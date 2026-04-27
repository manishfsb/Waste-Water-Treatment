"""
linear_modeling_exp2_s2_cyc.py

Trains OLS, Ridge, and ElasticNet on Experiment 2 Sub-experiment 2 datasets
with cyclic calendar encoding (month/dow replaced with sin/cos projections).

Feature set (21 features per target):
  Grab : Inlet pH/BOD/COD/TSS (Grab, 4) + all SEC_COLS (10) + Flow, Power, year,
         month_sin, month_cos, dow_sin, dow_cos (7) = 21 features
  Comp : Inlet pH/BOD/COD/TSS (Composite, 4) + all SEC_COLS (10) + COMMON_CYCLIC = 21 features

OLS uses LassoCV pre-screening (same as exp1_cyclic pattern).
OLS_full_* columns store the pre-LassoCV result.

Usage (from project root):
    .venv/bin/python3 modeling/models/linear/exp2_s2_cyc/linear_modeling_exp2_s2_cyc.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

ELNET_PARAM_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

# ── Feature sets ───────────────────────────────────────────────────────────────
GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]
SEC_COLS = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON_CYCLIC = [
    "Flow (MLD)", "Power Total (KW)", "year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]

E2S2_GRAB = GRAB_INLET + SEC_COLS + COMMON_CYCLIC   # 21 features
E2S2_COMP = COMP_INLET + SEC_COLS + COMMON_CYCLIC   # 21 features

# ── Dataset registry ───────────────────────────────────────────────────────────
def _cyc(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp2_cyclic",
                        f"{name}.xlsx")


DATASETS = [
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Grab_BOD", _cyc("grab_BOD"), E2S2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Grab_COD", _cyc("grab_COD"), E2S2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Grab_TSS", _cyc("grab_TSS"), E2S2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Grab_pH",  _cyc("grab_pH"),  E2S2_GRAB, "Effluent pH (Grab)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Comp_BOD", _cyc("comp_BOD"), E2S2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Comp_COD", _cyc("comp_COD"), E2S2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Comp_TSS", _cyc("comp_TSS"), E2S2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub2-Cyc", "E2S2Cyc_Comp_pH",  _cyc("comp_pH"),  E2S2_COMP, "Effluent pH (Composite)"),
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_R2":   float(r2_score(y_true, y_pred)),
        f"{prefix}_RMSE": _rmse(y_true, y_pred),
        f"{prefix}_MAE":  _mae(y_true, y_pred),
        f"{prefix}_MAPE": _mape(y_true, y_pred),
    }


# ── LassoCV feature selection for OLS ────────────────────────────────────────

def _lasso_select(X_tr_sc: np.ndarray, y_train: np.ndarray,
                  features: list, tscv) -> tuple:
    lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_sc, y_train)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        print("    LassoCV: all features zeroed — keeping full set")
        mask = np.ones(len(features), dtype=bool)
    n_in, n_kept = len(features), int(mask.sum())
    print(f"    LassoCV pre-screen → {n_kept}/{n_in} features kept", end="")
    if n_kept == n_in:
        print(" (no pruning)")
    else:
        dropped = [f for f, m in zip(features, mask) if not m]
        print(f" — dropped: {dropped}")
    selected = [f for f, m in zip(features, mask) if m]
    return mask, selected


# ── Run-number detection ───────────────────────────────────────────────────────

def get_run_number(subset_path: str) -> int:
    if not os.path.exists(subset_path):
        return 1
    cols = pd.read_excel(subset_path, nrows=0).columns.tolist()
    return sum(1 for c in cols if c.startswith("predicted_OLS_run_")) + 1


# ── Append predictions to subset file ─────────────────────────────────────────

def append_predictions(subset_path: str, pred_dict: dict):
    df = pd.read_excel(subset_path)
    for col, values in pred_dict.items():
        df[col] = values
    df.to_excel(subset_path, index=False)


# ── Core training ──────────────────────────────────────────────────────────────

def train_dataset(experiment, ds_id, path, features, target, run):
    df = pd.read_excel(path, parse_dates=["Date"])

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra    = df[df["year"] == 2020]
    if len(extra) > 0:
        train_df = pd.concat([extra, train_df]).drop_duplicates()

    test_df = df[df["year"] == TEST_YEAR]

    if len(test_df) == 0:
        print(f"  WARNING: no {TEST_YEAR} rows - skipping {ds_id}")
        return None, None

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing} - skipping {ds_id}")
        return None, None

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test  = test_df[features].values
    y_test  = test_df[target].values
    X_all   = df[features].values

    print(f"  Train: {len(train_df):>4d} rows | Test: {len(test_df):>3d} rows")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)

    tscv = TimeSeriesSplit(n_splits=3)
    n_in = len(features)

    # ── LassoCV feature selection for OLS ─────────────────────────────────────
    ols_mask, selected_ols = _lasso_select(X_tr_sc, y_train, features, tscv)
    X_tr_ols  = X_tr_sc[:, ols_mask]
    X_te_ols  = X_te_sc[:, ols_mask]
    X_all_ols = X_all_sc[:, ols_mask]
    n_sel_ols = int(ols_mask.sum())

    results = {
        "experiment":            experiment,
        "dataset":               ds_id,
        "target":                target,
        "run":                   run,
        "n_train":               len(train_df),
        "n_test":                len(test_df),
        "n_features":            n_in,
        "n_features_input":      n_in,
        "n_selected_ols":        n_sel_ols,
        "selected_features_ols": ", ".join(selected_ols),
    }
    preds = {}

    # ── OLS full set (pre-LassoCV baseline — stored as OLS_full_*) ────────────
    ols_full = LinearRegression()
    ols_full.fit(X_tr_sc, y_train)
    tr_ols_full = ols_full.predict(X_tr_sc)
    te_ols_full = ols_full.predict(X_te_sc)
    results["OLS_full_train_R2"]   = float(r2_score(y_train, tr_ols_full))
    results["OLS_full_test_R2"]    = float(r2_score(y_test,  te_ols_full))
    results["OLS_full_test_RMSE"]  = _rmse(y_test, te_ols_full)
    results["OLS_full_R2_gap"]     = results["OLS_full_train_R2"] - results["OLS_full_test_R2"]
    # Expose as R2_test_full / RMSE_test_full / R2_gap_full (unified report schema)
    results["R2_test_full"]        = results["OLS_full_test_R2"]
    results["RMSE_test_full"]      = results["OLS_full_test_RMSE"]
    results["R2_gap_full"]         = results["OLS_full_R2_gap"]
    print(f"    OLS_full - Train R²: {results['OLS_full_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_full_test_R2']:+.3f}  (all {n_in} features)")

    # ── OLS (LassoCV-selected features) ───────────────────────────────────────
    ols = LinearRegression()
    ols.fit(X_tr_ols, y_train)
    tr_ols = ols.predict(X_tr_ols)
    te_ols = ols.predict(X_te_ols)
    results.update(_metrics(y_train, tr_ols, "OLS_train"))
    results.update(_metrics(y_test,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    preds[f"predicted_OLS_run_{run}"] = np.round(ols.predict(X_all_ols), 3)
    joblib.dump({"scaler": scaler, "model": ols,
                 "selected_features": selected_ols, "feature_mask": ols_mask},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    - Train R²: {results['OLS_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_test_R2']:+.3f} | "
          f"RMSE: {results['OLS_test_RMSE']:.3f}  "
          f"({n_sel_ols}/{n_in} features via LassoCV)")

    # ── Ridge (full feature set) ───────────────────────────────────────────────
    ridge_gs = GridSearchCV(
        Ridge(), {"alpha": RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    ridge_gs.fit(X_tr_sc, y_train)
    ridge = ridge_gs.best_estimator_
    tr_ridge = ridge.predict(X_tr_sc)
    te_ridge = ridge.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ridge, "Ridge_train"))
    results.update(_metrics(y_test,  te_ridge, "Ridge_test"))
    results["Ridge_R2_gap"]  = results["Ridge_train_R2"] - results["Ridge_test_R2"]
    results["Ridge_CV_RMSE"] = float(-ridge_gs.best_score_)
    results["Ridge_alpha"]   = ridge_gs.best_params_["alpha"]
    preds[f"predicted_Ridge_run_{run}"] = np.round(ridge.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ridge},
                os.path.join(MODELS_DIR, f"{ds_id}_Ridge_run_{run}.pkl"))
    print(f"    Ridge  - Train R²: {results['Ridge_train_R2']:+.3f} | "
          f"Test R²: {results['Ridge_test_R2']:+.3f} | "
          f"α={results['Ridge_alpha']}  (full {n_in} features)")

    # ── ElasticNet (full set) ──────────────────────────────────────────────────
    elnet_gs = GridSearchCV(
        ElasticNet(max_iter=10000), ELNET_PARAM_GRID,
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    elnet_gs.fit(X_tr_sc, y_train)
    elnet = elnet_gs.best_estimator_
    tr_elnet = elnet.predict(X_tr_sc)
    te_elnet = elnet.predict(X_te_sc)
    results.update(_metrics(y_train, tr_elnet, "ElNet_train"))
    results.update(_metrics(y_test,  te_elnet, "ElNet_test"))
    results["ElNet_R2_gap"]   = results["ElNet_train_R2"] - results["ElNet_test_R2"]
    results["ElNet_CV_RMSE"]  = float(-elnet_gs.best_score_)
    results["ElNet_alpha"]    = elnet_gs.best_params_["alpha"]
    results["ElNet_l1_ratio"] = elnet_gs.best_params_["l1_ratio"]
    elnet_mask     = elnet.coef_ != 0
    elnet_selected = [f for f, m in zip(features, elnet_mask) if m]
    results["ElNet_n_selected"]        = int(elnet_mask.sum())
    results["ElNet_selected_features"] = ", ".join(elnet_selected)
    preds[f"predicted_ElNet_run_{run}"] = np.round(elnet.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": elnet,
                 "selected_features": elnet_selected, "feature_mask": elnet_mask},
                os.path.join(MODELS_DIR, f"{ds_id}_ElNet_run_{run}.pkl"))
    print(f"    ElNet  - Train R²: {results['ElNet_train_R2']:+.3f} | "
          f"Test R²: {results['ElNet_test_R2']:+.3f} | "
          f"α={results['ElNet_alpha']}, l1={results['ElNet_l1_ratio']} | "
          f"kept {results['ElNet_n_selected']}/{n_in} features")

    _plot_comparison(ds_id, run, y_test, te_ols, te_ridge, te_elnet,
                     test_df["Date"].values if "Date" in test_df.columns else None)

    return results, preds


# ── Plotting ───────────────────────────────────────────────────────────────────

MODEL_COLORS = {"OLS": "#E15252", "Ridge": "#4A90D9", "ElNet": "#5BAD6F"}


def _plot_comparison(ds_id, run, y_test, te_ols, te_ridge, te_elnet, dates):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"{ds_id}  |  Test Set Actual vs Predicted  (run {run})",
                 fontsize=13, fontweight="bold")
    for ax, (label, preds, color) in zip(axes, [
        ("OLS",   te_ols,   MODEL_COLORS["OLS"]),
        ("Ridge", te_ridge, MODEL_COLORS["Ridge"]),
        ("ElNet", te_elnet, MODEL_COLORS["ElNet"]),
    ]):
        ax.scatter(y_test, preds, color=color, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_test.min(), preds.min())
        hi = max(y_test.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
        r2 = r2_score(y_test, preds)
        ax.set_title(f"{label}\nR²={r2:+.3f}  RMSE={_rmse(y_test, preds):.3f}", fontsize=11)
        ax.set_xlabel("Actual", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Predicted", fontsize=10)
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_id}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot → {path}")


def _plot_r2_barchart(df_results: pd.DataFrame, run: int):
    labels = [d.split("_", 1)[1] if "_" in d else d for d in df_results["dataset"]]
    x = np.arange(len(df_results))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df_results) * 1.2), 5))
    ax.bar(x - w, df_results["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
    ax.bar(x,     df_results["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
    ax.bar(x + w, df_results["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test R²", fontsize=11)
    ax.set_title(f"Exp2-Sub2-Cyc  |  Test R² Comparison (run {run})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(-0.15,
        df_results[["OLS_test_R2","Ridge_test_R2","ElNet_test_R2"]].min().min() - 0.05))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"Exp2-Sub2-Cyc_r2_comparison_run_{run}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  R² bar chart → {path}")


# ── Results persistence ────────────────────────────────────────────────────────

def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"Results → {RESULTS_FILE}")
    key_cols = ["experiment", "dataset",
                "OLS_test_R2",   "Ridge_test_R2",   "ElNet_test_R2",
                "OLS_test_RMSE", "Ridge_test_RMSE", "ElNet_test_RMSE"]
    print(df_new[key_cols].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    first_path = DATASETS[0][2]
    run = get_run_number(first_path)
    print(f"Exp2-Sub2-Cyc Linear Modeling - Run {run}")
    print(f"Features: {len(E2S2_GRAB)} per target  "
          f"(cyclic calendar: month_sin/cos, dow_sin/cos; LassoCV FS for OLS)\n")

    all_results = []

    for experiment, ds_id, path, features, target in DATASETS:
        print(f"{'─'*60}")
        print(f"[{experiment}]  {ds_id}")
        print(f"  Target   : {target}")
        print(f"  Features : {len(features)}")

        if not os.path.exists(path):
            print(f"  WARNING: file not found - {path}")
            print(f"  Run make_sub2_cyclic_datasets.py first.")
            continue

        results, preds = train_dataset(experiment, ds_id, path, features, target, run)
        if results is None:
            continue

        append_predictions(path, preds)
        print(f"  Predictions appended → {path}")
        all_results.append(results)

    if not all_results:
        print("No results produced.")
        return

    df_results = pd.DataFrame(all_results)
    _plot_r2_barchart(df_results, run)
    save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
