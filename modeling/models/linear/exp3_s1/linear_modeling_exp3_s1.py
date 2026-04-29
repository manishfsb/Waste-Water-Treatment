"""
linear_modeling_exp3_s1.py - OLS, Ridge, and ElasticNet on Experiment 3 Sub-1 datasets.

Feature set: Exp2-Sub2 baseline (21 features) + ADD-tier features (7 Aeration cols)
= 28 features per target. No feature selection applied.

  ADD tier: Aeration DO/MLSS/SV30/SVI (Existing) + pH (Existing) + DO/SV30 (New)

Datasets: experiment3/sub_exp1/ (built by make_sub1_datasets.py)
Exp key: Exp3-S1

Usage (from project root):
    .venv/bin/python3 modeling/models/linear/exp3_s1/linear_modeling_exp3_s1.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
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

# ── Feature inference ──────────────────────────────────────────────────────────
_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [
        c for c in df.columns
        if c != target
        and c not in _EXCLUDE_COLS
        and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
    ]

# ── Dataset registry ───────────────────────────────────────────────────────────
def _ds(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1", f"{name}.xlsx")

DATASETS = [
    ("Exp3-S1", "Exp3S1_Grab_BOD", _ds("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Exp3-S1", "Exp3S1_Grab_COD", _ds("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Exp3-S1", "Exp3S1_Grab_TSS", _ds("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Exp3-S1", "Exp3S1_Grab_pH",  _ds("grab_pH"),  "Effluent pH (Grab)"),
    ("Exp3-S1", "Exp3S1_Comp_BOD", _ds("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Exp3-S1", "Exp3S1_Comp_COD", _ds("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Exp3-S1", "Exp3S1_Comp_TSS", _ds("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Exp3-S1", "Exp3S1_Comp_pH",  _ds("comp_pH"),  "Effluent pH (Composite)"),
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
        f"{prefix}_MAE":  _mae(y_true,  y_pred),
        f"{prefix}_MAPE": _mape(y_true, y_pred),
    }


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

    print(f"  Train: {len(train_df):>4d} rows | Test: {len(test_df):>3d} rows | Features: {len(features)}")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)

    tscv = TimeSeriesSplit(n_splits=3)

    results = {
        "experiment": experiment,
        "dataset":    ds_id,
        "target":     target,
        "run":        run,
        "n_train":    len(train_df),
        "n_test":     len(test_df),
        "n_features": len(features),
    }
    preds = {}

    # ── OLS ───────────────────────────────────────────────────────────────────
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_train)
    tr_ols = ols.predict(X_tr_sc)
    te_ols = ols.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ols, "OLS_train"))
    results.update(_metrics(y_test,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    col_ols = f"predicted_OLS_run_{run}"
    preds[col_ols] = np.round(ols.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ols},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    - Train R²: {results['OLS_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_test_R2']:+.3f} | "
          f"RMSE: {results['OLS_test_RMSE']:.3f}")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    ridge_gs = GridSearchCV(
        Ridge(), {"alpha": RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    ridge_gs.fit(X_tr_sc, y_train)
    ridge    = ridge_gs.best_estimator_
    tr_ridge = ridge.predict(X_tr_sc)
    te_ridge = ridge.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ridge, "Ridge_train"))
    results.update(_metrics(y_test,  te_ridge, "Ridge_test"))
    results["Ridge_R2_gap"]  = results["Ridge_train_R2"] - results["Ridge_test_R2"]
    results["Ridge_CV_RMSE"] = float(-ridge_gs.best_score_)
    results["Ridge_alpha"]   = ridge_gs.best_params_["alpha"]
    col_ridge = f"predicted_Ridge_run_{run}"
    preds[col_ridge] = np.round(ridge.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ridge},
                os.path.join(MODELS_DIR, f"{ds_id}_Ridge_run_{run}.pkl"))
    print(f"    Ridge  - Train R²: {results['Ridge_train_R2']:+.3f} | "
          f"Test R²: {results['Ridge_test_R2']:+.3f} | "
          f"α={results['Ridge_alpha']}")

    # ── ElasticNet ────────────────────────────────────────────────────────────
    elnet_gs = GridSearchCV(
        ElasticNet(max_iter=10000), ELNET_PARAM_GRID,
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    elnet_gs.fit(X_tr_sc, y_train)
    elnet    = elnet_gs.best_estimator_
    tr_elnet = elnet.predict(X_tr_sc)
    te_elnet = elnet.predict(X_te_sc)
    results.update(_metrics(y_train, tr_elnet, "ElNet_train"))
    results.update(_metrics(y_test,  te_elnet, "ElNet_test"))
    results["ElNet_R2_gap"]   = results["ElNet_train_R2"] - results["ElNet_test_R2"]
    results["ElNet_CV_RMSE"]  = float(-elnet_gs.best_score_)
    results["ElNet_alpha"]    = elnet_gs.best_params_["alpha"]
    results["ElNet_l1_ratio"] = elnet_gs.best_params_["l1_ratio"]
    col_elnet = f"predicted_ElNet_run_{run}"
    preds[col_elnet] = np.round(elnet.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": elnet},
                os.path.join(MODELS_DIR, f"{ds_id}_ElNet_run_{run}.pkl"))
    print(f"    ElNet  - Train R²: {results['ElNet_train_R2']:+.3f} | "
          f"Test R²: {results['ElNet_test_R2']:+.3f} | "
          f"α={results['ElNet_alpha']}, l1={results['ElNet_l1_ratio']}")

    _plot_scatter(ds_id, run, y_test, te_ols, te_ridge, te_elnet)

    return results, preds


# ── Plotting ───────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "OLS":   "#E15252",
    "Ridge": "#4A90D9",
    "ElNet": "#5BAD6F",
}


def _plot_scatter(ds_id, run, y_test, te_ols, te_ridge, te_elnet):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"{ds_id}  |  Test Set Actual vs Predicted  (run {run})",
                 fontsize=13, fontweight="bold")
    pairs = [
        ("OLS",   te_ols,   MODEL_COLORS["OLS"]),
        ("Ridge", te_ridge, MODEL_COLORS["Ridge"]),
        ("ElNet", te_elnet, MODEL_COLORS["ElNet"]),
    ]
    for ax, (label, preds_arr, color) in zip(axes, pairs):
        ax.scatter(y_test, preds_arr, color=color, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_test.min(), preds_arr.min())
        hi = max(y_test.max(), preds_arr.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
        r2v      = r2_score(y_test, preds_arr)
        rmse_val = _rmse(y_test, preds_arr)
        ax.set_title(f"{label}\nR²={r2v:+.3f}  RMSE={rmse_val:.3f}", fontsize=11)
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
    for exp in df_results["experiment"].unique():
        sub    = df_results[df_results["experiment"] == exp].copy()
        labels = [d.split("_", 1)[1] if "_" in d else d for d in sub["dataset"]]
        x, w   = np.arange(len(sub)), 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(sub) * 1.2), 5))
        ax.bar(x - w, sub["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
        ax.bar(x,     sub["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
        ax.bar(x + w, sub["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Test R²", fontsize=11)
        ax.set_title(f"{exp}  |  Test R² Comparison (run {run})", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=min(-0.15, sub[["OLS_test_R2","Ridge_test_R2","ElNet_test_R2"]].min().min() - 0.05))
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{exp}_r2_comparison_run_{run}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
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
                "OLS_test_RMSE", "Ridge_test_RMSE",  "ElNet_test_RMSE"]
    print(df_new[key_cols].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    first_path = DATASETS[0][2]
    run = get_run_number(first_path)
    print(f"Linear Modeling — Exp3-S1 (ADD-tier, no FS) — Run {run}")
    print(f"Datasets: {len(DATASETS)}  |  Models: OLS, Ridge, ElasticNet\n")

    all_results = []

    for experiment, ds_id, path, target in DATASETS:
        print(f"{'─'*60}")
        print(f"[{experiment}]  {ds_id}")
        print(f"  Target: {target}")

        if not os.path.exists(path):
            print(f"  WARNING: file not found - {path}")
            continue

        df_peek  = pd.read_excel(path, nrows=0)
        features = infer_features(df_peek, target)
        print(f"  Features (inferred): {len(features)}")

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
    print(f"\n{'='*60}")
    print("Generating summary plots...")
    _plot_r2_barchart(df_results, run)
    save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
