"""
linear_modeling_exp2_s6_fs.py - OLS (LassoCV-selected), Ridge, ElNet on Exp2 SE2a-FS.

Feature set: SECONDARY (10) = Sec Clarifier (5) + Sec Sedimentation (5).
OLS applies LassoCV pre-screening; Ridge/ElNet select implicitly via regularisation.
Identical datasets to exp2_s6 (no COMMON, no Inlet).

Exp key:  Exp2-S6-FS

Usage (from project root):
    .venv/bin/python3 modeling/models/linear/exp2_s6_fs/linear_modeling_exp2_s6_fs.py
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ELNET_PARAM_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [
        c for c in df.columns
        if c != target
        and c not in _EXCLUDE_COLS
        and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
    ]

def _ds(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp6", f"{name}.xlsx")

DATASETS = [
    ("Exp2-S6-FS", "E2S6FS_Grab_BOD", _ds("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Exp2-S6-FS", "E2S6FS_Grab_COD", _ds("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Exp2-S6-FS", "E2S6FS_Grab_TSS", _ds("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Exp2-S6-FS", "E2S6FS_Grab_pH",  _ds("grab_pH"),  "Effluent pH (Grab)"),
    ("Exp2-S6-FS", "E2S6FS_Comp_BOD", _ds("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Exp2-S6-FS", "E2S6FS_Comp_COD", _ds("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Exp2-S6-FS", "E2S6FS_Comp_TSS", _ds("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Exp2-S6-FS", "E2S6FS_Comp_pH",  _ds("comp_pH"),  "Effluent pH (Composite)"),
]

MODEL_COLORS = {"OLS": "#E15252", "Ridge": "#4A90D9", "ElNet": "#5BAD6F"}


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    if not mask.any(): return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_R2":   float(r2_score(y_true, y_pred)),
        f"{prefix}_RMSE": _rmse(y_true, y_pred),
        f"{prefix}_MAE":  _mae(y_true,  y_pred),
        f"{prefix}_MAPE": _mape(y_true, y_pred),
    }


def _lasso_select(X_tr_sc, y_train, features, tscv):
    lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_sc, y_train)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        print("    LassoCV: all zeroed - keeping full set")
        mask = np.ones(len(features), dtype=bool)
    n_kept = int(mask.sum())
    kept   = [f for f, m in zip(features, mask) if m]
    dropped = [f for f, m in zip(features, mask) if not m]
    print(f"    LassoCV: {n_kept}/{len(features)} kept", end="")
    if dropped: print(f"  dropped: {dropped}")
    else: print(" (no pruning)")
    return mask, kept


def get_run_number() -> int:
    if not os.path.exists(RESULTS_FILE): return 1
    df = pd.read_excel(RESULTS_FILE)
    return int(df["run"].max()) + 1 if "run" in df.columns else 1


def _plot_scatter(ds_id, run, y_test, te_ols, te_ridge, te_elnet):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"{ds_id}  |  Test Actual vs Predicted  (run {run})", fontsize=13, fontweight="bold")
    for ax, (lbl, preds, col) in zip(axes, [("OLS", te_ols, MODEL_COLORS["OLS"]),
                                             ("Ridge", te_ridge, MODEL_COLORS["Ridge"]),
                                             ("ElNet", te_elnet, MODEL_COLORS["ElNet"])]):
        ax.scatter(y_test, preds, color=col, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_test.min(), preds.min()); hi = max(y_test.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
        ax.set_title(f"{lbl}\nR2={r2_score(y_test, preds):+.3f}  RMSE={_rmse(y_test, preds):.3f}", fontsize=11)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted") if ax is axes[0] else None
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_id}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)


def train_dataset(experiment, ds_id, path, features, target, run):
    df = pd.read_excel(path, parse_dates=["Date"])
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra    = df[df["year"] == 2020]
    if len(extra) > 0:
        train_df = pd.concat([extra, train_df]).drop_duplicates()
    test_df = df[df["year"] == TEST_YEAR]

    if len(test_df) == 0:
        print(f"  WARNING: no {TEST_YEAR} rows - skipping {ds_id}"); return None, None

    X_train = train_df[features].values; y_train = train_df[target].values
    X_test  = test_df[features].values;  y_test  = test_df[target].values
    X_all   = df[features].values

    print(f"  Train: {len(train_df):>4d}  Test: {len(test_df):>3d}  Features: {len(features)}")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)
    tscv     = TimeSeriesSplit(n_splits=3)

    ols_mask, selected_ols = _lasso_select(X_tr_sc, y_train, features, tscv)
    X_tr_ols  = X_tr_sc[:, ols_mask]
    X_te_ols  = X_te_sc[:, ols_mask]
    X_all_ols = X_all_sc[:, ols_mask]

    results = {
        "experiment": experiment, "dataset": ds_id, "target": target, "run": run,
        "n_train": len(train_df), "n_test": len(test_df),
        "n_features": len(features), "n_features_input": len(features),
        "n_selected_ols": int(ols_mask.sum()),
        "selected_features_ols": ", ".join(selected_ols),
    }
    preds = {}

    # OLS full (pre-LassoCV baseline)
    ols_full = LinearRegression()
    ols_full.fit(X_tr_sc, y_train)
    results.update(_metrics(y_train, ols_full.predict(X_tr_sc), "OLS_full_train"))
    results.update(_metrics(y_test,  ols_full.predict(X_te_sc),  "OLS_full_test"))
    preds[f"predicted_OLS_full_fs_run_{run}"] = np.round(ols_full.predict(X_all_sc), 3)

    # OLS selected
    ols = LinearRegression()
    ols.fit(X_tr_ols, y_train)
    tr_ols = ols.predict(X_tr_ols); te_ols = ols.predict(X_te_ols)
    results.update(_metrics(y_train, tr_ols, "OLS_train"))
    results.update(_metrics(y_test,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    preds[f"predicted_OLS_fs_run_{run}"] = np.round(ols.predict(X_all_ols), 3)
    joblib.dump({"scaler": scaler, "mask": ols_mask, "model": ols},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    Train R2: {results['OLS_train_R2']:+.3f}  Test R2: {results['OLS_test_R2']:+.3f}")

    # Ridge
    ridge_gs = GridSearchCV(Ridge(), {"alpha": RIDGE_ALPHAS},
                            scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1)
    ridge_gs.fit(X_tr_sc, y_train)
    ridge = ridge_gs.best_estimator_
    tr_ridge = ridge.predict(X_tr_sc); te_ridge = ridge.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ridge, "Ridge_train"))
    results.update(_metrics(y_test,  te_ridge, "Ridge_test"))
    results["Ridge_R2_gap"]  = results["Ridge_train_R2"] - results["Ridge_test_R2"]
    results["Ridge_CV_RMSE"] = float(-ridge_gs.best_score_)
    results["Ridge_alpha"]   = ridge_gs.best_params_["alpha"]
    preds[f"predicted_Ridge_fs_run_{run}"] = np.round(ridge.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ridge},
                os.path.join(MODELS_DIR, f"{ds_id}_Ridge_run_{run}.pkl"))
    print(f"    Ridge  Train R2: {results['Ridge_train_R2']:+.3f}  Test R2: {results['Ridge_test_R2']:+.3f}  alpha={results['Ridge_alpha']}")

    # ElNet
    elnet_gs = GridSearchCV(ElasticNet(max_iter=10000), ELNET_PARAM_GRID,
                            scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1)
    elnet_gs.fit(X_tr_sc, y_train)
    elnet = elnet_gs.best_estimator_
    tr_elnet = elnet.predict(X_tr_sc); te_elnet = elnet.predict(X_te_sc)
    results.update(_metrics(y_train, tr_elnet, "ElNet_train"))
    results.update(_metrics(y_test,  te_elnet, "ElNet_test"))
    results["ElNet_R2_gap"]   = results["ElNet_train_R2"] - results["ElNet_test_R2"]
    results["ElNet_CV_RMSE"]  = float(-elnet_gs.best_score_)
    results["ElNet_alpha"]    = elnet_gs.best_params_["alpha"]
    results["ElNet_l1_ratio"] = elnet_gs.best_params_["l1_ratio"]
    preds[f"predicted_ElNet_fs_run_{run}"] = np.round(elnet.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": elnet},
                os.path.join(MODELS_DIR, f"{ds_id}_ElNet_run_{run}.pkl"))
    print(f"    ElNet  Train R2: {results['ElNet_train_R2']:+.3f}  Test R2: {results['ElNet_test_R2']:+.3f}")

    _plot_scatter(ds_id, run, y_test, te_ols, te_ridge, te_elnet)

    df2 = pd.read_excel(path)
    for col, vals in preds.items():
        df2[col] = vals
    df2.to_excel(path, index=False)

    return results, preds


def _plot_r2_barchart(df_results, run):
    labels = [d.split("_", 1)[1] if "_" in d else d for d in df_results["dataset"]]
    x, w = np.arange(len(df_results)), 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df_results) * 1.2), 5))
    ax.bar(x - w, df_results["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
    ax.bar(x,     df_results["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
    ax.bar(x + w, df_results["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test R2", fontsize=11)
    ax.set_title(f"Exp2-SE2a-FS (Secondary only, LassoCV/Ridge/ElNet)  |  Test R2 (run {run})", fontsize=12)
    ax.legend(fontsize=9); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"Exp2-S6-FS_r2_comparison_run_{run}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  R2 bar chart -> {path}")


def save_results(all_results, run):
    df_new = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    key_cols = ["experiment","dataset","OLS_test_R2","Ridge_test_R2","ElNet_test_R2",
                "OLS_test_RMSE","Ridge_test_RMSE","ElNet_test_RMSE","n_selected_ols"]
    print(f"\n{'='*60}")
    print(f"Results -> {RESULTS_FILE}")
    print(df_new[[c for c in key_cols if c in df_new.columns]].to_string(index=False))


def main():
    run = get_run_number()
    print(f"Linear Modeling  -  Exp2-SE2a-FS (Secondary only, LassoCV)  -  Run {run}")
    print(f"Datasets: {len(DATASETS)}  |  Models: OLS (LassoCV-FS), Ridge, ElNet\n")

    all_results = []
    for experiment, ds_id, path, target in DATASETS:
        print(f"{'--'*30}\n[{experiment}]  {ds_id}\n  Target: {target}")
        if not os.path.exists(path):
            print(f"  WARNING: file not found - {path}"); continue
        df_peek  = pd.read_excel(path, nrows=0)
        features = infer_features(df_peek, target)
        print(f"  Features (inferred): {len(features)}")
        results, preds = train_dataset(experiment, ds_id, path, features, target, run)
        if results is None: continue
        all_results.append(results)

    if not all_results:
        print("No results produced."); return

    df_results = pd.DataFrame(all_results)
    _plot_r2_barchart(df_results, run)
    save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
