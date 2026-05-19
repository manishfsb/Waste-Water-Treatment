"""
linear_modeling_exp9_s5.py - OLS, Ridge, ElNet on Experiment 9 SE5 datasets.

Recency hypothesis - W1 + log1p feature transform + log1p target transform (log-log model).
This is the combination of SE3 (log1p targets + Duan smearing) and SE4 (log1p features
in-place). The two transforms are orthogonal: SE4 addresses non-linear feature-target
functional form; SE3 addresses residual heteroscedasticity.

Feature set: identical to Exp9-SE1 (27 features), with the 12 right-skewed
concentration columns replaced in-place with their log1p values.
Target: BOD/COD/TSS are log1p-transformed before fitting; pH is left on its natural
scale. Duan smearing back-transforms predictions to the original scale.
All reported metrics are on the ORIGINAL scale.

Datasets: experiment9/sub_exp1/ (same as SE1)
Exp key : Exp9-SE5

Usage (from project root):
    .venv/bin/python3 modeling/models/linear/exp9_s5/linear_modeling_exp9_s5.py
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

# -- Paths ----------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -- Constants ------------------------------------------------------------------
TRAIN_YEARS = [2024]
TEST_YEAR   = 2025

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

ELNET_PARAM_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

# Same 12 columns as Exp7 (Phase 10) LOG_COL_NAMES that exist in the Exp9 feature set.
# pH columns excluded (already log-activity scale), as are Flow, Power, cyclic, Aeration.
LOG_FEATURES = {
    "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)",
    "Inlet TSS (mg/L, Grab)",
    "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)",
    "Inlet TSS (mg/L, Composite)",
    "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)",
    "Sec Clarifier TSS (mg/L)",
    "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)",
    "Sec Sed TSS (mg/L)",
}

# -- Feature inference ----------------------------------------------------------
_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [
        c for c in df.columns
        if c != target
        and c not in _EXCLUDE_COLS
        and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
    ]

def apply_log_features(df: pd.DataFrame, features: list) -> tuple:
    """Apply log1p in-place to LOG_FEATURES that exist in `features`. Returns
    modified copy and the list of columns actually transformed."""
    df = df.copy()
    transformed = []
    for col in features:
        if col in LOG_FEATURES and col in df.columns:
            df[col] = np.log1p(df[col])
            transformed.append(col)
    return df, transformed

# -- Dataset registry -----------------------------------------------------------
def _ds(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment9", "sub_exp1", f"{name}.xlsx")

# log_y=True for concentration targets; False for pH (already on log-activity scale)
DATASETS = [
    ("Exp9-SE5", "Exp9W1LogLF_Grab_BOD", _ds("grab_BOD"), "Effluent BOD (mg/L, Grab)",       True),
    ("Exp9-SE5", "Exp9W1LogLF_Grab_COD", _ds("grab_COD"), "Effluent COD (mg/L, Grab)",       True),
    ("Exp9-SE5", "Exp9W1LogLF_Grab_TSS", _ds("grab_TSS"), "Effluent TSS (mg/L, Grab)",       True),
    ("Exp9-SE5", "Exp9W1LogLF_Grab_pH",  _ds("grab_pH"),  "Effluent pH (Grab)",              False),
    ("Exp9-SE5", "Exp9W1LogLF_Comp_BOD", _ds("comp_BOD"), "Effluent BOD (mg/L, Composite)",  True),
    ("Exp9-SE5", "Exp9W1LogLF_Comp_COD", _ds("comp_COD"), "Effluent COD (mg/L, Composite)",  True),
    ("Exp9-SE5", "Exp9W1LogLF_Comp_TSS", _ds("comp_TSS"), "Effluent TSS (mg/L, Composite)",  True),
    ("Exp9-SE5", "Exp9W1LogLF_Comp_pH",  _ds("comp_pH"),  "Effluent pH (Composite)",         False),
]

# -- Metric helpers -------------------------------------------------------------
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

# -- Log1p / Duan smearing helpers ----------------------------------------------
def _compute_smear(y_train_log: np.ndarray, y_train_pred_log: np.ndarray) -> float:
    """Duan smearing factor: mean(exp(residuals on log scale))."""
    resid = y_train_log - y_train_pred_log
    return float(np.mean(np.exp(resid)))

def _back_transform(y_pred_log: np.ndarray, smear: float) -> np.ndarray:
    """Back-transform from log1p scale to original scale with Duan smearing."""
    return np.expm1(y_pred_log) * smear

# -- Run-number detection -------------------------------------------------------
def get_run_number() -> int:
    if not os.path.exists(RESULTS_FILE):
        return 1
    df = pd.read_excel(RESULTS_FILE)
    if "run" not in df.columns or df.empty:
        return 1
    return int(df["run"].max()) + 1

# -- Append predictions to dataset file -----------------------------------------
def append_predictions(subset_path: str, pred_dict: dict):
    df = pd.read_excel(subset_path)
    for col, values in pred_dict.items():
        df[col] = values
    df.to_excel(subset_path, index=False)

# -- Core training --------------------------------------------------------------
def train_dataset(experiment, ds_id, path, features, target, log_y, run):
    df_raw = pd.read_excel(path, parse_dates=["Date"])

    # Apply log1p to concentration features before any split
    df, transformed = apply_log_features(df_raw, features)
    n_transformed = len(transformed)

    train_df = df[df["year"].isin(TRAIN_YEARS)].dropna(subset=features + [target]).copy()
    test_df  = df[df["year"] == TEST_YEAR].dropna(subset=features + [target])

    if len(train_df) == 0:
        print(f"  WARNING: no training rows for {TRAIN_YEARS} - skipping {ds_id}")
        return None, None
    if len(test_df) == 0:
        print(f"  WARNING: no {TEST_YEAR} rows - skipping {ds_id}")
        return None, None

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test  = test_df[features].values
    y_test  = test_df[target].values
    X_all   = df[features].values

    print(f"  Train: {len(train_df):>4d} rows | Test: {len(test_df):>3d} rows | "
          f"Features: {len(features)} | log_feat={n_transformed} | log_y={log_y}")

    # Transform targets
    if log_y:
        y_train_t = np.log1p(y_train)
    else:
        y_train_t = y_train

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)
    tscv     = TimeSeriesSplit(n_splits=3)

    results = {
        "experiment":     experiment,
        "dataset":        ds_id,
        "target":         target,
        "run":            run,
        "log_y":          log_y,
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "n_features":     len(features),
        "n_log_features": n_transformed,
        "log_features":   ", ".join(transformed),
    }
    preds = {}

    # -- OLS --
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_train_t)
    tr_ols_t = ols.predict(X_tr_sc)
    te_ols_t = ols.predict(X_te_sc)
    if log_y:
        smear_ols = _compute_smear(y_train_t, tr_ols_t)
        tr_ols    = _back_transform(tr_ols_t, smear_ols)
        te_ols    = _back_transform(te_ols_t, smear_ols)
        all_ols   = _back_transform(ols.predict(X_all_sc), smear_ols)
        results["OLS_smear"] = round(smear_ols, 6)
    else:
        tr_ols  = tr_ols_t; te_ols = te_ols_t
        all_ols = ols.predict(X_all_sc)
    results.update(_metrics(y_train, tr_ols, "OLS_train"))
    results.update(_metrics(y_test,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    preds[f"predicted_OLS_loglogf_run_{run}"] = np.round(all_ols, 3)
    joblib.dump({"scaler": scaler, "model": ols, "log_y": log_y,
                 "log_features": transformed, "smear": results.get("OLS_smear", 1.0)},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    - Train R²: {results['OLS_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_test_R2']:+.3f} | "
          f"RMSE: {results['OLS_test_RMSE']:.3f}")

    # -- Ridge --
    ridge_gs = GridSearchCV(
        Ridge(), {"alpha": RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    ridge_gs.fit(X_tr_sc, y_train_t)
    ridge      = ridge_gs.best_estimator_
    tr_ridge_t = ridge.predict(X_tr_sc)
    te_ridge_t = ridge.predict(X_te_sc)
    if log_y:
        smear_ridge = _compute_smear(y_train_t, tr_ridge_t)
        tr_ridge    = _back_transform(tr_ridge_t, smear_ridge)
        te_ridge    = _back_transform(te_ridge_t, smear_ridge)
        all_ridge   = _back_transform(ridge.predict(X_all_sc), smear_ridge)
        results["Ridge_smear"] = round(smear_ridge, 6)
    else:
        tr_ridge  = tr_ridge_t; te_ridge = te_ridge_t
        all_ridge = ridge.predict(X_all_sc)
    results.update(_metrics(y_train, tr_ridge, "Ridge_train"))
    results.update(_metrics(y_test,  te_ridge, "Ridge_test"))
    results["Ridge_R2_gap"]  = results["Ridge_train_R2"] - results["Ridge_test_R2"]
    results["Ridge_CV_RMSE"] = float(-ridge_gs.best_score_)
    results["Ridge_alpha"]   = ridge_gs.best_params_["alpha"]
    preds[f"predicted_Ridge_loglogf_run_{run}"] = np.round(all_ridge, 3)
    joblib.dump({"scaler": scaler, "model": ridge, "log_y": log_y,
                 "log_features": transformed, "smear": results.get("Ridge_smear", 1.0)},
                os.path.join(MODELS_DIR, f"{ds_id}_Ridge_run_{run}.pkl"))
    print(f"    Ridge  - Train R²: {results['Ridge_train_R2']:+.3f} | "
          f"Test R²: {results['Ridge_test_R2']:+.3f} | "
          f"alpha={results['Ridge_alpha']}")

    # -- ElasticNet --
    elnet_gs = GridSearchCV(
        ElasticNet(max_iter=10000), ELNET_PARAM_GRID,
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    elnet_gs.fit(X_tr_sc, y_train_t)
    elnet      = elnet_gs.best_estimator_
    tr_elnet_t = elnet.predict(X_tr_sc)
    te_elnet_t = elnet.predict(X_te_sc)
    if log_y:
        smear_elnet = _compute_smear(y_train_t, tr_elnet_t)
        tr_elnet    = _back_transform(tr_elnet_t, smear_elnet)
        te_elnet    = _back_transform(te_elnet_t, smear_elnet)
        all_elnet   = _back_transform(elnet.predict(X_all_sc), smear_elnet)
        results["ElNet_smear"] = round(smear_elnet, 6)
    else:
        tr_elnet  = tr_elnet_t; te_elnet = te_elnet_t
        all_elnet = elnet.predict(X_all_sc)
    results.update(_metrics(y_train, tr_elnet, "ElNet_train"))
    results.update(_metrics(y_test,  te_elnet, "ElNet_test"))
    results["ElNet_R2_gap"]   = results["ElNet_train_R2"] - results["ElNet_test_R2"]
    results["ElNet_CV_RMSE"]  = float(-elnet_gs.best_score_)
    results["ElNet_alpha"]    = elnet_gs.best_params_["alpha"]
    results["ElNet_l1_ratio"] = elnet_gs.best_params_["l1_ratio"]
    preds[f"predicted_ElNet_loglogf_run_{run}"] = np.round(all_elnet, 3)
    joblib.dump({"scaler": scaler, "model": elnet, "log_y": log_y,
                 "log_features": transformed, "smear": results.get("ElNet_smear", 1.0)},
                os.path.join(MODELS_DIR, f"{ds_id}_ElNet_run_{run}.pkl"))
    print(f"    ElNet  - Train R²: {results['ElNet_train_R2']:+.3f} | "
          f"Test R²: {results['ElNet_test_R2']:+.3f} | "
          f"alpha={results['ElNet_alpha']} l1={results['ElNet_l1_ratio']}")

    _plot_scatter(ds_id, run, log_y, y_test, te_ols, te_ridge, te_elnet)
    return results, preds

# -- Plotting -------------------------------------------------------------------
MODEL_COLORS = {"OLS": "#E15252", "Ridge": "#4A90D9", "ElNet": "#5BAD6F"}

def _plot_scatter(ds_id, run, log_y, y_test, te_ols, te_ridge, te_elnet):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    tag = " (log1p target+feat)" if log_y else " (log1p feat only)"
    fig.suptitle(f"{ds_id}  |  Test Set  -  W1+LogLog{tag} (run {run})",
                 fontsize=12, fontweight="bold")
    for ax, (lbl, preds_arr, col) in zip(axes, [
        ("OLS",   te_ols,   MODEL_COLORS["OLS"]),
        ("Ridge", te_ridge, MODEL_COLORS["Ridge"]),
        ("ElNet", te_elnet, MODEL_COLORS["ElNet"]),
    ]):
        ax.scatter(y_test, preds_arr, color=col, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_test.min(), preds_arr.min())
        hi = max(y_test.max(), preds_arr.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
        ax.set_title(f"{lbl}\nR²={r2_score(y_test, preds_arr):+.3f}  "
                     f"RMSE={float(np.sqrt(mean_squared_error(y_test, preds_arr))):.3f}",
                     fontsize=10)
        ax.set_xlabel("Actual (original scale)", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Predicted (original scale)", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_id}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Plot -> {path}")


def _plot_r2_barchart(df_results: pd.DataFrame, run: int):
    labels = [d.split("_", 1)[1] if "_" in d else d for d in df_results["dataset"]]
    x, w   = np.arange(len(df_results)), 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df_results) * 1.2), 5))
    ax.bar(x - w, df_results["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
    ax.bar(x,     df_results["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
    ax.bar(x + w, df_results["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test R²", fontsize=11)
    ax.set_title(f"Exp9-SE5 (W1+LogLog)  |  Test R² (run {run})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(-0.15,
                df_results[["OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]].min().min() - 0.05))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"Exp9SE5_r2_comparison_run_{run}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  R² bar chart -> {path}")

# -- Results persistence --------------------------------------------------------
def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"Results -> {RESULTS_FILE}")
    key_cols = ["experiment", "dataset", "log_y", "n_log_features",
                "OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]
    print(df_new[key_cols].to_string(index=False))

# -- Main -----------------------------------------------------------------------
def main():
    run = get_run_number()
    print(f"Linear Modeling  -  Exp9-SE5 (W1+LogLog: 2024-only + log1p features + log1p targets)")
    print(f"Run {run}  |  Datasets: {len(DATASETS)}  |  Models: OLS, Ridge, ElNet")
    print(f"Log-transforming features (in-place): {len(LOG_FEATURES)} concentration columns\n")

    all_results = []
    for experiment, ds_id, path, target, log_y in DATASETS:
        print(f"{'─'*60}")
        print(f"[{experiment}]  {ds_id}")
        print(f"  Target: {target}")

        if not os.path.exists(path):
            print(f"  WARNING: file not found - {path}"); continue

        df_peek  = pd.read_excel(path, nrows=0)
        features = infer_features(df_peek, target)
        print(f"  Features (inferred): {len(features)}")

        results, preds = train_dataset(experiment, ds_id, path, features, target, log_y, run)
        if results is None:
            continue

        append_predictions(path, preds)
        print(f"  Predictions appended -> {path}")
        all_results.append(results)

    if not all_results:
        print("No results produced.")
        return

    df_results = pd.DataFrame(all_results)
    _plot_r2_barchart(df_results, run)
    save_results(all_results, run)
    print(f"\nExp9-SE5 Linear Modeling  -  Run {run} complete.")


if __name__ == "__main__":
    main()
