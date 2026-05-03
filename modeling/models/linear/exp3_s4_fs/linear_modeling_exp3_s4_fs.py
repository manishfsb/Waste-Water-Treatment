"""
linear_modeling_exp3_s4_fs.py - OLS (LassoCV FS), Ridge, ElNet on Experiment 3 KS-FS.

Reads all-features datasets from experiment3/sub_exp4/ for LassoCV screening, then:
  - OLS  : LassoCV selects features → rebuild from All_Years_Full with only those
           features (fewer missingness constraints → more usable rows) → final OLS fit.
           n_train / n_test in results reflect the expanded post-FS dataset.
           Pre-FS metrics stored as OLS_full_* / R2_test_full.
  - Ridge: full KS feature set (L2 handles collinearity; no row expansion possible).
  - ElNet: full KS feature set (L1+L2 selects internally; n_selected logged).

n_train_ks stores the original all-features row count for Ridge/ElNet reference.

Experiment key: Exp3-S4-FS

Usage (from project root):
    .venv/bin/python3 modeling/models/linear/exp3_s4_fs/linear_modeling_exp3_s4_fs.py
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
PROJECT_ROOT = os.path.abspath(os.path.join(MODELING_DIR, ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
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

# ── Dataset registry ───────────────────────────────────────────────────────────
def _e3ks(name):
    # Reads all-features (Sub-3) datasets
    return os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp4", f"{name}.xlsx")


DATASETS = [
    ("Exp3-S4-FS", "E3S3FS_Grab_BOD", _e3ks("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Exp3-S4-FS", "E3S3FS_Grab_COD", _e3ks("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Exp3-S4-FS", "E3S3FS_Grab_TSS", _e3ks("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Exp3-S4-FS", "E3S3FS_Grab_pH",  _e3ks("grab_pH"),  "Effluent pH (Grab)"),
    ("Exp3-S4-FS", "E3S3FS_Comp_BOD", _e3ks("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Exp3-S4-FS", "E3S3FS_Comp_COD", _e3ks("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Exp3-S4-FS", "E3S3FS_Comp_TSS", _e3ks("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Exp3-S4-FS", "E3S3FS_Comp_pH",  _e3ks("comp_pH"),  "Effluent pH (Composite)"),
]

# ── Feature inference ──────────────────────────────────────────────────────────
_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [c for c in df.columns
            if c != target
            and c not in _EXCLUDE_COLS
            and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)]


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


# ── Rebuild dataset from raw after LassoCV selection ──────────────────────────
def _rebuild_from_raw(selected_features: list, target: str,
                      train_years: list, test_year: int):
    """Rebuild train/test from All_Years_Full using only selected_features.

    After LassoCV selection, some high-missingness features are dropped. Rebuilding
    from raw without those columns unlocks rows previously lost to joint missingness,
    giving OLS more training data than the all-features baseline.

    Returns (train_df, test_df). Returns (None, None) on missing file or columns.
    Ridge/ElNet do NOT call this because they use the full KS feature set.
    """
    if not os.path.exists(RAW_FILE):
        print(f"    WARNING: raw file not found  -  {RAW_FILE}")
        return None, None

    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])
    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    missing_cols = [c for c in selected_features + [target] if c not in df.columns]
    if missing_cols:
        print(f"    WARNING: columns not in raw data  -  {missing_cols}")
        return None, None

    sub = df[["year"] + selected_features + [target]].dropna(
        subset=selected_features + [target])

    train = sub[sub["year"].isin(train_years)].copy()
    extra = sub[sub["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()
    test = sub[sub["year"] == test_year].copy()
    return train, test


# ── LassoCV feature selection for OLS ─────────────────────────────────────────
def _lasso_select(X_tr_sc: np.ndarray, y_train: np.ndarray,
                  features: list, tscv) -> tuple:
    lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42, n_jobs=-1)
    lasso.fit(X_tr_sc, y_train)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        print("    LassoCV: all features zeroed  -  keeping full set")
        mask = np.ones(len(features), dtype=bool)
    n_in, n_kept = len(features), int(mask.sum())
    print(f"    LassoCV pre-screen → {n_kept}/{n_in} features kept", end="")
    dropped = [f for f, m in zip(features, mask) if not m]
    if dropped:
        print(f"  -  dropped: {dropped}")
    else:
        print(" (no pruning)")
    selected = [f for f, m in zip(features, mask) if m]
    return mask, selected


# ── Run-number detection ───────────────────────────────────────────────────────
def get_run_number() -> int:
    if not os.path.exists(RESULTS_FILE):
        return 1
    df = pd.read_excel(RESULTS_FILE)
    if "run" not in df.columns or df.empty:
        return 1
    return int(df["run"].max()) + 1


# ── Append predictions to dataset file ────────────────────────────────────────
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
    n_in    = len(features)

    print(f"  Train: {len(train_df):>4d} rows | Test: {len(test_df):>3d} rows | "
          f"Features: {n_in}")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)
    tscv     = TimeSeriesSplit(n_splits=3)

    # ── LassoCV feature selection for OLS ─────────────────────────────────────
    ols_mask, selected_ols = _lasso_select(X_tr_sc, y_train, features, tscv)
    n_sel_ols = int(ols_mask.sum())

    # ── Rebuild dataset from All_Years_Full with only OLS-selected features ────
    # Ridge/ElNet use the full KS set, so they can't benefit from row expansion.
    train_ols, test_ols = _rebuild_from_raw(selected_ols, target, TRAIN_YEARS, TEST_YEAR)

    if train_ols is not None and len(train_ols) > 0 and len(test_ols) > 0:
        n_train_ols = len(train_ols)
        n_test_ols  = len(test_ols)
        scaler_ols  = StandardScaler()
        X_tr_ols    = scaler_ols.fit_transform(train_ols[selected_ols].values)
        X_te_ols    = scaler_ols.transform(test_ols[selected_ols].values)
        y_train_ols = train_ols[target].values
        y_test_ols  = test_ols[target].values
        # For KS xlsx predictions: apply new scaler to KS feature subset
        X_all_ols   = scaler_ols.transform(df[selected_ols].values)
        print(f"  OLS rebuild: {len(train_df)}→{n_train_ols} train rows, "
              f"{len(test_df)}→{n_test_ols} test rows ({n_sel_ols} features)")
    else:
        print(f"  OLS rebuild failed  -  falling back to KS subset")
        n_train_ols = len(train_df)
        n_test_ols  = len(test_df)
        scaler_ols  = scaler
        X_tr_ols    = X_tr_sc[:, ols_mask]
        X_te_ols    = X_te_sc[:, ols_mask]
        X_all_ols   = X_all_sc[:, ols_mask]
        y_train_ols = y_train
        y_test_ols  = y_test

    results = {
        "experiment":            experiment,
        "dataset":               ds_id,
        "target":                target,
        "run":                   run,
        "n_train":               n_train_ols,
        "n_test":                n_test_ols,
        "n_train_ks":            len(train_df),
        "n_test_ks":             len(test_df),
        "n_features":            n_in,
        "n_features_input":      n_in,
        "n_selected_ols":        n_sel_ols,
        "selected_features_ols": ", ".join(selected_ols),
    }
    preds = {}

    # ── OLS full (pre-LassoCV on KS data, for comparison) ────────────────────
    ols_full = LinearRegression()
    ols_full.fit(X_tr_sc, y_train)
    tr_ols_full = ols_full.predict(X_tr_sc)
    te_ols_full = ols_full.predict(X_te_sc)
    results["OLS_full_train_R2"]  = float(r2_score(y_train, tr_ols_full))
    results["OLS_full_test_R2"]   = float(r2_score(y_test,  te_ols_full))
    results["OLS_full_test_RMSE"] = _rmse(y_test, te_ols_full)
    results["OLS_full_R2_gap"]    = (results["OLS_full_train_R2"]
                                     - results["OLS_full_test_R2"])
    results["R2_test_full"]  = results["OLS_full_test_R2"]
    results["RMSE_test_full"]= results["OLS_full_test_RMSE"]
    results["R2_gap_full"]   = results["OLS_full_R2_gap"]
    print(f"    OLS_full - Train R²: {results['OLS_full_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_full_test_R2']:+.3f}  (all {n_in} features, KS data)")

    # ── OLS (LassoCV-selected, rebuilt from raw) ──────────────────────────────
    ols = LinearRegression()
    ols.fit(X_tr_ols, y_train_ols)
    tr_ols = ols.predict(X_tr_ols)
    te_ols = ols.predict(X_te_ols)
    results.update(_metrics(y_train_ols, tr_ols, "OLS_train"))
    results.update(_metrics(y_test_ols,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    preds[f"predicted_OLS_run_{run}"] = np.round(ols.predict(X_all_ols), 3)
    joblib.dump({"scaler": scaler_ols, "model": ols,
                 "selected_features": selected_ols, "feature_mask": ols_mask},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    - Train R²: {results['OLS_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_test_R2']:+.3f} | "
          f"RMSE: {results['OLS_test_RMSE']:.3f}  ({n_sel_ols}/{n_in} via LassoCV, "
          f"n_train={n_train_ols})")

    # ── Ridge (full feature set, L2 handles collinearity) ─────────────────────
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

    # ── ElasticNet (full feature set, L1+L2 selects internally) ──────────────
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
    elnet_mask = elnet.coef_ != 0
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

    _plot_comparison(ds_id, run, y_test_ols, y_test, te_ols, te_ridge, te_elnet)
    return results, preds


# ── Plotting ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {"OLS": "#E15252", "Ridge": "#4A90D9", "ElNet": "#5BAD6F"}


def _plot_comparison(ds_id, run, y_test_ols, y_test_lin, te_ols, te_ridge, te_elnet):
    # OLS is evaluated on the rebuilt (expanded) dataset; Ridge/ElNet on the KS set.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{ds_id}  |  Test Set  -  LassoCV FS / Full  (run {run})",
                 fontsize=12, fontweight="bold")
    for ax, (lbl, y_t, preds_arr, col) in zip(axes, [
        ("OLS (LassoCV FS)", y_test_ols,  te_ols,   MODEL_COLORS["OLS"]),
        ("Ridge (full)",     y_test_lin,  te_ridge, MODEL_COLORS["Ridge"]),
        ("ElNet (full)",     y_test_lin,  te_elnet, MODEL_COLORS["ElNet"]),
    ]):
        ax.scatter(y_t, preds_arr, color=col, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_t.min(), preds_arr.min())
        hi = max(y_t.max(), preds_arr.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
        ax.set_title(f"{lbl}\nR²={r2_score(y_t, preds_arr):+.3f}  "
                     f"RMSE={_rmse(y_t, preds_arr):.3f}", fontsize=10)
        ax.set_xlabel("Actual", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Predicted", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_id}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Plot → {path}")


def _plot_r2_barchart(df_results: pd.DataFrame, run: int):
    labels = [d.split("_", 1)[1] if "_" in d else d for d in df_results["dataset"]]
    x, w = np.arange(len(df_results)), 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df_results) * 1.2), 5))
    ax.bar(x - w, df_results["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
    ax.bar(x,     df_results["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
    ax.bar(x + w, df_results["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test R²", fontsize=11)
    ax.set_title(f"Exp3-S4-FS (LassoCV FS)  |  Test R² Comparison (run {run})", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"Exp3-S4-FS_r2_comparison_run_{run}.png")
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
                "OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]
    print(df_new[key_cols].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    run = get_run_number()
    print(f"Linear Modeling  -  Exp3-S4-FS (LassoCV FS on all-features features)")
    print(f"Run {run}  |  Datasets: {len(DATASETS)}  |  Models: OLS, Ridge, ElNet\n")

    all_results = []
    for experiment, ds_id, path, target in DATASETS:
        print(f"{'─'*60}")
        print(f"[{experiment}]  {ds_id}")
        print(f"  Target: {target}")

        if not os.path.exists(path):
            print(f"  WARNING: file not found - {path}"); continue

        df_peek  = pd.read_excel(path, nrows=0)
        features = infer_features(df_peek, target)
        print(f"  Kitchen-sink features: {len(features)}")

        results, preds = train_dataset(experiment, ds_id, path, features, target, run)
        if results is None:
            continue
        all_results.append(results)
        append_predictions(path, preds)

    if all_results:
        df_res = pd.DataFrame(all_results)
        save_results(all_results, run)
        _plot_r2_barchart(df_res, run)

    print(f"\nExp3-S4-FS Linear Modeling  -  Run {run} complete")


if __name__ == "__main__":
    main()
