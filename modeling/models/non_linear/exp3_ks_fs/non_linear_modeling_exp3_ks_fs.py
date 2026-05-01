"""
non_linear_modeling_exp3_ks_fs.py - RF, GB, XGB on Experiment 3 KS-FS (OOF FS).

Reads the all-features datasets from experiment3/sub_exp3/ for Phase 1+2 only:
  Phase 1 : CV search on all KS features → best estimator + full R²/RMSE
  Phase 2 : OOF permutation importance (threshold=0.05) → selected feature mask
  Phase 3 : rebuild dataset from All_Years_Full.xlsx using only selected features
             (fewer missingness constraints → more usable rows), then refit.

Because Phase 3 rebuilds from raw data, n_train / n_test / n_features stored in
results reflect the expanded post-FS dataset, NOT the original all-features row count.
n_train_ks stores the original KS count for reference.

Experiment key: Exp3-KS-FS

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp3_ks_fs/non_linear_modeling_exp3_ks_fs.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(MODELING_DIR, ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")

def _e3s1(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp3", f"{name}.xlsx")

# ── Splits ─────────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
RF_BASE  = dict(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=1)
GB_BASE  = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, random_state=42)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0)

RF_GRID  = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
GB_GRID  = {"max_depth": [2, 3, 4, 5],    "min_samples_leaf": [5, 10, 20]}
XGB_DIST = {"max_depth": [2, 3, 4, 5], "min_child_weight": [5, 10, 20],
            "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.5, 1.0, 5.0]}

# ── Feature inference ──────────────────────────────────────────────────────────
_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [c for c in df.columns
            if c != target
            and c not in _EXCLUDE_COLS
            and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)]

# ── Dataset registry ───────────────────────────────────────────────────────────
REGISTRY = [
    ("Exp3-S3-FS", "grab_BOD", _e3s1("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Exp3-S3-FS", "grab_COD", _e3s1("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Exp3-S3-FS", "grab_TSS", _e3s1("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Exp3-S3-FS", "grab_pH",  _e3s1("grab_pH"),  "Effluent pH (Grab)"),
    ("Exp3-S3-FS", "comp_BOD", _e3s1("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Exp3-S3-FS", "comp_COD", _e3s1("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Exp3-S3-FS", "comp_TSS", _e3s1("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Exp3-S3-FS", "comp_pH",  _e3s1("comp_pH"),  "Effluent pH (Composite)"),
]

YEAR_COLOURS  = {2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
                 2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801"}
MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}

# ── Metric helpers ─────────────────────────────────────────────────────────────
def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
def _r2(yt, yp):   return float(r2_score(yt, yp))


# ── Rebuild dataset from raw after feature selection ───────────────────────────
def _rebuild_from_raw(selected_features: list, target: str,
                      train_years: list, test_year: int):
    """Rebuild train/test from All_Years_Full using only selected_features.

    After OOF selection, some high-missingness features are dropped. Rebuilding
    from raw without those columns unlocks rows that were previously lost due to
    joint missingness, increasing n_train relative to the all-features baseline.

    Returns (train_df, test_df) DataFrames with columns [year] + selected_features
    + [target]. Returns (None, None) if RAW_FILE is missing or columns not found.
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

def _run_number(model_dir: str) -> int:
    rfile = os.path.join(model_dir, "results.xlsx")
    if not os.path.exists(rfile):
        return 1
    df = pd.read_excel(rfile)
    return int(df["run"].max()) + 1 if "run" in df.columns else 1


# ── OOF permutation importance feature selection ───────────────────────────────
def _oof_perm_select(fitted_estimator, X_tr, y_tr, features, tscv, threshold=0.05):
    fold_imps = np.zeros(len(features))
    for tr_idx, val_idx in tscv.split(X_tr):
        est = clone(fitted_estimator)
        est.fit(X_tr[tr_idx], y_tr[tr_idx])
        perm = permutation_importance(
            est, X_tr[val_idx], y_tr[val_idx],
            n_repeats=10, random_state=42, n_jobs=1)
        fold_imps += perm.importances_mean.clip(min=0)
    fold_imps /= tscv.n_splits
    total     = fold_imps.sum()
    norm_imps = fold_imps / total if total > 0 else fold_imps
    mask      = norm_imps >= threshold
    if mask.sum() == 0:
        mask = np.ones(len(features), dtype=bool)
    selected = [f for f, m in zip(features, mask) if m]
    return mask, selected, norm_imps


# ── Plotting ───────────────────────────────────────────────────────────────────
def _plot_scatter(plots_dir, name, model_tag, run, test_df, y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for yr in sorted(test_df["year"].unique()):
        m = test_df["year"].values == yr
        ax.scatter(y_test[m], y_pred[m], color=YEAR_COLOURS.get(yr, "#888"),
                   alpha=0.65, s=25, label=str(yr), edgecolors="none")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
    ax.set_title(f"{name} · {model_tag}\nR²={_r2(y_test,y_pred):+.3f}  "
                 f"RMSE={_rmse(y_test,y_pred):.3f}", fontsize=11)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)

def _plot_importance(plots_dir, name, model_tag, run, model, features):
    imps  = model.feature_importances_
    order = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.35)))
    ax.barh([features[i] for i in order], imps[order],
            color=MODEL_COLOURS.get(model_tag, "#888"))
    ax.set_title(f"{name} · {model_tag}  -  Feature Importance (run {run})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=150); plt.close(fig)


# ── Core training (3-phase OOF FS) ────────────────────────────────────────────
def train_one(experiment, name, subset_path, features, target,
              model_tag, estimator, param_grid, search_cls, model_dir, plots_dir, run):

    df = pd.read_excel(subset_path, parse_dates=["Date"])
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()
    test = df[df["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"  WARNING: no {TEST_YEAR} rows - skipping {name}")
        return None

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing} - skipping {name}")
        return None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values
    n_in = len(features)
    print(f"    n_train={len(train)} n_test={len(test)} n_features={n_in}")

    # Phase 1: CV search on full feature set
    print(f"    Phase 1  -  CV search ({n_in} features)...", end="", flush=True)
    gs1 = search_cls(estimator, param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True,
                     **({} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}))
    gs1.fit(X_tr, y_tr)
    best1         = gs1.best_estimator_
    cv_rmse1      = float(-gs1.best_score_)
    te_pred_full  = best1.predict(X_te)
    r2_test_full  = _r2(y_te, te_pred_full)
    r2_train_full = _r2(y_tr, best1.predict(X_tr))
    r2_gap_full   = r2_train_full - r2_test_full
    print(f"  CV_RMSE={cv_rmse1:.3f}  R²_test_full={r2_test_full:+.3f}")

    # Phase 2: OOF permutation importance selection
    print(f"    Phase 2  -  OOF selection...", end="", flush=True)
    oof_mask, selected_nl, norm_imps = _oof_perm_select(
        best1, X_tr, y_tr, features, TSCV, threshold=0.05)
    n_sel      = int(oof_mask.sum())
    dropped_nl = [f for f, m in zip(features, oof_mask) if not m]
    print(f"  {n_sel}/{n_in} features kept")
    if dropped_nl:
        print(f"    Dropped: {dropped_nl}")

    # Phase 3: rebuild dataset from All_Years_Full using only selected features,
    # then refit. This unlocks rows lost due to joint missingness of dropped features.
    print(f"    Phase 3  -  rebuilding from raw ({n_sel} features)...", end="", flush=True)
    train_r, test_r = _rebuild_from_raw(selected_nl, target, TRAIN_YEARS, TEST_YEAR)

    if train_r is not None and len(train_r) > 0 and len(test_r) > 0:
        X_tr_sel = train_r[selected_nl].values
        y_tr_fit = train_r[target].values
        X_te_sel = test_r[selected_nl].values
        y_te_fit = test_r[target].values
        n_train_final = len(train_r)
        n_test_final  = len(test_r)
        print(f"  n_train {len(train)}→{n_train_final}, n_test {len(test)}→{n_test_final}")
    else:
        print(f"  rebuild failed  -  falling back to KS subset")
        X_tr_sel = X_tr[:, oof_mask]
        X_te_sel = X_te[:, oof_mask]
        y_tr_fit = y_tr
        y_te_fit = y_te
        n_train_final = len(train)
        n_test_final  = len(test)

    gs2 = search_cls(estimator, param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True,
                     **({} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}))
    gs2.fit(X_tr_sel, y_tr_fit)
    best    = gs2.best_estimator_
    cv_rmse = float(-gs2.best_score_)
    print(f"  CV_RMSE={cv_rmse:.3f}")

    tr_pred  = best.predict(X_tr_sel)
    te_pred  = best.predict(X_te_sel)
    # Predictions stored back into the KS xlsx (original dataset rows)
    all_pred = best.predict(df[selected_nl].values)

    row = {
        "experiment":           experiment,
        "target":               target,
        "model":                model_tag,
        "run":                  run,
        "n_train":              n_train_final,
        "n_test":               n_test_final,
        "n_train_ks":           len(train),
        "n_test_ks":            len(test),
        "n_features_input":     n_in,
        "n_selected_nl":        n_sel,
        "selected_features_nl": ", ".join(selected_nl),
        "dropped_features_nl":  ", ".join(dropped_nl),
        "n_features":           n_sel,
        "CV_RMSE_initial":      round(cv_rmse1, 4),
        "R2_test_full":         round(r2_test_full,  4),
        "R2_train_full":        round(r2_train_full, 4),
        "R2_gap_full":          round(r2_gap_full,   4),
        "CV_RMSE":              round(cv_rmse, 4),
        "R2_train":             _r2(y_tr_fit, tr_pred),
        "R2_test":              _r2(y_te_fit, te_pred),
        "R2_gap":               _r2(y_tr_fit, tr_pred) - _r2(y_te_fit, te_pred),
        "RMSE_train":           _rmse(y_tr_fit, tr_pred),
        "RMSE_test":            _rmse(y_te_fit, te_pred),
        "MAE_test":             _mae(y_te_fit, te_pred),
        "best_params":          str(gs2.best_params_),
    }
    print(f"    R²_train={row['R2_train']:+.3f}  R²_test={row['R2_test']:+.3f}  "
          f"gap={row['R2_gap']:+.3f}  RMSE_test={row['RMSE_test']:.3f}")

    joblib.dump({
        "model":               best,
        "selected_features":   selected_nl,
        "feature_mask":        oof_mask,
        "all_features":        features,
        "norm_oof_importance": dict(zip(features, norm_imps.tolist())),
    }, os.path.join(model_dir, f"{name}_{model_tag}_run_{run}.pkl"))
    test_for_plot = test_r if (train_r is not None and len(test_r) > 0) else test
    y_te_for_plot = y_te_fit
    _plot_scatter(plots_dir, name, model_tag, run, test_for_plot, y_te_for_plot, te_pred, target)
    _plot_importance(plots_dir, name, model_tag, run, best, selected_nl)

    df_sub = pd.read_excel(subset_path)
    df_sub[f"predicted_{model_tag}_run_{run}"] = np.round(all_pred, 3)
    df_sub.to_excel(subset_path, index=False)

    return row


# ── Per-model runner ───────────────────────────────────────────────────────────
def run_model(model_tag: str, estimator, param_grid, search_cls):
    model_dir = os.path.join(SCRIPT_DIR, model_tag.lower())
    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    run = _run_number(model_dir)
    print(f"\n{'='*60}")
    print(f"{model_tag}  -  Exp3-KS-FS (OOF FS on all-features, run {run})")
    print(f"{'='*60}")

    results = []
    for experiment, name, path, target in REGISTRY:
        print(f"\n  [{experiment}]  {name}  →  {target}")
        if not os.path.exists(path):
            print(f"    WARNING: file not found - {path}"); continue

        df_peek  = pd.read_excel(path, nrows=0)
        features = infer_features(df_peek, target)
        print(f"    Kitchen-sink features: {len(features)}")

        row = train_one(experiment, name, path, features, target,
                        model_tag, estimator, param_grid, search_cls,
                        model_dir, plots_dir, run)
        if row:
            results.append(row)

    if not results:
        return

    df_new = pd.DataFrame(results)
    rfile  = os.path.join(model_dir, "results.xlsx")
    if os.path.exists(rfile):
        df_out = pd.concat([pd.read_excel(rfile), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(rfile, index=False)
    print(f"\n  Results → {rfile}")
    print(df_new[["target", "R2_test_full", "R2_test", "R2_gap",
                  "n_selected_nl", "RMSE_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Non-Linear Modeling  -  Exp3-KS-FS (RF / GB / XGB)")
    print("Kitchen-sink features · OOF permutation importance FS (3-phase)\n")

    run_model("RF",  RandomForestRegressor(**RF_BASE),     RF_GRID,  GridSearchCV)
    run_model("GB",  GradientBoostingRegressor(**GB_BASE), GB_GRID,  GridSearchCV)
    run_model("XGB", XGBRegressor(**XGB_BASE),             XGB_DIST, RandomizedSearchCV)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
