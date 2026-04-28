"""
non_linear_modeling_exp2_s2.py

Trains RF, GradientBoosting, and XGBoost on Experiment 2 Sub-experiment 2 datasets.
Calendar features use cyclic sin/cos encoding (standard for Exp2 onwards).
Includes OOF permutation importance feature selection (3-phase: full → select → refit).

Feature set (21 features per target):
  Grab : Inlet pH/BOD/COD/TSS (Grab, 4) + all SEC_COLS (10) + COMMON_CYCLIC (7) = 21
  Comp : Inlet pH/BOD/COD/TSS (Comp, 4) + all SEC_COLS (10) + COMMON_CYCLIC (7) = 21

Stores R2_test_full, R2_train_full, R2_gap_full, n_selected_nl, selected_features_nl
in addition to post-selection R2_test, R2_train, R2_gap.

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp2_s2/non_linear_modeling_exp2_s2.py
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

# ── Splits & fixed hyperparameters ────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)

RF_BASE  = dict(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=1)
GB_BASE  = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, random_state=42)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=1)

RF_GRID  = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
GB_GRID  = {"max_depth": [2, 3, 4, 5],    "min_samples_leaf": [5, 10, 20]}
XGB_DIST = {
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [5, 10, 20],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 5.0],
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
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp2",
                        f"{name}.xlsx")


REGISTRY = [
    ("Exp2-Sub2", "grab_BOD", _cyc("grab_BOD"), E2S2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub2", "grab_COD", _cyc("grab_COD"), E2S2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub2", "grab_TSS", _cyc("grab_TSS"), E2S2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub2", "grab_pH",  _cyc("grab_pH"),  E2S2_GRAB, "Effluent pH (Grab)"),
    ("Exp2-Sub2", "comp_BOD", _cyc("comp_BOD"), E2S2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub2", "comp_COD", _cyc("comp_COD"), E2S2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub2", "comp_TSS", _cyc("comp_TSS"), E2S2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub2", "comp_pH",  _cyc("comp_pH"),  E2S2_COMP, "Effluent pH (Composite)"),
]

MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}
YEAR_COLOURS  = {2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
                 2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801"}

# ── Metric helpers ─────────────────────────────────────────────────────────────
def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
def _r2(yt, yp):   return float(r2_score(yt, yp))

def _run_number(model_dir: str) -> int:
    rfile = os.path.join(model_dir, "results.xlsx")
    if not os.path.exists(rfile):
        return 1
    df = pd.read_excel(rfile)
    return int(df["run"].max()) + 1 if "run" in df.columns else 1


# ── OOF permutation importance feature selection ────────────────────────────────

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
    ax.set_title(f"{name} · {model_tag}\nR²={_r2(y_test,y_pred):+.3f}  RMSE={_rmse(y_test,y_pred):.3f}",
                 fontsize=11)
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
    ax.set_title(f"{name} · {model_tag} — Feature Importance (run {run})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=150); plt.close(fig)


# ── Core training (one model type) ────────────────────────────────────────────

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

    # Phase 1: initial CV search on full feature set
    print(f"    Phase 1 — CV search (all {n_in} features)...", end="", flush=True)
    gs1 = search_cls(estimator, param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True,
                     **({} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}))
    gs1.fit(X_tr, y_tr)
    best1    = gs1.best_estimator_
    cv_rmse1 = float(-gs1.best_score_)
    te_pred_full  = best1.predict(X_te)
    r2_test_full  = _r2(y_te, te_pred_full)
    r2_train_full = _r2(y_tr, best1.predict(X_tr))
    r2_gap_full   = r2_train_full - r2_test_full
    print(f"  CV_RMSE={cv_rmse1:.3f}  R²_test_full={r2_test_full:+.3f}")

    # Phase 2: OOF permutation importance selection
    print(f"    Phase 2 — OOF selection...", end="", flush=True)
    oof_mask, selected_nl, norm_imps = _oof_perm_select(
        best1, X_tr, y_tr, features, TSCV, threshold=0.05)
    n_sel = int(oof_mask.sum())
    dropped_nl = [f for f, m in zip(features, oof_mask) if not m]
    print(f"  {n_sel}/{n_in} features kept")
    if dropped_nl:
        print(f"    Dropped: {dropped_nl}")

    X_tr_sel = X_tr[:, oof_mask]
    X_te_sel = X_te[:, oof_mask]

    # Phase 3: refit on selected features
    print(f"    Phase 3 — refit ({n_sel} features)...", end="", flush=True)
    gs2 = search_cls(estimator, param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True,
                     **({} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}))
    gs2.fit(X_tr_sel, y_tr)
    best    = gs2.best_estimator_
    cv_rmse = float(-gs2.best_score_)
    print(f"  CV_RMSE={cv_rmse:.3f}")

    tr_pred  = best.predict(X_tr_sel)
    te_pred  = best.predict(X_te_sel)
    all_pred = best.predict(df[selected_nl].values)

    row = {
        "experiment":           experiment,
        "target":               target,
        "model":                model_tag,
        "run":                  run,
        "n_train":              len(train),
        "n_test":               len(test),
        "n_features_input":     n_in,
        "n_selected_nl":        n_sel,
        "selected_features_nl": ", ".join(selected_nl),
        "dropped_features_nl":  ", ".join(dropped_nl),
        "n_features":           n_sel,
        "CV_RMSE_initial":      round(cv_rmse1, 4),
        "R2_test_full":         round(r2_test_full,  4),
        "R2_train_full":        round(r2_train_full, 4),
        "R2_gap_full":          round(r2_gap_full,   4),
        "CV_RMSE":              round(cv_rmse,  4),
        "R2_train":             _r2(y_tr, tr_pred),
        "R2_test":              _r2(y_te, te_pred),
        "R2_gap":               _r2(y_tr, tr_pred) - _r2(y_te, te_pred),
        "RMSE_train":           _rmse(y_tr, tr_pred),
        "RMSE_test":            _rmse(y_te, te_pred),
        "MAE_test":             _mae(y_te, te_pred),
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
    _plot_scatter(plots_dir, name, model_tag, run, test, y_te, te_pred, target)
    _plot_importance(plots_dir, name, model_tag, run, best, selected_nl)

    # Append predictions to dataset file
    df_sub = pd.read_excel(subset_path)
    col    = f"predicted_{model_tag}_run_{run}"
    df_sub[col] = np.round(all_pred, 3)
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
    print(f"{model_tag} — Exp2-Sub2  (run {run})")
    print(f"{'='*60}")

    results = []
    for experiment, name, path, features, target in REGISTRY:
        print(f"\n  [{experiment}]  {name}  →  {target}")
        if not os.path.exists(path):
            print(f"    WARNING: file not found - {path}")
            print(f"    Run make_sub2_cyclic_datasets.py first.")
            continue
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
    print(df_new[["target", "R2_train", "R2_test", "R2_gap", "RMSE_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Non-Linear Modeling — Exp2-Sub2  (RF / GB / XGB)")
    print("21 features · cyclic calendar · OOF permutation importance FS (3-phase)\n")

    run_model("RF",  RandomForestRegressor(**RF_BASE),     RF_GRID,  GridSearchCV)
    run_model("GB",  GradientBoostingRegressor(**GB_BASE), GB_GRID,  GridSearchCV)
    run_model("XGB", XGBRegressor(**XGB_BASE),             XGB_DIST, RandomizedSearchCV)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
