"""
non_linear_modeling.py - Unified tree-ensemble training with hyperparameter tuning.

Trains Random Forest, Gradient Boosting, and XGBoost using the same CV protocol
as linear_modeling.py (GridSearchCV / TimeSeriesSplit n_splits=3, scored on
neg_root_mean_squared_error) for a fair apples-to-apples comparison.

  Experiment 1          - Inlet features only          (grab + composite)
  Experiment 2 Sub-1    - Secondary features only       (grab + composite)
  Experiment 2 Sub-2    - Inlet + Secondary features    (grab + composite)

Train = 2020-2024 (2020 rows included where available)  |  Test = 2025

Tuned parameters (via cross-validation):
  RF  - max_depth [4,6,8,None], min_samples_leaf [5,10,20]         GridSearchCV  (12 combos)
  GB  - max_depth [2,3,4,5],    min_samples_leaf [5,10,20]         GridSearchCV  (12 combos)
  XGB - max_depth [2,3,4,5],    min_child_weight [5,10,20],
         reg_alpha [0,0.1,1],    reg_lambda [0.5,1,5]              RandomizedSearchCV (n_iter=30)

Fixed (architectural / learning-rate schedule):
  All - n_estimators=300
  GB/XGB - learning_rate=0.05, subsample=0.8
  XGB    - colsample_bytree=0.8

Outputs per model (rf / gb / xgb):
  non_linear_modeling/{model}/results.xlsx
  non_linear_modeling/{model}/models/{name}_{MODEL}_run_N.pkl
  non_linear_modeling/{model}/plots/{name}_{MODEL}_run_N_{scatter|timeseries|importance}.png

Usage (from project root):
  .venv/bin/python3 21-25/modeling/non_linear_modeling/non_linear_modeling.py
"""

import os
import sys
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
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

_STAGE_DIRS = {
    "experiment1":    os.path.join("datasets", "experiment1", "sub_exp2"),
    "experiment2_s1": os.path.join("datasets", "experiment2", "sub_exp1"),
    "experiment2_s2": os.path.join("datasets", "experiment2", "sub_exp2"),
}

def _sub(stage_dir, name):
    """Resolve path to an existing subset Excel file."""
    return os.path.join(MODELING_DIR, _STAGE_DIRS[stage_dir], f"{name}.xlsx")

# ── Splits ─────────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]   # base years; 2020 rows appended where present
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)   # identical protocol to linear_modeling.py

# ── Fixed architectural hyperparameters ────────────────────────────────────────
RF_BASE = dict(
    n_estimators=300, max_features="sqrt", random_state=42,
    n_jobs=1,           # n_jobs on estimator set to 1; GridSearchCV uses n_jobs=-1
)
GB_BASE = dict(
    n_estimators=300, learning_rate=0.05,
    subsample=0.8, random_state=42,
)
XGB_BASE = dict(
    n_estimators=300, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    n_jobs=1, verbosity=0,
)

# ── Hyperparameter search grids ────────────────────────────────────────────────
RF_GRID = {
    "max_depth":        [4, 6, 8, None],
    "min_samples_leaf": [5, 10, 20],
}   # 12 combos × 3 CV folds = 36 fits per dataset

GB_GRID = {
    "max_depth":        [2, 3, 4, 5],
    "min_samples_leaf": [5, 10, 20],
}   # 12 combos × 3 CV folds = 36 fits per dataset

XGB_DIST = {
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [5, 10, 20],
    "reg_alpha":        [0.0, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 5.0],
}   # 108 combos → RandomizedSearchCV n_iter=30

# ── Feature sets ───────────────────────────────────────────────────────────────
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
              "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
              "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
              "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)"]
COMMON        = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]
COMMON_CYCLIC = ["Flow (MLD)", "Power Total (KW)", "year",
                 "month_sin", "month_cos", "dow_sin", "dow_cos"]

S1_GRAB   = GRAB_INLET + COMMON_CYCLIC      # 11 features (cyclic calendar)
S1_COMP   = COMP_INLET + COMMON_CYCLIC      # 11 features (cyclic calendar)
S2P1      = SEC_COLS   + COMMON             # 15 features
S2P2_GRAB = GRAB_INLET + SEC_COLS + COMMON  # 19 features
S2P2_COMP = COMP_INLET + SEC_COLS + COMMON  # 19 features

# ── Model registry ─────────────────────────────────────────────────────────────
# (experiment_label, model_name, subset_path, features, target)
REGISTRY = [
    # Experiment 1 - grab
    ("Experiment 1", "grab_BOD", _sub("experiment1", "grab_BOD"),
     S1_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 1", "grab_COD", _sub("experiment1", "grab_COD"),
     S1_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Experiment 1", "grab_TSS", _sub("experiment1", "grab_TSS"),
     S1_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 1", "grab_pH",  _sub("experiment1", "grab_pH"),
     S1_GRAB, "Effluent pH (Grab)"),
    # Experiment 1 - composite
    ("Experiment 1", "comp_BOD", _sub("experiment1", "comp_BOD"),
     S1_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 1", "comp_COD", _sub("experiment1", "comp_COD"),
     S1_COMP, "Effluent COD (mg/L, Composite)"),
    ("Experiment 1", "comp_TSS", _sub("experiment1", "comp_TSS"),
     S1_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 1", "comp_pH",  _sub("experiment1", "comp_pH"),
     S1_COMP, "Effluent pH (Composite)"),
    # Experiment 2 Sub-1 - grab
    ("Experiment 2 Sub-1", "grab_BOD",
     _sub("experiment2_s1","grab_BOD"), S2P1, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "grab_COD",
     _sub("experiment2_s1","grab_COD"), S2P1, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "grab_TSS",
     _sub("experiment2_s1","grab_TSS"), S2P1, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "grab_pH",
     _sub("experiment2_s1","grab_pH"),  S2P1, "Effluent pH (Grab)"),
    # Experiment 2 Sub-1 - composite
    ("Experiment 2 Sub-1", "comp_BOD",
     _sub("experiment2_s1","comp_BOD"), S2P1, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "comp_COD",
     _sub("experiment2_s1","comp_COD"), S2P1, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "comp_TSS",
     _sub("experiment2_s1","comp_TSS"), S2P1, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "comp_pH",
     _sub("experiment2_s1","comp_pH"),  S2P1, "Effluent pH (Composite)"),
    # Experiment 2 Sub-2 - grab
    ("Experiment 2 Sub-2", "grab_BOD",
     _sub("experiment2_s2","grab_BOD"), S2P2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "grab_COD",
     _sub("experiment2_s2","grab_COD"), S2P2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "grab_TSS",
     _sub("experiment2_s2","grab_TSS"), S2P2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "grab_pH",
     _sub("experiment2_s2","grab_pH"),  S2P2_GRAB, "Effluent pH (Grab)"),
    # Experiment 2 Sub-2 - composite
    ("Experiment 2 Sub-2", "comp_BOD",
     _sub("experiment2_s2","comp_BOD"), S2P2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "comp_COD",
     _sub("experiment2_s2","comp_COD"), S2P2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "comp_TSS",
     _sub("experiment2_s2","comp_TSS"), S2P2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "comp_pH",
     _sub("experiment2_s2","comp_pH"),  S2P2_COMP, "Effluent pH (Composite)"),
]

# ── Year / model colours ───────────────────────────────────────────────────────
YEAR_COLOURS = {2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
                2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801"}
MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}

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


# ── OOF permutation importance feature selection ────────────────────────────

def _oof_perm_select(fitted_estimator, X_tr: np.ndarray, y_tr: np.ndarray,
                     features: list, tscv,
                     threshold: float = 0.05) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Compute permutation importance on out-of-fold (validation) splits.

    Unlike training-set permutation importance (current broken approach),
    this evaluates each feature's contribution on data the model has NOT seen,
    giving an unbiased, generalization-aware importance estimate.

    Steps:
      1. For each CV fold: clone the fitted estimator (same hyperparams, unfitted),
         train on train_fold, compute perm. importance on val_fold.
      2. Average importance across folds, normalize to sum=1.
      3. Drop features with normalized importance < threshold.
      4. Fallback: if all features are dropped, return the full set.

    Returns:
      mask       - boolean array (True = keep)
      selected   - list of selected feature names
      norm_imps  - normalized OOF importance per feature (for logging)
    """
    fold_imps = np.zeros(len(features))
    for tr_idx, val_idx in tscv.split(X_tr):
        est = clone(fitted_estimator)       # same hyperparams, freshly unfitted
        est.fit(X_tr[tr_idx], y_tr[tr_idx])
        perm = permutation_importance(
            est, X_tr[val_idx], y_tr[val_idx],
            n_repeats=10, random_state=42, n_jobs=1,
        )
        fold_imps += perm.importances_mean.clip(min=0)

    fold_imps /= tscv.n_splits
    total     = fold_imps.sum()
    norm_imps = fold_imps / total if total > 0 else fold_imps
    mask      = norm_imps >= threshold

    if mask.sum() == 0:
        print("    OOF selection: all features below threshold  -  keeping full set")
        mask = np.ones(len(features), dtype=bool)

    selected = [f for f, m in zip(features, mask) if m]
    return mask, selected, norm_imps

# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_scatter(plots_dir, name, model_tag, run, test_df, y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for yr in sorted(test_df["year"].unique()):
        m = test_df["year"].values == yr
        ax.scatter(y_test[m], y_pred[m],
                   color=YEAR_COLOURS.get(yr, "#999"), label=str(yr),
                   alpha=0.75, s=28, edgecolors="none")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(f"Actual {target}", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.set_title(f"{model_tag} | {name}\nTest scatter (run {run})", fontsize=10)
    ax.legend(title="Year", fontsize=8)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_timeseries(plots_dir, name, model_tag, run, df_all, features, model, target):
    df_plot = df_all.sort_values("Date").copy()
    df_plot["_pred"] = model.predict(df_plot[features].values)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ts = df_plot[df_plot["year"] == TEST_YEAR]["Date"]
    if len(ts):
        ax.axvspan(ts.min(), ts.max(), alpha=0.10, color="orange",
                   label=f"Test ({TEST_YEAR})")
    ax.plot(df_plot["Date"], df_plot[target],
            color="#2171B5", linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(df_plot["Date"], df_plot["_pred"],
            color=MODEL_COLOURS[model_tag], linewidth=0.8,
            label=f"{model_tag} Predicted", alpha=0.9)
    ax.set_ylabel(target, fontsize=9)
    ax.set_title(f"{model_tag} | {name} | Time series (run {run})", fontsize=10)
    ax.legend(fontsize=8); fig.autofmt_xdate(); plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_timeseries.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_importance(plots_dir, name, model_tag, run, model, features):
    imps  = model.feature_importances_
    order = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.35)))
    ax.barh([features[i] for i in order], imps[order],
            color=MODEL_COLOURS[model_tag], edgecolor="white")
    ax.set_xlabel("Feature importance (impurity)", fontsize=9)
    ax.set_title(f"{model_tag} | {name}\nFeature importance - best-CV model (run {run})", fontsize=10)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path

# ── Core training loop ─────────────────────────────────────────────────────────

def train_one(experiment, name, subset_path, features, target,
              model_tag, search_factory, run, model_dir):
    """
    Train one (model, dataset) combination using a GridSearchCV /
    RandomizedSearchCV factory.

    Phase 1 (initial search): fit search on full feature set to find best
    hyperparameters.
    Phase 2 (OOF selection): use those best hyperparams to estimate per-feature
    importance on validation folds (unbiased). Drop features below threshold.
    Phase 3 (final fit): refit search on selected features only.
    """
    df = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    test = df[df["year"] == TEST_YEAR]
    if len(test) == 0:
        print(f"    SKIP - no test rows for {TEST_YEAR}")
        return None, None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values

    n_in = len(features)
    print(f"    n_train={len(train)} n_test={len(test)} n_features={n_in}")

    # ── Phase 1: initial CV search on full feature set ────────────────────────
    print(f"    Phase 1  -  initial CV search (all {n_in} features)...",
          end="", flush=True)
    search1 = search_factory()
    search1.fit(X_tr, y_tr)
    best1    = search1.best_estimator_
    cv_rmse1 = float(-search1.best_score_)
    best_p   = search1.best_params_
    print(f"  CV_RMSE={cv_rmse1:.3f}  best={best_p}")

    # Full-feature test metrics (for FS comparison)
    tr_pred_full = best1.predict(X_tr)
    te_pred_full = best1.predict(X_te)
    r2_train_full = round(_r2(y_tr, tr_pred_full),   4)
    r2_test_full  = round(_r2(y_te, te_pred_full),   4)
    rmse_test_full = round(_rmse(y_te, te_pred_full), 4)
    mae_test_full  = round(_mae(y_te,  te_pred_full), 4)
    r2_gap_full    = round(r2_train_full - r2_test_full, 4)

    # ── Phase 2: OOF permutation importance feature selection ─────────────────
    print(f"    Phase 2  -  OOF permutation importance selection...",
          end="", flush=True)
    oof_mask, selected_nl, norm_imps = _oof_perm_select(
        best1, X_tr, y_tr, features, TSCV, threshold=0.05
    )
    n_sel = int(oof_mask.sum())
    print(f"  selected {n_sel}/{n_in} features")
    print(f"    Kept : {selected_nl}")
    dropped_nl = [f for f, m in zip(features, oof_mask) if not m]
    if dropped_nl:
        print(f"    Dropped: {dropped_nl}")

    # Apply mask to training and test arrays
    X_tr_sel = X_tr[:, oof_mask]
    X_te_sel = X_te[:, oof_mask]

    # ── Phase 3: refit final CV search on selected features only ─────────────
    print(f"    Phase 3  -  final CV search ({n_sel} features)...",
          end="", flush=True)
    search2  = search_factory()
    search2.fit(X_tr_sel, y_tr)
    best     = search2.best_estimator_
    cv_rmse  = float(-search2.best_score_)
    best_p2  = search2.best_params_
    print(f"  CV_RMSE={cv_rmse:.3f}  best={best_p2}")

    tr_pred = best.predict(X_tr_sel)
    te_pred = best.predict(X_te_sel)

    row = {
        "experiment":          experiment,
        "model_name":          name,
        "run":                 run,
        "model":               model_tag,
        "target":              target,
        "n_train":             len(train),
        "n_test":              len(test),
        "n_features_input":    n_in,
        "n_selected_nl":       n_sel,
        "selected_features_nl": ", ".join(selected_nl),
        "dropped_features_nl":  ", ".join(dropped_nl),
        "CV_RMSE_initial":     round(cv_rmse1, 4),
        "R2_train_full":       r2_train_full,
        "R2_test_full":        r2_test_full,
        "RMSE_test_full":      rmse_test_full,
        "MAE_test_full":       mae_test_full,
        "R2_gap_full":         r2_gap_full,
        "CV_RMSE":             round(cv_rmse,  4),
        "R2_train":            round(_r2(y_tr, tr_pred),  4),
        "RMSE_train":          round(_rmse(y_tr, tr_pred), 4),
        "MAE_train":           round(_mae(y_tr,  tr_pred), 4),
        "R2_test":             round(_r2(y_te, te_pred),   4),
        "RMSE_test":           round(_rmse(y_te, te_pred), 4),
        "MAE_test":            round(_mae(y_te,  te_pred), 4),
        "R2_gap":              round(_r2(y_tr, tr_pred) - _r2(y_te, te_pred), 4),
        "best_params":         str(best_p2),
    }
    # n_features column (used by unified report for backward compat)
    row["n_features"] = n_sel

    # ── Save best estimator (with feature metadata) ──────────────────────────
    plots_dir  = os.path.join(model_dir, "plots")
    models_dir = os.path.join(model_dir, "models")
    pkl_path   = os.path.join(models_dir, f"{name}_{model_tag}_run_{run}.pkl")
    joblib.dump({
        "model":            best,
        "selected_features": selected_nl,
        "feature_mask":     oof_mask,
        "all_features":     features,
        "norm_oof_importance": dict(zip(features, norm_imps.tolist())),
    }, pkl_path)

    # ── Plots (using selected features) ───────────────────────────────────
    plots = {
        "scatter":    _plot_scatter(plots_dir, name, model_tag, run,
                                    test, y_te, te_pred, target),
        "timeseries": _plot_timeseries(plots_dir, name, model_tag, run,
                                       df, selected_nl, best, target),
        "importance": _plot_importance(plots_dir, name, model_tag, run,
                                       best, selected_nl),
    }

    # ── Append predictions to subset file ──────────────────────────────────
    df_sub = pd.read_excel(subset_path)
    col    = f"predicted_{model_tag}_NL_run_{run}"
    df_sub[col] = best.predict(df_sub[selected_nl].values).round(3)
    df_sub.to_excel(subset_path, index=False)

    return row, plots


def run_model(model_tag, search_factory):
    model_dir  = os.path.join(SCRIPT_DIR, model_tag.lower())
    results_fp = os.path.join(model_dir, "results.xlsx")
    run        = _run_number(model_dir)

    print(f"\n{'='*65}")
    print(f"  {model_tag} - run {run}  (tuned via GridSearchCV / TimeSeriesSplit n=3)")
    print(f"{'='*65}")

    all_rows  = []
    all_plots = {}

    for experiment, name, subset_path, features, target in REGISTRY:
        print(f"\n[{experiment}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP - file not found: {subset_path}"); continue

        row, plots = train_one(experiment, name, subset_path, features, target,
                               model_tag, search_factory, run, model_dir)
        if row is None:
            continue
        all_rows.append(row)
        all_plots[name] = plots
        print(f"    R²_train={row['R2_train']:+.3f}  R²_test={row['R2_test']:+.3f}"
              f"  RMSE_test={row['RMSE_test']:.3f}  MAE_test={row['MAE_test']:.3f}")

    # Save results
    df_new = pd.DataFrame(all_rows)
    if os.path.exists(results_fp):
        df_out = pd.concat([pd.read_excel(results_fp), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(results_fp, index=False)
    print(f"\n  Results → {results_fp}")
    return all_rows, all_plots, run


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    models = [
        ("RF",  lambda: GridSearchCV(
            RandomForestRegressor(**RF_BASE), RF_GRID,
            scoring="neg_root_mean_squared_error",
            cv=TSCV, n_jobs=-1, refit=True,
        )),
        ("GB",  lambda: GridSearchCV(
            GradientBoostingRegressor(**GB_BASE), GB_GRID,
            scoring="neg_root_mean_squared_error",
            cv=TSCV, n_jobs=-1, refit=True,
        )),
        ("XGB", lambda: RandomizedSearchCV(
            XGBRegressor(**XGB_BASE), XGB_DIST,
            n_iter=30, scoring="neg_root_mean_squared_error",
            cv=TSCV, n_jobs=-1, refit=True, random_state=42,
        )),
    ]

    summary = {}
    for tag, factory in models:
        rows, plots, run = run_model(tag, factory)
        summary[tag] = {"rows": rows, "plots": plots, "run": run}

    print("\n" + "="*65)
    print("  All models complete (tuned).")
    print(f"  Results in: {SCRIPT_DIR}/{{rf,gb,xgb}}/results.xlsx")
    print("="*65)


if __name__ == "__main__":
    main()
