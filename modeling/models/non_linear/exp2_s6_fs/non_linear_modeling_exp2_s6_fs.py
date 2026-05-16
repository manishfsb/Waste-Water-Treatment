"""
non_linear_modeling_exp2_s6_fs.py - RF, GB, XGB with OOF perm-importance FS on Exp2 SE2a-FS.

Feature set: SECONDARY (10) = Sec Clarifier (5) + Sec Sedimentation (5).
Three-phase protocol: full CV search -> OOF permutation importance selection -> refit on selected.
Identical datasets to exp2_s6 (no COMMON, no Inlet).

Exp key:  Exp2-S6-FS

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp2_s6_fs/non_linear_modeling_exp2_s6_fs.py
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

def _ds(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp6", f"{name}.xlsx")

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)

RF_BASE  = dict(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=1)
GB_BASE  = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, random_state=42)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0)

RF_GRID  = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
GB_GRID  = {"max_depth": [2, 3, 4, 5],    "min_samples_leaf": [5, 10, 20]}
XGB_DIST = {"max_depth": [2, 3, 4, 5], "min_child_weight": [5, 10, 20],
            "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.5, 1.0, 5.0]}

_EXCLUDE_COLS     = {"Date", "year"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [c for c in df.columns if c != target
            and c not in _EXCLUDE_COLS
            and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)]

REGISTRY = [
    ("Exp2-S6-FS", "grab_BOD", _ds("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Exp2-S6-FS", "grab_COD", _ds("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Exp2-S6-FS", "grab_TSS", _ds("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Exp2-S6-FS", "grab_pH",  _ds("grab_pH"),  "Effluent pH (Grab)"),
    ("Exp2-S6-FS", "comp_BOD", _ds("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Exp2-S6-FS", "comp_COD", _ds("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Exp2-S6-FS", "comp_TSS", _ds("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Exp2-S6-FS", "comp_pH",  _ds("comp_pH"),  "Effluent pH (Composite)"),
]

YEAR_COLOURS  = {2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
                 2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801"}
MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}


def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
def _r2(yt, yp):   return float(r2_score(yt, yp))

def _run_number(model_dir: str) -> int:
    rfile = os.path.join(model_dir, "results.xlsx")
    if not os.path.exists(rfile): return 1
    df = pd.read_excel(rfile)
    return int(df["run"].max()) + 1 if "run" in df.columns else 1


def _oof_perm_select(fitted_estimator, X_tr, y_tr, features, tscv, threshold=0.05):
    fold_imps = np.zeros(len(features))
    for tr_idx, val_idx in tscv.split(X_tr):
        est = clone(fitted_estimator)
        est.fit(X_tr[tr_idx], y_tr[tr_idx])
        perm = permutation_importance(est, X_tr[val_idx], y_tr[val_idx],
                                      n_repeats=10, random_state=42, n_jobs=1)
        fold_imps += perm.importances_mean.clip(min=0)
    fold_imps /= tscv.n_splits
    total = fold_imps.sum()
    norm_imps = fold_imps / total if total > 0 else fold_imps
    mask = norm_imps >= threshold
    if mask.sum() == 0:
        mask = np.ones(len(features), dtype=bool)
    return mask, [f for f, m in zip(features, mask) if m], norm_imps


def _plot_scatter(plots_dir, name, model_tag, run, test_df, y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for yr in sorted(test_df["year"].unique()):
        m = test_df["year"].values == yr
        ax.scatter(y_test[m], y_pred[m], color=YEAR_COLOURS.get(yr, "#999"),
                   label=str(yr), alpha=0.75, s=28, edgecolors="none")
    lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(f"Actual {target}", fontsize=10); ax.set_ylabel("Predicted", fontsize=10)
    ax.set_title(f"{model_tag} | {name}\nTest scatter (run {run})", fontsize=10)
    ax.legend(title="Year", fontsize=8); plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _plot_importance(plots_dir, name, model_tag, run, model, features):
    imps  = model.feature_importances_
    order = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.35)))
    ax.barh([features[i] for i in order], imps[order],
            color=MODEL_COLOURS[model_tag], edgecolor="white")
    ax.set_xlabel("Feature importance (impurity)", fontsize=9)
    ax.set_title(f"{model_tag} | {name}\nFeature importance (run {run})", fontsize=10)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def train_one(experiment, name, subset_path, features, target,
              model_tag, estimator, param_grid, search_cls, model_dir, plots_dir, run):

    df = pd.read_excel(subset_path, parse_dates=["Date"])
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()
    test = df[df["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"    SKIP - no test rows for {TEST_YEAR}"); return None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values
    n_in = len(features)
    print(f"    n_train={len(train)} n_test={len(test)} n_feat={n_in}")

    # Phase 1: CV search on full feature set
    print(f"    Phase 1 - CV search ({n_in} feat)...", end="", flush=True)
    search_kwargs = {} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}
    gs1 = search_cls(estimator, param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True, **search_kwargs)
    gs1.fit(X_tr, y_tr)
    best1 = gs1.best_estimator_
    print(f" CV_RMSE={-gs1.best_score_:.3f}")

    tr_full = best1.predict(X_tr); te_full = best1.predict(X_te)
    r2_train_full = _r2(y_tr, tr_full); r2_test_full = _r2(y_te, te_full)
    r2_gap_full   = r2_train_full - r2_test_full

    # Phase 2: OOF permutation importance selection
    print(f"    Phase 2 - OOF perm importance selection...", end="", flush=True)
    mask, selected, norm_imps = _oof_perm_select(best1, X_tr, y_tr, features, TSCV)
    n_sel = int(mask.sum())
    dropped = [f for f, m in zip(features, mask) if not m]
    print(f" {n_sel}/{n_in} selected", end="")
    if dropped: print(f"  dropped: {dropped}")
    else: print(" (no pruning)")

    X_tr_s = X_tr[:, mask]; X_te_s = X_te[:, mask]

    # Phase 3: refit on selected features
    print(f"    Phase 3 - refit ({n_sel} feat)...", end="", flush=True)
    gs2 = search_cls(clone(estimator), param_grid,
                     scoring="neg_root_mean_squared_error",
                     cv=TSCV, n_jobs=-1, refit=True, **search_kwargs)
    gs2.fit(X_tr_s, y_tr)
    best2 = gs2.best_estimator_
    cv_rmse2 = float(-gs2.best_score_)
    print(f" CV_RMSE={cv_rmse2:.3f}  best={gs2.best_params_}")

    tr_pred = best2.predict(X_tr_s); te_pred = best2.predict(X_te_s)

    row = {
        "experiment":           experiment,
        "model_name":           name,
        "run":                  run,
        "model":                model_tag,
        "target":               target,
        "n_train":              len(train),
        "n_test":               len(test),
        "n_features":           n_sel,
        "n_features_input":     n_in,
        "CV_RMSE":              round(cv_rmse2, 4),
        "R2_train_full":        round(r2_train_full, 4),
        "R2_test_full":         round(r2_test_full,  4),
        "R2_gap_full":          round(r2_gap_full,   4),
        "R2_train":             round(_r2(y_tr, tr_pred),  4),
        "RMSE_train":           round(_rmse(y_tr, tr_pred), 4),
        "MAE_train":            round(_mae(y_tr,  tr_pred), 4),
        "R2_test":              round(_r2(y_te, te_pred),   4),
        "RMSE_test":            round(_rmse(y_te, te_pred), 4),
        "MAE_test":             round(_mae(y_te,  te_pred), 4),
        "R2_gap":               round(_r2(y_tr, tr_pred) - _r2(y_te, te_pred), 4),
        "n_selected_nl":        n_sel,
        "selected_features_nl": ", ".join(selected),
        "best_params":          str(gs2.best_params_),
    }

    os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    joblib.dump({"mask": mask, "model": best2},
                os.path.join(model_dir, "models", f"{name}_{model_tag}_run_{run}.pkl"))
    _plot_scatter(plots_dir, name, model_tag, run, test, y_te, te_pred, target)
    _plot_importance(plots_dir, name, model_tag, run, best2, selected)

    df_sub = pd.read_excel(subset_path)
    X_all  = df_sub[features].values
    col    = f"predicted_{model_tag}_fs_run_{run}"
    df_sub[col] = best2.predict(X_all[:, mask]).round(3)
    df_sub.to_excel(subset_path, index=False)

    return row


def run_model(model_tag, estimator, param_grid, search_cls):
    model_dir  = os.path.join(SCRIPT_DIR, model_tag.lower())
    results_fp = os.path.join(model_dir, "results.xlsx")
    plots_dir  = os.path.join(model_dir, "plots")
    run        = _run_number(model_dir)

    os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {model_tag} (Exp2-SE2a-FS, Secondary only, OOF perm FS) - run {run}")
    print(f"{'='*65}")

    all_rows = []
    for experiment, name, subset_path, target in REGISTRY:
        print(f"\n[{experiment}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP - file not found: {subset_path}"); continue
        df_peek  = pd.read_excel(subset_path, nrows=0)
        features = infer_features(df_peek, target)
        row = train_one(experiment, name, subset_path, features, target,
                        model_tag, estimator, param_grid, search_cls,
                        model_dir, plots_dir, run)
        if row is None: continue
        all_rows.append(row)
        print(f"    R2_train={row['R2_train']:+.3f}  R2_test={row['R2_test']:+.3f}"
              f"  RMSE_test={row['RMSE_test']:.3f}  n_sel={row['n_selected_nl']}")

    df_new = pd.DataFrame(all_rows)
    if os.path.exists(results_fp):
        df_out = pd.concat([pd.read_excel(results_fp), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(results_fp, index=False)
    print(f"\n  Results -> {results_fp}")
    return all_rows, run


def main():
    models = [
        ("RF",  RandomForestRegressor(**RF_BASE),          RF_GRID,  GridSearchCV),
        ("GB",  GradientBoostingRegressor(**GB_BASE),      GB_GRID,  GridSearchCV),
        ("XGB", XGBRegressor(**XGB_BASE),                  XGB_DIST, RandomizedSearchCV),
    ]
    for tag, est, grid, search_cls in models:
        run_model(tag, est, grid, search_cls)

    print("\n" + "="*65)
    print("  Exp2-SE2a-FS (Secondary only, OOF perm FS)  -  all models complete.")
    print(f"  Results in: {SCRIPT_DIR}/{{rf,gb,xgb}}/results.xlsx")
    print("="*65)


if __name__ == "__main__":
    main()
