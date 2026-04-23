"""
non_linear_modeling_exp4_s2.py - RF, GB, XGBoost on Experiment 4 Sub-2 subsets.

Experiment 4 Phase 2: iterative VIF pruning (threshold=10) applied to the
Exp4-S1 feature pool. Surviving features: 5-10 per target (all pH features,
MLSS, Inlet BOD/COD, and one of Flow/Power dropped).

Row gain vs Exp4-S1: +47 to +293 rows (removed high-missingness correlated
features unlocked more complete observation days, especially for composites).

All other protocol details identical to non_linear_modeling_exp4_s1.py.

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp4_s2/non_linear_modeling_exp4_s2.py
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

def _e4s2(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment4", "sub_exp2", f"{name}.xlsx")

# ── Splits ─────────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
RF_BASE  = dict(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=1)
GB_BASE  = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, random_state=42)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0)

# ── Search grids ───────────────────────────────────────────────────────────────
RF_GRID  = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
GB_GRID  = {"max_depth": [2, 3, 4, 5],    "min_samples_leaf": [5, 10, 20]}
XGB_DIST = {"max_depth": [2, 3, 4, 5], "min_child_weight": [5, 10, 20],
            "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.5, 1.0, 5.0]}

# ── Feature inference ──────────────────────────────────────────────────────────
_EXCLUDE_COLS     = {"Date", "year", "month", "day_of_week"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [
        c for c in df.columns
        if c != target
        and c not in _EXCLUDE_COLS
        and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
    ]

# ── Registry ───────────────────────────────────────────────────────────────────
REGISTRY = [
    ("Experiment 4 Sub-2", "grab_BOD", _e4s2("grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Experiment 4 Sub-2", "grab_COD", _e4s2("grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Experiment 4 Sub-2", "grab_TSS", _e4s2("grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Experiment 4 Sub-2", "grab_pH",  _e4s2("grab_pH"),  "Effluent pH (Grab)"),
    ("Experiment 4 Sub-2", "comp_BOD", _e4s2("comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Experiment 4 Sub-2", "comp_COD", _e4s2("comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Experiment 4 Sub-2", "comp_TSS", _e4s2("comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Experiment 4 Sub-2", "comp_pH",  _e4s2("comp_pH"),  "Effluent pH (Composite)"),
]

# ── Colours ────────────────────────────────────────────────────────────────────
YEAR_COLOURS  = {2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
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

# ── Plotting ───────────────────────────────────────────────────────────────────
def _plot_scatter(plots_dir, name, model_tag, run, test_df, y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for yr in sorted(test_df["year"].unique()):
        m = test_df["year"].values == yr
        ax.scatter(y_test[m], y_pred[m], color=YEAR_COLOURS.get(yr, "#999"),
                   label=str(yr), alpha=0.75, s=28, edgecolors="none")
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
    ax.set_title(f"{model_tag} | {name}\nFeature importance (run {run})", fontsize=10)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path

# ── Core training loop ─────────────────────────────────────────────────────────
def train_one(experiment, name, subset_path, features, target,
              model_tag, search_factory, run, model_dir):
    df = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    test = df[df["year"] == TEST_YEAR]
    if len(test) == 0:
        print(f"    SKIP - no test rows for {TEST_YEAR}"); return None, None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values

    print(f"    n_train={len(train)} n_test={len(test)} n_feat={len(features)} - tuning...",
          end="", flush=True)

    search = search_factory()
    search.fit(X_tr, y_tr)
    best    = search.best_estimator_
    cv_rmse = float(-search.best_score_)
    best_p  = search.best_params_
    print(f"  CV_RMSE={cv_rmse:.3f}  best={best_p}")

    tr_pred = best.predict(X_tr)
    te_pred = best.predict(X_te)

    row = {
        "experiment": experiment,
        "model_name": name,
        "run":        run,
        "model":      model_tag,
        "target":     target,
        "n_train":    len(train),
        "n_test":     len(test),
        "n_features": len(features),
        "CV_RMSE":    round(cv_rmse, 4),
        "R2_train":   round(_r2(y_tr, tr_pred),  4),
        "RMSE_train": round(_rmse(y_tr, tr_pred), 4),
        "MAE_train":  round(_mae(y_tr,  tr_pred), 4),
        "R2_test":    round(_r2(y_te, te_pred),   4),
        "RMSE_test":  round(_rmse(y_te, te_pred), 4),
        "MAE_test":   round(_mae(y_te,  te_pred), 4),
        "R2_gap":     round(_r2(y_tr, tr_pred) - _r2(y_te, te_pred), 4),
        "best_params": str(best_p),
    }

    plots_dir  = os.path.join(model_dir, "plots")
    models_dir = os.path.join(model_dir, "models")
    pkl_path   = os.path.join(models_dir, f"{name}_{model_tag}_run_{run}.pkl")
    joblib.dump(best, pkl_path)

    _plot_scatter(plots_dir, name, model_tag, run, test, y_te, te_pred, target)
    _plot_timeseries(plots_dir, name, model_tag, run, df, features, best, target)
    _plot_importance(plots_dir, name, model_tag, run, best, features)

    df_sub = pd.read_excel(subset_path)
    col    = f"predicted_{model_tag}_NL_run_{run}"
    df_sub[col] = best.predict(df_sub[features].values).round(3)
    df_sub.to_excel(subset_path, index=False)

    return row, None


def run_model(model_tag, search_factory):
    model_dir  = os.path.join(SCRIPT_DIR, model_tag.lower())
    results_fp = os.path.join(model_dir, "results.xlsx")
    run        = _run_number(model_dir)

    os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "plots"),  exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {model_tag} Exp4-S2 - run {run}")
    print(f"{'='*65}")

    all_rows = []

    for experiment, name, subset_path, target in REGISTRY:
        print(f"\n[{experiment}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP - file not found: {subset_path}"); continue

        df_peek  = pd.read_excel(subset_path, nrows=0)
        features = infer_features(df_peek, target)

        row, _ = train_one(experiment, name, subset_path, features, target,
                           model_tag, search_factory, run, model_dir)
        if row is None:
            continue
        all_rows.append(row)
        print(f"    R²_train={row['R2_train']:+.3f}  R²_test={row['R2_test']:+.3f}"
              f"  R²_gap={row['R2_gap']:+.3f}  RMSE_test={row['RMSE_test']:.3f}")

    df_new = pd.DataFrame(all_rows)
    if os.path.exists(results_fp):
        df_out = pd.concat([pd.read_excel(results_fp), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(results_fp, index=False)
    print(f"\n  Results → {results_fp}")
    return all_rows, run


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

    for tag, factory in models:
        run_model(tag, factory)

    print("\n" + "="*65)
    print("  All models complete (Exp4-S2).")
    print(f"  Results in: {SCRIPT_DIR}/{{rf,gb,xgb}}/results.xlsx")
    print("="*65)

if __name__ == "__main__":
    main()
