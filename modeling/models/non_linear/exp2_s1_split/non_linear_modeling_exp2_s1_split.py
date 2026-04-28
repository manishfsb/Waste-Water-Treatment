"""
non_linear_modeling_exp2_s1_split.py

Trains RF, GradientBoosting, and XGBoost on Experiment 2 Sub-experiment 1 split variants:
  - Exp2-Sub1-Clr : Sec Clarifier features + COMMON_CYCLIC (12 features)
  - Exp2-Sub1-Sed : Sec Sedimentation features + COMMON_CYCLIC (12 features)

These are baseline runs (no OOF feature selection). A single model is trained
per dataset without the 3-phase full→select→refit pipeline.

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp2_s1_split/non_linear_modeling_exp2_s1_split.py
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
SEC_CLARIFIER_COLS = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
]
SEC_SED_COLS = [
    "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON_CYCLIC = ["Flow (MLD)", "Power Total (KW)", "year", "month_sin", "month_cos", "dow_sin", "dow_cos"]

CLR_FEAT = SEC_CLARIFIER_COLS + COMMON_CYCLIC   # 12 features
SED_FEAT = SEC_SED_COLS + COMMON_CYCLIC         # 12 features

# ── Dataset registry ───────────────────────────────────────────────────────────
def _clr(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp1",
                        "sec_clarifier", f"{name}.xlsx")

def _sed(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp1",
                        "sec_sed", f"{name}.xlsx")


REGISTRY = [
    # Clarifier
    ("Exp2-Sub1-Clr", "grab_BOD_clr", _clr("grab_BOD"), CLR_FEAT, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub1-Clr", "grab_COD_clr", _clr("grab_COD"), CLR_FEAT, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub1-Clr", "grab_TSS_clr", _clr("grab_TSS"), CLR_FEAT, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub1-Clr", "grab_pH_clr",  _clr("grab_pH"),  CLR_FEAT, "Effluent pH (Grab)"),
    ("Exp2-Sub1-Clr", "comp_BOD_clr", _clr("comp_BOD"), CLR_FEAT, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub1-Clr", "comp_COD_clr", _clr("comp_COD"), CLR_FEAT, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub1-Clr", "comp_TSS_clr", _clr("comp_TSS"), CLR_FEAT, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub1-Clr", "comp_pH_clr",  _clr("comp_pH"),  CLR_FEAT, "Effluent pH (Composite)"),
    # Sedimentation
    ("Exp2-Sub1-Sed", "grab_BOD_sed", _sed("grab_BOD"), SED_FEAT, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub1-Sed", "grab_COD_sed", _sed("grab_COD"), SED_FEAT, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub1-Sed", "grab_TSS_sed", _sed("grab_TSS"), SED_FEAT, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub1-Sed", "grab_pH_sed",  _sed("grab_pH"),  SED_FEAT, "Effluent pH (Grab)"),
    ("Exp2-Sub1-Sed", "comp_BOD_sed", _sed("comp_BOD"), SED_FEAT, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub1-Sed", "comp_COD_sed", _sed("comp_COD"), SED_FEAT, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub1-Sed", "comp_TSS_sed", _sed("comp_TSS"), SED_FEAT, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub1-Sed", "comp_pH_sed",  _sed("comp_pH"),  SED_FEAT, "Effluent pH (Composite)"),
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

    # Single CV search on full feature set (no OOF FS phase for baseline runs)
    print(f"    CV search ({n_in} features)...", end="", flush=True)
    gs = search_cls(estimator, param_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=TSCV, n_jobs=-1, refit=True,
                    **({} if search_cls is GridSearchCV else {"n_iter": 30, "random_state": 42}))
    gs.fit(X_tr, y_tr)
    best    = gs.best_estimator_
    cv_rmse = float(-gs.best_score_)
    print(f"  CV_RMSE={cv_rmse:.3f}")

    tr_pred  = best.predict(X_tr)
    te_pred  = best.predict(X_te)
    all_pred = best.predict(df[features].values)

    row = {
        "experiment":       experiment,
        "target":           target,
        "model":            model_tag,
        "run":              run,
        "n_train":          len(train),
        "n_test":           len(test),
        "n_features":       n_in,
        "n_features_input": n_in,
        "CV_RMSE":          round(cv_rmse, 4),
        "R2_train":         _r2(y_tr, tr_pred),
        "R2_test":          _r2(y_te, te_pred),
        "R2_gap":           _r2(y_tr, tr_pred) - _r2(y_te, te_pred),
        "RMSE_train":       _rmse(y_tr, tr_pred),
        "RMSE_test":        _rmse(y_te, te_pred),
        "MAE_test":         _mae(y_te, te_pred),
        "best_params":      str(gs.best_params_),
    }
    print(f"    R²_train={row['R2_train']:+.3f}  R²_test={row['R2_test']:+.3f}  "
          f"gap={row['R2_gap']:+.3f}  RMSE_test={row['RMSE_test']:.3f}")

    joblib.dump({"model": best, "all_features": features},
                os.path.join(model_dir, f"{name}_{model_tag}_run_{run}.pkl"))
    _plot_scatter(plots_dir, name, model_tag, run, test, y_te, te_pred, target)
    _plot_importance(plots_dir, name, model_tag, run, best, features)

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
    print(f"{model_tag} — Exp2-Sub1-Split  (run {run})")
    print(f"{'='*60}")

    results = []
    for experiment, name, path, features, target in REGISTRY:
        print(f"\n  [{experiment}]  {name}  →  {target}")
        if not os.path.exists(path):
            print(f"    WARNING: file not found - {path}")
            print(f"    Run make_sub1_split_datasets.py first.")
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
    print(df_new[["experiment", "target", "R2_train", "R2_test", "R2_gap", "RMSE_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Non-Linear Modeling — Exp2-Sub1-Split  (RF / GB / XGB)")
    print("Clarifier: 10 features | Sed: 10 features | No OOF FS (baseline runs)\n")

    run_model("RF",  RandomForestRegressor(**RF_BASE),     RF_GRID,  GridSearchCV)
    run_model("GB",  GradientBoostingRegressor(**GB_BASE), GB_GRID,  GridSearchCV)
    run_model("XGB", XGBRegressor(**XGB_BASE),             XGB_DIST, RandomizedSearchCV)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
