"""
non_linear_modeling_exp9_s5.py - RF, GB, XGBoost on Experiment 9 SE5 datasets.

Recency hypothesis - W1 + log1p feature transform + log1p target transform (log-log model).
Combination of SE3 (log1p targets + Duan smearing) and SE4 (log1p features in-place).

Feature set: identical to Exp9-SE1 (27 features), with the 12 right-skewed
concentration columns replaced in-place with their log1p values before fitting.
Target: BOD/COD/TSS are log1p-transformed; pH is left on its natural scale.
Duan smearing back-transforms predictions to the original scale.
All reported metrics are on the ORIGINAL scale.

Datasets: experiment9/sub_exp1/ (same as SE1)
Exp key : Exp9-SE5

Usage (from project root):
    .venv/bin/python3 modeling/models/non_linear/exp9_s5/non_linear_modeling_exp9_s5.py
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

# -- Paths ----------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

def _ds(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment9", "sub_exp1", f"{name}.xlsx")

# -- Splits ---------------------------------------------------------------------
TRAIN_YEARS = [2024]
TEST_YEAR   = 2025
TSCV        = TimeSeriesSplit(n_splits=3)

# -- Fixed hyperparameters ------------------------------------------------------
RF_BASE  = dict(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=1)
GB_BASE  = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, random_state=42)
XGB_BASE = dict(n_estimators=300, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0)

RF_GRID  = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
GB_GRID  = {"max_depth": [2, 3, 4, 5],    "min_samples_leaf": [5, 10, 20]}
XGB_DIST = {"max_depth": [2, 3, 4, 5], "min_child_weight": [5, 10, 20],
            "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.5, 1.0, 5.0]}

# Same 12 columns as Exp7 (Phase 10) that exist in the Exp9 feature set.
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
    return [c for c in df.columns if c != target
            and c not in _EXCLUDE_COLS
            and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)]

def apply_log_features(df: pd.DataFrame, features: list) -> tuple:
    df = df.copy()
    transformed = [c for c in features if c in LOG_FEATURES and c in df.columns]
    for col in transformed:
        df[col] = np.log1p(df[col])
    return df, transformed

# -- Registry - log_y=True for concentration targets, False for pH --------------
REGISTRY = [
    ("Exp9-SE5", "grab_BOD", _ds("grab_BOD"), "Effluent BOD (mg/L, Grab)",       True),
    ("Exp9-SE5", "grab_COD", _ds("grab_COD"), "Effluent COD (mg/L, Grab)",       True),
    ("Exp9-SE5", "grab_TSS", _ds("grab_TSS"), "Effluent TSS (mg/L, Grab)",       True),
    ("Exp9-SE5", "grab_pH",  _ds("grab_pH"),  "Effluent pH (Grab)",              False),
    ("Exp9-SE5", "comp_BOD", _ds("comp_BOD"), "Effluent BOD (mg/L, Composite)",  True),
    ("Exp9-SE5", "comp_COD", _ds("comp_COD"), "Effluent COD (mg/L, Composite)",  True),
    ("Exp9-SE5", "comp_TSS", _ds("comp_TSS"), "Effluent TSS (mg/L, Composite)",  True),
    ("Exp9-SE5", "comp_pH",  _ds("comp_pH"),  "Effluent pH (Composite)",         False),
]

YEAR_COLOURS  = {2021: "#2171B5", 2022: "#74C476", 2023: "#238B45",
                 2024: "#FD8D3C", 2025: "#D94801"}
MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}

def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
def _r2(yt, yp):   return float(r2_score(yt, yp))

def _run_number(model_dir: str) -> int:
    rfile = os.path.join(model_dir, "results.xlsx")
    if not os.path.exists(rfile):
        return 1
    df = pd.read_excel(rfile)
    return int(df["run"].max()) + 1 if "run" in df.columns else 1

# -- Duan smearing helpers ------------------------------------------------------
def _compute_smear(y_train_log: np.ndarray, y_train_pred_log: np.ndarray) -> float:
    resid = y_train_log - y_train_pred_log
    return float(np.mean(np.exp(resid)))

def _back_transform(y_pred_log: np.ndarray, smear: float) -> np.ndarray:
    return np.expm1(y_pred_log) * smear

# -- Plotting -------------------------------------------------------------------
def _plot_scatter(plots_dir, name, model_tag, run, test_df, y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(6, 5))
    for yr in sorted(test_df["year"].unique()):
        m = test_df["year"].values == yr
        ax.scatter(y_test[m], y_pred[m], color=YEAR_COLOURS.get(yr, "#888"),
                   alpha=0.65, s=25, label=str(yr), edgecolors="none")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
    ax.set_title(f"{name} - {model_tag} (W1+LogLog)\nR²={_r2(y_test, y_pred):+.3f}  "
                 f"RMSE={_rmse(y_test, y_pred):.3f}", fontsize=11)
    ax.set_xlabel("Actual (original scale)"); ax.set_ylabel("Predicted (original scale)")
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
    ax.set_title(f"{name} - {model_tag} (W1+LogLog) | Feature Importance (run {run})",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_importance.png")
    fig.savefig(path, dpi=150); plt.close(fig)

# -- Core training loop ---------------------------------------------------------
def train_one(experiment, name, subset_path, features, target, log_y,
              model_tag, search_factory, run, model_dir, plots_dir):
    df_raw = pd.read_excel(subset_path, parse_dates=["Date"])
    df, transformed = apply_log_features(df_raw, features)

    train = df[df["year"].isin(TRAIN_YEARS)].dropna(subset=features + [target]).copy()
    test  = df[df["year"] == TEST_YEAR].dropna(subset=features + [target])

    if len(train) == 0:
        print(f"    SKIP - no training rows for {TRAIN_YEARS}")
        return None
    if len(test) == 0:
        print(f"    SKIP - no test rows for {TEST_YEAR}")
        return None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values

    # Transform targets
    if log_y:
        y_tr_t = np.log1p(y_tr)
    else:
        y_tr_t = y_tr

    print(f"    n_train={len(train)} n_test={len(test)} n_feat={len(features)} "
          f"log_feat={len(transformed)} log_y={log_y} - tuning...", end="", flush=True)

    search = search_factory()
    search.fit(X_tr, y_tr_t)
    best    = search.best_estimator_
    cv_rmse = float(-search.best_score_)
    best_p  = search.best_params_
    print(f"  CV_RMSE={cv_rmse:.3f}  best={best_p}")

    tr_pred_t = best.predict(X_tr)
    te_pred_t = best.predict(X_te)

    # Back-transform with Duan smearing
    if log_y:
        smear   = _compute_smear(y_tr_t, tr_pred_t)
        tr_pred = _back_transform(tr_pred_t, smear)
        te_pred = _back_transform(te_pred_t, smear)
        df_sub  = pd.read_excel(subset_path)
        df_sub_t, _ = apply_log_features(df_sub, features)
        all_pred = _back_transform(best.predict(df_sub_t[features].values), smear)
    else:
        smear   = 1.0
        tr_pred = tr_pred_t; te_pred = te_pred_t
        df_sub  = pd.read_excel(subset_path)
        df_sub_t, _ = apply_log_features(df_sub, features)
        all_pred = best.predict(df_sub_t[features].values)

    row = {
        "experiment":     experiment,
        "model_name":     name,
        "run":            run,
        "model":          model_tag,
        "target":         target,
        "log_y":          log_y,
        "smear":          round(smear, 6),
        "n_train":        len(train),
        "n_test":         len(test),
        "n_features":     len(features),
        "n_log_features": len(transformed),
        "CV_RMSE":        round(cv_rmse, 4),
        "R2_train":       round(_r2(y_tr, tr_pred),  4),
        "RMSE_train":     round(_rmse(y_tr, tr_pred), 4),
        "MAE_train":      round(_mae(y_tr,  tr_pred), 4),
        "R2_test":        round(_r2(y_te, te_pred),   4),
        "RMSE_test":      round(_rmse(y_te, te_pred), 4),
        "MAE_test":       round(_mae(y_te,  te_pred), 4),
        "R2_gap":         round(_r2(y_tr, tr_pred) - _r2(y_te, te_pred), 4),
        "best_params":    str(best_p),
    }

    col = f"predicted_{model_tag}_loglogf_run_{run}"
    df_sub[col] = np.round(all_pred, 3)
    df_sub.to_excel(subset_path, index=False)

    joblib.dump({"model": best, "log_y": log_y, "smear": smear,
                 "log_features": transformed, "features": features},
                os.path.join(model_dir, f"{name}_{model_tag}_run_{run}.pkl"))
    _plot_scatter(plots_dir, name, model_tag, run, test, y_te, te_pred, target)
    _plot_importance(plots_dir, name, model_tag, run, best, features)

    return row


def run_model(model_tag, search_factory):
    model_dir  = os.path.join(SCRIPT_DIR, model_tag.lower())
    plots_dir  = os.path.join(model_dir, "plots")
    results_fp = os.path.join(model_dir, "results.xlsx")
    run        = _run_number(model_dir)

    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {model_tag} (Exp9-SE5, W1+LogLog: 2024-only + log feat + log target) - run {run}")
    print(f"{'='*65}")

    all_rows = []
    for experiment, name, subset_path, target, log_y in REGISTRY:
        print(f"\n[{experiment}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP - file not found: {subset_path}"); continue

        df_peek  = pd.read_excel(subset_path, nrows=0)
        features = infer_features(df_peek, target)

        row = train_one(experiment, name, subset_path, features, target, log_y,
                        model_tag, search_factory, run, model_dir, plots_dir)
        if row is None:
            continue
        all_rows.append(row)
        print(f"    R2_train={row['R2_train']:+.3f}  R2_test={row['R2_test']:+.3f}"
              f"  RMSE_test={row['RMSE_test']:.3f}  MAE_test={row['MAE_test']:.3f}")

    df_new = pd.DataFrame(all_rows)
    if os.path.exists(results_fp):
        df_out = pd.concat([pd.read_excel(results_fp), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(results_fp, index=False)
    print(f"\n  Results -> {results_fp}")
    return all_rows, run


# -- Main -----------------------------------------------------------------------
def main():
    models = [
        ("RF",  lambda: GridSearchCV(
            RandomForestRegressor(**RF_BASE), RF_GRID,
            scoring="neg_root_mean_squared_error", cv=TSCV, n_jobs=-1, refit=True,
        )),
        ("GB",  lambda: GridSearchCV(
            GradientBoostingRegressor(**GB_BASE), GB_GRID,
            scoring="neg_root_mean_squared_error", cv=TSCV, n_jobs=-1, refit=True,
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
    print("  Exp9-SE5 (W1+LogLog)  -  all models complete.")
    print(f"  Results in: {SCRIPT_DIR}/{{rf,gb,xgb}}/results.xlsx")
    print("="*65)


if __name__ == "__main__":
    main()
