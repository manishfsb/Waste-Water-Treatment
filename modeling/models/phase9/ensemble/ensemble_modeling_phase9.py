"""
ensemble_modeling_phase9.py — Voting and Stacking ensembles for Phase 9.

Uses Experiment 3 Sub-2 datasets. Builds two ensemble strategies:
  1. Voting  — VotingRegressor: ElNet + RF + XGBoost (equal weights)
             (replaces original Ridge+ElNet+RF which had L2 redundancy)
  2. Stacking — Walk-forward stacking with TimeSeriesSplit OOF
             Base: ElNet + RF + XGBoost → Meta: Ridge
             (replaces KFold-based StackingRegressor which had look-ahead bias)

Changes vs run 1:
  - Voting: dropped redundant Ridge; added XGBoost for structural diversity
  - Stacking: replaced KFold StackingRegressor with manual walk-forward OOF
    using TimeSeriesSplit(n_splits=5). Only past data is used when generating
    OOF predictions for each sample. First ~1/6 of training samples (no OOF fold
    coverage) are excluded from meta-learner training — no look-ahead bias.
  - Diagnostics: added y_train_std, y_test_std, MAE_2024, NRMSE_test columns
    to results for R²-vs-variance diagnosis (esp. Composite COD).

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase9/ensemble/ensemble_modeling_phase9.py
"""

import os
import pickle
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, learning_curve,
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DS_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = [
    {"name": "s2_stage3_grab_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_BOD.xlsx"),  "target": "Effluent BOD (mg/L, Grab)",        "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_grab_COD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_COD.xlsx"),  "target": "Effluent COD (mg/L, Grab)",        "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_grab_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_grab_TSS.xlsx"),  "target": "Effluent TSS (mg/L, Grab)",        "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_grab_pH",   "file": os.path.join(DS_DIR, "s2_stage3_grab_pH.xlsx"),   "target": "Effluent pH (Grab)",               "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_comp_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_BOD.xlsx"),  "target": "Effluent BOD (mg/L, Composite)",   "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_comp_COD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_COD.xlsx"),  "target": "Effluent COD (mg/L, Composite)",   "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_comp_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_comp_TSS.xlsx"),  "target": "Effluent TSS (mg/L, Composite)",   "experiment": "Phase9-Ensemble"},
    {"name": "s2_stage3_comp_pH",   "file": os.path.join(DS_DIR, "s2_stage3_comp_pH.xlsx"),   "target": "Effluent pH (Composite)",          "experiment": "Phase9-Ensemble"},
]

# ── Hyperparameter grids ───────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ELNET_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "max_iter": [5000],
}

RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt",
    max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1,
)

XGB_BASE = dict(
    n_estimators=300, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0,
)
XGB_DIST = {
    "max_depth":       [2, 3, 4, 5],
    "min_child_weight":[5, 10, 20],
    "reg_alpha":       [0.0, 0.1, 1.0],
    "reg_lambda":      [0.5, 1.0, 5.0],
}


# ── Walk-forward stacking predictor ────────────────────────────────────────────

class WalkForwardStackingPredictor:
    """
    Minimal sklearn-compatible wrapper for manual walk-forward stacking.
    Holds three fully-retrained base models and a Ridge meta-learner.
    Pickleable.
    """
    def __init__(self, base_models, meta_model, meta_alpha, oof_coverage):
        # base_models: list of (name, fitted estimator)
        self.base_models   = base_models
        self.meta_model    = meta_model
        self.meta_alpha    = meta_alpha
        self.oof_coverage  = oof_coverage   # fraction of train rows that had OOF preds

    def predict(self, X):
        meta_X = np.column_stack([m.predict(X) for _, m in self.base_models])
        return self.meta_model.predict(meta_X)

    # Enough for sklearn learning_curve (won't be perfect but won't crash)
    def fit(self, X, y):
        for _, m in self.base_models:
            m.fit(X, y)
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_features(df: pd.DataFrame, target: str) -> list:
    exclude = {"Date", "year", "month", "day_of_week", target}
    return [c for c in df.columns
            if c not in exclude and not c.startswith("predicted_")]


def _mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _next_run(results_file: str) -> int:
    if not os.path.exists(results_file):
        return 1
    df = pd.read_excel(results_file)
    return int(df["run"].max()) + 1 if len(df) > 0 else 1


def _metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    return r2, rmse, mae, mape


def _nrmse(rmse, y):
    rng = y.max() - y.min()
    return rmse / rng if rng > 0 else np.nan


# ── Plotting ───────────────────────────────────────────────────────────────────

def _scatter_plot(y_train, y_train_pred, y_test, y_test_pred,
                  r2_train, r2_test, name, model_tag, run):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, yt, yp, r2, label, color in [
        (ax1, y_train, y_train_pred, r2_train, "Train", "#4A90D9"),
        (ax2, y_test,  y_test_pred,  r2_test,  "Test",  "#E15252"),
    ]:
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "w--", lw=0.8, alpha=0.5)
        ax.scatter(yt, yp, alpha=0.55, s=18, color=color)
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{label} — R²={r2:+.3f}", fontsize=9)
    fig.suptitle(f"{model_tag} — {name}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _timeseries_plot(df_full, target, y_pred_full, name, model_tag, run):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_full["Date"], df_full[target], color="white", lw=1.2,
            alpha=0.8, label="Actual")
    ax.plot(df_full["Date"], y_pred_full, color="#B07FD4", lw=1.0,
            alpha=0.85, label=f"{model_tag} Predicted")
    split_date = df_full.loc[df_full["Date"].dt.year == 2025, "Date"].min()
    if pd.notna(split_date):
        ax.axvline(split_date, color="#F0B849", lw=1.2, linestyle="--",
                   alpha=0.7, label="Train | Test")
    ax.set_title(f"{model_tag} — {target}", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_timeseries.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _learning_curve_plot(model, X_train, y_train, name, model_tag, run):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes = np.linspace(0.2, 1.0, 7)
    try:
        ts, train_sc, val_sc = learning_curve(
            model, X_train, y_train,
            cv=tscv, train_sizes=train_sizes,
            scoring="r2", n_jobs=1,
        )
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, train_sc.mean(axis=1), "o-", color="#4A90D9", lw=1.5, label="Train R²")
    ax.fill_between(ts,
                    train_sc.mean(axis=1) - train_sc.std(axis=1),
                    train_sc.mean(axis=1) + train_sc.std(axis=1),
                    alpha=0.15, color="#4A90D9")
    ax.plot(ts, val_sc.mean(axis=1), "s-", color="#E15252", lw=1.5, label="CV Val R²")
    ax.fill_between(ts,
                    val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1),
                    alpha=0.15, color="#E15252")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Training examples", fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title(f"Learning Curve — {model_tag} — {name}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_lc.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Base model tuning helpers ──────────────────────────────────────────────────

def _tune_elnet(Xs, y_tr):
    """GridSearchCV for ElasticNet on already-scaled data."""
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        ElasticNet(max_iter=5000), ELNET_GRID, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    gs.fit(Xs, y_tr)
    return gs.best_params_


def _tune_xgb(Xs, y_tr):
    """RandomizedSearchCV for XGBoost (scale-invariant; receives scaled data)."""
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(
        XGBRegressor(**XGB_BASE), XGB_DIST, n_iter=20, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=1, random_state=42,
    )
    rs.fit(Xs, y_tr)
    return rs.best_params_


def _tune_ridge_alpha(Xs, y_tr):
    """Manual CV for Ridge alpha on already-scaled data."""
    tscv = TimeSeriesSplit(n_splits=3)
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [
            r2_score(y_tr[vi], Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
            for ti, vi in tscv.split(Xs)
        ]
        if np.mean(scores) > best_sc:
            best_sc, best_alpha = np.mean(scores), a
    return best_alpha


# ── Voting builder ─────────────────────────────────────────────────────────────

def _build_voting(X_tr, y_tr):
    """
    VotingRegressor: ElNet (tuned) + RF (fixed) + XGBoost (tuned).

    Replaces original Ridge+ElNet+RF which over-weighted L2 linear regularisation.
    Ridge removed; XGBoost added for structural diversity (gradient boosting
    vs. bagging vs. regularised regression).
    """
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)

    best_en_p  = _tune_elnet(Xs, y_tr)
    best_xgb_p = _tune_xgb(Xs, y_tr)

    estimators = [
        ("elnet", ElasticNet(
            alpha=best_en_p["alpha"], l1_ratio=best_en_p["l1_ratio"], max_iter=5000)),
        ("rf",    RandomForestRegressor(**RF_PARAMS)),
        ("xgb",   XGBRegressor(**XGB_BASE, **best_xgb_p)),
    ]
    # Pipeline: StandardScaler → VotingRegressor
    # RF and XGB are scale-invariant; scaling is harmless for them and
    # required for ElNet.
    voting = Pipeline([
        ("sc",     StandardScaler()),
        ("voting", VotingRegressor(estimators=estimators)),
    ])
    voting.fit(X_tr, y_tr)
    return voting, {"elnet": best_en_p, "xgb": best_xgb_p}


# ── Stacking builder ───────────────────────────────────────────────────────────

def _build_stacking(X_tr, y_tr):
    """
    Walk-forward stacking with TimeSeriesSplit(n_splits=5) OOF generation.

    Replaces the previous KFold(n_splits=5, shuffle=False) StackingRegressor
    which allowed future data to train base models when predicting past samples
    (look-ahead bias). Here, for every OOF prediction of sample i, the base
    models are trained ONLY on samples 0..(i-1), i.e. strictly past data.

    Approx. first 1/6 of training samples have no OOF coverage (TimeSeriesSplit
    constraint) and are excluded from meta-learner training. The fraction
    excluded is stored as `oof_coverage` in the predictor for reporting.

    Base models: ElNet + RF + XGBoost (same as Voting, consistent diversity).
    Meta-learner: Ridge (tuned alpha via TimeSeriesSplit on OOF data).
    Base models are retrained on full training set before deployment.
    """
    n = len(y_tr)
    tscv_outer = TimeSeriesSplit(n_splits=5)
    scaler_tmp = StandardScaler()
    Xs = scaler_tmp.fit_transform(X_tr)

    # Tune hyperparameters using inner CV on full training data
    best_en_p  = _tune_elnet(Xs, y_tr)
    best_xgb_p = _tune_xgb(Xs, y_tr)

    # ── Walk-forward OOF generation ────────────────────────────────────────────
    oof_en  = np.full(n, np.nan)
    oof_rf  = np.full(n, np.nan)
    oof_xgb = np.full(n, np.nan)

    for tr_idx, te_idx in tscv_outer.split(X_tr):
        # ElNet (needs scaling)
        en_fold = Pipeline([
            ("sc", StandardScaler()),
            ("en", ElasticNet(alpha=best_en_p["alpha"], l1_ratio=best_en_p["l1_ratio"],
                              max_iter=5000)),
        ])
        en_fold.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_en[te_idx] = en_fold.predict(X_tr[te_idx])

        # RF
        rf_fold = RandomForestRegressor(**RF_PARAMS)
        rf_fold.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_rf[te_idx] = rf_fold.predict(X_tr[te_idx])

        # XGB
        xgb_fold = XGBRegressor(**XGB_BASE, **best_xgb_p)
        xgb_fold.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_xgb[te_idx] = xgb_fold.predict(X_tr[te_idx])

    # Exclude samples with no OOF coverage (first fold's training portion)
    valid_mask = ~(np.isnan(oof_en) | np.isnan(oof_rf) | np.isnan(oof_xgb))
    n_valid     = valid_mask.sum()
    oof_coverage = n_valid / n

    meta_X_oof = np.column_stack([oof_en[valid_mask], oof_rf[valid_mask], oof_xgb[valid_mask]])
    meta_y_oof = y_tr[valid_mask]

    # Tune Ridge meta-learner on OOF meta-features (strictly past data only)
    tscv_meta = TimeSeriesSplit(n_splits=3)
    meta_alpha = _tune_ridge_alpha(meta_X_oof, meta_y_oof)
    meta_model = Ridge(alpha=meta_alpha)
    meta_model.fit(meta_X_oof, meta_y_oof)

    print(f"      Stacking OOF coverage: {n_valid}/{n} ({oof_coverage:.1%}), meta α={meta_alpha}")

    # ── Retrain base models on full training set ───────────────────────────────
    en_full = Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNet(alpha=best_en_p["alpha"], l1_ratio=best_en_p["l1_ratio"],
                          max_iter=5000)),
    ])
    en_full.fit(X_tr, y_tr)

    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(X_tr, y_tr)

    xgb_full = XGBRegressor(**XGB_BASE, **best_xgb_p)
    xgb_full.fit(X_tr, y_tr)

    predictor = WalkForwardStackingPredictor(
        base_models=[("en", en_full), ("rf", rf_full), ("xgb", xgb_full)],
        meta_model=meta_model,
        meta_alpha=meta_alpha,
        oof_coverage=oof_coverage,
    )
    return predictor, {"elnet": best_en_p, "xgb": best_xgb_p,
                       "meta_alpha": meta_alpha, "n_oof_valid": int(n_valid)}


# ── Per-dataset training ───────────────────────────────────────────────────────

def _per_year_metrics(model, train_df, feat_cols, target):
    """Return dict {year: (mae, rmse)} for each training year with ≥5 rows."""
    result = {}
    for yr in sorted(train_df["Date"].dt.year.unique()):
        mask = train_df["Date"].dt.year == yr
        if mask.sum() < 5:
            continue
        X_yr = train_df.loc[mask, feat_cols].values
        y_yr = train_df.loc[mask, target].values
        preds = model.predict(X_yr)
        result[yr] = (
            mean_absolute_error(y_yr, preds),
            float(np.sqrt(mean_squared_error(y_yr, preds))),
        )
    return result


def train_dataset(ds: dict, run: int) -> list:
    name   = ds["name"]
    target = ds["target"]

    df = pd.read_excel(ds["file"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feat_cols = infer_features(df, target)
    df_clean  = df[["Date"] + feat_cols + [target]].dropna()

    train = df_clean[df_clean["Date"].dt.year < 2025]
    test  = df_clean[df_clean["Date"].dt.year == 2025]

    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    records   = []
    df_orig   = pd.read_excel(ds["file"])
    df_orig["Date"] = pd.to_datetime(df_orig["Date"])

    for model_tag, builder_fn in [
        ("Voting",   lambda: _build_voting(X_train, y_train)),
        ("Stacking", lambda: _build_stacking(X_train, y_train)),
    ]:
        print(f"    [{model_tag}] Fitting...")
        try:
            result = builder_fn()
            model  = result[0]
            params = result[1] if len(result) > 1 else {}

            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)

            r2_tr, rmse_tr, mae_tr, _        = _metrics(y_train, y_tr_pred)
            r2_te, rmse_te, mae_te, mape_te  = _metrics(y_test,  y_te_pred)
            gap = r2_tr - r2_te

            print(f"      Train R²={r2_tr:+.3f}  Test R²={r2_te:+.3f}  "
                  f"Gap={gap:+.3f}  MAE_test={mae_te:.3f}")

            # ── Diagnostic metrics (Point 4) ───────────────────────────────
            y_train_std = float(np.std(y_train))
            y_test_std  = float(np.std(y_test))
            nrmse_test  = _nrmse(rmse_te, y_test)

            # MAE and RMSE on 2024 training data as holdout baseline
            yr_metrics = _per_year_metrics(model, train, feat_cols, target)
            mae_2024  = yr_metrics.get(2024, (np.nan, np.nan))[0]
            rmse_2024 = yr_metrics.get(2024, (np.nan, np.nan))[1]

            print(f"      MAE_2024={mae_2024:.3f}  MAE_2025={mae_te:.3f}  "
                  f"RMSE_2024={rmse_2024:.3f}  RMSE_2025={rmse_te:.3f}  "
                  f"y_train_std={y_train_std:.3f}  y_test_std={y_test_std:.3f}")

            # Save model
            model_path = os.path.join(MODELS_DIR,
                                      f"{name}_{model_tag.lower()}_run_{run}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Plots
            _scatter_plot(y_train, y_tr_pred, y_test, y_te_pred,
                          r2_tr, r2_te, name, model_tag, run)
            _learning_curve_plot(model, X_train, y_train, name, model_tag, run)

            df_full = df[["Date"] + feat_cols + [target]].dropna().copy()
            _timeseries_plot(df_full, target,
                             model.predict(df_full[feat_cols].values),
                             name, model_tag, run)

            # Append predictions to dataset xlsx
            pred_col   = f"predicted_{model_tag}_run_{run}"
            feat_avail = [c for c in feat_cols if c in df_orig.columns]
            if target in df_orig.columns and all(c in df_orig.columns for c in feat_avail):
                mask_orig = df_orig[feat_avail + [target]].notna().all(axis=1)
                df_orig.loc[:, pred_col] = np.nan
                df_orig.loc[mask_orig, pred_col] = model.predict(
                    df_orig.loc[mask_orig, feat_avail].values
                )

            records.append({
                "experiment": ds["experiment"],
                "model_name": (f"p9_{model_tag.lower()}_"
                               f"{name.split('_')[-2]}_{name.split('_')[-1]}"),
                "run":        run,
                "model":      model_tag,
                "target":     target,
                "n_train":    len(train),
                "n_test":     len(test),
                "n_features": len(feat_cols),
                "R2_train":   r2_tr,
                "RMSE_train": rmse_tr,
                "MAE_train":  mae_tr,
                "R2_test":    r2_te,
                "RMSE_test":  rmse_te,
                "MAE_test":   mae_te,
                "NRMSE_test": nrmse_test,
                "MAPE_test":  mape_te,
                "R2_gap":     gap,
                "MAE_2024":    mae_2024,
                "RMSE_2024":   rmse_2024,
                "y_train_std": y_train_std,
                "y_test_std":  y_test_std,
                "best_params": str(params),
            })
        except Exception as e:
            print(f"      ERROR in {model_tag}: {e}")
            import traceback; traceback.print_exc()

    # Write back dataset xlsx with both new prediction columns
    df_orig.to_excel(ds["file"], index=False)
    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run = _next_run(RESULTS_FILE)
    print(f"=== Phase 9 — Ensemble Models (Voting + Stacking) — Run {run} ===")
    print(f"  Voting:   ElNet + RF + XGB (equal weights)")
    print(f"  Stacking: ElNet + RF + XGB base → Ridge meta (walk-forward OOF)")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n")

    all_records = []
    for ds in DATASETS:
        print(f"\n[{ds['name']}] target={ds['target']}")
        try:
            recs = train_dataset(ds, run)
            all_records.extend(recs)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if all_records:
        results = pd.DataFrame(all_records)
        if os.path.exists(RESULTS_FILE):
            old = pd.read_excel(RESULTS_FILE)
            results = pd.concat([old, results], ignore_index=True)
        results.to_excel(RESULTS_FILE, index=False)
        print(f"\nResults saved → {RESULTS_FILE}")

        print("\n=== Summary ===")
        for model_tag in ["Voting", "Stacking"]:
            sub = [r for r in all_records if r["model"] == model_tag]
            if not sub:
                continue
            avg_r2 = np.mean([r["R2_test"] for r in sub])
            print(f"\n  {model_tag} (ElNet+RF+XGB):")
            for r in sub:
                print(f"    {r['target']:45s}  "
                      f"R²={r['R2_test']:+.3f}  MAE={r['MAE_test']:.3f}  "
                      f"MAE_2024={r['MAE_2024']:.3f}  "
                      f"σ_train={r['y_train_std']:.3f}  σ_test={r['y_test_std']:.3f}")
            print(f"    Avg Test R²: {avg_r2:+.3f}")

        # Variance diagnosis for targets with negative R²
        print("\n  === Variance Collapse Diagnosis (R² < 0 targets) ===")
        for r in all_records:
            if r["R2_test"] < 0:
                ratio = r["y_test_std"] / r["y_train_std"] if r["y_train_std"] > 0 else np.nan
                delta_mae  = r["MAE_test"]  - r["MAE_2024"]
                delta_rmse = r["RMSE_test"] - r["RMSE_2024"]
                print(f"    {r['model']:10s} {r['target']:45s}  "
                      f"R²={r['R2_test']:+.3f}  "
                      f"σ-ratio={ratio:.2f}  "
                      f"ΔMAE={delta_mae:+.3f} (2024:{r['MAE_2024']:.3f}→2025:{r['MAE_test']:.3f})  "
                      f"ΔRMSE={delta_rmse:+.3f} (2024:{r['RMSE_2024']:.3f}→2025:{r['RMSE_test']:.3f})")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
