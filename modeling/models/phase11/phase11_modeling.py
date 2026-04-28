"""
phase11_modeling.py - Lag / rolling features + log-transformed targets.

Two modeling changes layered on Exp3-S2 datasets:

  1. TEMPORAL FEATURES - for each continuous base feature we add:
       shift(1), shift(3), shift(7)      (previous observations)
       rolling(7D).mean()                (calendar-aware 7-day mean)
     Added only to continuous features (inlets, Flow, Power, secondary, aeration).
     year is left untouched (not lag-expanded).

  2. LOG-TRANSFORMED TARGETS - BOD/COD/TSS are right-skewed non-negative
     concentrations. We fit on log1p(y) and back-transform predictions with
     Duan's smearing estimator:
            ŷ = expm1(μ_log) * mean(exp(resid_log))
     Grab/Comp pH is left untransformed (already on a log-activity scale).

Models: Ridge, ElNet, RF, XGB, Voting (ElNet+RF+XGB).
Tuning: GridSearch/RandomizedSearch with TimeSeriesSplit(n_splits=5).

Results and diagnostics include CV-R² so that the downstream selection
script can use Gap_generalisation = CV_R² − Test_R²  as a distribution-shift
indicator instead of Train R² − Test R² which blends two failure modes.

Run:  .venv/bin/python3 21-25/modeling/models/phase11/phase11_modeling.py
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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DS_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = [
    {"name": "grab_BOD", "target": "Effluent BOD (mg/L, Grab)",      "log_y": True},
    {"name": "grab_COD", "target": "Effluent COD (mg/L, Grab)",      "log_y": True},
    {"name": "grab_TSS", "target": "Effluent TSS (mg/L, Grab)",      "log_y": True},
    {"name": "grab_pH",  "target": "Effluent pH (Grab)",             "log_y": False},
    {"name": "comp_BOD", "target": "Effluent BOD (mg/L, Composite)", "log_y": True},
    {"name": "comp_COD", "target": "Effluent COD (mg/L, Composite)", "log_y": True},
    {"name": "comp_TSS", "target": "Effluent TSS (mg/L, Composite)", "log_y": True},
    {"name": "comp_pH",  "target": "Effluent pH (Composite)",        "log_y": False},
]

# Columns that should NOT get lag/rolling expansion
NO_TEMPORAL     = {"Date", "year"}

# Lag and rolling window configuration
LAG_SHIFTS      = [1, 3, 7]
ROLL_WINDOW     = "7D"

# Hyperparameter grids
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
    "max_depth":        [2, 3, 4, 5],
    "min_child_weight": [5, 10, 20],
    "reg_alpha":        [0.0, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.0, 5.0],
    "gamma":            [0.0, 0.1, 1.0],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def build_temporal_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    For a restricted subset of high-leverage continuous features, append:
      - shift(k) for k in LAG_SHIFTS
      - rolling(ROLL_WINDOW) mean (calendar-aware, closed='left')

    Rolling mean excludes the current row (closed='left') to avoid label-
    adjacent leakage. Scope is limited to upstream inlets + hydraulic/energy
    drivers so that the feature-to-sample ratio stays tractable on small
    composite datasets (n_train ≈ 290). Aeration/secondary columns are left
    as-is - they already reflect smoothed operational state.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()
    date_idx = pd.DatetimeIndex(df["Date"])

    # Only expand columns matching these substrings (case-insensitive).
    # This caps engineered features at ~20-28 per target instead of ~100.
    TEMPORAL_KEYS = ("inlet", "flow (mld)", "power total")

    def _is_temporal(c: str) -> bool:
        lc = c.lower()
        return any(k in lc for k in TEMPORAL_KEYS)

    base_cols = [c for c in df.columns
                 if c not in NO_TEMPORAL and c != target
                 and not c.startswith("predicted_")
                 and pd.api.types.is_numeric_dtype(df[c])
                 and _is_temporal(c)]

    new_frames = [df]
    for col in base_cols:
        # Shifts (previous observations)
        lag_df = pd.DataFrame({
            f"{col}__lag{k}": df[col].shift(k) for k in LAG_SHIFTS
        })
        # Calendar rolling mean using closed='left' (does NOT include current row).
        s = pd.Series(df[col].values, index=date_idx)
        roll = s.rolling(ROLL_WINDOW, closed="left", min_periods=2).mean()
        roll_df = pd.DataFrame({f"{col}__roll7d": roll.values})
        new_frames.extend([lag_df, roll_df])

    out = pd.concat(new_frames, axis=1)
    return out


def infer_features(df: pd.DataFrame, target: str) -> list:
    exclude = {"Date", target}
    return [c for c in df.columns
            if c not in exclude and not c.startswith("predicted_")]


# ═══════════════════════════════════════════════════════════════════════════════
# Target transformation (log1p + Duan smearing)
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_transform_y(y_train: np.ndarray, log_y: bool):
    """Return (y_trans, meta-dict). Meta holds info for back-transform."""
    if not log_y:
        return y_train, {"log_y": False, "smear": 1.0}
    # Guard: all training values must be ≥ 0 for log1p
    if (y_train < 0).any():
        # Shift minimally so log1p remains defined
        offset = -y_train.min() + 1e-3
    else:
        offset = 0.0
    y_log = np.log1p(y_train + offset)
    return y_log, {"log_y": True, "offset": offset, "smear": None}


def _back_transform(y_pred_trans: np.ndarray, y_train_trans: np.ndarray,
                    y_train_orig: np.ndarray, meta: dict) -> np.ndarray:
    """Duan's smearing estimator for log1p targets; identity otherwise."""
    if not meta["log_y"]:
        return y_pred_trans
    # Compute smearing on TRAINING residuals: e_i = y_train_log − ŷ_train_log.
    # Smear factor = mean(exp(e_i)). Applied multiplicatively in exp-space.
    # Since we use log1p, back-transform is expm1, then subtract offset.
    offset = meta.get("offset", 0.0)
    # Approximation: we don't have ŷ_train_trans here, pass it separately via meta
    smear = meta.get("smear", 1.0)
    return np.expm1(y_pred_trans) * smear - offset


def _compute_smear(y_train_trans, y_train_pred_trans):
    resid = y_train_trans - y_train_pred_trans
    return float(np.mean(np.exp(resid)))


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    return r2, rmse, mae, mape


def _cv_r2(estimator, X, y, n_splits=5):
    """Mean R² across TimeSeriesSplit folds. estimator must be cloneable."""
    from sklearn.base import clone
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        m = clone(estimator)
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[va], m.predict(X[va])))
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders  (all operate on the transformed target y_t)
# ═══════════════════════════════════════════════════════════════════════════════

def _tune_ridge(Xs, y_t):
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        sc = [
            r2_score(y_t[va], Ridge(alpha=a).fit(Xs[tr], y_t[tr]).predict(Xs[va]))
            for tr, va in tscv.split(Xs)
        ]
        if np.mean(sc) > best_sc:
            best_sc, best_alpha = np.mean(sc), a
    return best_alpha


def _tune_elnet(Xs, y_t):
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        ElasticNet(max_iter=5000), ELNET_GRID, cv=tscv,
        scoring="r2", n_jobs=-1,
    )
    gs.fit(Xs, y_t)
    return gs.best_params_


def _tune_xgb(Xs, y_t):
    tscv = TimeSeriesSplit(n_splits=5)
    rs = RandomizedSearchCV(
        XGBRegressor(**XGB_BASE), XGB_DIST, n_iter=30, cv=tscv,
        scoring="r2", n_jobs=1, random_state=42,
    )
    rs.fit(Xs, y_t)
    return rs.best_params_


def build_ridge(X_tr, y_t):
    pipe_probe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    # Tune on scaled data
    Xs = StandardScaler().fit_transform(X_tr)
    alpha = _tune_ridge(Xs, y_t)
    pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])
    pipe.fit(X_tr, y_t)
    return pipe, {"alpha": alpha}


def build_elnet(X_tr, y_t):
    Xs = StandardScaler().fit_transform(X_tr)
    best = _tune_elnet(Xs, y_t)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("m",  ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], max_iter=5000)),
    ])
    pipe.fit(X_tr, y_t)
    return pipe, best


def build_rf(X_tr, y_t):
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X_tr, y_t)
    return m, RF_PARAMS


def build_xgb(X_tr, y_t):
    Xs = StandardScaler().fit_transform(X_tr)  # tuning is scale-invariant but consistent
    best = _tune_xgb(Xs, y_t)
    m = XGBRegressor(**XGB_BASE, **best)
    m.fit(X_tr, y_t)
    return m, best


def build_voting(X_tr, y_t):
    Xs = StandardScaler().fit_transform(X_tr)
    best_en  = _tune_elnet(Xs, y_t)
    best_xgb = _tune_xgb(Xs, y_t)
    estimators = [
        ("elnet", ElasticNet(alpha=best_en["alpha"], l1_ratio=best_en["l1_ratio"], max_iter=5000)),
        ("rf",    RandomForestRegressor(**RF_PARAMS)),
        ("xgb",   XGBRegressor(**XGB_BASE, **best_xgb)),
    ]
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("v",  VotingRegressor(estimators=estimators)),
    ])
    pipe.fit(X_tr, y_t)
    return pipe, {"elnet": best_en, "xgb": best_xgb}


MODEL_BUILDERS = [
    ("Ridge",  build_ridge),
    ("ElNet",  build_elnet),
    ("RF",     build_rf),
    ("XGB",    build_xgb),
    ("Voting", build_voting),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _scatter_plot(y_train, y_train_pred, y_test, y_test_pred,
                  r2_train, r2_test, name, model_tag, run):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, yt, yp, r2, label, color in [
        (ax1, y_train, y_train_pred, r2_train, "Train", "#4A90D9"),
        (ax2, y_test,  y_test_pred,  r2_test,  "Test",  "#E15252"),
    ]:
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "w--", lw=0.8, alpha=0.5)
        ax.scatter(yt, yp, alpha=0.55, s=18, color=color)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{label} - R²={r2:+.3f}", fontsize=9)
    fig.suptitle(f"{model_tag} - {name}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset training
# ═══════════════════════════════════════════════════════════════════════════════

def _next_run() -> int:
    if not os.path.exists(RESULTS_FILE):
        return 1
    df = pd.read_excel(RESULTS_FILE)
    return int(df["run"].max()) + 1 if len(df) > 0 else 1


def _per_year_mae(model, train_df, feat_cols, target, meta, smear):
    """Return {year: (mae, rmse)} on original scale for each training year."""
    result = {}
    for yr in sorted(train_df["Date"].dt.year.unique()):
        mask = train_df["Date"].dt.year == yr
        if mask.sum() < 5:
            continue
        X_yr = train_df.loc[mask, feat_cols].values
        y_yr = train_df.loc[mask, target].values
        pred_t = model.predict(X_yr)
        meta_use = {**meta, "smear": smear}
        pred_o = _back_transform(pred_t, None, None, meta_use)
        result[yr] = (
            float(mean_absolute_error(y_yr, pred_o)),
            float(np.sqrt(mean_squared_error(y_yr, pred_o))),
        )
    return result


def train_dataset(ds: dict, run: int) -> list:
    name   = ds["name"]
    target = ds["target"]
    log_y  = ds["log_y"]

    path = os.path.join(DS_DIR, f"{name}.xlsx")
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop old predicted_* columns to keep feature set clean
    df = df[[c for c in df.columns if not c.startswith("predicted_")]]

    # Build temporal features
    df_feat = build_temporal_features(df, target)

    feat_cols = infer_features(df_feat, target)
    n_base = sum(1 for c in feat_cols if "__" not in c)
    n_lag  = sum(1 for c in feat_cols if "__lag" in c)
    n_roll = sum(1 for c in feat_cols if "__roll" in c)

    # Drop rows with NaN in features or target (first 7 rows will be NaN from lags/roll)
    df_clean = df_feat[["Date"] + feat_cols + [target]].dropna()
    train = df_clean[df_clean["Date"].dt.year < 2025]
    test  = df_clean[df_clean["Date"].dt.year == 2025]

    if len(train) < 30 or len(test) < 10:
        print(f"  SKIP (n_train={len(train)}, n_test={len(test)})")
        return []

    X_train = train[feat_cols].values.astype(float)
    X_test  = test[feat_cols].values.astype(float)
    y_train = train[target].values.astype(float)
    y_test  = test[target].values.astype(float)

    y_train_t, meta = _fit_transform_y(y_train, log_y)

    print(f"  n_train={len(train)}  n_test={len(test)}  "
          f"features: base={n_base}  lag={n_lag}  roll={n_roll}  log_y={log_y}")

    records = []
    for model_tag, builder in MODEL_BUILDERS:
        print(f"    [{model_tag}] fitting…")
        try:
            model, params = builder(X_train, y_train_t)

            # In-sample predictions (transformed scale)
            y_tr_pred_t = model.predict(X_train)
            y_te_pred_t = model.predict(X_test)

            # Smearing factor from training residuals
            smear = _compute_smear(y_train_t, y_tr_pred_t) if log_y else 1.0
            meta_use = {**meta, "smear": smear}

            # Back-transform to original scale
            y_tr_pred = _back_transform(y_tr_pred_t, None, None, meta_use)
            y_te_pred = _back_transform(y_te_pred_t, None, None, meta_use)

            # Metrics on ORIGINAL scale
            r2_tr, rmse_tr, mae_tr, _        = _metrics(y_train, y_tr_pred)
            r2_te, rmse_te, mae_te, mape_te  = _metrics(y_test,  y_te_pred)
            gap = r2_tr - r2_te

            # CV R² on transformed scale with the tuned estimator's template.
            # For Pipelines we clone safely; for Voting we skip (expensive).
            try:
                if model_tag != "Voting":
                    cv_r2 = _cv_r2(model, X_train, y_train_t, n_splits=5)
                else:
                    cv_r2 = np.nan
            except Exception:
                cv_r2 = np.nan

            gap_gen = (cv_r2 - r2_te) if not np.isnan(cv_r2) else np.nan

            # Per-year MAE for distribution-shift diagnosis
            yr_metrics = _per_year_mae(model, train, feat_cols, target, meta, smear)
            mae_2024, rmse_2024 = yr_metrics.get(2024, (np.nan, np.nan))

            print(f"      Train R²={r2_tr:+.3f}  Test R²={r2_te:+.3f}  "
                  f"Gap={gap:+.3f}  CV R²={cv_r2:+.3f}  "
                  f"GenGap={gap_gen:+.3f}  MAE2024={mae_2024:.3f}→MAE2025={mae_te:.3f}")

            # Save artefacts
            mp = os.path.join(MODELS_DIR, f"{name}_{model_tag.lower()}_run_{run}.pkl")
            with open(mp, "wb") as f:
                pickle.dump({"model": model, "meta": meta_use, "feat_cols": feat_cols}, f)

            _scatter_plot(y_train, y_tr_pred, y_test, y_te_pred,
                          r2_tr, r2_te, name, model_tag, run)

            records.append({
                "experiment":  "Phase11-Temporal-LogY",
                "model_name":  f"p11_{model_tag.lower()}_"
                               f"{name.split('_')[-2]}_{name.split('_')[-1]}",
                "run":         run,
                "model":       model_tag,
                "target":      target,
                "log_y":       log_y,
                "n_train":     len(train),
                "n_test":      len(test),
                "n_features":  len(feat_cols),
                "n_base_features": n_base,
                "n_lag":       n_lag,
                "n_roll":      n_roll,
                "R2_train":    r2_tr,
                "RMSE_train":  rmse_tr,
                "MAE_train":   mae_tr,
                "R2_test":     r2_te,
                "RMSE_test":   rmse_te,
                "MAE_test":    mae_te,
                "MAPE_test":   mape_te,
                "R2_gap":      gap,
                "CV_R2":       cv_r2,
                "Gap_gen":     gap_gen,
                "MAE_2024":    mae_2024,
                "RMSE_2024":   rmse_2024,
                "smear":       smear,
                "best_params": str(params),
            })
        except Exception as e:
            print(f"      ERROR in {model_tag}: {e}")
            import traceback; traceback.print_exc()

    return records


def main():
    run = _next_run()
    print(f"=== Phase 11 - Temporal features + Log-y - Run {run} ===")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n")

    all_records = []
    for ds in DATASETS:
        print(f"\n[{ds['name']}] target={ds['target']}  log_y={ds['log_y']}")
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

        print("\n=== Summary (Test R², gap, CV R²) ===")
        for mt in ["Ridge", "ElNet", "RF", "XGB", "Voting"]:
            sub = [r for r in all_records if r["model"] == mt]
            if not sub:
                continue
            avg_r2 = float(np.mean([r["R2_test"] for r in sub]))
            avg_gap = float(np.mean([r["R2_gap"] for r in sub]))
            print(f"\n  {mt}  (avg Test R² = {avg_r2:+.3f}, avg gap = {avg_gap:+.3f})")
            for r in sub:
                print(f"    {r['target']:45s}  "
                      f"R²_te={r['R2_test']:+.3f}  gap={r['R2_gap']:+.3f}  "
                      f"CV={r['CV_R2']:+.3f}  MAE={r['MAE_test']:.3f}")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
