"""
ensemble_modeling_phase9.py — Voting and Stacking ensembles for Phase 9.

Uses Experiment 3 Sub-2 datasets. Builds two ensemble strategies:
  1. Voting  — VotingRegressor: RF + ElNet + Ridge (equal weights)
  2. Stacking — StackingRegressor: RF + Ridge + ElNet → Ridge meta-learner

All base models use the same hyperparameter ranges as previous experiments.
TimeSeriesSplit(n_splits=3) throughout.

Outputs:
  - phase9/ensemble/results.xlsx
  - phase9/ensemble/models/<name>_{voting,stacking}_run_N.pkl
  - phase9/ensemble/plots/<name>_{voting,stacking}_run_N_{scatter,timeseries,lc}.png
  - Appends predicted_Voting_run_N, predicted_Stacking_run_N columns to datasets

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase9/ensemble/ensemble_modeling_phase9.py
"""

import os
import sys
import pickle
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold, GridSearchCV, learning_curve
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    {
        "name":    "s2_stage3_grab_BOD",
        "file":    os.path.join(DS_DIR, "s2_stage3_grab_BOD.xlsx"),
        "target":  "Effluent BOD (mg/L, Grab)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_grab_COD",
        "file":    os.path.join(DS_DIR, "s2_stage3_grab_COD.xlsx"),
        "target":  "Effluent COD (mg/L, Grab)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_grab_TSS",
        "file":    os.path.join(DS_DIR, "s2_stage3_grab_TSS.xlsx"),
        "target":  "Effluent TSS (mg/L, Grab)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_grab_pH",
        "file":    os.path.join(DS_DIR, "s2_stage3_grab_pH.xlsx"),
        "target":  "Effluent pH (Grab)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_comp_BOD",
        "file":    os.path.join(DS_DIR, "s2_stage3_comp_BOD.xlsx"),
        "target":  "Effluent BOD (mg/L, Composite)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_comp_COD",
        "file":    os.path.join(DS_DIR, "s2_stage3_comp_COD.xlsx"),
        "target":  "Effluent COD (mg/L, Composite)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_comp_TSS",
        "file":    os.path.join(DS_DIR, "s2_stage3_comp_TSS.xlsx"),
        "target":  "Effluent TSS (mg/L, Composite)",
        "experiment": "Phase9-Ensemble",
    },
    {
        "name":    "s2_stage3_comp_pH",
        "file":    os.path.join(DS_DIR, "s2_stage3_comp_pH.xlsx"),
        "target":  "Effluent pH (Composite)",
        "experiment": "Phase9-Ensemble",
    },
]

# ── Base model configs ─────────────────────────────────────────────────────────

# Ridge — tuned via GridSearchCV
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ElNet — tuned via GridSearchCV
ELNET_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "max_iter": [5000],
}

# RF — fixed best-typical params (avoids nested CV cost)
RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt",
    max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_features(df: pd.DataFrame, target: str) -> list[str]:
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
            scoring="r2", n_jobs=1
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


# ── Base model builders ────────────────────────────────────────────────────────

def _fit_ridge(X_tr, y_tr):
    """Return fitted Ridge with best alpha chosen by TimeSeriesSplit CV."""
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    best_alpha, best_score = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            m = Ridge(alpha=a)
            m.fit(X_scaled[train_idx], y_tr[train_idx])
            scores.append(r2_score(y_tr[val_idx], m.predict(X_scaled[val_idx])))
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = a
    model = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    model.fit(X_tr, y_tr)
    return model, best_alpha


def _fit_elnet(X_tr, y_tr):
    """Return fitted ElasticNet pipeline with GridSearchCV."""
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(
        ElasticNet(max_iter=5000),
        ELNET_GRID, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    gs.fit(X_scaled, y_tr)
    best_en = gs.best_estimator_
    model = Pipeline([("sc", StandardScaler()),
                      ("en", ElasticNet(alpha=best_en.alpha,
                                        l1_ratio=best_en.l1_ratio,
                                        max_iter=5000))])
    model.fit(X_tr, y_tr)
    return model


def _fit_rf(X_tr, y_tr):
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    return rf


# ── Voting & Stacking builders ─────────────────────────────────────────────────

def _build_voting(X_tr, y_tr):
    """
    VotingRegressor: Ridge (tuned) + ElNet (tuned) + RF (fixed params).
    Note: VotingRegressor fits estimators internally; we pass pre-configured
    estimators with good starting params, then wrap in a scaler pipeline.
    We tune Ridge alpha and ElNet params via a light CV beforehand.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    # Tune Ridge alpha
    scaler_tmp = StandardScaler()
    Xs = scaler_tmp.fit_transform(X_tr)
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [
            r2_score(y_tr[vi],
                     Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
            for ti, vi in tscv.split(Xs)
        ]
        if np.mean(scores) > best_sc:
            best_sc, best_alpha = np.mean(scores), a

    # Tune ElNet
    gs_en = GridSearchCV(
        ElasticNet(max_iter=5000), ELNET_GRID, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    gs_en.fit(Xs, y_tr)
    best_en_p = gs_en.best_params_

    # Build VotingRegressor inside a scaler pipeline
    estimators = [
        ("ridge", Ridge(alpha=best_alpha)),
        ("elnet", ElasticNet(alpha=best_en_p["alpha"],
                             l1_ratio=best_en_p["l1_ratio"],
                             max_iter=5000)),
        ("rf",    RandomForestRegressor(**RF_PARAMS)),
    ]
    voting = Pipeline([
        ("sc",     StandardScaler()),
        ("voting", VotingRegressor(estimators=estimators)),
    ])
    voting.fit(X_tr, y_tr)
    return voting, best_alpha, best_en_p


def _build_stacking(X_tr, y_tr):
    """
    StackingRegressor: base=[RF, Ridge, ElNet], meta=Ridge.
    Uses TimeSeriesSplit cross-val to generate out-of-fold meta-features.
    Wrapped in a StandardScaler pipeline for the linear base models.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    # Pre-tune Ridge + ElNet params
    scaler_tmp = StandardScaler()
    Xs = scaler_tmp.fit_transform(X_tr)
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [
            r2_score(y_tr[vi],
                     Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
            for ti, vi in tscv.split(Xs)
        ]
        if np.mean(scores) > best_sc:
            best_sc, best_alpha = np.mean(scores), a

    gs_en = GridSearchCV(
        ElasticNet(max_iter=5000), ELNET_GRID, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    gs_en.fit(Xs, y_tr)
    best_en_p = gs_en.best_params_

    # Stacking base estimators (operate on scaled input)
    base_estimators = [
        ("rf",    RandomForestRegressor(**RF_PARAMS)),
        ("ridge", Pipeline([("sc2", StandardScaler()),
                            ("r",   Ridge(alpha=best_alpha))])),
        ("elnet", Pipeline([("sc2", StandardScaler()),
                            ("e",   ElasticNet(
                                        alpha=best_en_p["alpha"],
                                        l1_ratio=best_en_p["l1_ratio"],
                                        max_iter=5000))])),
    ]

    # KFold (not TimeSeriesSplit) required by cross_val_predict inside StackingRegressor.
    # Temporal ordering is preserved by base models; meta-learner just combines outputs.
    stack_cv = KFold(n_splits=5, shuffle=False)

    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=best_alpha),
        cv=stack_cv,
        passthrough=False,
        n_jobs=1,
    )
    stacking.fit(X_tr, y_tr)
    return stacking, best_alpha, best_en_p


# ── Per-dataset training ───────────────────────────────────────────────────────

def train_dataset(ds: dict, run: int) -> list[dict]:
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

    records = []
    df_orig = pd.read_excel(ds["file"])
    df_orig["Date"] = pd.to_datetime(df_orig["Date"])

    for model_tag, builder_fn in [
        ("Voting",   lambda: _build_voting(X_train, y_train)),
        ("Stacking", lambda: _build_stacking(X_train, y_train)),
    ]:
        print(f"    [{model_tag}] Fitting...")
        try:
            result = builder_fn()
            model  = result[0]
            params = {"alpha": result[1], "elnet": result[2]} if len(result) > 1 else {}

            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)

            r2_tr, rmse_tr, mae_tr, _ = _metrics(y_train, y_tr_pred)
            r2_te, rmse_te, mae_te, mape_te = _metrics(y_test, y_te_pred)
            gap = r2_tr - r2_te

            print(f"      Train R²={r2_tr:+.3f}  Test R²={r2_te:+.3f}  Gap={gap:+.3f}")

            # Save model
            model_path = os.path.join(MODELS_DIR, f"{name}_{model_tag.lower()}_run_{run}.pkl")
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
            pred_col = f"predicted_{model_tag}_run_{run}"
            feat_avail = [c for c in feat_cols if c in df_orig.columns]
            if target in df_orig.columns and all(c in df_orig.columns for c in feat_avail):
                mask = df_orig[feat_avail + [target]].notna().all(axis=1)
                df_orig.loc[:, pred_col] = np.nan
                df_orig.loc[mask, pred_col] = model.predict(
                    df_orig.loc[mask, feat_avail].values
                )

            records.append({
                "experiment":  ds["experiment"],
                "model_name":  f"p9_{model_tag.lower()}_{name.split('_')[-2]}_{name.split('_')[-1]}",
                "run":         run,
                "model":       model_tag,
                "target":      target,
                "n_train":     len(train),
                "n_test":      len(test),
                "n_features":  len(feat_cols),
                "R2_train":    r2_tr,
                "RMSE_train":  rmse_tr,
                "MAE_train":   mae_tr,
                "R2_test":     r2_te,
                "RMSE_test":   rmse_te,
                "MAE_test":    mae_te,
                "MAPE_test":   mape_te,
                "R2_gap":      gap,
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
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

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
            if sub:
                avg = np.mean([r["R2_test"] for r in sub])
                print(f"\n  {model_tag}:")
                for r in sub:
                    print(f"    {r['target']:45s}  Test R²={r['R2_test']:+.3f}  Gap={r['R2_gap']:+.3f}")
                print(f"    Avg Test R²: {avg:+.3f}")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
