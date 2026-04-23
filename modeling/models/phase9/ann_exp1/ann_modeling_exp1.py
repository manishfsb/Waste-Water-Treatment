"""
ann_modeling_exp1.py - ANN (MLPRegressor) on Experiment 1 datasets.

Diagnostic follow-up to Phase 9 ANN (Exp3-S2, avg Test R²=−1.12).
Experiment 1 has ~1175 Grab / ~800 Composite training rows vs ~470/290 in Exp3-S2.
Tests whether ANN failure was data-volume limited rather than architecture limited.

Features: Inlet pH, Inlet BOD, Inlet COD, Inlet TSS + Flow, Power (9 features).
Architecture: same as Phase 9 ANN — StandardScaler + MLPRegressor, GridSearchCV
(TimeSeriesSplit). Larger architectures added to param grid given more data.

Outputs:
  - phase9/ann_exp1/results.xlsx
  - phase9/ann_exp1/models/<name>_ann_run_N.pkl
  - phase9/ann_exp1/plots/<name>_ann_run_N_{scatter,timeseries,lc}.png

Usage (from project root):
    .venv/bin/python3 modeling/models/phase9/ann_exp1/ann_modeling_exp1.py
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DS_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment1", "sub_exp2")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = [
    {
        "name":       "grab_BOD",
        "file":       os.path.join(DS_DIR, "stage1_grab_BOD.xlsx"),
        "target":     "Effluent BOD (mg/L, Grab)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "grab_COD",
        "file":       os.path.join(DS_DIR, "stage1_grab_COD.xlsx"),
        "target":     "Effluent COD (mg/L, Grab)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "grab_TSS",
        "file":       os.path.join(DS_DIR, "stage1_grab_TSS.xlsx"),
        "target":     "Effluent TSS (mg/L, Grab)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "grab_pH",
        "file":       os.path.join(DS_DIR, "stage1_grab_pH.xlsx"),
        "target":     "Effluent pH (Grab)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "comp_BOD",
        "file":       os.path.join(DS_DIR, "stage1_comp_BOD.xlsx"),
        "target":     "Effluent BOD (mg/L, Composite)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "comp_COD",
        "file":       os.path.join(DS_DIR, "stage1_comp_COD.xlsx"),
        "target":     "Effluent COD (mg/L, Composite)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "comp_TSS",
        "file":       os.path.join(DS_DIR, "stage1_comp_TSS.xlsx"),
        "target":     "Effluent TSS (mg/L, Composite)",
        "experiment": "ANN-Exp1",
    },
    {
        "name":       "comp_pH",
        "file":       os.path.join(DS_DIR, "stage1_comp_pH.xlsx"),
        "target":     "Effluent pH (Composite)",
        "experiment": "ANN-Exp1",
    },
]

# ── Hyperparameter grid ────────────────────────────────────────────────────────
# Larger architectures added vs Phase 9 ANN: more data (1175 rows) supports deeper nets.
MLP_PARAM_GRID = {
    "mlp__hidden_layer_sizes": [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (64, 32, 16),
        (256, 128),
        (128, 64, 32),
    ],
    "mlp__alpha": [0.001, 0.01, 0.1, 1.0],
}

MLP_BASE = dict(
    activation="relu",
    solver="adam",
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    learning_rate_init=0.001,
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


# ── Plotting ───────────────────────────────────────────────────────────────────

def _scatter_plot(y_train, y_train_pred, y_test, y_test_pred,
                  r2_train, r2_test, name, run):
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
        ax.set_title(f"{label} - R²={r2:+.3f}", fontsize=9)
    fig.suptitle(f"ANN-Exp1 - {name}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_ANN_run_{run}_scatter.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _timeseries_plot(df_full, target, y_pred_full, name, run):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_full["Date"], df_full[target], color="white", lw=1.2,
            alpha=0.8, label="Actual")
    ax.plot(df_full["Date"], y_pred_full, color="#4FC3F7", lw=1.0,
            alpha=0.85, label="ANN Predicted")
    split_date = df_full.loc[df_full["Date"].dt.year == 2025, "Date"].min()
    if pd.notna(split_date):
        ax.axvline(split_date, color="#F0B849", lw=1.2, linestyle="--",
                   alpha=0.7, label="Train | Test")
    ax.set_title(f"ANN-Exp1 - {target}", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_ANN_run_{run}_timeseries.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _learning_curve_plot(pipeline, X_train, y_train, name, run):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes = np.linspace(0.2, 1.0, 8)
    try:
        ts, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train,
            cv=tscv, train_sizes=train_sizes,
            scoring="r2", n_jobs=1
        )
    except Exception:
        return

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, train_mean, "o-", color="#4A90D9", lw=1.5, label="Train R²")
    ax.fill_between(ts, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#4A90D9")
    ax.plot(ts, val_mean, "s-", color="#E15252", lw=1.5, label="CV Val R²")
    ax.fill_between(ts, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#E15252")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Training examples", fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title(f"Learning Curve - ANN-Exp1 - {name}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_ANN_run_{run}_lc.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Training ───────────────────────────────────────────────────────────────────

def train_dataset(ds: dict, run: int) -> dict:
    name   = ds["name"]
    target = ds["target"]
    print(f"  [{name}] Loading data...")

    df = pd.read_excel(ds["file"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feat_cols = infer_features(df, target)
    df_clean  = df[["Date"] + feat_cols + [target]].dropna()

    train = df_clean[df_clean["Date"].dt.year < 2025]
    test  = df_clean[df_clean["Date"].dt.year == 2025]

    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    print(f"    n_train={len(train)}, n_test={len(test)}, n_features={len(feat_cols)}")

    # ── Build pipeline & tune ────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(**MLP_BASE)),
    ])
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        pipeline, MLP_PARAM_GRID,
        cv=tscv, scoring="neg_root_mean_squared_error",
        n_jobs=-1, refit=True, error_score="raise"
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_params = grid.best_params_

    # ── Metrics ──────────────────────────────────────────────────────────────
    y_train_pred = best.predict(X_train)
    y_test_pred  = best.predict(X_test)

    r2_train   = r2_score(y_train, y_train_pred)
    r2_test    = r2_score(y_test,  y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
    mae_train  = mean_absolute_error(y_train, y_train_pred)
    mae_test   = mean_absolute_error(y_test,  y_test_pred)
    mape_test  = _mape(y_test, y_test_pred)
    r2_gap     = r2_train - r2_test

    print(f"    Train R²={r2_train:+.3f}  Test R²={r2_test:+.3f}  Gap={r2_gap:+.3f}")
    print(f"    Best params: {best_params}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"{name}_ann_run_{run}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best, f)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _scatter_plot(y_train, y_train_pred, y_test, y_test_pred,
                  r2_train, r2_test, name, run)
    _learning_curve_plot(best, X_train, y_train, name, run)

    df_full = df[["Date"] + feat_cols + [target]].dropna().copy()
    y_pred_full = best.predict(df_full[feat_cols].values)
    _timeseries_plot(df_full, target, y_pred_full, name, run)

    return {
        "experiment":  ds["experiment"],
        "run":         run,
        "model":       "ANN",
        "target":      target,
        "n_train":     len(train),
        "n_test":      len(test),
        "n_features":  len(feat_cols),
        "R2_train":    r2_train,
        "RMSE_train":  rmse_train,
        "MAE_train":   mae_train,
        "R2_test":     r2_test,
        "RMSE_test":   rmse_test,
        "MAE_test":    mae_test,
        "MAPE_test":   mape_test,
        "R2_gap":      r2_gap,
        "CV_RMSE":     -grid.best_score_,
        "best_params": str(best_params),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run = _next_run(RESULTS_FILE)
    print(f"=== ANN - Experiment 1 Datasets - Run {run} ===")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    records = []
    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        try:
            rec = train_dataset(ds, run)
            records.append(rec)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if records:
        results = pd.DataFrame(records)
        if os.path.exists(RESULTS_FILE):
            old = pd.read_excel(RESULTS_FILE)
            results = pd.concat([old, results], ignore_index=True)
        results.to_excel(RESULTS_FILE, index=False)
        print(f"\nResults saved → {RESULTS_FILE}")

        print("\n=== Summary ===")
        for rec in records:
            print(f"  {rec['target']:45s}  "
                  f"Test R²={rec['R2_test']:+.3f}  Gap={rec['R2_gap']:+.3f}")
        avg_r2 = np.mean([r["R2_test"] for r in records])
        print(f"\n  Avg Test R²: {avg_r2:+.3f}")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
