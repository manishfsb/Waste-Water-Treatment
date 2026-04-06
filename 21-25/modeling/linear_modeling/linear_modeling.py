"""
linear_modeling.py — OLS, Ridge, and ElasticNet across Experiments 1 & 2.

Reads existing subset Excel files produced by the stage modeling scripts.
Trains three models per dataset:
  - OLS  (LinearRegression, no regularisation)
  - Ridge (L2-regularised, alpha tuned via GridSearchCV + TimeSeriesSplit)
  - ElasticNet (L1+L2, alpha & l1_ratio tuned)

Results saved to linear_modeling/results.xlsx.
Predictions appended to each subset file as:
  predicted_OLS_run_N  /  predicted_Ridge_run_N  /  predicted_ElNet_run_N

Experiments:
  Experiment 1        — Inlet features       (grab + composite)
  Experiment 2 Sub-1  — Secondary only       (grab + composite)
  Experiment 2 Sub-2  — Inlet + Secondary    (grab + composite)

Usage (from project root):
    .venv/bin/python3 21-25/modeling/linear_modeling/linear_modeling.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

ELNET_PARAM_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
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
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
    "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

EXP1_GRAB  = GRAB_INLET + COMMON          # 9 features
EXP1_COMP  = COMP_INLET + COMMON          # 9 features
EXP2S1     = SEC_COLS   + COMMON          # 15 features
EXP2S2_G   = GRAB_INLET + SEC_COLS + COMMON  # 19 features
EXP2S2_C   = COMP_INLET + SEC_COLS + COMMON  # 19 features

# ── Dataset registry ───────────────────────────────────────────────────────────
def _s1(name):
    """Experiment 1 files: modeling/stage1/<name>.xlsx  (no data/ subdir)"""
    return os.path.join(MODELING_DIR, "stage1", f"{name}.xlsx")

def _s2a(name):
    """Experiment 2 Sub-1: modeling/stage2_phase1/data/<name>.xlsx"""
    return os.path.join(MODELING_DIR, "stage2_phase1", "data", f"{name}.xlsx")

def _s2b(name):
    """Experiment 2 Sub-2: modeling/stage2_phase2/data/<name>.xlsx"""
    return os.path.join(MODELING_DIR, "stage2_phase2", "data", f"{name}.xlsx")


# (experiment_label, dataset_id, file_path, features, target)
DATASETS = [
    # ── Experiment 1 — Grab ──────────────────────────────────────────────────
    ("Exp1", "Exp1_Grab_BOD", _s1("stage1_grab_BOD"), EXP1_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Exp1", "Exp1_Grab_COD", _s1("stage1_grab_COD"), EXP1_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Exp1", "Exp1_Grab_TSS", _s1("stage1_grab_TSS"), EXP1_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Exp1", "Exp1_Grab_pH",  _s1("stage1_grab_pH"),  EXP1_GRAB, "Effluent pH (Grab)"),
    # ── Experiment 1 — Composite ─────────────────────────────────────────────
    ("Exp1", "Exp1_Comp_BOD", _s1("stage1_comp_BOD"), EXP1_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Exp1", "Exp1_Comp_COD", _s1("stage1_comp_COD"), EXP1_COMP, "Effluent COD (mg/L, Composite)"),
    ("Exp1", "Exp1_Comp_TSS", _s1("stage1_comp_TSS"), EXP1_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Exp1", "Exp1_Comp_pH",  _s1("stage1_comp_pH"),  EXP1_COMP, "Effluent pH (Composite)"),
    # ── Experiment 2 Sub-1 — Grab ─────────────────────────────────────────────
    ("Exp2-Sub1", "Exp2S1_Grab_BOD", _s2a("stage2_p1_grab_BOD"), EXP2S1, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub1", "Exp2S1_Grab_COD", _s2a("stage2_p1_grab_COD"), EXP2S1, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub1", "Exp2S1_Grab_TSS", _s2a("stage2_p1_grab_TSS"), EXP2S1, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub1", "Exp2S1_Grab_pH",  _s2a("stage2_p1_grab_pH"),  EXP2S1, "Effluent pH (Grab)"),
    # ── Experiment 2 Sub-1 — Composite ───────────────────────────────────────
    ("Exp2-Sub1", "Exp2S1_Comp_BOD", _s2a("stage2_p1_comp_BOD"), EXP2S1, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub1", "Exp2S1_Comp_COD", _s2a("stage2_p1_comp_COD"), EXP2S1, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub1", "Exp2S1_Comp_TSS", _s2a("stage2_p1_comp_TSS"), EXP2S1, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub1", "Exp2S1_Comp_pH",  _s2a("stage2_p1_comp_pH"),  EXP2S1, "Effluent pH (Composite)"),
    # ── Experiment 2 Sub-2 — Grab ─────────────────────────────────────────────
    ("Exp2-Sub2", "Exp2S2_Grab_BOD", _s2b("stage2_p2_grab_BOD"), EXP2S2_G, "Effluent BOD (mg/L, Grab)"),
    ("Exp2-Sub2", "Exp2S2_Grab_COD", _s2b("stage2_p2_grab_COD"), EXP2S2_G, "Effluent COD (mg/L, Grab)"),
    ("Exp2-Sub2", "Exp2S2_Grab_TSS", _s2b("stage2_p2_grab_TSS"), EXP2S2_G, "Effluent TSS (mg/L, Grab)"),
    ("Exp2-Sub2", "Exp2S2_Grab_pH",  _s2b("stage2_p2_grab_pH"),  EXP2S2_G, "Effluent pH (Grab)"),
    # ── Experiment 2 Sub-2 — Composite ───────────────────────────────────────
    ("Exp2-Sub2", "Exp2S2_Comp_BOD", _s2b("stage2_p2_comp_BOD"), EXP2S2_C, "Effluent BOD (mg/L, Composite)"),
    ("Exp2-Sub2", "Exp2S2_Comp_COD", _s2b("stage2_p2_comp_COD"), EXP2S2_C, "Effluent COD (mg/L, Composite)"),
    ("Exp2-Sub2", "Exp2S2_Comp_TSS", _s2b("stage2_p2_comp_TSS"), EXP2S2_C, "Effluent TSS (mg/L, Composite)"),
    ("Exp2-Sub2", "Exp2S2_Comp_pH",  _s2b("stage2_p2_comp_pH"),  EXP2S2_C, "Effluent pH (Composite)"),
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_R2":   float(r2_score(y_true, y_pred)),
        f"{prefix}_RMSE": _rmse(y_true, y_pred),
        f"{prefix}_MAE":  _mae(y_true,  y_pred),
        f"{prefix}_MAPE": _mape(y_true, y_pred),
    }


# ── Run-number detection ───────────────────────────────────────────────────────

def get_run_number(subset_path: str) -> int:
    """Return the next unified run number based on existing OLS prediction columns."""
    if not os.path.exists(subset_path):
        return 1
    cols = pd.read_excel(subset_path, nrows=0).columns.tolist()
    return sum(1 for c in cols if c.startswith("predicted_OLS_run_")) + 1


# ── Append predictions to subset file ─────────────────────────────────────────

def append_predictions(subset_path: str, pred_dict: dict):
    """Add one column per model to the subset Excel file."""
    df = pd.read_excel(subset_path)
    for col, values in pred_dict.items():
        df[col] = values
    df.to_excel(subset_path, index=False)


# ── Core training ──────────────────────────────────────────────────────────────

def train_dataset(experiment, ds_id, path, features, target, run):
    """Train OLS, Ridge, ElasticNet on one dataset. Returns (results_dict, preds_dict)."""

    df = pd.read_excel(path, parse_dates=["Date"])

    # Build train / test splits (include 2020 rows if present)
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra    = df[df["year"] == 2020]
    if len(extra) > 0:
        train_df = pd.concat([extra, train_df]).drop_duplicates()

    test_df = df[df["year"] == TEST_YEAR]

    if len(test_df) == 0:
        print(f"  WARNING: no {TEST_YEAR} rows — skipping {ds_id}")
        return None, None

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing} — skipping {ds_id}")
        return None, None

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test  = test_df[features].values
    y_test  = test_df[target].values
    X_all   = df[features].values

    print(f"  Train: {len(train_df):>4d} rows | Test: {len(test_df):>3d} rows")

    # Shared scaler (fitted on train)
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_te_sc   = scaler.transform(X_test)
    X_all_sc  = scaler.transform(X_all)

    tscv = TimeSeriesSplit(n_splits=3)

    results = {
        "experiment":  experiment,
        "dataset":     ds_id,
        "target":      target,
        "run":         run,
        "n_train":     len(train_df),
        "n_test":      len(test_df),
        "n_features":  len(features),
    }
    preds = {}

    # ── OLS ───────────────────────────────────────────────────────────────────
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_train)
    tr_ols = ols.predict(X_tr_sc)
    te_ols = ols.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ols, "OLS_train"))
    results.update(_metrics(y_test,  te_ols, "OLS_test"))
    results["OLS_R2_gap"] = results["OLS_train_R2"] - results["OLS_test_R2"]
    col_ols = f"predicted_OLS_run_{run}"
    preds[col_ols] = np.round(ols.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ols},
                os.path.join(MODELS_DIR, f"{ds_id}_OLS_run_{run}.pkl"))
    print(f"    OLS    — Train R²: {results['OLS_train_R2']:+.3f} | "
          f"Test R²: {results['OLS_test_R2']:+.3f} | "
          f"RMSE: {results['OLS_test_RMSE']:.3f}")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    ridge_gs = GridSearchCV(
        Ridge(), {"alpha": RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    ridge_gs.fit(X_tr_sc, y_train)
    ridge = ridge_gs.best_estimator_
    tr_ridge = ridge.predict(X_tr_sc)
    te_ridge = ridge.predict(X_te_sc)
    results.update(_metrics(y_train, tr_ridge, "Ridge_train"))
    results.update(_metrics(y_test,  te_ridge, "Ridge_test"))
    results["Ridge_R2_gap"]  = results["Ridge_train_R2"] - results["Ridge_test_R2"]
    results["Ridge_CV_RMSE"] = float(-ridge_gs.best_score_)
    results["Ridge_alpha"]   = ridge_gs.best_params_["alpha"]
    col_ridge = f"predicted_Ridge_run_{run}"
    preds[col_ridge] = np.round(ridge.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": ridge},
                os.path.join(MODELS_DIR, f"{ds_id}_Ridge_run_{run}.pkl"))
    print(f"    Ridge  — Train R²: {results['Ridge_train_R2']:+.3f} | "
          f"Test R²: {results['Ridge_test_R2']:+.3f} | "
          f"α={results['Ridge_alpha']}")

    # ── ElasticNet ────────────────────────────────────────────────────────────
    elnet_gs = GridSearchCV(
        ElasticNet(max_iter=10000), ELNET_PARAM_GRID,
        scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, refit=True,
    )
    elnet_gs.fit(X_tr_sc, y_train)
    elnet = elnet_gs.best_estimator_
    tr_elnet = elnet.predict(X_tr_sc)
    te_elnet = elnet.predict(X_te_sc)
    results.update(_metrics(y_train, tr_elnet, "ElNet_train"))
    results.update(_metrics(y_test,  te_elnet, "ElNet_test"))
    results["ElNet_R2_gap"]  = results["ElNet_train_R2"] - results["ElNet_test_R2"]
    results["ElNet_CV_RMSE"] = float(-elnet_gs.best_score_)
    results["ElNet_alpha"]   = elnet_gs.best_params_["alpha"]
    results["ElNet_l1_ratio"] = elnet_gs.best_params_["l1_ratio"]
    col_elnet = f"predicted_ElNet_run_{run}"
    preds[col_elnet] = np.round(elnet.predict(X_all_sc), 3)
    joblib.dump({"scaler": scaler, "model": elnet},
                os.path.join(MODELS_DIR, f"{ds_id}_ElNet_run_{run}.pkl"))
    print(f"    ElNet  — Train R²: {results['ElNet_train_R2']:+.3f} | "
          f"Test R²: {results['ElNet_test_R2']:+.3f} | "
          f"α={results['ElNet_alpha']}, l1={results['ElNet_l1_ratio']}")

    # ── Per-dataset comparison plot ───────────────────────────────────────────
    _plot_comparison(
        ds_id, run,
        y_test, te_ols, te_ridge, te_elnet,
        test_df["Date"].values if "Date" in test_df.columns else None,
    )

    return results, preds


# ── Plotting ───────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "OLS":   "#E15252",   # warm red
    "Ridge": "#4A90D9",   # blue
    "ElNet": "#5BAD6F",   # green
}


def _plot_comparison(ds_id, run, y_test, te_ols, te_ridge, te_elnet, dates):
    """Actual-vs-predicted scatter for all three models (test set)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"{ds_id}  |  Test Set Actual vs Predicted  (run {run})",
                 fontsize=13, fontweight="bold")

    pairs = [
        ("OLS",   te_ols,   MODEL_COLORS["OLS"]),
        ("Ridge", te_ridge, MODEL_COLORS["Ridge"]),
        ("ElNet", te_elnet, MODEL_COLORS["ElNet"]),
    ]
    for ax, (label, preds, color) in zip(axes, pairs):
        ax.scatter(y_test, preds, color=color, alpha=0.65, s=25, edgecolors="none")
        lo = min(y_test.min(), preds.min())
        hi = max(y_test.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1, label="Perfect")
        r2 = r2_score(y_test, preds)
        rmse_val = _rmse(y_test, preds)
        ax.set_title(f"{label}\nR²={r2:+.3f}  RMSE={rmse_val:.3f}", fontsize=11)
        ax.set_xlabel("Actual", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Predicted", fontsize=10)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_id}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot → {path}")


def _plot_r2_barchart(df_results: pd.DataFrame, run: int):
    """
    Grouped bar chart of Test R² per experiment, one group per dataset.
    Saved per experiment to plots/.
    """
    for exp in df_results["experiment"].unique():
        sub = df_results[df_results["experiment"] == exp].copy()
        labels = [d.split("_", 1)[1] if "_" in d else d for d in sub["dataset"]]
        x = np.arange(len(sub))
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(sub) * 1.2), 5))
        ax.bar(x - w, sub["OLS_test_R2"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
        ax.bar(x,     sub["Ridge_test_R2"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
        ax.bar(x + w, sub["ElNet_test_R2"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Test R²", fontsize=11)
        ax.set_title(f"{exp}  |  Test R² Comparison (run {run})", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=min(-0.15, sub[["OLS_test_R2","Ridge_test_R2","ElNet_test_R2"]].min().min() - 0.05))
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{exp}_r2_comparison_run_{run}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  R² bar chart → {path}")


def _plot_rmse_barchart(df_results: pd.DataFrame, run: int):
    """Grouped bar chart of Test RMSE per experiment."""
    for exp in df_results["experiment"].unique():
        sub = df_results[df_results["experiment"] == exp].copy()
        labels = [d.split("_", 1)[1] if "_" in d else d for d in sub["dataset"]]
        x = np.arange(len(sub))
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(sub) * 1.2), 5))
        ax.bar(x - w, sub["OLS_test_RMSE"],   w, label="OLS",   color=MODEL_COLORS["OLS"])
        ax.bar(x,     sub["Ridge_test_RMSE"],  w, label="Ridge", color=MODEL_COLORS["Ridge"])
        ax.bar(x + w, sub["ElNet_test_RMSE"],  w, label="ElNet", color=MODEL_COLORS["ElNet"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Test RMSE", fontsize=11)
        ax.set_title(f"{exp}  |  Test RMSE Comparison (run {run})", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{exp}_rmse_comparison_run_{run}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  RMSE bar chart → {path}")


# ── Results persistence ────────────────────────────────────────────────────────

def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"Results → {RESULTS_FILE}")
    key_cols = ["experiment", "dataset",
                "OLS_test_R2",   "Ridge_test_R2",   "ElNet_test_R2",
                "OLS_test_RMSE", "Ridge_test_RMSE",  "ElNet_test_RMSE"]
    print(df_new[key_cols].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Determine run number from the first dataset file
    first_path = DATASETS[0][2]
    run = get_run_number(first_path)
    print(f"Linear Modeling — Run {run}")
    print(f"Datasets: {len(DATASETS)}  |  Models: OLS, Ridge, ElasticNet\n")

    all_results = []

    for experiment, ds_id, path, features, target in DATASETS:
        print(f"{'─'*60}")
        print(f"[{experiment}]  {ds_id}")
        print(f"  Target   : {target}")
        print(f"  Features : {len(features)}")

        if not os.path.exists(path):
            print(f"  WARNING: file not found — {path}")
            continue

        results, preds = train_dataset(experiment, ds_id, path, features, target, run)
        if results is None:
            continue

        append_predictions(path, preds)
        print(f"  Predictions appended → {path}")
        all_results.append(results)

    if not all_results:
        print("No results produced — check warnings above.")
        return

    df_results = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("Generating summary plots...")
    _plot_r2_barchart(df_results, run)
    _plot_rmse_barchart(df_results, run)

    save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
