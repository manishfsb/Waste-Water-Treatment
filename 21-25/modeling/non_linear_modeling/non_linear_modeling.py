"""
non_linear_modeling.py — Unified tree-ensemble training across all experiments.

Trains Random Forest, Gradient Boosting, and XGBoost with consistent fixed
hyperparameters across all three experiments and eight targets.

  Experiment 1          — Inlet features only          (grab + composite)
  Experiment 2 Sub-1    — Secondary features only       (grab + composite)
  Experiment 2 Sub-2    — Inlet + Secondary features    (grab + composite)

Train = 2021–2024  |  Test = 2025

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)

def _sub(stage_dir, name):
    """Resolve path to an existing subset Excel file."""
    if stage_dir == "stage1":
        return os.path.join(MODELING_DIR, "stage1", f"{name}.xlsx")
    return os.path.join(MODELING_DIR, stage_dir, "data", f"{name}.xlsx")

# ── Splits ─────────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

# ── Hyperparameters — fixed for all three models (fair comparison) ─────────────
RF_PARAMS = dict(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    max_features="sqrt", random_state=42, n_jobs=-1,
)
GB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=5, random_state=42,
)
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, verbosity=0,
)

# ── Feature sets ───────────────────────────────────────────────────────────────
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
              "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
              "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
              "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)"]
COMMON     = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

S1_GRAB   = GRAB_INLET + COMMON
S1_COMP   = COMP_INLET + COMMON
S2P1      = SEC_COLS   + COMMON
S2P2_GRAB = GRAB_INLET + SEC_COLS + COMMON
S2P2_COMP = COMP_INLET + SEC_COLS + COMMON

# ── Model registry ─────────────────────────────────────────────────────────────
# (experiment_label, model_name, subset_path, features, target)
REGISTRY = [
    # Experiment 1 — grab
    ("Experiment 1",       "stage1_grab_BOD", _sub("stage1","stage1_grab_BOD"),
     S1_GRAB,   "Effluent BOD (mg/L, Grab)"),
    ("Experiment 1",       "stage1_grab_COD", _sub("stage1","stage1_grab_COD"),
     S1_GRAB,   "Effluent COD (mg/L, Grab)"),
    ("Experiment 1",       "stage1_grab_TSS", _sub("stage1","stage1_grab_TSS"),
     S1_GRAB,   "Effluent TSS (mg/L, Grab)"),
    ("Experiment 1",       "stage1_grab_pH",  _sub("stage1","stage1_grab_pH"),
     S1_GRAB,   "Effluent pH (Grab)"),
    # Experiment 1 — composite
    ("Experiment 1",       "stage1_comp_BOD", _sub("stage1","stage1_comp_BOD"),
     S1_COMP,   "Effluent BOD (mg/L, Composite)"),
    ("Experiment 1",       "stage1_comp_COD", _sub("stage1","stage1_comp_COD"),
     S1_COMP,   "Effluent COD (mg/L, Composite)"),
    ("Experiment 1",       "stage1_comp_TSS", _sub("stage1","stage1_comp_TSS"),
     S1_COMP,   "Effluent TSS (mg/L, Composite)"),
    ("Experiment 1",       "stage1_comp_pH",  _sub("stage1","stage1_comp_pH"),
     S1_COMP,   "Effluent pH (Composite)"),
    # Experiment 2 Sub-1 — grab
    ("Experiment 2 Sub-1", "stage2_p1_grab_BOD",
     _sub("stage2_phase1","stage2_p1_grab_BOD"), S2P1, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "stage2_p1_grab_COD",
     _sub("stage2_phase1","stage2_p1_grab_COD"), S2P1, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "stage2_p1_grab_TSS",
     _sub("stage2_phase1","stage2_p1_grab_TSS"), S2P1, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "stage2_p1_grab_pH",
     _sub("stage2_phase1","stage2_p1_grab_pH"),  S2P1, "Effluent pH (Grab)"),
    # Experiment 2 Sub-1 — composite
    ("Experiment 2 Sub-1", "stage2_p1_comp_BOD",
     _sub("stage2_phase1","stage2_p1_comp_BOD"), S2P1, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "stage2_p1_comp_COD",
     _sub("stage2_phase1","stage2_p1_comp_COD"), S2P1, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "stage2_p1_comp_TSS",
     _sub("stage2_phase1","stage2_p1_comp_TSS"), S2P1, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "stage2_p1_comp_pH",
     _sub("stage2_phase1","stage2_p1_comp_pH"),  S2P1, "Effluent pH (Composite)"),
    # Experiment 2 Sub-2 — grab
    ("Experiment 2 Sub-2", "stage2_p2_grab_BOD",
     _sub("stage2_phase2","stage2_p2_grab_BOD"), S2P2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "stage2_p2_grab_COD",
     _sub("stage2_phase2","stage2_p2_grab_COD"), S2P2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "stage2_p2_grab_TSS",
     _sub("stage2_phase2","stage2_p2_grab_TSS"), S2P2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "stage2_p2_grab_pH",
     _sub("stage2_phase2","stage2_p2_grab_pH"),  S2P2_GRAB, "Effluent pH (Grab)"),
    # Experiment 2 Sub-2 — composite
    ("Experiment 2 Sub-2", "stage2_p2_comp_BOD",
     _sub("stage2_phase2","stage2_p2_comp_BOD"), S2P2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "stage2_p2_comp_COD",
     _sub("stage2_phase2","stage2_p2_comp_COD"), S2P2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "stage2_p2_comp_TSS",
     _sub("stage2_phase2","stage2_p2_comp_TSS"), S2P2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "stage2_p2_comp_pH",
     _sub("stage2_phase2","stage2_p2_comp_pH"),  S2P2_COMP, "Effluent pH (Composite)"),
]

# ── Year colours (consistent across project) ───────────────────────────────────
YEAR_COLOURS = {2021:"#2171B5", 2022:"#74C476", 2023:"#238B45",
                2024:"#FD8D3C", 2025:"#D94801"}

MODEL_COLOURS = {"RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801"}

# ── Metric helpers ─────────────────────────────────────────────────────────────

def _rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def _mae(yt, yp):  return float(mean_absolute_error(yt, yp))
def _r2(yt, yp):   return float(r2_score(yt, yp))

def _run_number(model_dir: str, prefix: str) -> int:
    """Next run number based on existing results.xlsx entries."""
    rfile = os.path.join(model_dir, "results.xlsx")
    if not os.path.exists(rfile):
        return 1
    df = pd.read_excel(rfile, nrows=0)
    run_col = [c for c in df.columns if c == "run"]
    if not run_col:
        return 1
    df_full = pd.read_excel(rfile)
    return int(df_full["run"].max()) + 1

# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_scatter(plots_dir, name, model_tag, run,
                  test_df, y_test, y_pred, target):
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
    ax.set_ylabel(f"Predicted", fontsize=10)
    ax.set_title(f"{model_tag} | {name}\nTest scatter (run {run})", fontsize=10)
    ax.legend(title="Year", fontsize=8)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _plot_timeseries(plots_dir, name, model_tag, run,
                     df_all, features, model, target):
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
              model_tag, estimator, run, model_dir):
    """Train one (model, dataset) combination. Return metrics dict + plot paths."""
    df = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)]
    test  = df[df["year"] == TEST_YEAR]
    if len(test) == 0:
        print(f"    SKIP — no test rows for {TEST_YEAR}")
        return None, None

    X_tr, y_tr = train[features].values, train[target].values
    X_te, y_te = test[features].values,  test[target].values

    print(f"    Train {len(train):4d}  Test {len(test):3d}")
    estimator.fit(X_tr, y_tr)

    tr_pred = estimator.predict(X_tr)
    te_pred = estimator.predict(X_te)

    row = {
        "experiment":  experiment,
        "model_name":  name,
        "run":         run,
        "model":       model_tag,
        "target":      target,
        "n_train":     len(train),
        "n_test":      len(test),
        "R2_train":    round(_r2(y_tr, tr_pred),  4),
        "RMSE_train":  round(_rmse(y_tr, tr_pred), 4),
        "MAE_train":   round(_mae(y_tr,  tr_pred), 4),
        "R2_test":     round(_r2(y_te, te_pred),   4),
        "RMSE_test":   round(_rmse(y_te, te_pred), 4),
        "MAE_test":    round(_mae(y_te,  te_pred), 4),
        "R2_gap":      round(_r2(y_tr, tr_pred) - _r2(y_te, te_pred), 4),
    }

    # Save model
    plots_dir  = os.path.join(model_dir, "plots")
    models_dir = os.path.join(model_dir, "models")
    pkl_path   = os.path.join(models_dir, f"{name}_{model_tag}_run_{run}.pkl")
    joblib.dump(estimator, pkl_path)

    # Plots
    plots = {
        "scatter":    _plot_scatter(plots_dir, name, model_tag, run,
                                    test, y_te, te_pred, target),
        "timeseries": _plot_timeseries(plots_dir, name, model_tag, run,
                                       df, features, estimator, target),
        "importance": _plot_importance(plots_dir, name, model_tag, run,
                                       estimator, features),
    }

    # Append prediction column to subset file
    df_sub = pd.read_excel(subset_path)
    col    = f"predicted_{model_tag}_NL_run_{run}"
    df_sub[col] = estimator.predict(df_sub[features].values).round(3)
    df_sub.to_excel(subset_path, index=False)

    return row, plots


def run_model(model_tag, estimator_cls, params):
    model_dir  = os.path.join(SCRIPT_DIR, model_tag.lower())
    results_fp = os.path.join(model_dir, "results.xlsx")
    run        = _run_number(model_dir, model_tag)

    print(f"\n{'='*65}")
    print(f"  {model_tag} — run {run}")
    print(f"{'='*65}")

    all_rows  = []
    all_plots = {}   # name → {scatter, timeseries, importance}

    for experiment, name, subset_path, features, target in REGISTRY:
        print(f"\n[{experiment}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP — file not found: {subset_path}"); continue

        estimator = estimator_cls(**params)
        row, plots = train_one(experiment, name, subset_path, features, target,
                               model_tag, estimator, run, model_dir)
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
        ("RF",  RandomForestRegressor,    RF_PARAMS),
        ("GB",  GradientBoostingRegressor, GB_PARAMS),
        ("XGB", XGBRegressor,              XGB_PARAMS),
    ]

    summary = {}
    for tag, cls, params in models:
        rows, plots, run = run_model(tag, cls, params)
        summary[tag] = {"rows": rows, "plots": plots, "run": run}

    print("\n" + "="*65)
    print("  All models complete.")
    print(f"  Results in: {SCRIPT_DIR}/{{rf,gb,xgb}}/results.xlsx")
    print("="*65)


if __name__ == "__main__":
    main()
