"""
xgb_modeling.py — XGBoost run across all stages and phases.

Trains XGBRegressor on the same data subsets used for RF and GB,
using identical train/test splits. Results saved to xgb/results.xlsx.
XGB predictions appended as 'predicted_XGB_run_N' to existing subset Excel files.

Covers:
  Stage 1       — Inlet features     (grab + composite)
  Stage 2 P1    — Secondary only     (grab + composite targets)
  Stage 2 P2    — Inlet + Secondary  (grab + composite)

Usage (from project root):
    .venv/bin/python3 21-25/modeling/xgb/xgb_modeling.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
XGB_PARAMS  = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# ── Feature sets (must match the modeling scripts exactly) ─────────────────────
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
def _p(stage_dir, name):
    return os.path.join(MODELING_DIR, stage_dir, "data", f"{name}.xlsx")

MODELS = [
    # Stage 1 — grab
    ("Stage 1", "stage1_grab_BOD", _p("","stage1_grab_BOD"), S1_GRAB,
     "Effluent BOD (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_COD", _p("","stage1_grab_COD"), S1_GRAB,
     "Effluent COD (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_TSS", _p("","stage1_grab_TSS"), S1_GRAB,
     "Effluent TSS (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_pH",  _p("","stage1_grab_pH"),  S1_GRAB,
     "Effluent pH (Grab)"),
    # Stage 1 — composite
    ("Stage 1", "stage1_comp_BOD", _p("","stage1_comp_BOD"), S1_COMP,
     "Effluent BOD (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_COD", _p("","stage1_comp_COD"), S1_COMP,
     "Effluent COD (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_TSS", _p("","stage1_comp_TSS"), S1_COMP,
     "Effluent TSS (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_pH",  _p("","stage1_comp_pH"),  S1_COMP,
     "Effluent pH (Composite)"),
    # Stage 2 P1 — grab
    ("Stage 2 P1", "stage2_p1_grab_BOD", _p("stage2_phase1","stage2_p1_grab_BOD"),
     S2P1, "Effluent BOD (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_COD", _p("stage2_phase1","stage2_p1_grab_COD"),
     S2P1, "Effluent COD (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_TSS", _p("stage2_phase1","stage2_p1_grab_TSS"),
     S2P1, "Effluent TSS (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_pH",  _p("stage2_phase1","stage2_p1_grab_pH"),
     S2P1, "Effluent pH (Grab)"),
    # Stage 2 P1 — composite
    ("Stage 2 P1", "stage2_p1_comp_BOD", _p("stage2_phase1","stage2_p1_comp_BOD"),
     S2P1, "Effluent BOD (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_COD", _p("stage2_phase1","stage2_p1_comp_COD"),
     S2P1, "Effluent COD (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_TSS", _p("stage2_phase1","stage2_p1_comp_TSS"),
     S2P1, "Effluent TSS (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_pH",  _p("stage2_phase1","stage2_p1_comp_pH"),
     S2P1, "Effluent pH (Composite)"),
    # Stage 2 P2 — grab
    ("Stage 2 P2", "stage2_p2_grab_BOD", _p("stage2_phase2","stage2_p2_grab_BOD"),
     S2P2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_COD", _p("stage2_phase2","stage2_p2_grab_COD"),
     S2P2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_TSS", _p("stage2_phase2","stage2_p2_grab_TSS"),
     S2P2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_pH",  _p("stage2_phase2","stage2_p2_grab_pH"),
     S2P2_GRAB, "Effluent pH (Grab)"),
    # Stage 2 P2 — composite
    ("Stage 2 P2", "stage2_p2_comp_BOD", _p("stage2_phase2","stage2_p2_comp_BOD"),
     S2P2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_COD", _p("stage2_phase2","stage2_p2_comp_COD"),
     S2P2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_TSS", _p("stage2_phase2","stage2_p2_comp_TSS"),
     S2P2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_pH",  _p("stage2_phase2","stage2_p2_comp_pH"),
     S2P2_COMP, "Effluent pH (Composite)"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_xgb_run_number(subset_path: str) -> int:
    """Check existing XGB prediction columns to determine next run number."""
    if not os.path.exists(subset_path):
        return 1
    cols = pd.read_excel(subset_path, nrows=0).columns.tolist()
    return sum(1 for c in cols if c.startswith("predicted_XGB_run_")) + 1


def append_prediction(subset_path: str, predictions: np.ndarray, run: int):
    """Append XGB predictions as a new column to the existing subset Excel."""
    df = pd.read_excel(subset_path)
    df[f"predicted_XGB_run_{run}"] = predictions.round(3)
    df.to_excel(subset_path, index=False)


# ── Core ───────────────────────────────────────────────────────────────────────

def train_model(stage, name, subset_path, features, target):
    df = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)]
    # Also include 2020 rows in training if present (Stage 1 grab models)
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    test = df[df["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"    WARNING: no test rows for {TEST_YEAR} — skipping")
        return None, None

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"    WARNING: missing features {missing_feats} — skipping")
        return None, None

    X_train = train[features].values
    y_train = train[target].values
    X_test  = test[features].values
    y_test  = test[target].values

    print(f"    Train: {len(train)} | Test: {len(test)}")

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_train, y_train)

    results = {
        "stage":           stage,
        "model":           name,
        "XGB_RMSE_train":  rmse(y_train, xgb.predict(X_train)),
        "XGB_R2_train":    r2_score(y_train, xgb.predict(X_train)),
        "XGB_RMSE_test":   rmse(y_test,  xgb.predict(X_test)),
        "XGB_R2_test":     r2_score(y_test,  xgb.predict(X_test)),
    }
    return results, xgb


def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    df_new.insert(2, "run", run)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\nResults → {RESULTS_FILE}")
    print(df_new[["stage", "model", "XGB_RMSE_test", "XGB_R2_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run = get_xgb_run_number(MODELS[0][2])
    print(f"Starting XGB run {run} across {len(MODELS)} models...\n")

    all_results = []

    for stage, name, subset_path, features, target in MODELS:
        print(f"{'─'*60}")
        print(f"[{stage}]  {name}")
        print(f"  Target: {target}")

        if not os.path.exists(subset_path):
            print(f"  WARNING: subset file not found — {subset_path}")
            continue

        result, xgb = train_model(stage, name, subset_path, features, target)
        if result is None:
            continue

        print(f"    XGB — RMSE: {result['XGB_RMSE_test']:.3f}  "
              f"R²: {result['XGB_R2_test']:.3f}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}_XGB_run_{run}.pkl")
        joblib.dump(xgb, model_path)
        print(f"    Saved model → {model_path}")

        # Append predictions to existing subset file (all rows)
        df_all = pd.read_excel(subset_path)
        all_preds = xgb.predict(df_all[features].values)
        append_prediction(subset_path, all_preds, run)
        print(f"    Updated subset → {subset_path}")

        all_results.append(result)

    if all_results:
        save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
