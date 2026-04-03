"""
elasticnet_modeling.py — ElasticNet regression across all stages and phases.

Reads existing subset Excel files produced by the stage modeling scripts.
Tunes alpha + l1_ratio via GridSearchCV + TimeSeriesSplit. Results saved to
elasticnet/results.xlsx. Predictions appended as 'predicted_ElNet_run_N'.

Covers:
  Stage 1       — Inlet features     (grab + composite)
  Stage 2 P1    — Secondary only     (grab + composite targets)
  Stage 2 P2    — Inlet + Secondary  (grab + composite)

Usage (from project root):
    .venv/bin/python3 21-25/modeling/elasticnet/elasticnet_modeling.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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

ELNET_PARAM_GRID = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

# ── Feature sets (must match stage modeling scripts exactly) ──────────────────
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
    ("Stage 1", "stage1_grab_BOD", _p("", "stage1_grab_BOD"), S1_GRAB,  "Effluent BOD (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_COD", _p("", "stage1_grab_COD"), S1_GRAB,  "Effluent COD (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_TSS", _p("", "stage1_grab_TSS"), S1_GRAB,  "Effluent TSS (mg/L, Grab)"),
    ("Stage 1", "stage1_grab_pH",  _p("", "stage1_grab_pH"),  S1_GRAB,  "Effluent pH (Grab)"),
    # Stage 1 — composite
    ("Stage 1", "stage1_comp_BOD", _p("", "stage1_comp_BOD"), S1_COMP,  "Effluent BOD (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_COD", _p("", "stage1_comp_COD"), S1_COMP,  "Effluent COD (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_TSS", _p("", "stage1_comp_TSS"), S1_COMP,  "Effluent TSS (mg/L, Composite)"),
    ("Stage 1", "stage1_comp_pH",  _p("", "stage1_comp_pH"),  S1_COMP,  "Effluent pH (Composite)"),
    # Stage 2 P1 — grab
    ("Stage 2 P1", "stage2_p1_grab_BOD", _p("stage2_phase1", "stage2_p1_grab_BOD"), S2P1, "Effluent BOD (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_COD", _p("stage2_phase1", "stage2_p1_grab_COD"), S2P1, "Effluent COD (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_TSS", _p("stage2_phase1", "stage2_p1_grab_TSS"), S2P1, "Effluent TSS (mg/L, Grab)"),
    ("Stage 2 P1", "stage2_p1_grab_pH",  _p("stage2_phase1", "stage2_p1_grab_pH"),  S2P1, "Effluent pH (Grab)"),
    # Stage 2 P1 — composite
    ("Stage 2 P1", "stage2_p1_comp_BOD", _p("stage2_phase1", "stage2_p1_comp_BOD"), S2P1, "Effluent BOD (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_COD", _p("stage2_phase1", "stage2_p1_comp_COD"), S2P1, "Effluent COD (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_TSS", _p("stage2_phase1", "stage2_p1_comp_TSS"), S2P1, "Effluent TSS (mg/L, Composite)"),
    ("Stage 2 P1", "stage2_p1_comp_pH",  _p("stage2_phase1", "stage2_p1_comp_pH"),  S2P1, "Effluent pH (Composite)"),
    # Stage 2 P2 — grab
    ("Stage 2 P2", "stage2_p2_grab_BOD", _p("stage2_phase2", "stage2_p2_grab_BOD"), S2P2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_COD", _p("stage2_phase2", "stage2_p2_grab_COD"), S2P2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_TSS", _p("stage2_phase2", "stage2_p2_grab_TSS"), S2P2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Stage 2 P2", "stage2_p2_grab_pH",  _p("stage2_phase2", "stage2_p2_grab_pH"),  S2P2_GRAB, "Effluent pH (Grab)"),
    # Stage 2 P2 — composite
    ("Stage 2 P2", "stage2_p2_comp_BOD", _p("stage2_phase2", "stage2_p2_comp_BOD"), S2P2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_COD", _p("stage2_phase2", "stage2_p2_comp_COD"), S2P2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_TSS", _p("stage2_phase2", "stage2_p2_comp_TSS"), S2P2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Stage 2 P2", "stage2_p2_comp_pH",  _p("stage2_phase2", "stage2_p2_comp_pH"),  S2P2_COMP, "Effluent pH (Composite)"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def get_run_number(subset_path: str) -> int:
    if not os.path.exists(subset_path):
        return 1
    cols = pd.read_excel(subset_path, nrows=0).columns.tolist()
    return sum(1 for c in cols if c.startswith("predicted_ElNet_run_")) + 1


def append_prediction(subset_path: str, predictions: np.ndarray, run: int):
    df = pd.read_excel(subset_path)
    df[f"predicted_ElNet_run_{run}"] = predictions.round(3)
    df.to_excel(subset_path, index=False)


# ── Tuning ─────────────────────────────────────────────────────────────────────

def tune_elasticnet(X_train: np.ndarray, y_train: np.ndarray):
    """GridSearchCV over alpha + l1_ratio with TimeSeriesSplit."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    tscv   = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(
        ElasticNet(max_iter=10000),
        ELNET_PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_scaled, y_train)
    cv_rmse    = float(-search.best_score_)
    best_alpha = search.best_params_["alpha"]
    best_l1    = search.best_params_["l1_ratio"]
    print(f"    Best alpha={best_alpha}, l1_ratio={best_l1}")
    print(f"    CV RMSE   : {cv_rmse:.3f}")
    return scaler, search.best_estimator_, cv_rmse, best_alpha, best_l1


# ── Core ───────────────────────────────────────────────────────────────────────

def train_model(stage, name, subset_path, features, target):
    df = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)]
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    test = df[df["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"    WARNING: no test rows for {TEST_YEAR} — skipping")
        return None, None, None

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"    WARNING: missing features {missing} — skipping")
        return None, None, None

    X_train = train[features].values
    y_train = train[target].values
    X_test  = test[features].values
    y_test  = test[target].values

    print(f"    Train: {len(train)} | Test: {len(test)}")

    scaler, elnet, cv_rmse, best_alpha, best_l1 = tune_elasticnet(X_train, y_train)
    train_pred = elnet.predict(scaler.transform(X_train))
    test_pred  = elnet.predict(scaler.transform(X_test))

    results = {
        "stage":            stage,
        "model":            name,
        "ElNet_RMSE_train": rmse(y_train, train_pred),
        "ElNet_MAE_train":  mae(y_train,  train_pred),
        "ElNet_R2_train":   r2_score(y_train, train_pred),
        "ElNet_CV_RMSE":    cv_rmse,
        "ElNet_RMSE_test":  rmse(y_test,  test_pred),
        "ElNet_MAE_test":   mae(y_test,   test_pred),
        "ElNet_MAPE_test":  mape(y_test,  test_pred),
        "ElNet_R2_test":    r2_score(y_test, test_pred),
        "ElNet_alpha":      best_alpha,
        "ElNet_l1_ratio":   best_l1,
    }

    all_preds = elnet.predict(scaler.transform(df[features].values))
    return results, (scaler, elnet), all_preds


def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    df_new.insert(2, "run", run)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\nResults → {RESULTS_FILE}")
    print(df_new[["stage", "model", "ElNet_RMSE_test", "ElNet_R2_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run = get_run_number(MODELS[0][2])
    print(f"Starting ElasticNet run {run} across {len(MODELS)} models...\n")

    all_results = []

    for stage, name, subset_path, features, target in MODELS:
        print(f"{'─'*60}")
        print(f"[{stage}]  {name}")
        print(f"  Target: {target}")

        if not os.path.exists(subset_path):
            print(f"  WARNING: subset file not found — {subset_path}")
            continue

        result, model_pair, all_preds = train_model(stage, name, subset_path, features, target)
        if result is None:
            continue

        scaler, elnet = model_pair
        print(f"    ElNet — RMSE: {result['ElNet_RMSE_test']:.3f}  "
              f"R²: {result['ElNet_R2_test']:.3f}")

        model_path = os.path.join(MODELS_DIR, f"{name}_ElNet_run_{run}.pkl")
        joblib.dump({"scaler": scaler, "model": elnet}, model_path)
        print(f"    Saved model → {model_path}")

        append_prediction(subset_path, all_preds, run)
        print(f"    Updated subset → {subset_path}")

        all_results.append(result)

    if all_results:
        save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
