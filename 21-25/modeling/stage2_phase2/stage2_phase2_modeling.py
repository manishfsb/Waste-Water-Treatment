"""
stage2_phase2_modeling.py — Stage 2 Phase 2: Inlet + Secondary clarifier → Effluent targets

Augments inlet features with secondary clarifier + secondary sedimentation measurements.
Grab models use grab inlet features; composite models use composite inlet features.
Both are then combined with the same secondary clarifier feature set.

Feature sets
------------
  Grab models    : Inlet grab (pH, BOD, COD, TSS)
                 + Sec Clarifier (pH, TSS, BOD, COD, RAS)
                 + Sec Sed (pH, TSS, BOD, COD, RAS New)
                 + Flow, Power Total, month, day_of_week, year

  Composite models: Inlet composite (pH, BOD, COD, TSS)
                 + Sec Clarifier (pH, TSS, BOD, COD, RAS)
                 + Sec Sed (pH, TSS, BOD, COD, RAS New)
                 + Flow, Power Total, month, day_of_week, year

Train : 2021–2024  (2020 excluded — no secondary data)
Test  : 2025

Outputs
-------
  stage2_phase2/data/    — subset Excel files with predictions
  stage2_phase2/models/  — saved RF models (.pkl)
  stage2_phase2/plots/   — scatter, time-series, feature importance plots
  stage2_phase2/results.xlsx
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_FILE    = os.path.join(os.path.dirname(BASE_DIR), "All_Years_Merged.xlsx")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
RF_PARAMS   = dict(n_estimators=200, random_state=42, n_jobs=-1)  # kept for reference

RF_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [4, 6, 8, 10, 12, None],
    "min_samples_leaf":  [2, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", 0.5, 0.7],
}
RF_TUNE_ITER = 40

YEAR_COLOURS = {
    2021: "#2171B5", 2022: "#74C476",
    2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801",
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
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)", "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

GRAB_FEATURES = GRAB_INLET + SEC_COLS + COMMON
COMP_FEATURES = COMP_INLET + SEC_COLS + COMMON

# (model_name, features, target_column, min_year)
MODELS = [
    ("stage2_p2_grab_BOD", GRAB_FEATURES, "Effluent BOD (mg/L, Grab)",      2021),
    ("stage2_p2_grab_COD", GRAB_FEATURES, "Effluent COD (mg/L, Grab)",      2021),
    ("stage2_p2_grab_TSS", GRAB_FEATURES, "Effluent TSS (mg/L, Grab)",      2021),
    ("stage2_p2_grab_pH",  GRAB_FEATURES, "Effluent pH (Grab)",             2021),
    ("stage2_p2_comp_BOD", COMP_FEATURES, "Effluent BOD (mg/L, Composite)", 2022),
    ("stage2_p2_comp_COD", COMP_FEATURES, "Effluent COD (mg/L, Composite)", 2022),
    ("stage2_p2_comp_TSS", COMP_FEATURES, "Effluent TSS (mg/L, Composite)", 2022),
    ("stage2_p2_comp_pH",  COMP_FEATURES, "Effluent pH (Composite)",        2022),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_base_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    df["Date"]        = pd.to_datetime(df["Date"])
    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    return df


def prepare_subset(df: pd.DataFrame, features: list, target: str,
                   min_year: int) -> pd.DataFrame:
    cols   = list(dict.fromkeys(["Date", "year", "month", "day_of_week"]
                                + features + [target]))
    sub    = df[df["year"] >= min_year][cols].copy()
    before = len(sub)
    sub    = sub.dropna(subset=features + [target])
    print(f"    Rows after dropna: {len(sub)}  (dropped {before - len(sub)})")
    return sub.reset_index(drop=True)


def get_run_number(path: str) -> int:
    if not os.path.exists(path):
        return 1
    cols = pd.read_excel(path, nrows=0).columns.tolist()
    return sum(1 for c in cols if c.startswith("predicted_RF_run_")) + 1


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """RandomizedSearchCV with TimeSeriesSplit to find best RF hyperparameters."""
    tscv   = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        RF_PARAM_DIST,
        n_iter=RF_TUNE_ITER,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"    Best params : {search.best_params_}")
    print(f"    Best CV RMSE: {-search.best_score_:.3f}")
    return search.best_estimator_, float(-search.best_score_)


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(name: str, df_sub: pd.DataFrame,
                features: list, target: str):
    train = df_sub[df_sub["year"].isin(TRAIN_YEARS)].sort_values("Date")
    test  = df_sub[df_sub["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"    WARNING: no test rows for {TEST_YEAR} — skipping")
        return None, None, None

    X_train, y_train = train[features].values, train[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    print(f"    Train: {len(train)} rows | Test: {len(test)} rows")

    # Random Forest (tuned)
    rf, rf_cv_rmse = tune_rf(X_train, y_train)
    rf_train_pred = rf.predict(X_train)
    rf_test_pred  = rf.predict(X_test)

    scaler = StandardScaler()
    lr     = LinearRegression()
    lr.fit(scaler.fit_transform(X_train), y_train)
    lr_train_pred = lr.predict(scaler.transform(X_train))
    lr_test_pred  = lr.predict(scaler.transform(X_test))

    results = {
        "model":                name,
        "RF_RMSE_train":        rmse(y_train, rf_train_pred),
        "RF_MAE_train":         mae(y_train,  rf_train_pred),
        "RF_R2_train":          r2_score(y_train, rf_train_pred),
        "RF_CV_RMSE":           rf_cv_rmse,
        "RF_RMSE_test":         rmse(y_test,  rf_test_pred),
        "RF_MAE_test":          mae(y_test,   rf_test_pred),
        "RF_MAPE_test":         mape(y_test,  rf_test_pred),
        "RF_R2_test":           r2_score(y_test,  rf_test_pred),
        "RF_n_estimators":      rf.n_estimators,
        "RF_max_depth":         rf.max_depth if rf.max_depth is not None else "None",
        "RF_min_samples_leaf":  rf.min_samples_leaf,
        "RF_min_samples_split": rf.min_samples_split,
        "RF_max_features":      rf.max_features,
        "LR_RMSE_train":        rmse(y_train, lr_train_pred),
        "LR_R2_train":          r2_score(y_train, lr_train_pred),
        "LR_RMSE_test":         rmse(y_test,  lr_test_pred),
        "LR_R2_test":           r2_score(y_test,  lr_test_pred),
    }
    return results, rf, (test, rf_test_pred, lr_test_pred)


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_model(name: str, rf, run: int):
    path = os.path.join(MODELS_DIR, f"{name}_run_{run}.pkl")
    joblib.dump(rf, path)
    print(f"    Saved model  → {path}")


def save_subset(name: str, df_sub: pd.DataFrame, features: list,
                target: str, rf, run: int):
    path     = os.path.join(DATA_DIR, f"{name}.xlsx")
    rf_preds = rf.predict(df_sub[features].values).round(3)
    if os.path.exists(path):
        df_existing = pd.read_excel(path)
        df_existing[f"predicted_RF_run_{run}"] = rf_preds
        df_existing.to_excel(path, index=False)
    else:
        df_out = df_sub.copy()
        df_out[f"predicted_RF_run_{run}"] = rf_preds
        df_out.to_excel(path, index=False)
    print(f"    Saved subset → {path}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_scatter(name: str, test_df: pd.DataFrame, target: str,
                 rf_preds: np.ndarray, run: int):
    fig, ax = plt.subplots(figsize=(7, 6))
    for yr in sorted(test_df["year"].unique()):
        mask = test_df["year"].values == yr
        ax.scatter(test_df[target].values[mask], rf_preds[mask],
                   color=YEAR_COLOURS.get(yr, "#999"), label=str(yr),
                   alpha=0.7, s=30, edgecolors="none")
    lo = min(test_df[target].min(), rf_preds.min())
    hi = max(test_df[target].max(), rf_preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel(f"Actual {target}", fontsize=11)
    ax.set_ylabel(f"Predicted {target}", fontsize=11)
    ax.set_title(f"{name} | Scatter (test {TEST_YEAR}, run {run})", fontsize=11)
    ax.legend(title="Year", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Saved plot   → {path}")


def plot_timeseries(name: str, df_sub: pd.DataFrame, features: list,
                    target: str, rf, run: int):
    df_plot = df_sub.sort_values("Date").copy()
    df_plot["rf_pred"] = rf.predict(df_plot[features].values)
    test_rows = df_plot[df_plot["year"] == TEST_YEAR]
    fig, ax = plt.subplots(figsize=(14, 4))
    if len(test_rows):
        ax.axvspan(test_rows["Date"].min(), test_rows["Date"].max(),
                   alpha=0.12, color="orange", label=f"Test ({TEST_YEAR})")
    ax.plot(df_plot["Date"], df_plot[target],
            color="#2171B5", linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(df_plot["Date"], df_plot["rf_pred"],
            color="#D94801", linewidth=0.8, label="RF Predicted", alpha=0.9)
    ax.set_xlabel("Date"); ax.set_ylabel(target)
    ax.set_title(f"{name} | Time Series (run {run})", fontsize=11)
    ax.legend(fontsize=9); fig.autofmt_xdate(); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_timeseries.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Saved plot   → {path}")


def plot_importance(name: str, rf, features: list, run: int):
    imps  = rf.feature_importances_
    order = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh([features[i] for i in order], imps[order],
            color="#2E4057", edgecolor="white")
    ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=10)
    ax.set_title(f"{name} | Feature Importances (run {run})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_importance.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Saved plot   → {path}")


# ── Results ────────────────────────────────────────────────────────────────────

def save_results(all_results: list, run: int):
    df_new = pd.DataFrame(all_results)
    df_new.insert(1, "run", run)
    if os.path.exists(RESULTS_FILE):
        df_out = pd.concat([pd.read_excel(RESULTS_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.round(4).to_excel(RESULTS_FILE, index=False)
    print(f"\nResults saved → {RESULTS_FILE}")
    print(df_new[["model", "RF_RMSE_test", "RF_R2_test",
                  "LR_RMSE_test", "LR_R2_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_base_data()

    all_results, run = [], None

    for name, features, target, min_year in MODELS:
        print(f"\n{'─'*60}")
        print(f"Model  : {name}")
        print(f"Target : {target}")
        print(f"Years  : {min_year}–2025  (train ≤{max(TRAIN_YEARS)}, test {TEST_YEAR})")

        subset_path = os.path.join(DATA_DIR, f"{name}.xlsx")
        model_run   = get_run_number(subset_path)
        if run is None:
            run = model_run

        df_sub = prepare_subset(df, features, target, min_year)
        result, rf, eval_data = train_model(name, df_sub, features, target)
        if result is None:
            continue

        test_df, rf_test_preds, _ = eval_data
        print(f"    RF  — RMSE: {result['RF_RMSE_test']:.3f}  R²: {result['RF_R2_test']:.3f}")
        print(f"    LR  — RMSE: {result['LR_RMSE_test']:.3f}  R²: {result['LR_R2_test']:.3f}")

        save_model(name, rf, run)
        save_subset(name, df_sub, features, target, rf, run)
        plot_scatter(name, test_df, target, rf_test_preds, run)
        plot_timeseries(name, df_sub, features, target, rf, run)
        plot_importance(name, rf, features, run)
        all_results.append(result)

    if all_results:
        save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
