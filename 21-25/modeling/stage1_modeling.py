"""
stage1_modeling.py — Stage 1: Inlet features → Effluent targets

Trains a Random Forest and Linear Regression baseline for each of 8 models:
  - 4 grab models    (BOD, COD, TSS, pH)  using data from 2020–2025
  - 4 composite models (BOD, COD, TSS, pH) using data from 2022–2025

Train split : 2020–2023
Test split  : 2024
Holdout     : 2025 (untouched)

Outputs
-------
  data/   — one Excel file per model with features, target, and predictions
  models/ — saved RF models (.pkl)
  plots/  — scatter and time-series plots per model per run
  results.xlsx — summary of RMSE and R² for every model and run
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(os.path.dirname(BASE_DIR), "All_Years_Merged.xlsx")
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")
RESULTS_FILE = os.path.join(BASE_DIR, "results.xlsx")

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
TEST_YEAR   = 2025
RF_PARAMS   = dict(n_estimators=200, random_state=42, n_jobs=-1)

# ── Feature sets ───────────────────────────────────────────────────────────────
GRAB_FEATURES = [
    "Inlet pH (Grab)",
    "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)",
    "Inlet TSS (mg/L, Grab)",
    "Flow (MLD)",
    "Power Total (KW)",
    "month",
    "day_of_week",
    "year",
]

COMP_FEATURES = [
    "Inlet pH (Composite)",
    "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)",
    "Inlet TSS (mg/L, Composite)",
    "Flow (MLD)",
    "Power Total (KW)",
    "month",
    "day_of_week",
    "year",
]

# (model_name, features, target_column, first_year_with_data)
MODELS = [
    ("stage1_grab_BOD", GRAB_FEATURES, "Effluent BOD (mg/L, Grab)",      2020),
    ("stage1_grab_COD", GRAB_FEATURES, "Effluent COD (mg/L, Grab)",      2020),
    ("stage1_grab_TSS", GRAB_FEATURES, "Effluent TSS (mg/L, Grab)",      2020),
    ("stage1_grab_pH",  GRAB_FEATURES, "Effluent pH (Grab)",             2021),
    ("stage1_comp_BOD", COMP_FEATURES, "Effluent BOD (mg/L, Composite)", 2022),
    ("stage1_comp_COD", COMP_FEATURES, "Effluent COD (mg/L, Composite)", 2022),
    ("stage1_comp_TSS", COMP_FEATURES, "Effluent TSS (mg/L, Composite)", 2022),
    ("stage1_comp_pH",  COMP_FEATURES, "Effluent pH (Composite)",        2022),
]

# ── Colour palette for year-coded scatter plots ────────────────────────────────
YEAR_COLOURS = {
    2020: "#6BAED6",
    2021: "#2171B5",
    2022: "#74C476",
    2023: "#238B45",
    2024: "#FD8D3C",
    2025: "#D94801",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_base_data() -> pd.DataFrame:
    """Load All_Years_Merged.xlsx and add temporal feature columns."""
    df = pd.read_excel(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"]       = df["Date"].dt.year
    df["month"]      = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek   # 0 = Monday
    return df


def prepare_subset(df: pd.DataFrame, features: list, target: str,
                   min_year: int) -> pd.DataFrame:
    """
    Filter by year range, keep only relevant columns, and drop rows that
    have any NaN in the feature columns or the target column.
    """
    cols_needed = ["Date", "year", "month", "day_of_week"] + features + [target]
    # de-duplicate (year/month/day_of_week already in features list)
    cols_needed = list(dict.fromkeys(cols_needed))

    sub = df[df["year"] >= min_year][cols_needed].copy()
    before = len(sub)
    sub = sub.dropna(subset=features + [target])
    after  = len(sub)
    print(f"    Rows after dropna: {after}  (dropped {before - after})")
    return sub.reset_index(drop=True)


def get_run_number(subset_path: str) -> int:
    """Return the next run number based on existing prediction columns."""
    if not os.path.exists(subset_path):
        return 1
    existing = pd.read_excel(subset_path, nrows=0).columns.tolist()
    pred_cols = [c for c in existing if c.startswith("predicted_RF_run_")]
    return len(pred_cols) + 1


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ── Core training function ─────────────────────────────────────────────────────

def train_model(name: str, df_sub: pd.DataFrame, features: list,
                target: str) -> dict:
    """
    Train RF + LR on TRAIN_YEARS, evaluate on TEST_YEAR.
    Returns a results dict and the fitted RF model.
    """
    train = df_sub[df_sub["year"].isin(TRAIN_YEARS)]
    test  = df_sub[df_sub["year"] == TEST_YEAR]

    if len(test) == 0:
        print(f"    WARNING: no test rows for {TEST_YEAR} — skipping {name}")
        return None, None, None

    X_train, y_train = train[features].values, train[target].values
    X_test,  y_test  = test[features].values,  test[target].values

    print(f"    Train: {len(train)} rows | Test: {len(test)} rows")

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_train_pred = rf.predict(X_train)
    rf_test_pred  = rf.predict(X_test)

    rf_metrics = {
        "RF_RMSE_train": rmse(y_train, rf_train_pred),
        "RF_R2_train":   r2_score(y_train, rf_train_pred),
        "RF_RMSE_test":  rmse(y_test,  rf_test_pred),
        "RF_R2_test":    r2_score(y_test,  rf_test_pred),
    }

    # ── Linear Regression baseline ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_train_pred = lr.predict(X_train_sc)
    lr_test_pred  = lr.predict(X_test_sc)

    lr_metrics = {
        "LR_RMSE_train": rmse(y_train, lr_train_pred),
        "LR_R2_train":   r2_score(y_train, lr_train_pred),
        "LR_RMSE_test":  rmse(y_test,  lr_test_pred),
        "LR_R2_test":    r2_score(y_test,  lr_test_pred),
    }

    results = {"model": name, **rf_metrics, **lr_metrics}
    return results, rf, (test, rf_test_pred, lr_test_pred, y_test,
                         train, rf_train_pred, lr_train_pred)


# ── Saving helpers ─────────────────────────────────────────────────────────────

def save_subset_with_predictions(name: str, df_sub: pd.DataFrame, features: list,
                                  target: str, rf: object, run: int):
    """
    Save (or update) the subset Excel file with RF and LR prediction columns.
    Predictions are generated for ALL rows in the subset (train + test + 2025).
    """
    subset_path = os.path.join(DATA_DIR, f"{name}.xlsx")

    # Generate predictions for all rows
    X_all = df_sub[features].values
    rf_all_preds = rf.predict(X_all)

    df_out = df_sub.copy()
    df_out[f"predicted_RF_run_{run}"] = rf_all_preds.round(3)

    if os.path.exists(subset_path):
        # Load existing file and append new prediction column
        df_existing = pd.read_excel(subset_path)
        df_existing[f"predicted_RF_run_{run}"] = rf_all_preds.round(3)
        df_existing.to_excel(subset_path, index=False)
    else:
        df_out.to_excel(subset_path, index=False)

    print(f"    Saved subset → {subset_path}")


def save_model(name: str, rf, run: int):
    path = os.path.join(MODELS_DIR, f"{name}_run_{run}.pkl")
    joblib.dump(rf, path)
    print(f"    Saved model  → {path}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_scatter(name: str, test_df: pd.DataFrame, target: str,
                 rf_preds: np.ndarray, run: int):
    """Actual vs predicted scatter, points coloured by year."""
    fig, ax = plt.subplots(figsize=(7, 6))

    years = test_df["year"].values
    for yr in sorted(set(years)):
        mask = years == yr
        ax.scatter(test_df[target].values[mask], rf_preds[mask],
                   color=YEAR_COLOURS.get(yr, "#999999"), label=str(yr),
                   alpha=0.7, s=30, edgecolors="none")

    # Diagonal reference line
    lo = min(test_df[target].min(), rf_preds.min())
    hi = max(test_df[target].max(), rf_preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")

    ax.set_xlabel(f"Actual {target}", fontsize=11)
    ax.set_ylabel(f"Predicted {target}", fontsize=11)
    ax.set_title(f"{name}  |  Scatter (test set, run {run})", fontsize=12)
    ax.legend(title="Year", fontsize=9)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved plot   → {path}")


def plot_timeseries(name: str, df_sub: pd.DataFrame, target: str,
                    rf: object, features: list, run: int):
    """
    Time-series overlay of actual vs RF predicted for the full subset
    (train shown in lighter colour, test highlighted).
    """
    df_plot = df_sub.sort_values("Date").copy()
    df_plot["rf_pred"] = rf.predict(df_plot[features].values)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade the test period
    test_start = df_plot[df_plot["year"] == TEST_YEAR]["Date"].min()
    test_end   = df_plot[df_plot["year"] == TEST_YEAR]["Date"].max()
    if pd.notna(test_start):
        ax.axvspan(test_start, test_end, alpha=0.12, color="orange",
                   label=f"Test ({TEST_YEAR})")

    ax.plot(df_plot["Date"], df_plot[target],
            color="#2171B5", linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(df_plot["Date"], df_plot["rf_pred"],
            color="#D94801", linewidth=0.8, label="RF Predicted", alpha=0.9)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(target, fontsize=11)
    ax.set_title(f"{name}  |  Time Series (run {run})", fontsize=12)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved plot   → {path}")


def plot_feature_importance(name: str, rf, features: list, run: int):
    """Horizontal bar chart of RF feature importances."""
    importances = rf.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([features[i] for i in order], importances[order],
            color="#2171B5", edgecolor="white")
    ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=10)
    ax.set_title(f"{name}  |  Feature Importances (run {run})", fontsize=12)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{name}_run_{run}_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved plot   → {path}")


# ── Results summary ────────────────────────────────────────────────────────────

def save_results(all_results: list, run: int):
    """Append results for this run to results.xlsx."""
    df_new = pd.DataFrame(all_results)
    df_new.insert(1, "run", run)

    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_excel(RESULTS_FILE)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out = df_out.round(4)
    df_out.to_excel(RESULTS_FILE, index=False)
    print(f"\nResults saved → {RESULTS_FILE}")
    print(df_new[["model", "RF_RMSE_test", "RF_R2_test",
                  "LR_RMSE_test", "LR_R2_test"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_base_data()

    all_results = []
    run = None  # determined from the first model's existing file

    for name, features, target, min_year in MODELS:
        print(f"\n{'─'*60}")
        print(f"Model: {name}")
        print(f"  Target : {target}")
        print(f"  Years  : {min_year}–2025 (train ≤{max(TRAIN_YEARS)}, test {TEST_YEAR})")

        # Determine run number from this model's subset file
        subset_path = os.path.join(DATA_DIR, f"{name}.xlsx")
        model_run   = get_run_number(subset_path)
        if run is None:
            run = model_run   # all models in a session share the same run number

        df_sub = prepare_subset(df, features, target, min_year)

        result, rf, eval_data = train_model(name, df_sub, features, target)
        if result is None:
            continue

        test_df, rf_test_preds, lr_test_preds, y_test, _, _, _ = eval_data

        print(f"    RF  — RMSE: {result['RF_RMSE_test']:.3f}  R²: {result['RF_R2_test']:.3f}")
        print(f"    LR  — RMSE: {result['LR_RMSE_test']:.3f}  R²: {result['LR_R2_test']:.3f}")

        save_model(name, rf, run)
        save_subset_with_predictions(name, df_sub, features, target, rf, run)
        plot_scatter(name, test_df, target, rf_test_preds, run)
        plot_timeseries(name, df_sub, target, rf, features, run)
        plot_feature_importance(name, rf, features, run)

        all_results.append(result)

    if all_results:
        save_results(all_results, run)
    print("\nDone.")


if __name__ == "__main__":
    main()
