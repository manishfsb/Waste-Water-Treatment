"""
make_sub2_datasets.py - Experiment 10 SE2: Same-day features + Lag-5 BOD/COD.

Extends SE1 by adding the 6 BOD and COD columns that were removed, shifted
back by 5 days. On any given day X, the most recent BOD/COD lab result
available is from day X-5 (collected 5 days ago; BOD5 result returned today).
COD is treated as a 5-day lab result pending confirmation.

SE1 same-day features (22):
  Inlet pH/TSS + Sec pH/TSS/RAS (x2) + ADD-tier Aeration + Flow + Power + COMMON_CYCLIC

SE2 adds lag-5 versions of the 6 removed BOD/COD columns (6 new features):
  Grab datasets:      Inlet BOD (Grab) lag-5, Inlet COD (Grab) lag-5,
                      Sec Clarifier BOD lag-5, Sec Clarifier COD lag-5,
                      Sec Sed BOD lag-5, Sec Sed COD lag-5
  Composite datasets: Inlet BOD (Composite) lag-5, Inlet COD (Composite) lag-5,
                      Sec Clarifier BOD lag-5, Sec Clarifier COD lag-5,
                      Sec Sed BOD lag-5, Sec Sed COD lag-5

Total: 28 features per dataset.

Lag implementation: data is sorted by Date; .shift(5) on the raw column
produces the value from 5 calendar rows earlier (not 5 calendar days, since
the dataset has daily rows including weekends). For continuous daily data this
is equivalent to 5-day lag.

Exp key: Exp10-SE2

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment10/sub_exp2/make_sub2_datasets.py
"""

import os
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# -- Same-day features (identical to SE1) -------------------------------------
GRAB_INLET_SD = [
    "Inlet pH (Grab)",
    "Inlet TSS (mg/L, Grab)",
]
COMP_INLET_SD = [
    "Inlet pH (Composite)",
    "Inlet TSS (mg/L, Composite)",
]
SEC_COLS_SD = [
    "Sec Clarifier pH",
    "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier RAS",
    "Sec Sed pH",
    "Sec Sed TSS (mg/L)",
    "Sec Sed RAS (New)",
]
ADD_FEATURES = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

# -- Columns to lag by 5 days (grab and composite variants) -------------------
GRAB_LAG_COLS = [
    "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)",
    "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)",
    "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)",
]
COMP_LAG_COLS = [
    "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)",
    "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)",
    "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)",
]

# -- Target definitions --------------------------------------------------------
DATASETS = [
    ("grab_BOD", "Grab",      GRAB_INLET_SD, GRAB_LAG_COLS, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", "Grab",      GRAB_INLET_SD, GRAB_LAG_COLS, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", "Grab",      GRAB_INLET_SD, GRAB_LAG_COLS, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  "Grab",      GRAB_INLET_SD, GRAB_LAG_COLS, "Effluent pH (Grab)"),
    ("comp_BOD", "Composite", COMP_INLET_SD, COMP_LAG_COLS, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", "Composite", COMP_INLET_SD, COMP_LAG_COLS, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", "Composite", COMP_INLET_SD, COMP_LAG_COLS, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  "Composite", COMP_INLET_SD, COMP_LAG_COLS, "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Calendar features
    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    # Build lag-5 columns for all relevant raw columns (deduped)
    all_lag_sources = list(dict.fromkeys(GRAB_LAG_COLS + COMP_LAG_COLS))
    lag_col_map = {}  # original_col -> lag_col_name
    for col in all_lag_sources:
        lag_name = col + " (lag5)"
        df[lag_name] = df[col].shift(5)
        lag_col_map[col] = lag_name

    print(f"  Created {len(lag_col_map)} lag-5 columns.")

    for name, kind, inlet_cols, lag_source_cols, target in DATASETS:
        lag_cols     = [lag_col_map[c] for c in lag_source_cols]
        feature_cols = inlet_cols + SEC_COLS_SD + COMMON_BASE + CYCLIC + ADD_FEATURES + lag_cols
        cols         = ["Date"] + feature_cols + [target]

        subset   = df[cols].dropna().copy()
        train_n  = int((subset["year"] < 2025).sum())
        test_n   = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
