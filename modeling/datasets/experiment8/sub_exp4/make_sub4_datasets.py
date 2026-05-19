"""
make_sub4_datasets.py - Experiment 8 SE4: Full Exp3-SE2 features with availability-aware lags.

Takes the complete Exp3-SE2 feature set (31 features) and replaces the columns
that require lab processing with their appropriate lag versions:

  BOD columns (5-day BOD5 incubation) -> lag-5:
    Inlet BOD (Grab/Composite), Sec Clarifier BOD, Sec Sed BOD

  COD columns (treated as 5-day pending confirmation) -> lag-5:
    Inlet COD (Grab/Composite), Sec Clarifier COD, Sec Sed COD

  Total Coliform (24-hour culture plate) -> lag-1:
    Inlet Total Coliform (CFU/100ml, Grab)

All other features remain as same-day values:
  pH everywhere (probe, minutes), TSS (gravimetric, same-day),
  RAS (meter), all Aeration columns, Flow, Power, Primary Sludge Totalizer,
  calendar features.

This gives 31 features per dataset - identical count to Exp3-SE2 - but with
the 7 lab-delayed columns shifted to match their actual operational availability.

Contrast with SE1 (lags all inlet + Flow + Power indiscriminately) and SE3
(same-day-only base + lag-5 BOD/COD on reduced 22-feature set). SE4 is the
principled middle ground: full feature richness, lags only where justified.

Exp key: Exp8-SE4

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment8/sub_exp4/make_sub4_datasets.py
"""

import os
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# -- Same-day inlet features ---------------------------------------------------
GRAB_INLET_SD = [
    "Inlet pH (Grab)",
    "Inlet TSS (mg/L, Grab)",
]
COMP_INLET_SD = [
    "Inlet pH (Composite)",
    "Inlet TSS (mg/L, Composite)",
]

# -- Secondary features (pH, TSS, RAS: all same-day) --------------------------
SEC_COLS_SD = [
    "Sec Clarifier pH",
    "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier RAS",
    "Sec Sed pH",
    "Sec Sed TSS (mg/L)",
    "Sec Sed RAS (New)",
]

# -- Aeration and operational (all same-day) -----------------------------------
ADD_FEATURES = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
    "Aeration SVI (New)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year", "Primary Sludge Totalizer (m3)"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

# -- BOD/COD columns to lag by 5 days -----------------------------------------
# Shared secondary BOD/COD columns (same for grab and composite)
SEC_BOD_COD = [
    "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)",
    "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)",
]

GRAB_LAG5_COLS = [
    "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)",
] + SEC_BOD_COD

COMP_LAG5_COLS = [
    "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)",
] + SEC_BOD_COD

# -- Coliform column to lag by 1 day ------------------------------------------
# Grab coliform is only available 2021-2023; composite coliform covers 2023-2025.
# Using composite column for both grab and composite datasets (same precedent as
# Exp3-SE2 which uses grab coliform for both types).
COLIFORM_COL = "Inlet Total Coliform (CFU/100ml, Composite)"

# -- Target definitions --------------------------------------------------------
DATASETS = [
    ("grab_BOD", GRAB_INLET_SD, GRAB_LAG5_COLS, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", GRAB_INLET_SD, GRAB_LAG5_COLS, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", GRAB_INLET_SD, GRAB_LAG5_COLS, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  GRAB_INLET_SD, GRAB_LAG5_COLS, "Effluent pH (Grab)"),
    ("comp_BOD", COMP_INLET_SD, COMP_LAG5_COLS, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", COMP_INLET_SD, COMP_LAG5_COLS, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", COMP_INLET_SD, COMP_LAG5_COLS, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  COMP_INLET_SD, COMP_LAG5_COLS, "Effluent pH (Composite)"),
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

    # Build lag-5 columns for all BOD/COD sources (deduped across grab/comp)
    all_lag5_sources = list(dict.fromkeys(GRAB_LAG5_COLS + COMP_LAG5_COLS))
    lag5_map = {}
    for col in all_lag5_sources:
        lag_name = col + " (lag5)"
        df[lag_name] = df[col].shift(5)
        lag5_map[col] = lag_name
    print(f"  Created {len(lag5_map)} lag-5 columns.")

    # Build lag-1 column for Coliform
    coliform_lag1 = COLIFORM_COL + " (lag1)"
    df[coliform_lag1] = df[COLIFORM_COL].shift(1)
    print(f"  Created 1 lag-1 column: {coliform_lag1}")

    for name, inlet_cols, lag5_source_cols, target in DATASETS:
        lag5_cols    = [lag5_map[c] for c in lag5_source_cols]
        feature_cols = (inlet_cols + SEC_COLS_SD + COMMON_BASE + CYCLIC
                        + ADD_FEATURES + lag5_cols + [coliform_lag1])
        cols         = ["Date"] + feature_cols + [target]

        subset  = df[cols].dropna().copy()
        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
