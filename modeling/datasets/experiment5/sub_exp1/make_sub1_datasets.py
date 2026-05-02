"""
make_sub1_datasets.py - Experiment 5 Sub-experiment 1.

Hypothesis: adding Grab inlet measurements (pH, BOD, COD, TSS) to the
composite-target dataset improves composite effluent prediction.

Base feature set: Exp3-S1 ADD-tier (Inlet + Secondary + COMMON + ADD aeration, 28 features)
Added:            GRAB_INLET (4 features) to the 4 composite-target datasets
Total:            32 features per composite dataset

Row-cost analysis (row_cost_analysis.py):
  Exp3-S1 composite baseline: ~813-816 train rows
  After adding GRAB_INLET:    ~629-631 train rows
  Marginal cost:              ~22.6% (OK, below 25% threshold)

Only composite targets are built here (4 datasets). Grab targets are in Sub2.

Cyclic calendar encoding is the standard from Exp2 onwards.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment5/sub_exp1/make_sub1_datasets.py
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

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
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]
ADD_FEATURES = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
]

# Sub1: Composite targets with GRAB_INLET added
DATASETS = [
    ("comp_BOD", COMP_INLET + GRAB_INLET, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", COMP_INLET + GRAB_INLET, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", COMP_INLET + GRAB_INLET, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  COMP_INLET + GRAB_INLET, "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    for name, inlet_cols, target in DATASETS:
        feature_cols = inlet_cols + SEC_COLS + COMMON_BASE + CYCLIC + ADD_FEATURES
        cols         = ["Date"] + feature_cols + [target]
        subset       = df[cols].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
