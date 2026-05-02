"""
make_sub2_datasets.py - Experiment 5 Sub-experiment 2.

Hypothesis: adding Composite inlet measurements (pH, BOD, COD, TSS) to the
grab-target dataset improves grab effluent prediction.

Base feature set: Exp3-S1 ADD-tier (Inlet + Secondary + COMMON + ADD aeration, 28 features)
Added:            COMP_INLET (4 features) to the 4 grab-target datasets
Total:            32 features per grab dataset

Row-cost analysis (row_cost_analysis.py):
  Exp3-S1 grab baseline:   ~1018-1021 train rows
  After adding COMP_INLET: ~630-631 train rows
  Marginal cost:           ~38.2% (HIGH COST - unavoidable, due to joint
                           availability of grab and composite measurements)
  Retained n (~630) is comparable to Exp2-Sub2 grab training size and
  adequate for all 6 model families.

Only grab targets are built here (4 datasets). Composite targets are in Sub1.

Cyclic calendar encoding is the standard from Exp2 onwards.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment5/sub_exp2/make_sub2_datasets.py
"""

import os
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

# Sub2: Grab targets with COMP_INLET added
DATASETS = [
    ("grab_BOD", GRAB_INLET + COMP_INLET, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", GRAB_INLET + COMP_INLET, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", GRAB_INLET + COMP_INLET, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  GRAB_INLET + COMP_INLET, "Effluent pH (Grab)"),
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
