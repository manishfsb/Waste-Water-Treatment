"""
make_sub3_datasets.py - Experiment 2 Sub-experiment 3: Primary Stage + GRAB_INLET + COMMON_CYCLIC.

Tests whether primary treatment measurements (TSS, BOD, COD, pH, Grit, Sludge Totalizer)
combined with inlet and calendar features can predict effluent quality.

Feature set (4 + 6 + 7 = 17 features for grab targets, 4 + 6 + 7 = 17 for composite):
  GRAB_INLET (4):     Inlet pH, BOD, COD, TSS (Grab)   [grab targets]
  COMP_INLET (4):     Inlet pH, BOD, COD, TSS (Composite) [composite targets]
  PRIMARY (6):        Grit Classifier TSS, Primary Clarifier pH,
                      Primary TSS, Primary BOD, Primary COD,
                      Primary Sludge Totalizer
  COMMON (3):         Flow (MLD), Power Total (KW), year
  Cyclic calendar (4): month_sin, month_cos, dow_sin, dow_cos

VIF notes (these are flagged; tree models unaffected):
  Primary Clarifier pH: VIF ~2743 (collinear with other pH cols) - included for completeness
  Primary BOD:          VIF ~17.8  (above >10 threshold for linear models)
  Primary COD:          VIF ~21.6  (above >10 threshold for linear models)
  Primary TSS:          VIF ~7.1   (clean)
  Grit Classifier TSS:  VIF ~7.6   (clean)
  Primary Sludge Totalizer: VIF ~7.4 (clean)

Row counts (train, vs Exp1-SE2-Cyclic baseline of 1175):
  Grab targets with all 6 primary cols + GRAB_INLET:  ~778 rows
  Comp targets with all 6 primary cols + COMP_INLET:  ~767 rows

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp3/make_sub3_datasets.py
"""

import os
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# -- Feature groups ------------------------------------------------------------
GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]
PRIMARY = [
    "Grit Classifier TSS (mg/L)",
    "Primary Clarifier pH",
    "Primary TSS (mg/L)",
    "Primary BOD (mg/L)",
    "Primary COD (mg/L)",
    "Primary Sludge Totalizer (m3)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

# -- Targets -------------------------------------------------------------------
DATASETS = [
    ("grab_BOD", GRAB_INLET, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", GRAB_INLET, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", GRAB_INLET, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  GRAB_INLET, "Effluent pH (Grab)"),
    ("comp_BOD", COMP_INLET, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", COMP_INLET, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", COMP_INLET, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  COMP_INLET, "Effluent pH (Composite)"),
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
        feature_cols = inlet_cols + PRIMARY + COMMON_BASE + CYCLIC
        cols         = ["Date"] + feature_cols + [target]
        subset       = df[cols].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  Saved {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
