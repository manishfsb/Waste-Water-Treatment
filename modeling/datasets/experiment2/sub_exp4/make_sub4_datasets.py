"""
make_sub4_datasets.py - Experiment 2 SE3: Inlet + Primary + Secondary + COMMON (27 features).

Combines all three upstream monitoring points - core inlet (4), primary clarifier + grit (6),
and secondary clarifier + sedimentation (10) - with COMMON (7). This is the full stage-based
feature set for Experiment 2, with model-specific feature selection applied in the modeling scripts.

Feature set (grab targets): GRAB_INLET (4) + PRIMARY (6) + SECONDARY (10) + COMMON (7) = 27
Feature set (comp targets): COMP_INLET (4) + PRIMARY (6) + SECONDARY (10) + COMMON (7) = 27

Row counts (~764 grab train / ~595 comp train):
  Secondary co-occurs almost perfectly with Primary measurement days (~14-row additional
  cost vs Exp2-SE1 which had ~778 grab train). Binding constraint remains Primary/Grit.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp4/make_sub4_datasets.py
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
PRIMARY = [
    "Grit Classifier TSS (mg/L)",
    "Primary Clarifier pH",
    "Primary TSS (mg/L)",
    "Primary BOD (mg/L)",
    "Primary COD (mg/L)",
    "Primary Sludge Totalizer (m3)",
]
SECONDARY = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

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
        feature_cols = inlet_cols + PRIMARY + SECONDARY + COMMON_BASE + CYCLIC
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
