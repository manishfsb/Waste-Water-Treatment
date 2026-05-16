"""
make_sub6_datasets.py - Experiment 2 SE2a: All Secondary only (10 features, no COMMON).

Isolates whether secondary treatment features carry predictive signal WITHOUT
COMMON operational features (Flow, Power, calendar). Contrasts with sub_exp1
(All Secondary + COMMON, 17 features, exp_key Exp2-Sub1) to attribute signal
between secondary measurements and plant-level operational context.

Motivated by Exp1 finding that COMMON sometimes hurts grab targets and helps
composite targets - this experiment separates those contributions for SE2.

Feature set: SECONDARY (10) = Sec Clarifier (5) + Sec Sedimentation (5).
Both grab and comp targets use the same 10 features.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp6/make_sub6_datasets.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

SECONDARY = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]

DATASETS = [
    ("grab_BOD", "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  "Effluent pH (Grab)"),
    ("comp_BOD", "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    df["year"] = df["Date"].dt.year

    for name, target in DATASETS:
        cols   = ["Date", "year"] + SECONDARY + [target]
        subset = df[cols].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  Saved {name}.xlsx  -  {len(SECONDARY)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
