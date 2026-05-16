"""
make_sub5_datasets.py - Experiment 2 SE1a: Primary + COMMON only (13 features).

Isolates whether primary treatment features carry any predictive signal WITHOUT
inlet features. Contrasts with sub_exp3 (Primary + Inlet + COMMON, 17 features,
exp_key Exp2-S3) to reveal whether inlet adds or destroys signal at the primary stage.

Feature set: PRIMARY (6) + COMMON (7) = 13 features
Both grab and comp targets use the same 13 features (PRIMARY has no grab/comp distinction).

Row counts: slightly higher than sub_exp3 since inlet NaN rows are no longer binding.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp5/make_sub5_datasets.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

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

    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    feature_cols = PRIMARY + COMMON_BASE + CYCLIC

    for name, target in DATASETS:
        cols   = ["Date"] + feature_cols + [target]
        subset = df[cols].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  Saved {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
