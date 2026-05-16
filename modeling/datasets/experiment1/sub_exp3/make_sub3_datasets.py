"""
make_sub3_datasets.py - Experiment 1 Sub-experiment 3: Extended Composite Inlet + COMMON_CYCLIC.

Produces 4 composite-target datasets only (grab datasets retired).

DESIGN CHANGE (2026-05): After correcting the measurement-type mislabelling in
extraction, the extended inlet features (TKN/NH3-N, O&G, PO4/TP, Coliform) were
found to be grab-measured only in 2021 (TKN/O&G) or 2021-2022 (PO4/Coliform).
From 2022 or 2023 onwards they are composite measurements. Correctly using only
grab-sourced values in grab datasets would leave at most 51 training rows -
effectively reducing grab SE3 to the same 11 features as SE2. Grab SE3 datasets
are therefore retired; SE3 is composite-only.

Feature set (composite targets):
  COMP_INLET (4):   Inlet pH, BOD, COD, TSS (Composite)
  EXTRA_COMP (2):   Inlet NH3-N (Composite), Inlet O&G (Composite)
                    -- PO4 and Coliform composite excluded (too sparse before 2024)
  COMMON_CYCLIC (7): Flow, Power, year, month_sin/cos, dow_sin/cos
  Total: 13 features | ~737 train rows (2022-2024)

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment1/sub_exp3/make_sub3_datasets.py
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
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]
# Composite-sourced extended features (consistently available 2022+)
EXTRA_COMP = [
    "Inlet NH3-N (mg/L, Composite)",
    "Inlet O&G (mg/L, Composite)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

COMP_DATASETS = [
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

    comp_feat = COMP_INLET + EXTRA_COMP + COMMON_BASE + CYCLIC
    print(f"\n-- Composite datasets ({len(comp_feat)} features: COMP_INLET + NH3-N + O&G + COMMON_CYCLIC) --")
    for name, target in COMP_DATASETS:
        cols   = ["Date"] + comp_feat + [target]
        subset = df[cols].dropna().copy()
        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())
        subset.to_excel(os.path.join(OUT_DIR, f"{name}.xlsx"), index=False)
        print(f"  {name}.xlsx  -  {len(comp_feat)} features  (train={train_n}, test={test_n})")

    print("\nNote: grab SE3 datasets retired - see module docstring for explanation.")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
