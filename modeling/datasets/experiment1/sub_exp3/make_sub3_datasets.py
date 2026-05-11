"""
make_sub3_datasets.py - Experiment 1 Sub-experiment 3: All Grab Inlet + COMMON_CYCLIC.

Extends SE2-Cyclic by adding the five supplementary grab inlet measurements:
  - Inlet TKN/NH3-N (mg/L, Grab)
  - Inlet O&G (mg/L, Grab)
  - Inlet PO4/TP (mg/L, Grab)
  - Inlet Total Coliform (CFU/100ml, Grab)
  - Inlet Fecal Coliform (CFU/100ml, Grab)

Only grab effluent targets are produced (4 files). Composite inlet features have no
extended equivalents, so composite targets are unchanged from SE2-Cyclic and are omitted
here to avoid redundant datasets.

Feature set (9 inlet + 7 COMMON_CYCLIC = 16 features per dataset):
  GRAB_INLET (4):     Inlet pH, BOD, COD, TSS (Grab)
  EXTRA_INLET (5):    Inlet TKN/NH3-N, O&G, PO4/TP, Total Coliform, Fecal Coliform (Grab)
  COMMON (3):         Flow (MLD), Power Total (KW), year
  Cyclic calendar (4): month_sin, month_cos, dow_sin, dow_cos

Row cost (train, vs SE2-Cyclic baseline of 1175):
  All 9 inlet cols present: ~393 rows (~33.5% of train)

Note: Inlet O&G and TKN/NH3-N have ~65-68% missingness individually; their inclusion
drives the joint row count down. This SE is intentionally the "full inlet signal" test.

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
GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
EXTRA_INLET = [
    "Inlet TKN/NH3-N (mg/L, Grab)",
    "Inlet O&G (mg/L, Grab)",
    "Inlet PO4/TP (mg/L, Grab)",
    "Inlet Total Coliform (CFU/100ml, Grab)",
    "Inlet Fecal Coliform (CFU/100ml, Grab)",
]
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

# -- Targets (grab only - composite has no extended inlet equivalents) ----------
DATASETS = [
    ("grab_BOD", "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  "Effluent pH (Grab)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    feature_cols = GRAB_INLET + EXTRA_INLET + COMMON_BASE + CYCLIC

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
