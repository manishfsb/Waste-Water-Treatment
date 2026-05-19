"""
make_sub1_datasets.py - Experiment 10 SE1: Same-day available features only.

Builds one xlsx per target using the Exp3-SE1 feature set (Inlet + Secondary +
ADD-tier Aeration + COMMON_CYCLIC) with all BOD and COD input columns removed.

Motivation: BOD requires 5-day lab incubation (BOD5); COD turnaround is also
treated as 5 days pending confirmation. Removing these columns leaves only
features that are genuinely available on the day of sampling:
  - Meters (Flow, Power) - real-time
  - pH everywhere - electrochemical probe, minutes
  - TSS - gravimetric (~3-4h) or sensor, same-day
  - Aeration DO, MLSS, SV30, SVI - same-day lab/physical tests
  - RAS flow ratios - meter-based
  - Calendar features - trivially known

Same-day grab features (2):  Inlet pH (Grab),  Inlet TSS (Grab)
Same-day comp features (2):  Inlet pH (Composite), Inlet TSS (Composite)
Same-day secondary (6):      Sec Clarifier pH/TSS/RAS + Sec Sed pH/TSS/RAS
ADD-tier aeration (7):       DO/MLSS/SV30/SVI/pH (Existing) + DO/SV30 (New)
COMMON_CYCLIC (7):           Flow, Power, year, month_sin/cos, dow_sin/cos
Total per dataset: 22 features.

Exp key: Exp10-SE1

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment10/sub_exp1/make_sub1_datasets.py
"""

import os
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# -- Same-day inlet features (BOD and COD removed) ----------------------------
GRAB_INLET_SD = [
    "Inlet pH (Grab)",
    "Inlet TSS (mg/L, Grab)",
]
COMP_INLET_SD = [
    "Inlet pH (Composite)",
    "Inlet TSS (mg/L, Composite)",
]

# -- Secondary features (BOD and COD removed) ---------------------------------
SEC_COLS_SD = [
    "Sec Clarifier pH",
    "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier RAS",
    "Sec Sed pH",
    "Sec Sed TSS (mg/L)",
    "Sec Sed RAS (New)",
]

# -- ADD-tier aeration (all same-day, unchanged from Exp3-SE1) ----------------
ADD_FEATURES = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
]

# -- Calendar/operational (real-time or trivially known) ----------------------
COMMON_BASE = ["Flow (MLD)", "Power Total (KW)", "year"]
CYCLIC      = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

# -- Target definitions --------------------------------------------------------
DATASETS = [
    ("grab_BOD", "Grab",      GRAB_INLET_SD, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", "Grab",      GRAB_INLET_SD, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", "Grab",      GRAB_INLET_SD, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  "Grab",      GRAB_INLET_SD, "Effluent pH (Grab)"),
    ("comp_BOD", "Composite", COMP_INLET_SD, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", "Composite", COMP_INLET_SD, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", "Composite", COMP_INLET_SD, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  "Composite", COMP_INLET_SD, "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    for name, kind, inlet_cols, target in DATASETS:
        feature_cols = inlet_cols + SEC_COLS_SD + COMMON_BASE + CYCLIC + ADD_FEATURES
        cols = ["Date"] + feature_cols + [target]

        subset = df[cols].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  {name}.xlsx  -  {len(feature_cols)} features  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
