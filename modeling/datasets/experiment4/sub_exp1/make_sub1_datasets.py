"""
make_sub1_datasets.py - Experiment 4 Sub-experiment 1: Manual Group Removal datasets.

Builds one xlsx per target by starting from the Exp3-SE2 31-feature pool and removing
three redundant/correlated feature groups:

  Removed groups:
    (1) Aeration SVI (Existing + New)  - derived from SV30/MLSS ratio, zero independent information
    (2) All New aeration tank columns  - correlated r=0.74-0.84 with Existing tanks;
                                         Existing-tank features are the primary ADD-tier contributors
    (3) Sec Clarifier (5 columns)      - Exp2 split analysis showed Sec Sedimentation was the
                                         stronger standalone predictor; Sec Clarifier removed here
                                         to test whether Sec Sed alone suffices

  Remaining features per dataset (18):
    - Inlet (4): Grab or Composite
    - Sec Sedimentation (5): pH, TSS, BOD, COD, RAS (New)
    - COMMON (3): Flow (MLD), Power Total (KW), year
    - Existing aeration minus SVI (4): DO, MLSS, SV30, pH
    - CONSIDER minus removed (2): Inlet Total Coliform, Primary Sludge Totalizer

  Cyclic calendar features (month/dow sin/cos) are NOT included here - they are computed
  on the fly by the training scripts (linear_modeling_exp4_s1.py / non_linear_modeling_exp4_s1.py).

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment4/sub_exp1/make_sub1_datasets.py
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# ── Feature groups ──────────────────────────────────────────────────────────────
GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]
SEC_SED = [
    "Sec Sed pH", "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON = ["Flow (MLD)", "Power Total (KW)", "year"]
AERATION_EXISTING = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration pH (Existing)",
    # Aeration SVI (Existing) excluded: derived directly from SV30/MLSS ratio
]
CONSIDER = [
    # Aeration SVI (New) excluded: derived
    # Power GE (KW) excluded: Power Total = Power GE + Power NEA (redundant)
    "Inlet Total Coliform (CFU/100ml, Grab)",
    "Primary Sludge Totalizer (m3)",
]

FEATURE_SET = SEC_SED + AERATION_EXISTING + CONSIDER

# ── Target definitions ─────────────────────────────────────────────────────────
DATASETS = [
    ("grab_BOD",  GRAB_INLET, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD",  GRAB_INLET, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS",  GRAB_INLET, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",   GRAB_INLET, "Effluent pH (Grab)"),
    ("comp_BOD",  COMP_INLET, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD",  COMP_INLET, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS",  COMP_INLET, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",   COMP_INLET, "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])
    df["year"] = df["Date"].dt.year

    for name, inlet_cols, target in DATASETS:
        feature_cols = inlet_cols + COMMON + FEATURE_SET
        cols = ["Date"] + feature_cols + [target]

        # Only keep columns that exist in the raw file
        cols = [c for c in cols if c in df.columns or c == "Date"]
        feature_cols = [c for c in feature_cols if c in df.columns]

        subset = df[["Date"] + feature_cols + [target]].dropna().copy()

        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  Saved {name}.xlsx - {len(feature_cols)} features "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
