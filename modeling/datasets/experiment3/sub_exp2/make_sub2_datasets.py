"""
make_sub2_datasets.py - Experiment 3 Sub-experiment 2: ADD+CONSIDER-tier datasets.

Builds one xlsx per target using the Inlet + Secondary + COMMON baseline features (21 cols) PLUS the
ADD- and CONSIDER-tier features from the Phase 6 Feature Audit.

  ADD-tier (MI >= 0.20, marginal cost <= 20%):
    Aeration DO  (mg/L, Existing)
    Aeration MLSS(mg/L, Existing)
    Aeration SV30(ml/L, Existing)
    Aeration SVI (Existing)
    Aeration pH  (Existing)
    Aeration DO  (mg/L, New)
    Aeration SV30(ml/L, New)

  CONSIDER-tier additions (MI >= 0.15 & cost <= 35%, or MI >= 0.25 & cost <= 50%):
    Aeration SVI (New)
    Power GE (KW)
    Inlet Total Coliform (CFU/100ml, Grab)   [included in both Grab & Composite]
    Primary Sludge Totalizer (m3)

Total: 28 baseline + 4 CONSIDER = 32 features per dataset.

Cyclic calendar encoding (month/dow → sin/cos) is the unconditional standard
from Exp2 onwards.

Feature selection (LassoCV for OLS; OOF permutation importance for trees) is
run in the training scripts (exp3_s2/) on these datasets  -  not at dataset-build time.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment3/sub_exp2/make_sub2_datasets.py
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# ── Baseline feature groups (Inlet + Secondary + COMMON) ──────────────────────
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

# ── ADD-tier extras ────────────────────────────────────────────────────────────
ADD_FEATURES = [
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
]

# ── CONSIDER-tier extras ───────────────────────────────────────────────────────
# Power GE removed: Power Total (KW) = Power GE + Power NEA, so both cannot be features simultaneously.
CONSIDER_FEATURES = [
    "Aeration SVI (New)",
    "Inlet Total Coliform (CFU/100ml, Grab)",
    "Primary Sludge Totalizer (m3)",
]

# ── Target definitions ─────────────────────────────────────────────────────────
DATASETS = [
    ("grab_BOD",  "Grab",      GRAB_INLET, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD",  "Grab",      GRAB_INLET, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS",  "Grab",      GRAB_INLET, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",   "Grab",      GRAB_INLET, "Effluent pH (Grab)"),
    ("comp_BOD",  "Composite", COMP_INLET, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD",  "Composite", COMP_INLET, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS",  "Composite", COMP_INLET, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",   "Composite", COMP_INLET, "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    # Cyclic calendar features
    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    extra = ADD_FEATURES + CONSIDER_FEATURES

    for name, kind, inlet_cols, target in DATASETS:
        feature_cols = inlet_cols + SEC_COLS + COMMON_BASE + CYCLIC + extra
        cols = ["Date"] + feature_cols + [target]

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
