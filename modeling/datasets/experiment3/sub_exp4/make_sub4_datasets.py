"""
make_sub4_datasets.py - Experiment 3 Sub-experiment 1: all-features datasets.

Builds one xlsx per target using EVERY available feature from All_Years_Full
except those explicitly excluded:
  - Data-leakage:  effluent non-target measurements (FRC, O&G, NH3-N, coliforms)
  - Redundant:     MLVSS variants (highly collinear with MLSS)
                   Power NEA (Power GE + Power NEA = Power Total → perfect collinearity)
  - Calendar raw:  month, day_of_week  (replaced by sin/cos cyclic encoding)
  - Identity:      Date, year  (year added back as plain int feature)
  - Targets:       all 8 effluent quality columns

Grab datasets use grab-specific inlet features; composite datasets use composite inlets.
Calendar cyclic features (month_sin/cos, dow_sin/cos) are added at build time.

Resulting feature counts:
  Grab      : ~39 process features + 5 calendar (month_sin/cos, dow_sin/cos, year) = ~44
  Composite : ~34 process features + 5 calendar = ~39

Row counts are much lower than Exp2-Sub2 because including all features (especially
Grit Classifier TSS, Primary COD/TSS, Inlet TKN/O&G/Fecal Coliform) causes joint
missingness loss.  This illustrates WHY the Phase 6 Feature Suggestions Audit was
necessary  -  see EDA report section s11 for tier assignments.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment3/sub_exp4/make_sub4_datasets.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
RAW_FILE     = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = SCRIPT_DIR

# ── Exclusion lists ────────────────────────────────────────────────────────────
LEAKAGE = {
    "Effluent FRC (mg/L)",
    "Effluent Fecal Coliform (CFU/100ml)",
    "Effluent NH3-N (mg/L)",
    "Effluent O&G (mg/L)",
    "Effluent Total Coliform (CFU/100ml)",
}
REDUNDANT = {
    "Aeration MLVSS (mg/L, Existing)",   # ~0.95 corr with MLSS Existing
    "Aeration MLVSS (mg/L, New)",         # ~0.95 corr with MLSS New
    "Power NEA (KW)",                     # Power GE + Power NEA = Power Total
}
CALENDAR_RAW = {"month", "day_of_week"}
META        = {"Date"}   # year kept as a feature

TARGETS = {
    "Effluent pH (Grab)", "Effluent BOD (mg/L, Grab)",
    "Effluent COD (mg/L, Grab)", "Effluent TSS (mg/L, Grab)",
    "Effluent pH (Composite)", "Effluent BOD (mg/L, Composite)",
    "Effluent COD (mg/L, Composite)", "Effluent TSS (mg/L, Composite)",
}

EXCLUDE_ALL = LEAKAGE | REDUNDANT | CALENDAR_RAW | META | TARGETS

# ── Dataset definitions ────────────────────────────────────────────────────────
DATASETS = [
    ("grab_BOD",  "Grab",      "Effluent BOD (mg/L, Grab)"),
    ("grab_COD",  "Grab",      "Effluent COD (mg/L, Grab)"),
    ("grab_TSS",  "Grab",      "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",   "Grab",      "Effluent pH (Grab)"),
    ("comp_BOD",  "Composite", "Effluent BOD (mg/L, Composite)"),
    ("comp_COD",  "Composite", "Effluent COD (mg/L, Composite)"),
    ("comp_TSS",  "Composite", "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",   "Composite", "Effluent pH (Composite)"),
]


def build_datasets():
    print(f"Loading {RAW_FILE} ...")
    df = pd.read_excel(RAW_FILE, parse_dates=["Date"])

    # ── Derived calendar features ──────────────────────────────────────────────
    df["year"]      = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7)

    # ── Build base feature pools per sample type ───────────────────────────────
    # Grab: exclude composite-only inlets; Composite: exclude grab-only inlets
    grab_exclude = EXCLUDE_ALL | {c for c in df.columns if "Composite" in c and c not in TARGETS}
    comp_exclude = EXCLUDE_ALL | {c for c in df.columns if "Grab" in c and c not in TARGETS}

    GRAB_KS = [c for c in df.columns if c not in grab_exclude]
    COMP_KS = [c for c in df.columns if c not in comp_exclude]

    print(f"  Grab all-features features ({len(GRAB_KS)}): {GRAB_KS}")
    print(f"  Comp all-features features ({len(COMP_KS)}): {COMP_KS}")

    CYCLIC = ["month_sin", "month_cos", "dow_sin", "dow_cos"]

    for name, kind, target in DATASETS:
        pool = GRAB_KS if kind == "Grab" else COMP_KS
        cols = pool + [target]
        subset = df[cols].dropna().copy()
        subset.insert(0, "Date", df.loc[subset.index, "Date"])

        # Re-order: Date first, then features, then target
        feat_cols = [c for c in subset.columns if c not in {target, "Date"}]
        subset = subset[["Date"] + feat_cols + [target]]

        # Train/test summary
        train_n = int((subset["year"] < 2025).sum())
        test_n  = int((subset["year"] == 2025).sum())

        out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
        subset.to_excel(out_path, index=False)
        print(f"  Saved {name}.xlsx   -  {len(feat_cols)} feature cols  "
              f"(train={train_n}, test={test_n})")


if __name__ == "__main__":
    build_datasets()
    print("\nDone.")
