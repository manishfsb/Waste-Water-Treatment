"""
make_sub1_datasets.py

Creates 8 Experiment 1 Sub-1 datasets: inlet features only, no Flow / Power /
calendar columns.  This is the minimal predictor set  -  just what enters the
plant (Grab or Composite) predicts what leaves.

Feature set per target (4 features):
  Grab:      Inlet pH (Grab), Inlet BOD (mg/L, Grab), Inlet COD (mg/L, Grab),
             Inlet TSS (mg/L, Grab)
  Composite: Inlet pH (Composite), Inlet BOD (mg/L, Composite),
             Inlet COD (mg/L, Composite), Inlet TSS (mg/L, Composite)

Source: experiment1/sub_exp2/*.xlsx  (already contain these columns plus more)
Output: experiment1/sub_exp1/{grab|comp}_{BOD|COD|TSS|pH}.xlsx

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment1/sub_exp1/make_sub1_datasets.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SRC_DIR      = os.path.join(MODELING_DIR, "datasets", "experiment1", "sub_exp2")
DEST_DIR     = SCRIPT_DIR

GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]

REGISTRY = [
    ("grab_BOD", GRAB_INLET, "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", GRAB_INLET, "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", GRAB_INLET, "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  GRAB_INLET, "Effluent pH (Grab)"),
    ("comp_BOD", COMP_INLET, "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", COMP_INLET, "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", COMP_INLET, "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  COMP_INLET, "Effluent pH (Composite)"),
]

KEEP_COLS = ["Date", "year"]   # preserved for splitting; not used as features


def main():
    print(f"Source      : {SRC_DIR}")
    print(f"Destination : {DEST_DIR}\n")

    for stem, features, target in REGISTRY:
        src  = os.path.join(SRC_DIR, f"{stem}.xlsx")
        dest = os.path.join(DEST_DIR, f"{stem}.xlsx")

        if not os.path.exists(src):
            print(f"  SKIP (not found): {stem}.xlsx")
            continue

        df = pd.read_excel(src, parse_dates=["Date"])
        cols = KEEP_COLS + features + [target]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  SKIP (missing cols {missing}): {stem}.xlsx")
            continue

        df_out = df[cols].dropna(subset=features + [target]).copy()
        df_out.to_excel(dest, index=False)
        print(f"  {stem}.xlsx: {len(df_out)} rows, {len(features)} features  → {dest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
