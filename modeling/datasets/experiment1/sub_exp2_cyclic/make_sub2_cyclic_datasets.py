"""
make_exp1_cyclic_datasets.py

Reads the 8 Experiment 1 dataset files from datasets/experiment1/ and writes
cyclic-encoded versions to datasets/experiment1_cyclic/.

Changes applied:
  - month (1-12)     → month_sin, month_cos   (computed from Date)
  - day_of_week (0-6) → dow_sin,   dow_cos     (computed from Date)
  - Raw month and day_of_week columns dropped
  - All predicted_* columns stripped (clean slate for new experiment)
  - year and Date columns preserved unchanged

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment1_cyclic/make_exp1_cyclic_datasets.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SRC_DIR  = os.path.join(MODELING_DIR, "datasets", "experiment1", "sub_exp2")
DEST_DIR = SCRIPT_DIR

DATASETS = [
    "stage1_grab_BOD.xlsx",
    "stage1_grab_COD.xlsx",
    "stage1_grab_TSS.xlsx",
    "stage1_grab_pH.xlsx",
    "stage1_comp_BOD.xlsx",
    "stage1_comp_COD.xlsx",
    "stage1_comp_TSS.xlsx",
    "stage1_comp_pH.xlsx",
]


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derive from Date column to guarantee consistency with the Date values
    m = df["Date"].dt.month.values.astype(float)
    d = df["Date"].dt.dayofweek.values.astype(float)   # 0=Monday, 6=Sunday (matches day_of_week)

    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * d / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * d / 7)

    # Drop raw cyclic-encoded originals and any stale predictions
    drop_cols = [c for c in df.columns
                 if c in ("month", "day_of_week") or c.startswith("predicted_")]
    df.drop(columns=drop_cols, inplace=True)

    return df


def main():
    print(f"Source      : {SRC_DIR}")
    print(f"Destination : {DEST_DIR}\n")

    for fname in DATASETS:
        src  = os.path.join(SRC_DIR, fname)
        dest = os.path.join(DEST_DIR, fname)

        if not os.path.exists(src):
            print(f"  SKIP (not found): {fname}")
            continue

        df = pd.read_excel(src, parse_dates=["Date"])
        n_before = len(df.columns)
        df = add_cyclic_features(df)
        n_after = len(df.columns)

        df.to_excel(dest, index=False)
        print(f"  {fname}: {n_before} → {n_after} cols  ({len(df)} rows)  → {dest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
