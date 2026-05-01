"""
make_sub2_datasets.py

Applies cyclic calendar encoding in-place to the 8 Experiment 1 Sub-2 datasets.

Changes applied to each file:
  - month (1-12)     → month_sin, month_cos   (derived from Date)
  - day_of_week (0-6) → dow_sin,  dow_cos      (derived from Date)
  - Raw month and day_of_week columns dropped
  - All predicted_* columns stripped (clean slate for new run)
  - Date, year, and all other columns preserved unchanged

After running this script the sub_exp2 datasets use COMMON_CYCLIC:
  ["Flow (MLD)", "Power Total (KW)", "year",
   "month_sin", "month_cos", "dow_sin", "dow_cos"]

Re-run the downstream generators if needed:
  - sub_exp1/make_sub1_datasets.py   (reads sub_exp2, unaffected  -  inlet cols only)
  - sub_exp2_cyclic datasets are now redundant (sub_exp2 IS cyclic)

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment1/sub_exp2/make_sub2_datasets.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    "grab_BOD.xlsx",
    "grab_COD.xlsx",
    "grab_TSS.xlsx",
    "grab_pH.xlsx",
    "comp_BOD.xlsx",
    "comp_COD.xlsx",
    "comp_TSS.xlsx",
    "comp_pH.xlsx",
]


def apply_cyclic_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    m = df["Date"].dt.month.values.astype(float)
    d = df["Date"].dt.dayofweek.values.astype(float)

    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * d / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * d / 7)

    drop_cols = [c for c in df.columns
                 if c in ("month", "day_of_week") or c.startswith("predicted_")]
    df.drop(columns=drop_cols, inplace=True)

    return df


def main():
    print(f"Directory : {SCRIPT_DIR}\n")

    for fname in DATASETS:
        path = os.path.join(SCRIPT_DIR, fname)

        if not os.path.exists(path):
            print(f"  SKIP (not found): {fname}")
            continue

        df = pd.read_excel(path, parse_dates=["Date"])
        n_cols_before = len(df.columns)
        dropped_pred  = sum(1 for c in df.columns if c.startswith("predicted_"))
        had_raw_cal   = any(c in df.columns for c in ("month", "day_of_week"))

        df = apply_cyclic_encoding(df)

        n_cols_after = len(df.columns)
        df.to_excel(path, index=False)

        notes = []
        if had_raw_cal:
            notes.append("month/dow → sin/cos")
        if dropped_pred:
            notes.append(f"{dropped_pred} predicted_* cols stripped")
        if not notes:
            notes.append("no changes (already cyclic?)")

        print(f"  {fname}: {n_cols_before} → {n_cols_after} cols "
              f"({len(df)} rows)  [{', '.join(notes)}]")

    print("\nDone. Run modeling scripts to produce fresh results.")


if __name__ == "__main__":
    main()
