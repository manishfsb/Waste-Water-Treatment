"""
make_sub1_datasets.py - Experiment 9 SE1 (W1: 2024-only training window).

Copies the Exp3-SE1 feature set (27 features: Inlet + Secondary + COMMON_CYCLIC
+ ADD-tier aeration) but strips the Exp3-S1 prediction columns so the modeling
scripts start clean. All rows (2021-2025) are retained; the modeling scripts
filter TRAIN_YEARS = [2024] and TEST_YEAR = 2025.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment9/sub_exp1/make_sub1_datasets.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..",
                                           "datasets", "experiment3", "sub_exp1"))
OUT_DIR    = SCRIPT_DIR

DATASETS = [
    "grab_BOD", "grab_COD", "grab_TSS", "grab_pH",
    "comp_BOD", "comp_COD", "comp_TSS", "comp_pH",
]


def main():
    print("Experiment 9 SE1 - W1 dataset creation (2024-only training window)")
    print(f"Source : {SRC_DIR}")
    print(f"Output : {OUT_DIR}\n")

    for name in DATASETS:
        src = os.path.join(SRC_DIR, f"{name}.xlsx")
        out = os.path.join(OUT_DIR, f"{name}.xlsx")

        df = pd.read_excel(src, parse_dates=["Date"])

        # Drop Exp3-S1 prediction columns - Exp9 starts with clean datasets
        drop_cols = [c for c in df.columns if c.startswith("predicted_")]
        df = df.drop(columns=drop_cols)

        train_rows = (df["year"] == 2024).sum()
        test_rows  = (df["year"] == 2025).sum()

        df.to_excel(out, index=False)
        print(f"  {name}.xlsx -> 2024 train rows: {train_rows} | "
              f"2025 test rows: {test_rows} | cols: {len(df.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
