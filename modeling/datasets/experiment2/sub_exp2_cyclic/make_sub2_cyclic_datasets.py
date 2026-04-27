"""
make_sub2_cyclic_datasets.py

Reads existing sub_exp2/ xlsx files and rewrites them with cyclic calendar
encoding to this directory (sub_exp2_cyclic/).

Changes applied:
  - Adds month_sin = sin(2π·month/12), month_cos = cos(2π·month/12)
  - Adds dow_sin   = sin(2π·dayofweek/7), dow_cos = cos(2π·dayofweek/7)
  - Drops raw month and day_of_week columns
  - Drops all predicted_* columns
  - All other columns (Inlet, Sec, Flow, Power, year, Date, target) retained

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp2_cyclic/make_sub2_cyclic_datasets.py
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "sub_exp2"))

# ── Dataset registry ───────────────────────────────────────────────────────────
STEMS = [
    ("grab_BOD", "Effluent BOD (mg/L, Grab)"),
    ("grab_COD", "Effluent COD (mg/L, Grab)"),
    ("grab_TSS", "Effluent TSS (mg/L, Grab)"),
    ("grab_pH",  "Effluent pH (Grab)"),
    ("comp_BOD", "Effluent BOD (mg/L, Composite)"),
    ("comp_COD", "Effluent COD (mg/L, Composite)"),
    ("comp_TSS", "Effluent TSS (mg/L, Composite)"),
    ("comp_pH",  "Effluent pH (Composite)"),
]


def make_cyclic(stem: str, target: str):
    src = os.path.join(SRC_DIR, f"{stem}.xlsx")
    if not os.path.exists(src):
        print(f"  WARNING: source not found — {src}")
        return

    df = pd.read_excel(src, parse_dates=["Date"])

    # Drop predicted_* columns
    pred_cols = [c for c in df.columns if c.startswith("predicted_")]
    if pred_cols:
        df = df.drop(columns=pred_cols)

    # Derive cyclic encodings from Date (authoritative source)
    if "Date" in df.columns:
        df["month"]      = df["Date"].dt.month
        df["day_of_week"] = df["Date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Drop raw integer calendar columns
    df = df.drop(columns=[c for c in ["month", "day_of_week"] if c in df.columns])

    out_path = os.path.join(SCRIPT_DIR, f"{stem}.xlsx")
    df.to_excel(out_path, index=False)
    print(f"  {stem}: {len(df)} rows × {len(df.columns)} cols → {out_path}")


def main():
    print("=== make_sub2_cyclic_datasets.py ===")
    print(f"Source:  {SRC_DIR}")
    print(f"Output:  {SCRIPT_DIR}\n")
    for stem, target in STEMS:
        make_cyclic(stem, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
