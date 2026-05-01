"""
make_sub1_split_datasets.py

Reads existing sub_exp1/ xlsx files (which contain ALL SEC_COLS + COMMON_CYCLIC)
and subsets them into two new directories:
  - sec_clarifier/  : 8 xlsx with only SEC_CLARIFIER_COLS + COMMON_CYCLIC (12 features)
  - sec_sed/        : 8 xlsx with only SEC_SED_COLS + COMMON_CYCLIC (12 features)

Calendar features use cyclic (sin/cos) encoding  -  no raw month/day_of_week columns.

Usage (from project root):
    .venv/bin/python3 modeling/datasets/experiment2/sub_exp1/make_sub1_split_datasets.py
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLR_DIR    = os.path.join(SCRIPT_DIR, "sec_clarifier")
SED_DIR    = os.path.join(SCRIPT_DIR, "sec_sed")
os.makedirs(CLR_DIR, exist_ok=True)
os.makedirs(SED_DIR, exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────────
SEC_CLARIFIER_COLS = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
]
SEC_SED_COLS = [
    "Sec Sed pH", "Sec Sed TSS (mg/L)",
    "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON_CYCLIC = ["Flow (MLD)", "Power Total (KW)", "year", "month_sin", "month_cos", "dow_sin", "dow_cos"]

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


def make_split(stem: str, target: str):
    src = os.path.join(SCRIPT_DIR, f"{stem}.xlsx")
    if not os.path.exists(src):
        print(f"  WARNING: source not found  -  {src}")
        return

    df = pd.read_excel(src, parse_dates=["Date"])

    # Drop any predicted_* columns
    pred_cols = [c for c in df.columns if c.startswith("predicted_")]
    if pred_cols:
        df = df.drop(columns=pred_cols)

    for label, feat_cols, out_dir in [
        ("Clarifier", SEC_CLARIFIER_COLS, CLR_DIR),
        ("Sed",       SEC_SED_COLS,       SED_DIR),
    ]:
        keep = ["Date"] + feat_cols + COMMON_CYCLIC + [target]
        missing = [c for c in keep if c not in df.columns and c != "Date"]
        if missing:
            print(f"  WARNING [{label}] {stem}: missing columns {missing}  -  skipping")
            continue

        # Keep only available columns
        avail = [c for c in keep if c in df.columns]
        sub = df[avail].dropna(subset=feat_cols + [target])
        out_path = os.path.join(out_dir, f"{stem}.xlsx")
        sub.to_excel(out_path, index=False)
        print(f"  [{label}] {stem}: {len(sub)} rows × {len(sub.columns)} cols → {out_path}")


def main():
    print("=== make_sub1_split_datasets.py ===")
    print(f"Output dirs: sec_clarifier/ and sec_sed/\n")
    for stem, target in STEMS:
        make_split(stem, target)
    print("\nDone.")


if __name__ == "__main__":
    main()
