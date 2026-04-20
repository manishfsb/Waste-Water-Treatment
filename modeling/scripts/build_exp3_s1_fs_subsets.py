"""
build_exp3_s1_fs_subsets.py - Create feature-selected subset Excel files for Exp3 Sub-1.

Reads feature_importance_exp3_s1.xlsx (output of feature_selection_exp3_s1.py) and builds
new subset files that use only Core + Useful features (normalised perm importance >= 0.03),
dropping Weak features. Because dropna() is applied to fewer columns, more rows are retained
compared to the Exp3-S1 baseline subsets.

Outputs:
  datasets/experiment3/sub_exp1/feature_selected_datasets/  - 8 xlsx files

Row-count comparison (Exp3-S1 baseline vs feature-selected) is printed for each dataset.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/scripts/build_exp3_s1_fs_subsets.py
"""

import os

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE    = os.path.join(os.path.dirname(MODELING_DIR), "raw_data", "All_Years_Full.xlsx")
FI_FILE      = os.path.join(MODELING_DIR, "feature_analysis", "selection",
                             "feature_importance_exp3_s1.xlsx")

FS_DIR   = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1",
                         "feature_selected_datasets")
BASE_DIR = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1")

# ── Feature selection threshold ────────────────────────────────────────────────
FS_THRESHOLD = 0.03   # Core (>= 0.08) + Useful (0.03-0.07); drop Weak (< 0.03)

# ── Registry ───────────────────────────────────────────────────────────────────
# (exp_label, variant, dataset_name, target, min_year)
REGISTRY = [
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_BOD", "Effluent BOD (mg/L, Grab)",      2021),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_COD", "Effluent COD (mg/L, Grab)",      2021),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_TSS", "Effluent TSS (mg/L, Grab)",      2021),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_pH",  "Effluent pH (Grab)",             2021),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_BOD", "Effluent BOD (mg/L, Composite)", 2021),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_COD", "Effluent COD (mg/L, Composite)", 2021),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_TSS", "Effluent TSS (mg/L, Composite)", 2021),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_pH",  "Effluent pH (Composite)",        2021),
]


def main():
    print("build_exp3_s1_fs_subsets.py - Feature-Selected Subset Creation (Exp3 Sub-1)")
    print("=" * 70)
    print(f"Threshold: perm_imp_norm >= {FS_THRESHOLD} (Core + Useful)")
    print()

    # ── Load source data ───────────────────────────────────────────────────────
    print(f"Loading {DATA_FILE} ...")
    raw = pd.read_excel(DATA_FILE, parse_dates=["Date"])
    raw["year"]        = raw["Date"].dt.year
    raw["month"]       = raw["Date"].dt.month
    raw["day_of_week"] = raw["Date"].dt.dayofweek
    print(f"  {len(raw)} total rows, {raw['year'].min()}-{raw['year'].max()}")
    print()

    # ── Load feature importance ────────────────────────────────────────────────
    fi = pd.read_excel(FI_FILE)
    print(f"Loaded feature_importance_exp3_s1.xlsx - {len(fi)} feature×target rows")
    print()

    # ── Create output directory ────────────────────────────────────────────────
    os.makedirs(FS_DIR, exist_ok=True)

    # ── Build each subset ──────────────────────────────────────────────────────
    summary_rows = []

    for exp, variant, name, target, min_year in REGISTRY:
        print(f"[{exp} / {variant}]  {name}")

        # Selected features for this (experiment, variant, target) combination
        mask = (
            (fi["experiment"] == exp) &
            (fi["variant"]    == variant) &
            (fi["target"]     == target) &
            (fi["perm_imp_norm"] >= FS_THRESHOLD)
        )
        selected = fi.loc[mask, "feature"].tolist()

        if not selected:
            print(f"  WARNING: no features above threshold - skipping")
            continue

        n_all     = fi.loc[(fi["experiment"] == exp) & (fi["variant"] == variant) &
                           (fi["target"] == target)].shape[0]
        n_dropped = n_all - len(selected)
        print(f"  Features: {len(selected)} kept, {n_dropped} dropped")

        # Filter year range
        sub = raw[raw["year"] >= min_year].copy()

        # Keep only needed columns
        cols_needed = ["Date", "year", "month", "day_of_week"] + selected + [target]
        cols_needed = list(dict.fromkeys(cols_needed))   # preserve order, dedup

        missing_cols = [c for c in cols_needed if c not in sub.columns]
        if missing_cols:
            print(f"  WARNING: columns missing from source data: {missing_cols} - skipping")
            continue

        sub = sub[cols_needed].copy()

        # Drop rows missing any selected feature or the target
        sub = sub.dropna(subset=selected + [target]).reset_index(drop=True)
        after = len(sub)

        # Baseline row count for comparison
        base_path = os.path.join(BASE_DIR, f"{name}.xlsx")
        if os.path.exists(base_path):
            base_df   = pd.read_excel(base_path)
            base_rows = len(base_df)
            delta     = after - base_rows
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
            print(f"  Rows: {after} (Exp3-S1 baseline: {base_rows}, delta: {delta_str})")
        else:
            print(f"  Rows: {after}  (baseline file not found for comparison)")
            base_rows = None
            delta_str = "n/a"

        # Write
        out_path = os.path.join(FS_DIR, f"{name}.xlsx")
        sub.to_excel(out_path, index=False)
        print(f"  → {out_path}")

        summary_rows.append({
            "experiment":         exp,
            "variant":            variant,
            "dataset":            name,
            "target":             target,
            "n_features_fs":      len(selected),
            "n_features_dropped": n_dropped,
            "rows_fs":            after,
            "rows_baseline":      base_rows,
            "row_delta":          delta_str,
        })
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    col_w = 24
    print(f"{'Dataset':<{col_w}}  {'Feats kept':>10}  {'Feats dropped':>13}  "
          f"{'Rows FS':>8}  {'Rows Base':>9}  {'Delta':>6}")
    print("-" * 76)
    for r in summary_rows:
        print(f"{r['dataset']:<{col_w}}  {r['n_features_fs']:>10}  "
              f"{r['n_features_dropped']:>13}  "
              f"{r['rows_fs']:>8}  {str(r['rows_baseline'] or 'n/a'):>9}  "
              f"{r['row_delta']:>6}")

    print()
    print("Done. Feature-selected subset files written to:")
    print(f"  {FS_DIR}")


if __name__ == "__main__":
    main()
