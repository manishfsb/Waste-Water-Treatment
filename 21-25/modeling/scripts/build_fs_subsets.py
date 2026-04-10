"""
build_fs_subsets.py — Create feature-selected subset Excel files.

Reads feature_importance.xlsx (Phase 5 output) and builds new subset files
that use only Core + Useful features (normalised perm importance >= 0.03),
dropping Weak features. Because dropna() is applied to fewer columns, more
rows are retained compared to the baseline subsets.

Outputs (mirroring baseline naming):
  experiment1_fs/                 — Experiment 1 subset files (no data/ subdir)
  experiment2_s1_fs/data/         — Experiment 2 Sub-1 subset files
  experiment2_s2_fs/data/         — Experiment 2 Sub-2 subset files

Row-count comparison (baseline vs feature-selected) is printed for each dataset.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/build_fs_subsets.py
"""

import os

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(os.path.dirname(SCRIPT_DIR), "All_Years_Full.xlsx")
FI_FILE      = os.path.join(SCRIPT_DIR, "feature_selection", "feature_importance.xlsx")

EXP1_FS_DIR   = os.path.join(SCRIPT_DIR, "experiment1_fs")
EXP2S1_FS_DIR = os.path.join(SCRIPT_DIR, "experiment2_s1_fs", "data")
EXP2S2_FS_DIR = os.path.join(SCRIPT_DIR, "experiment2_s2_fs", "data")

# Baseline subset dirs (for row-count comparison)
EXP1_BASE_DIR   = os.path.join(SCRIPT_DIR, "experiment1")
EXP2S1_BASE_DIR = os.path.join(SCRIPT_DIR, "experiment2_s1", "data")
EXP2S2_BASE_DIR = os.path.join(SCRIPT_DIR, "experiment2_s2", "data")

# ── Feature selection threshold ────────────────────────────────────────────────
FS_THRESHOLD = 0.03   # Core (>= 0.08) + Useful (0.03–0.07); drop Weak (< 0.03)

# ── Registry ───────────────────────────────────────────────────────────────────
# (exp_label, variant, dataset_name, output_dir, baseline_dir, target, min_year)
# exp_label and variant must match feature_importance.xlsx exactly.
# min_year matches the baseline subset creation in stage1_modeling.py.
REGISTRY = [
    # Experiment 1 — Grab
    ("Experiment 1", "Grab", "stage1_grab_BOD",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent BOD (mg/L, Grab)", 2020),
    ("Experiment 1", "Grab", "stage1_grab_COD",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent COD (mg/L, Grab)", 2020),
    ("Experiment 1", "Grab", "stage1_grab_TSS",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent TSS (mg/L, Grab)", 2020),
    ("Experiment 1", "Grab", "stage1_grab_pH",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent pH (Grab)", 2021),
    # Experiment 1 — Composite
    ("Experiment 1", "Composite", "stage1_comp_BOD",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent BOD (mg/L, Composite)", 2022),
    ("Experiment 1", "Composite", "stage1_comp_COD",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent COD (mg/L, Composite)", 2022),
    ("Experiment 1", "Composite", "stage1_comp_TSS",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent TSS (mg/L, Composite)", 2022),
    ("Experiment 1", "Composite", "stage1_comp_pH",
     EXP1_FS_DIR, EXP1_BASE_DIR, "Effluent pH (Composite)", 2022),
    # Experiment 2 Sub-1 — Grab
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_BOD",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent BOD (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_COD",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent COD (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_TSS",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent TSS (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_pH",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent pH (Grab)", 2020),
    # Experiment 2 Sub-1 — Composite
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_BOD",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent BOD (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_COD",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent COD (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_TSS",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent TSS (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_pH",
     EXP2S1_FS_DIR, EXP2S1_BASE_DIR, "Effluent pH (Composite)", 2022),
    # Experiment 2 Sub-2 — Grab
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_BOD",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent BOD (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_COD",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent COD (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_TSS",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent TSS (mg/L, Grab)", 2020),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_pH",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent pH (Grab)", 2020),
    # Experiment 2 Sub-2 — Composite
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_BOD",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent BOD (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_COD",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent COD (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_TSS",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent TSS (mg/L, Composite)", 2022),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_pH",
     EXP2S2_FS_DIR, EXP2S2_BASE_DIR, "Effluent pH (Composite)", 2022),
]


def main():
    print("build_fs_subsets.py — Feature-Selected Subset Creation")
    print("=" * 60)
    print(f"Threshold: perm_imp_norm >= {FS_THRESHOLD} (Core + Useful)")
    print()

    # ── Load source data ───────────────────────────────────────────────────────
    print(f"Loading {DATA_FILE} ...")
    raw = pd.read_excel(DATA_FILE, parse_dates=["Date"])
    raw["year"]        = raw["Date"].dt.year
    raw["month"]       = raw["Date"].dt.month
    raw["day_of_week"] = raw["Date"].dt.dayofweek
    print(f"  {len(raw)} total rows, {raw['year'].min()}–{raw['year'].max()}")
    print()

    # ── Load feature importance ────────────────────────────────────────────────
    fi = pd.read_excel(FI_FILE)
    print(f"Loaded feature_importance.xlsx — {len(fi)} feature×target rows")
    print()

    # ── Create output directories ──────────────────────────────────────────────
    for d in [EXP1_FS_DIR, EXP2S1_FS_DIR, EXP2S2_FS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Build each subset ──────────────────────────────────────────────────────
    summary_rows = []

    for exp, variant, name, out_dir, base_dir, target, min_year in REGISTRY:
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
            print(f"  WARNING: no features above threshold — skipping")
            continue

        n_all   = fi.loc[(fi["experiment"] == exp) & (fi["variant"] == variant) &
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
            print(f"  WARNING: columns missing from source data: {missing_cols} — skipping")
            continue

        sub = sub[cols_needed].copy()

        # Drop rows missing any selected feature or the target
        before = len(sub)
        sub = sub.dropna(subset=selected + [target]).reset_index(drop=True)
        after = len(sub)

        # Baseline row count for comparison
        base_path = os.path.join(base_dir, f"{name}.xlsx")
        if os.path.exists(base_path):
            base_df   = pd.read_excel(base_path)
            base_rows = len(base_df)
            delta     = after - base_rows
            delta_str = f"+{delta}" if delta >= 0 else str(delta)
            print(f"  Rows: {after} (baseline: {base_rows}, delta: {delta_str})")
        else:
            print(f"  Rows: {after}  (baseline file not found for comparison)")
            base_rows = None
            delta_str = "n/a"

        # Write
        out_path = os.path.join(out_dir, f"{name}.xlsx")
        sub.to_excel(out_path, index=False)
        print(f"  → {out_path}")

        summary_rows.append({
            "experiment":    exp,
            "variant":       variant,
            "dataset":       name,
            "target":        target,
            "n_features_fs": len(selected),
            "n_features_dropped": n_dropped,
            "rows_fs":       after,
            "rows_baseline": base_rows,
            "row_delta":     delta_str,
        })
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    col_w = 28
    print(f"{'Dataset':<{col_w}}  {'Feats kept':>10}  {'Feats dropped':>13}  {'Rows FS':>8}  {'Rows Base':>9}  {'Delta':>6}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['dataset']:<{col_w}}  {r['n_features_fs']:>10}  {r['n_features_dropped']:>13}  "
              f"{r['rows_fs']:>8}  {str(r['rows_baseline'] or 'n/a'):>9}  {r['row_delta']:>6}")

    print()
    print("Done. Feature-selected subset files written to:")
    print(f"  {EXP1_FS_DIR}")
    print(f"  {EXP2S1_FS_DIR}")
    print(f"  {EXP2S2_FS_DIR}")


if __name__ == "__main__":
    main()
