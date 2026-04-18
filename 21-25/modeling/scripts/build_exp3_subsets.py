"""
build_exp3_subsets.py — Create Experiment 3 subset files (Sub-1 and Sub-2).

Reads feature_audit_exp3/feature_audit.xlsx to get per-target ADD and CONSIDER
feature tiers, then builds two sets of 8 datasets from All_Years_Full.xlsx:

  Experiment 3 Sub-1 (S1): Exp2 Sub-2 baseline + ADD tier features per target
  Experiment 3 Sub-2 (S2): Exp2 Sub-2 baseline + ADD + CONSIDER tier features

Baseline (Exp2 Sub-2) per variant:
  Grab      : GRAB_INLET + SEC_COLS + COMMON  (19 features)
  Composite : COMP_INLET + SEC_COLS + COMMON  (19 features)

Outputs:
  modeling/experiment3_s1/data/stage3_s1_<variant>_<param>.xlsx  (8 files)
  modeling/experiment3_s2/data/stage3_s2_<variant>_<param>.xlsx  (8 files)

Usage (from project root):
  .venv/bin/python3 21-25/modeling/build_exp3_subsets.py
"""

import os
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE    = os.path.join(os.path.dirname(MODELING_DIR), "raw_data", "All_Years_Full.xlsx")
AUDIT_FILE   = os.path.join(MODELING_DIR, "feature_analysis", "audit", "feature_audit.xlsx")
EXP2S2_GRAB  = os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp2")  # row-count reference
EXP2S2_COMP  = EXP2S2_GRAB
S1_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1")
S2_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")

os.makedirs(S1_DIR, exist_ok=True)
os.makedirs(S2_DIR, exist_ok=True)

# ── Exp2 Sub-2 baseline feature sets ────────────────────────────────────────────
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
              "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
              "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
              "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)"]
COMMON     = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

EXP2_GRAB = GRAB_INLET + SEC_COLS + COMMON
EXP2_COMP = COMP_INLET + SEC_COLS + COMMON

# ── Registry ────────────────────────────────────────────────────────────────────
# (target_col, variant, file_stem, min_year)
# min_year matches prior subset conventions (2020 for most grab, 2022 for composite)
REGISTRY = [
    ("Effluent BOD (mg/L, Grab)",      "Grab",      "stage3_grab_BOD",  2020),
    ("Effluent COD (mg/L, Grab)",      "Grab",      "stage3_grab_COD",  2020),
    ("Effluent TSS (mg/L, Grab)",      "Grab",      "stage3_grab_TSS",  2020),
    ("Effluent pH (Grab)",             "Grab",      "stage3_grab_pH",   2020),
    ("Effluent BOD (mg/L, Composite)", "Composite", "stage3_comp_BOD",  2022),
    ("Effluent COD (mg/L, Composite)", "Composite", "stage3_comp_COD",  2022),
    ("Effluent TSS (mg/L, Composite)", "Composite", "stage3_comp_TSS",  2022),
    ("Effluent pH (Composite)",        "Composite", "stage3_comp_pH",   2022),
]


def build_subset(raw, target, baseline_feats, new_feats, min_year, out_path, label):
    """
    Filter raw to min_year+, keep only needed cols, dropna on features+target,
    save to out_path. Returns row count.
    """
    all_feats = baseline_feats + new_feats
    cols_needed = ["Date"] + all_feats + [target]
    # Only keep cols that actually exist in raw
    cols_needed = [c for c in cols_needed if c in raw.columns]
    feats_present = [f for f in all_feats if f in raw.columns]

    sub = raw[raw["year"] >= min_year][cols_needed].copy()
    sub = sub.dropna(subset=feats_present + [target]).reset_index(drop=True)
    sub.to_excel(out_path, index=False)
    return len(sub)


def main():
    print("Build Experiment 3 subsets")
    print("=" * 65)

    # ── Load raw data ────────────────────────────────────────────────────────────
    print(f"\nLoading {DATA_FILE} ...")
    raw = pd.read_excel(DATA_FILE)
    raw["month"]       = raw["Date"].dt.month
    raw["day_of_week"] = raw["Date"].dt.dayofweek
    raw["year"]        = raw["Date"].dt.year
    n_total = len(raw)
    print(f"  {n_total} total rows, {int(raw['year'].min())}–{int(raw['year'].max())}")

    # ── Load audit tiers ─────────────────────────────────────────────────────────
    print(f"\nLoading {AUDIT_FILE} ...")
    audit = pd.read_excel(AUDIT_FILE)
    # Keep only the latest run
    latest_run = audit["run"].max()
    audit = audit[audit["run"] == latest_run]

    # Build per-target tier maps
    def _features_for(target, tiers):
        mask = (audit["target"] == target) & (audit["tier"].isin(tiers))
        return audit.loc[mask, "feature"].tolist()

    # ── Build datasets ───────────────────────────────────────────────────────────
    summary_rows = []

    for target, variant, stem, min_year in REGISTRY:
        baseline = EXP2_GRAB if variant == "Grab" else EXP2_COMP

        add_feats  = _features_for(target, ["ADD"])
        cons_feats = _features_for(target, ["CONSIDER"])
        s2_new     = add_feats + cons_feats   # ADD + CONSIDER combined

        # Exp2 Sub-2 baseline row count (for delta reference)
        exp2_file = os.path.join(EXP2S2_GRAB, f"stage2_p2_{stem.replace('stage3_', '')}.xlsx")
        n_exp2 = len(pd.read_excel(exp2_file)) if os.path.exists(exp2_file) else None

        # ── S1: baseline + ADD ───────────────────────────────────────────────────
        s1_path = os.path.join(S1_DIR, f"s1_{stem}.xlsx")
        n_s1 = build_subset(raw, target, baseline, add_feats, min_year, s1_path,
                             f"S1 {stem}")

        # ── S2: baseline + ADD + CONSIDER ────────────────────────────────────────
        s2_path = os.path.join(S2_DIR, f"s2_{stem}.xlsx")
        n_s2 = build_subset(raw, target, baseline, s2_new, min_year, s2_path,
                             f"S2 {stem}")

        # Rows available in All_Years for this target+year range
        n_allyears = int(raw[(raw["year"] >= min_year) & raw[target].notna()].shape[0])

        summary_rows.append(dict(
            target=target, variant=variant,
            n_allyears=n_allyears,
            n_exp2=n_exp2,
            n_s1=n_s1,  add_feats=len(add_feats),
            n_s2=n_s2,  s2_feats=len(s2_new),
            drop_s1=n_allyears - n_s1,
            drop_s2=n_allyears - n_s2,
            pct_kept_s1=round(n_s1 / n_allyears * 100, 1) if n_allyears else 0,
            pct_kept_s2=round(n_s2 / n_allyears * 100, 1) if n_allyears else 0,
            s1_vs_exp2=(f"+{n_s1-n_exp2}" if n_exp2 and n_s1 >= n_exp2
                        else str(n_s1-n_exp2) if n_exp2 else "—"),
            s2_vs_exp2=(f"+{n_s2-n_exp2}" if n_exp2 and n_s2 >= n_exp2
                        else str(n_s2-n_exp2) if n_exp2 else "—"),
            add_note="(same as Exp2 Sub-2)" if len(add_feats) == 0 else "",
        ))

        print(f"\n[{variant}] {stem}")
        print(f"  ADD features    ({len(add_feats)}): {add_feats}")
        print(f"  CONSIDER feats  ({len(cons_feats)}): {cons_feats}")
        print(f"  All_Years rows with target (≥{min_year}): {n_allyears}")
        if n_exp2:
            print(f"  Exp2 Sub-2 baseline rows            : {n_exp2}")
        print(f"  S1 rows (baseline + ADD)            : {n_s1}  "
              f"(kept {n_s1/n_allyears*100:.1f}%,  dropped {n_allyears-n_s1})")
        print(f"  S2 rows (baseline + ADD + CONSIDER) : {n_s2}  "
              f"(kept {n_s2/n_allyears*100:.1f}%,  dropped {n_allyears-n_s2})")
        if len(add_feats) == 0:
            print(f"  ⚠  No ADD features — S1 is identical to Exp2 Sub-2 baseline")

    # ── Summary table ────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    hdr = (f"{'Dataset':<28} {'AllYrs':>7} {'Exp2S2':>7} "
           f"{'S1 rows':>8} {'S1 kept%':>9} {'S1Δ vs Exp2':>11}  "
           f"{'S2 rows':>8} {'S2 kept%':>9} {'S2Δ vs Exp2':>11}  Note")
    print(hdr)
    print("-" * 95)
    for r in summary_rows:
        stem = r['target'].replace("Effluent ", "").replace(" (mg/L, ", " ").replace(")", "")
        exp2_str = str(r['n_exp2']) if r['n_exp2'] else "—"
        print(f"{stem:<28} {r['n_allyears']:>7} {exp2_str:>7} "
              f"{r['n_s1']:>8} {r['pct_kept_s1']:>8.1f}% {r['s1_vs_exp2']:>11}  "
              f"{r['n_s2']:>8} {r['pct_kept_s2']:>8.1f}% {r['s2_vs_exp2']:>11}  "
              f"{r['add_note']}")

    print(f"\nFiles written to:")
    print(f"  {S1_DIR}")
    print(f"  {S2_DIR}")


if __name__ == "__main__":
    main()
