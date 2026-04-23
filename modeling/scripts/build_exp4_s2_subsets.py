"""
build_exp4_s2_subsets.py - Create Experiment 4 Sub-2 datasets.

Experiment 4 Sub-2 hypothesis: within the Exp4-S1 feature set, apply automated
iterative VIF pruning (threshold = 10) to remove intra-group collinear features.
The pruning runs on training rows only (2020-2024) so no 2025 data leaks into
the feature selection decision.

VIF excluded from analysis (always kept in dataset):
  - month, day_of_week, year  - temporal/categorical, not subject to VIF

Starting pool per target = the Exp4-S1 feature set for that target
(Exp3-S2 minus SVI, (New) aeration columns, Sec Sed columns).

Output: modeling/datasets/experiment4/sub_exp2/e4_s2_{variant}_{param}.xlsx

Usage (from project root):
    .venv/bin/python3 modeling/scripts/build_exp4_s2_subsets.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE    = os.path.join(os.path.dirname(MODELING_DIR), "raw_data", "All_Years_Full.xlsx")
OUT_DIR      = os.path.join(MODELING_DIR, "datasets", "experiment4", "sub_exp2")
E4S1_DIR     = os.path.join(MODELING_DIR, "datasets", "experiment4", "sub_exp1")
os.makedirs(OUT_DIR, exist_ok=True)

VIF_THRESHOLD = 10
TRAIN_YEARS   = [2020, 2021, 2022, 2023, 2024]

# ── Always-include temporal features (excluded from VIF, always in dataset) ────
TEMPORAL = ["month", "day_of_week", "year"]

# ── Exp4-S1 feature pool per target ───────────────────────────────────────────
# Derived from Exp3-S2 by removing: SVI (Existing/New), all (New) aeration cols,
# all Sec Sedimentation cols.  These pools are the starting point for VIF pruning.

GRAB_BASE = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Flow (MLD)", "Power Total (KW)",
]
COMP_BASE = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
    "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Flow (MLD)", "Power Total (KW)",
]
AERATION_EXISTING = [
    "Aeration DO (mg/L, Existing)", "Aeration MLSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)", "Aeration pH (Existing)",
]

# Registry: (target, variant, output_stem, min_year, extra_features)
# extra_features = Exp3-S2 ADD/CONSIDER features that were in Exp4-S1 for this target
REGISTRY = [
    (
        "Effluent BOD (mg/L, Grab)", "Grab", "grab_BOD", 2020,
        GRAB_BASE + AERATION_EXISTING + [
            "Inlet Total Coliform (CFU/100ml, Grab)", "Primary Sludge Totalizer (m3)",
        ],
    ),
    (
        "Effluent COD (mg/L, Grab)", "Grab", "grab_COD", 2020,
        GRAB_BASE + [
            "Inlet Total Coliform (CFU/100ml, Grab)",
        ],
    ),
    (
        "Effluent TSS (mg/L, Grab)", "Grab", "grab_TSS", 2020,
        GRAB_BASE + AERATION_EXISTING + [
            "Inlet Total Coliform (CFU/100ml, Grab)", "Primary BOD (mg/L)",
            "Primary Sludge Totalizer (m3)", "Power GE (KW)",
        ],
    ),
    (
        "Effluent pH (Grab)", "Grab", "grab_pH", 2020,
        GRAB_BASE + [
            "Aeration pH (Existing)", "Aeration DO (mg/L, Existing)",
            "Primary Clarifier pH", "Aeration SV30 (ml/L, Existing)",
        ],
    ),
    (
        "Effluent BOD (mg/L, Composite)", "Composite", "comp_BOD", 2021,
        COMP_BASE + AERATION_EXISTING + [
            "Inlet Total Coliform (CFU/100ml, Grab)", "Primary Sludge Totalizer (m3)",
            "Power GE (KW)",
        ],
    ),
    (
        "Effluent COD (mg/L, Composite)", "Composite", "comp_COD", 2021,
        COMP_BASE + [
            "Inlet Total Coliform (CFU/100ml, Grab)",
        ],
    ),
    (
        "Effluent TSS (mg/L, Composite)", "Composite", "comp_TSS", 2021,
        COMP_BASE + AERATION_EXISTING + [
            "Inlet Total Coliform (CFU/100ml, Grab)", "Inlet PO4/TP (mg/L, Grab)",
            "Power GE (KW)", "Primary Sludge Totalizer (m3)",
        ],
    ),
    (
        "Effluent pH (Composite)", "Composite", "comp_pH", 2021,
        COMP_BASE + [
            "Aeration pH (Existing)", "Aeration DO (mg/L, Existing)",
            "Primary Clarifier pH", "Aeration SV30 (ml/L, Existing)",
        ],
    ),
]


# ── VIF pruning ────────────────────────────────────────────────────────────────

def iterative_vif_prune(X_df: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> list:
    """Iteratively drop the highest-VIF feature until all VIF <= threshold."""
    feats = list(X_df.columns)
    dropped = []
    iteration = 0
    while True:
        X = X_df[feats].values.astype(float)
        vifs = np.array([variance_inflation_factor(X, i) for i in range(len(feats))])
        max_vif = vifs.max()
        if max_vif <= threshold:
            break
        worst = feats[int(np.argmax(vifs))]
        print(f"    [VIF iter {iteration+1}] Drop '{worst}' (VIF={max_vif:.1f})")
        dropped.append((worst, round(float(max_vif), 1)))
        feats.remove(worst)
        iteration += 1
    # Print final VIF scores
    X_final = X_df[feats].values.astype(float)
    final_vifs = {feats[i]: round(float(variance_inflation_factor(X_final, i)), 2)
                  for i in range(len(feats))}
    return feats, dropped, final_vifs


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_subset(raw: pd.DataFrame, target: str, variant: str,
                 stem: str, min_year: int, pool: list) -> dict:
    # Deduplicate pool
    pool = list(dict.fromkeys(pool))

    # Only keep pool features that exist in the raw data
    pool = [f for f in pool if f in raw.columns]

    # Restrict to min_year+ rows that have the target
    sub = raw[raw["year"] >= min_year].copy()

    # Compute VIF on training rows only, using pool features with no missing values
    train_mask = sub["year"].isin(TRAIN_YEARS)
    train_sub  = sub[train_mask][pool].dropna()
    if len(train_sub) < 30:
        print(f"  WARNING: only {len(train_sub)} training rows with complete pool - skipping VIF")
        surviving = pool
        dropped_info = []
        final_vifs = {}
    else:
        print(f"  Pool ({len(pool)} feats): {pool}")
        print(f"  VIF pruning on {len(train_sub)} training rows...")
        surviving, dropped_info, final_vifs = iterative_vif_prune(train_sub)

    # Build final dataset
    all_cols = ["Date"] + surviving + TEMPORAL + [target]
    all_cols = [c for c in all_cols if c in sub.columns]

    df_out = sub[all_cols].dropna(subset=surviving + [target]).reset_index(drop=True)
    return {
        "df": df_out,
        "n_rows": len(df_out),
        "pool_size": len(pool),
        "surviving": surviving,
        "n_surviving": len(surviving),
        "dropped": dropped_info,
        "final_vifs": final_vifs,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Build Experiment 4 Sub-2 datasets (VIF-pruned from Exp4-S1 pool)")
    print("=" * 70)

    # Load raw
    print(f"\nLoading {DATA_FILE} ...")
    raw = pd.read_excel(DATA_FILE, parse_dates=["Date"])
    raw["month"]       = raw["Date"].dt.month
    raw["day_of_week"] = raw["Date"].dt.dayofweek
    raw["year"]        = raw["Date"].dt.year
    print(f"  {len(raw)} rows, {int(raw['year'].min())}-{int(raw['year'].max())}")

    # Load Exp4-S1 row counts for delta comparison
    s1_counts = {}
    for stem in ["grab_BOD","grab_COD","grab_TSS","grab_pH",
                 "comp_BOD","comp_COD","comp_TSS","comp_pH"]:
        p = os.path.join(E4S1_DIR, f"{stem}.xlsx")
        if os.path.exists(p):
            s1_counts[stem] = len(pd.read_excel(p))

    summary_rows = []

    for target, variant, stem, min_year, pool in REGISTRY:
        print(f"\n{'─'*60}")
        print(f"[{variant}] {stem}  →  {target}")

        result = build_subset(raw, target, variant, stem, min_year, pool)

        out_path = os.path.join(OUT_DIR, f"{stem}.xlsx")
        result["df"].to_excel(out_path, index=False)

        n_s1 = s1_counts.get(stem, "-")
        delta = (result["n_rows"] - n_s1) if isinstance(n_s1, int) else "-"
        delta_str = f"+{delta}" if isinstance(delta, int) and delta >= 0 else str(delta)

        print(f"  Surviving features ({result['n_surviving']}): {result['surviving']}")
        print(f"  Final VIFs: {result['final_vifs']}")
        print(f"  Rows → {result['n_rows']}  (Exp4-S1 had {n_s1}, Δ={delta_str})")
        print(f"  Written → {out_path}")

        summary_rows.append({
            "target": target,
            "variant": variant,
            "min_year": min_year,
            "pool_size": result["pool_size"],
            "n_surviving": result["n_surviving"],
            "dropped_features": "; ".join(f"{f}(VIF={v})" for f, v in result["dropped"]),
            "surviving_features": "; ".join(result["surviving"]),
            "n_rows": result["n_rows"],
            "n_rows_s1": n_s1,
            "delta_vs_s1": delta_str,
        })

    # Print summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in summary_rows:
        stem = r["target"].replace("Effluent ", "").replace(" (mg/L, ", " ").replace(")", "")
        print(f"  {stem:<30} pool={r['pool_size']:>2d}  kept={r['n_surviving']:>2d}  "
              f"rows={r['n_rows']:>4d}  (S1={r['n_rows_s1']}, Δ={r['delta_vs_s1']})")

    print(f"\nFiles written to: {OUT_DIR}")

    # Save feature log
    log_path = os.path.join(OUT_DIR, "vif_feature_log.xlsx")
    pd.DataFrame(summary_rows).to_excel(log_path, index=False)
    print(f"Feature log     → {log_path}")


if __name__ == "__main__":
    main()
