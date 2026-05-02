"""Row-cost analysis for Experiment 5.

Tests how many rows remain when cross-type inlet features are added to the
best-candidate base datasets from Experiment 3 (S1 and S2).

Sub1 hypothesis: adding GRAB_INLET to composite-target datasets helps predict
                 composite effluent quality.
Sub2 hypothesis: adding COMP_INLET to grab-target datasets helps predict grab
                 effluent quality.

Marginal row cost is computed relative to the base Exp3 datasets (after their
own dropna). A cost above 25% flags the configuration as high-risk.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

RAW_DATA = os.path.join(ROOT, "raw_data", "All_Years_Full.xlsx")
MODELING  = os.path.join(ROOT, "modeling")
EXP3_S1  = os.path.join(MODELING, "datasets", "experiment3", "sub_exp1")
EXP3_S2  = os.path.join(MODELING, "datasets", "experiment3", "sub_exp2")

GRAB_TARGETS = [
    "Effluent BOD (mg/L, Grab)", "Effluent COD (mg/L, Grab)",
    "Effluent TSS (mg/L, Grab)", "Effluent pH (Grab)",
]
COMP_TARGETS = [
    "Effluent BOD (mg/L, Composite)", "Effluent COD (mg/L, Composite)",
    "Effluent TSS (mg/L, Composite)", "Effluent pH (Composite)",
]

GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]

DATASET_SLUG = {
    "Effluent BOD (mg/L, Grab)":       "grab_BOD",
    "Effluent COD (mg/L, Grab)":       "grab_COD",
    "Effluent TSS (mg/L, Grab)":       "grab_TSS",
    "Effluent pH (Grab)":              "grab_pH",
    "Effluent BOD (mg/L, Composite)":  "comp_BOD",
    "Effluent COD (mg/L, Composite)":  "comp_COD",
    "Effluent TSS (mg/L, Composite)":  "comp_TSS",
    "Effluent pH (Composite)":         "comp_pH",
}


def _feature_cols(df, target):
    """Non-target, non-meta, non-prediction columns."""
    skip = {"Date", "year", "month", "day_of_week"}
    return [c for c in df.columns
            if c != target and c not in skip
            and not c.startswith("predicted_")]


def _load_base(exp_dir, target):
    slug  = DATASET_SLUG[target]
    path  = os.path.join(exp_dir, f"{slug}.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, parse_dates=["Date"])
    feat_cols = _feature_cols(df, target)
    df_clean  = df[[c for c in feat_cols if c != target] + [target]].dropna()
    return df_clean


def _n_train(df):
    return int((df.index.map(lambda i: True) if df.empty else
                (df["Date"] if "Date" in df.columns
                 else pd.Series(range(len(df)))).apply(lambda _: True)).sum()) \
        if df is None else (len(df[df.get("year", pd.Series()) < 2025])
                            if "year" in df.columns else len(df))


def analyze(raw_df, exp_dir, targets, cross_cols, direction_label):
    """
    For each target in `targets`, load the base dataset from `exp_dir`,
    then compute how many rows remain after adding `cross_cols` via a
    fresh dropna on All_Years_Full.xlsx.

    Returns a list of dicts for tabular display.
    """
    rows = []
    for tgt in targets:
        slug    = DATASET_SLUG[tgt]
        base_df = _load_base(exp_dir, tgt)
        if base_df is None:
            rows.append({"target": slug, "base_n": None, "cross_n": None,
                         "cost_pct": None, "flag": "NO FILE"})
            continue

        feat_cols = _feature_cols(base_df, tgt)
        train_mask = base_df.get("year", pd.Series(dtype=float)) < 2025 \
            if "year" in base_df.columns \
            else pd.Series([True] * len(base_df))
        base_n_train = int(train_mask.sum()) if "year" in base_df.columns \
            else len(base_df)

        # Build equivalent subset from All_Years_Full with cross-type cols added.
        # Cyclic calendar columns are derived from Date, not stored in raw file.
        raw_work = raw_df.copy()
        raw_work["year"]      = raw_work["Date"].dt.year
        raw_work["month"]     = raw_work["Date"].dt.month
        raw_work["day_of_week"] = raw_work["Date"].dt.dayofweek
        raw_work["month_sin"] = np.sin(2 * np.pi * raw_work["month"] / 12)
        raw_work["month_cos"] = np.cos(2 * np.pi * raw_work["month"] / 12)
        raw_work["dow_sin"]   = np.sin(2 * np.pi * raw_work["day_of_week"] / 7)
        raw_work["dow_cos"]   = np.cos(2 * np.pi * raw_work["day_of_week"] / 7)

        needed = feat_cols + cross_cols + [tgt]
        available = [c for c in needed if c in raw_work.columns]
        missing   = [c for c in needed if c not in raw_work.columns]
        if missing:
            rows.append({"target": slug, "base_n": base_n_train,
                         "cross_n": None, "cost_pct": None,
                         "flag": f"MISSING COLS: {missing}"})
            continue

        cross_df    = raw_work[available + ["Date", "year"]].dropna()
        cross_train = cross_df[cross_df["year"] < 2025]
        cross_n_train = len(cross_train)

        cost_pct  = (base_n_train - cross_n_train) / base_n_train * 100 \
            if base_n_train > 0 else float("nan")
        flag = "HIGH COST" if cost_pct > 25 else ("OK" if cost_pct >= 0 else "GAIN")

        rows.append({
            "target":    slug,
            "base_n":    base_n_train,
            "cross_n":   cross_n_train,
            "cost_pct":  round(cost_pct, 1),
            "flag":      flag,
        })
    return rows


def print_table(label, rows_s1, rows_s2):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    hdr = f"{'Target':<18} {'Base(S1)':>10} {'With+':>10} {'Cost%':>8} {'Flag':<12} "
    hdr += f"{'Base(S2)':>10} {'With+':>10} {'Cost%':>8} {'Flag'}"
    print(hdr)
    print("-" * 95)
    for r1, r2 in zip(rows_s1, rows_s2):
        b1  = str(r1["base_n"])  if r1["base_n"]  is not None else "-"
        c1  = str(r1["cross_n"]) if r1["cross_n"] is not None else "-"
        p1  = f"{r1['cost_pct']:.1f}%" if r1["cost_pct"] is not None else "-"
        b2  = str(r2["base_n"])  if r2["base_n"]  is not None else "-"
        c2  = str(r2["cross_n"]) if r2["cross_n"] is not None else "-"
        p2  = f"{r2['cost_pct']:.1f}%" if r2["cost_pct"] is not None else "-"
        flag1 = r1["flag"]; flag2 = r2["flag"]
        print(f"{r1['target']:<18} {b1:>10} {c1:>10} {p1:>8} {flag1:<12} "
              f"{b2:>10} {c2:>10} {p2:>8} {flag2}")
    print()


def main():
    print("Loading All_Years_Full.xlsx ...")
    raw = pd.read_excel(RAW_DATA, parse_dates=["Date"])

    print("\nAnalyzing Sub1 (Composite targets + GRAB_INLET cross-type features) ...")
    comp_s1 = analyze(raw, EXP3_S1, COMP_TARGETS, GRAB_INLET, "Composite+GrabInlet")
    comp_s2 = analyze(raw, EXP3_S2, COMP_TARGETS, GRAB_INLET, "Composite+GrabInlet")

    print("Analyzing Sub2 (Grab targets + COMP_INLET cross-type features) ...")
    grab_s1 = analyze(raw, EXP3_S1, GRAB_TARGETS, COMP_INLET, "Grab+CompInlet")
    grab_s2 = analyze(raw, EXP3_S2, GRAB_TARGETS, COMP_INLET, "Grab+CompInlet")

    print_table(
        "Sub1: Composite targets + GRAB_INLET  |  Exp3-S1 base vs Exp3-S2 base",
        comp_s1, comp_s2,
    )
    print_table(
        "Sub2: Grab targets + COMP_INLET  |  Exp3-S1 base vs Exp3-S2 base",
        grab_s1, grab_s2,
    )

    print("SUMMARY")
    print("-" * 50)
    for label, rows in [("Sub1 on Exp3-S1", comp_s1), ("Sub1 on Exp3-S2", comp_s2),
                        ("Sub2 on Exp3-S1", grab_s1), ("Sub2 on Exp3-S2", grab_s2)]:
        high  = sum(1 for r in rows if r["flag"] == "HIGH COST")
        costs = [r["cost_pct"] for r in rows if r["cost_pct"] is not None]
        avg   = f"{np.mean(costs):.1f}%" if costs else "-"
        print(f"  {label:<22}: avg cost {avg}, {high}/4 targets HIGH COST")
    print()


if __name__ == "__main__":
    main()
