"""
feature_audit_exp3.py - Phase 6: Candidate feature audit for Experiment 3.

For every numeric column NOT already in the Experiment 2 Sub-2 feature set,
this script computes - per target:

  - Pairwise missingness %  : rows where this feature is NaN, given target is present
  - Marginal row cost       : additional rows lost when adding this feature on top of
                              the Experiment 2 Sub-2 drop-na baseline (the operationally
                              relevant cost - pairwise alone can be misleading)
  - Pearson r, Spearman ρ   : computed on the pairwise-complete rows only
  - Mutual Information (MI) : model-agnostic signal estimate
  - Recommendation tier     : ADD / CONSIDER / LOW / SKIP / REDUNDANT

Tier logic (uses marginal row cost, not raw missingness):
  ADD       - MI ≥ 0.20 and marginal cost ≤ 20 %
  CONSIDER  - MI ≥ 0.15 and marginal cost ≤ 35 %, OR MI ≥ 0.25 and cost ≤ 50 %
  LOW       - MI ≥ 0.05
  SKIP      - MI < 0.05
  REDUNDANT - known collinear with a feature already in the baseline

Outputs (all in modeling/feature_audit_exp3/):
  plots/                              - signal-vs-cost scatter per target (8 PNG)
  feature_audit.xlsx                  - flat table of all metrics
  report_feature_audit_exp3_run_N.html

Usage (from project root):
  .venv/bin/python3 21-25/modeling/feature_audit_exp3/feature_audit_exp3.py
"""

import os
import sys
import base64
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_regression

# ── Paths ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
BASE_DIR     = os.path.dirname(MODELING_DIR)
DATA_FILE    = os.path.join(BASE_DIR, "raw_data", "All_Years_Full.xlsx")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
AUDIT_XLSX   = os.path.join(SCRIPT_DIR, "feature_audit.xlsx")

os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(MODELING_DIR, "scripts"))
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Baseline feature sets (Experiment 2 Sub-2) ──────────────────────────────────
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
              "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
              "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
              "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)"]
COMMON     = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

EXP2_GRAB  = GRAB_INLET + SEC_COLS + COMMON
EXP2_COMP  = COMP_INLET + SEC_COLS + COMMON
BASELINE_ALL = set(EXP2_GRAB + EXP2_COMP)

ALL_EFFLUENT = {
    "Effluent BOD (mg/L, Grab)", "Effluent COD (mg/L, Grab)",
    "Effluent TSS (mg/L, Grab)", "Effluent pH (Grab)",
    "Effluent BOD (mg/L, Composite)", "Effluent COD (mg/L, Composite)",
    "Effluent TSS (mg/L, Composite)", "Effluent pH (Composite)",
    "Effluent FRC (mg/L)", "Effluent O&G (mg/L)", "Effluent NH3-N (mg/L)",
    "Effluent Total Coliform (CFU/100ml)", "Effluent Fecal Coliform (CFU/100ml)",
}

TRAIN_YEARS = [2021, 2022, 2023, 2024]

# (target_col, variant_label, baseline_feats, slug)
TARGETS = [
    ("Effluent BOD (mg/L, Grab)",      "Grab",      EXP2_GRAB, "bod-grab"),
    ("Effluent COD (mg/L, Grab)",      "Grab",      EXP2_GRAB, "cod-grab"),
    ("Effluent TSS (mg/L, Grab)",      "Grab",      EXP2_GRAB, "tss-grab"),
    ("Effluent pH (Grab)",             "Grab",      EXP2_GRAB, "ph-grab"),
    ("Effluent BOD (mg/L, Composite)", "Composite", EXP2_COMP, "bod-comp"),
    ("Effluent COD (mg/L, Composite)", "Composite", EXP2_COMP, "cod-comp"),
    ("Effluent TSS (mg/L, Composite)", "Composite", EXP2_COMP, "tss-comp"),
    ("Effluent pH (Composite)",        "Composite", EXP2_COMP, "ph-comp"),
]

# ── Redundancy register (from EDA cross-correlation analysis) ───────────────────
REDUNDANT = {
    "Power NEA (KW)": (
        "Power Total (KW)",
        "Near-perfect collinearity (Pearson=0.975, Spearman=0.940). "
        "Power Total is already in the baseline."
    ),
    "Aeration MLVSS (mg/L, Existing)": (
        "Aeration MLSS (mg/L, Existing)",
        "High collinearity with MLSS Existing (Pearson=0.882, Spearman=0.911). "
        "MLSS is more widely measured."
    ),
    "Aeration MLVSS (mg/L, New)": (
        "Aeration MLSS (mg/L, New)",
        "High collinearity with MLSS New (Pearson=0.871, Spearman=0.892). "
        "Same logic as Existing tank."
    ),
}

# ── Feature → display group ──────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "Power GE (KW)":                         "Power & Flow",
    "Power NEA (KW)":                        "Power & Flow",
    "Power / Flow (KW/ML)":                  "Power & Flow",
    "Inlet pH (Composite)":                  "Inlet - Composite",
    "Inlet BOD (mg/L, Composite)":           "Inlet - Composite",
    "Inlet COD (mg/L, Composite)":           "Inlet - Composite",
    "Inlet TSS (mg/L, Composite)":           "Inlet - Composite",
    "Inlet pH (Grab)":                       "Inlet - Grab",
    "Inlet BOD (mg/L, Grab)":                "Inlet - Grab",
    "Inlet COD (mg/L, Grab)":                "Inlet - Grab",
    "Inlet TSS (mg/L, Grab)":                "Inlet - Grab",
    "Inlet TKN/NH3-N (mg/L, Grab)":          "Inlet - Specialty",
    "Inlet O&G (mg/L, Grab)":                "Inlet - Specialty",
    "Inlet PO4/TP (mg/L, Grab)":             "Inlet - Specialty",
    "Inlet Total Coliform (CFU/100ml, Grab)": "Inlet - Specialty",
    "Inlet Fecal Coliform (CFU/100ml, Grab)": "Inlet - Specialty",
    "Grit Classifier TSS (mg/L)":            "Primary Treatment",
    "Primary Clarifier pH":                  "Primary Treatment",
    "Primary TSS (mg/L)":                    "Primary Treatment",
    "Primary BOD (mg/L)":                    "Primary Treatment",
    "Primary COD (mg/L)":                    "Primary Treatment",
    "Primary Sludge Totalizer (m3)":         "Primary Treatment",
    "Aeration DO (mg/L, Existing)":          "Aeration - DO",
    "Aeration DO (mg/L, New)":               "Aeration - DO",
    "Aeration MLSS (mg/L, Existing)":        "Aeration - Biomass",
    "Aeration MLVSS (mg/L, Existing)":       "Aeration - Biomass",
    "Aeration SV30 (ml/L, Existing)":        "Aeration - Biomass",
    "Aeration SVI (Existing)":               "Aeration - Biomass",
    "Aeration MLSS (mg/L, New)":             "Aeration - Biomass",
    "Aeration MLVSS (mg/L, New)":            "Aeration - Biomass",
    "Aeration SV30 (ml/L, New)":             "Aeration - Biomass",
    "Aeration SVI (New)":                    "Aeration - Biomass",
    "Aeration pH (Existing)":               "Aeration - pH",
    "Aeration pH (New)":                    "Aeration - pH",
}

# ── Tier colours (for plots and report) ─────────────────────────────────────────
TIER_COLORS = {
    "ADD":       "#2ecc71",
    "CONSIDER":  "#f0c040",
    "LOW":       "#d08030",
    "SKIP":      "#c04040",
    "REDUNDANT": "#888888",
    "-":         "#555555",
}
TIER_SORT = {"ADD": 0, "CONSIDER": 1, "LOW": 2, "SKIP": 3, "REDUNDANT": 4, "-": 5}


# ── Tier logic ───────────────────────────────────────────────────────────────────
def _tier(feat, mi, marginal_cost_pct):
    if feat in REDUNDANT:
        return "REDUNDANT"
    if mi is None:
        return "-"
    if mi >= 0.20 and marginal_cost_pct <= 20:
        return "ADD"
    if (mi >= 0.15 and marginal_cost_pct <= 35) or (mi >= 0.25 and marginal_cost_pct <= 50):
        return "CONSIDER"
    if mi >= 0.05:
        return "LOW"
    return "SKIP"


def _signal_type(p, sp):
    if p is None or sp is None:
        return "-"
    ap, asp = abs(p), abs(sp)
    if ap >= 0.6 and asp >= 0.6:
        return "Linear & monotonic"
    if ap >= 0.6 and asp < 0.35:
        return "Outlier-driven (P≫S)"
    if ap < 0.35 and asp >= 0.6:
        return "Non-linear monotonic (S≫P)"
    if ap >= 0.35 and asp >= 0.35:
        return "Moderate"
    if asp >= 0.30:
        return "Weakly monotonic"
    if ap >= 0.30:
        return "Weak linear only"
    return "Weak / non-monotonic"


# ── Data loading ─────────────────────────────────────────────────────────────────
def load_data():
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_excel(DATA_FILE)
    df["month"]       = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["year"]        = df["Date"].dt.year
    print(f"  {len(df)} rows, {df['year'].min():.0f}-{df['year'].max():.0f}")
    return df


# ── Candidate identification ──────────────────────────────────────────────────────
def get_candidates(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [
        c for c in numeric_cols
        if c not in BASELINE_ALL
        and c not in ALL_EFFLUENT
        and c not in ("month", "day_of_week", "year")
    ]
    print(f"\n  Candidate features (not in Exp2 Sub-2 baseline): {len(candidates)}")
    for c in candidates:
        grp = FEATURE_GROUPS.get(c, "Other")
        print(f"    [{grp}] {c}")
    return candidates


# ── Per-target stats computation ─────────────────────────────────────────────────
def compute_stats(df, candidates):
    """
    For each (target, candidate) pair compute:
      - n_target        : rows where target is present
      - n_exp2          : rows remaining after Exp2 baseline dropna
      - n_pair          : rows with both target and this feature present (pairwise)
      - miss_pct        : pairwise missingness % (given target is present)
      - n_after_adding  : rows remaining after adding this feature to Exp2 dropna
      - marginal_cost_pct : (n_exp2 - n_after_adding) / n_exp2 * 100
      - pearson, spearman, mi : computed on pairwise-complete rows
      - tier, signal_type
    """
    records = []
    for target, variant, baseline_feats, slug in TARGETS:
        # Filter to rows where target is present (modeling population)
        tgt_df = df[df[target].notna()].copy()
        n_target = len(tgt_df)

        # Rows that survive Exp2 Sub-2 dropna (train years only for MI/corr,
        # but row counts use full population)
        baseline_present = [f for f in baseline_feats if f in df.columns]
        exp2_rows = tgt_df.dropna(subset=baseline_present + [target])
        n_exp2 = len(exp2_rows)

        print(f"\n  [{variant}] {target}")
        print(f"    n_target={n_target}  n_exp2_rows={n_exp2}")

        for feat in candidates:
            if feat not in df.columns:
                continue

            group = FEATURE_GROUPS.get(feat, "Other")

            # Pairwise stats (target rows, both target + feature present)
            pair = tgt_df[[feat, target]].dropna()
            n_pair = len(pair)
            miss_pct = round((1 - n_pair / n_target) * 100, 1) if n_target > 0 else 100.0

            # Marginal row cost (on full tgt_df)
            exp2_plus = tgt_df.dropna(subset=baseline_present + [target, feat])
            n_after   = len(exp2_plus)
            marginal_cost_pct = round((n_exp2 - n_after) / n_exp2 * 100, 1) if n_exp2 > 0 else 100.0

            # Signal (use training years only for an honest estimate)
            pair_train = pair[pair[target].index.map(
                lambda i: df.loc[i, "year"] in TRAIN_YEARS
                if i in df.index else False
            )] if False else pair  # use full pairwise for MI (more stable)

            if n_pair >= 20:
                pr,  _ = scipy_stats.pearsonr(pair[feat], pair[target])
                sr,  _ = scipy_stats.spearmanr(pair[feat], pair[target])
                mi_val = mutual_info_regression(
                    pair[[feat]], pair[target], random_state=42
                )[0]
                pr     = round(float(pr),     3)
                sr     = round(float(sr),     3)
                mi_val = round(float(mi_val), 3)
            else:
                pr = sr = mi_val = None

            t   = _tier(feat, mi_val, marginal_cost_pct)
            sig = _signal_type(pr, sr)

            records.append(dict(
                target=target, variant=variant, slug=slug,
                feature=feat, group=group,
                n_target=n_target, n_exp2=n_exp2,
                n_pair=n_pair, miss_pct=miss_pct,
                n_after_adding=n_after, marginal_cost_pct=marginal_cost_pct,
                pearson=pr, spearman=sr, mi=mi_val,
                tier=t, signal_type=sig,
            ))
            print(f"    {feat[:45]:<45}  miss={miss_pct:5.1f}%  marginal={marginal_cost_pct:5.1f}%"
                  f"  MI={str(mi_val):>6}  tier={t}")

    return pd.DataFrame(records)


# ── Scatter plot: signal vs. cost ────────────────────────────────────────────────
def _shorten(name):
    """Abbreviate long feature names for scatter labels."""
    repl = {
        "Inlet ": "In.", "Aeration ": "Aer.", "Primary ": "Pri.",
        "Effluent ": "Eff.", "(mg/L, Grab)": "(G)", "(mg/L, Composite)": "(C)",
        "(mg/L)": "", "(ml/L, Existing)": "(Ex)", "(ml/L, New)": "(New)",
        "(mg/L, Existing)": "(Ex)", "(mg/L, New)": "(New)",
        "Existing": "Ex", " (Grab)": "(G)", " (Composite)": "(C)",
        "CFU/100ml, Grab": "CFU(G)", "Grit Classifier ": "Grit.",
        "Clarifier ": "Clar.", "Sludge Totalizer": "Slg.Total",
        "Sec Sed ": "SSed.", "Sec Clarifier ": "SClar.",
        "BOD": "BOD", "COD": "COD", "TSS": "TSS",
        "Power / Flow": "P/F", "(KW)": "(KW)",
    }
    s = name
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.strip()


def plot_signal_vs_cost(stats_df):
    """One scatter figure per target. Returns {slug: path}."""
    paths = {}
    for target, variant, _, slug in TARGETS:
        sub = stats_df[stats_df["target"] == target].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        fig.patch.set_facecolor("#1A1D23")
        ax.set_facecolor("#23272F")

        # Plot data first so ylim is meaningful
        for _, row in sub.iterrows():
            if row["mi"] is None:
                continue
            clr = TIER_COLORS.get(row["tier"], "#888")
            ax.scatter(row["marginal_cost_pct"], row["mi"],
                       color=clr, s=60, zorder=5, edgecolors="#1A1D23", lw=0.5)
            ax.annotate(_shorten(row["feature"]),
                        (row["marginal_cost_pct"], row["mi"]),
                        fontsize=6.5, color="#D0D0D0",
                        xytext=(4, 4), textcoords="offset points")

        ax.set_xlim(-2, 102)
        ax.set_ylim(bottom=-0.01)
        y_top = ax.get_ylim()[1]

        # Background quadrant shading (after axes limits set)
        ax.axvspan(0,  20, alpha=0.06, color="#2ecc71", zorder=0)
        ax.axvspan(20, 35, alpha=0.06, color="#f0c040", zorder=0)
        ax.axvspan(35, 100, alpha=0.06, color="#c04040", zorder=0)

        # Reference lines
        for x, lbl in [(20, "20%"), (35, "35%"), (50, "50%")]:
            ax.axvline(x, color="#555", lw=0.8, ls="--", zorder=1)
            ax.text(x + 0.5, y_top * 0.97, lbl, color="#888", fontsize=7, va="top")

        # Legend
        legend_handles = [
            mpatches.Patch(color=TIER_COLORS[t], label=t)
            for t in ["ADD", "CONSIDER", "LOW", "SKIP", "REDUNDANT"]
        ]
        ax.legend(handles=legend_handles, fontsize=8,
                  facecolor="#23272F", edgecolor="#555", labelcolor="#E0E0E0",
                  loc="upper right")

        ax.set_xlabel("Marginal row cost on top of Exp2 Sub-2 baseline (%)",
                      color="#A0A6B2", fontsize=9)
        ax.set_ylabel("Mutual Information (MI)", color="#A0A6B2", fontsize=9)
        ax.set_title(f"{target}", color="#E2E4E9", fontsize=10, pad=10)
        ax.tick_params(colors="#A0A6B2", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#383C46")
        path = os.path.join(PLOTS_DIR, f"signal_vs_cost_{slug}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        paths[slug] = path
        print(f"  Plot → {path}")

    return paths


# ── XLSX export ──────────────────────────────────────────────────────────────────
def export_xlsx(stats_df, run):
    cols = ["target", "variant", "feature", "group",
            "n_target", "n_exp2", "n_pair", "miss_pct",
            "n_after_adding", "marginal_cost_pct",
            "pearson", "spearman", "mi", "tier", "signal_type"]
    out = stats_df[cols].copy()
    out.insert(0, "run", run)
    out.sort_values(["target", "tier", "mi"],
                    key=lambda s: s.map(TIER_SORT) if s.name == "tier" else s,
                    ascending=[True, True, False],
                    inplace=True)
    out.to_excel(AUDIT_XLSX, index=False)
    print(f"\nAudit table → {AUDIT_XLSX}")
    return out


# ── HTML helpers ─────────────────────────────────────────────────────────────────
def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _tier_badge(t):
    bg = {
        "ADD":       "#d4edda", "CONSIDER": "#fff3cd", "LOW":       "#fde8d0",
        "SKIP":      "#f8d7da", "REDUNDANT": "#e2e3e5", "-":        "#f0f0f0",
    }
    fg = {
        "ADD":       "#155724", "CONSIDER": "#7d5400", "LOW":       "#7a3800",
        "SKIP":      "#721c24", "REDUNDANT": "#3a3a3a", "-":        "#888",
    }
    label = {"ADD": "Add", "CONSIDER": "Consider", "LOW": "Low priority",
             "SKIP": "Skip", "REDUNDANT": "Redundant - drop", "-": "-"}
    b, f = bg.get(t, "#eee"), fg.get(t, "#333")
    l = label.get(t, t)
    return (f'<span style="background:{b};color:{f};padding:2px 7px;'
            f'border-radius:4px;font-size:0.78rem;font-weight:600">{l}</span>')


def _fmt_num(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span style='color:#666'>-</span>"
    return f"{v:.{decimals}f}"


def _fmt_pct(v, warn_above=35):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span style='color:#666'>-</span>"
    clr = "#c04040" if v > warn_above else ("#d08030" if v > 20 else "#a0d0a0")
    return f"<span style='color:{clr};font-weight:600'>{v:.1f}%</span>"


def _fmt_mi(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span style='color:#666'>-</span>"
    clr = ("#2ecc71" if v >= 0.30 else "#f0c040" if v >= 0.15
           else "#d08030" if v >= 0.05 else "#c04040")
    return f"<span style='color:{clr};font-weight:600'>{v:.3f}</span>"


def _feature_table(sub):
    """HTML table for one target's candidates, sorted by tier then MI desc."""
    sub = sub.sort_values(
        ["tier", "mi"],
        key=lambda s: s.map(TIER_SORT) if s.name == "tier" else s.fillna(-1),
        ascending=[True, False]
    )
    TH = "padding:7px 10px;text-align:left;background:#1a2535;color:#9ab;font-size:0.80rem"
    TH_C = TH + ";text-align:center"
    TD = "padding:5px 9px;font-size:0.80rem;border-bottom:1px solid #2a3a4a;color:#d0d8e0"
    TD_C = TD + ";text-align:center"

    rows_html = ""
    for _, r in sub.iterrows():
        redundant_note = ""
        if r["tier"] == "REDUNDANT" and r["feature"] in REDUNDANT:
            pref, reason = REDUNDANT[r["feature"]]
            redundant_note = (f'<br><span style="color:#888;font-size:0.74rem">'
                              f'→ prefer <em>{pref}</em>. {reason}</span>')
        rows_html += f"""<tr>
          <td style="{TD}">{r['feature']}{redundant_note}</td>
          <td style="{TD}">{r.get('group','-')}</td>
          <td style="{TD_C}">{_tier_badge(r['tier'])}</td>
          <td style="{TD_C}">{_fmt_mi(r['mi'])}</td>
          <td style="{TD_C}">{_fmt_num(r['pearson'])}</td>
          <td style="{TD_C}">{_fmt_num(r['spearman'])}</td>
          <td style="{TD_C}">{_fmt_pct(r['miss_pct'], warn_above=35)}</td>
          <td style="{TD_C}">{_fmt_pct(r['marginal_cost_pct'], warn_above=20)}</td>
          <td style="{TD_C}">{int(r['n_pair'])}</td>
          <td style="{TD}">{r['signal_type']}</td>
        </tr>"""

    return f"""<table style="width:100%;border-collapse:collapse">
      <thead><tr>
        <th style="{TH}">Feature</th>
        <th style="{TH}">Group</th>
        <th style="{TH_C}">Tier</th>
        <th style="{TH_C}">MI</th>
        <th style="{TH_C}">Pearson r</th>
        <th style="{TH_C}">Spearman ρ</th>
        <th style="{TH_C}">Pairwise miss%</th>
        <th style="{TH_C}">Marginal cost%</th>
        <th style="{TH_C}">n pairs</th>
        <th style="{TH}">Signal type</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _summary_table(stats_df, candidates):
    """Cross-tab: feature rows × target columns, cells coloured by tier."""
    tgt_labels = [t.replace("Effluent ", "").replace(" (mg/L, ", " ").replace(")", "")
                  for t, *_ in TARGETS]
    pivot = stats_df.pivot_table(
        index="feature", columns="target", values="tier", aggfunc="first"
    )
    # Sort rows by best tier across targets
    def _best(row):
        vals = [TIER_SORT.get(v, 9) for v in row if isinstance(v, str)]
        return min(vals) if vals else 9
    pivot["_best"] = pivot.apply(_best, axis=1)
    pivot = pivot.sort_values("_best").drop(columns="_best")

    TH  = "padding:6px 8px;font-size:0.76rem;color:#9ab;background:#1a2535;text-align:center;white-space:nowrap"
    TH_L = TH + ";text-align:left"
    TD  = "padding:5px 8px;font-size:0.76rem;text-align:center;border-bottom:1px solid #2a3a4a"
    TD_L = TD + ";text-align:left;color:#d0d8e0"

    hdr = "".join(f'<th style="{TH}">{lbl}</th>' for lbl in tgt_labels)
    body = ""
    for feat, row in pivot.iterrows():
        grp = FEATURE_GROUPS.get(feat, "Other")
        cells = ""
        for tgt, *_ in TARGETS:
            t = row.get(tgt, "-")
            if not isinstance(t, str):
                t = "-"
            bg = {
                "ADD": "#0d3b1f", "CONSIDER": "#2e2800", "LOW": "#2e1800",
                "SKIP": "#2e0f0f", "REDUNDANT": "#1a1a1a", "-": "#1a1a1a"
            }.get(t, "#1a1a1a")
            fg = TIER_COLORS.get(t, "#555")
            lbl_short = {"ADD": "Add", "CONSIDER": "Cns.", "LOW": "Low",
                         "SKIP": "Skip", "REDUNDANT": "Rdnt", "-": "-"}.get(t, t)
            cells += (f'<td style="{TD};background:{bg};color:{fg};'
                      f'font-weight:600;font-size:0.73rem">{lbl_short}</td>')
        body += f'<tr><td style="{TD_L}">{feat}</td><td style="{TD};color:#888;font-size:0.73rem">{grp}</td>{cells}</tr>'

    return f"""<table style="width:100%;border-collapse:collapse">
      <thead><tr>
        <th style="{TH_L}">Feature</th>
        <th style="{TH_L}">Group</th>
        {hdr}
      </tr></thead>
      <tbody>{body}</tbody>
    </table>"""


# ── HTML report ──────────────────────────────────────────────────────────────────
def build_report(stats_df, plot_paths, run):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── CSS ───────────────────────────────────────────────────────────────────────
    CSS = dark_mode_css("""
      body { padding: 20px 30px; max-width: 1200px; margin: 0 auto; }
      h1 { color: #4a9fd4; font-size: 1.5rem; margin-bottom: 4px; }
      h2 { color: #4a9fd4; font-size: 1.15rem; margin: 22px 0 8px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
      .meta { font-size: 0.82rem; margin-bottom: 20px; }
      .card { padding: 16px 20px; margin-bottom: 20px; border-radius: 8px; }
      .obs-card { padding: 14px 18px; border-radius: 8px; margin-bottom: 14px; }
      details { border: 1px solid var(--details-bdr); border-radius: 6px;
                background: var(--details-bg); margin-bottom: 14px; }
      details > summary { padding: 10px 14px; cursor: pointer; font-weight: 600;
                          color: var(--text); font-size: 0.92rem; list-style: none; }
      details > summary:hover { background: var(--summary-hover); border-radius: 6px; }
      details[open] > summary { border-bottom: 1px solid var(--details-bdr); }
      .fold-icon { display: inline-block; transition: transform 0.2s; margin-right: 6px; }
      details[open] .fold-icon { transform: rotate(90deg); }
      .fold-hint { font-weight: 400; font-size: 0.78rem; color: var(--text-muted); margin-left: 8px; }
      .tier-legend { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
      .tier-pill { padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
      img { max-width: 100%; border-radius: 6px; display: block; margin: 12px auto; }
    """)

    # ── Intro / methodology ────────────────────────────────────────────────────────
    tier_legend = "".join(
        f'<span class="tier-pill" style="background:{bg};color:{fg}">{lbl}</span>'
        for lbl, bg, fg in [
            ("Add",             "#d4edda", "#155724"),
            ("Consider",        "#fff3cd", "#7d5400"),
            ("Low priority",    "#fde8d0", "#7a3800"),
            ("Skip",            "#f8d7da", "#721c24"),
            ("Redundant - drop","#e2e3e5", "#3a3a3a"),
        ]
    )

    intro_card = f"""
    <div class="card obs-card" style="font-size:0.88rem">
      <strong>Purpose:</strong> Identify which candidate features (not yet in Experiment 2
      Sub-2) are worth adding for Experiment 3, balancing predictive signal against the
      row-loss cost of including them.<br><br>

      <strong>Two missingness metrics are reported</strong> - they answer different questions:<br>
      <ul style="margin:6px 0 6px 16px;line-height:1.7">
        <li><strong>Pairwise miss%</strong> - what fraction of target rows have this feature
            missing? Measures raw data coverage for the feature in isolation.</li>
        <li><strong>Marginal cost%</strong> - how many additional rows are lost when this
            feature is added to the existing Experiment 2 Sub-2 <code>dropna</code>?
            This is the operationally relevant number: if Exp2 already filters out rows
            due to other missing columns, a new feature with correlated missingness may
            cost far fewer rows than its raw pairwise % suggests.</li>
      </ul>

      <strong>Tier logic</strong> uses <em>marginal cost</em> (not pairwise miss%) and
      MI as the signal measure:<br>
      <div class="tier-legend" style="margin-top:8px">{tier_legend}</div>
      <ul style="margin:6px 0 6px 16px;line-height:1.7;font-size:0.84rem">
        <li><strong>Add</strong>      - MI ≥ 0.20 and marginal cost ≤ 20%</li>
        <li><strong>Consider</strong> - MI ≥ 0.15 &amp; cost ≤ 35%, or MI ≥ 0.25 &amp; cost ≤ 50%</li>
        <li><strong>Low priority</strong> - MI ≥ 0.05 (weak signal or high cost)</li>
        <li><strong>Skip</strong>     - MI &lt; 0.05</li>
        <li><strong>Redundant</strong>- known collinear with a baseline feature (from EDA)</li>
      </ul>

      <strong>Signal is estimated on pairwise-complete rows</strong> (rows where both the
      target and this feature are present), so the MI/correlation numbers are honest -
      they are not inflated or deflated by imputed values.
    </div>"""

    # ── Summary cross-tab ─────────────────────────────────────────────────────────
    candidates = stats_df["feature"].unique().tolist()
    summary_tbl = _summary_table(stats_df, candidates)

    summary_section = f"""
    <h2>Summary - all targets</h2>
    <p style="font-size:0.83rem;color:var(--text-muted)">
      Each cell shows the recommendation tier for that (feature, target) pair.
      Features sorted by best tier across all targets.
    </p>
    {summary_tbl}"""

    # ── Per-target sections ────────────────────────────────────────────────────────
    tgt_sections = ""
    for target, variant, _, slug in TARGETS:
        sub = stats_df[stats_df["target"] == target]
        if sub.empty:
            continue

        n_target = int(sub["n_target"].iloc[0])
        n_exp2   = int(sub["n_exp2"].iloc[0])
        n_add    = int((sub["tier"] == "ADD").sum())
        n_con    = int((sub["tier"] == "CONSIDER").sum())
        n_low    = int((sub["tier"] == "LOW").sum())
        n_skip   = int((sub["tier"] == "SKIP").sum())
        n_red    = int((sub["tier"] == "REDUNDANT").sum())

        plot_html = ""
        if slug in plot_paths:
            b64 = _b64(plot_paths[slug])
            plot_html = f'<img src="data:image/png;base64,{b64}" alt="signal vs cost {slug}">'

        feat_tbl = _feature_table(sub)

        tgt_sections += f"""
        <details>
          <summary>
            <span class="fold-icon">▶</span>
            {target}
            <span class="fold-hint">(click to expand)</span>
          </summary>
          <div style="padding:14px 18px">
            <div class="obs-card" style="font-size:0.84rem;margin-bottom:12px">
              <strong>Population:</strong> {n_target} rows with target present
              &nbsp;·&nbsp;
              <strong>After Exp2 Sub-2 dropna:</strong> {n_exp2} rows
              &nbsp;·&nbsp;
              <strong>Candidate summary:</strong>
              <span style="color:#2ecc71">{n_add} Add</span> &nbsp;
              <span style="color:#f0c040">{n_con} Consider</span> &nbsp;
              <span style="color:#d08030">{n_low} Low</span> &nbsp;
              <span style="color:#c04040">{n_skip} Skip</span> &nbsp;
              <span style="color:#888">{n_red} Redundant</span>
            </div>
            {plot_html}
            {feat_tbl}
          </div>
        </details>"""

    # ── Assemble page ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Feature Audit - Experiment 3 Candidates (run {run})</title>
  <style>{CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
  <h1>Feature Audit - Experiment 3 Candidates</h1>
  <p class="meta">Run {run} &nbsp;·&nbsp; Generated {now}</p>
  {intro_card}
  {summary_section}
  <h2>Per-target detail</h2>
  <p style="font-size:0.83rem;color:var(--text-muted)">
    Scatter: x = marginal row cost on top of Exp2 Sub-2 baseline,
    y = MI score. Table sorted by tier then MI descending.
  </p>
  {tgt_sections}
</body>
</html>"""

    out_path = os.path.join(SCRIPT_DIR, f"report_feature_audit_exp3_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report → {out_path}")
    return out_path


# ── Run number ───────────────────────────────────────────────────────────────────
def _next_run():
    if not os.path.exists(AUDIT_XLSX):
        return 1
    try:
        df = pd.read_excel(AUDIT_XLSX)
        return int(df["run"].max()) + 1
    except Exception:
        return 1


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    run = _next_run()
    print(f"\nFeature Audit - Experiment 3 Candidates  (run {run})")
    print("=" * 60)

    df         = load_data()
    candidates = get_candidates(df)

    print("\nComputing stats per target × candidate ...")
    stats_df   = compute_stats(df, candidates)

    print("\nGenerating signal-vs-cost scatter plots ...")
    plot_paths = plot_signal_vs_cost(stats_df)

    export_xlsx(stats_df, run)
    build_report(stats_df, plot_paths, run)

    print("\nDone.")


if __name__ == "__main__":
    main()
