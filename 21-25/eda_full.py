"""
Full EDA for All_Years_Full.xlsx.

Reads 21-25/All_Years_Full.xlsx (60 columns, 1918 rows) and generates:

  1.  missing_new_columns     — coverage heatmap for the 24 new columns by month
  2.  pearson_vs_spearman     — side-by-side feature correlations vs effluent targets
  3.  mutual_information      — MI scores per feature per target (bar charts)
  4.  feature_scatter_grids   — top features × each effluent target scatter grids
  5.  aeration_timeseries     — DO / MLSS / SVI time series for both tanks
  6.  cross_stage_heatmap     — all features × effluent targets (Pearson + Spearman)
  7.  stage_removal           — BOD/COD/TSS concentrations at each treatment stage
  8.  do_vs_effluent          — aeration DO threshold vs effluent BOD / COD
  9.  svi_mlss_vs_tss         — SVI / MLSS vs effluent TSS scatter
  10. linearity_residuals     — Ridge residuals: linearity diagnostic per target

Outputs: 21-25/eda_full_plots/  (PNG files)
         21-25/eda_full_report.html
"""

import base64
import os
import sys as _sys
import warnings

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling"))
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EXCEL_IN  = os.path.join(BASE_DIR, "All_Years_Full.xlsx")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_full_plots")
REPORT    = os.path.join(BASE_DIR, "eda_full_report.html")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Year colour palette (consistent with modeling scripts) ────────────────────
YEAR_COLOURS = {
    2021: "#2171B5",
    2022: "#74C476",
    2023: "#238B45",
    2024: "#FD8D3C",
    2025: "#D94801",
}

# ── Column groups ──────────────────────────────────────────────────────────────
GRAB_TARGETS = [
    "Effluent BOD (mg/L, Grab)",
    "Effluent COD (mg/L, Grab)",
    "Effluent TSS (mg/L, Grab)",
    "Effluent pH (Grab)",
]
COMP_TARGETS = [
    "Effluent BOD (mg/L, Composite)",
    "Effluent COD (mg/L, Composite)",
    "Effluent TSS (mg/L, Composite)",
    "Effluent pH (Composite)",
]
ALL_TARGETS = GRAB_TARGETS + COMP_TARGETS

NEW_COLUMNS = [
    "Inlet TKN/NH3-N (mg/L, Grab)",
    "Inlet O&G (mg/L, Grab)",
    "Inlet PO4/TP (mg/L, Grab)",
    "Inlet Total Coliform (CFU/100ml, Grab)",
    "Inlet Fecal Coliform (CFU/100ml, Grab)",
    "Grit Classifier TSS (mg/L)",
    "Primary Clarifier pH",
    "Primary Sludge Totalizer (m3)",
    "Aeration pH (Existing)",
    "Aeration DO (mg/L, Existing)",
    "Aeration MLSS (mg/L, Existing)",
    "Aeration MLVSS (mg/L, Existing)",
    "Aeration SV30 (ml/L, Existing)",
    "Aeration SVI (Existing)",
    "Aeration pH (New)",
    "Aeration DO (mg/L, New)",
    "Aeration MLSS (mg/L, New)",
    "Aeration MLVSS (mg/L, New)",
    "Aeration SV30 (ml/L, New)",
    "Aeration SVI (New)",
    "Effluent O&G (mg/L)",
    "Effluent NH3-N (mg/L)",
    "Effluent Total Coliform (CFU/100ml)",
    "Effluent Fecal Coliform (CFU/100ml)",
]

# Short labels for axes — written to be self-explanatory without domain knowledge
SHORT = {
    # Power / Operational
    "Power GE (KW)":                          "Power — Gas Engine (KW)",
    "Power NEA (KW)":                         "Power — NEA Grid (KW)",
    "Power Total (KW)":                       "Power — Total (KW)",
    "Power / Flow (KW/ML)":                   "Power per Flow (KW/ML)",
    "Flow (MLD)":                             "Flow (MLD)",
    # Inlet — grab
    "Inlet pH (Grab)":                        "Inlet pH (Grab)",
    "Inlet BOD (mg/L, Grab)":                "Inlet BOD (Grab)",
    "Inlet COD (mg/L, Grab)":                "Inlet COD (Grab)",
    "Inlet TSS (mg/L, Grab)":                "Inlet TSS (Grab)",
    "Inlet TKN/NH3-N (mg/L, Grab)":          "Inlet TKN/NH3 (Grab)",
    "Inlet O&G (mg/L, Grab)":                "Inlet Oil & Grease (Grab)",
    "Inlet PO4/TP (mg/L, Grab)":             "Inlet PO4/TP (Grab)",
    "Inlet Total Coliform (CFU/100ml, Grab)": "Inlet Total Coliform (Grab)",
    "Inlet Fecal Coliform (CFU/100ml, Grab)": "Inlet Fecal Coliform (Grab)",
    # Inlet — composite
    "Inlet pH (Composite)":                   "Inlet pH (Composite)",
    "Inlet BOD (mg/L, Composite)":           "Inlet BOD (Composite)",
    "Inlet COD (mg/L, Composite)":           "Inlet COD (Composite)",
    "Inlet TSS (mg/L, Composite)":           "Inlet TSS (Composite)",
    # Grit
    "Grit Classifier TSS (mg/L)":            "Grit Classifier TSS",
    # Primary Clarifier
    "Primary Clarifier pH":                   "Primary Clarifier pH",
    "Primary TSS (mg/L)":                     "Primary Clarifier TSS",
    "Primary BOD (mg/L)":                     "Primary Clarifier BOD",
    "Primary COD (mg/L)":                     "Primary Clarifier COD",
    "Primary Sludge Totalizer (m3)":          "Primary Sludge Volume (m3)",
    # Secondary Clarifier
    "Sec Clarifier pH":                       "Sec Clarifier pH",
    "Sec Clarifier TSS (mg/L)":              "Sec Clarifier TSS",
    "Sec Clarifier BOD (mg/L)":              "Sec Clarifier BOD",
    "Sec Clarifier COD (mg/L)":              "Sec Clarifier COD",
    "Sec Clarifier RAS":                      "Sec Clarifier RAS",
    # Secondary Sedimentation
    "Sec Sed pH":                             "Sec Sedimentation pH",
    "Sec Sed TSS (mg/L)":                    "Sec Sedimentation TSS",
    "Sec Sed BOD (mg/L)":                    "Sec Sedimentation BOD",
    "Sec Sed COD (mg/L)":                    "Sec Sedimentation COD",
    "Sec Sed RAS (New)":                     "Sec Sedimentation RAS",
    # Aeration — existing tank
    "Aeration pH (Existing)":                "Aeration pH (Existing Tank)",
    "Aeration DO (mg/L, Existing)":          "Aeration DO (Existing Tank)",
    "Aeration MLSS (mg/L, Existing)":        "Aeration MLSS (Existing Tank)",
    "Aeration MLVSS (mg/L, Existing)":       "Aeration MLVSS (Existing Tank)",
    "Aeration SV30 (ml/L, Existing)":        "Aeration SV30 (Existing Tank)",
    "Aeration SVI (Existing)":               "Aeration SVI (Existing Tank)",
    # Aeration — new tank
    "Aeration pH (New)":                     "Aeration pH (New Tank)",
    "Aeration DO (mg/L, New)":               "Aeration DO (New Tank)",
    "Aeration MLSS (mg/L, New)":             "Aeration MLSS (New Tank)",
    "Aeration MLVSS (mg/L, New)":            "Aeration MLVSS (New Tank)",
    "Aeration SV30 (ml/L, New)":             "Aeration SV30 (New Tank)",
    "Aeration SVI (New)":                    "Aeration SVI (New Tank)",
    # Effluent — grab
    "Effluent pH (Grab)":                    "Effluent pH (Grab)",
    "Effluent BOD (mg/L, Grab)":             "Effluent BOD (Grab)",
    "Effluent COD (mg/L, Grab)":             "Effluent COD (Grab)",
    "Effluent TSS (mg/L, Grab)":             "Effluent TSS (Grab)",
    "Effluent FRC (mg/L)":                   "Effluent Free Residual Chlorine",
    "Effluent O&G (mg/L)":                   "Effluent Oil & Grease",
    "Effluent NH3-N (mg/L)":                "Effluent NH3-N",
    "Effluent Total Coliform (CFU/100ml)":   "Effluent Total Coliform",
    "Effluent Fecal Coliform (CFU/100ml)":   "Effluent Fecal Coliform",
    # Effluent — composite
    "Effluent pH (Composite)":               "Effluent pH (Composite)",
    "Effluent BOD (mg/L, Composite)":        "Effluent BOD (Composite)",
    "Effluent COD (mg/L, Composite)":        "Effluent COD (Composite)",
    "Effluent TSS (mg/L, Composite)":        "Effluent TSS (Composite)",
}

def s(col):
    return SHORT.get(col, col)


# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}.png")
    return path


def feature_cols(df, exclude_targets=True):
    """All numeric non-date, non-target columns."""
    drop = {"Date"}
    if exclude_targets:
        drop |= set(ALL_TARGETS) | {"Effluent FRC (mg/L)", "Effluent O&G (mg/L)",
                                     "Effluent NH3-N (mg/L)",
                                     "Effluent Total Coliform (CFU/100ml)",
                                     "Effluent Fecal Coliform (CFU/100ml)"}
    return [c for c in df.select_dtypes(include="number").columns if c not in drop]


def pearson_spearman(df, features, target):
    """Return DataFrame with Pearson and Spearman r for each feature vs target."""
    rows = []
    t = df[target].dropna()
    for f in features:
        sub = df[[f, target]].dropna()
        if len(sub) < 10:
            rows.append({"feature": f, "pearson": np.nan, "spearman": np.nan})
            continue
        pr, _ = stats.pearsonr(sub[f], sub[target])
        sr, _ = stats.spearmanr(sub[f], sub[target])
        rows.append({"feature": f, "pearson": pr, "spearman": sr})
    return pd.DataFrame(rows).set_index("feature")


# ── Chart 1: Missing data — all columns ───────────────────────────────────────

# Ordered column groups for the missing-data heatmap rows
_COL_SECTIONS = [
    ("Power & Operational", [
        "Power — Gas Engine (KW)", "Power — NEA Grid (KW)", "Power — Total (KW)",
        "Power per Flow (KW/ML)", "Flow (MLD)",
    ]),
    ("Inlet (Grab)", [
        "Inlet pH (Grab)", "Inlet BOD (Grab)", "Inlet COD (Grab)", "Inlet TSS (Grab)",
        "Inlet TKN/NH3 (Grab)", "Inlet Oil & Grease (Grab)", "Inlet PO4/TP (Grab)",
        "Inlet Total Coliform (Grab)", "Inlet Fecal Coliform (Grab)",
    ]),
    ("Inlet (Composite)", [
        "Inlet pH (Composite)", "Inlet BOD (Composite)",
        "Inlet COD (Composite)", "Inlet TSS (Composite)",
    ]),
    ("Grit & Primary", [
        "Grit Classifier TSS", "Primary Clarifier pH", "Primary Clarifier TSS",
        "Primary Clarifier BOD", "Primary Clarifier COD", "Primary Sludge Volume (m3)",
    ]),
    ("Sec Clarifier", [
        "Sec Clarifier pH", "Sec Clarifier TSS", "Sec Clarifier BOD",
        "Sec Clarifier COD", "Sec Clarifier RAS",
    ]),
    ("Sec Sedimentation", [
        "Sec Sedimentation pH", "Sec Sedimentation TSS", "Sec Sedimentation BOD",
        "Sec Sedimentation COD", "Sec Sedimentation RAS",
    ]),
    ("Aeration — Existing Tank", [
        "Aeration pH (Existing Tank)", "Aeration DO (Existing Tank)",
        "Aeration MLSS (Existing Tank)", "Aeration MLVSS (Existing Tank)",
        "Aeration SV30 (Existing Tank)", "Aeration SVI (Existing Tank)",
    ]),
    ("Aeration — New Tank", [
        "Aeration pH (New Tank)", "Aeration DO (New Tank)",
        "Aeration MLSS (New Tank)", "Aeration MLVSS (New Tank)",
        "Aeration SV30 (New Tank)", "Aeration SVI (New Tank)",
    ]),
    ("Effluent (Grab)", [
        "Effluent pH (Grab)", "Effluent BOD (Grab)", "Effluent COD (Grab)",
        "Effluent TSS (Grab)", "Effluent Free Residual Chlorine",
        "Effluent Oil & Grease", "Effluent NH3-N",
        "Effluent Total Coliform", "Effluent Fecal Coliform",
    ]),
    ("Effluent (Composite)", [
        "Effluent pH (Composite)", "Effluent BOD (Composite)",
        "Effluent COD (Composite)", "Effluent TSS (Composite)",
    ]),
]


def plot_missing_all(df):
    print("1. Missing data — all columns...")

    # Build short-name → original-name reverse map
    rev = {v: k for k, v in SHORT.items()}

    # Monthly coverage for every numeric column
    num_cols = df.select_dtypes(include="number").columns.tolist()
    tmp = df[["Date"] + num_cols].copy()
    tmp["YearMonth"] = tmp["Date"].dt.to_period("M")
    monthly = tmp.groupby("YearMonth")[num_cols].apply(
        lambda g: g.notna().mean() * 100
    )
    monthly.index = monthly.index.astype(str)

    # Build ordered row list: short_name → original_col
    ordered_rows = []   # (display_label, original_col)
    section_boundaries = []   # (y_position, section_name)
    pos = 0
    for sec_name, short_names in _COL_SECTIONS:
        section_boundaries.append((pos, sec_name))
        for sn in short_names:
            orig = rev.get(sn)
            if orig and orig in monthly.columns:
                ordered_rows.append((sn, orig))
                pos += 1

    labels  = [r[0] for r in ordered_rows]
    orig_cols = [r[1] for r in ordered_rows]
    data = monthly[orig_cols].T   # shape: features × months
    data.index = labels

    n_rows = len(labels)
    fig, ax = plt.subplots(figsize=(22, max(10, n_rows * 0.32)))
    sns.heatmap(
        data, ax=ax, cmap="YlGn", vmin=0, vmax=100,
        linewidths=0.25, linecolor="#e8e8e8",
        cbar_kws={"label": "% days with data", "shrink": 0.5},
        yticklabels=labels,
    )
    ax.set_title("Data coverage (%) — all columns, by month  (darker green = more complete)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("")

    # Draw section dividers
    for y_pos, sec_name in section_boundaries:
        if y_pos > 0:
            ax.axhline(y_pos, color="white", linewidth=2.5)
        ax.text(-0.5, y_pos + 0.5, sec_name,
                ha="right", va="top", fontsize=7.5, color="#444",
                fontweight="bold", transform=ax.get_yaxis_transform())

    # Sparse x-axis labels — every 6 months
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    step = max(1, len(xlabels) // 12)
    ax.set_xticks(np.arange(0, len(xlabels), step) + 0.5)
    ax.set_xticklabels(xlabels[::step], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return save(fig, "01_missing_all_columns")


# ── Chart 2: Pearson vs Spearman (one figure per target) ─────────────────────

def _pearson_spearman_one(df, target, feats, filename):
    """Single horizontal bar chart for one target. Returns saved path."""
    corr = pearson_spearman(df, feats, target).dropna()
    corr["abs_sp"] = corr["spearman"].abs()
    corr = corr.sort_values("abs_sp", ascending=True)
    short_feats = [s(f) for f in corr.index]

    n = len(corr)
    bar_h = 0.38
    fig, ax = plt.subplots(figsize=(15, max(7, n * 0.30)))

    y = np.arange(n)
    ax.barh(y + bar_h / 2, corr["pearson"],  bar_h, color="#2171B5",
            alpha=0.85, label="Pearson r  (linear correlation)")
    ax.barh(y - bar_h / 2, corr["spearman"], bar_h, color="#FD8D3C",
            alpha=0.85, label="Spearman r  (rank / monotonic correlation)")

    ax.set_yticks(y)
    ax.set_yticklabels(short_feats, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation coefficient  (−1 = perfect negative, +1 = perfect positive)",
                  fontsize=9)
    ax.set_title(f"Pearson vs Spearman — {s(target)}\n"
                 "Features ranked by |Spearman r| (strongest at top)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Annotate top-5 Spearman values
    top5_thresh = corr["abs_sp"].nlargest(5).min()
    for i, (pr, sr, abs_sp) in enumerate(
            zip(corr["pearson"], corr["spearman"], corr["abs_sp"])):
        if abs_sp >= top5_thresh:
            xp = pr + 0.02 if pr >= 0 else pr - 0.02
            xs = sr + 0.02 if sr >= 0 else sr - 0.02
            ha_p = "left" if pr >= 0 else "right"
            ha_s = "left" if sr >= 0 else "right"
            ax.text(xp, i + bar_h / 2, f"{pr:.2f}", va="center",
                    fontsize=7, ha=ha_p, color="#1a3a6e")
            ax.text(xs, i - bar_h / 2, f"{sr:.2f}", va="center",
                    fontsize=7, ha=ha_s, color="#7a3000")

    plt.tight_layout()
    return save(fig, filename)


def plot_pearson_vs_spearman(df):
    print("2. Pearson vs Spearman...")
    feats = feature_cols(df)
    grab_paths, comp_paths = [], []

    for target in GRAB_TARGETS:
        slug = (target.replace(" ", "_").replace("/", "").replace("(", "")
                      .replace(")", "").replace(",", ""))[:30]
        grab_paths.append(_pearson_spearman_one(df, target, feats,
                                                f"02_pearson_spearman_{slug}"))
    for target in COMP_TARGETS:
        slug = (target.replace(" ", "_").replace("/", "").replace("(", "")
                      .replace(")", "").replace(",", ""))[:30]
        comp_paths.append(_pearson_spearman_one(df, target, feats,
                                                f"02_pearson_spearman_comp_{slug}"))
    return {"grab": grab_paths, "comp": comp_paths}


# ── Chart 3: Mutual Information (one figure per target) ───────────────────────

def _mi_one(df, target, feats, filename):
    """Single MI bar chart for one target. Returns (path, mi_series)."""
    sub = df[feats + [target]].dropna()
    if len(sub) < 20:
        return None, None
    X = sub[feats].values
    y = sub[target].values
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=feats).sort_values(ascending=True)

    short_feats = [s(f) for f in mi_series.index]
    q75 = mi_series.quantile(0.75)
    colors = ["#D94801" if v >= q75 else "#2171B5" for v in mi_series.values]

    n = len(mi_series)
    fig, ax = plt.subplots(figsize=(15, max(7, n * 0.30)))
    ax.barh(range(n), mi_series.values, color=colors, alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_feats, fontsize=8)
    ax.set_title(f"Mutual Information scores — {s(target)}\n"
                 "Features ranked by MI score  (red bars = top 25%)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mutual Information score  (higher = more predictive, regardless of relationship shape)",
                  fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Annotate top-5
    top5_thresh = mi_series.nlargest(5).min()
    for i, v in enumerate(mi_series.values):
        if v >= top5_thresh:
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7.5, color="#6B0000")

    plt.tight_layout()
    return save(fig, filename), mi_series


def plot_mutual_information(df):
    print("3. Mutual information...")
    feats = feature_cols(df)
    grab_paths, comp_paths = [], []
    all_mi = {}

    for target in GRAB_TARGETS:
        slug = (target.replace(" ", "_").replace("/", "").replace("(", "")
                      .replace(")", "").replace(",", ""))[:30]
        path, mi_s = _mi_one(df, target, feats, f"03_mi_{slug}")
        grab_paths.append(path)
        if mi_s is not None:
            all_mi[target] = mi_s

    for target in COMP_TARGETS:
        slug = (target.replace(" ", "_").replace("/", "").replace("(", "")
                      .replace(")", "").replace(",", ""))[:30]
        path, mi_s = _mi_one(df, target, feats, f"03_mi_comp_{slug}")
        comp_paths.append(path)
        if mi_s is not None:
            all_mi[target] = mi_s

    return {"grab": grab_paths, "comp": comp_paths, "mi_data": all_mi}


# ── Chart 4: Feature vs target scatter grids ──────────────────────────────────

def plot_scatter_grids(df, all_mi):
    print("4. Feature scatter grids...")
    paths = []
    feats = feature_cols(df)

    for target in GRAB_TARGETS:
        # Rank features: use MI if available, else |Pearson|
        if target in all_mi:
            ranked = all_mi[target].sort_values(ascending=False)
        else:
            corr = pearson_spearman(df, feats, target).dropna()
            ranked = corr["spearman"].abs().sort_values(ascending=False)

        top_feats = [f for f in ranked.index if f in df.columns][:15]

        ncols = 5
        nrows = int(np.ceil(len(top_feats) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5))
        fig.suptitle(f"Top features vs {s(target)} (coloured by year)",
                     fontsize=13, fontweight="bold")

        axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
        for ax, feat in zip(axes_flat, top_feats):
            sub = df[["Date", feat, target]].dropna().copy()
            sub["year"] = sub["Date"].dt.year
            for yr, grp in sub.groupby("year"):
                ax.scatter(grp[feat], grp[target],
                           color=YEAR_COLOURS.get(yr, "#999"),
                           alpha=0.4, s=10, label=str(yr))
            # Regression line
            if len(sub) >= 10:
                m, b, r, _, _ = stats.linregress(sub[feat], sub[target])
                xr = np.array([sub[feat].min(), sub[feat].max()])
                ax.plot(xr, m * xr + b, color="black", linewidth=1.2, linestyle="--")
                ax.set_title(f"{s(feat)}\n(r={r:.2f})", fontsize=8, pad=2)
            else:
                ax.set_title(s(feat), fontsize=8, pad=2)
            ax.set_xlabel(s(feat), fontsize=7)
            ax.set_ylabel(s(target), fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for ax in list(axes_flat)[len(top_feats):]:
            ax.set_visible(False)

        # Legend (years) — one shared legend
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=YEAR_COLOURS[yr], markersize=6)
                   for yr in sorted(YEAR_COLOURS)]
        fig.legend(handles, [str(y) for y in sorted(YEAR_COLOURS)],
                   title="Year", loc="lower right", ncol=5, fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        slug = target.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "").replace(",", "")
        p = save(fig, f"04_scatter_{slug[:30]}")
        paths.append(p)
    return paths


# ── Chart 5: Aeration time series ─────────────────────────────────────────────

def plot_aeration_timeseries(df):
    print("5. Aeration time series...")
    aer_df = df[df["Date"].dt.year >= 2021].copy()

    params = [
        ("Aeration DO (mg/L, Existing)",   "Aeration DO (mg/L, New)",   "DO (mg/L)",  0.5),
        ("Aeration MLSS (mg/L, Existing)", "Aeration MLSS (mg/L, New)", "MLSS (mg/L)", None),
        ("Aeration SVI (Existing)",         "Aeration SVI (New)",        "SVI",         None),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle("Aeration Tank — DO / MLSS / SVI over time",
                 fontsize=13, fontweight="bold")

    for ax, (col_e, col_n, ylabel, threshold) in zip(axes, params):
        ax.plot(aer_df["Date"], aer_df[col_e],
                color="#2171B5", alpha=0.7, linewidth=0.8, label="Existing tank")
        ax.plot(aer_df["Date"], aer_df[col_n],
                color="#FD8D3C", alpha=0.7, linewidth=0.8, label="New tank")
        if threshold is not None:
            ax.axhline(threshold, color="red", linewidth=1.2, linestyle="--",
                       label=f"Min threshold ({threshold})")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        # Year shading
        for yr, colour in YEAR_COLOURS.items():
            ys = aer_df[aer_df["Date"].dt.year == yr]["Date"]
            if len(ys):
                ax.axvspan(ys.min(), ys.max(), alpha=0.04, color=colour)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return save(fig, "05_aeration_timeseries")


# ── Chart 6: Cross-stage correlation heatmap ──────────────────────────────────

def plot_cross_stage_heatmap(df):
    print("6. Cross-stage correlation heatmap...")
    feats = feature_cols(df)
    targets = GRAB_TARGETS + COMP_TARGETS

    # Build Pearson and Spearman matrices
    pears = pd.DataFrame(index=feats, columns=[s(t) for t in targets], dtype=float)
    spear = pd.DataFrame(index=feats, columns=[s(t) for t in targets], dtype=float)

    for target in targets:
        for feat in feats:
            sub = df[[feat, target]].dropna()
            if len(sub) < 10:
                pears.loc[feat, s(target)] = np.nan
                spear.loc[feat, s(target)] = np.nan
            else:
                pr, _ = stats.pearsonr(sub[feat], sub[target])
                sr, _ = stats.spearmanr(sub[feat], sub[target])
                pears.loc[feat, s(target)] = pr
                spear.loc[feat, s(target)] = sr

    pears = pears.astype(float)
    spear = spear.astype(float)

    # Sort rows by mean absolute Pearson across grab targets
    grab_cols = [s(t) for t in GRAB_TARGETS]
    row_order = pears[grab_cols].abs().mean(axis=1).sort_values(ascending=False).index
    pears = pears.loc[row_order]
    spear = spear.loc[row_order]
    short_rows = [s(f) for f in pears.index]

    fig, axes = plt.subplots(1, 2, figsize=(22, 16))
    kws = dict(cmap="RdBu_r", vmin=-1, vmax=1,
               linewidths=0.3, linecolor="white",
               cbar_kws={"shrink": 0.7, "label": "r"},
               annot=True, fmt=".2f", annot_kws={"size": 6})

    sns.heatmap(pears.rename(index=dict(zip(pears.index, short_rows))),
                ax=axes[0], **kws)
    axes[0].set_title("Pearson correlation", fontsize=12, fontweight="bold")

    sns.heatmap(spear.rename(index=dict(zip(spear.index, short_rows))),
                ax=axes[1], **kws)
    axes[1].set_title("Spearman correlation", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle("All features vs effluent targets — Pearson (left) and Spearman (right)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return save(fig, "06_cross_stage_heatmap")


# ── Chart 7: Stage removal efficiency ─────────────────────────────────────────

def plot_stage_removal(df):
    print("7. Stage removal efficiency...")
    params = [
        ("BOD", [
            "Inlet BOD (mg/L, Grab)",
            "Primary BOD (mg/L)",
            "Sec Clarifier BOD (mg/L)",
            "Sec Sed BOD (mg/L)",
            "Effluent BOD (mg/L, Grab)",
        ]),
        ("COD", [
            "Inlet COD (mg/L, Grab)",
            "Primary COD (mg/L)",
            "Sec Clarifier COD (mg/L)",
            "Sec Sed COD (mg/L)",
            "Effluent COD (mg/L, Grab)",
        ]),
        ("TSS", [
            "Inlet TSS (mg/L, Grab)",
            "Primary TSS (mg/L)",
            "Sec Clarifier TSS (mg/L)",
            "Sec Sed TSS (mg/L)",
            "Effluent TSS (mg/L, Grab)",
        ]),
    ]
    stage_labels = ["Inlet", "Primary\nClarifier", "Sec\nClarifier", "Sec\nSed", "Effluent"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Concentration at each treatment stage — median ± IQR (2021–2025)",
                 fontsize=13, fontweight="bold")

    for ax, (param, cols) in zip(axes, params):
        present = [c for c in cols if c in df.columns]
        medians = [df[c].median() for c in present]
        q25     = [df[c].quantile(0.25) for c in present]
        q75     = [df[c].quantile(0.75) for c in present]
        labels  = stage_labels[:len(present)]
        x       = np.arange(len(present))

        ax.fill_between(x, q25, q75, alpha=0.25, color="#2171B5", label="IQR")
        ax.plot(x, medians, "o-", color="#2171B5", linewidth=2, markersize=8, label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Concentration (mg/L)", fontsize=10)
        ax.set_title(f"{param} removal by stage", fontsize=11, fontweight="bold")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # Annotate % removal from inlet to effluent
        if medians[0] and medians[-1] and medians[0] > 0:
            pct = (1 - medians[-1] / medians[0]) * 100
            ax.text(len(present) - 1, medians[-1] * 1.5,
                    f"−{pct:.0f}%\noverall", ha="center", fontsize=9,
                    color="#D94801", fontweight="bold")

    plt.tight_layout()
    return save(fig, "07_stage_removal")


# ── Chart 8: Aeration DO vs effluent BOD / COD ───────────────────────────────

def plot_do_vs_effluent(df):
    print("8. Aeration DO vs effluent BOD/COD...")
    do_col = "Aeration DO (mg/L, Existing)"
    targets_plot = [
        ("Effluent BOD (mg/L, Grab)", "Eff BOD (G)"),
        ("Effluent COD (mg/L, Grab)", "Eff COD (G)"),
        ("Effluent TSS (mg/L, Grab)", "Eff TSS (G)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Aeration DO (Existing tank) vs effluent quality\n"
                 "(dashed line = DO = 0.5 mg/L minimum threshold)",
                 fontsize=13, fontweight="bold")

    for ax, (tcol, tlabel) in zip(axes, targets_plot):
        sub = df[[do_col, tcol, "Date"]].dropna().copy()
        sub["year"] = sub["Date"].dt.year
        for yr, grp in sub.groupby("year"):
            ax.scatter(grp[do_col], grp[tcol],
                       color=YEAR_COLOURS.get(yr, "#999"),
                       alpha=0.45, s=18, label=str(yr))
        ax.axvline(0.5, color="red", linewidth=1.5, linestyle="--", label="DO = 0.5")
        ax.set_xlabel("Aeration DO (mg/L, Existing)", fontsize=10)
        ax.set_ylabel(tlabel, fontsize=10)
        ax.set_title(tlabel, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="Year", ncol=2)

    plt.tight_layout()
    return save(fig, "08_do_vs_effluent")


# ── Chart 9: SVI / MLSS vs effluent TSS ───────────────────────────────────────

def plot_svi_mlss_vs_tss(df):
    print("9. SVI / MLSS vs effluent TSS...")
    pairs = [
        ("Aeration SVI (Existing)",         "SVI (Existing)"),
        ("Aeration MLSS (mg/L, Existing)",  "MLSS (Existing)"),
        ("Aeration SV30 (ml/L, Existing)",  "SV30 (Existing)"),
        ("Aeration SVI (New)",              "SVI (New)"),
        ("Aeration MLSS (mg/L, New)",       "MLSS (New)"),
        ("Aeration SV30 (ml/L, New)",       "SV30 (New)"),
    ]
    tss_col = "Effluent TSS (mg/L, Grab)"

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    fig.suptitle("Aeration settleability / biomass indicators vs Effluent TSS",
                 fontsize=13, fontweight="bold")

    for ax, (feat, label) in zip(axes.flat, pairs):
        sub = df[[feat, tss_col, "Date"]].dropna().copy()
        sub["year"] = sub["Date"].dt.year
        for yr, grp in sub.groupby("year"):
            ax.scatter(grp[feat], grp[tss_col],
                       color=YEAR_COLOURS.get(yr, "#999"),
                       alpha=0.45, s=18, label=str(yr))
        if len(sub) >= 10:
            m, b, r, _, _ = stats.linregress(sub[feat], sub[tss_col])
            xr = np.array([sub[feat].min(), sub[feat].max()])
            ax.plot(xr, m * xr + b, color="black", linewidth=1.2, linestyle="--")
            ax.set_title(f"{label}  (r = {r:.2f})", fontsize=10, fontweight="bold")
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Effluent TSS (mg/L, Grab)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, title="Year", ncol=2)

    plt.tight_layout()
    return save(fig, "09_svi_mlss_vs_tss")


# ── Chart 10: Ridge residuals (linearity diagnostic) ─────────────────────────

def plot_linearity_residuals(df):
    print("10. Linearity residuals (Ridge)...")
    feats = feature_cols(df)
    train_df = df[df["Date"].dt.year.isin([2021, 2022, 2023, 2024])]
    test_df  = df[df["Date"].dt.year == 2025]

    all_targets = GRAB_TARGETS + COMP_TARGETS
    fig, big_axes = plt.subplots(len(all_targets), 3, figsize=(20, len(all_targets) * 5))
    fig.suptitle("Ridge regression residuals — linearity diagnostic\n"
                 "(random scatter = linear relationship; patterns = non-linearity)\n"
                 "Top block: Grab effluent targets  |  Bottom block: Composite effluent targets",
                 fontsize=13, fontweight="bold")

    for row_ax, target in zip(big_axes, all_targets):
        usable_feats = [f for f in feats if f in df.columns]
        tr = train_df[usable_feats + [target]].dropna()
        te = test_df[usable_feats + [target]].dropna()

        if len(tr) < 20:
            for ax in row_ax:
                ax.set_visible(False)
            continue

        X_tr = tr[usable_feats].values
        y_tr = tr[target].values
        X_te = te[usable_feats].values
        y_te = te[target].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else X_te

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)

        yhat_tr = ridge.predict(X_tr_s)
        resid_tr = y_tr - yhat_tr

        # Panel A: residuals vs fitted (train)
        ax = row_ax[0]
        ax.scatter(yhat_tr, resid_tr, color="#2171B5", alpha=0.3, s=12)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Fitted values (train)")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{s(target)}\nResid vs Fitted", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

        # Panel B: residuals over time (train)
        ax = row_ax[1]
        ax.scatter(tr.index, resid_tr, color="#2171B5", alpha=0.3, s=8)
        if len(te) > 0:
            yhat_te = ridge.predict(X_te_s)
            resid_te = y_te - yhat_te
            ax.scatter(te.index, resid_te, color="#D94801", alpha=0.4, s=8, label="Test 2025")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Row index (time)")
        ax.set_ylabel("Residuals")
        ax.set_title("Resid vs Time", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Panel C: residuals vs top feature by |coefficient|
        coef = pd.Series(np.abs(ridge.coef_), index=usable_feats)
        top_feat = coef.idxmax()
        ax = row_ax[2]
        ax.scatter(tr[top_feat].values, resid_tr, color="#2171B5", alpha=0.3, s=12)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel(s(top_feat))
        ax.set_ylabel("Residuals")
        ax.set_title(f"Resid vs {s(top_feat)}\n(highest |coef|)", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return save(fig, "10_linearity_residuals")


# ── Data-driven observations ──────────────────────────────────────────────────

def build_observations(df):
    """
    Compute data-driven HTML observations for all 10 EDA sections.
    Returns dict: section_id (str "s1"…"s10") → HTML string.
    """
    from sklearn.metrics import r2_score, mean_squared_error as mse

    feats  = feature_cols(df)
    usable = [f for f in feats if f in df.columns]
    trndf  = df[df["Date"].dt.year.isin([2021, 2022, 2023, 2024])]
    tstdf  = df[df["Date"].dt.year == 2025]
    obs    = {}

    # ── S1: Ridge metrics per grab and composite target ──────────────────────
    def _ridge_row(tgt, group_label):
        tr = trndf[usable + [tgt]].dropna()
        te = tstdf[usable + [tgt]].dropna()
        if len(tr) < 20:
            return None
        Xtr, ytr = tr[usable].values, tr[tgt].values
        Xte, yte = te[usable].values, te[tgt].values
        sc    = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(Xtr), ytr)
        yhat_tr   = ridge.predict(sc.transform(Xtr))
        r2_tr     = r2_score(ytr, yhat_tr)
        rmse_tr   = np.sqrt(mse(ytr, yhat_tr))
        resid_tr  = ytr - yhat_tr
        funnel_r, _ = stats.spearmanr(yhat_tr, np.abs(resid_tr))
        top_feat  = pd.Series(np.abs(ridge.coef_), index=usable).idxmax()
        row = {"tgt": s(tgt), "group": group_label, "r2_tr": r2_tr, "rmse_tr": rmse_tr,
               "top": s(top_feat), "funnel": funnel_r}
        if len(te) >= 5:
            yhat_te      = ridge.predict(sc.transform(Xte))
            row["r2_te"]   = r2_score(yte, yhat_te)
            row["rmse_te"] = np.sqrt(mse(yte, yhat_te))
            row["gap"]     = r2_tr - row["r2_te"]
        return row

    rows_s1 = []
    for tgt in GRAB_TARGETS:
        row = _ridge_row(tgt, "Grab")
        if row:
            rows_s1.append(row)
    for tgt in COMP_TARGETS:
        row = _ridge_row(tgt, "Composite")
        if row:
            rows_s1.append(row)

    def _tbl_rows(rows):
        html = ""
        prev_group = None
        for r in rows:
            r2_te_s   = f"{r['r2_te']:.3f}"   if "r2_te"   in r else "—"
            rmse_te_s = f"{r['rmse_te']:.2f}" if "rmse_te" in r else "—"
            gap_s     = f"{r['gap']:.3f}"     if "gap"     in r else "—"
            r2_color  = "#C0392B" if r.get("r2_te", 1) < 0 else "#27AE60"
            gap_color = ("#C0392B" if r.get("gap", 0) > 0.15
                         else "#27AE60" if r.get("gap", 0) < 0.05
                         else "inherit")
            group_cell = ""
            if r["group"] != prev_group:
                group_count = sum(1 for x in rows if x["group"] == r["group"])
                group_cell = (f"<td rowspan='{group_count}' "
                              f"style='font-weight:bold;vertical-align:middle;"
                              f"text-align:center;background:#EEF2F7'>{r['group']}</td>")
                prev_group = r["group"]
            html += (
                f"<tr>{group_cell}"
                f"<td>{r['tgt']}</td>"
                f"<td style='text-align:center'>{r['r2_tr']:.3f}</td>"
                f"<td style='text-align:center;color:{r2_color};font-weight:bold'>{r2_te_s}</td>"
                f"<td style='text-align:center;color:{gap_color}'>{gap_s}</td>"
                f"<td style='text-align:center'>{r['rmse_tr']:.2f}</td>"
                f"<td style='text-align:center'>{rmse_te_s}</td>"
                f"<td style='font-size:0.85em'>{r['top']}</td></tr>\n"
            )
        return html

    tbl_s1 = _tbl_rows(rows_s1)
    n_neg = sum(1 for r in rows_s1 if r.get("r2_te", 0) < 0)
    funnel_targets = [r["tgt"] for r in rows_s1 if abs(r.get("funnel", 0)) > 0.20]

    if n_neg == len(rows_s1) and len(rows_s1) > 0:
        neg_note = ("<li><strong>Negative test R² for all targets:</strong> A negative R² means "
                    "the Ridge model performs <em>worse</em> than predicting the mean — the model "
                    "adds no predictive value on 2025 data.</li>")
    elif n_neg > 0:
        neg_note = (f"<li><strong>Negative test R² for {n_neg} target(s):</strong> Negative R² "
                    "means Ridge is worse than predicting the mean. Combined with a reasonable "
                    "train R², this suggests the linear structure learned from 2021–2024 does not "
                    "transfer to 2025 operating conditions.</li>")
    else:
        neg_note = ("<li><strong>Degraded test R²:</strong> A notable drop from train to test R² "
                    "suggests the model may not generalise well to 2025 conditions.</li>")

    funnel_note = ""
    if funnel_targets:
        tgts_str = ", ".join(funnel_targets)
        vals_str = ", ".join(
            f"{r['funnel']:.2f}" for r in rows_s1 if abs(r.get("funnel", 0)) > 0.20
        )
        funnel_note = (
            f"<li><strong>Potential heteroscedasticity for: {tgts_str}.</strong> "
            "The Spearman correlation between absolute residuals and fitted values "
            f"(the model's training-set predictions) is {vals_str}. "
            "This is a standalone statistical test on the training residuals — it is not derived "
            "from the Pearson/Spearman feature correlations shown in Section 2. A positive "
            "correlation means larger predictions tend to have larger errors; a negative one means "
            "errors shrink as predictions rise. Inspect the Resid vs Fitted panels directly to "
            "confirm whether this is visually apparent.</li>"
        )

    obs["s1"] = (
        "<p>Ridge regression (α = 1, StandardScaler) trained on 2021–2024, tested on 2025. "
        "Columns are ordered to make the train R², test R², and R² gap easy to compare:</p>"
        "<table class='obs-table' style='width:100%;margin:10px 0'>"
        "<thead><tr>"
        "<th>Sample type</th><th>Target</th>"
        "<th>Train R²</th><th>Test R²</th><th>R² Gap</th>"
        "<th>Train RMSE</th><th>Test RMSE</th>"
        "<th>Top feature (|coef|)</th>"
        "</tr></thead>"
        f"<tbody>{tbl_s1}</tbody></table>"
        f"<ul>{neg_note}"
        "<li><strong>R² gap:</strong> A gap between train and test R² indicates the model may "
        "be overfitting to the training period. Whether that gap is meaningful depends on its "
        "size relative to the test R² — a small gap on a poor test R² is still a poor model.</li>"
        f"{funnel_note}"
        "<li><strong>Top relied-on feature:</strong> The feature with the largest Ridge coefficient "
        "magnitude is what the linear model leans on most heavily. If that feature has a curved "
        "relationship with the target (visible in Section 4 scatter plots), the model will "
        "systematically under- or over-predict at extreme concentrations.</li>"
        "</ul>"
    )

    # ── S2: Pearson vs Spearman ────────────────────────────────────────────────
    ps_blocks = []
    for tgt in GRAB_TARGETS:
        sub = df[usable + [tgt]].dropna()
        if len(sub) < 20:
            continue
        X, y = sub[usable], sub[tgt]
        p_r = X.apply(lambda c: stats.pearsonr(c, y)[0])
        sp_r = X.apply(lambda c: stats.spearmanr(c, y)[0])
        gap  = sp_r.abs() - p_r.abs()
        top5 = sp_r.abs().nlargest(5).index
        pct_nl = (gap > 0.10).mean() * 100

        tbl = ""
        for f in top5:
            gap_color = "#C0392B" if gap[f] > 0.10 else "var(--text)"
            tbl += (
                f"<tr><td>{s(f)}</td>"
                f"<td style='color:#2171B5'>{p_r[f]:+.3f}</td>"
                f"<td style='color:#FD8D3C'>{sp_r[f]:+.3f}</td>"
                f"<td style='color:{gap_color}'>{gap[f]:+.3f}</td></tr>"
            )
        ps_blocks.append(
            f"<h5 style='margin:16px 0 6px 0;color:var(--text)'>{s(tgt)}</h5>"
            "<table class='summary-table' style='width:auto;min-width:500px;font-size:0.88em'>"
            "<thead><tr><th>Feature</th><th>Pearson r</th><th>Spearman r</th>"
            "<th>Gap (|Spear|−|Pear|)</th></tr></thead>"
            f"<tbody>{tbl}</tbody></table>"
            f"<p style='font-size:0.87em;color:var(--text-muted);margin:4px 0 0 0'>"
            f"{pct_nl:.0f}% of all {len(usable)} features show |Spearman| &gt; |Pearson| by "
            f"more than 0.10 for this target — indicating predominantly curved or non-monotonic "
            f"relationships.</p>"
        )

    obs["s2"] = (
        "".join(ps_blocks)
        + "<ul style='margin-top:18px'>"
        "<li><strong>Positive Gap (red):</strong> Spearman exceeds Pearson — the feature moves "
        "with the target but not proportionally. The relationship is curved (e.g., logarithmic, "
        "exponential, or step-like). Linear models undervalue these features; Random Forest "
        "naturally handles curvature by splitting across ranges of the feature space.</li>"
        "<li><strong>Negative Gap:</strong> A few extreme outlier observations are inflating the "
        "Pearson correlation. Spearman (rank-based) is unaffected. These features may appear "
        "deceptively important in a linear model but be unreliable predictors. Investigate the "
        "outlier days — they may represent instrument errors or genuine process upsets.</li>"
        "<li><strong>Both magnitudes small:</strong> Low Pearson AND Spearman does not mean "
        "the feature is uninformative — it may have a non-monotonic relationship (e.g., an optimal "
        "aeration DO range with an inverted-U effect). Check Mutual Information in Section 3, "
        "which detects non-monotonic patterns that rank correlation misses entirely.</li>"
        "<li><strong>Model selection take-away:</strong> When the majority of top-ranked features "
        "show positive gaps, a linear model systematically mis-specifies their contributions. "
        "The prevalence of non-zero gaps across all four grab targets is consistent with the "
        "Ridge failure observed in Section 1.</li>"
        "</ul>"
    )

    # ── S3: Mutual Information ─────────────────────────────────────────────────
    mi_blocks = []
    for tgt in GRAB_TARGETS:
        sub = df[usable + [tgt]].dropna()
        if len(sub) < 20:
            continue
        X_vals, y_vals = sub[usable].values, sub[tgt].values
        mi_scores = mutual_info_regression(X_vals, y_vals, random_state=42)
        mi_s = pd.Series(mi_scores, index=usable)
        sp_r = pd.Series(
            [stats.spearmanr(sub[f], sub[tgt])[0] for f in usable], index=usable
        )
        top5_mi  = mi_s.nlargest(5)
        mi_thr   = mi_s.quantile(0.75)
        nonlin   = mi_s[(mi_s >= mi_thr) & (sp_r.abs() < 0.30)].nlargest(3)

        tbl = ""
        for f in top5_mi.index:
            is_nl = (mi_s[f] >= mi_thr) and (abs(sp_r[f]) < 0.30)
            nl_color = "#C0392B" if is_nl else "var(--text)"
            nl_label = "⚠ Non-monotonic" if is_nl else "Monotonic"
            tbl += (
                f"<tr><td>{s(f)}</td>"
                f"<td>{mi_s[f]:.4f}</td>"
                f"<td style='color:#FD8D3C'>{sp_r[f]:+.3f}</td>"
                f"<td style='color:{nl_color}'>{nl_label}</td></tr>"
            )

        nonlin_note = ""
        if len(nonlin) > 0:
            names = ", ".join(s(f) for f in nonlin.index)
            nonlin_note = (
                f"<p style='font-size:0.87em;color:#C0392B;margin:6px 0'>"
                f"⚠ <strong>Non-monotonic features (top-quartile MI, |Spearman| &lt; 0.30):</strong> "
                f"{names}. These carry real predictive information but in a pattern that only "
                f"non-linear models can exploit — they will be ignored or penalised by Ridge.</p>"
            )

        mi_blocks.append(
            f"<h5 style='margin:16px 0 6px 0;color:var(--text)'>{s(tgt)}</h5>"
            "<table class='summary-table' style='width:auto;min-width:500px;font-size:0.88em'>"
            "<thead><tr><th>Feature</th><th>MI score</th><th>Spearman r</th>"
            "<th>Relationship type</th></tr></thead>"
            f"<tbody>{tbl}</tbody></table>"
            f"{nonlin_note}"
        )

    obs["s3"] = (
        "".join(mi_blocks)
        + "<ul style='margin-top:18px'>"
        "<li><strong>MI vs Spearman together:</strong> When MI is high but Spearman is near zero, "
        "the feature and target are statistically dependent but not monotonically so — there is a "
        "real relationship (e.g., an optimal operating range, a threshold, a seasonal interaction) "
        "that purely rank-based methods miss entirely. Random Forest detects these patterns through "
        "axis-aligned splits.</li>"
        "<li><strong>Features to prioritise for model training:</strong> Features with both high MI "
        "and high |Spearman| are the most reliable predictors — they carry a strong, consistent "
        "signal regardless of the model family. Features with high MI but low Spearman are "
        "specifically valuable for Random Forest and should be retained during feature selection "
        "(RFECV/SFS) even if linear-model-based selectors would drop them.</li>"
        "<li><strong>Low MI features:</strong> Features with near-zero MI scores carry essentially "
        "no information about the target (linear, monotonic, or otherwise) and are strong candidates "
        "for exclusion during feature selection, reducing noise and improving generalisation.</li>"
        "<li><strong>Model selection take-away:</strong> The presence of non-monotonic features "
        "(high MI, low Spearman) confirms that a model capable of detecting arbitrary feature–target "
        "interactions — such as Random Forest — is better suited than Ridge regression for this "
        "prediction task.</li>"
        "</ul>"
    )

    # ── S4: Scatter grids ─────────────────────────────────────────────────────
    obs["s4"] = (
        "<ul>"
        "<li><strong>What a good linear scatter looks like:</strong> Points cluster tightly around "
        "the dashed regression line without systematic deviation. All year colours (2021–2025) align "
        "on the same band. A linear model would perform well for features showing this pattern.</li>"
        "<li><strong>Curved scatter (non-linearity):</strong> If points form a concave or convex arc "
        "around the regression line, the relationship is non-linear. A linear model will "
        "over-predict in the middle range and under-predict at extremes (or vice versa). "
        "Common in wastewater where treatment efficiency plateaus at very high or very low loads.</li>"
        "<li><strong>Fan shape (heteroscedasticity):</strong> If scatter expands as the feature "
        "value increases, error variance is not constant. Linear models assume constant variance; "
        "Random Forest adapts naturally because splits are value-driven, not variance-assumption-driven.</li>"
        "<li><strong>Year-cluster separation:</strong> If different years occupy distinct regions of "
        "the scatter plot, there is inter-annual variation in operating regime. A model without a "
        "year feature will produce large residuals for years that drift from the historical average. "
        "Consider whether year (or a seasonal proxy) should be included as a feature.</li>"
        "<li><strong>Isolated outliers:</strong> Extreme points far from the main cloud may be "
        "measurement errors, equipment failures, or genuine process upsets. They inflate Pearson "
        "correlation while leaving Spearman unaffected — one reason to cross-check both metrics. "
        "Investigate high-leverage days before finalising the feature set.</li>"
        "<li><strong>Cross-reference with Sections 2 and 3:</strong> The scatter plots show the "
        "top 15 features by Mutual Information. Features identified as non-monotonic in Section 3 "
        "(high MI, low Spearman) may show no obvious pattern in a simple scatter but still carry "
        "information — look for cluster-based structure or threshold behaviour rather than a "
        "clean monotonic trend.</li>"
        "</ul>"
    )

    # ── S5: Cross-stage heatmap ────────────────────────────────────────────────
    stage_groups = {
        "Inlet":         [f for f in usable if f.lower().startswith("inlet")],
        "Primary":       [f for f in usable if f.lower().startswith("primary")],
        "Aeration":      [f for f in usable if f.lower().startswith("aeration")],
        "Sec Clarifier": [f for f in usable if "sec clarifier" in f.lower()],
        "Sec Sed":       [f for f in usable if "sec sed" in f.lower()],
    }
    stage_mean_corr = {}
    for stage_name, cols in stage_groups.items():
        vals = []
        for f in cols:
            for tgt in GRAB_TARGETS:
                sub = df[[f, tgt]].dropna()
                if len(sub) >= 20:
                    r, _ = stats.pearsonr(sub[f], sub[tgt])
                    vals.append(abs(r))
        if vals:
            stage_mean_corr[stage_name] = float(np.mean(vals))
    ranked = sorted(stage_mean_corr.items(), key=lambda x: x[1], reverse=True)
    rank_li = "".join(
        f"<li><strong>{sn}</strong> — mean |Pearson r| across grab targets = {v:.3f}</li>"
        for sn, v in ranked
    )

    obs["s5"] = (
        f"<p><strong>Treatment stage ranking by mean |Pearson r| with effluent grab targets:</strong></p>"
        f"<ol style='margin:4px 0 12px 20px'>{rank_li}</ol>"
        "<ul>"
        "<li><strong>Strong inlet–effluent correlations</strong> indicate that incoming load "
        "variation propagates directly to the final effluent — the treatment process cannot fully "
        "buffer large fluctuations in feed concentration. Features from high-ranking stages are "
        "likely to be retained through feature selection regardless of model type.</li>"
        "<li><strong>Intermediate stage correlations</strong> (e.g., secondary clarifier parameters) "
        "may be weaker because daily grab samples do not account for hydraulic retention time lag — "
        "the effluent quality on a given day may reflect the inlet load from 12–24 hours earlier. "
        "A lagged feature (e.g., inlet BOD lagged by 1 day) may be more informative than same-day "
        "measurements and is worth testing during feature engineering.</li>"
        "<li><strong>Comparing Pearson vs Spearman panels:</strong> Cells where Spearman is "
        "notably darker than Pearson identify stage–target combinations where the relationship is "
        "curved or non-monotonic. These are the cells that most strongly justify using Random Forest "
        "over a linear model — Ridge would underestimate the influence of those stage parameters.</li>"
        "<li><strong>Cross-stage correlations:</strong> Strong correlations between inlet and "
        "intermediate-stage parameters (e.g., inlet BOD vs secondary clarifier BOD) are expected "
        "due to the sequential treatment process. If intermediate measurements are not always "
        "available, inlet measurements alone may be sufficient predictors — a consideration for "
        "imputation strategy when intermediate columns have lower coverage.</li>"
        "</ul>"
    )

    # ── S6: Data coverage ─────────────────────────────────────────────────────
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    coverage  = df[num_cols].notna().mean() * 100
    low_cols  = coverage[coverage < 50].sort_values()
    med_cols  = coverage[(coverage >= 50) & (coverage < 80)].sort_values()
    high_cols = coverage[coverage >= 80]

    low_list = ", ".join(f"{s(c)} ({v:.0f}%)" for c, v in low_cols.items()) or "None"
    med_list = ", ".join(f"{s(c)} ({v:.0f}%)" for c, v in med_cols.items()) or "None"

    obs["s6"] = (
        f"<ul>"
        f"<li><strong>High coverage (≥ 80%, {len(high_cols)} columns):</strong> These columns are "
        f"suitable for direct use as features or targets with minimal imputation. Core grab-sample "
        f"parameters and most secondary-stage measurements fall in this tier.</li>"
        f"<li><strong>Medium coverage (50–80%, {len(med_cols)} columns):</strong> {med_list}. "
        f"These columns provide useful signal but will reduce the effective training sample size "
        f"when used together (row-wise dropna). Evaluate whether imputation (forward-fill, median) "
        f"is justified or whether the column should be treated as an optional feature.</li>"
        f"<li><strong>Low coverage (&lt; 50%, {len(low_cols)} columns):</strong> {low_list}. "
        f"Using these as features will substantially shrink the training set. Consider either "
        f"excluding them from the base feature set or imputing with care — forward-fill is "
        f"appropriate for readings that are stable over days (e.g., MLVSS); random missingness "
        f"benefits more from median imputation.</li>"
        f"<li><strong>Temporal gap patterns:</strong> The monthly heatmap reveals whether gaps "
        f"are clustered in specific years or spread randomly. Systematic gaps in 2021 for composite "
        f"columns (expected — composite sampling began in 2022) are predictable and do not "
        f"constitute a data quality issue. Random scattered gaps may reflect recording lapses "
        f"and are better candidates for imputation than structured missingness.</li>"
        f"<li><strong>Impact on modelling:</strong> For features with medium-to-low coverage, "
        f"run a sensitivity check: compare model performance with and without that feature. "
        f"If adding it improves test R² despite fewer training rows, include it; if the reduced "
        f"sample causes instability, exclude it.</li>"
        f"</ul>"
    )

    # ── S7: Aeration time series ───────────────────────────────────────────────
    do_col  = "Aeration DO (mg/L, Existing)"
    svi_col = "Aeration SVI (Existing)"
    mlss_col = "Aeration MLSS (mg/L, Existing)"

    do_note = svi_note = mlss_note = ""
    if do_col in df.columns:
        do_v = df[do_col].dropna()
        pct_below = (do_v < 0.5).mean() * 100
        do_note = (
            f"<li><strong>DO below minimum threshold:</strong> {pct_below:.1f}% of all daily "
            f"readings for the Existing tank fall below 0.5 mg/L (median = {do_v.median():.2f} "
            f"mg/L, P10 = {do_v.quantile(0.10):.2f}, P90 = {do_v.quantile(0.90):.2f} mg/L). "
            f"{'Frequent sub-threshold events' if pct_below > 10 else 'Occasional sub-threshold events'} "
            f"are expected to coincide with elevated effluent BOD and COD due to insufficient "
            f"aerobic mineralisation — confirmed in Section 9.</li>"
        )
    if svi_col in df.columns:
        svi_v = df[svi_col].dropna()
        svi_note = (
            f"<li><strong>Sludge settleability (SVI):</strong> Median SVI = {svi_v.median():.0f} ml/g, "
            f"P90 = {svi_v.quantile(0.90):.0f} ml/g. "
            f"{'Values regularly exceed 150 ml/g (bulking threshold), indicating episodic poor settleability.' if svi_v.quantile(0.75) > 150 else 'Most values are below the 150 ml/g bulking threshold, but episodic spikes occur.'} "
            f"These spikes are likely to appear as outliers in effluent TSS — visible in Section 10.</li>"
        )
    if mlss_col in df.columns:
        mlss_v = df[mlss_col].dropna()
        mlss_note = (
            f"<li><strong>Biomass concentration (MLSS):</strong> Median MLSS = {mlss_v.median():.0f} mg/L "
            f"(Existing tank, range = {mlss_v.quantile(0.10):.0f}–{mlss_v.quantile(0.90):.0f} mg/L). "
            f"Sustained low MLSS periods indicate washout risk; very high MLSS combined with low DO "
            f"creates oxygen limitation conditions. Both extremes degrade effluent quality non-linearly, "
            f"making MLSS a candidate non-linear feature for Random Forest.</li>"
        )

    obs["s7"] = (
        f"<ul>"
        f"{do_note}"
        f"{svi_note}"
        f"{mlss_note}"
        "<li><strong>Divergence between Existing and New tanks:</strong> If the two tanks track "
        "differently over time, they may operate under different aeration regimes. The MI and "
        "correlation charts in Sections 2–3 show which tank's parameters rank higher for each "
        "target — it may be worth including features from both tanks and letting feature selection "
        "decide which is more predictive.</li>"
        "<li><strong>Seasonal patterns:</strong> If DO dips or SVI spikes cluster in specific "
        "months (e.g., summer months with higher organic load or temperature effects on biological "
        "activity), a month-of-year feature or season indicator could improve model accuracy by "
        "providing the model a cue to expect different operating conditions.</li>"
        "</ul>"
    )

    # ── S8: Stage removal efficiency ──────────────────────────────────────────
    removal_rows = []
    for param in ["BOD", "COD", "TSS"]:
        i_col = f"Inlet {param} (mg/L, Grab)"
        e_col = f"Effluent {param} (mg/L, Grab)"
        if i_col not in df.columns or e_col not in df.columns:
            continue
        i_med = df[i_col].median()
        e_med = df[e_col].median()
        if i_med <= 0:
            continue
        i_iqr  = df[i_col].quantile(0.75) - df[i_col].quantile(0.25)
        e_iqr  = df[e_col].quantile(0.75) - df[e_col].quantile(0.25)
        pct_r  = (1 - e_med / i_med) * 100
        i_cv   = i_iqr / i_med
        e_cv   = e_iqr / e_med if e_med > 0 else None
        removal_rows.append({
            "param": param, "i_med": i_med, "e_med": e_med,
            "pct_r": pct_r, "i_cv": i_cv, "e_cv": e_cv,
        })

    rem_tbl = ""
    for r in removal_rows:
        i_cv_s = f"{r['i_cv']:.2f}" if r["i_cv"] is not None else "—"
        e_cv_s = f"{r['e_cv']:.2f}" if r["e_cv"] is not None else "—"
        rem_tbl += (
            f"<tr><td><strong>{r['param']}</strong></td>"
            f"<td>{r['i_med']:.1f}</td><td>{r['e_med']:.2f}</td>"
            f"<td><strong>{r['pct_r']:.0f}%</strong></td>"
            f"<td>{i_cv_s}</td><td>{e_cv_s}</td></tr>\n"
        )

    # Flag parameters where effluent CV > inlet CV (treatment amplifying variability)
    amp_params = [r["param"] for r in removal_rows
                  if r["e_cv"] is not None and r["i_cv"] is not None and r["e_cv"] > r["i_cv"]]
    amp_note = ""
    if amp_params:
        amp_note = (
            f"<li><strong>Amplified variability in effluent:</strong> For {', '.join(amp_params)}, "
            "the effluent IQR/median ratio exceeds the inlet ratio — secondary treatment "
            "<em>amplifies</em> concentration variability rather than just reducing it. "
            "This occurs because biological treatment is sensitive to operating conditions "
            "(DO, MLSS, HRT), which fluctuate independently of inlet load. "
            "A model must capture these process-driven fluctuations, not just inlet concentration, "
            "to predict effluent quality accurately.</li>"
        )

    obs["s8"] = (
        "<table class='summary-table' style='width:auto;font-size:0.88em;margin:10px 0'>"
        "<thead><tr><th>Parameter</th><th>Inlet median (mg/L)</th>"
        "<th>Effluent median (mg/L)</th><th>Overall removal</th>"
        "<th>Inlet IQR/median</th><th>Effluent IQR/median</th>"
        f"</tr></thead><tbody>{rem_tbl}</tbody></table>"
        "<ul>"
        "<li><strong>IQR/median (relative variability):</strong> A ratio above 0.5 indicates high "
        "day-to-day variability. Compare inlet vs effluent ratios — if effluent variability is "
        "comparable to or higher than inlet variability, the treatment process itself is introducing "
        "noise, not just attenuating the inlet signal. This process-driven variability is what "
        "the model needs to explain.</li>"
        "<li><strong>Where most removal occurs:</strong> The log-scale stage plot shows whether "
        "Primary Clarifier or Secondary stages dominate removal. The stage responsible for the "
        "largest single drop is also likely to produce the most variability in effluent quality "
        "when it under-performs — making its upstream parameters (e.g., primary TSS, aeration DO) "
        "the most important features for prediction.</li>"
        f"{amp_note}"
        "<li><strong>Compliance risk days:</strong> The median effluent is typically well below "
        "regulatory thresholds, but the P90 values (visible from the IQR shading in the chart) "
        "may reach compliance limits on some days. Understanding what combination of conditions "
        "drives those exceedance days is the primary modelling objective — and it is better served "
        "by a non-linear model capable of identifying the multi-condition thresholds that lead to "
        "exceedances, rather than a linear model that only sees average effects.</li>"
        "</ul>"
    )

    # ── S9: DO vs effluent quality ─────────────────────────────────────────────
    do_col = "Aeration DO (mg/L, Existing)"
    do_corr_rows = []
    if do_col in df.columns:
        for tgt_col, tgt_lbl in [
            ("Effluent BOD (mg/L, Grab)", "Effluent BOD"),
            ("Effluent COD (mg/L, Grab)", "Effluent COD"),
            ("Effluent TSS (mg/L, Grab)", "Effluent TSS"),
        ]:
            sub = df[[do_col, tgt_col]].dropna()
            if len(sub) < 20:
                continue
            p_r, _ = stats.pearsonr(sub[do_col], sub[tgt_col])
            sp_r, _ = stats.spearmanr(sub[do_col], sub[tgt_col])
            below = sub[sub[do_col] < 0.5]
            above = sub[sub[do_col] >= 0.5]
            do_corr_rows.append({
                "lbl": tgt_lbl, "pearson": p_r, "spearman": sp_r,
                "n_below": len(below), "n_above": len(above),
                "mean_below": below[tgt_col].mean() if len(below) > 0 else None,
                "mean_above": above[tgt_col].mean() if len(above) > 0 else None,
            })

    do_tbl = ""
    threshold_effects = []
    for r in do_corr_rows:
        mb = f"{r['mean_below']:.1f}" if r["mean_below"] is not None else "—"
        ma = f"{r['mean_above']:.1f}" if r["mean_above"] is not None else "—"
        do_tbl += (
            f"<tr><td>{r['lbl']}</td>"
            f"<td style='color:#2171B5'>{r['pearson']:+.3f}</td>"
            f"<td style='color:#FD8D3C'>{r['spearman']:+.3f}</td>"
            f"<td>{r['n_below']}</td><td>{mb}</td>"
            f"<td>{r['n_above']}</td><td>{ma}</td></tr>\n"
        )
        if (r["mean_below"] is not None and r["mean_above"] is not None
                and r["mean_above"] > 0):
            lift = (r["mean_below"] - r["mean_above"]) / r["mean_above"] * 100
            if lift > 20:
                threshold_effects.append(
                    f"{r['lbl']} ({lift:.0f}% higher when DO &lt; 0.5)"
                )

    thr_note = ""
    if threshold_effects:
        thr_note = (
            "<li><strong>Threshold effect confirmed:</strong> Mean concentration is substantially "
            "higher when DO &lt; 0.5 mg/L: " + "; ".join(threshold_effects) + ". "
            "This step-change at a specific threshold is exactly the kind of non-linearity that "
            "Random Forest can capture through a single split but Ridge regression cannot. "
            "It also quantifies the operational risk: maintaining DO above 0.5 mg/L is a "
            "direct lever for reducing effluent non-compliance events.</li>"
        )

    obs["s9"] = (
        "<table class='summary-table' style='width:auto;font-size:0.88em;margin:10px 0'>"
        "<thead><tr><th>Target</th><th>Pearson r</th><th>Spearman r</th>"
        "<th>n (DO &lt; 0.5)</th><th>Mean effl. when DO &lt; 0.5</th>"
        "<th>n (DO ≥ 0.5)</th><th>Mean effl. when DO ≥ 0.5</th>"
        f"</tr></thead><tbody>{do_tbl}</tbody></table>"
        f"<ul>"
        f"{thr_note}"
        "<li><strong>Overall correlation direction:</strong> A negative Pearson/Spearman value "
        "confirms that higher DO is associated with lower effluent concentrations — aerobic "
        "treatment is working as expected. The question is whether this relationship is linear "
        "(a fixed slope per mg/L of DO) or threshold-based (a step-change below 0.5 mg/L).</li>"
        "<li><strong>Pearson vs Spearman gap:</strong> If |Spearman| &gt; |Pearson|, the "
        "DO–effluent relationship is curved or step-like. This is direct evidence that the "
        "0.5 mg/L threshold acts as a breakpoint rather than a linear slope — a structure "
        "that Ridge cannot capture but Random Forest handles through splits.</li>"
        "<li><strong>TSS vs BOD/COD:</strong> TSS is controlled primarily by sedimentation "
        "(SVI, MLSS) rather than aeration chemistry, so a weaker DO–TSS correlation than "
        "DO–BOD/COD is expected. If DO still correlates with TSS, it may be through an indirect "
        "pathway: poor DO → elevated MLSS/SVI → particles carry over to effluent.</li>"
        "</ul>"
    )

    # ── S10: SVI/MLSS vs effluent TSS ─────────────────────────────────────────
    tss_col  = "Effluent TSS (mg/L, Grab)"
    svi_rows = []
    for feat in [
        "Aeration SVI (Existing)", "Aeration MLSS (mg/L, Existing)",
        "Aeration SV30 (ml/L, Existing)",
        "Aeration SVI (New)", "Aeration MLSS (mg/L, New)",
        "Aeration SV30 (ml/L, New)",
    ]:
        if feat not in df.columns or tss_col not in df.columns:
            continue
        sub = df[[feat, tss_col]].dropna()
        if len(sub) < 10:
            continue
        p_r, _  = stats.pearsonr(sub[feat], sub[tss_col])
        sp_r, _ = stats.spearmanr(sub[feat], sub[tss_col])
        svi_rows.append({"feat": s(feat), "pearson": p_r, "spearman": sp_r, "n": len(sub)})

    svi_tbl = ""
    for r in svi_rows:
        is_curved = abs(r["spearman"]) > abs(r["pearson"]) + 0.08
        curve_col = "#C0392B" if is_curved else "var(--text)"
        curve_lbl = "Curved ↑" if is_curved else "Approximately linear"
        svi_tbl += (
            f"<tr><td>{r['feat']}</td>"
            f"<td style='color:#2171B5'>{r['pearson']:+.3f}</td>"
            f"<td style='color:#FD8D3C'>{r['spearman']:+.3f}</td>"
            f"<td style='color:{curve_col}'>{curve_lbl}</td>"
            f"<td>{r['n']}</td></tr>"
        )

    useful = [r for r in svi_rows if abs(r["spearman"]) > 0.20]
    useful_note = ""
    if useful:
        u_str = ", ".join(
            f"{r['feat']} (Spearman = {r['spearman']:+.2f})" for r in useful
        )
        useful_note = (
            f"<li><strong>Predictive SVI/MLSS features:</strong> {u_str} show a statistically "
            f"meaningful monotonic relationship with effluent TSS. These should be included in "
            f"the feature selection candidate pool for any TSS prediction model.</li>"
        )
    else:
        useful_note = (
            "<li><strong>Weak direct correlations:</strong> SVI and MLSS show weak rank correlations "
            "with effluent TSS at daily resolution. This does not mean they are uninformative — "
            "check the MI chart in Section 3. A non-monotonic pattern (e.g., TSS spikes only "
            "above a SVI bulking threshold) would produce near-zero Spearman but positive MI.</li>"
        )

    obs["s10"] = (
        "<table class='summary-table' style='width:auto;font-size:0.88em;margin:10px 0'>"
        "<thead><tr><th>Feature</th><th>Pearson r</th><th>Spearman r</th>"
        "<th>Relationship type</th><th>n pairs</th>"
        f"</tr></thead><tbody>{svi_tbl}</tbody></table>"
        f"<ul>"
        f"{useful_note}"
        "<li><strong>Physical mechanism:</strong> SVI quantifies sludge settling speed. A high SVI "
        "means biomass settles slowly, increasing the chance that suspended solids carry over into "
        "the final effluent. The relationship is inherently non-linear: at low SVI, sedimentation "
        "is efficient and TSS is limited by other factors; above a bulking threshold (often cited "
        "at SVI ≥ 150 ml/g) TSS rises sharply. A linear model misses this inflection point.</li>"
        "<li><strong>Curved Spearman &gt; Pearson:</strong> If Spearman exceeds Pearson by more "
        "than 0.08 (flagged in red above), the settleability–TSS relationship is non-proportional. "
        "This is independent confirmation — beyond Sections 1–3 — that non-linear models are "
        "more appropriate for TSS prediction specifically.</li>"
        "<li><strong>MLSS vs SVI trade-off:</strong> Very high MLSS increases the sludge blanket "
        "depth and extends settling time requirements. If MLSS shows a stronger correlation than "
        "SVI, it may indicate that biomass concentration (rather than individual particle "
        "settleability) is the binding constraint — which would point to HRT or wasting rate "
        "as the key operational variable to control effluent TSS.</li>"
        "</ul>"
    )

    return obs


# ── HTML report generator ─────────────────────────────────────────────────────

def img_tag(path):
    if path is None or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0;display:block;">'


def _fold(summary, content_html, open_by_default=False):
    """Return a <details> foldable block."""
    open_attr = " open" if open_by_default else ""
    return (f'<details{open_attr}>\n'
            f'  <summary>{summary}</summary>\n'
            f'  <div class="fold-inner">{content_html}</div>\n'
            f'</details>\n')


def _obs(html):
    """Wrap observation HTML in a styled observations card."""
    if not html:
        return ""
    return f'<div class="obs-card"><h4>Observations</h4>{html}</div>\n'


def build_html(data):
    """
    data keys:
      residuals, pearson (dict grab/comp), mi (dict grab/comp/mi_data),
      scatter (list), aeration_ts, cross_heatmap, stage_removal,
      do_effluent, svi_tss, missing_all, obs (dict s1…s10)
    """
    obs = data.get("obs", {})
    title = "EDA Full Report — All_Years_Full.xlsx"

    CSS = dark_mode_css("""
    body  { font-family: Calibri, Arial, sans-serif; max-width: 1500px;
            margin: 0 auto; padding: 24px; }
    h1    { color: #1F4E79; border-bottom: 3px solid #1F4E79; padding-bottom: 8px;
            font-size: 1.8em; }
    h2    { color: #1F4E79; margin-top: 48px; font-size: 1.25em;
            border-left: 6px solid #2E75B6; padding-left: 12px; }
    p, ul, ol { max-width: 950px; line-height: 1.7; }
    li    { margin-bottom: 4px; }
    .card { padding: 24px 28px; border-radius: 10px; margin-bottom: 28px; }
    .toc  { border-left: 5px solid #2E75B6; border-radius: 6px;
            padding: 16px 24px; margin-bottom: 36px; }
    .toc a { color: #4a90d9; text-decoration: none; }
    .toc a:hover { text-decoration: underline; }
    .badge { display: inline-block; background: #2E75B6; color: white;
             font-size: 0.75em; padding: 2px 8px; border-radius: 10px;
             vertical-align: middle; margin-right: 6px; }
    .badge-decision { background: #C0392B; }
    img   { border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.12); }
    details > summary { cursor: pointer; font-weight: bold; font-size: 1em;
                        color: #2E75B6; padding: 10px 16px; list-style: none;
                        user-select: none; border-radius: 6px; }
    details[open] > summary { border-radius: 6px 6px 0 0; }
    .fold-inner { padding: 12px 16px; }
    .interp { border-radius: 6px; padding: 14px 18px; margin: 14px 0;
              max-width: 950px; }
    .interp h4 { margin: 0 0 8px 0; color: #7a5800; }
    .obs-card { border-left: 5px solid #2171B5; border-radius: 0 6px 6px 0;
                padding: 14px 20px; margin: 20px 0; max-width: 1100px;
                background: var(--obs-bg); }
    .obs-card h4 { margin: 0 0 10px 0; color: #1F4E79; font-size: 1.05em;
                   letter-spacing: 0.02em; }
    .obs-card p, .obs-card ul, .obs-card ol { max-width: 1060px; }
    .obs-table { border-collapse: collapse; width: 100%; font-size: 13px;
                 margin: 10px 0; }
    .obs-table th { background: #4A6FA5; color: white; font-weight: 600;
                    padding: 7px 12px; text-align: center; border: 1px solid #3a5a8a; }
    .obs-table th:first-child, .obs-table th:nth-child(2) { text-align: left; }
    .obs-table td { border: 1px solid #d0d9e8; padding: 6px 12px;
                    vertical-align: middle; }
    .obs-table td:first-child { font-weight: 600; background: #EEF2F7; }
    .obs-table tbody tr:nth-child(even) { background: #F7F9FC; }
    .obs-table tbody tr:nth-child(odd)  { background: #FFFFFF; }
    hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }
    """)

    TARGET_LABELS = {
        "Effluent BOD (mg/L, Grab)":        "Effluent BOD — Grab sample",
        "Effluent COD (mg/L, Grab)":        "Effluent COD — Grab sample",
        "Effluent TSS (mg/L, Grab)":        "Effluent TSS — Grab sample",
        "Effluent pH (Grab)":               "Effluent pH — Grab sample",
        "Effluent BOD (mg/L, Composite)":   "Effluent BOD — Composite sample",
        "Effluent COD (mg/L, Composite)":   "Effluent COD — Composite sample",
        "Effluent TSS (mg/L, Composite)":   "Effluent TSS — Composite sample",
        "Effluent pH (Composite)":          "Effluent pH — Composite sample",
    }

    SCATTER_LABELS = {
        "Effluent BOD (mg/L, Grab)":  "4a — Effluent BOD (Grab)",
        "Effluent COD (mg/L, Grab)":  "4b — Effluent COD (Grab)",
        "Effluent TSS (mg/L, Grab)":  "4c — Effluent TSS (Grab)",
        "Effluent pH (Grab)":         "4d — Effluent pH (Grab)",
    }

    PEARSON_INTERP = """
    <div class="interp">
    <h4>How to read this chart</h4>
    <p>Each feature has two horizontal bars:</p>
    <ul>
      <li><strong style="color:#2171B5">Blue — Pearson r</strong>: measures the
          <em>linear</em> relationship. It assumes that as the feature increases by one
          unit, the target increases by a fixed amount.</li>
      <li><strong style="color:#FD8D3C">Orange — Spearman r</strong>: measures the
          <em>rank / monotonic</em> relationship. It only asks whether the two variables
          move in the same direction, regardless of whether the relationship is a straight
          line or a curve.</li>
    </ul>
    <p><strong>What the gap between the two bars tells you:</strong></p>
    <ul>
      <li><strong>Blue ≈ Orange, both sizable</strong> — the relationship is
          approximately linear. A linear model handles this well.</li>
      <li><strong>Orange noticeably longer than Blue</strong> — the relationship is
          monotonic but <em>curved</em> (e.g., logarithmic, exponential, or step-like).
          As one variable increases the other consistently rises or falls, but not
          proportionally. Linear models will underestimate the strength of this feature;
          non-linear models (RF) capture it naturally.</li>
      <li><strong>Blue noticeably longer than Orange</strong> — a few extreme outliers
          are boosting the linear correlation. Spearman is more robust to outliers.
          Investigate whether those points are genuine or measurement errors.</li>
      <li><strong>Both bars small</strong> — this feature has little direct predictive
          power for this target, or the relationship is complex and non-monotonic.
          Check the Mutual Information chart, which can detect non-monotonic patterns.</li>
    </ul>
    <p><strong>Model selection rule of thumb:</strong> if the top-ranked features
       consistently show Orange &gt; Blue by a wide margin, non-linear models are
       justified. If Pearson ≈ Spearman throughout, linear models are worth trying first.</p>
    </div>"""

    MI_INTERP = """
    <div class="interp">
    <h4>How to read this chart</h4>
    <p><strong>Mutual Information (MI)</strong> measures how much knowing a feature
       reduces uncertainty about the target. Unlike Pearson or Spearman, it is
       completely model-agnostic — it detects linear, curved, step-function, and even
       fully non-monotonic dependencies.</p>
    <ul>
      <li><strong>High MI + high Spearman</strong> — strong, monotonic relationship.
          Both linear and non-linear models benefit from this feature.</li>
      <li><strong>High MI + low Spearman</strong> — the relationship is important but
          non-monotonic (e.g., there is an optimal range). Only non-linear models can
          exploit this. This is the clearest signal that RF is needed.</li>
      <li><strong>Low MI</strong> — the feature contributes little to predicting this
          target and is a candidate for exclusion in feature selection.</li>
    </ul>
    <p>Red bars mark the top 25% of features by MI score for this target.</p>
    </div>"""

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
<h1>{title}</h1>
<p>Full exploratory data analysis on the enriched 60-column dataset
(<code>All_Years_Full.xlsx</code>, 1918 daily rows, 2020–2025).
Sections are ordered by relevance to the key question:
<strong>are non-linear models (Random Forest) justified over linear ones?</strong></p>

<div class="toc">
<strong>Contents</strong>
<ol>
  <li><a href="#s1">Ridge residuals — linearity diagnostic</a>
      <span class="badge badge-decision">Model selection</span></li>
  <li><a href="#s2">Pearson vs Spearman correlation</a>
      <span class="badge badge-decision">Model selection</span></li>
  <li><a href="#s3">Mutual Information scores</a>
      <span class="badge badge-decision">Model selection</span></li>
  <li><a href="#s4">Feature vs target scatter grids</a>
      <span class="badge badge-decision">Model selection</span></li>
  <li><a href="#s5">Cross-stage correlation heatmap</a></li>
  <li><a href="#s6">Data coverage — all columns</a></li>
  <li><a href="#s7">Aeration tank time series</a></li>
  <li><a href="#s8">Stage removal efficiency</a></li>
  <li><a href="#s9">Aeration DO vs effluent quality</a></li>
  <li><a href="#s10">Settleability &amp; biomass vs effluent TSS</a></li>
</ol>
</div>
"""]

    # ── Section 1: Ridge residuals ──────────────────────────────────────────────
    parts.append("""<div class="card" id="s1">
<h2><span class="badge badge-decision">Model selection</span>
1. Ridge residuals — linearity diagnostic</h2>
<p>A Ridge regression model (α = 1, StandardScaler) is trained on process features
(2021–2024) and tested on 2025.
Three residual plots are shown per effluent target:</p>
<ul>
  <li><strong>Residuals vs Fitted</strong> — a random horizontal band around zero
      means linear assumptions hold; any curve, funnel, or pattern means they do not.</li>
  <li><strong>Residuals vs Time</strong> — systematic drift or seasonality that the
      linear model cannot capture (blue = train, red = 2025 test).</li>
  <li><strong>Residuals vs Top Feature</strong> — if residuals correlate with the
      feature most relied upon by Ridge, the relationship is non-linear.</li>
</ul>
""")
    parts.append(img_tag(data.get("residuals")))
    parts.append(_obs(obs.get("s1", "")))
    parts.append("</div>\n")

    # ── Section 2: Pearson vs Spearman ─────────────────────────────────────────
    ps = data.get("pearson", {})
    grab_paths_ps = ps.get("grab", [])
    comp_paths_ps = ps.get("comp", [])

    grab_imgs_ps = "".join(img_tag(p) for p in grab_paths_ps)
    comp_imgs_ps = "".join(img_tag(p) for p in comp_paths_ps)

    parts.append(f"""<div class="card" id="s2">
<h2><span class="badge badge-decision">Model selection</span>
2. Pearson vs Spearman correlation</h2>
{PEARSON_INTERP}
{_fold("▶ Grab effluent targets (click to expand)", grab_imgs_ps, open_by_default=True)}
{_fold("▶ Composite effluent targets (click to expand)", comp_imgs_ps)}
{_obs(obs.get("s2", ""))}
</div>
""")

    # ── Section 3: Mutual Information ──────────────────────────────────────────
    mi = data.get("mi", {})
    grab_paths_mi = [p for p in mi.get("grab", []) if p]
    comp_paths_mi = [p for p in mi.get("comp", []) if p]

    grab_imgs_mi = "".join(img_tag(p) for p in grab_paths_mi)
    comp_imgs_mi = "".join(img_tag(p) for p in comp_paths_mi)

    parts.append(f"""<div class="card" id="s3">
<h2><span class="badge badge-decision">Model selection</span>
3. Mutual Information scores</h2>
{MI_INTERP}
{_fold("▶ Grab effluent targets (click to expand)", grab_imgs_mi, open_by_default=True)}
{_fold("▶ Composite effluent targets (click to expand)", comp_imgs_mi)}
{_obs(obs.get("s3", ""))}
</div>
""")

    # ── Section 4: Scatter grids ───────────────────────────────────────────────
    scatter_paths = data.get("scatter", [])
    scatter_labels = list(SCATTER_LABELS.values())

    scatter_folds = ""
    for path, label in zip(scatter_paths, scatter_labels):
        scatter_folds += _fold(f"▶ {label}", img_tag(path))

    parts.append(f"""<div class="card" id="s4">
<h2><span class="badge badge-decision">Model selection</span>
4. Feature vs target scatter grids</h2>
<p>Top 15 features (ranked by Mutual Information) plotted against each effluent
target. Points are coloured by year. The dashed black line is the linear regression fit.
Look for curvature, fan shapes (heteroscedasticity), or clusters — all indicate that
a linear model will leave systematic error on the table.</p>
{scatter_folds}
{_obs(obs.get("s4", ""))}
</div>
""")

    # ── Section 5: Cross-stage heatmap ─────────────────────────────────────────
    parts.append(f"""<div class="card" id="s5">
<h2>5. Cross-stage correlation heatmap</h2>
<p>All process features vs all eight effluent targets.
Left panel = Pearson, right panel = Spearman. Rows are sorted by mean |Pearson r|
across the four grab targets. This gives a single-view summary of which process
stages and parameters are most associated with effluent quality.</p>
{img_tag(data.get("cross_heatmap"))}
{_obs(obs.get("s5", ""))}
</div>
""")

    # ── Section 6: Missing data ────────────────────────────────────────────────
    parts.append(f"""<div class="card" id="s6">
<h2>6. Data coverage — all columns</h2>
<p>Monthly data coverage (% of days with a recorded value) for every column in
the dataset, grouped by treatment stage. White = no data; darker green = complete.
Use this to understand where gaps fall before modelling — columns with very low
coverage may need imputation or exclusion depending on the missing-data pattern.</p>
{img_tag(data.get("missing_all"))}
{_obs(obs.get("s6", ""))}
</div>
""")

    # ── Section 7: Aeration time series ───────────────────────────────────────
    parts.append(f"""<div class="card" id="s7">
<h2>7. Aeration tank time series</h2>
<p>Dissolved Oxygen (DO), Mixed Liquor Suspended Solids (MLSS), and Sludge Volume
Index (SVI) for the Existing (blue) and New (orange) aeration tanks over 2021–2025.
The dashed red line on DO marks the 0.5 mg/L minimum required for aerobic treatment.
Extended periods below this threshold are expected to correspond to poorer effluent
quality — visible in the scatter charts of Section 9.</p>
{img_tag(data.get("aeration_ts"))}
{_obs(obs.get("s7", ""))}
</div>
""")

    # ── Section 8: Stage removal ───────────────────────────────────────────────
    parts.append(f"""<div class="card" id="s8">
<h2>8. Stage removal efficiency</h2>
<p>Median concentration ± IQR (shaded band) at each treatment stage for BOD, COD, and TSS.
Y-axis is logarithmic. The annotation shows overall % removal from inlet to final effluent.
This chart answers <em>where</em> in the process most removal occurs and how much
variability exists at each stage.</p>
{img_tag(data.get("stage_removal"))}
{_obs(obs.get("s8", ""))}
</div>
""")

    # ── Section 9: DO vs effluent ──────────────────────────────────────────────
    parts.append(f"""<div class="card" id="s9">
<h2>9. Aeration DO vs effluent quality</h2>
<p>Scatter plots of Aeration DO (Existing Tank) against effluent BOD, COD, and TSS.
The dashed red line marks the DO = 0.5 mg/L minimum threshold.
A sharp degradation in effluent quality <em>below</em> the threshold — rather than a
smooth linear decline — would be strong evidence for a threshold / step non-linearity
that RF can capture but a linear model cannot.</p>
{img_tag(data.get("do_effluent"))}
{_obs(obs.get("s9", ""))}
</div>
""")

    # ── Section 10: SVI/MLSS vs TSS ───────────────────────────────────────────
    parts.append(f"""<div class="card" id="s10">
<h2>10. Settleability &amp; biomass vs effluent TSS</h2>
<p>SVI (Sludge Volume Index), MLSS, and SV30 for both aeration tanks plotted against
effluent TSS. Poor settling (high SVI) should produce high effluent TSS because
particles fail to settle out of the final effluent. Pearson r is shown in each panel
title. If the relationship is present but non-linear, SVI and SV30 are strong
candidates for the feature selection pool.</p>
{img_tag(data.get("svi_tss"))}
{_obs(obs.get("s10", ""))}
</div>
""")

    parts.append("</body>\n</html>")
    html = "\n".join(parts)
    with open(REPORT, "w") as f:
        f.write(html)
    print(f"  report: {REPORT}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {EXCEL_IN}...")
    df = pd.read_excel(EXCEL_IN, parse_dates=["Date"])
    print(f"  {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Generating charts...")
    missing_all   = plot_missing_all(df)
    pearson_data  = plot_pearson_vs_spearman(df)
    mi_result     = plot_mutual_information(df)
    all_mi        = mi_result["mi_data"]
    scatter_paths = plot_scatter_grids(df, all_mi)
    aeration_ts   = plot_aeration_timeseries(df)
    cross_heatmap = plot_cross_stage_heatmap(df)
    stage_removal = plot_stage_removal(df)
    do_effluent   = plot_do_vs_effluent(df)
    svi_tss       = plot_svi_mlss_vs_tss(df)
    residuals     = plot_linearity_residuals(df)

    n_charts = (1 + len(pearson_data["grab"]) + len(pearson_data["comp"])
                + len([p for p in mi_result["grab"] if p])
                + len([p for p in mi_result["comp"] if p])
                + len(scatter_paths) + 5 + 1)

    print("\nComputing observations...")
    section_obs = build_observations(df)

    print("Building HTML report...")
    build_html({
        "residuals":    residuals,
        "pearson":      pearson_data,
        "mi":           mi_result,
        "scatter":      scatter_paths,
        "aeration_ts":  aeration_ts,
        "cross_heatmap":cross_heatmap,
        "stage_removal":stage_removal,
        "do_effluent":  do_effluent,
        "svi_tss":      svi_tss,
        "missing_all":  missing_all,
        "obs":          section_obs,
    })
    print(f"\nDone. ~{n_charts} chart files → {PLOTS_DIR}/")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
