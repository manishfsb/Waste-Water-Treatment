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

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modeling"))
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

    # ── Interpretation threshold lines ────────────────────────────────────
    _thresholds = [
        (0.20, "Weak",     "#999999", ":"),
        (0.40, "Moderate", "#E07B20", "--"),
        (0.60, "Strong",   "#C0392B", "--"),
    ]
    for val, lbl, col, ls in _thresholds:
        for sign in (1, -1):
            ax.axvline(sign * val, color=col, linestyle=ls,
                       linewidth=0.9, alpha=0.70, zorder=0)
        # label at top of the positive line only
        ax.text(val + 0.012, 0.995, lbl,
                transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=col, va="top", ha="left",
                alpha=0.90, rotation=90)

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

    # ── Interpretation threshold lines ────────────────────────────────────
    _mi_thresholds = [
        (0.05, "Weak",     "#999999", ":"),
        (0.15, "Moderate", "#E07B20", "--"),
        (0.30, "Strong",   "#C0392B", "--"),
    ]
    for val, lbl, col, ls in _mi_thresholds:
        ax.axvline(val, color=col, linestyle=ls,
                   linewidth=0.9, alpha=0.70, zorder=0)
        ax.text(val + 0.004, 0.995, lbl,
                transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=col, va="top", ha="left",
                alpha=0.90, rotation=90)

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

def compute_top_ridge_features(df):
    """Return {target: feature_name} — the feature with the largest Ridge |coef| per target."""
    feats    = feature_cols(df)
    usable   = [f for f in feats if f in df.columns]
    train_df = df[df["Date"].dt.year.isin([2021, 2022, 2023, 2024])]
    top = {}
    for tgt in GRAB_TARGETS + COMP_TARGETS:
        tr = train_df[usable + [tgt]].dropna()
        if len(tr) < 20:
            continue
        sc    = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(tr[usable].values), tr[tgt].values)
        top[tgt] = pd.Series(np.abs(ridge.coef_), index=usable).idxmax()
    return top


def plot_ridge_coefficients(df):
    """
    For each target, fit Ridge on training data and plot the top 10 features by |coefficient|
    as a signed horizontal bar chart.  Returns {"grab": {tgt: path}, "comp": {tgt: path}}.
    Bars are coloured by sign (blue = positive relationship, red = inverse).
    X-axis shows signed coefficients on the StandardScaler scale.
    """
    print("10b. Ridge coefficient charts...")
    feats    = feature_cols(df)
    usable   = [f for f in feats if f in df.columns]
    train_df = df[df["Date"].dt.year.isin([2021, 2022, 2023, 2024])]
    results  = {"grab": {}, "comp": {}}

    for tgt in GRAB_TARGETS + COMP_TARGETS:
        group = "grab" if tgt in GRAB_TARGETS else "comp"
        slug  = (tgt.replace("/", "_").replace(" ", "_")
                    .replace("(", "").replace(")", "").replace(",", ""))

        tr = train_df[usable + [tgt]].dropna()
        if len(tr) < 20:
            results[group][tgt] = None
            continue

        sc    = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(tr[usable].values), tr[tgt].values)

        coef_s   = pd.Series(ridge.coef_, index=usable)
        top10    = coef_s.abs().nlargest(10).index
        top10_s  = coef_s[top10].sort_values()          # ascending so largest |coef| is at top
        colours  = ["#D94801" if v < 0 else "#2171B5" for v in top10_s]

        fig, ax = plt.subplots(figsize=(11, 4))
        bars = ax.barh(range(len(top10_s)), top10_s.values, color=colours, height=0.6)
        ax.set_yticks(range(len(top10_s)))
        ax.set_yticklabels([s(f) for f in top10_s.index], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Ridge coefficient (StandardScaler scale)", fontsize=9)
        ax.set_title(
            f"Ridge coefficient weights — {s(tgt)}\n"
            "Top 10 features by |coefficient|  •  "
            "Blue = positive relationship  •  Red = inverse",
            fontsize=10, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        # Dim note about multicollinearity
        fig.text(
            0.01, 0.01,
            "⚠  Correlated features share weight — individual bars understate importance of "
            "collinear groups.  These weights reflect what Ridge relied on, not general feature importance.",
            fontsize=7.5, color="#666", style="italic"
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        results[group][tgt] = save(fig, f"10b_coef_{slug}")

    return results


def plot_scatter_grids(df, all_mi, top_ridge_feats=None):
    """top_ridge_feats: dict {target: feature_name} — subplot for that feature gets an orange border."""
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

        top_feats   = [f for f in ranked.index if f in df.columns][:15]
        highlight   = (top_ridge_feats or {}).get(target)

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
                title_txt = f"{s(feat)}\n(r={r:.2f})"
            else:
                title_txt = s(feat)

            is_top_ridge = (highlight is not None and feat == highlight)
            if is_top_ridge:
                # Orange border + subtle background to flag the top Ridge coefficient feature
                for spine in ax.spines.values():
                    spine.set_edgecolor("#E07B20")
                    spine.set_linewidth(2.5)
                ax.set_facecolor("#FFF8F0")
                ax.set_title(f"{title_txt}\n★ top Ridge coef", fontsize=8, pad=2,
                             color="#E07B20", fontweight="bold")
            else:
                ax.set_title(title_txt, fontsize=8, pad=2)

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
        if highlight:
            handles.append(plt.Line2D([0], [0], color="#E07B20", linewidth=3))
            labels = [str(y) for y in sorted(YEAR_COLOURS)] + ["★ top Ridge coef"]
        else:
            labels = [str(y) for y in sorted(YEAR_COLOURS)]
        fig.legend(handles, labels, title="Year", loc="lower right", ncol=6, fontsize=8)

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
    """
    Full square correlation matrix: all features + all targets × themselves.
    Returns (path, table_html) where table_html is an HTML string listing
    every pair with max(|Pearson|, |Spearman|) >= THRESHOLD.
    """
    print("6. Cross-stage correlation heatmap (full matrix)...")
    feats   = feature_cols(df)
    targets = GRAB_TARGETS + COMP_TARGETS
    all_cols = feats + [t for t in targets if t not in feats]

    # ── Build full Pearson and Spearman matrices ────────────────────────────────
    n = len(all_cols)
    pears_arr = np.full((n, n), np.nan)
    spear_arr = np.full((n, n), np.nan)

    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i == j:          # keep diagonal as NaN (self-correlation)
                continue
            sub = df[[ci, cj]].dropna()
            if len(sub) < 10:
                continue
            pears_arr[i, j], _ = stats.pearsonr(sub[ci],  sub[cj])
            spear_arr[i, j], _ = stats.spearmanr(sub[ci], sub[cj])

    short = [s(c) for c in all_cols]
    pears_df = pd.DataFrame(pears_arr, index=short, columns=short)
    spear_df = pd.DataFrame(spear_arr, index=short, columns=short)

    # ── Threshold filter: keep rows AND columns where any |r| > 0.5 ─────────────
    THRESHOLD = 0.5
    max_abs = pears_df.abs().combine(spear_df.abs(), np.fmax)   # NaN-safe element-wise max
    active  = max_abs.max(axis=1) > THRESHOLD                   # rows with signal

    n_total  = n
    n_shown  = int(active.sum())
    n_hidden = n_total - n_shown

    pears_f = pears_df.loc[active, active]
    spear_f = spear_df.loc[active, active]

    subtitle = (
        f"Showing {n_shown} of {n_total} columns  ·  "
        f"threshold: |Pearson| or |Spearman| > {THRESHOLD}  ·  "
        f"{n_hidden} column{'s' if n_hidden != 1 else ''} below threshold hidden"
    )

    # ── Compact figure (square matrix — no need to be huge) ────────────────────
    cell = max(0.55, min(1.0, 14.0 / n_shown))   # cell size in inches
    side = max(8, n_shown * cell)
    side = min(side, 20)                          # cap at 20 inches
    fig, axes = plt.subplots(2, 1, figsize=(side, side * 2 + 1))

    kws = dict(
        cmap="RdBu_r", vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.5, "label": "r"},
        annot=True, fmt=".2f", annot_kws={"size": max(5, min(8, int(120 / n_shown)))},
    )

    sns.heatmap(pears_f, ax=axes[0], **kws)
    axes[0].set_title("Pearson correlation", fontsize=11, fontweight="bold", pad=8)

    sns.heatmap(spear_f, ax=axes[1], **kws)
    axes[1].set_title("Spearman correlation", fontsize=11, fontweight="bold", pad=8)

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                           fontsize=max(6, min(9, int(110 / n_shown))))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=max(6, min(9, int(110 / n_shown))))
        ax.set_xlabel(""); ax.set_ylabel("")

    fig.suptitle(
        "Full correlation matrix — all features & targets vs each other\n" + subtitle,
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = save(fig, "06_cross_stage_heatmap")

    # ── Build pair table (upper triangle, sorted by |Pearson| desc) ─────────────
    pairs = []
    cols_f = list(pears_f.columns)
    for ii, ca in enumerate(cols_f):
        for jj, cb in enumerate(cols_f):
            if jj <= ii:      # lower triangle + diagonal → skip
                continue
            pr = pears_f.loc[ca, cb]
            sr = spear_f.loc[ca, cb]
            if pd.isna(pr) and pd.isna(sr):
                continue
            max_r = max(abs(pr) if pd.notna(pr) else 0,
                        abs(sr) if pd.notna(sr) else 0)
            if max_r >= THRESHOLD:
                pairs.append((ca, cb, pr, sr, max_r))

    pairs.sort(key=lambda x: x[4], reverse=True)   # sort by max |r|

    def _rfmt(v):
        if pd.isna(v): return "—"
        clr = "#C0392B" if abs(v) >= 0.75 else ("#E07B20" if abs(v) >= 0.5 else "")
        sign = "+" if v > 0 else ""
        txt  = f"{sign}{v:.2f}"
        return f"<span style='color:{clr};font-weight:600'>{txt}</span>" if clr else txt

    TD = "padding:6px 12px;border-bottom:1px solid #2e3d52;color:#b8c8dc"
    TD_R = f"{TD};text-align:right;font-variant-numeric:tabular-nums"
    TD_ARROW = "padding:6px 4px;border-bottom:1px solid #2e3d52;color:#4a5a70;text-align:center"

    # ── Classify pairs into sub-sections ────────────────────────────────────────
    GRAB_TGT_SHORT = {s(t) for t in GRAB_TARGETS}
    COMP_TGT_SHORT = {s(t) for t in COMP_TARGETS}
    ALL_TGT_SHORT  = GRAB_TGT_SHORT | COMP_TGT_SHORT

    feat_feat   = []
    feat_grab   = []
    feat_comp   = []
    tgt_gc      = []   # grab vs composite
    tgt_gg      = []   # grab vs grab
    tgt_cc      = []   # composite vs composite

    for pair in pairs:
        ca, cb = pair[0], pair[1]
        a_tgt = ca in ALL_TGT_SHORT
        b_tgt = cb in ALL_TGT_SHORT
        if not a_tgt and not b_tgt:
            feat_feat.append(pair)
        elif a_tgt and b_tgt:
            a_grab = ca in GRAB_TGT_SHORT
            b_grab = cb in GRAB_TGT_SHORT
            if a_grab != b_grab:
                tgt_gc.append(pair)
            elif a_grab:
                tgt_gg.append(pair)
            else:
                tgt_cc.append(pair)
        else:
            # one is a target — normalise so the target is always cb for label purposes
            tgt_col = ca if a_tgt else cb
            if tgt_col in GRAB_TGT_SHORT:
                feat_grab.append(pair)
            else:
                feat_comp.append(pair)

    TH = "padding:8px 12px;text-align:left;color:#8aa8cc;font-weight:600;letter-spacing:0.03em"
    TH_R = "padding:8px 12px;text-align:right;color:#8aa8cc;font-weight:600;letter-spacing:0.03em"

    def _sub_table(sub_pairs, heading, open_by_default=False):
        if not sub_pairs:
            return ""
        rows = ""
        for i, (ca, cb, pr, sr, _) in enumerate(sub_pairs):
            row_bg = "#141e2d" if i % 2 == 0 else "#192336"
            rows += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{TD}'>{ca}</td>"
                f"<td style='{TD_ARROW}'>↔</td>"
                f"<td style='{TD}'>{cb}</td>"
                f"<td style='{TD_R}'>{_rfmt(pr)}</td>"
                f"<td style='{TD_R}'>{_rfmt(sr)}</td>"
                f"</tr>"
            )
        open_attr = " open" if open_by_default else ""
        n = len(sub_pairs)
        return f"""
      <details{open_attr} style='margin:8px 0 8px 16px'>
        <summary style='font-weight:600;font-size:0.88rem;cursor:pointer;padding:4px 0;
                        color:#7a98bc;letter-spacing:0.01em'>
          {heading} &mdash; {n} pair{'s' if n != 1 else ''}
        </summary>
        <div style='overflow-x:auto;margin-top:8px;border-radius:6px;
                    border:1px solid #2e3d52;overflow:hidden'>
        <table style='border-collapse:collapse;font-size:0.84rem;width:100%;
                      max-width:760px;background:#111929'>
          <thead>
            <tr style='background:#1d2f45;border-bottom:2px solid #3a5070'>
              <th style='{TH}'>Column A</th>
              <th style='padding:8px 4px'></th>
              <th style='{TH}'>Column B</th>
              <th style='{TH_R}'>Pearson r</th>
              <th style='{TH_R}'>Spearman r</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
      </details>"""

    feat_feat_block = _sub_table(feat_feat, "Features vs Features", open_by_default=True)

    feat_tgt_subs = (
        _sub_table(feat_grab, "Features vs Grab Targets", open_by_default=True)
        + _sub_table(feat_comp, "Features vs Composite Targets", open_by_default=True)
    )
    n_feat_tgt = len(feat_grab) + len(feat_comp)
    feat_tgt_block = f"""
      <details open style='margin:8px 0'>
        <summary style='font-weight:600;font-size:0.91rem;cursor:pointer;padding:5px 0;
                        color:#8aa8cc;letter-spacing:0.01em'>
          Features vs Targets &mdash; {n_feat_tgt} pair{'s' if n_feat_tgt != 1 else ''}
        </summary>
        {feat_tgt_subs}
      </details>""" if n_feat_tgt else ""

    tgt_tgt_subs = (
        _sub_table(tgt_gg, "Grab vs Grab")
        + _sub_table(tgt_cc, "Composite vs Composite")
        + _sub_table(tgt_gc, "Grab vs Composite")
    )
    n_tgt_tgt = len(tgt_gg) + len(tgt_cc) + len(tgt_gc)
    tgt_tgt_block = f"""
      <details style='margin:8px 0'>
        <summary style='font-weight:600;font-size:0.91rem;cursor:pointer;padding:5px 0;
                        color:#8aa8cc;letter-spacing:0.01em'>
          Targets vs Targets &mdash; {n_tgt_tgt} pair{'s' if n_tgt_tgt != 1 else ''}
          &nbsp;<span style='font-size:0.78rem;color:#556070;font-weight:400'>
            (grab and composite targets never appear in the same model dataset)</span>
        </summary>
        {tgt_tgt_subs}
      </details>""" if n_tgt_tgt else ""

    feat_feat_outer = f"""
      <details open style='margin:8px 0'>
        <summary style='font-weight:600;font-size:0.91rem;cursor:pointer;padding:5px 0;
                        color:#8aa8cc;letter-spacing:0.01em'>
          Features vs Features &mdash; {len(feat_feat)} pair{'s' if len(feat_feat) != 1 else ''}
          &nbsp;<span style='font-size:0.78rem;color:#556070;font-weight:400'>
            (collinearity — check these before linear modelling)</span>
        </summary>
        {feat_feat_block}
      </details>""" if feat_feat else ""

    table_html = f"""
    <details open style='margin:16px 0'>
      <summary style='font-weight:600;font-size:0.93rem;cursor:pointer;padding:6px 0;
                      color:#8aa8cc;letter-spacing:0.01em'>
        Correlated pairs &mdash; {len(pairs)} pair{'s' if len(pairs)!=1 else ''}
        with |Pearson| or |Spearman| &ge; {THRESHOLD}
        &nbsp;&middot;&nbsp; sorted by strongest correlation within each group
      </summary>
      <p style='font-size:0.77rem;color:#556070;margin:8px 0 4px'>
        <span style='color:#C0392B;font-weight:600'>Red</span> = |r| &ge; 0.75 (strong)
        &nbsp;&middot;&nbsp;
        <span style='color:#E07B20;font-weight:600'>Orange</span> = |r| 0.50&ndash;0.74 (moderate)
      </p>
      {feat_feat_outer}
      {feat_tgt_block}
      {tgt_tgt_block}
    </details>
    """

    return path, table_html



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
    """Return {"grab": {target: path}, "comp": {target: path}} — one 1×3 figure per target."""
    print("10. Linearity residuals (Ridge)...")
    feats = feature_cols(df)
    usable_feats = [f for f in feats if f in df.columns]
    train_df = df[df["Date"].dt.year.isin([2021, 2022, 2023, 2024])]
    test_df  = df[df["Date"].dt.year == 2025]

    results = {"grab": {}, "comp": {}}

    for target in GRAB_TARGETS + COMP_TARGETS:
        group = "grab" if target in GRAB_TARGETS else "comp"
        slug  = target.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        tr = train_df[usable_feats + [target]].dropna()
        te = test_df[usable_feats  + [target]].dropna()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Ridge residuals — {s(target)}\n"
                     "(random scatter = linear relationship; patterns = non-linearity)",
                     fontsize=11, fontweight="bold")

        if len(tr) < 20:
            for ax in axes:
                ax.set_visible(False)
            results[group][target] = save(fig, f"10_resid_{slug}")
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

        yhat_tr  = ridge.predict(X_tr_s)
        resid_tr = y_tr - yhat_tr

        # Panel A: residuals vs fitted (train)
        ax = axes[0]
        ax.scatter(yhat_tr, resid_tr, color="#2171B5", alpha=0.3, s=12)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Fitted values — Ridge predictions on training set")
        ax.set_ylabel("Residuals (actual − predicted)")
        ax.set_title("Residuals vs Fitted", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        # Panel B: residuals over time
        ax = axes[1]
        ax.scatter(tr.index, resid_tr, color="#2171B5", alpha=0.3, s=8, label="Train 2021–2024")
        if len(te) > 0:
            yhat_te  = ridge.predict(X_te_s)
            resid_te = y_te - yhat_te
            ax.scatter(te.index, resid_te, color="#D94801", alpha=0.4, s=8, label="Test 2025")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel("Row index (time)")
        ax.set_ylabel("Residuals (actual − predicted)")
        ax.set_title("Residuals vs Time", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel C: residuals vs top feature by |coefficient|
        coef     = pd.Series(np.abs(ridge.coef_), index=usable_feats)
        top_feat = coef.idxmax()
        ax = axes[2]
        ax.scatter(tr[top_feat].values, resid_tr, color="#2171B5", alpha=0.3, s=12)
        ax.axhline(0, color="red", linewidth=1)
        ax.set_xlabel(s(top_feat))
        ax.set_ylabel("Residuals (actual − predicted)")
        ax.set_title(f"Residuals vs {s(top_feat)}\n(highest |coef| feature)", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        results[group][target] = save(fig, f"10_resid_{slug}")

    return results


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

    # ── S1: Ridge metrics — one observation block per target ─────────────────
    def _ridge_metrics(tgt):
        """Fit Ridge, return a dict of metrics for one target."""
        # ── Row count accounting ─────────────────────────────────────────────
        n_total        = len(df)
        n_train_tgt    = int(trndf[tgt].notna().sum())   # rows with target in 2021–2024
        n_test_tgt     = int(tstdf[tgt].notna().sum())   # rows with target in 2025
        n_feats        = len(usable)

        tr = trndf[usable + [tgt]].dropna()
        te = tstdf[usable + [tgt]].dropna()
        if len(tr) < 20:
            return None

        n_train_used    = len(tr)
        n_test_used     = len(te)
        n_dropped_train = n_train_tgt - n_train_used
        pct_dropped     = (n_dropped_train / n_train_tgt * 100) if n_train_tgt > 0 else 0.0

        Xtr, ytr = tr[usable].values, tr[tgt].values
        Xte, yte = te[usable].values, te[tgt].values
        sc    = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(Xtr), ytr)
        yhat_tr    = ridge.predict(sc.transform(Xtr))
        resid_tr   = ytr - yhat_tr
        funnel_r, _ = stats.spearmanr(yhat_tr, np.abs(resid_tr))
        top_feat   = pd.Series(np.abs(ridge.coef_), index=usable).idxmax()
        m = {
            "tgt":            s(tgt),
            "r2_tr":          r2_score(ytr, yhat_tr),
            "rmse_tr":        np.sqrt(mse(ytr, yhat_tr)),
            "top":            s(top_feat),
            "funnel":         funnel_r,
            # row accounting
            "n_total":        n_total,
            "n_train_tgt":    n_train_tgt,
            "n_train_used":   n_train_used,
            "n_test_used":    n_test_used,
            "n_dropped_train": n_dropped_train,
            "pct_dropped":    pct_dropped,
            "n_feats":        n_feats,
        }
        if len(te) >= 5:
            yhat_te     = ridge.predict(sc.transform(Xte))
            m["r2_te"]  = r2_score(yte, yhat_te)
            m["rmse_te"] = np.sqrt(mse(yte, yhat_te))
            m["gap"]    = m["r2_tr"] - m["r2_te"]
        return m

    def _s1_obs_for(m):
        """Build observation HTML for a single target's Ridge metrics dict."""
        r2_te_s   = f"{m['r2_te']:.3f}"   if "r2_te"   in m else "—"
        rmse_te_s = f"{m['rmse_te']:.2f}" if "rmse_te" in m else "—"
        gap_s     = f"{m['gap']:.3f}"     if "gap"     in m else "—"
        r2_color  = "#C0392B" if m.get("r2_te", 1) < 0 else "#27AE60"
        gap_color = ("#C0392B" if m.get("gap", 0) > 0.15
                     else "#27AE60" if m.get("gap", 0) < 0.05
                     else "inherit")

        row_info = (
            f"<p style='font-size:0.86em;margin:0 0 10px 0;border-left:3px solid #4A6FA5;"
            f"padding-left:10px;color:var(--text)'>"
            f"<strong>Rows used for this model:</strong>&nbsp; "
            f"{m['n_total']:,} total rows in dataset &nbsp;→&nbsp; "
            f"{m['n_train_tgt']:,} rows with target present in 2021–2024 &nbsp;→&nbsp; "
            f"<strong>{m['n_train_used']:,} rows used for Ridge training</strong> "
            f"({m['n_dropped_train']:,} dropped, {m['pct_dropped']:.1f}%). "
            f"&nbsp; Test (2025): {m['n_test_used']:,} rows."
            f"<br><em style='color:#888'>Criteria: any row missing a value in any of the "
            f"{m['n_feats']} feature columns is excluded from that target's model.</em>"
            f"</p>"
        )

        tbl = (
            "<table style='width:100%;margin:10px 0'>"
            "<thead><tr>"
            "<th>Train R²</th><th>Test R²</th><th>R² Gap (train − test)</th>"
            "<th>Train RMSE</th><th>Test RMSE</th>"
            "<th>Top feature (|coef|)</th>"
            "</tr></thead><tbody><tr>"
            f"<td>{m['r2_tr']:.3f}</td>"
            f"<td style='color:{r2_color};font-weight:bold'>{r2_te_s}</td>"
            f"<td style='color:{gap_color}'>{gap_s}</td>"
            f"<td>{m['rmse_tr']:.2f}</td>"
            f"<td>{rmse_te_s}</td>"
            f"<td style='font-size:0.9em'>{m['top']}</td>"
            "</tr></tbody></table>"
        )

        bullets = []
        if m.get("r2_te", 1) < 0:
            bullets.append(
                "<li><strong>Negative test R²:</strong> Ridge performs worse than predicting "
                "the training mean — it adds no predictive value on 2025 data.</li>")
        elif m.get("gap", 0) > 0.15:
            bullets.append(
                "<li><strong>Notable R² gap:</strong> Train and test R² diverge significantly, "
                "suggesting the linear pattern learned from 2021–2024 does not fully transfer "
                "to 2025 conditions.</li>")

        if abs(m.get("funnel", 0)) > 0.20:
            direction = "grows" if m["funnel"] > 0 else "shrinks"
            bullets.append(
                f"<li><strong>Potential heteroscedasticity</strong> (Spearman r = {m['funnel']:.2f} "
                f"between |residuals| and fitted values): error {direction} with the predicted value. "
                "This is a standalone statistical test on the training residuals — unrelated to the "
                "feature correlations in Section 2. Inspect the Residuals vs Fitted panel above "
                "to judge whether this is visually apparent.</li>")

        bullets.append(
            f"<li><strong>Top relied-on feature:</strong> <em>{m['top']}</em> carries the largest "
            "Ridge coefficient magnitude. If this feature has a curved relationship with the target "
            "Its scatter panel is highlighted with an orange border in Section 2 — check it "
            "for curvature or fan shapes.</li>")

        ul = "<ul>" + "".join(bullets) + "</ul>" if bullets else ""
        return row_info + tbl + ul

    obs["s1"] = {}
    for tgt in GRAB_TARGETS + COMP_TARGETS:
        m = _ridge_metrics(tgt)
        if m:
            obs["s1"][tgt] = _s1_obs_for(m)

    # ── S2: Pearson vs Spearman — per-target blocks ───────────────────────────
    def _ps_block(tgt):
        sub = df[usable + [tgt]].dropna()
        if len(sub) < 20:
            return None
        X, y  = sub[usable], sub[tgt]
        p_r   = X.apply(lambda c: stats.pearsonr(c,  y)[0])
        sp_r  = X.apply(lambda c: stats.spearmanr(c, y)[0])
        gap   = sp_r.abs() - p_r.abs()
        top5  = sp_r.abs().nlargest(5).index
        pct_nl = (gap > 0.10).mean() * 100
        tbl = ""
        for f in top5:
            gc = "#C0392B" if gap[f] > 0.10 else "var(--text)"
            tbl += (f"<tr><td>{s(f)}</td>"
                    f"<td style='color:#2171B5'>{p_r[f]:+.3f}</td>"
                    f"<td style='color:#FD8D3C'>{sp_r[f]:+.3f}</td>"
                    f"<td style='color:{gc}'>{gap[f]:+.3f}</td></tr>")
        return (
            "<table style='width:auto;min-width:500px;font-size:0.88em'>"
            "<thead><tr><th>Feature</th><th>Pearson r</th><th>Spearman r</th>"
            "<th>Gap (|Spear|−|Pear|)</th></tr></thead>"
            f"<tbody>{tbl}</tbody></table>"
            f"<p style='font-size:0.87em;margin:4px 0 0 0'>"
            f"{pct_nl:.0f}% of all {len(usable)} features show |Spearman| &gt; |Pearson| "
            f"by more than 0.10 — suggesting predominantly curved or non-monotonic "
            f"relationships for this target.</p>"
        )

    s2_general = (
        "<ul style='margin-top:4px'>"
        "<li><strong>Positive Gap (red):</strong> Spearman exceeds Pearson — the relationship "
        "is curved (logarithmic, exponential, or step-like). A linear model undervalues these "
        "features.</li>"
        "<li><strong>Negative Gap:</strong> Extreme outliers inflate Pearson while Spearman is "
        "unaffected. These features may appear deceptively important in a linear model.</li>"
        "<li><strong>Both magnitudes small:</strong> Low Pearson AND Spearman does not mean "
        "uninformative — check Mutual Information in Section 4, which detects non-monotonic "
        "patterns that rank correlation misses entirely.</li>"
        "<li><strong>Ridge failure context:</strong> When most top-ranked features show positive "
        "gaps, a linear model systematically mis-specifies their contributions — consistent with "
        "the Ridge residual patterns in Section 1.</li>"
        "</ul>"
    )

    obs["s2"] = {
        "grab":    {tgt: _ps_block(tgt) for tgt in GRAB_TARGETS},
        "comp":    {tgt: _ps_block(tgt) for tgt in COMP_TARGETS},
        "general": s2_general,
    }

    # ── S3: Mutual Information — per-target blocks ────────────────────────────
    def _mi_block(tgt):
        sub = df[usable + [tgt]].dropna()
        if len(sub) < 20:
            return None
        X_vals, y_vals = sub[usable].values, sub[tgt].values
        mi_scores = mutual_info_regression(X_vals, y_vals, random_state=42)
        mi_s  = pd.Series(mi_scores, index=usable)
        sp_r  = pd.Series(
            [stats.spearmanr(sub[f], sub[tgt])[0] for f in usable], index=usable
        )
        top5_mi = mi_s.nlargest(5)
        mi_thr  = mi_s.quantile(0.75)
        nonlin  = mi_s[(mi_s >= mi_thr) & (sp_r.abs() < 0.30)].nlargest(3)
        tbl = ""
        for f in top5_mi.index:
            is_nl    = (mi_s[f] >= mi_thr) and (abs(sp_r[f]) < 0.30)
            nl_color = "#C0392B" if is_nl else "var(--text)"
            nl_label = "⚠ Non-monotonic" if is_nl else "Monotonic"
            tbl += (f"<tr><td>{s(f)}</td>"
                    f"<td>{mi_s[f]:.4f}</td>"
                    f"<td style='color:#FD8D3C'>{sp_r[f]:+.3f}</td>"
                    f"<td style='color:{nl_color}'>{nl_label}</td></tr>")
        nonlin_note = ""
        if len(nonlin) > 0:
            names = ", ".join(s(f) for f in nonlin.index)
            nonlin_note = (
                f"<p style='font-size:0.87em;color:#C0392B;margin:6px 0'>"
                f"⚠ <strong>Non-monotonic (top-quartile MI, |Spearman| &lt; 0.30):</strong> "
                f"{names} — real predictive signal, but only exploitable by non-linear models.</p>"
            )
        return (
            "<table style='width:auto;min-width:500px;font-size:0.88em'>"
            "<thead><tr><th>Feature</th><th>MI score</th><th>Spearman r</th>"
            "<th>Relationship type</th></tr></thead>"
            f"<tbody>{tbl}</tbody></table>{nonlin_note}"
        )

    s3_general = (
        "<ul style='margin-top:4px'>"
        "<li><strong>High MI, low |Spearman|:</strong> The feature has a real but non-monotonic "
        "relationship with the target — an optimal range, a threshold, or seasonal interaction "
        "that rank correlation misses entirely.</li>"
        "<li><strong>High MI and high |Spearman|:</strong> The most reliable predictors — strong "
        "signal regardless of model family.</li>"
        "<li><strong>Near-zero MI:</strong> Strong candidates for exclusion — they add noise "
        "without information.</li>"
        "</ul>"
    )

    obs["s3"] = {
        "grab":    {tgt: _mi_block(tgt) for tgt in GRAB_TARGETS},
        "comp":    {tgt: _mi_block(tgt) for tgt in COMP_TARGETS},
        "general": s3_general,
    }

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
        "<li><strong>Cross-reference with Sections 3 and 4:</strong> The scatter plots show the "
        "top 15 features by Mutual Information. Features identified as non-monotonic in Section 4 "
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
    """Return a <details> foldable block.
    The ▶ icon is wrapped in .fold-icon so CSS can rotate it when open.
    JS (added at end of body) dynamically swaps the hint text.
    Strip any caller-supplied hint like '(click to expand)' — it's added automatically.
    """
    import re as _re
    label = _re.sub(r"\s*\(click to expand\)", "", summary, flags=_re.IGNORECASE).strip()
    # Move leading arrow into its own span; if caller passes ▶ prefix, extract it
    if label.startswith("▶"):
        icon  = '<span class="fold-icon">▶</span>'
        label = label[1:].strip()
    else:
        icon  = '<span class="fold-icon">▶</span>'
    open_attr = " open" if open_by_default else ""
    return (f'<details{open_attr}>\n'
            f'  <summary>{icon} <span class="fold-label">{label}</span>'
            f'<span class="fold-hint"></span></summary>\n'
            f'  <div class="fold-inner">{content_html}</div>\n'
            f'</details>\n')


def _obs(html):
    """Wrap observation HTML in a styled observations card."""
    if not html:
        return ""
    return f'<div class="obs-card"><h4>Observations</h4>{html}</div>\n'


# ── Section 11: Feature suggestions for Experiment 3 ──────────────────────────

def build_suggestions(df):
    """
    Per grab-effluent target, apply the 4-question framework to every candidate
    feature not already in Experiment 2 Sub-2 and return an HTML string for
    Section 11 of the report.

    Framework:
      Q1 — Is there any signal?  → MI score
      Q2 — What kind of signal?  → Pearson + Spearman pattern
      Q3 — What does it cost?    → missingness %
      Q4 — Is it redundant?      → known collinear pairs
    """
    print("11. Feature suggestions (Experiment 3)...")

    # ── Feature sets ──────────────────────────────────────────────────────────
    EXP2_FEATS = {
        "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)", "Inlet COD (mg/L, Grab)",
        "Inlet TSS (mg/L, Grab)",
        "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)", "Sec Clarifier BOD (mg/L)",
        "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
        "Sec Sed pH", "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)",
        "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
        "Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year",
    }
    ALL_EFFLUENT = {
        "Effluent BOD (mg/L, Grab)", "Effluent COD (mg/L, Grab)",
        "Effluent TSS (mg/L, Grab)", "Effluent pH (Grab)",
        "Effluent BOD (mg/L, Composite)", "Effluent COD (mg/L, Composite)",
        "Effluent TSS (mg/L, Composite)", "Effluent pH (Composite)",
        "Effluent FRC (mg/L)", "Effluent O&G (mg/L)", "Effluent NH3-N (mg/L)",
        "Effluent Total Coliform (CFU/100ml)", "Effluent Fecal Coliform (CFU/100ml)",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if c not in EXP2_FEATS and c not in ALL_EFFLUENT]

    # ── Redundancy register ────────────────────────────────────────────────────
    # (feature_to_drop → (preferred_alternative, explanation))
    REDUNDANT = {
        "Power NEA (KW)": (
            "Power Total (KW)",
            "Near-perfect collinearity (Pearson = 0.975, Spearman = 0.940). "
            "Power Total already incorporates NEA's contribution and is in every experiment."
        ),
        "Aeration MLVSS (mg/L, Existing)": (
            "Aeration MLSS (mg/L, Existing)",
            "High collinearity with MLSS (Pearson = 0.882, Spearman = 0.911). "
            "MLSS is more widely measured; keep MLVSS only if volatile fraction "
            "is specifically hypothesised to add independent signal."
        ),
        "Aeration MLVSS (mg/L, New)": (
            "Aeration MLSS (mg/L, New)",
            "High collinearity with MLSS (Pearson = 0.871, Spearman = 0.892). Same logic as Existing tank."
        ),
    }
    # Note: Power GE is NOT redundant with Power Total (P=0.159) — separate pump system.
    # Aeration pH Existing / New are correlated (P=0.839) but kept separately as
    # the two tanks can operate at different set-points.
    # Aeration DO Existing / New are correlated (P=0.769) but tank-level DO differences
    # are operationally meaningful; flagged in narrative rather than hard-dropped.

    # ── Ordered feature groups for table layout ────────────────────────────────
    GROUPS = [
        ("Power &amp; Flow", [
            "Power GE (KW)", "Power NEA (KW)", "Power / Flow (KW/ML)",
        ]),
        ("Inlet — Composite", [
            "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
            "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
        ]),
        ("Inlet — Specialty", [
            "Inlet TKN/NH3-N (mg/L, Grab)", "Inlet O&G (mg/L, Grab)",
            "Inlet PO4/TP (mg/L, Grab)",
            "Inlet Total Coliform (CFU/100ml, Grab)",
            "Inlet Fecal Coliform (CFU/100ml, Grab)",
        ]),
        ("Primary Treatment", [
            "Grit Classifier TSS (mg/L)", "Primary Clarifier pH",
            "Primary TSS (mg/L)", "Primary BOD (mg/L)",
            "Primary COD (mg/L)", "Primary Sludge Totalizer (m3)",
        ]),
        ("Aeration — DO", [
            "Aeration DO (mg/L, Existing)", "Aeration DO (mg/L, New)",
        ]),
        ("Aeration — Biomass &amp; Settling", [
            "Aeration MLSS (mg/L, Existing)", "Aeration MLVSS (mg/L, Existing)",
            "Aeration SV30 (ml/L, Existing)", "Aeration SVI (Existing)",
            "Aeration MLSS (mg/L, New)", "Aeration MLVSS (mg/L, New)",
            "Aeration SV30 (ml/L, New)", "Aeration SVI (New)",
        ]),
        ("Aeration — pH", [
            "Aeration pH (Existing)", "Aeration pH (New)",
        ]),
    ]

    # ── Classification helpers ─────────────────────────────────────────────────
    def _signal_type(p, sp):
        """Classify relationship type from Pearson + Spearman."""
        if p is None or sp is None:
            return "—"
        ap, asp = abs(p), abs(sp)
        if ap >= 0.6 and asp >= 0.6:
            return "Linear &amp; monotonic"
        if ap >= 0.6 and asp < 0.35:
            return "Outlier-driven (P&gg;S)"
        if ap < 0.35 and asp >= 0.6:
            return "Non-linear monotonic (S&gg;P)"
        if ap >= 0.35 and asp >= 0.35:
            return "Moderate — linear + monotonic"
        if asp >= 0.30:
            return "Weakly monotonic"
        if ap >= 0.30:
            return "Weak linear only"
        return "Weak / non-monotonic"

    def _tier(feat, mi, miss_pct):
        """Four-level recommendation tier."""
        if feat in REDUNDANT:
            return "REDUNDANT"
        if mi is None:
            return "—"
        if mi >= 0.20 and miss_pct <= 35:
            return "ADD"
        if (mi >= 0.15 and miss_pct <= 35) or (mi >= 0.25 and miss_pct <= 55):
            return "CONSIDER"
        if mi >= 0.05:
            return "LOW"
        return "SKIP"

    TIER_LABEL = {
        "ADD":       "Add",
        "CONSIDER":  "Consider",
        "LOW":       "Low priority",
        "SKIP":      "Skip",
        "REDUNDANT": "Redundant — drop",
        "—":         "—",
    }
    TIER_CLR = {
        "ADD":       ("#0d3b1f", "#2ecc71"),
        "CONSIDER":  ("#2e2800", "#f0c040"),
        "LOW":       ("#2e1800", "#d08030"),
        "SKIP":      ("#2e0f0f", "#c04040"),
        "REDUNDANT": ("#1a1a1a", "#666666"),
        "—":         ("#1a1a1a", "#555555"),
    }

    # ── Grab targets ──────────────────────────────────────────────────────────
    GRAB_TGTS = [
        ("Effluent BOD (mg/L, Grab)",  "bod-grab",  "Effluent BOD (Grab)"),
        ("Effluent COD (mg/L, Grab)",  "cod-grab",  "Effluent COD (Grab)"),
        ("Effluent TSS (mg/L, Grab)",  "tss-grab",  "Effluent TSS (Grab)"),
        ("Effluent pH (Grab)",         "ph-grab",   "Effluent pH (Grab)"),
    ]

    # ── Compute stats per target × candidate ──────────────────────────────────
    all_stats = {}
    for tgt, slug, label in GRAB_TGTS:
        tgt_rows = df[df[tgt].notna()].copy()
        n_total  = len(tgt_rows)
        feat_stats = {}
        for feat in candidates:
            pair     = tgt_rows[[feat, tgt]].dropna()
            n        = len(pair)
            miss_pct = round((1 - n / n_total) * 100, 1)
            if n < 20:
                feat_stats[feat] = dict(n=n, miss_pct=miss_pct, p=None, sp=None,
                                        mi=None, tier="—", signal="—")
                continue
            pr, _  = stats.pearsonr(pair[feat], pair[tgt])
            sr, _  = stats.spearmanr(pair[feat], pair[tgt])
            mi_val = mutual_info_regression(pair[[feat]], pair[tgt], random_state=42)[0]
            pr, sr, mi_val = round(pr, 3), round(sr, 3), round(mi_val, 3)
            t   = _tier(feat, mi_val, miss_pct)
            sig = _signal_type(pr, sr)
            feat_stats[feat] = dict(n=n, miss_pct=miss_pct, p=pr, sp=sr,
                                    mi=mi_val, tier=t, signal=sig)
        all_stats[tgt] = feat_stats

    # ── HTML helpers ──────────────────────────────────────────────────────────
    # Light-theme table styles (white/grey/black — sits inside dark report card)
    TH    = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem"
    TH_C  = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem"
    TD_BASE = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"

    TIER_SORT = {"ADD": 0, "CONSIDER": 1, "LOW": 2, "SKIP": 3, "REDUNDANT": 4, "—": 5}

    def _val(v, is_pct=False):
        if v is None:
            return "<span style='color:#999'>—</span>"
        if is_pct:
            clr = "#721c24" if v > 40 else ("#7a3800" if v > 20 else "#1a1a1a")
            return f"<span style='color:{clr}'>{v:.1f}%</span>"
        return f"{v:.3f}"

    def _mi_fmt(v):
        if v is None:
            return "<span style='color:#999'>—</span>"
        clr = ("#155724" if v >= 0.30 else
               "#7d5400" if v >= 0.15 else
               "#7a3800" if v >= 0.05 else "#721c24")
        return f"<span style='color:{clr};font-weight:600'>{v:.3f}</span>"

    def _corr_fmt(v):
        """Plain dark text — no colour coding; let MI carry the signal story."""
        if v is None:
            return "<span style='color:#999'>—</span>"
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.3f}"

    def _tier_badge(t):
        # Bootstrap-style light-background badge colours
        styles = {
            "ADD":       ("background:#d4edda;color:#155724;border:1px solid #c3e6cb"),
            "CONSIDER":  ("background:#fff3cd;color:#7d5400;border:1px solid #ffeeba"),
            "LOW":       ("background:#fde8d0;color:#7a3800;border:1px solid #f5c6a0"),
            "SKIP":      ("background:#f8d7da;color:#721c24;border:1px solid #f5c6cb"),
            "REDUNDANT": ("background:#e2e3e5;color:#3a3a3a;border:1px solid #d3d3d3"),
            "—":         ("background:#f0f0f0;color:#888;border:1px solid #ddd"),
        }
        label = TIER_LABEL.get(t, t)
        st    = styles.get(t, styles["—"])
        return (f"<span style='{st};padding:2px 7px;border-radius:3px;"
                f"font-size:0.75rem;font-weight:600;white-space:nowrap'>{label}</span>")

    def _group_rows(group_feats, feat_stats):
        # Sort within group by tier order, then by MI desc within same tier
        ordered = [f for f in group_feats if f in feat_stats]
        ordered.sort(key=lambda f: (
            TIER_SORT.get(feat_stats[f]["tier"], 99),
            -(feat_stats[f]["mi"] or 0)
        ))
        rows = ""
        for i, feat in enumerate(ordered):
            v  = feat_stats[feat]
            t  = v["tier"]
            bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
            if t == "REDUNDANT":
                alt, reason = REDUNDANT[feat]
                reason_cell = (f"<td colspan='5' style='{TD_BASE};color:#888;"
                               f"font-style:italic;background:{bg}'>"
                               f"Redundant with <em>{alt}</em>. {reason}</td>")
            else:
                reason_cell = None

            row  = f"<tr style='background:{bg}'>"
            row += f"<td style='{TD_BASE}'>{feat}</td>"
            row += f"<td style='{TD_BASE};text-align:center'>{_tier_badge(t)}</td>"
            if reason_cell:
                row += reason_cell
            else:
                row += f"<td style='{TD_BASE};text-align:center'>{_mi_fmt(v['mi'])}</td>"
                row += f"<td style='{TD_BASE};text-align:center'>{_corr_fmt(v['p'])}</td>"
                row += f"<td style='{TD_BASE};text-align:center'>{_corr_fmt(v['sp'])}</td>"
                row += f"<td style='{TD_BASE};text-align:center'>{_val(v['miss_pct'], is_pct=True)}</td>"
                row += f"<td style='{TD_BASE};color:#555;font-size:0.78rem'>{v['signal']}</td>"
            row += "</tr>"
            rows += row
        return rows

    def _narrative(tgt, feat_stats):
        """Generate evidence-backed bullet points for the narrative callout box."""
        bullets = []

        # Collect features by tier, sorted by MI desc
        add_feats = sorted(
            [(f, v) for f, v in feat_stats.items() if v["tier"] == "ADD" and f not in REDUNDANT],
            key=lambda x: (x[1]["mi"] or 0), reverse=True
        )
        consider_feats = sorted(
            [(f, v) for f, v in feat_stats.items() if v["tier"] == "CONSIDER" and f not in REDUNDANT],
            key=lambda x: (x[1]["mi"] or 0), reverse=True
        )
        low_feats = sorted(
            [(f, v) for f, v in feat_stats.items() if v["tier"] == "LOW"],
            key=lambda x: (x[1]["mi"] or 0), reverse=True
        )
        skip_feats = [f for f, v in feat_stats.items() if v["tier"] == "SKIP"]
        redundant_feats = [f for f in REDUNDANT if f in feat_stats]

        # ADD bullets — grouped by signal type where patterns emerge
        if add_feats:
            # Check for DO threshold pattern
            do_feats = [(f, v) for f, v in add_feats
                        if "DO" in f and v["p"] is not None and abs(v["p"]) < 0.30]
            non_do_add = [(f, v) for f, v in add_feats if f not in dict(do_feats)]

            for feat, v in non_do_add:
                sig_note = ""
                if v["signal"] == "Linear &amp; monotonic":
                    sig_note = "strong linear and monotonic signal — reliable for both linear and non-linear models"
                elif "Non-linear" in v["signal"] or "S&gg;P" in v["signal"]:
                    sig_note = "monotonic but curved (Spearman > Pearson) — benefits non-linear models more than linear"
                elif "Moderate" in v["signal"]:
                    sig_note = "moderate linear + monotonic signal"
                else:
                    sig_note = v["signal"].lower()
                bullets.append(
                    f"<li><strong>{feat}</strong>: MI = {v['mi']:.3f}, "
                    f"Pearson = {v['p']:+.3f}, Spearman = {v['sp']:+.3f}, "
                    f"missingness = {v['miss_pct']:.1f}%. {sig_note.capitalize()}. "
                    f"Worth the imputation effort.</li>"
                )

            if do_feats:
                do_feat_names = ", ".join(f"<em>{f}</em>" for f, _ in do_feats)
                do_mis = [f"{v['mi']:.3f}" for _, v in do_feats]
                do_miss = [f"{v['miss_pct']:.1f}%" for _, v in do_feats]
                bullets.append(
                    f"<li><strong>Aeration DO (Existing &amp; New)</strong>: "
                    f"MI = {'/'.join(do_mis)} but Pearson and Spearman are both weak "
                    f"(&lt; 0.20). This is the hallmark of a <em>threshold / non-monotonic</em> "
                    f"relationship — Section 9 confirms that BOD and TSS spike disproportionately "
                    f"when DO falls below ~0.5&nbsp;mg/L. A linear model will miss this entirely; "
                    f"a tree model will find the split automatically. "
                    f"Missingness: {'/'.join(do_miss)}. Strongly recommended for Experiment 3 "
                    f"with non-linear models.</li>"
                )

        # CONSIDER bullets — highlight only those with non-trivial MI
        for feat, v in consider_feats[:5]:  # cap at 5 to keep narrative concise
            miss_note = (f"high missingness ({v['miss_pct']:.1f}%) means imputation is required "
                         f"and should be validated" if v["miss_pct"] > 40
                         else f"missingness ({v['miss_pct']:.1f}%) is manageable")
            bullets.append(
                f"<li><strong>{feat}</strong>: MI = {v['mi']:.3f}, "
                f"Pearson = {v['p']:+.3f}, Spearman = {v['sp']:+.3f}. "
                f"{miss_note.capitalize()}. Include if imputation quality is acceptable.</li>"
            )

        # LOW-priority features — single grouped bullet
        if low_feats:
            lf_names = ", ".join(f"<em>{f}</em>" for f, _ in low_feats[:8])
            bullets.append(
                f"<li><strong>Low priority</strong>: {lf_names}"
                + (f" and {len(low_feats)-8} more" if len(low_feats) > 8 else "")
                + ". MI scores between 0.05 and 0.15. Marginal signal — include only "
                  "if they are already imputed cheaply alongside stronger features.</li>"
            )

        # SKIP grouped bullet
        if skip_feats:
            sk_names = ", ".join(f"<em>{f}</em>" for f in skip_feats)
            bullets.append(
                f"<li><strong>Skip</strong>: {sk_names}. "
                f"MI ≈ 0 — no detectable statistical relationship with this target. "
                f"Adding them only increases noise and imputation burden.</li>"
            )

        # Redundancy bullets
        for feat in redundant_feats:
            alt, reason = REDUNDANT[feat]
            bullets.append(
                f"<li><strong>Drop <em>{feat}</em></strong>: {reason} "
                f"Use <em>{alt}</em> instead.</li>"
            )

        return "<ul style='margin:0;padding-left:1.2em'>" + "".join(bullets) + "</ul>"

    # ── Build per-target blocks ────────────────────────────────────────────────
    target_blocks = ""
    for tgt, slug, label in GRAB_TGTS:
        feat_stats = all_stats[tgt]
        tgt_rows   = df[df[tgt].notna()]
        n_total    = len(tgt_rows)

        table_head = f"""
        <table style='border-collapse:collapse;width:100%;background:#ffffff;
                      font-size:0.82rem;table-layout:auto;color:#1a1a1a'>
          <thead>
            <tr style='background:#eeeeee;border-bottom:2px solid #cccccc'>
              <th style='{TH}'>Feature</th>
              <th style='{TH_C}'>Recommendation</th>
              <th style='{TH_C}'>MI</th>
              <th style='{TH_C}'>Pearson r</th>
              <th style='{TH_C}'>Spearman r</th>
              <th style='{TH_C}'>Miss%</th>
              <th style='{TH}'>Signal type</th>
            </tr>
          </thead>
          <tbody>"""

        table_body = ""
        for grp_label, grp_feats in GROUPS:
            visible = [f for f in grp_feats if f in feat_stats]
            if not visible:
                continue
            table_body += (
                f"<tr style='background:#e8e8e8'>"
                f"<td colspan='7' style='padding:5px 10px;font-size:0.75rem;"
                f"font-weight:700;color:#555555;letter-spacing:0.06em;"
                f"text-transform:uppercase;border-bottom:1px solid #d0d0d0'>"
                f"{grp_label}</td></tr>"
            )
            table_body += _group_rows(visible, feat_stats)

        table_foot = """
          </tbody>
        </table>"""

        # MI legend
        legend = """
        <p style='font-size:0.76rem;color:#666;margin:6px 0 0'>
          MI: <span style='color:#155724;font-weight:600'>green ≥ 0.30</span>
          &nbsp;·&nbsp; <span style='color:#7d5400;font-weight:600'>amber 0.15–0.29</span>
          &nbsp;·&nbsp; <span style='color:#7a3800;font-weight:600'>orange 0.05–0.14</span>
          &nbsp;·&nbsp; <span style='color:#721c24;font-weight:600'>red &lt; 0.05</span>
          &nbsp;·&nbsp; Pearson and Spearman r shown as plain values (no colour coding).
          Miss% in <span style='color:#721c24;font-weight:600'>red</span> &gt; 40%,
          <span style='color:#7a3800;font-weight:600'>orange</span> 21–40%.
        </p>"""

        narrative_html = _narrative(tgt, feat_stats)

        target_blocks += f"""
<details id="s11-{slug}" style='margin:12px 0;border:1px solid #2e3d52;
           border-radius:6px;overflow:hidden'>
  <summary style='padding:10px 14px;cursor:pointer;background:#1a2d42;
                  font-weight:700;font-size:0.95rem;color:#b8d0f0;
                  list-style:none;display:flex;align-items:center;gap:8px'>
    <span class='fold-icon' style='display:inline-block;transition:transform 0.2s'>&#9654;</span>
    {label}
    <span style='font-size:0.78rem;font-weight:400;color:#5a7a9a;margin-left:4px'>
      — {n_total} rows with target present (training + test combined)
    </span>
  </summary>
  <div style='padding:14px'>
    <div style='overflow-x:auto;border-radius:4px;border:1px solid #2e3d52'>
      {table_head}{table_body}{table_foot}
    </div>
    {legend}
    <div class='obs-card' style='margin-top:14px'>
      <h4>Evidence-backed suggestions</h4>
      {narrative_html}
    </div>
  </div>
</details>"""

    # ── Assemble section ───────────────────────────────────────────────────────
    intro = """
<p>For each grab-sample effluent target, the table below lists every candidate feature
<em>not already included in Experiment&nbsp;2 Sub&#8209;2</em> (which uses inlet grab +
secondary clarifier/sedimentation columns + COMMON). Each feature is evaluated against
four questions:</p>
<ol style='margin:6px 0 10px;padding-left:1.4em;font-size:0.92rem'>
  <li><strong>Is there any signal?</strong> — Mutual Information (MI) score, computed on rows
      where both the feature and target are present.</li>
  <li><strong>What kind of signal?</strong> — Pearson (linear) vs Spearman (monotonic) pattern,
      classified into relationship types that indicate which model class will benefit.</li>
  <li><strong>What does it cost?</strong> — Missingness % after filtering to rows where the
      target is present. High missingness means imputation is required before this feature
      can be used.</li>
  <li><strong>Is it redundant?</strong> — Known collinear pairs identified from the
      Section&nbsp;5 Features&#8209;vs&#8209;Features table.</li>
</ol>
<p style='font-size:0.87rem;color:#7a9ab8'>
  Recommendation tiers:
  <strong style='color:#2ecc71'>Add</strong> (MI &ge; 0.20, miss &le; 35%) &nbsp;&middot;&nbsp;
  <strong style='color:#f0c040'>Consider</strong> (MI &ge; 0.15 or MI &ge; 0.25 with miss &le; 55%) &nbsp;&middot;&nbsp;
  <strong style='color:#d08030'>Low priority</strong> (MI 0.05&ndash;0.15) &nbsp;&middot;&nbsp;
  <strong style='color:#c04040'>Skip</strong> (MI &lt; 0.05) &nbsp;&middot;&nbsp;
  <strong style='color:#666'>Redundant&nbsp;&mdash;&nbsp;drop</strong>
</p>"""

    return f"""<div class="card" id="s11">
<h2>11. Feature Suggestions &mdash; Experiment&nbsp;3 (Grab Effluents)</h2>
{intro}
{target_blocks}
</div>
"""


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
    body  { font-family: Calibri, Arial, sans-serif; margin: 0; padding: 0; }
    #main-content { margin-left: 265px; max-width: 1500px; padding: 24px 28px; }
    h1    { color: #1F4E79; border-bottom: 3px solid #1F4E79; padding-bottom: 8px;
            font-size: 1.8em; }
    h2    { color: #1F4E79; margin-top: 48px; font-size: 1.25em;
            border-left: 6px solid #2E75B6; padding-left: 12px; }
    p, ul, ol { max-width: 950px; line-height: 1.7; }
    li    { margin-bottom: 4px; }
    .card { padding: 24px 28px; border-radius: 10px; margin-bottom: 28px; }
    .badge { display: inline-block; background: #2E75B6; color: white;
             font-size: 0.75em; padding: 2px 8px; border-radius: 10px;
             vertical-align: middle; margin-right: 6px; }
    .badge-decision { background: #C0392B; }
    img   { border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.12); }
    /* ── Main-content fold styles (scoped away from sidenav) ── */
    #main-content details > summary {
        cursor: pointer; font-weight: bold; font-size: 1em;
        color: #2E75B6; padding: 10px 16px; list-style: none;
        user-select: none; border-radius: 6px; }
    #main-content details > summary .fold-icon {
        display: inline-block; transition: transform 0.2s ease; }
    #main-content details[open] > summary .fold-icon { transform: rotate(90deg); }
    #main-content details[open] > summary { border-radius: 6px 6px 0 0; }
    .fold-hint { font-weight: normal; font-size: 0.82em;
                 opacity: 0.7; margin-left: 8px; }
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
    .obs-card table { border-collapse: collapse; width: 100%; font-size: 13px;
                      margin: 10px 0; }
    .obs-card table th { background: #4A6FA5; color: white; font-weight: 600;
                         padding: 7px 14px; text-align: center;
                         border: 1px solid #3a5a8a; white-space: nowrap; }
    .obs-card table th:first-child { text-align: left; }
    .obs-card table td { border: 1px solid #d0d9e8; padding: 6px 14px;
                         vertical-align: middle; text-align: center; }
    .obs-card table td:first-child { text-align: left; font-weight: 600; }
    .obs-card table tbody tr:nth-child(even) { background: #F4F7FB; }
    .obs-card table tbody tr:nth-child(odd)  { background: #FFFFFF; }
    hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }
    /* ── Left sidebar navigation ── */
    #sidenav {
        position: fixed; top: 0; left: 0; width: 255px; height: 100vh;
        overflow-y: auto; overflow-x: hidden;
        background: var(--bg); border-right: 2px solid var(--border);
        z-index: 500; font-size: 0.79em; line-height: 1.4;
        box-shadow: 2px 0 8px rgba(0,0,0,0.08); }
    #sidenav .nav-header {
        font-weight: 700; font-size: 1.0em; color: #1F4E79;
        padding: 14px 14px 10px; border-bottom: 2px solid #2E75B6;
        letter-spacing: 0.04em; }
    #sidenav ul { list-style: none; margin: 0; padding: 0; }
    #sidenav li { margin: 0; }
    #sidenav a { display: block; color: #3a7bbf; text-decoration: none;
                 padding: 3px 14px 3px 14px;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #sidenav a:hover { background: var(--hover-bg, rgba(46,117,182,0.10));
                       color: #1F4E79; }
    /* top-level section entries */
    #sidenav .nav-s > a {
        font-weight: 600; font-size: 1.0em; color: var(--text);
        padding: 6px 14px 2px; border-top: 1px solid var(--border); }
    /* group level (Grab / Composite) */
    #sidenav .nav-g > a {
        font-size: 0.95em; font-style: italic; color: #555;
        padding: 3px 14px 3px 22px; }
    /* target level (BOD / COD / TSS / pH) */
    #sidenav .nav-t > a { padding: 2px 14px 2px 34px; font-size: 0.92em; }
    /* collapsible toggle button in nav */
    #sidenav .nav-toggle {
        cursor: pointer; user-select: none;
        display: flex; align-items: center; justify-content: space-between; }
    #sidenav .nav-toggle::after { content: "▸"; font-size: 0.75em;
        margin-right: 10px; transition: transform 0.15s ease;
        flex-shrink: 0; }
    #sidenav .nav-toggle.open::after { transform: rotate(90deg); }
    #sidenav .nav-children { display: none; }
    #sidenav .nav-children.open { display: block; }
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

    # Stable anchor slugs for each effluent target (used in sidebar nav + section folds)
    _TGT_SLUG = {
        "Effluent BOD (mg/L, Grab)":       "bod-grab",
        "Effluent COD (mg/L, Grab)":       "cod-grab",
        "Effluent TSS (mg/L, Grab)":       "tss-grab",
        "Effluent pH (Grab)":              "ph-grab",
        "Effluent BOD (mg/L, Composite)":  "bod-comp",
        "Effluent COD (mg/L, Composite)":  "cod-comp",
        "Effluent TSS (mg/L, Composite)":  "tss-comp",
        "Effluent pH (Composite)":         "ph-comp",
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

    # ── sidebar nav HTML ──────────────────────────────────────────────────────
    def _nav_targets(section, targets):
        """Return <li> entries for individual target links."""
        items = ""
        for tgt in targets:
            slug  = _TGT_SLUG.get(tgt, "")
            label = tgt.replace("Effluent ", "").split(" (")[0]   # "BOD", "COD", etc.
            items += f'<li class="nav-t"><a href="#{section}-{slug}">{label}</a></li>\n'
        return items

    def _nav_group(section, group_id, group_label, targets):
        """Return a collapsible group entry (Grab / Composite)."""
        tgt_items = _nav_targets(section, targets)
        return (
            f'<li class="nav-g">'
            f'<a class="nav-toggle" href="#{group_id}">{group_label}</a>'
            f'<ul class="nav-children">{tgt_items}</ul>'
            f'</li>\n'
        )

    SIDENAV = f"""<nav id="sidenav">
  <div class="nav-header">Navigation</div>
  <ul>
    <li class="nav-s"><a class="nav-toggle" href="#s1">1. Ridge Residuals</a>
      <ul class="nav-children">
        {_nav_group("s1", "s1-grab", "Grab Effluent", GRAB_TARGETS)}
        {_nav_group("s1", "s1-comp", "Composite Effluent", COMP_TARGETS)}
      </ul>
    </li>
    <li class="nav-s"><a class="nav-toggle" href="#s4">2. Scatter Grids</a>
      <ul class="nav-children">
        {_nav_targets("s4", GRAB_TARGETS)}
      </ul>
    </li>
    <li class="nav-s"><a class="nav-toggle" href="#s2">3. Pearson vs Spearman</a>
      <ul class="nav-children">
        {_nav_group("s2", "s2-grab", "Grab Effluent", GRAB_TARGETS)}
        {_nav_group("s2", "s2-comp", "Composite Effluent", COMP_TARGETS)}
      </ul>
    </li>
    <li class="nav-s"><a class="nav-toggle" href="#s3">4. Mutual Information</a>
      <ul class="nav-children">
        {_nav_group("s3", "s3-grab", "Grab Effluent", GRAB_TARGETS)}
        {_nav_group("s3", "s3-comp", "Composite Effluent", COMP_TARGETS)}
      </ul>
    </li>
    <li class="nav-s"><a href="#s5">5. Cross-stage Heatmap</a></li>
    <li class="nav-s"><a href="#s6">6. Data Coverage</a></li>
    <li class="nav-s"><a href="#s7">7. Aeration Time Series</a></li>
    <li class="nav-s"><a href="#s8">8. Stage Removal</a></li>
    <li class="nav-s"><a href="#s9">9. DO vs Effluent</a></li>
    <li class="nav-s"><a href="#s10">10. Settleability vs TSS</a></li>
    <li class="nav-s"><a href="#s11">11. Feature Suggestions — Exp 3</a></li>
  </ul>
</nav>"""

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
{SIDENAV}
<div id="main-content">
<h1>{title}</h1>
<p>Full exploratory data analysis on the enriched 60-column dataset
(<code>All_Years_Full.xlsx</code>, 1918 daily rows, 2020–2025).
Sections are ordered by relevance to the key question:
<strong>are non-linear models (Random Forest) justified over linear ones?</strong>
Use the navigation panel on the left to jump to any section or individual target.</p>
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
    resid_data = data.get("residuals",   {})
    coef_data  = data.get("ridge_coefs", {})
    s1_obs     = obs.get("s1", {})

    def _resid_fold(targets, key, label, open_by_default=False,
                    section_id="s1", group_id=""):
        content = ""
        for tgt in targets:
            slug       = _TGT_SLUG.get(tgt, "")
            resid_path = resid_data.get(key, {}).get(tgt)
            coef_path  = coef_data.get(key,  {}).get(tgt)
            tgt_obs    = s1_obs.get(tgt, "")

            # per-target anchor so sidebar nav can scroll directly here
            if slug:
                content += f'<a id="{section_id}-{slug}" style="display:block;position:relative;top:-80px;visibility:hidden"></a>\n'

            # 1 — residual diagnostic plots
            content += img_tag(resid_path)

            # 2 — coefficient bar chart (foldable, collapsed by default)
            if coef_path:
                coef_note = (
                    "<p style='font-size:0.82em;color:#666;margin:4px 0 0 0;font-style:italic'>"
                    "Coefficients are on the StandardScaler scale (unit-free). "
                    "Correlated features split weight between them — individual bars "
                    "understate the importance of collinear groups. "
                    "Use Mutual Information (Section 4) for model-agnostic feature ranking."
                    "</p>"
                )
                content += _fold(
                    "▶ Coefficient weights — what Ridge relied on (top 10)",
                    img_tag(coef_path) + coef_note
                )

            # 3 — per-target observations
            if tgt_obs:
                content += _obs(tgt_obs)

        fold = _fold(f"▶ {label} (click to expand)", content,
                     open_by_default=open_by_default)
        # prepend group anchor so sidebar can also link to the whole group fold
        if group_id:
            return f'<a id="{group_id}" style="display:block;position:relative;top:-80px;visibility:hidden"></a>\n' + fold
        return fold

    parts.append(_resid_fold(GRAB_TARGETS, "grab", "Grab effluent targets",
                             open_by_default=True, section_id="s1", group_id="s1-grab"))
    parts.append(_resid_fold(COMP_TARGETS, "comp", "Composite effluent targets",
                             section_id="s1", group_id="s1-comp"))
    parts.append("</div>\n")

    # helper: build a fold whose content interleaves one image per target with its obs
    def _per_target_fold(targets, img_paths, obs_dict, label, open_by_default=False,
                         section_id="", group_id=""):
        content = ""
        for tgt, path in zip(targets, img_paths):
            slug = _TGT_SLUG.get(tgt, "")
            if slug and section_id:
                content += f'<a id="{section_id}-{slug}" style="display:block;position:relative;top:-80px;visibility:hidden"></a>\n'
            content += img_tag(path)
            tgt_obs = (obs_dict or {}).get(tgt)
            if tgt_obs:
                content += _obs(tgt_obs)
        fold = _fold(f"▶ {label} (click to expand)", content,
                     open_by_default=open_by_default)
        if group_id:
            return f'<a id="{group_id}" style="display:block;position:relative;top:-80px;visibility:hidden"></a>\n' + fold
        return fold

    # ── Section 2 (display): Scatter grids ────────────────────────────────────
    scatter_paths  = data.get("scatter", [])
    scatter_labels = list(SCATTER_LABELS.values())
    s4_obs         = obs.get("s4", "")   # general how-to-read guidance

    scatter_folds = ""
    for tgt, path, lbl in zip(GRAB_TARGETS, scatter_paths, scatter_labels):
        slug   = _TGT_SLUG.get(tgt, "")
        anchor = f'<a id="s4-{slug}" style="display:block;position:relative;top:-80px;visibility:hidden"></a>\n' if slug else ""
        scatter_folds += anchor + _fold(f"▶ {lbl}", img_tag(path))

    parts.append(f"""<div class="card" id="s4">
<h2><span class="badge badge-decision">Model selection</span>
2. Feature vs target scatter grids</h2>
<p>Top 15 features (ranked by Mutual Information) plotted against each effluent target.
Points are coloured by year. The dashed black line is the linear regression fit.
The subplot highlighted in <strong style="color:#E07B20">orange</strong> is the feature
with the highest Ridge coefficient magnitude — the variable Section 1 residuals rely on most.
Look for curvature, fan shapes, or year-cluster separation in that panel first.</p>
{_obs(s4_obs)}
{scatter_folds}
</div>
""")

    # ── Section 3 (display): Pearson vs Spearman ──────────────────────────────
    ps            = data.get("pearson", {})
    grab_paths_ps = ps.get("grab", [])
    comp_paths_ps = ps.get("comp", [])
    s2_obs        = obs.get("s2", {})

    grab_ps_fold = _per_target_fold(
        GRAB_TARGETS, grab_paths_ps, s2_obs.get("grab", {}),
        "Grab effluent targets", open_by_default=True,
        section_id="s2", group_id="s2-grab")
    comp_ps_fold = _per_target_fold(
        COMP_TARGETS, comp_paths_ps, s2_obs.get("comp", {}),
        "Composite effluent targets",
        section_id="s2", group_id="s2-comp")

    parts.append(f"""<div class="card" id="s2">
<h2><span class="badge badge-decision">Model selection</span>
3. Pearson vs Spearman correlation</h2>
{PEARSON_INTERP}
{grab_ps_fold}
{comp_ps_fold}
{_obs(s2_obs.get("general", ""))}
</div>
""")

    # ── Section 4 (display): Mutual Information ────────────────────────────────
    mi            = data.get("mi", {})
    grab_paths_mi = [p for p in mi.get("grab", []) if p]
    comp_paths_mi = [p for p in mi.get("comp", []) if p]
    s3_obs        = obs.get("s3", {})

    grab_mi_fold = _per_target_fold(
        GRAB_TARGETS, grab_paths_mi, s3_obs.get("grab", {}),
        "Grab effluent targets", open_by_default=True,
        section_id="s3", group_id="s3-grab")
    comp_mi_fold = _per_target_fold(
        COMP_TARGETS, comp_paths_mi, s3_obs.get("comp", {}),
        "Composite effluent targets",
        section_id="s3", group_id="s3-comp")

    parts.append(f"""<div class="card" id="s3">
<h2><span class="badge badge-decision">Model selection</span>
4. Mutual Information scores</h2>
{MI_INTERP}
{grab_mi_fold}
{comp_mi_fold}
{_obs(s3_obs.get("general", ""))}
</div>
""")

    # ── Section 5: Cross-stage heatmap ─────────────────────────────────────────
    cross_data = data.get("cross_heatmap", (None, ""))
    cross_img  = cross_data[0] if isinstance(cross_data, tuple) else cross_data
    cross_tbl  = cross_data[1] if isinstance(cross_data, tuple) else ""
    parts.append(f"""<div class="card" id="s5">
<h2>5. Full correlation matrix — features &amp; targets</h2>
<p>All process features <em>and</em> all eight effluent targets correlated against
each other. Features below the threshold (|Pearson| and |Spearman| both &lt; 0.5 for
every pair) are hidden. This reveals both <strong>feature–target signal</strong>
(which inputs associate with which outputs) and <strong>inter-feature collinearity</strong>
(which inputs carry redundant information).
Top = Pearson, Bottom = Spearman.
Diagonal self-correlations are excluded.</p>
{img_tag(cross_img)}
{cross_tbl}
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

    parts.append(data.get("suggestions", ""))

    parts.append("""
</div><!-- /#main-content -->
<script>
(function() {
  // ── fold-hint text ───────────────────────────────────────────────────────
  function updateHint(details) {
    var hint = details.querySelector(':scope > summary .fold-hint');
    if (!hint) return;
    hint.textContent = details.open ? '(click to minimize)' : '(click to expand)';
  }
  document.querySelectorAll('#main-content details').forEach(function(d) {
    updateHint(d);
    d.addEventListener('toggle', function() { updateHint(d); });
  });

  // ── sidebar: open-ancestor-details + smooth scroll on hash nav ──────────
  function openAncestors(el) {
    var p = el.parentElement;
    while (p) {
      if (p.tagName === 'DETAILS') { p.open = true; }
      p = p.parentElement;
    }
  }
  function goToHash(hash) {
    if (!hash) return;
    var id  = hash.replace('#', '');
    var el  = document.getElementById(id);
    if (!el) return;
    openAncestors(el);
    setTimeout(function() {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 60);
  }
  if (window.location.hash) { goToHash(window.location.hash); }
  window.addEventListener('hashchange', function() { goToHash(window.location.hash); });

  // ── sidebar: collapsible toggle (▸ icon rotates) ─────────────────────────
  document.querySelectorAll('#sidenav .nav-toggle').forEach(function(a) {
    a.addEventListener('click', function(e) {
      // Allow navigation via href, but also toggle children
      var li       = a.closest('li');
      var children = li ? li.querySelector('.nav-children') : null;
      if (children) {
        var isOpen = children.classList.toggle('open');
        a.classList.toggle('open', isOpen);
      }
    });
  });

  // ── sidebar: highlight currently visible section ─────────────────────────
  var navLinks = Array.from(document.querySelectorAll('#sidenav a[href^="#"]'));
  function setActive() {
    var scrollY = window.scrollY + 120;
    var best = null;
    navLinks.forEach(function(link) {
      var id  = link.getAttribute('href').replace('#', '');
      var el  = document.getElementById(id);
      if (el && el.getBoundingClientRect().top + window.scrollY <= scrollY) {
        best = link;
      }
    });
    navLinks.forEach(function(l) { l.style.fontWeight = ''; l.style.color = ''; });
    if (best) { best.style.fontWeight = '700'; best.style.color = '#1F4E79'; }
  }
  window.addEventListener('scroll', setActive, { passive: true });
  setActive();
})();
</script>
</body>
</html>""")
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
    mi_result       = plot_mutual_information(df)
    all_mi          = mi_result["mi_data"]
    top_ridge_feats = compute_top_ridge_features(df)
    scatter_paths   = plot_scatter_grids(df, all_mi, top_ridge_feats)
    aeration_ts   = plot_aeration_timeseries(df)
    cross_heatmap_path, cross_heatmap_tbl = plot_cross_stage_heatmap(df)
    stage_removal = plot_stage_removal(df)
    do_effluent   = plot_do_vs_effluent(df)
    svi_tss       = plot_svi_mlss_vs_tss(df)
    residuals     = plot_linearity_residuals(df)
    ridge_coefs   = plot_ridge_coefficients(df)

    n_charts = (1 + len(pearson_data["grab"]) + len(pearson_data["comp"])
                + len([p for p in mi_result["grab"] if p])
                + len([p for p in mi_result["comp"] if p])
                + len(scatter_paths) + 5 + 1)

    print("\nComputing observations...")
    section_obs  = build_observations(df)
    suggestions_html = build_suggestions(df)

    print("Building HTML report...")
    build_html({
        "residuals":    residuals,
        "ridge_coefs":  ridge_coefs,
        "pearson":      pearson_data,
        "mi":           mi_result,
        "scatter":      scatter_paths,
        "aeration_ts":  aeration_ts,
        "cross_heatmap":(cross_heatmap_path, cross_heatmap_tbl),
        "stage_removal":stage_removal,
        "do_effluent":  do_effluent,
        "svi_tss":      svi_tss,
        "missing_all":  missing_all,
        "obs":          section_obs,
        "suggestions":  suggestions_html,
    })
    print(f"\nDone. ~{n_charts} chart files → {PLOTS_DIR}/")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
