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
import warnings

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

# Short labels for axes
SHORT = {
    "Power GE (KW)":                         "Power GE",
    "Power NEA (KW)":                        "Power NEA",
    "Power Total (KW)":                      "Power Total",
    "Power / Flow (KW/ML)":                  "Power/Flow",
    "Flow (MLD)":                            "Flow",
    "Inlet pH (Grab)":                       "In pH (G)",
    "Inlet BOD (mg/L, Grab)":               "In BOD (G)",
    "Inlet COD (mg/L, Grab)":               "In COD (G)",
    "Inlet TSS (mg/L, Grab)":               "In TSS (G)",
    "Inlet TKN/NH3-N (mg/L, Grab)":         "In TKN (G)",
    "Inlet O&G (mg/L, Grab)":               "In O&G (G)",
    "Inlet PO4/TP (mg/L, Grab)":            "In PO4 (G)",
    "Inlet Total Coliform (CFU/100ml, Grab)":"In T.Coli (G)",
    "Inlet Fecal Coliform (CFU/100ml, Grab)":"In F.Coli (G)",
    "Inlet pH (Composite)":                  "In pH (C)",
    "Inlet BOD (mg/L, Composite)":          "In BOD (C)",
    "Inlet COD (mg/L, Composite)":          "In COD (C)",
    "Inlet TSS (mg/L, Composite)":          "In TSS (C)",
    "Grit Classifier TSS (mg/L)":           "Grit TSS",
    "Primary Clarifier pH":                  "Prim pH",
    "Primary TSS (mg/L)":                    "Prim TSS",
    "Primary BOD (mg/L)":                    "Prim BOD",
    "Primary COD (mg/L)":                    "Prim COD",
    "Primary Sludge Totalizer (m3)":         "Prim Sludge",
    "Sec Clarifier pH":                      "SecC pH",
    "Sec Clarifier TSS (mg/L)":             "SecC TSS",
    "Sec Clarifier BOD (mg/L)":             "SecC BOD",
    "Sec Clarifier COD (mg/L)":             "SecC COD",
    "Sec Clarifier RAS":                     "SecC RAS",
    "Sec Sed pH":                            "SecS pH",
    "Sec Sed TSS (mg/L)":                   "SecS TSS",
    "Sec Sed BOD (mg/L)":                   "SecS BOD",
    "Sec Sed COD (mg/L)":                   "SecS COD",
    "Sec Sed RAS (New)":                    "SecS RAS",
    "Aeration pH (Existing)":               "Aer pH (E)",
    "Aeration DO (mg/L, Existing)":         "Aer DO (E)",
    "Aeration MLSS (mg/L, Existing)":       "Aer MLSS (E)",
    "Aeration MLVSS (mg/L, Existing)":      "Aer MLVSS (E)",
    "Aeration SV30 (ml/L, Existing)":       "Aer SV30 (E)",
    "Aeration SVI (Existing)":              "Aer SVI (E)",
    "Aeration pH (New)":                    "Aer pH (N)",
    "Aeration DO (mg/L, New)":              "Aer DO (N)",
    "Aeration MLSS (mg/L, New)":            "Aer MLSS (N)",
    "Aeration MLVSS (mg/L, New)":           "Aer MLVSS (N)",
    "Aeration SV30 (ml/L, New)":            "Aer SV30 (N)",
    "Aeration SVI (New)":                   "Aer SVI (N)",
    "Effluent pH (Grab)":                   "Eff pH (G)",
    "Effluent BOD (mg/L, Grab)":            "Eff BOD (G)",
    "Effluent COD (mg/L, Grab)":            "Eff COD (G)",
    "Effluent TSS (mg/L, Grab)":            "Eff TSS (G)",
    "Effluent FRC (mg/L)":                  "Eff FRC",
    "Effluent O&G (mg/L)":                  "Eff O&G",
    "Effluent NH3-N (mg/L)":               "Eff NH3",
    "Effluent Total Coliform (CFU/100ml)":  "Eff T.Coli",
    "Effluent Fecal Coliform (CFU/100ml)":  "Eff F.Coli",
    "Effluent pH (Composite)":              "Eff pH (C)",
    "Effluent BOD (mg/L, Composite)":       "Eff BOD (C)",
    "Effluent COD (mg/L, Composite)":       "Eff COD (C)",
    "Effluent TSS (mg/L, Composite)":       "Eff TSS (C)",
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


# ── Chart 1: Missing data for new columns ─────────────────────────────────────

def plot_missing_new(df):
    print("1. Missing data — new columns...")
    # Monthly coverage % per new column
    tmp = df[["Date"] + [c for c in NEW_COLUMNS if c in df.columns]].copy()
    tmp["YearMonth"] = tmp["Date"].dt.to_period("M")
    monthly = tmp.groupby("YearMonth")[[c for c in NEW_COLUMNS if c in df.columns]].apply(
        lambda g: g.notna().mean() * 100
    )
    monthly.index = monthly.index.astype(str)
    short_cols = [s(c) for c in monthly.columns]

    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(
        monthly.T,
        ax=ax, cmap="YlGn", vmin=0, vmax=100,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "% coverage", "shrink": 0.6},
        yticklabels=short_cols,
    )
    ax.set_title("Coverage (%) of new columns by month", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    # Thin x-axis tick labels — show every 6 months
    xticks = ax.get_xticks()
    xlabels = [l.get_text() for l in ax.get_xticklabels()]
    ax.set_xticks(xticks[::6])
    ax.set_xticklabels(xlabels[::6], rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return save(fig, "01_missing_new_columns")


# ── Chart 2: Pearson vs Spearman ──────────────────────────────────────────────

def plot_pearson_vs_spearman(df):
    print("2. Pearson vs Spearman...")
    feats = feature_cols(df)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Pearson vs Spearman correlation — features vs effluent targets\n"
                 "(gap between bars = non-linearity / non-monotonicity)",
                 fontsize=13, fontweight="bold")

    for ax, target in zip(axes.flat, GRAB_TARGETS):
        corr = pearson_spearman(df, feats, target).dropna()
        # Sort by |Spearman|
        corr["abs_sp"] = corr["spearman"].abs()
        corr = corr.sort_values("abs_sp", ascending=True)
        short_feats = [s(f) for f in corr.index]

        y = np.arange(len(corr))
        bar_h = 0.35
        ax.barh(y + bar_h/2, corr["pearson"], bar_h, color="#2171B5",
                alpha=0.8, label="Pearson")
        ax.barh(y - bar_h/2, corr["spearman"], bar_h, color="#FD8D3C",
                alpha=0.8, label="Spearman")
        ax.set_yticks(y)
        ax.set_yticklabels(short_feats, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(s(target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Correlation coefficient")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return save(fig, "02_pearson_vs_spearman")


# ── Chart 3: Mutual Information ────────────────────────────────────────────────

def plot_mutual_information(df):
    print("3. Mutual information...")
    feats = feature_cols(df)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Mutual Information scores — features vs effluent targets\n"
                 "(captures non-linear and non-monotonic relationships)",
                 fontsize=13, fontweight="bold")

    all_mi = {}
    for ax, target in zip(axes.flat, GRAB_TARGETS):
        sub = df[feats + [target]].dropna()
        if len(sub) < 20:
            ax.set_visible(False)
            continue
        X = sub[feats].values
        y = sub[target].values
        mi = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi, index=feats).sort_values(ascending=True)
        all_mi[target] = mi_series

        short_feats = [s(f) for f in mi_series.index]
        colors = ["#D94801" if v > mi_series.quantile(0.75) else "#2171B5"
                  for v in mi_series.values]
        ax.barh(range(len(mi_series)), mi_series.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(mi_series)))
        ax.set_yticklabels(short_feats, fontsize=7)
        ax.set_title(s(target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mutual Information score")
        ax.grid(axis="x", alpha=0.3)
        # Annotate top 5
        for i, (v, feat) in enumerate(zip(mi_series.values, mi_series.index)):
            if v > mi_series.quantile(0.80):
                ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7, color="#8B0000")

    plt.tight_layout()
    return save(fig, "03_mutual_information"), all_mi


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

    fig, big_axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle("Ridge regression residuals — linearity diagnostic\n"
                 "(random scatter = linear relationship; patterns = non-linearity)",
                 fontsize=13, fontweight="bold")

    for row_ax, target in zip(big_axes, GRAB_TARGETS):
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


# ── HTML report generator ─────────────────────────────────────────────────────

def img_tag(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0;">'


CHART_META = [
    ("01_missing_new_columns",   "1. Missing data — new columns",
     "Coverage (%) of each new column by calendar month. "
     "White = no data; darker green = more complete. Identifies which months/years "
     "introduced new parameters and where gaps remain."),
    ("02_pearson_vs_spearman",   "2. Pearson vs Spearman correlation",
     "For each effluent target, features are ranked by |Spearman r|. "
     "A large gap between the two bars for a feature signals a non-linear but "
     "monotonic relationship — direct justification for using RF over linear models."),
    ("03_mutual_information",    "3. Mutual Information scores",
     "MI captures non-monotonic and interaction-based relationships that both Pearson "
     "and Spearman miss. High MI + low Pearson = strongly non-linear feature. "
     "Red bars = top quartile."),
    ("04_scatter_Effluent_BOD_mgL_Grab",   "4a. Feature scatter grid — Effluent BOD",
     "Top 15 features vs Effluent BOD (Grab). Dashed line = linear regression fit. "
     "Colour = year. Curvature or fan shapes indicate non-linearity."),
    ("04_scatter_Effluent_COD_mgL_Grab",   "4b. Feature scatter grid — Effluent COD", ""),
    ("04_scatter_Effluent_TSS_mgL_Grab",   "4c. Feature scatter grid — Effluent TSS", ""),
    ("04_scatter_Effluent_pH_Grab",         "4d. Feature scatter grid — Effluent pH", ""),
    ("05_aeration_timeseries",   "5. Aeration tank time series",
     "Existing (blue) and New (orange) aeration tank DO, MLSS, and SVI over the full "
     "study period. The dashed red line on DO marks the 0.5 mg/L minimum threshold. "
     "Extended periods below threshold correlate with degraded effluent quality."),
    ("06_cross_stage_heatmap",   "6. Cross-stage correlation heatmap",
     "All process features vs all effluent targets. Left = Pearson, Right = Spearman. "
     "Rows sorted by mean |Pearson| across the four grab targets. "
     "Large Pearson–Spearman differences highlight non-linear relationships."),
    ("07_stage_removal",         "7. Stage removal efficiency",
     "Median concentration ± IQR (shaded band) at each treatment stage for BOD, COD, TSS. "
     "Log y-axis. Overall removal % annotated at the effluent stage."),
    ("08_do_vs_effluent",        "8. Aeration DO vs effluent quality",
     "Scatter of DO (existing tank) against effluent BOD, COD, and TSS. "
     "If a threshold effect exists near DO = 0.5 mg/L, effluent quality degrades "
     "sharply — evidence for non-linearity that RF can capture but Ridge cannot."),
    ("09_svi_mlss_vs_tss",       "9. Settleability / biomass vs effluent TSS",
     "SVI, MLSS, and SV30 (existing and new tanks) vs effluent TSS. "
     "Poor settling (high SVI) is expected to drive high effluent TSS. "
     "Correlation coefficient r shown in each title."),
    ("10_linearity_residuals",   "10. Ridge residuals — linearity diagnostic",
     "Ridge (alpha=1, StandardScaler) trained on 2021–2024 process features, tested on 2025. "
     "Panels per target: residuals vs fitted values; vs time (blue=train, red=test); "
     "vs highest-coefficient feature. Systematic patterns confirm non-linearity."),
]

def build_html(plot_paths):
    title = "EDA Full Report — All_Years_Full.xlsx"
    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: Calibri, Arial, sans-serif; max-width: 1400px;
          margin: 0 auto; padding: 20px; background: #f5f5f5; color: #222; }}
  h1   {{ color: #1F4E79; border-bottom: 3px solid #1F4E79; padding-bottom: 8px; }}
  h2   {{ color: #2E75B6; margin-top: 40px; border-left: 5px solid #2E75B6;
          padding-left: 10px; }}
  p    {{ max-width: 900px; line-height: 1.6; }}
  .card {{ background: white; padding: 20px; border-radius: 8px;
           box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
  img  {{ border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Full exploratory data analysis using the enriched 60-column dataset
(<code>All_Years_Full.xlsx</code>). Charts include correlation analysis,
mutual information, aeration diagnostics, and Ridge residual checks for
linearity. These inform feature selection and model choice.</p>
"""]

    path_map = {os.path.splitext(os.path.basename(p))[0]: p
                for p in plot_paths}

    for slug, heading, desc in CHART_META:
        if slug not in path_map:
            continue
        parts.append(f'<div class="card">\n<h2>{heading}</h2>\n')
        if desc:
            parts.append(f"<p>{desc}</p>\n")
        parts.append(img_tag(path_map[slug]))
        parts.append("\n</div>\n")

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

    all_paths = []

    print("Generating charts...")
    all_paths.append(plot_missing_new(df))
    all_paths.append(plot_pearson_vs_spearman(df))
    mi_path, all_mi = plot_mutual_information(df)
    all_paths.append(mi_path)
    all_paths += plot_scatter_grids(df, all_mi)
    all_paths.append(plot_aeration_timeseries(df))
    all_paths.append(plot_cross_stage_heatmap(df))
    all_paths.append(plot_stage_removal(df))
    all_paths.append(plot_do_vs_effluent(df))
    all_paths.append(plot_svi_mlss_vs_tss(df))
    all_paths.append(plot_linearity_residuals(df))

    print("\nBuilding HTML report...")
    build_html(all_paths)
    print(f"\nDone. {len(all_paths)} charts → {PLOTS_DIR}/")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
