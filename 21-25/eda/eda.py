"""
EDA script for the merged 2021–2025 WWTP dataset.

Reads All_Years_Merged.xlsx and produces a set of diagnostic plots saved to
21-25/eda_plots/. Run after merge_all_years.py.

Plots generated:
  1.  missing_values_heatmap  — % missing per attribute, heatmap over time
  2.  missing_values_bar      — bar chart of overall % missing per column
  3.  timeseries_core         — line plots of flow, power/flow, inlet/effluent params
  4.  distributions           — histograms + KDE for each numeric column
  5.  boxplot_by_year         — box plots per parameter grouped by year
  6.  boxplot_by_month        — box plots per parameter grouped by calendar month
  7.  correlation_heatmap     — Pearson correlation matrix of numeric columns
  8.  outliers_scatter        — scatter plots with control limit lines
  9.  removal_efficiency      — BOD/COD/TSS removal % over time + distributions
  10. compliance_over_time    — monthly pass/fail/missing rate for each effluent param
  11. power_vs_flow           — power_per_flow scatter vs. flow, coloured by year
  12. seasonal_heatmap        — monthly average heatmap (param × month)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EXCEL_IN  = os.path.join(BASE_DIR, "All_Years_Merged.xlsx")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Control limits (for outlier scatter plots) ─────────────────────────────────
CONTROL_LIMITS = {
    "inlet_ph":      {"type": "range", "min": 6.0,  "max": 9.0},
    "inlet_bod":     {"type": "max",   "max": 300},
    "inlet_cod":     {"type": "max",   "max": 800},
    "inlet_tss":     {"type": "max",   "max": 400},
    "effluent_ph":   {"type": "range", "min": 6.5,  "max": 8.0},
    "effluent_bod":  {"type": "max",   "max": 10},
    "effluent_cod":  {"type": "max",   "max": 250},
    "effluent_tss":  {"type": "max",   "max": 10},
    "power_per_flow":{"type": "max",   "max": 482.02},
}

REMOVAL_PAIRS = [
    ("inlet_bod", "effluent_bod", "BOD"),
    ("inlet_cod", "effluent_cod", "COD"),
    ("inlet_tss", "effluent_tss", "TSS"),
]

NUMERIC_COLS = [
    "power_ge", "power_nea", "power_total", "power_per_flow", "flow",
    "inlet_ph", "inlet_bod", "inlet_cod", "inlet_tss",
    "inlet_ph_comp", "inlet_bod_comp", "inlet_cod_comp", "inlet_tss_comp",
    "effluent_ph", "effluent_bod", "effluent_cod", "effluent_tss", "effluent_frc",
    "effluent_ph_comp", "effluent_bod_comp", "effluent_cod_comp", "effluent_tss_comp",
]

PARAM_LABELS = {
    "power_ge":          "Power GE (KW)",
    "power_nea":         "Power NEA (KW)",
    "power_total":       "Power Total (KW)",
    "power_per_flow":    "Power/Flow (KW/ML)",
    "flow":              "Flow (MLD)",
    "inlet_ph":          "Inlet pH",
    "inlet_bod":         "Inlet BOD (mg/L)",
    "inlet_cod":         "Inlet COD (mg/L)",
    "inlet_tss":         "Inlet TSS (mg/L)",
    "inlet_ph_comp":     "Inlet pH (Comp.)",
    "inlet_bod_comp":    "Inlet BOD (mg/L, Comp.)",
    "inlet_cod_comp":    "Inlet COD (mg/L, Comp.)",
    "inlet_tss_comp":    "Inlet TSS (mg/L, Comp.)",
    "effluent_ph":       "Effluent pH",
    "effluent_bod":      "Effluent BOD (mg/L)",
    "effluent_cod":      "Effluent COD (mg/L)",
    "effluent_tss":      "Effluent TSS (mg/L)",
    "effluent_frc":      "Effluent FRC (mg/L)",
    "effluent_ph_comp":  "Effluent pH (Comp.)",
    "effluent_bod_comp": "Effluent BOD (mg/L, Comp.)",
    "effluent_cod_comp": "Effluent COD (mg/L, Comp.)",
    "effluent_tss_comp": "Effluent TSS (mg/L, Comp.)",
}

sns.set_theme(style="whitegrid", palette="tab10", font_scale=0.9)
YEAR_PALETTE = {2020: "#a65628", 2021: "#e41a1c", 2022: "#377eb8",
                2023: "#4daf4a", 2024: "#984ea3", 2025: "#ff7f00"}


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(EXCEL_IN):
        raise FileNotFoundError(f"Merged Excel not found: {EXCEL_IN}\nRun merge_all_years.py first.")

    df = pd.read_excel(EXCEL_IN, sheet_name=0, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # Rename columns to JSON keys using header text
    rename = {v[0]: v[1] for v in [
        ("Date",                       "date"),
        ("Power GE (KW)",              "power_ge"),
        ("Power NEA (KW)",             "power_nea"),
        ("Power Total (KW)",           "power_total"),
        ("Power / Flow (KW/ML)",       "power_per_flow"),
        ("Flow (MLD)",                 "flow"),
        ("Inlet pH (Grab)",            "inlet_ph"),
        ("Inlet BOD (mg/L, Grab)",     "inlet_bod"),
        ("Inlet COD (mg/L, Grab)",     "inlet_cod"),
        ("Inlet TSS (mg/L, Grab)",     "inlet_tss"),
        ("Inlet pH (Composite)",       "inlet_ph_comp"),
        ("Inlet BOD (mg/L, Composite)","inlet_bod_comp"),
        ("Inlet COD (mg/L, Composite)","inlet_cod_comp"),
        ("Inlet TSS (mg/L, Composite)","inlet_tss_comp"),
        ("Effluent pH (Grab)",         "effluent_ph"),
        ("Effluent BOD (mg/L, Grab)",  "effluent_bod"),
        ("Effluent COD (mg/L, Grab)",  "effluent_cod"),
        ("Effluent TSS (mg/L, Grab)",  "effluent_tss"),
        ("Effluent FRC (mg/L)",        "effluent_frc"),
        ("Effluent pH (Composite)",    "effluent_ph_comp"),
        ("Effluent BOD (mg/L, Composite)", "effluent_bod_comp"),
        ("Effluent COD (mg/L, Composite)", "effluent_cod_comp"),
        ("Effluent TSS (mg/L, Composite)", "effluent_tss_comp"),
    ]}
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df)} rows, {df['year'].nunique()} years "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png")


# ── 1 & 2. Missing values ──────────────────────────────────────────────────────

def plot_missing_bar(df):
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    pct = df[cols].isnull().mean() * 100
    pct = pct.sort_values(ascending=False)
    labels = [PARAM_LABELS.get(c, c) for c in pct.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, pct.values, color=sns.color_palette("RdYlGn_r", len(pct)))
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Attribute (2021–2025)", fontsize=13, fontweight="bold")
    ax.axvline(50, color="red", lw=1, linestyle="--", alpha=0.6, label="50% threshold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, pct.values):
        if val > 1:
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "02_missing_values_bar")


def plot_missing_heatmap(df):
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    # Resample to monthly % missing
    tmp = df.set_index("date")[cols].copy()
    monthly_missing = tmp.resample("ME").apply(lambda x: x.isnull().mean() * 100)
    monthly_missing.index = monthly_missing.index.to_period("M").astype(str)

    fig, ax = plt.subplots(figsize=(18, 8))
    labels = [PARAM_LABELS.get(c, c) for c in monthly_missing.columns]
    sns.heatmap(
        monthly_missing.T,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0, vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "% Missing"},
        xticklabels=4,
    )
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_title("Monthly Missing Value Rate per Attribute (%)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    save(fig, "01_missing_values_heatmap")


# ── 3. Time-series ─────────────────────────────────────────────────────────────

def plot_timeseries(df):
    groups = [
        ("Flow & Power", ["flow", "power_per_flow"]),
        ("Inlet Quality (Grab)", ["inlet_ph", "inlet_bod", "inlet_cod", "inlet_tss"]),
        ("Effluent Quality (Grab)", ["effluent_ph", "effluent_bod", "effluent_cod", "effluent_tss"]),
        ("Power Components (KWh)", ["power_ge", "power_nea", "power_total"]),
    ]
    for group_name, cols in groups:
        cols = [c for c in cols if c in df.columns]
        fig, axes = plt.subplots(len(cols), 1, figsize=(16, 3 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            # 7-day rolling mean to reduce noise
            series = df.set_index("date")[col].rolling("7D").mean()
            ax.plot(series.index, series.values, lw=0.9, alpha=0.85)
            # Control limit
            if col in CONTROL_LIMITS:
                cl = CONTROL_LIMITS[col]
                if cl["type"] == "max":
                    ax.axhline(cl["max"], color="red", lw=1, linestyle="--",
                               alpha=0.7, label=f"Limit: {cl['max']}")
                elif cl["type"] == "range":
                    ax.axhline(cl["min"], color="orange", lw=1, linestyle="--", alpha=0.7)
                    ax.axhline(cl["max"], color="orange", lw=1, linestyle="--", alpha=0.7)
                ax.legend(fontsize=7)
            ax.set_ylabel(PARAM_LABELS.get(col, col), fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        axes[-1].set_xlabel("Date")
        fig.suptitle(f"Time Series — {group_name} (7-day rolling mean)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        safe_name = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        save(fig, f"03_timeseries_{safe_name}")


# ── 4. Distributions ───────────────────────────────────────────────────────────

def plot_distributions(df):
    cols = [c for c in NUMERIC_COLS if c in df.columns and df[c].notna().sum() > 20]
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="none")
        try:
            data.plot.kde(ax=axes[i], color="darkblue", lw=1.5)
        except Exception:
            pass  # scipy not available
        if col in CONTROL_LIMITS:
            cl = CONTROL_LIMITS[col]
            if cl["type"] == "max":
                axes[i].axvline(cl["max"], color="red", lw=1.2, linestyle="--")
            elif cl["type"] == "range":
                axes[i].axvline(cl["min"], color="orange", lw=1.2, linestyle="--")
                axes[i].axvline(cl["max"], color="orange", lw=1.2, linestyle="--")
        axes[i].set_title(PARAM_LABELS.get(col, col), fontsize=8)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distributions of All Attributes (2021–2025)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_distributions")


# ── 4b. Effluent distributions (dedicated) ────────────────────────────────────

EFFLUENT_GRAB_PARAMS = [
    # (col, short_label, compliance_type, limit_lo, limit_hi)
    ("effluent_bod", "BOD (mg/L)",  "max",   None, 10),
    ("effluent_cod", "COD (mg/L)",  "max",   None, 250),
    ("effluent_tss", "TSS (mg/L)",  "max",   None, 10),
    ("effluent_ph",  "pH",          "range",  6.5,  8.0),
]
EFFLUENT_COMP_PARAMS = [
    ("effluent_bod_comp", "BOD (mg/L)",  "max",   None, 10),
    ("effluent_cod_comp", "COD (mg/L)",  "max",   None, 250),
    ("effluent_tss_comp", "TSS (mg/L)",  "max",   None, 10),
    ("effluent_ph_comp",  "pH",          "range",  6.5,  8.0),
]
LOG_SCALE_COLS = {"effluent_bod", "effluent_cod", "effluent_tss",
                  "effluent_bod_comp", "effluent_cod_comp", "effluent_tss_comp"}


def _draw_effluent_dist_panel(df, params, ax_v, ax_h, col, label,
                               ctype, lo, hi):
    """Draw one violin (ax_v) + histogram/KDE (ax_h) row for a single parameter."""
    use_log  = col in LOG_SCALE_COLS
    data_all = df[[col, "year"]].dropna().copy()
    data_all["year"] = data_all["year"].astype(int)

    all_years   = sorted(data_all["year"].unique())
    valid_years = [y for y in all_years
                   if len(data_all.loc[data_all["year"] == y, col]) > 1]
    groups      = [data_all.loc[data_all["year"] == y, col].values
                   for y in valid_years]

    # ── Violin ────────────────────────────────────────────────────────────────
    parts = ax_v.violinplot(groups, positions=range(len(valid_years)),
                            showmedians=True, showextrema=True)
    for body, yr in zip(parts["bodies"], valid_years):
        body.set_facecolor(YEAR_PALETTE.get(yr, "#999"))
        body.set_alpha(0.65)
        body.set_edgecolor("white")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_color("#555")
        parts[key].set_linewidth(1)

    ax_v.set_xticks(range(len(valid_years)))
    ax_v.set_xticklabels([str(y) for y in valid_years], fontsize=9)
    ax_v.set_ylabel(label, fontsize=9)
    ax_v.set_title(f"{label} — by year", fontsize=10)

    if use_log:
        ax_v.set_yscale("log")
        ax_v.yaxis.set_major_formatter(mticker.ScalarFormatter())

    if ctype == "max" and hi is not None:
        ax_v.axhline(hi, color="red", lw=1.2, linestyle="--",
                     label=f"Limit {hi}", zorder=5)
        ax_v.legend(fontsize=8)
    elif ctype == "range":
        ax_v.axhline(lo, color="orange", lw=1.2, linestyle="--",
                     label=f"Min {lo}", zorder=5)
        ax_v.axhline(hi, color="red",    lw=1.2, linestyle="--",
                     label=f"Max {hi}", zorder=5)
        ax_v.legend(fontsize=8)

    n_obs = len(data_all)
    med   = data_all[col].median()
    ax_v.annotate(f"n={n_obs}  median={med:.2f}",
                  xy=(0.01, 0.97), xycoords="axes fraction",
                  va="top", fontsize=8, color="#444")

    # ── Histogram + KDE ───────────────────────────────────────────────────────
    vals = data_all[col].values
    if use_log:
        vals_pos  = vals[vals > 0]
        vals_plot = np.log10(vals_pos)
        ax_h.hist(vals_plot, bins=35, density=True,
                  color="steelblue", alpha=0.55, edgecolor="none")
        try:
            pd.Series(vals_plot).plot.kde(ax=ax_h, color="darkblue", lw=1.6)
        except Exception:
            pass
        if ctype == "max" and hi is not None:
            ax_h.axvline(np.log10(hi), color="red", lw=1.2, linestyle="--",
                         label=f"Limit {hi}")
        tick_vals = [v for v in [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 300]
                     if len(vals_pos) and
                     np.log10(vals_pos.max()) >= np.log10(v) >=
                     np.log10(max(0.1, vals_pos.min()))]
        ax_h.set_xticks([np.log10(v) for v in tick_vals])
        ax_h.set_xticklabels([str(v) for v in tick_vals], fontsize=7)
        ax_h.set_xlabel(f"log scale  ({label})", fontsize=8)
    else:
        ax_h.hist(vals, bins=35, density=True,
                  color="steelblue", alpha=0.55, edgecolor="none")
        try:
            pd.Series(vals).plot.kde(ax=ax_h, color="darkblue", lw=1.6)
        except Exception:
            pass
        if ctype == "max" and hi is not None:
            ax_h.axvline(hi, color="red", lw=1.2, linestyle="--",
                         label=f"Limit {hi}")
        elif ctype == "range":
            ax_h.axvline(lo, color="orange", lw=1.2, linestyle="--")
            ax_h.axvline(hi, color="red",    lw=1.2, linestyle="--")
        ax_h.set_xlabel(label, fontsize=8)

    ax_h.set_ylabel("Density", fontsize=8)
    ax_h.set_title("All years" + (" (log₁₀ x-axis)" if use_log else ""), fontsize=9)
    ax_h.tick_params(labelsize=7)
    if ax_h.get_legend_handles_labels()[0]:
        ax_h.legend(fontsize=7)


def _plot_effluent_dist_group(df, params, sample_type, filename):
    """
    Generate violin + histogram panels for one group of effluent parameters.
    Adds a shared legend and a how-to-read annotation block.
    """
    avail = [(c, lbl, ct, lo, hi) for c, lbl, ct, lo, hi in params
             if c in df.columns and df[c].notna().sum() > 20]
    if not avail:
        return

    n   = len(avail)
    fig = plt.figure(figsize=(14, 4.5 * n + 1.4))

    # Top row: how-to-read box; remaining rows: one per parameter
    gs  = fig.add_gridspec(n + 1, 2,
                           height_ratios=[0.30] + [1.0] * n,
                           width_ratios=[2, 1],
                           hspace=0.52, wspace=0.3,
                           top=0.97, bottom=0.03)

    # ── How-to-read annotation ─────────────────────────────────────────────────
    ax_note = fig.add_subplot(gs[0, :])
    ax_note.axis("off")
    how_to = (
        "HOW TO READ THESE CHARTS\n"
        "Left panel (violin by year)  — Each coloured shape shows the full spread of daily values "
        "for one year. Wider = more days at that concentration. The horizontal black bar is the "
        "median. Whiskers extend to the min/max. The red dashed line is the compliance limit.\n"
        "Right panel (histogram + curve)  — Bars show how often each value range occurred across "
        "all years combined. The smooth curve (KDE) is a fitted density. "
        "BOD, COD and TSS use a log₁₀ x-axis so the long right tail (rare high-pollution days) "
        "is visible — tick labels show original mg/L values. pH uses a linear axis."
    )
    ax_note.text(0.0, 1.0, how_to, transform=ax_note.transAxes,
                 va="top", ha="left", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8",
                           edgecolor="#BBBBBB", linewidth=0.8),
                 wrap=True, linespacing=1.55)

    # ── Parameter rows ─────────────────────────────────────────────────────────
    for row, (col, label, ctype, lo, hi) in enumerate(avail):
        ax_v = fig.add_subplot(gs[row + 1, 0])
        ax_h = fig.add_subplot(gs[row + 1, 1])
        _draw_effluent_dist_panel(df, params, ax_v, ax_h, col, label, ctype, lo, hi)

    fig.savefig(os.path.join(PLOTS_DIR, f"{filename}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}.png")


def plot_effluent_distributions(df):
    _plot_effluent_dist_group(df, EFFLUENT_GRAB_PARAMS,
                              "Grab Samples",
                              "04b_effluent_distributions_grab")
    _plot_effluent_dist_group(df, EFFLUENT_COMP_PARAMS,
                              "Composite Samples",
                              "04b_effluent_distributions_comp")


# ── 5. Box plots by year ───────────────────────────────────────────────────────

def plot_boxplot_by_year(df):
    core_cols = [c for c in [
        "flow", "power_per_flow", "inlet_ph", "inlet_bod", "inlet_cod", "inlet_tss",
        "effluent_ph", "effluent_bod", "effluent_cod", "effluent_tss",
    ] if c in df.columns]

    ncols = 2
    nrows = (len(core_cols) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(core_cols):
        data = df[["year", col]].dropna()
        years = sorted(data["year"].unique())
        groups = [data.loc[data["year"] == y, col].values for y in years]
        bp = axes[i].boxplot(groups, patch_artist=True, notch=False,
                             flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, yr in zip(bp["boxes"], years):
            patch.set_facecolor(YEAR_PALETTE.get(yr, "steelblue"))
            patch.set_alpha(0.7)
        if col in CONTROL_LIMITS:
            cl = CONTROL_LIMITS[col]
            if cl["type"] == "max":
                axes[i].axhline(cl["max"], color="red", lw=1, linestyle="--", alpha=0.7)
            elif cl["type"] == "range":
                axes[i].axhline(cl["min"], color="orange", lw=1, linestyle="--", alpha=0.7)
                axes[i].axhline(cl["max"], color="orange", lw=1, linestyle="--", alpha=0.7)
        axes[i].set_xticklabels([str(y) for y in years])
        axes[i].set_title(PARAM_LABELS.get(col, col), fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=YEAR_PALETTE.get(y, "steelblue"), label=str(y))
                  for y in sorted(YEAR_PALETTE.keys())]
    fig.legend(handles=legend_els, title="Year", loc="lower right", ncol=5, fontsize=8)

    fig.suptitle("Annual Distributions by Year (box = IQR, whiskers = 1.5×IQR)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_boxplot_by_year")


# ── 6. Box plots by calendar month ────────────────────────────────────────────

def plot_boxplot_by_month(df):
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    core_cols = [c for c in ["flow", "inlet_bod", "inlet_cod", "inlet_tss",
                              "effluent_bod", "effluent_cod", "effluent_tss"]
                 if c in df.columns]
    ncols = 2
    nrows = (len(core_cols) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(core_cols):
        data = df[["month", col]].dropna()
        groups = [data.loc[data["month"] == m, col].values for m in range(1, 13)]
        bp = axes[i].boxplot(groups, patch_artist=True,
                             flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        if col in CONTROL_LIMITS:
            cl = CONTROL_LIMITS[col]
            if cl["type"] == "max":
                axes[i].axhline(cl["max"], color="red", lw=1, linestyle="--", alpha=0.7)
        axes[i].set_xticklabels(MONTH_NAMES, fontsize=7)
        axes[i].set_title(PARAM_LABELS.get(col, col), fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Seasonal Distributions (all years combined)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_boxplot_by_month")


# ── 7. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation(df):
    grab_cols = [c for c in [
        "flow", "power_per_flow",
        "inlet_ph", "inlet_bod", "inlet_cod", "inlet_tss",
        "effluent_ph", "effluent_bod", "effluent_cod", "effluent_tss", "effluent_frc",
    ] if c in df.columns]

    corr = df[grab_cols].corr()
    labels = [PARAM_LABELS.get(c, c) for c in grab_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8},
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Pearson Correlation Matrix (Grab Samples)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    save(fig, "07_correlation_heatmap")


# ── 8. Outlier scatter ─────────────────────────────────────────────────────────

def plot_outliers(df):
    params = [c for c in CONTROL_LIMITS if c in df.columns]
    ncols = 2
    nrows = (len(params) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False)
    axes = axes.flatten()

    for i, col in enumerate(params):
        cl = CONTROL_LIMITS[col]
        data = df[["date", "year", col]].dropna()
        colors = [YEAR_PALETTE.get(y, "grey") for y in data["year"]]
        axes[i].scatter(data["date"], data[col], c=colors, s=4, alpha=0.5)
        if cl["type"] == "max":
            axes[i].axhline(cl["max"], color="red", lw=1.2, linestyle="--",
                            label=f"Limit: {cl['max']}")
        elif cl["type"] == "range":
            axes[i].axhline(cl["min"], color="orange", lw=1.2, linestyle="--",
                            label=f"Min: {cl['min']}")
            axes[i].axhline(cl["max"], color="red", lw=1.2, linestyle="--",
                            label=f"Max: {cl['max']}")
        axes[i].set_title(PARAM_LABELS.get(col, col), fontsize=9)
        axes[i].legend(fontsize=7)
        axes[i].tick_params(axis="x", labelsize=6, rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Daily Values vs. Control Limits (coloured by year)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_outliers_scatter")


# ── 9. Removal efficiency ──────────────────────────────────────────────────────

def plot_removal_efficiency(df):
    eff_data = {}
    for inlet_col, eff_col, label in REMOVAL_PAIRS:
        if inlet_col not in df.columns or eff_col not in df.columns:
            continue
        both = df[[inlet_col, eff_col, "date", "year"]].dropna()
        both = both[both[inlet_col] > 0]
        both["removal"] = (both[inlet_col] - both[eff_col]) / both[inlet_col] * 100
        eff_data[label] = both

    if not eff_data:
        print("  No removal efficiency data available.")
        return

    n = len(eff_data)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for j, (label, both) in enumerate(eff_data.items()):
        # Time series (7-day rolling)
        ts = both.set_index("date")["removal"].rolling("7D").mean()
        axes[0][j].plot(ts.index, ts.values, lw=0.9, color="steelblue")
        axes[0][j].set_title(f"{label} Removal Efficiency (%)", fontsize=9)
        axes[0][j].set_ylabel("Removal (%)")
        axes[0][j].axhline(90, color="orange", lw=1, linestyle="--", label="90%")
        axes[0][j].axhline(95, color="red",    lw=1, linestyle="--", label="95%")
        axes[0][j].legend(fontsize=7)
        axes[0][j].tick_params(axis="x", labelsize=6)

        # Distribution by year
        groups = [both.loc[both["year"] == y, "removal"].values
                  for y in sorted(both["year"].unique())]
        years  = sorted(both["year"].unique())
        bp = axes[1][j].boxplot(groups, patch_artist=True,
                                 flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, yr in zip(bp["boxes"], years):
            patch.set_facecolor(YEAR_PALETTE.get(yr, "steelblue"))
            patch.set_alpha(0.7)
        axes[1][j].set_xticklabels([str(y) for y in years])
        axes[1][j].set_ylabel("Removal (%)")
        axes[1][j].axhline(90, color="orange", lw=1, linestyle="--")
        axes[1][j].axhline(95, color="red",    lw=1, linestyle="--")

    fig.suptitle("Treatment Removal Efficiency — BOD, COD, TSS",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_removal_efficiency")


# ── 10. Compliance over time ────────────────────────────────────────────────────

def plot_compliance(df):
    COMPLIANCE_PARAMS = [
        ("effluent_ph",  "Effluent pH",  "range", 6.5, 8.0),
        ("effluent_bod", "Effluent BOD", "max",   None, 10),
        ("effluent_cod", "Effluent COD", "max",   None, 250),
        ("effluent_tss", "Effluent TSS", "max",   None, 10),
    ]
    COMPLIANCE_PARAMS = [(p, l, t, mn, mx) for p, l, t, mn, mx in COMPLIANCE_PARAMS
                         if p in df.columns]

    n = len(COMPLIANCE_PARAMS)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    tmp = df.set_index("date").resample("ME")

    for ax, (col, label, ctype, cmin, cmax) in zip(axes, COMPLIANCE_PARAMS):
        def classify(series):
            out = pd.Series(index=series.index, dtype=float)
            for idx, val in series.items():
                if pd.isna(val):
                    out[idx] = np.nan
                elif ctype == "range":
                    out[idx] = 1.0 if cmin <= val <= cmax else 0.0
                else:  # max
                    out[idx] = 1.0 if val <= cmax else 0.0
            return out

        monthly_pass = (
            df.set_index("date")[[col]]
            .resample("ME")
            .apply(lambda g: classify(g[col]).mean())
        )
        pass_rate = monthly_pass
        missing_rate = (
            df.set_index("date")[[col]]
            .resample("ME")
            .apply(lambda g: g[col].isna().mean())
        )

        x = pass_rate.index
        ax.fill_between(x, pass_rate.values * 100, alpha=0.7, color="green", label="Pass %")
        ax.fill_between(x, missing_rate.values * 100, alpha=0.5, color="grey", label="Missing %")
        ax.axhline(80, color="orange", lw=1, linestyle="--", label="80%")
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7, loc="lower left")

    axes[-1].set_xlabel("Month")
    fig.suptitle("Monthly Compliance Rate (%) — Effluent Parameters",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_compliance_over_time")


# ── 11. Power vs. flow ─────────────────────────────────────────────────────────

def plot_power_vs_flow(df):
    data = df[["flow", "power_per_flow", "year"]].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    for yr in sorted(data["year"].unique()):
        sub = data[data["year"] == yr]
        ax.scatter(sub["flow"], sub["power_per_flow"],
                   s=12, alpha=0.5, color=YEAR_PALETTE.get(yr, "grey"), label=str(yr))
    ax.axhline(482.02, color="red", lw=1.2, linestyle="--", label="Limit: 482.02 KWh/ML")
    ax.set_xlabel("Flow (MLD)")
    ax.set_ylabel("Power / Flow (KWh/ML)")
    ax.set_title("Power Efficiency vs. Flow (coloured by year)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "11_power_vs_flow")


# ── 12. Seasonal heatmap ───────────────────────────────────────────────────────

def plot_seasonal_heatmap(df):
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [c for c in [
        "flow", "power_per_flow", "inlet_bod", "inlet_cod", "inlet_tss",
        "effluent_bod", "effluent_cod", "effluent_tss",
    ] if c in df.columns]

    monthly_avg = df.groupby("month")[cols].mean()
    monthly_avg.index = MONTH_NAMES

    # Normalise each column for visual clarity
    normed = (monthly_avg - monthly_avg.min()) / (monthly_avg.max() - monthly_avg.min())

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(normed.T, ax=ax, cmap="YlOrRd", annot=monthly_avg.T.round(1),
                fmt=".1f", linewidths=0.5, annot_kws={"size": 8},
                cbar_kws={"label": "Normalised value"})
    ylabels = [PARAM_LABELS.get(c, c) for c in cols]
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xticklabels(MONTH_NAMES, rotation=0)
    ax.set_title("Seasonal Heatmap — Monthly Average (annotated with actual values)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "12_seasonal_heatmap")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading data from {EXCEL_IN}...")
    df = load_data()
    print(f"Generating plots in {PLOTS_DIR}/")

    print("\n[1/12] Missing values heatmap...")
    plot_missing_heatmap(df)

    print("[2/12] Missing values bar chart...")
    plot_missing_bar(df)

    print("[3/12] Time series...")
    plot_timeseries(df)

    print("[4/12] Distributions...")
    plot_distributions(df)

    print("[4b/12] Effluent distributions — grab & composite (violin + histogram)...")
    plot_effluent_distributions(df)

    print("[5/12] Box plots by year...")
    plot_boxplot_by_year(df)

    print("[6/12] Box plots by month...")
    plot_boxplot_by_month(df)

    print("[7/12] Correlation heatmap...")
    plot_correlation(df)

    print("[8/12] Outlier scatter plots...")
    plot_outliers(df)

    print("[9/12] Removal efficiency...")
    plot_removal_efficiency(df)

    print("[10/12] Compliance over time...")
    plot_compliance(df)

    print("[11/12] Power vs. flow...")
    plot_power_vs_flow(df)

    print("[12/12] Seasonal heatmap...")
    plot_seasonal_heatmap(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
