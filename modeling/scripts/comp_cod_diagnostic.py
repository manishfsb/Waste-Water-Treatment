"""
comp_cod_diagnostic.py - Time-series diagnostic for Effluent COD (mg/L, Composite).

Plots raw daily measurements year-by-year, overlaid with the training-set
mean ± 1 std band (computed on 2021-2024 only, never 2025).

Outputs:
    reports/comp_cod_diagnostic.png
    reports/comp_cod_diagnostic_monthly.png   - monthly median by year

Usage:
    .venv/bin/python3 modeling/scripts/comp_cod_diagnostic.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR  = os.path.dirname(MODELING_DIR)
DATA_FILE    = os.path.join(PROJECT_DIR, "raw_data", "All_Years_Full.xlsx")
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET    = "Effluent COD (mg/L, Composite)"
TRAIN_MAX_YEAR = 2024
TEST_YEAR      = 2025

YEAR_COLORS = {
    2020: "#6BAED6",
    2021: "#2171B5",
    2022: "#74C476",
    2023: "#238B45",
    2024: "#FD8D3C",
    2025: "#D94801",
}

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading {DATA_FILE} ...")
df = pd.read_excel(DATA_FILE, parse_dates=["Date"])
df = df[["Date", TARGET]].dropna(subset=[TARGET]).sort_values("Date").reset_index(drop=True)
df["year"]  = df["Date"].dt.year
df["month"] = df["Date"].dt.month

train = df[df["year"] <= TRAIN_MAX_YEAR]
test  = df[df["year"] == TEST_YEAR]

train_mean = train[TARGET].mean()
train_std  = train[TARGET].std()
train_p25  = train[TARGET].quantile(0.25)
train_p75  = train[TARGET].quantile(0.75)

print(f"Target: {TARGET}")
print(f"Train (≤{TRAIN_MAX_YEAR}): n={len(train):,}  mean={train_mean:.1f}  "
      f"std={train_std:.1f}  IQR=[{train_p25:.1f}, {train_p75:.1f}]")
print(f"Test  (={TEST_YEAR}):  n={len(test):,}  mean={test[TARGET].mean():.1f}  "
      f"std={test[TARGET].std():.1f}")

# ── Plot 1: Raw time series by year ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Training mean ± 1 std band
ax.axhspan(train_mean - train_std, train_mean + train_std,
           color="#ffffff", alpha=0.07, label="Train mean ± 1 std")
ax.axhline(train_mean, color="#ffffff", lw=1.2, linestyle="--",
           alpha=0.5, label=f"Train mean ({train_mean:.1f})")

# Per-year scatter/lines
for year, grp in df.groupby("year"):
    color = YEAR_COLORS.get(year, "#aaaaaa")
    alpha = 0.95 if year == TEST_YEAR else 0.70
    lw    = 1.6 if year == TEST_YEAR else 1.0
    ax.plot(grp["Date"], grp[TARGET], color=color, lw=lw, alpha=alpha,
            label=str(year))
    ax.scatter(grp["Date"], grp[TARGET], color=color, s=12, alpha=alpha,
               edgecolors="none")

# Train/test boundary
boundary = pd.Timestamp(f"{TEST_YEAR}-01-01")
ax.axvline(boundary, color="#F0B849", lw=1.5, linestyle=":", alpha=0.9,
           label="Train | Test boundary")

# Year-level stats annotation
for year, grp in df.groupby("year"):
    m = grp[TARGET].mean()
    s = grp[TARGET].std()
    n = len(grp)
    ax.annotate(
        f"{year}\nμ={m:.0f}\nσ={s:.0f}\nn={n}",
        xy=(grp["Date"].mean(), grp[TARGET].max()),
        fontsize=6.5, color=YEAR_COLORS.get(year, "#aaaaaa"),
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="none", alpha=0.7),
    )

ax.set_title(f"Effluent COD (Composite) - Raw Measurements by Year\n"
             f"Training mean = {train_mean:.1f} mg/L  |  2025 mean = {test[TARGET].mean():.1f} mg/L  "
             f"|  2025 std = {test[TARGET].std():.1f} mg/L",
             fontsize=11, color="white", pad=10)
ax.set_xlabel("Date", fontsize=9, color="#cccccc")
ax.set_ylabel("Effluent COD (mg/L, Composite)", fontsize=9, color="#cccccc")
ax.tick_params(colors="#cccccc", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.legend(fontsize=7.5, ncol=5, loc="upper left",
          facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white")

plt.tight_layout()
out1 = os.path.join(REPORTS_DIR, "comp_cod_diagnostic.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out1}")

# ── Plot 2: Monthly median by year (seasonal pattern) ─────────────────────────
monthly = df.groupby(["year", "month"])[TARGET].agg(["median", "mean", "std", "count"]).reset_index()
monthly.columns = ["year", "month", "median", "mean", "std", "count"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
fig.patch.set_facecolor("#1a1a2e")
years_plot = sorted(df["year"].unique())

for ax, year in zip(axes.flat, years_plot):
    ax.set_facecolor("#1a1a2e")
    grp = monthly[monthly["year"] == year]
    color = YEAR_COLORS.get(year, "#aaaaaa")

    ax.bar(grp["month"], grp["median"], color=color, alpha=0.75, width=0.7)
    ax.errorbar(grp["month"], grp["median"], yerr=grp["std"].fillna(0),
                fmt="none", color="white", alpha=0.4, capsize=3, lw=0.8)

    # overlay training mean
    ax.axhline(train_mean, color="#F0B849", lw=1.0, linestyle="--",
               alpha=0.7, label=f"Train mean ({train_mean:.0f})")

    ax.set_title(f"{year}  (n={grp['count'].sum():.0f})",
                 color=color, fontsize=10, fontweight="bold")
    ax.set_xlabel("Month", fontsize=8, color="#cccccc")
    ax.set_ylabel("Median COD (mg/L)", fontsize=8, color="#cccccc")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                        fontsize=7, color="#cccccc")
    ax.tick_params(colors="#cccccc", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    if year == years_plot[0]:
        ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444444",
                  labelcolor="white")

fig.suptitle("Comp COD - Monthly Median by Year  (training mean dashed gold)",
             fontsize=12, color="white", y=1.01)
plt.tight_layout()
out2 = os.path.join(REPORTS_DIR, "comp_cod_diagnostic_monthly.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out2}")

# ── Plot 3: 2025 vs training distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

bins = np.linspace(0, max(df[TARGET].quantile(0.99), 1), 40)
ax.hist(train[TARGET], bins=bins, color="#4A90D9", alpha=0.6, label=f"Train 2021-{TRAIN_MAX_YEAR} (n={len(train)})")
ax.hist(test[TARGET],  bins=bins, color="#D94801", alpha=0.75, label=f"Test 2025 (n={len(test)})")
ax.axvline(train_mean, color="#4A90D9", lw=1.5, linestyle="--", alpha=0.9)
ax.axvline(test[TARGET].mean(), color="#D94801", lw=1.5, linestyle="--", alpha=0.9)

ax.set_title(f"Comp COD - Training vs 2025 Distribution\n"
             f"Train: μ={train_mean:.1f}, σ={train_std:.1f}  |  "
             f"2025: μ={test[TARGET].mean():.1f}, σ={test[TARGET].std():.1f}",
             fontsize=10, color="white")
ax.set_xlabel("Effluent COD (mg/L, Composite)", fontsize=9, color="#cccccc")
ax.set_ylabel("Count", fontsize=9, color="#cccccc")
ax.tick_params(colors="#cccccc")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white")

plt.tight_layout()
out3 = os.path.join(REPORTS_DIR, "comp_cod_diagnostic_dist.png")
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out3}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n── Per-year summary ────────────────────────────────────────────────────")
for year, grp in df.groupby("year"):
    tag = " ← TEST" if year == TEST_YEAR else ""
    vals = grp[TARGET]
    print(f"  {year}{tag:8s}  n={len(vals):4d}  "
          f"mean={vals.mean():6.1f}  std={vals.std():5.1f}  "
          f"min={vals.min():5.1f}  max={vals.max():7.1f}  "
          f"IQR=[{vals.quantile(0.25):.1f}, {vals.quantile(0.75):.1f}]")

print("\n── Distribution shift check ────────────────────────────────────────────")
frac_in_train_iqr = ((test[TARGET] >= train_p25) & (test[TARGET] <= train_p75)).mean()
frac_above_train_p75 = (test[TARGET] > train_p75).mean()
print(f"  Fraction of 2025 values within training IQR [{train_p25:.1f}, {train_p75:.1f}]: "
      f"{frac_in_train_iqr:.1%}")
print(f"  Fraction of 2025 values above training P75 ({train_p75:.1f}):  "
      f"{frac_above_train_p75:.1%}")
overlap = 1 - abs(test[TARGET].mean() - train_mean) / (train_std + 1e-9)
print(f"  Mean shift relative to training std: "
      f"{abs(test[TARGET].mean() - train_mean) / train_std:.2f}x")
print(f"\nPlots written to {REPORTS_DIR}/")
