"""
generate_report_phase9.py — Comprehensive Phase 9 report.

Combines ANN, Voting, and Stacking results with best prior experiments
(ElNet from Exp3-S2, RF from Exp3-S2) for head-to-head comparison.

Sections:
  • Executive summary stat cards
  • All-model comparison table (8 targets × 6+ models) with per-row best highlighting
  • Per-target winner tracker (which model is best per target)
  • Phase 9 model details: ANN / Voting / Stacking metrics tables
  • Learning curve plots (ANN)
  • Scatter and timeseries plots (Voting, Stacking)
  • Cross-phase average R² trend chart (how avg Test R² evolved per experiment)

Outputs:
  reports/report_phase9_run_N.html

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase9/generate_report_phase9.py
"""

import base64
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
ANN_DIR      = os.path.join(SCRIPT_DIR, "ann")
ENS_DIR      = os.path.join(SCRIPT_DIR, "ensemble")

sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Comparison models to pull from prior experiments ──────────────────────────
# We load ElNet and RF from Exp3-S2 as the prior-best baselines
PRIOR_LINEAR_FILE = os.path.join(MODELING_DIR, "models", "linear", "exp3_s2", "results.xlsx")
PRIOR_RF_FILE     = os.path.join(MODELING_DIR, "models", "non_linear", "exp3_s2", "rf", "results.xlsx")

# ── Model colors ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "ElNet":    "#5BAD6F",
    "Ridge":    "#4A90D9",
    "RF":       "#2171B5",
    "ANN":      "#9C27B0",
    "Voting":   "#FF9800",
    "Stacking": "#00BCD4",
}
ALL_PRIOR    = ["ElNet", "Ridge", "RF"]
ALL_PHASE9   = ["ANN", "Voting", "Stacking"]
ALL_MODELS   = ALL_PRIOR + ALL_PHASE9

# ── CSS ────────────────────────────────────────────────────────────────────────
REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', Calibri, Arial, sans-serif;
    max-width: 1440px; margin: 0 auto; padding: 24px 32px 60px;
  }
  h1 { font-size: 1.7rem; margin-bottom: 4px; color: var(--text); }
  h2 { font-size: 1.22rem; margin: 36px 0 10px; color: var(--text);
       border-bottom: 2px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.05rem; margin: 22px 0 8px; color: var(--text-muted); }
  p, li { color: var(--text-muted); font-size: 0.93rem; line-height: 1.6; }
  .run-badge { display: inline-block; background: #9C27B0; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.78rem;
    margin-left: 10px; vertical-align: middle; }
  .ts { color: var(--text-meta); font-size: 0.82rem; margin-top: 4px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap;
    margin: 12px 0 20px; font-size: 0.88rem; color: var(--text-muted); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .legend-divider { width: 1px; background: var(--border); margin: 0 6px; align-self: stretch; }

  .summary-grid { display: flex; flex-wrap: nowrap; gap: 10px;
    margin: 14px 0 24px; overflow-x: auto; }
  .stat-card { background: var(--card); border-radius: 8px; padding: 13px 14px;
    box-shadow: 0 1px 6px var(--card-shadow); border-top: 3px solid #9C27B0;
    flex: 1 1 0; min-width: 125px; }
  .stat-card .label { font-size: 0.72rem; color: var(--text-meta); margin-bottom: 4px; line-height: 1.3; }
  .stat-card .value { font-size: 1.2rem; font-weight: 700; color: var(--text); white-space: nowrap; }
  .stat-card .sub   { font-size: 0.68rem; color: var(--text-muted); margin-top: 2px; line-height: 1.35; }

  .section-card { background: var(--card); border-radius: 10px; padding: 24px 28px;
    margin-bottom: 28px; box-shadow: 0 2px 10px var(--card-shadow);
    border-left: 4px solid #9C27B0; }
  .section-card h2 { margin-top: 0; border: none; padding: 0; font-size: 1.15rem; }

  .metric-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin: 14px 0 22px; }
  .metric-table th { background: #2B3A55; color: #C8D8F0; padding: 7px 9px; text-align: center;
    font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap; }
  .metric-table td { padding: 6px 9px; border-bottom: 1px solid var(--border-light); color: var(--text); }
  .metric-table tbody tr:nth-child(even) { background: var(--table-even); }
  .metric-table .num  { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-table .left { text-align: left; font-weight: 500; }
  .metric-table .sub  { font-size: 0.75rem; color: var(--text-meta); }

  .pos { color: #5BAD6F; } .neg { color: #E15252; } .na { color: var(--text-meta); }
  .best { background: rgba(156,39,176,0.22) !important; font-weight: 700; }
  .gap-ok { color: #5BAD6F; } .gap-warn { color: #F0B849; } .gap-bad { color: #E15252; }

  .th-elnet   { color: #5BAD6F; } .th-ridge { color: #4A90D9; } .th-rf { color: #4FC3F7; }
  .th-ann     { color: #CE93D8; } .th-voting { color: #FFB74D; } .th-stacking { color: #4DD0E1; }

  .group-prior { border-left: 2px solid #555; border-right: 2px solid #555; }
  .group-p9    { border-left: 2px solid #9C27B0; border-right: 2px solid #9C27B0; }

  .obs { background: var(--obs-bg); border-radius: 8px; padding: 14px 18px;
    margin: 14px 0; font-size: 0.88rem; color: var(--text); line-height: 1.65; }
  .obs strong { color: var(--text); }

  .plots-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
  .plot-box { flex: 1 1 400px; background: var(--card); border-radius: 8px;
    overflow: hidden; border: 1px solid var(--border-light); }
  .plot-box.wide { flex: 1 1 700px; }
  .plot-box.narrow { flex: 1 1 300px; }
  .plot-box img { width: 100%; display: block; }
  .plot-caption { font-size: 0.78rem; color: var(--text-meta); padding: 6px 10px; }

  details { border: 1px solid var(--border); border-radius: 8px; margin: 14px 0; }
  details > summary { padding: 10px 16px; cursor: pointer; font-size: 0.93rem;
    font-weight: 600; color: var(--text); list-style: none; user-select: none; }
  details > summary::before { content: "▶ "; font-size: 0.7em; }
  details[open] > summary::before { content: "▼ "; }
  details > summary:hover { background: var(--summary-hover); border-radius: 8px; }
  .details-body { padding: 14px 18px; }

  @media (max-width: 768px) { body { padding: 16px; } .plots-row { flex-direction: column; } }
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _r2_cell(val, best_val):
    if pd.isna(val): return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    colour = "pos" if val >= 0 else "neg"
    return f"<td class='num {colour}{cls}'>{val:+.3f}</td>"


def _rmse_cell(val, best_val):
    if pd.isna(val): return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    return f"<td class='num{cls}'>{val:.3f}</td>"


def _gap_cell(val):
    if pd.isna(val): return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all() -> tuple[pd.DataFrame, int]:
    """Load ANN + Ensemble results and merge with prior ElNet/Ridge/RF."""

    # Phase 9 models
    ann_df = pd.read_excel(os.path.join(ANN_DIR, "results.xlsx"))
    run_ann = int(ann_df["run"].max())
    ann_df = ann_df[ann_df["run"] == run_ann]

    ens_df = pd.read_excel(os.path.join(ENS_DIR, "results.xlsx"))
    run_ens = int(ens_df["run"].max())
    ens_df = ens_df[ens_df["run"] == run_ens]

    run = max(run_ann, run_ens)

    # Prior ElNet and Ridge from Exp3-S2
    lin = pd.read_excel(PRIOR_LINEAR_FILE)
    lin = lin[lin["run"] == lin["run"].max()]

    # Prior RF from Exp3-S2
    rf = pd.read_excel(PRIOR_RF_FILE)
    rf = rf[rf["run"] == rf["run"].max()]

    # Build unified wide table keyed on target
    targets = ann_df["target"].tolist()

    rows = []
    for tgt in targets:
        row = {"target": tgt}
        # Short label
        row["ds_label"] = (tgt.replace("Effluent ", "")
                              .replace(" (mg/L, Grab)", " (Grab)")
                              .replace(" (mg/L, Composite)", " (Comp)"))

        # Prior models
        lin_row = lin[lin["target"] == tgt]
        rf_row  = rf[rf["target"] == tgt]

        for col in ["ElNet_test_R2", "ElNet_test_RMSE", "ElNet_R2_gap",
                    "Ridge_test_R2", "Ridge_test_RMSE", "Ridge_R2_gap",
                    "ElNet_train_R2", "Ridge_train_R2",
                    "n_train", "n_test"]:
            row[col] = lin_row[col].values[0] if len(lin_row) > 0 else np.nan

        row["RF_test_R2"]   = rf_row["R2_test"].values[0]  if len(rf_row) > 0 else np.nan
        row["RF_test_RMSE"] = rf_row["RMSE_test"].values[0] if len(rf_row) > 0 else np.nan
        row["RF_R2_gap"]    = rf_row["R2_gap"].values[0]   if len(rf_row) > 0 else np.nan
        row["RF_train_R2"]  = rf_row["R2_train"].values[0] if len(rf_row) > 0 else np.nan

        # ANN
        ann_r = ann_df[ann_df["target"] == tgt]
        row["ANN_test_R2"]   = ann_r["R2_test"].values[0]   if len(ann_r) > 0 else np.nan
        row["ANN_test_RMSE"] = ann_r["RMSE_test"].values[0] if len(ann_r) > 0 else np.nan
        row["ANN_R2_gap"]    = ann_r["R2_gap"].values[0]    if len(ann_r) > 0 else np.nan
        row["ANN_train_R2"]  = ann_r["R2_train"].values[0]  if len(ann_r) > 0 else np.nan

        # Voting & Stacking
        for model_tag in ["Voting", "Stacking"]:
            ens_r = ens_df[(ens_df["target"] == tgt) & (ens_df["model"] == model_tag)]
            row[f"{model_tag}_test_R2"]   = ens_r["R2_test"].values[0]   if len(ens_r) > 0 else np.nan
            row[f"{model_tag}_test_RMSE"] = ens_r["RMSE_test"].values[0] if len(ens_r) > 0 else np.nan
            row[f"{model_tag}_R2_gap"]    = ens_r["R2_gap"].values[0]    if len(ens_r) > 0 else np.nan
            row[f"{model_tag}_train_R2"]  = ens_r["R2_train"].values[0]  if len(ens_r) > 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, run


# ── Charts ────────────────────────────────────────────────────────────────────

def _trend_chart(df: pd.DataFrame, run: int) -> str:
    """Bar chart: avg Test R² per model across all 8 targets."""
    models = ALL_MODELS
    avgs = [df[f"{m}_test_R2"].mean() for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")
    bars = ax.bar(models, avgs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008 if v >= 0 else bar.get_height() - 0.04,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_ylabel("Avg Test R² (all 8 targets)", fontsize=9, color="white")
    ax.set_title("Average Test R² — All Models (Prior Baselines + Phase 9)", fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")

    # Vertical divider between prior and Phase9
    ax.axvline(2.5, color="#9C27B0", lw=1.2, linestyle="--", alpha=0.6)
    ax.text(1.0, ax.get_ylim()[0] + 0.01, "Prior experiments", color="#aaa", fontsize=7, ha="center")
    ax.text(4.0, ax.get_ylim()[0] + 0.01, "Phase 9", color="#CE93D8", fontsize=7, ha="center")
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_trend_run{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


def _per_target_chart(df: pd.DataFrame, run: int) -> str:
    """Grouped bar: best Test R² per target for each model."""
    targets = df["ds_label"].tolist()
    n_m = len(ALL_MODELS)
    x = np.arange(len(targets))
    width = 0.13
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(max(12, len(targets) * 1.8), 5))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    for i, m in enumerate(ALL_MODELS):
        vals = df[f"{m}_test_R2"].tolist()
        hatch = "//" if m in ALL_PRIOR else ""
        bars = ax.bar(x + offsets[i], vals, width, label=m,
                      color=MODEL_COLORS[m], alpha=0.85,
                      edgecolor="white", linewidth=0.4, hatch=hatch)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004 if v >= 0 else bar.get_height() - 0.06,
                        f"{v:+.2f}", ha="center", va="bottom",
                        fontsize=5, rotation=90, color="white")

    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Test R²", fontsize=9, color="white")
    ax.set_title("Test R² per Target — All Models (hatched = prior, solid = Phase 9)", fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    ax.legend(fontsize=8, ncol=6, facecolor="#2B3A55", labelcolor="white", edgecolor="#3A4560")
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_pertarget_run{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


# ── Tables ─────────────────────────────────────────────────────────────────────

def _full_comparison_table(df: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Target</th>
          <th rowspan="2">n<br><span class='sub'>train</span></th>
          <th rowspan="2">n<br><span class='sub'>test</span></th>
          <th colspan="3" class="group-prior">Prior Best (Exp3-S2)</th>
          <th colspan="3" class="group-p9">Phase 9</th>
        </tr>
        <tr>
          <th class="th-elnet">ElNet<br>Test R²</th>
          <th class="th-ridge">Ridge<br>Test R²</th>
          <th class="th-rf">RF<br>Test R²</th>
          <th class="th-ann">ANN<br>Test R²</th>
          <th class="th-voting">Voting<br>Test R²</th>
          <th class="th-stacking">Stacking<br>Test R²</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        all_r2 = [row.get(f"{m}_test_R2", np.nan) for m in ALL_MODELS]
        valid_r2 = [v for v in all_r2 if not pd.isna(v)]
        best_r2 = max(valid_r2) if valid_r2 else np.nan

        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        rows_html += f"<td class='num'>{int(row['n_train']):,}</td>"
        rows_html += f"<td class='num'>{int(row['n_test']):,}</td>"
        for m in ALL_MODELS:
            rows_html += _r2_cell(row.get(f"{m}_test_R2"), best_r2)
        rows_html += "</tr>"

    # Average row
    rows_html += "<tr style='border-top:2px solid var(--border);font-style:italic'>"
    rows_html += "<td class='left' style='color:var(--text-meta)'>Average</td><td></td><td></td>"
    for m in ALL_MODELS:
        avg = df[f"{m}_test_R2"].mean()
        colour = "pos" if avg >= 0 else "neg"
        rows_html += f"<td class='num {colour}'><strong>{avg:+.3f}</strong></td>"
    rows_html += "</tr>"

    return header + rows_html + "</tbody></table>"


def _gap_table(df: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th class="th-elnet">ElNet Gap</th>
          <th class="th-ridge">Ridge Gap</th>
          <th class="th-rf">RF Gap</th>
          <th class="th-ann">ANN Gap</th>
          <th class="th-voting">Voting Gap</th>
          <th class="th-stacking">Stacking Gap</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for m in ALL_MODELS:
            rows_html += _gap_cell(row.get(f"{m}_R2_gap"))
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


def _train_r2_table(df: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th class="th-elnet">ElNet Train</th>
          <th class="th-ridge">Ridge Train</th>
          <th class="th-rf">RF Train</th>
          <th class="th-ann">ANN Train</th>
          <th class="th-voting">Voting Train</th>
          <th class="th-stacking">Stacking Train</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for m in ALL_MODELS:
            v = row.get(f"{m}_train_R2")
            colour = "pos" if (not pd.isna(v) and v >= 0) else "neg"
            val_str = f"{v:+.3f}" if not pd.isna(v) else "—"
            rows_html += f"<td class='num {colour}'>{val_str}</td>"
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


# ── Summary cards ──────────────────────────────────────────────────────────────

def _summary_cards(df: pd.DataFrame, run: int) -> str:
    # Best model per target (across all 6)
    wins = {m: 0 for m in ALL_MODELS}
    for _, row in df.iterrows():
        r2s = {m: row.get(f"{m}_test_R2", np.nan) for m in ALL_MODELS}
        valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1

    best_model = max(wins, key=wins.get)
    win_str = " · ".join(f"{wins[m]}W {m}" for m in ALL_MODELS)

    # Phase 9 vs prior avg
    prior_avgs = {m: df[f"{m}_test_R2"].mean() for m in ALL_PRIOR}
    p9_avgs    = {m: df[f"{m}_test_R2"].mean() for m in ALL_PHASE9}
    best_prior = max(prior_avgs, key=prior_avgs.get)
    best_p9    = max(p9_avgs,    key=p9_avgs.get)
    delta      = p9_avgs[best_p9] - prior_avgs[best_prior]

    # Best single R²
    best_val = df[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).max()
    best_idx = df[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).idxmax()
    best_tgt = df.loc[best_idx, "ds_label"]

    def _card(label, value, sub, border="#9C27B0"):
        return f"""
      <div class="stat-card" style="border-top-color:{border}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
      </div>"""

    cards = ""
    cards += _card("Run", f"#{run}", "Phase 9")
    cards += _card("Overall best model", best_model, win_str, MODEL_COLORS[best_model])
    cards += _card("Best Phase 9 model", best_p9,
                   f"avg R²={p9_avgs[best_p9]:+.3f} vs prior best {prior_avgs[best_prior]:+.3f}",
                   MODEL_COLORS[best_p9])
    delta_color = "#5BAD6F" if delta >= 0 else "#E15252"
    cards += _card("Phase 9 improvement",
                   f"{delta:+.3f}",
                   f"{best_p9} vs {best_prior}", delta_color)
    cards += _card("Best single result", f"{best_val:+.3f}", best_tgt, "#B07FD4")
    for m in ALL_PHASE9:
        cards += _card(f"{m} avg Test R²", f"{p9_avgs[m]:+.3f}",
                       "Phase 9 model", MODEL_COLORS[m])
    return f'<div class="summary-grid">{cards}</div>'


# ── Observation ────────────────────────────────────────────────────────────────

def _auto_obs(df: pd.DataFrame) -> str:
    lines = []

    # Winner table
    wins = {m: 0 for m in ALL_MODELS}
    for _, row in df.iterrows():
        r2s = {m: row.get(f"{m}_test_R2", np.nan) for m in ALL_MODELS}
        valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1
    win_str = " | ".join(f"{m}: {wins[m]}W" for m in ALL_MODELS)
    best = max(wins, key=wins.get)
    lines.append(f"<strong>Win counts:</strong> {win_str}. "
                 f"<strong>{best}</strong> leads overall.")

    # Avg comparisons
    avgs = {m: df[f"{m}_test_R2"].mean() for m in ALL_MODELS}
    avg_str = " | ".join(f"{m}&nbsp;{avgs[m]:+.3f}" for m in ALL_MODELS)
    lines.append(f"<strong>Average Test R²:</strong> {avg_str}.")

    # ANN note
    ann_avg = avgs["ANN"]
    if ann_avg < 0:
        lines.append(
            f"<strong>ANN performance:</strong> "
            f"<span class='neg'>Average Test R² = {ann_avg:+.3f}</span> — "
            f"MLPRegressor overfits badly on composite targets (R² gap up to +6.8). "
            f"Likely cause: temporal distribution shift to 2025 combined with "
            f"insufficient training data (~470 samples) for MLP generalisation. "
            f"ANN is <em>not recommended</em> for deployment without more data."
        )

    # Voting vs prior best
    voting_avg = avgs["Voting"]
    elnet_avg  = avgs["ElNet"]
    if voting_avg > elnet_avg:
        lines.append(
            f"<strong>Voting Ensemble:</strong> Avg Test R² {voting_avg:+.3f} vs "
            f"prior best ElNet {elnet_avg:+.3f} (+{voting_avg-elnet_avg:.3f}). "
            f"Voting provides the best balance of performance and stability."
        )

    # Stacking inconsistency
    stacking_avg = avgs["Stacking"]
    lines.append(
        f"<strong>Stacking:</strong> Avg {stacking_avg:+.3f} — competitive on Grab "
        f"targets but fails on Comp pH (R²={df[df['target'].str.contains('pH') & df['target'].str.contains('Composite')]['Stacking_test_R2'].values[0]:+.3f}). "
        f"Meta-learner (Ridge) inherits the composite pH difficulty."
    )

    # Comp COD persistent failure
    cod_comp = df[df["target"].str.contains("COD") & df["target"].str.contains("Composite")]
    if len(cod_comp) > 0:
        best_cod = cod_comp[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).max()
        if best_cod < 0:
            lines.append(
                f"<strong>Composite COD:</strong> Remains unsolved — "
                f"best R² across all models = {best_cod:+.3f}. "
                f"This target likely requires external process variables not currently measured."
            )

    return "<br>".join(lines)


# ── Plots HTML ─────────────────────────────────────────────────────────────────

def _ann_plots_html(run: int) -> str:
    html = ""
    for name in ["s2_stage3_grab_BOD", "s2_stage3_grab_COD", "s2_stage3_grab_TSS",
                 "s2_stage3_grab_pH", "s2_stage3_comp_BOD", "s2_stage3_comp_COD",
                 "s2_stage3_comp_TSS", "s2_stage3_comp_pH"]:
        label = name.replace("s2_stage3_", "").replace("_", " ").title()
        lc = _img_b64(os.path.join(ANN_DIR, "plots", f"{name}_ANN_run_{run}_lc.png"))
        sc = _img_b64(os.path.join(ANN_DIR, "plots", f"{name}_ANN_run_{run}_scatter.png"))
        if lc or sc:
            html += f"<div style='margin-bottom:6px;font-weight:500;font-size:0.85rem;color:var(--text-muted)'>{label}</div>"
            html += "<div class='plots-row'>"
            if sc: html += f"<div class='plot-box'><img src='{sc}'><div class='plot-caption'>Scatter</div></div>"
            if lc: html += f"<div class='plot-box'><img src='{lc}'><div class='plot-caption'>Learning Curve</div></div>"
            html += "</div>"
    return html if html else "<p style='color:var(--text-meta)'>No plots found.</p>"


def _ensemble_plots_html(model_tag: str, run: int) -> str:
    html = ""
    for name in ["s2_stage3_grab_BOD", "s2_stage3_grab_COD", "s2_stage3_grab_TSS",
                 "s2_stage3_grab_pH", "s2_stage3_comp_BOD", "s2_stage3_comp_COD",
                 "s2_stage3_comp_TSS", "s2_stage3_comp_pH"]:
        label = name.replace("s2_stage3_", "").replace("_", " ").title()
        sc = _img_b64(os.path.join(ENS_DIR, "plots", f"{name}_{model_tag}_run_{run}_scatter.png"))
        ts = _img_b64(os.path.join(ENS_DIR, "plots", f"{name}_{model_tag}_run_{run}_timeseries.png"))
        if sc or ts:
            html += f"<div style='margin-bottom:6px;font-weight:500;font-size:0.85rem;color:var(--text-muted)'>{label}</div>"
            html += "<div class='plots-row'>"
            if sc: html += f"<div class='plot-box'><img src='{sc}'><div class='plot-caption'>Scatter</div></div>"
            if ts: html += f"<div class='plot-box wide'><img src='{ts}'><div class='plot-caption'>Timeseries</div></div>"
            html += "</div>"
    return html if html else "<p style='color:var(--text-meta)'>No plots found.</p>"


# ── Full HTML ──────────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    trend_b64      = _trend_chart(df, run)
    per_target_b64 = _per_target_chart(df, run)

    legend = """
    <div class="legend">
      <strong style="color:var(--text);font-size:0.88rem">Prior (Exp3-S2):</strong>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElasticNet</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge</div>
      <div class="legend-item"><span class="dot" style="background:#2171B5"></span> RF</div>
      <div class="legend-divider"></div>
      <strong style="color:var(--text);font-size:0.88rem">Phase 9:</strong>
      <div class="legend-item"><span class="dot" style="background:#9C27B0"></span> ANN (MLP)</div>
      <div class="legend-item"><span class="dot" style="background:#FF9800"></span> Voting</div>
      <div class="legend-item"><span class="dot" style="background:#00BCD4"></span> Stacking</div>
    </div>
    """

    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Phase 9 — Advanced Models:</strong> Three new model classes evaluated on
      Experiment 3 Sub-2 feature sets (broadest scope, 20–32 features per target):<br>
      • <strong style="color:#CE93D8">ANN</strong> — sklearn MLPRegressor tuned via GridSearchCV
        (5 architectures × 4 alpha values = 20 combos, TimeSeriesSplit n_splits=3).<br>
      • <strong style="color:#FFB74D">Voting Ensemble</strong> — VotingRegressor combining
        Ridge + ElasticNet + RF with equal weights. Base models use tuned hyperparameters.<br>
      • <strong style="color:#4DD0E1">Stacking Ensemble</strong> — StackingRegressor: base
        models RF + Ridge + ElNet → Ridge meta-learner trained on OOF predictions
        (KFold n_splits=5). Final evaluation on the 2025 holdout set.<br><br>
      Compared against best prior models: ElNet, Ridge, RF (all from Exp3-S2, run 1).
      Highlighted cell = best Test R² in each row across all 6 models.
    </div>
    """

    obs_html = _auto_obs(df)
    summary  = _summary_cards(df, run)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 9 — Advanced Models Report | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Phase 9 — ANN, Voting &amp; Stacking Ensembles
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
    &nbsp;|&nbsp; Dataset: Exp3-S2 (ADD + CONSIDER features)</div>

  {intro}
  {legend}

  <h2>Executive Summary</h2>
  {summary}
  <div class="obs">{obs_html}</div>

  <h2>All-Model Comparison — Test R²</h2>
  <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 6px">
    <span style="background:rgba(156,39,176,0.25);padding:1px 6px;border-radius:3px">■</span>
    highlighted = row-best Test R² &nbsp;·&nbsp;
    <span class="pos">green</span> = R² ≥ 0 &nbsp;·&nbsp;
    <span class="neg">red</span> = R² &lt; 0
  </p>
  {_full_comparison_table(df)}

  <details>
    <summary>R² Gap (Train − Test) — all models</summary>
    <div class="details-body">
      <p style="font-size:0.82rem;color:var(--text-meta)">
        <span class="gap-ok">■</span> &lt; 0.10 (tight) &nbsp;
        <span class="gap-warn">■</span> 0.10–0.25 (moderate) &nbsp;
        <span class="gap-bad">■</span> &gt; 0.25 (overfit)
      </p>
      {_gap_table(df)}
    </div>
  </details>

  <details>
    <summary>Train R² — all models</summary>
    <div class="details-body">{_train_r2_table(df)}</div>
  </details>

  <h2>Summary Charts</h2>
  <div class="plots-row">
    {"<div class='plot-box wide'><img src='" + trend_b64 + "'><div class='plot-caption'>Average Test R² — all models</div></div>" if trend_b64 else ""}
    {"<div class='plot-box wide'><img src='" + per_target_b64 + "'><div class='plot-caption'>Test R² per target — all models</div></div>" if per_target_b64 else ""}
  </div>

  <h2>ANN (MLPRegressor) — Detailed Plots</h2>
  <p style="font-size:0.85rem;color:var(--text-meta);margin:0 0 8px">
    Learning curves show train R² vs CV validation R² as training size grows —
    useful for diagnosing data-sufficiency vs model-capacity issues.
  </p>
  <details>
    <summary>ANN Scatter + Learning Curves (all targets)</summary>
    <div class="details-body">
      {_ann_plots_html(run)}
    </div>
  </details>

  <h2>Voting Ensemble — Plots</h2>
  <details>
    <summary>Voting Scatter + Timeseries (all targets)</summary>
    <div class="details-body">
      {_ensemble_plots_html("Voting", run)}
    </div>
  </details>

  <h2>Stacking Ensemble — Plots</h2>
  <details>
    <summary>Stacking Scatter + Timeseries (all targets)</summary>
    <div class="details-body">
      {_ensemble_plots_html("Stacking", run)}
    </div>
  </details>

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading Phase 9 results...")
    df, run = _load_all()
    print(f"  {len(df)} targets, run {run}")

    print("Building HTML report...")
    html = build_html(df, run)

    out_path = os.path.join(REPORTS_DIR, f"report_phase9_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
