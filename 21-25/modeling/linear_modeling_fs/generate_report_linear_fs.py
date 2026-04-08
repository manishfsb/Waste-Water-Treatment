"""
generate_report_linear_fs.py — HTML comparison report for feature-selected linear models.

Reads linear_modeling_fs/results.xlsx and the plots in linear_modeling_fs/plots/.
Produces linear_modeling_fs/report_linear_fs_run_N.html — a self-contained,
dark-mode-aware report with:
  - Executive summary per experiment
  - Metric tables (Train R², Test R², R² Gap, Train RMSE, Test RMSE, MAPE)
  - Best-model highlighting per row
  - Embedded bar charts and scatter plots
  - Hyperparameter table (Ridge alpha, ElNet alpha / l1_ratio)

Usage (from project root):
    .venv/bin/python3 21-25/modeling/linear_modeling/generate_report_linear.py
"""

import base64
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")

# ── Import shared theme ────────────────────────────────────────────────────────
sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Design tokens ─────────────────────────────────────────────────────────────
OLS_COLOR   = "#E15252"
RIDGE_COLOR = "#4A90D9"
ELNET_COLOR = "#5BAD6F"
BEST_BG     = "#1a3a1a"   # dark-mode "winner" highlight
BEST_BG_LIGHT = "#e8f5e9"

EXPERIMENT_LABELS = {
    "Exp1":     "Experiment 1 — Inlet Features Only",
    "Exp2-Sub1": "Experiment 2 Sub-1 — Secondary Features Only",
    "Exp2-Sub2": "Experiment 2 Sub-2 — Inlet + Secondary Features",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path: str) -> str:
    """Return a base64 data-URI for a PNG image, or empty string if missing."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _fmt(val, digits=3, pct=False):
    if pd.isna(val):
        return "—"
    if pct:
        return f"{val:.1f}%"
    return f"{val:+.{digits}f}" if digits == 3 else f"{val:.{digits}f}"


def _r2_cell(val, best_val):
    """Colour a Test R² cell; highlight the best."""
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    colour = "pos" if val >= 0 else "neg"
    return f"<td class='num {colour}{cls}'>{val:+.3f}</td>"


def _rmse_cell(val, best_val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    return f"<td class='num{cls}'>{val:.3f}</td>"


def _gap_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


def _mape_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "mape-ok" if val < 20 else ("mape-warn" if val < 40 else "mape-bad")
    return f"<td class='num {cls}'>{val:.1f}%</td>"


# ── CSS ────────────────────────────────────────────────────────────────────────

REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }

  body {
    font-family: 'Inter', Calibri, Arial, sans-serif;
    max-width: 1350px;
    margin: 0 auto;
    padding: 24px 32px 60px;
  }

  h1 { font-size: 1.7rem; margin-bottom: 4px; color: var(--text); }
  h2 { font-size: 1.22rem; margin: 36px 0 10px; color: var(--text); border-bottom: 2px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.05rem; margin: 22px 0 8px; color: var(--text-muted); }
  p, li { color: var(--text-muted); font-size: 0.93rem; line-height: 1.6; }

  .run-badge {
    display: inline-block;
    background: #3A7BD5;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin-left: 10px;
    vertical-align: middle;
  }
  .ts { color: var(--text-meta); font-size: 0.82rem; margin-top: 4px; }

  /* ── Legend ── */
  .legend {
    display: flex; gap: 20px; flex-wrap: wrap;
    margin: 12px 0 20px;
    font-size: 0.88rem; color: var(--text-muted);
  }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }

  /* ── Experiment section card ── */
  .exp-card {
    background: var(--card);
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 36px;
    box-shadow: 0 2px 10px var(--card-shadow);
    border-left: 4px solid #3A7BD5;
  }
  .exp-card h2 { margin-top: 0; border: none; padding: 0; font-size: 1.15rem; }

  /* ── Metric table ── */
  .metric-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    margin: 14px 0 22px;
  }
  .metric-table th {
    background: #2B3A55;
    color: #C8D8F0;
    padding: 8px 10px;
    text-align: center;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
  }
  .metric-table td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text);
  }
  .metric-table tbody tr:nth-child(even) { background: var(--table-even); }
  .metric-table .num { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-table .left { text-align: left; font-weight: 500; }
  .metric-table .sub  { font-size: 0.78rem; color: var(--text-meta); }

  /* R² colours */
  .pos  { color: #5BAD6F; }
  .neg  { color: #E15252; }
  .na   { color: var(--text-meta); }

  /* Best-model highlight */
  .best { background: rgba(74,144,217,0.18) !important; font-weight: 700; }

  /* R² Gap colours */
  .gap-ok   { color: #5BAD6F; }
  .gap-warn { color: #F0B849; }
  .gap-bad  { color: #E15252; }

  /* MAPE colours */
  .mape-ok   { color: #5BAD6F; }
  .mape-warn { color: #F0B849; }
  .mape-bad  { color: #E15252; }

  /* ── Model header spans ── */
  .th-ols   { color: #E15252; }
  .th-ridge { color: #4A90D9; }
  .th-elnet { color: #5BAD6F; }

  /* ── Section divider ── */
  .section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-meta);
    padding: 6px 10px 2px;
    border-bottom: 1px solid var(--border-light);
  }

  /* ── Hyperparameter table ── */
  .hp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    margin: 10px 0 20px;
  }
  .hp-table th {
    background: var(--details-bg);
    color: var(--text-muted);
    padding: 7px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .hp-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-light);
    color: var(--text);
    font-size: 0.82rem;
  }

  /* ── Plot containers ── */
  .plots-row {
    display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;
  }
  .plot-box {
    flex: 1 1 400px;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-light);
  }
  .plot-box img { width: 100%; display: block; }
  .plot-caption { font-size: 0.78rem; color: var(--text-meta); padding: 6px 10px; }

  /* ── Obs callout ── */
  .obs {
    background: var(--obs-bg);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    color: var(--text);
    line-height: 1.65;
  }
  .obs strong { color: var(--text); }

  /* ── Summary section ── */
  .summary-grid {
    display: flex;
    flex-wrap: nowrap;
    gap: 10px;
    margin: 14px 0 24px;
    overflow-x: auto;
  }
  .stat-card {
    background: var(--card);
    border-radius: 8px;
    padding: 13px 14px;
    box-shadow: 0 1px 6px var(--card-shadow);
    border-top: 3px solid #3A7BD5;
    flex: 1 1 0;
    min-width: 130px;
  }
  .stat-card .label { font-size: 0.72rem; color: var(--text-meta); margin-bottom: 4px; line-height: 1.3; }
  .stat-card .value { font-size: 1.25rem; font-weight: 700; color: var(--text); white-space: nowrap; }
  .stat-card .sub   { font-size: 0.70rem; color: var(--text-muted); margin-top: 2px; line-height: 1.35; }

  details {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 14px 0;
  }
  details > summary {
    padding: 10px 16px;
    cursor: pointer;
    font-size: 0.93rem;
    font-weight: 600;
    color: var(--text);
    list-style: none;
    user-select: none;
  }
  details > summary::before { content: "▶ "; font-size: 0.7em; }
  details[open] > summary::before { content: "▼ "; }
  details > summary:hover { background: var(--summary-hover); border-radius: 8px; }
  .details-body { padding: 14px 18px; }

  @media (max-width: 768px) {
    body { padding: 16px; }
    .plots-row { flex-direction: column; }
  }
""")


# ── Section builders ───────────────────────────────────────────────────────────

def _metric_table_html(sub: pd.DataFrame) -> str:
    """Build the main metrics table for one experiment."""

    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Dataset</th>
          <th rowspan="2">n<br><span class='sub'>train</span></th>
          <th rowspan="2">n<br><span class='sub'>test</span></th>
          <th colspan="3" class="th-ols">OLS</th>
          <th colspan="3" class="th-ridge">Ridge</th>
          <th colspan="3" class="th-elnet">ElasticNet</th>
        </tr>
        <tr>
          <th class="th-ols">Test R²</th>
          <th class="th-ols">Test RMSE</th>
          <th class="th-ols">R² Gap</th>
          <th class="th-ridge">Test R²</th>
          <th class="th-ridge">Test RMSE</th>
          <th class="th-ridge">R² Gap</th>
          <th class="th-elnet">Test R²</th>
          <th class="th-elnet">Test RMSE</th>
          <th class="th-elnet">R² Gap</th>
        </tr>
      </thead>
      <tbody>
    """

    rows = ""
    for _, row in sub.iterrows():
        ds = row["dataset"].replace("Exp1_", "").replace("Exp2S1_", "").replace("Exp2S2_", "")

        # best Test R² (highest)
        best_r2 = max(row.get("OLS_test_R2", -99),
                      row.get("Ridge_test_R2", -99),
                      row.get("ElNet_test_R2", -99))
        # best Test RMSE (lowest)
        best_rmse = min(v for v in [row.get("OLS_test_RMSE"), row.get("Ridge_test_RMSE"), row.get("ElNet_test_RMSE")] if pd.notna(v))

        rows += f"<tr>"
        rows += f"<td class='left'>{ds}</td>"
        rows += f"<td class='num'>{row['n_train']:,}</td>"
        rows += f"<td class='num'>{row['n_test']:,}</td>"
        # OLS
        rows += _r2_cell(row.get("OLS_test_R2"), best_r2)
        rows += _rmse_cell(row.get("OLS_test_RMSE"), best_rmse)
        rows += _gap_cell(row.get("OLS_R2_gap"))
        # Ridge
        rows += _r2_cell(row.get("Ridge_test_R2"), best_r2)
        rows += _rmse_cell(row.get("Ridge_test_RMSE"), best_rmse)
        rows += _gap_cell(row.get("Ridge_R2_gap"))
        # ElNet
        rows += _r2_cell(row.get("ElNet_test_R2"), best_r2)
        rows += _rmse_cell(row.get("ElNet_test_RMSE"), best_rmse)
        rows += _gap_cell(row.get("ElNet_R2_gap"))
        rows += "</tr>"

    return header + rows + "</tbody></table>"


def _mape_table_html(sub: pd.DataFrame) -> str:
    """Secondary table with MAPE and train R² columns."""
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Dataset</th>
          <th class="th-ols">OLS Train R²</th>
          <th class="th-ols">OLS MAPE</th>
          <th class="th-ridge">Ridge Train R²</th>
          <th class="th-ridge">Ridge MAPE</th>
          <th class="th-ridge">Ridge α</th>
          <th class="th-elnet">ElNet Train R²</th>
          <th class="th-elnet">ElNet MAPE</th>
          <th class="th-elnet">ElNet α</th>
          <th class="th-elnet">ElNet l1</th>
        </tr>
      </thead>
      <tbody>
    """
    rows = ""
    for _, row in sub.iterrows():
        ds = row["dataset"].replace("Exp1_", "").replace("Exp2S1_", "").replace("Exp2S2_", "")
        rows += "<tr>"
        rows += f"<td class='left'>{ds}</td>"
        rows += f"<td class='num pos'>{_fmt(row.get('OLS_train_R2'))}</td>"
        rows += _mape_cell(row.get("OLS_test_MAPE"))
        rows += f"<td class='num pos'>{_fmt(row.get('Ridge_train_R2'))}</td>"
        rows += _mape_cell(row.get("Ridge_test_MAPE"))
        rows += f"<td class='num'>{row.get('Ridge_alpha', '—')}</td>"
        rows += f"<td class='num pos'>{_fmt(row.get('ElNet_train_R2'))}</td>"
        rows += _mape_cell(row.get("ElNet_test_MAPE"))
        rows += f"<td class='num'>{row.get('ElNet_alpha', '—')}</td>"
        rows += f"<td class='num'>{row.get('ElNet_l1_ratio', '—')}</td>"
        rows += "</tr>"
    return header + rows + "</tbody></table>"


def _auto_observation(sub: pd.DataFrame, exp: str) -> str:
    """Generate a plain-language observation paragraph for the experiment."""
    lines = []

    best_counts = {"OLS": 0, "Ridge": 0, "ElNet": 0}
    for _, row in sub.iterrows():
        r2s = {
            "OLS":   row.get("OLS_test_R2", -99),
            "Ridge": row.get("Ridge_test_R2", -99),
            "ElNet": row.get("ElNet_test_R2", -99),
        }
        winner = max(r2s, key=r2s.get)
        best_counts[winner] += 1

    dominant = max(best_counts, key=best_counts.get)
    lines.append(
        f"<strong>Best model:</strong> {dominant} wins on Test R² in "
        f"{best_counts[dominant]}/{len(sub)} datasets in this experiment."
    )

    avg_r2 = {
        "OLS":   sub["OLS_test_R2"].mean(),
        "Ridge": sub["Ridge_test_R2"].mean(),
        "ElNet": sub["ElNet_test_R2"].mean(),
    }
    lines.append(
        f"<strong>Average Test R²:</strong> "
        f"OLS&nbsp;{avg_r2['OLS']:+.3f} | "
        f"Ridge&nbsp;{avg_r2['Ridge']:+.3f} | "
        f"ElNet&nbsp;{avg_r2['ElNet']:+.3f}."
    )

    avg_gap = {
        "OLS":   sub["OLS_R2_gap"].mean(),
        "Ridge": sub["Ridge_R2_gap"].mean(),
        "ElNet": sub["ElNet_R2_gap"].mean(),
    }
    gap_notes = []
    for m, g in avg_gap.items():
        if g > 0.25:
            gap_notes.append(f"<span class='gap-bad'>{m} shows high overfit (avg gap {g:+.2f})</span>")
        elif g > 0.10:
            gap_notes.append(f"<span class='gap-warn'>{m} shows moderate overfit (avg gap {g:+.2f})</span>")
    if gap_notes:
        lines.append("<strong>Generalisation gaps:</strong> " + "; ".join(gap_notes) + ".")
    else:
        lines.append("<strong>Generalisation:</strong> All models show small train–test gaps — no obvious overfitting.")

    # pH note
    ph_rows = sub[sub["dataset"].str.endswith("pH")]
    if len(ph_rows) > 0:
        avg_ph_r2 = ph_rows[["OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]].max(axis=1).mean()
        if avg_ph_r2 < 0.10:
            lines.append(
                "<strong>pH targets:</strong> All linear models fail on pH (best avg Test R² "
                f"{avg_ph_r2:+.3f}). pH is buffered by the biology and has low variance — "
                "non-linear or domain-specific models would be needed."
            )

    return "<br>".join(lines)


def _exp_section_html(sub: pd.DataFrame, exp: str, run: int) -> str:
    label = EXPERIMENT_LABELS.get(exp, exp)
    n_feat = int(sub["n_features"].iloc[0]) if "n_features" in sub.columns else "?"

    obs_html = _auto_observation(sub, exp)

    # Plots (bar charts)
    r2_chart_path  = os.path.join(PLOTS_DIR, f"{exp}_r2_comparison_run_{run}.png")
    rmse_chart_path = os.path.join(PLOTS_DIR, f"{exp}_rmse_comparison_run_{run}.png")
    r2_b64   = _img_b64(r2_chart_path)
    rmse_b64 = _img_b64(rmse_chart_path)

    # Per-dataset scatter plots
    scatter_plots_html = ""
    for _, row in sub.iterrows():
        sc_path = os.path.join(PLOTS_DIR, f"{row['dataset']}_run_{run}_scatter.png")
        b64 = _img_b64(sc_path)
        if b64:
            ds_label = row["dataset"].replace("Exp1_","").replace("Exp2S1_","").replace("Exp2S2_","")
            scatter_plots_html += f"""
              <div class="plot-box">
                <img src="{b64}" alt="{ds_label} scatter">
                <div class="plot-caption">Actual vs Predicted — {ds_label}</div>
              </div>
            """

    charts_html = ""
    if r2_b64:
        charts_html += f"""
          <div class="plot-box">
            <img src="{r2_b64}" alt="R² comparison {exp}">
            <div class="plot-caption">Test R² — all models & targets</div>
          </div>
        """
    if rmse_b64:
        charts_html += f"""
          <div class="plot-box">
            <img src="{rmse_b64}" alt="RMSE comparison {exp}">
            <div class="plot-caption">Test RMSE — all models & targets</div>
          </div>
        """

    return f"""
    <div class="exp-card" id="{exp.replace(' ','-')}">
      <h2>{label} <span class="run-badge">run {run}</span></h2>
      <p style="margin:0 0 12px; font-size:0.85rem; color:var(--text-meta);">
        Feature count: <strong>{n_feat}</strong> &nbsp;|&nbsp;
        Datasets: <strong>{len(sub)}</strong> &nbsp;|&nbsp;
        Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
      </p>

      <div class="obs">{obs_html}</div>

      <h3>Primary Metrics — Test Set</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 4px;">
        <span class="gap-ok">■</span> gap &lt; 0.10 &nbsp;
        <span class="gap-warn">■</span> gap 0.10–0.25 (moderate) &nbsp;
        <span class="gap-bad">■</span> gap &gt; 0.25 (overfit) &nbsp;&nbsp;
        R² Gap = Train R² − Test R² (lower is better)
      </p>
      {_metric_table_html(sub)}

      <details>
        <summary>Train R², MAPE &amp; Hyperparameters</summary>
        <div class="details-body">
          {_mape_table_html(sub)}
        </div>
      </details>

      <h3>Summary Charts</h3>
      <div class="plots-row">{charts_html}</div>

      <details>
        <summary>Per-Dataset Actual vs Predicted (test set)</summary>
        <div class="details-body">
          <div class="plots-row">{scatter_plots_html}</div>
        </div>
      </details>
    </div>
    """


def _win_counts(subset: pd.DataFrame) -> dict:
    """Count how many datasets each model achieves the highest Test R²."""
    wins = {"OLS": 0, "Ridge": 0, "ElNet": 0}
    for _, row in subset.iterrows():
        r2s = {
            "OLS":   row.get("OLS_test_R2",   -9999),
            "Ridge": row.get("Ridge_test_R2", -9999),
            "ElNet": row.get("ElNet_test_R2", -9999),
        }
        wins[max(r2s, key=r2s.get)] += 1
    return wins


def _best_by_wins(subset: pd.DataFrame) -> tuple:
    """Return (winner_name, wins_dict) for the given subset."""
    wins = _win_counts(subset)
    return max(wins, key=wins.get), wins


def _top_summary_cards(df: pd.DataFrame) -> str:
    """Aggregate stat cards across all experiments — single scrollable row."""
    n_ds = len(df)

    # ── Average Test R² (shown per model, not used for selection) ──────────────
    avg_ols   = df["OLS_test_R2"].mean()
    avg_ridge = df["Ridge_test_R2"].mean()
    avg_elnet = df["ElNet_test_R2"].mean()

    # ── Overall winner: win-count across all 24 datasets ───────────────────────
    overall_best, overall_wins = _best_by_wins(df)
    overall_sub = f"{overall_wins['OLS']}W OLS · {overall_wins['Ridge']}W Ridge · {overall_wins['ElNet']}W ElNet"

    # ── Grab vs Composite best: win-count, filtered by target type ─────────────
    grab_df = df[df["dataset"].str.contains("Grab")]
    comp_df = df[df["dataset"].str.contains("Comp")]

    grab_best, grab_wins = _best_by_wins(grab_df)
    comp_best, comp_wins = _best_by_wins(comp_df)

    grab_sub = f"{grab_wins['OLS']}W OLS · {grab_wins['Ridge']}W Ridge · {grab_wins['ElNet']}W ElNet  ({len(grab_df)} datasets)"
    comp_sub = f"{comp_wins['OLS']}W OLS · {comp_wins['Ridge']}W Ridge · {comp_wins['ElNet']}W ElNet  ({len(comp_df)} datasets)"

    # ── Best single result ─────────────────────────────────────────────────────
    best_r2_val = df[["OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]].max(axis=1).max()
    best_r2_row = df.loc[df[["OLS_test_R2", "Ridge_test_R2", "ElNet_test_R2"]].max(axis=1).idxmax()]

    return f"""
    <div class="summary-grid">
      <div class="stat-card">
        <div class="label">Datasets evaluated</div>
        <div class="value">{n_ds}</div>
        <div class="sub">3 experiments × 8 targets</div>
      </div>
      <div class="stat-card" style="border-top-color:#F0B849">
        <div class="label">Best on Grab Effluents <span style="font-weight:400;opacity:0.7">(wins)</span></div>
        <div class="value">{grab_best}</div>
        <div class="sub">{grab_sub}</div>
      </div>
      <div class="stat-card" style="border-top-color:#B07FD4">
        <div class="label">Best on Composite Effluents <span style="font-weight:400;opacity:0.7">(wins)</span></div>
        <div class="value">{comp_best}</div>
        <div class="sub">{comp_sub}</div>
      </div>
      <div class="stat-card">
        <div class="label">Overall best model <span style="font-weight:400;opacity:0.7">(wins)</span></div>
        <div class="value">{overall_best}</div>
        <div class="sub">{overall_sub}</div>
      </div>
      <div class="stat-card" style="border-top-color:{OLS_COLOR}">
        <div class="label">OLS avg Test R²</div>
        <div class="value">{avg_ols:+.3f}</div>
        <div class="sub">Unregularised baseline</div>
      </div>
      <div class="stat-card" style="border-top-color:{RIDGE_COLOR}">
        <div class="label">Ridge avg Test R²</div>
        <div class="value">{avg_ridge:+.3f}</div>
        <div class="sub">L2 regularised, α tuned</div>
      </div>
      <div class="stat-card" style="border-top-color:{ELNET_COLOR}">
        <div class="label">ElasticNet avg Test R²</div>
        <div class="value">{avg_elnet:+.3f}</div>
        <div class="sub">L1+L2, α &amp; l1_ratio tuned</div>
      </div>
      <div class="stat-card">
        <div class="label">Best single result</div>
        <div class="value">{best_r2_val:+.3f}</div>
        <div class="sub">{best_r2_row['dataset']}</div>
      </div>
    </div>
    """


def _cross_exp_table(df: pd.DataFrame) -> str:
    """Compact cross-experiment average comparison table."""
    rows = ""
    for exp, grp in df.groupby("experiment", sort=False):
        label = EXPERIMENT_LABELS.get(exp, exp)
        rows += f"""
        <tr>
          <td class="left">{label}</td>
          <td class="num">{int(grp['n_features'].iloc[0])}</td>
          <td class="num">{grp['OLS_test_R2'].mean():+.3f}</td>
          <td class="num">{grp['Ridge_test_R2'].mean():+.3f}</td>
          <td class="num">{grp['ElNet_test_R2'].mean():+.3f}</td>
          <td class="num">{grp['OLS_test_RMSE'].mean():.3f}</td>
          <td class="num">{grp['Ridge_test_RMSE'].mean():.3f}</td>
          <td class="num">{grp['ElNet_test_RMSE'].mean():.3f}</td>
        </tr>"""

    return f"""
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Experiment</th>
          <th># features</th>
          <th class="th-ols">OLS<br>avg R²</th>
          <th class="th-ridge">Ridge<br>avg R²</th>
          <th class="th-elnet">ElNet<br>avg R²</th>
          <th class="th-ols">OLS<br>avg RMSE</th>
          <th class="th-ridge">Ridge<br>avg RMSE</th>
          <th class="th-elnet">ElNet<br>avg RMSE</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


# ── Full HTML assembly ─────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    legend = """
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background:#E15252"></span> OLS (unregularised)</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge (L2, tuned α)</div>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElasticNet (L1+L2, tuned)</div>
    </div>
    """

    exp_sections = ""
    for exp in df["experiment"].unique():
        sub = df[df["experiment"] == exp].copy()
        exp_sections += _exp_section_html(sub, exp, run)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Linear Models Report (Feature Selected) — Experiments 1 &amp; 2 | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Linear Models Comparison Report — Feature Selected
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025</div>

  <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);border-radius:8px;font-size:0.88rem;color:var(--text);">
    <strong>Purpose:</strong> Evaluate three linear models — OLS (unregularised baseline),
    Ridge (L2), and ElasticNet (L1+L2) — across Experiments 1 and 2. OLS is included
    deliberately to demonstrate where regularisation provides meaningful value and where
    collinearity or overfitting degradation becomes apparent. All feature sets use
    <code>StandardScaler</code>. Ridge α and ElasticNet α/l1_ratio are tuned via
    <code>GridSearchCV + TimeSeriesSplit(n_splits=3)</code>.<br>
    <strong style="color:var(--text)">Selection criterion for &ldquo;best model&rdquo; cards:</strong>
    <strong>Win count</strong> — the number of individual datasets on which a model achieves
    the highest Test R² (ties go to the first model that reaches the maximum). Average Test R²
    is shown separately but <em>not</em> used for selection, because OLS produces
    catastrophically negative R² values on some datasets (e.g. −13.5 for Comp TSS) that
    would unfairly distort its mean without reflecting its typical performance.
  </div>

  {legend}

  <h2>Cross-Experiment Summary</h2>
  {_top_summary_cards(df)}
  {_cross_exp_table(df)}

  {exp_sections}

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: results file not found — {RESULTS_FILE}")
        print("Run linear_modeling.py first.")
        sys.exit(1)

    df = pd.read_excel(RESULTS_FILE)

    # Use the latest run
    run = int(df["run"].max())
    df  = df[df["run"] == run].copy()

    print(f"Building report for run {run}  ({len(df)} datasets)...")

    html = build_html(df, run)

    out_path = os.path.join(SCRIPT_DIR, f"report_linear_fs_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
