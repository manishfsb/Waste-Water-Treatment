"""
generate_report_linear_exp3_s1_fs.py — HTML report for Exp3 Sub-1 Feature-Selected linear models.

Reads models/linear/exp3_s1_fs/results.xlsx and plots in models/linear/exp3_s1_fs/plots/.
Produces models/linear/exp3_s1_fs/report_linear_exp3_s1_fs_run_N.html — a self-contained,
dark-mode-aware report.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/linear/exp3_s1_fs/generate_report_linear_exp3_s1_fs.py
"""

import base64
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")

sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Design tokens ─────────────────────────────────────────────────────────────
OLS_COLOR   = "#E15252"
RIDGE_COLOR = "#4A90D9"
ELNET_COLOR = "#5BAD6F"

EXPERIMENT_LABELS = {
    "Exp3-S1-FS": "Experiment 3 Sub-1 — Feature Selected",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path: str) -> str:
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
  body { font-family:'Inter',Calibri,Arial,sans-serif;
         max-width:1350px; margin:0 auto; padding:24px 32px 60px; }
  h1 { font-size:1.7rem; margin-bottom:4px; color:var(--text); }
  h2 { font-size:1.22rem; margin:36px 0 10px; color:var(--text);
       border-bottom:2px solid var(--border); padding-bottom:6px; }
  h3 { font-size:1.05rem; margin:22px 0 8px; color:var(--text-muted); }
  p, li { color:var(--text-muted); font-size:0.93rem; line-height:1.6; }
  .run-badge { display:inline-block; background:#3A7BD5; color:white;
               padding:2px 10px; border-radius:12px; font-size:0.78rem;
               margin-left:10px; vertical-align:middle; }
  .ts { color:var(--text-meta); font-size:0.82rem; margin-top:4px; }
  .legend { display:flex; gap:20px; flex-wrap:wrap;
            margin:12px 0 20px; font-size:0.88rem; color:var(--text-muted); }
  .legend-item { display:flex; align-items:center; gap:6px; }
  .dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
  .exp-card { background:var(--card); border-radius:10px; padding:24px 28px;
              margin-bottom:36px; box-shadow:0 2px 10px var(--card-shadow);
              border-left:4px solid #3A7BD5; }
  .exp-card h2 { margin-top:0; border:none; padding:0; font-size:1.15rem; }
  .metric-table { width:100%; border-collapse:collapse;
                  font-size:0.86rem; margin:14px 0 22px; }
  .metric-table th { background:#2B3A55; color:#C8D8F0; padding:8px 10px;
                     text-align:center; font-weight:600;
                     border-bottom:2px solid var(--border); white-space:nowrap; }
  .metric-table td { padding:7px 10px; border-bottom:1px solid var(--border-light);
                     color:var(--text); }
  .metric-table tbody tr:nth-child(even) { background:var(--table-even); }
  .metric-table .num  { text-align:right; font-variant-numeric:tabular-nums; }
  .metric-table .left { text-align:left; font-weight:500; }
  .metric-table .sub  { font-size:0.78rem; color:var(--text-meta); }
  .pos { color:#5BAD6F; } .neg { color:#E15252; } .na { color:var(--text-meta); }
  .best { background:rgba(74,144,217,0.18) !important; font-weight:700; }
  .gap-ok   { color:#5BAD6F; } .gap-warn { color:#F0B849; } .gap-bad  { color:#E15252; }
  .mape-ok  { color:#5BAD6F; } .mape-warn{ color:#F0B849; } .mape-bad { color:#E15252; }
  .th-ols   { color:#E15252; } .th-ridge { color:#4A90D9; } .th-elnet { color:#5BAD6F; }
  .hp-table { width:100%; border-collapse:collapse; font-size:0.84rem; margin:10px 0 20px; }
  .hp-table th { background:var(--details-bg); color:var(--text-muted);
                 padding:7px 10px; text-align:left; border-bottom:1px solid var(--border); }
  .hp-table td { padding:6px 10px; border-bottom:1px solid var(--border-light);
                 color:var(--text); font-size:0.82rem; }
  .plots-row { display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }
  .plot-box { flex:1 1 400px; background:var(--card); border-radius:8px;
              overflow:hidden; border:1px solid var(--border-light); }
  .plot-box img { width:100%; display:block; }
  .plot-caption { font-size:0.78rem; color:var(--text-meta); padding:6px 10px; }
  .obs { background:var(--obs-bg); border-radius:8px; padding:14px 18px;
         margin:14px 0; font-size:0.88rem; color:var(--text); line-height:1.65; }
  .obs strong { color:var(--text); }
  .summary-grid { display:flex; flex-wrap:nowrap; gap:10px;
                  margin:14px 0 24px; overflow-x:auto; }
  .stat-card { background:var(--card); border-radius:8px; padding:13px 14px;
               box-shadow:0 1px 6px var(--card-shadow); border-top:3px solid #3A7BD5;
               flex:1 1 0; min-width:130px; }
  .stat-card .label { font-size:0.72rem; color:var(--text-meta); margin-bottom:4px; line-height:1.3; }
  .stat-card .value { font-size:1.25rem; font-weight:700; color:var(--text); white-space:nowrap; }
  .stat-card .sub   { font-size:0.70rem; color:var(--text-muted); margin-top:2px; line-height:1.35; }
  details { border:1px solid var(--border); border-radius:8px; margin:14px 0; }
  details > summary { padding:10px 16px; cursor:pointer; font-size:0.93rem;
                      font-weight:600; color:var(--text); list-style:none; user-select:none; }
  details > summary::before { content:"▶ "; font-size:0.7em; }
  details[open] > summary::before { content:"▼ "; }
  details > summary:hover { background:var(--summary-hover); border-radius:8px; }
  .details-body { padding:14px 18px; }
""")


# ── Section builders ───────────────────────────────────────────────────────────

def _strip_ds(name):
    return name.replace("Exp3S1FS_", "").replace("_", " ")


def _metric_table_html(sub: pd.DataFrame) -> str:
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
          <th class="th-ols">Test R²</th><th class="th-ols">Test RMSE</th><th class="th-ols">R² Gap</th>
          <th class="th-ridge">Test R²</th><th class="th-ridge">Test RMSE</th><th class="th-ridge">R² Gap</th>
          <th class="th-elnet">Test R²</th><th class="th-elnet">Test RMSE</th><th class="th-elnet">R² Gap</th>
        </tr>
      </thead>
      <tbody>
    """
    rows = ""
    for _, row in sub.iterrows():
        best_r2   = max(row.get("OLS_test_R2", -99), row.get("Ridge_test_R2", -99), row.get("ElNet_test_R2", -99))
        best_rmse = min(v for v in [row.get("OLS_test_RMSE"), row.get("Ridge_test_RMSE"), row.get("ElNet_test_RMSE")] if pd.notna(v))
        rows += f"<tr><td class='left'>{_strip_ds(row['dataset'])}</td>"
        rows += f"<td class='num'>{row['n_train']:,}</td><td class='num'>{row['n_test']:,}</td>"
        rows += _r2_cell(row.get("OLS_test_R2"), best_r2) + _rmse_cell(row.get("OLS_test_RMSE"), best_rmse) + _gap_cell(row.get("OLS_R2_gap"))
        rows += _r2_cell(row.get("Ridge_test_R2"), best_r2) + _rmse_cell(row.get("Ridge_test_RMSE"), best_rmse) + _gap_cell(row.get("Ridge_R2_gap"))
        rows += _r2_cell(row.get("ElNet_test_R2"), best_r2) + _rmse_cell(row.get("ElNet_test_RMSE"), best_rmse) + _gap_cell(row.get("ElNet_R2_gap"))
        rows += "</tr>"
    return header + rows + "</tbody></table>"


def _mape_table_html(sub: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead><tr>
        <th class="left">Dataset</th>
        <th class="th-ols">OLS Train R²</th><th class="th-ols">OLS MAPE</th>
        <th class="th-ridge">Ridge Train R²</th><th class="th-ridge">Ridge MAPE</th><th class="th-ridge">Ridge α</th>
        <th class="th-elnet">ElNet Train R²</th><th class="th-elnet">ElNet MAPE</th><th class="th-elnet">ElNet α</th><th class="th-elnet">ElNet l1</th>
      </tr></thead><tbody>
    """
    rows = ""
    for _, row in sub.iterrows():
        rows += f"<tr><td class='left'>{_strip_ds(row['dataset'])}</td>"
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


def _auto_observation(sub: pd.DataFrame) -> str:
    best_counts = {"OLS": 0, "Ridge": 0, "ElNet": 0}
    for _, row in sub.iterrows():
        r2s = {"OLS": row.get("OLS_test_R2", -99), "Ridge": row.get("Ridge_test_R2", -99), "ElNet": row.get("ElNet_test_R2", -99)}
        best_counts[max(r2s, key=r2s.get)] += 1
    dominant = max(best_counts, key=best_counts.get)
    lines = [
        f"<strong>Best model:</strong> {dominant} wins on Test R² in {best_counts[dominant]}/{len(sub)} datasets.",
        f"<strong>Average Test R²:</strong> OLS&nbsp;{sub['OLS_test_R2'].mean():+.3f} | "
        f"Ridge&nbsp;{sub['Ridge_test_R2'].mean():+.3f} | ElNet&nbsp;{sub['ElNet_test_R2'].mean():+.3f}.",
    ]
    avg_gap = {m: sub[f"{m}_R2_gap"].mean() for m in ["OLS", "Ridge", "ElNet"]}
    gap_notes = []
    for m, g in avg_gap.items():
        if g > 0.25:
            gap_notes.append(f"<span class='gap-bad'>{m} high overfit (avg gap {g:+.2f})</span>")
        elif g > 0.10:
            gap_notes.append(f"<span class='gap-warn'>{m} moderate overfit (avg gap {g:+.2f})</span>")
    if gap_notes:
        lines.append("<strong>Generalisation gaps:</strong> " + "; ".join(gap_notes) + ".")
    else:
        lines.append("<strong>Generalisation:</strong> All models show small train–test gaps.")
    return "<br>".join(lines)


def _exp_section_html(sub: pd.DataFrame, exp: str, run: int) -> str:
    label  = EXPERIMENT_LABELS.get(exp, exp)
    n_feat = int(sub["n_features"].iloc[0]) if "n_features" in sub.columns else "?"
    r2_b64   = _img_b64(os.path.join(PLOTS_DIR, f"{exp}_r2_comparison_run_{run}.png"))
    rmse_b64 = _img_b64(os.path.join(PLOTS_DIR, f"{exp}_rmse_comparison_run_{run}.png"))

    scatter_html = ""
    for _, row in sub.iterrows():
        b64 = _img_b64(os.path.join(PLOTS_DIR, f"{row['dataset']}_run_{run}_scatter.png"))
        if b64:
            scatter_html += f'<div class="plot-box"><img src="{b64}" alt="scatter"><div class="plot-caption">{_strip_ds(row["dataset"])}</div></div>'

    charts = ""
    if r2_b64:   charts += f'<div class="plot-box"><img src="{r2_b64}" alt="R²"><div class="plot-caption">Test R² all targets</div></div>'
    if rmse_b64: charts += f'<div class="plot-box"><img src="{rmse_b64}" alt="RMSE"><div class="plot-caption">Test RMSE all targets</div></div>'

    return f"""
    <div class="exp-card" id="{exp.replace(' ','-')}">
      <h2>{label} <span class="run-badge">run {run}</span></h2>
      <p style="margin:0 0 12px;font-size:0.85rem;color:var(--text-meta);">
        Feature count: <strong>{n_feat}</strong> &nbsp;|&nbsp;
        Datasets: <strong>{len(sub)}</strong> &nbsp;|&nbsp;
        Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
      </p>
      <div class="obs">{_auto_observation(sub)}</div>
      <h3>Primary Metrics — Test Set</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 4px;">
        <span class="gap-ok">■</span> gap &lt; 0.10 &nbsp;
        <span class="gap-warn">■</span> gap 0.10–0.25 &nbsp;
        <span class="gap-bad">■</span> gap &gt; 0.25 (overfit)
      </p>
      {_metric_table_html(sub)}
      <details>
        <summary>Train R², MAPE &amp; Hyperparameters</summary>
        <div class="details-body">{_mape_table_html(sub)}</div>
      </details>
      <h3>Summary Charts</h3>
      <div class="plots-row">{charts}</div>
      <details>
        <summary>Per-Dataset Actual vs Predicted (test set)</summary>
        <div class="details-body"><div class="plots-row">{scatter_html}</div></div>
      </details>
    </div>
    """


def _top_summary_cards(df: pd.DataFrame) -> str:
    def _wins(sub):
        w = {"OLS": 0, "Ridge": 0, "ElNet": 0}
        for _, r in sub.iterrows():
            r2s = {m: r.get(f"{m}_test_R2", -9999) for m in w}
            w[max(r2s, key=r2s.get)] += 1
        return w

    overall_wins = _wins(df)
    overall_best = max(overall_wins, key=overall_wins.get)
    grab_wins  = _wins(df[df["dataset"].str.contains("Grab")])
    comp_wins  = _wins(df[df["dataset"].str.contains("Comp")])
    grab_best  = max(grab_wins, key=grab_wins.get)
    comp_best  = max(comp_wins, key=comp_wins.get)
    avg_ols    = df["OLS_test_R2"].mean()
    avg_ridge  = df["Ridge_test_R2"].mean()
    avg_elnet  = df["ElNet_test_R2"].mean()
    best_r2_val = df[["OLS_test_R2","Ridge_test_R2","ElNet_test_R2"]].max(axis=1).max()
    best_r2_row = df.loc[df[["OLS_test_R2","Ridge_test_R2","ElNet_test_R2"]].max(axis=1).idxmax()]

    return f"""
    <div class="summary-grid">
      <div class="stat-card">
        <div class="label">Datasets evaluated</div>
        <div class="value">{len(df)}</div>
        <div class="sub">8 targets (Grab + Composite)</div>
      </div>
      <div class="stat-card" style="border-top-color:#F0B849">
        <div class="label">Best on Grab</div>
        <div class="value">{grab_best}</div>
        <div class="sub">{grab_wins['OLS']}W OLS · {grab_wins['Ridge']}W Ridge · {grab_wins['ElNet']}W ElNet</div>
      </div>
      <div class="stat-card" style="border-top-color:#B07FD4">
        <div class="label">Best on Composite</div>
        <div class="value">{comp_best}</div>
        <div class="sub">{comp_wins['OLS']}W OLS · {comp_wins['Ridge']}W Ridge · {comp_wins['ElNet']}W ElNet</div>
      </div>
      <div class="stat-card">
        <div class="label">Overall best model</div>
        <div class="value">{overall_best}</div>
        <div class="sub">{overall_wins['OLS']}W OLS · {overall_wins['Ridge']}W Ridge · {overall_wins['ElNet']}W ElNet</div>
      </div>
      <div class="stat-card" style="border-top-color:{OLS_COLOR}">
        <div class="label">OLS avg Test R²</div>
        <div class="value">{avg_ols:+.3f}</div><div class="sub">Unregularised baseline</div>
      </div>
      <div class="stat-card" style="border-top-color:{RIDGE_COLOR}">
        <div class="label">Ridge avg Test R²</div>
        <div class="value">{avg_ridge:+.3f}</div><div class="sub">L2 regularised</div>
      </div>
      <div class="stat-card" style="border-top-color:{ELNET_COLOR}">
        <div class="label">ElasticNet avg Test R²</div>
        <div class="value">{avg_elnet:+.3f}</div><div class="sub">L1+L2, tuned</div>
      </div>
      <div class="stat-card">
        <div class="label">Best single result</div>
        <div class="value">{best_r2_val:+.3f}</div>
        <div class="sub">{best_r2_row['dataset']}</div>
      </div>
    </div>
    """


def _cross_exp_table(df: pd.DataFrame) -> str:
    rows = ""
    for exp, grp in df.groupby("experiment", sort=False):
        label = EXPERIMENT_LABELS.get(exp, exp)
        rows += f"""<tr>
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
      <thead><tr>
        <th class="left">Experiment</th><th># features</th>
        <th class="th-ols">OLS avg R²</th><th class="th-ridge">Ridge avg R²</th><th class="th-elnet">ElNet avg R²</th>
        <th class="th-ols">OLS avg RMSE</th><th class="th-ridge">Ridge avg RMSE</th><th class="th-elnet">ElNet avg RMSE</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    legend = """
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background:#E15252"></span> OLS</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge (L2)</div>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElasticNet (L1+L2)</div>
    </div>"""

    exp_sections = ""
    for exp in df["experiment"].unique():
        exp_sections += _exp_section_html(df[df["experiment"] == exp].copy(), exp, run)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Linear Models — Exp3 Sub-1 (Feature Selected) | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
  <h1>Linear Models — Experiment 3 Sub-1 (Feature Selected)
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025</div>
  <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);border-radius:8px;font-size:0.88rem;color:var(--text)">
    <strong>Purpose:</strong> OLS, Ridge, and ElasticNet on Exp3 Sub-1 feature-selected subsets
    (Core + Useful features only, norm perm imp ≥ 0.03). Feature counts are now comparable to
    Exp2-FS (5–8 features vs 16–22 in the full Exp3-S1), recovering rows via fewer dropna constraints.
    Comparison with Exp3-S1 and Exp2-FS reveals whether the ADD-tier features earn their keep
    after selection.
  </div>
  {legend}
  <h2>Summary</h2>
  {_top_summary_cards(df)}
  {_cross_exp_table(df)}
  {exp_sections}
</body>
</html>"""


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run linear_modeling_exp3_s1_fs.py first.")
        sys.exit(1)

    df  = pd.read_excel(RESULTS_FILE)
    run = int(df["run"].max())
    df  = df[df["run"] == run].copy()

    print(f"Building linear FS report for run {run}  ({len(df)} datasets)...")
    html = build_html(df, run)

    out_path = os.path.join(SCRIPT_DIR, f"report_linear_exp3_s1_fs_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
