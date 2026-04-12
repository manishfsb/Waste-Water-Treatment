"""
generate_report_nonlinear_exp3_s2.py — HTML report for Experiment 3 Sub-2 tree-ensemble models.

Reads non_linear/exp3_s2/{rf,gb,xgb}/results.xlsx and the plots produced by
non_linear_modeling_exp3_s2.py.  Produces:
    non_linear/exp3_s2/report_nonlinear_exp3_s2_run_N.html

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/non_linear/exp3_s2/generate_report_nonlinear_exp3_s2.py
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
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

RF_DIR   = os.path.join(SCRIPT_DIR, "rf")
GB_DIR   = os.path.join(SCRIPT_DIR, "gb")
XGB_DIR  = os.path.join(SCRIPT_DIR, "xgb")

sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Design tokens ──────────────────────────────────────────────────────────────
RF_COLOR  = "#2171B5"
GB_COLOR  = "#238B45"
XGB_COLOR = "#D94801"
MODELS    = ["RF", "GB", "XGB"]

EXPERIMENT_LABELS = {
    "Experiment 3 Sub-2": "Experiment 3 Sub-2 — Exp2 Sub-2 Baseline + ADD + CONSIDER-tier Features",
}
EXPERIMENT_NFEAT = {
    "Experiment 3 Sub-2": "20–32",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }

  body {
    font-family: 'Inter', Calibri, Arial, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px 60px;
  }

  h1 { font-size: 1.7rem; margin-bottom: 4px; color: var(--text); }
  h2 { font-size: 1.22rem; margin: 36px 0 10px; color: var(--text);
       border-bottom: 2px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.05rem; margin: 22px 0 8px; color: var(--text-muted); }
  p, li { color: var(--text-muted); font-size: 0.93rem; line-height: 1.6; }

  .run-badge {
    display: inline-block; background: #3A7BD5; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.78rem;
    margin-left: 10px; vertical-align: middle;
  }
  .ts { color: var(--text-meta); font-size: 0.82rem; margin-top: 4px; }

  .legend { display: flex; gap: 20px; flex-wrap: wrap;
            margin: 12px 0 20px; font-size: 0.88rem; color: var(--text-muted); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }

  .summary-grid {
    display: flex; flex-wrap: nowrap; gap: 10px;
    margin: 14px 0 24px; overflow-x: auto;
  }
  .stat-card {
    background: var(--card); border-radius: 8px; padding: 13px 14px;
    box-shadow: 0 1px 6px var(--card-shadow); border-top: 3px solid #3A7BD5;
    flex: 1 1 0; min-width: 130px;
  }
  .stat-card .label { font-size: 0.72rem; color: var(--text-meta); margin-bottom: 4px; line-height: 1.3; }
  .stat-card .value { font-size: 1.25rem; font-weight: 700; color: var(--text); white-space: nowrap; }
  .stat-card .sub   { font-size: 0.70rem; color: var(--text-muted); margin-top: 2px; line-height: 1.35; }

  .exp-card {
    background: var(--card); border-radius: 10px; padding: 24px 28px;
    margin-bottom: 36px; box-shadow: 0 2px 10px var(--card-shadow);
    border-left: 4px solid #3A7BD5;
  }
  .exp-card h2 { margin-top: 0; border: none; padding: 0; font-size: 1.15rem; }

  .metric-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 14px 0 22px; }
  .metric-table th {
    background: #2B3A55; color: #C8D8F0; padding: 8px 10px; text-align: center;
    font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap;
  }
  .metric-table td { padding: 7px 10px; border-bottom: 1px solid var(--border-light); color: var(--text); }
  .metric-table tbody tr:nth-child(even) { background: var(--table-even); }
  .metric-table .num  { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-table .left { text-align: left; font-weight: 500; }
  .metric-table .sub  { font-size: 0.78rem; color: var(--text-meta); }

  .pos { color: #5BAD6F; }
  .neg { color: #E15252; }
  .na  { color: var(--text-meta); }

  .best { background: rgba(74,144,217,0.18) !important; font-weight: 700; }

  .gap-ok   { color: #5BAD6F; }
  .gap-warn { color: #F0B849; }
  .gap-bad  { color: #E15252; }

  .th-rf  { color: #4FC3F7; }
  .th-gb  { color: #81C784; }
  .th-xgb { color: #FF8A65; }

  .section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-meta);
    padding: 6px 10px 2px; border-bottom: 1px solid var(--border-light);
  }

  .obs {
    background: var(--obs-bg); border-radius: 8px; padding: 14px 18px;
    margin: 14px 0; font-size: 0.88rem; color: var(--text); line-height: 1.65;
  }
  .obs strong { color: var(--text); }

  .plots-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
  .plot-box {
    flex: 1 1 400px; background: var(--card); border-radius: 8px;
    overflow: hidden; border: 1px solid var(--border-light);
  }
  .plot-box.wide { flex: 1 1 700px; }
  .plot-box.narrow { flex: 1 1 300px; }
  .plot-box img { width: 100%; display: block; }
  .plot-caption { font-size: 0.78rem; color: var(--text-meta); padding: 6px 10px; }

  details {
    border: 1px solid var(--border); border-radius: 8px; margin: 14px 0;
  }
  details > summary {
    padding: 10px 16px; cursor: pointer; font-size: 0.93rem; font-weight: 600;
    color: var(--text); list-style: none; user-select: none;
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _fmt(val, digits=3):
    if pd.isna(val):
        return "—"
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


def _mae_cell(val, best_val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    return f"<td class='num{cls}'>{val:.3f}</td>"


def _gap_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_results() -> tuple[pd.DataFrame, int]:
    dfs = {}
    for tag, model_dir in [("RF", RF_DIR), ("GB", GB_DIR), ("XGB", XGB_DIR)]:
        fp = os.path.join(model_dir, "results.xlsx")
        if not os.path.exists(fp):
            print(f"ERROR: {fp} not found. Run non_linear_modeling_exp3_s2.py first.")
            sys.exit(1)
        df = pd.read_excel(fp)
        run = int(df["run"].max())
        dfs[tag] = df[df["run"] == run].copy()

    runs = {tag: int(df["run"].max()) for tag, df in dfs.items()}
    if len(set(runs.values())) > 1:
        print(f"WARNING: run numbers differ across models — {runs}. Using RF run.")
    run = runs["RF"]

    base = dfs["RF"][["experiment", "model_name", "target", "n_train", "n_test"]].copy()
    for tag in MODELS:
        suffix = f"_{tag}"
        sub = dfs[tag][["model_name", "R2_train", "RMSE_train", "MAE_train",
                         "R2_test", "RMSE_test", "MAE_test", "R2_gap"]].copy()
        sub.columns = ["model_name"] + [c + suffix for c in
                                        ["R2_train", "RMSE_train", "MAE_train",
                                         "R2_test", "RMSE_test", "MAE_test", "R2_gap"]]
        base = base.merge(sub, on="model_name", how="left")

    return base, run


# ── Chart generation ───────────────────────────────────────────────────────────

def _make_bar_chart(sub: pd.DataFrame, metric: str, label: str,
                    exp: str, run: int) -> str:
    short_labels = []
    for tgt in sub["target"]:
        parts = tgt.replace("Effluent ", "").replace(" (mg/L, ", " ").replace(")", "")
        short_labels.append(parts)

    x = np.arange(len(sub))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 1.2), 4))

    colors = {"RF": RF_COLOR, "GB": GB_COLOR, "XGB": XGB_COLOR}
    for i, tag in enumerate(MODELS):
        vals = sub[f"{metric}_{tag}"].tolist()
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=tag, color=colors[tag],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if metric.startswith("R2") else 0),
                        f"{v:+.2f}" if metric.startswith("R2") else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)

    if metric.startswith("R2"):
        ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(f"{label} — {EXPERIMENT_LABELS.get(exp, exp)}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_{exp.replace(' ','')}_{metric}_run{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


# ── Auto-observation ───────────────────────────────────────────────────────────

def _auto_obs(sub: pd.DataFrame, exp: str) -> str:
    lines = []

    wins = {tag: 0 for tag in MODELS}
    for _, row in sub.iterrows():
        r2s = {tag: row.get(f"R2_test_{tag}", -9999) for tag in MODELS}
        w = max(r2s, key=r2s.get)
        wins[w] += 1
    dominant = max(wins, key=wins.get)
    win_str = " | ".join(f"{tag}: {wins[tag]}W" for tag in MODELS)
    lines.append(
        f"<strong>Winner (Test R²):</strong> <strong>{dominant}</strong> leads "
        f"in {wins[dominant]}/{len(sub)} datasets ({win_str})."
    )

    avg_r2 = {tag: sub[f"R2_test_{tag}"].mean() for tag in MODELS}
    r2_str = " | ".join(f"{tag}&nbsp;{avg_r2[tag]:+.3f}" for tag in MODELS)
    lines.append(f"<strong>Average Test R²:</strong> {r2_str}.")

    gap_notes = []
    for tag in MODELS:
        g = sub[f"R2_gap_{tag}"].mean()
        if g > 0.35:
            gap_notes.append(f"<span class='gap-bad'>{tag} heavily overfit (avg gap {g:+.2f})</span>")
        elif g > 0.15:
            gap_notes.append(f"<span class='gap-warn'>{tag} moderately overfit (avg gap {g:+.2f})</span>")
    if gap_notes:
        lines.append("<strong>Overfitting:</strong> " + "; ".join(gap_notes) + ".")
    else:
        lines.append("<strong>Generalisation:</strong> All models show small train–test gaps.")

    ph_rows = sub[sub["target"].str.endswith("pH") | sub["target"].str.endswith("pH)")]
    if len(ph_rows) > 0:
        best_ph = ph_rows[[f"R2_test_{t}" for t in MODELS]].max(axis=1).mean()
        if best_ph < 0.15:
            lines.append(
                f"<strong>pH targets:</strong> All models struggle (best avg Test R² "
                f"{best_ph:+.3f}). Effluent pH is tightly buffered — low variance target."
            )

    all_r2 = sub[[f"R2_test_{t}" for t in MODELS]]
    best_val = all_r2.max(axis=1).max()
    best_tgt = sub.loc[all_r2.max(axis=1).idxmax(), "target"]
    best_col = all_r2.loc[all_r2.max(axis=1).idxmax()].idxmax()
    best_model = best_col.replace("R2_test_", "")
    lines.append(
        f"<strong>Best single result:</strong> {best_model} on <em>{best_tgt}</em> "
        f"— Test R² = {best_val:+.3f}."
    )

    return "<br>".join(lines)


# ── Metric tables ──────────────────────────────────────────────────────────────

def _primary_table(sub: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Dataset</th>
          <th rowspan="2">n<br><span class='sub'>train</span></th>
          <th rowspan="2">n<br><span class='sub'>test</span></th>
          <th colspan="4" class="th-rf">RF</th>
          <th colspan="4" class="th-gb">GB</th>
          <th colspan="4" class="th-xgb">XGB</th>
        </tr>
        <tr>
          <th class="th-rf">Test R²</th>
          <th class="th-rf">RMSE</th>
          <th class="th-rf">MAE</th>
          <th class="th-rf">R² Gap</th>
          <th class="th-gb">Test R²</th>
          <th class="th-gb">RMSE</th>
          <th class="th-gb">MAE</th>
          <th class="th-gb">R² Gap</th>
          <th class="th-xgb">Test R²</th>
          <th class="th-xgb">RMSE</th>
          <th class="th-xgb">MAE</th>
          <th class="th-xgb">R² Gap</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in sub.iterrows():
        label = (row["target"]
                 .replace("Effluent ", "")
                 .replace(" (mg/L, Grab)", " (Grab)")
                 .replace(" (mg/L, Composite)", " (Comp)"))

        best_r2   = max(row.get(f"R2_test_{t}",   -9999) for t in MODELS)
        best_rmse = min(row.get(f"RMSE_test_{t}", 9999)  for t in MODELS
                        if not pd.isna(row.get(f"RMSE_test_{t}")))
        best_mae  = min(row.get(f"MAE_test_{t}",  9999)  for t in MODELS
                        if not pd.isna(row.get(f"MAE_test_{t}")))

        rows_html += "<tr>"
        rows_html += f"<td class='left'>{label}</td>"
        rows_html += f"<td class='num'>{row['n_train']:,}</td>"
        rows_html += f"<td class='num'>{row['n_test']:,}</td>"
        for tag in MODELS:
            rows_html += _r2_cell(row.get(f"R2_test_{tag}"),   best_r2)
            rows_html += _rmse_cell(row.get(f"RMSE_test_{tag}"), best_rmse)
            rows_html += _mae_cell(row.get(f"MAE_test_{tag}"),  best_mae)
            rows_html += _gap_cell(row.get(f"R2_gap_{tag}"))
        rows_html += "</tr>"

    return header + rows_html + "</tbody></table>"


def _train_table(sub: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Dataset</th>
          <th class="th-rf">RF Train R²</th>
          <th class="th-rf">RF Train RMSE</th>
          <th class="th-gb">GB Train R²</th>
          <th class="th-gb">GB Train RMSE</th>
          <th class="th-xgb">XGB Train R²</th>
          <th class="th-xgb">XGB Train RMSE</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in sub.iterrows():
        label = (row["target"]
                 .replace("Effluent ", "")
                 .replace(" (mg/L, Grab)", " (Grab)")
                 .replace(" (mg/L, Composite)", " (Comp)"))
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{label}</td>"
        for tag in MODELS:
            r2t = row.get(f"R2_train_{tag}")
            rmse_t = row.get(f"RMSE_train_{tag}")
            colour = "pos" if (not pd.isna(r2t) and r2t >= 0) else "neg"
            rows_html += f"<td class='num {colour}'>{_fmt(r2t)}</td>"
            rows_html += f"<td class='num'>{rmse_t:.3f}</td>"
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


# ── Per-experiment section ─────────────────────────────────────────────────────

def _exp_section(sub: pd.DataFrame, exp: str, run: int) -> str:
    label  = EXPERIMENT_LABELS.get(exp, exp)
    n_feat = EXPERIMENT_NFEAT.get(exp, "?")
    obs    = _auto_obs(sub, exp)

    r2_b64   = _make_bar_chart(sub, "R2_test",   "Test R²",   exp, run)
    rmse_b64 = _make_bar_chart(sub, "RMSE_test", "Test RMSE", exp, run)

    charts_html = ""
    for b64, caption in [(r2_b64, "Test R² — all models &amp; targets"),
                         (rmse_b64, "Test RMSE — all models &amp; targets")]:
        if b64:
            charts_html += f"""
            <div class="plot-box wide">
              <img src="{b64}" alt="{caption}">
              <div class="plot-caption">{caption}</div>
            </div>"""

    scatter_html    = ""
    timeseries_html = ""
    importance_html = ""

    for _, row in sub.iterrows():
        name = row["model_name"]
        label_ds = (row["target"]
                    .replace("Effluent ", "")
                    .replace(" (mg/L, Grab)", " (Grab)")
                    .replace(" (mg/L, Composite)", " (Comp)"))

        sc_row = ""
        for tag, model_dir in [("RF", RF_DIR), ("GB", GB_DIR), ("XGB", XGB_DIR)]:
            path = os.path.join(model_dir, "plots", f"{name}_{tag}_run_{run}_scatter.png")
            b64  = _img_b64(path)
            if b64:
                sc_row += f"""
                <div class="plot-box narrow">
                  <img src="{b64}" alt="{label_ds} {tag} scatter">
                  <div class="plot-caption">{tag} — {label_ds}</div>
                </div>"""
        if sc_row:
            scatter_html += f"<div style='margin-bottom:8px;font-weight:500;font-size:0.85rem;color:var(--text-muted)'>{label_ds}</div>"
            scatter_html += f"<div class='plots-row'>{sc_row}</div>"

        for tag, model_dir in [("RF", RF_DIR), ("GB", GB_DIR), ("XGB", XGB_DIR)]:
            path = os.path.join(model_dir, "plots", f"{name}_{tag}_run_{run}_timeseries.png")
            b64  = _img_b64(path)
            if b64:
                timeseries_html += f"""
                <div class="plot-box" style="flex:1 1 100%;max-width:100%">
                  <img src="{b64}" alt="{label_ds} {tag} timeseries">
                  <div class="plot-caption">{tag} — {label_ds} timeseries</div>
                </div>"""

        imp_row = ""
        for tag, model_dir in [("RF", RF_DIR), ("GB", GB_DIR), ("XGB", XGB_DIR)]:
            path = os.path.join(model_dir, "plots", f"{name}_{tag}_run_{run}_importance.png")
            b64  = _img_b64(path)
            if b64:
                imp_row += f"""
                <div class="plot-box narrow">
                  <img src="{b64}" alt="{label_ds} {tag} importance">
                  <div class="plot-caption">{tag} — {label_ds}</div>
                </div>"""
        if imp_row:
            importance_html += f"<div style='margin-bottom:8px;font-weight:500;font-size:0.85rem;color:var(--text-muted)'>{label_ds}</div>"
            importance_html += f"<div class='plots-row'>{imp_row}</div>"

    exp_id = exp.replace(" ", "-")
    return f"""
    <div class="exp-card" id="{exp_id}">
      <h2>{label} <span class="run-badge">run {run}</span></h2>
      <p style="margin:0 0 12px;font-size:0.85rem;color:var(--text-meta)">
        Features: <strong>{n_feat}</strong> &nbsp;|&nbsp;
        Datasets: <strong>{len(sub)}</strong> &nbsp;|&nbsp;
        Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
      </p>

      <div class="obs">{obs}</div>

      <h3>Primary Metrics — Test Set</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 4px">
        <span class="gap-ok">■</span> gap &lt; 0.10 &nbsp;
        <span class="gap-warn">■</span> gap 0.10–0.25 (moderate) &nbsp;
        <span class="gap-bad">■</span> gap &gt; 0.25 (overfit) &nbsp;&nbsp;
        R² Gap = Train R² − Test R² (lower is better)
      </p>
      {_primary_table(sub)}

      <details>
        <summary>Train-Set Metrics</summary>
        <div class="details-body">
          {_train_table(sub)}
        </div>
      </details>

      <h3>Summary Charts</h3>
      <div class="plots-row">{charts_html}</div>

      <details>
        <summary>Feature Importance (impurity-based, per dataset)</summary>
        <div class="details-body">
          {importance_html if importance_html else "<p style='color:var(--text-meta)'>No importance plots found.</p>"}
        </div>
      </details>

      <details>
        <summary>Actual vs Predicted — Test Set Scatter</summary>
        <div class="details-body">
          {scatter_html if scatter_html else "<p style='color:var(--text-meta)'>No scatter plots found.</p>"}
        </div>
      </details>

      <details>
        <summary>Time Series — Full Dataset</summary>
        <div class="details-body">
          <div class="plots-row" style="flex-direction:column">
            {timeseries_html if timeseries_html else "<p style='color:var(--text-meta)'>No timeseries plots found.</p>"}
          </div>
        </div>
      </details>
    </div>
    """


# ── Cross-experiment summary ───────────────────────────────────────────────────

def _top_summary_cards(df: pd.DataFrame, run: int) -> str:
    n_ds = len(df)

    wins = {tag: 0 for tag in MODELS}
    for _, row in df.iterrows():
        r2s = {tag: row.get(f"R2_test_{tag}", -9999) for tag in MODELS}
        wins[max(r2s, key=r2s.get)] += 1
    overall_best = max(wins, key=wins.get)
    win_str = " · ".join(f"{wins[t]}W {t}" for t in MODELS)

    grab_df = df[df["target"].str.contains("Grab")]
    comp_df = df[df["target"].str.contains("Composite")]

    def _winner(sub):
        w = {t: 0 for t in MODELS}
        for _, row in sub.iterrows():
            r2s = {t: row.get(f"R2_test_{t}", -9999) for t in MODELS}
            w[max(r2s, key=r2s.get)] += 1
        return max(w, key=w.get), w

    grab_best, grab_wins = _winner(grab_df)
    comp_best, comp_wins = _winner(comp_df)
    grab_str = " · ".join(f"{grab_wins[t]}W {t}" for t in MODELS)
    comp_str = " · ".join(f"{comp_wins[t]}W {t}" for t in MODELS)

    avg_r2 = {tag: df[f"R2_test_{tag}"].mean() for tag in MODELS}

    best_val = df[[f"R2_test_{t}" for t in MODELS]].max(axis=1).max()
    best_row = df.loc[df[[f"R2_test_{t}" for t in MODELS]].max(axis=1).idxmax()]

    return f"""
    <div class="summary-grid">
      <div class="stat-card">
        <div class="label">Datasets evaluated</div>
        <div class="value">{n_ds}</div>
        <div class="sub">8 datasets</div>
      </div>
      <div class="stat-card" style="border-top-color:#F0B849">
        <div class="label">Best on Grab Effluents</div>
        <div class="value">{grab_best}</div>
        <div class="sub">{grab_str} ({len(grab_df)} datasets)</div>
      </div>
      <div class="stat-card" style="border-top-color:#B07FD4">
        <div class="label">Best on Composite Effluents</div>
        <div class="value">{comp_best}</div>
        <div class="sub">{comp_str} ({len(comp_df)} datasets)</div>
      </div>
      <div class="stat-card">
        <div class="label">Overall best model</div>
        <div class="value">{overall_best}</div>
        <div class="sub">{win_str}</div>
      </div>
      <div class="stat-card" style="border-top-color:{RF_COLOR}">
        <div class="label">RF avg Test R²</div>
        <div class="value">{avg_r2['RF']:+.3f}</div>
        <div class="sub">300 trees, depth tuned</div>
      </div>
      <div class="stat-card" style="border-top-color:{GB_COLOR}">
        <div class="label">GB avg Test R²</div>
        <div class="value">{avg_r2['GB']:+.3f}</div>
        <div class="sub">300 estimators, lr=0.05</div>
      </div>
      <div class="stat-card" style="border-top-color:{XGB_COLOR}">
        <div class="label">XGB avg Test R²</div>
        <div class="value">{avg_r2['XGB']:+.3f}</div>
        <div class="sub">300 estimators, lr=0.05</div>
      </div>
      <div class="stat-card">
        <div class="label">Best single result</div>
        <div class="value">{best_val:+.3f}</div>
        <div class="sub">{best_row['target'].replace('Effluent ','')}</div>
      </div>
    </div>
    """


def _cross_exp_table(df: pd.DataFrame) -> str:
    rows_html = ""
    for exp in ["Experiment 3 Sub-2"]:
        grp = df[df["experiment"] == exp]
        if grp.empty:
            continue
        label  = EXPERIMENT_LABELS.get(exp, exp)
        n_feat = EXPERIMENT_NFEAT.get(exp, "?")
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{label}</td>"
        rows_html += f"<td class='num'>{n_feat}</td>"
        for tag in MODELS:
            rows_html += f"<td class='num'>{grp[f'R2_test_{tag}'].mean():+.3f}</td>"
        for tag in MODELS:
            rows_html += f"<td class='num'>{grp[f'RMSE_test_{tag}'].mean():.3f}</td>"
        rows_html += "</tr>"

    return f"""
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Experiment</th>
          <th># features</th>
          <th class="th-rf">RF avg R²</th>
          <th class="th-gb">GB avg R²</th>
          <th class="th-xgb">XGB avg R²</th>
          <th class="th-rf">RF avg RMSE</th>
          <th class="th-gb">GB avg RMSE</th>
          <th class="th-xgb">XGB avg RMSE</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


# ── Full HTML assembly ─────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    legend = """
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background:#2171B5"></span> Random Forest (RF)</div>
      <div class="legend-item"><span class="dot" style="background:#238B45"></span> Gradient Boosting (GB)</div>
      <div class="legend-item"><span class="dot" style="background:#D94801"></span> XGBoost (XGB)</div>
    </div>
    """

    exp_sections = ""
    for exp in ["Experiment 3 Sub-2"]:
        sub = df[df["experiment"] == exp].copy()
        if sub.empty:
            continue
        exp_sections += _exp_section(sub, exp, run)

    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Purpose:</strong> Compare three tree-ensemble models — Random Forest (RF),
      Gradient Boosting (GB), and XGBoost (XGB) — on Experiment 3 Sub-2 datasets using
      <strong>tuned hyperparameters</strong> (Exp2 Sub-2 baseline + ADD-tier + CONSIDER-tier
      candidate features). This is the broadest feature set tested to date — CONSIDER-tier
      features have higher marginal row cost (up to 35–50%) but potentially higher MI signal.
      RF and GB are tuned via <code>GridSearchCV</code> (12 combos); XGB via
      <code>RandomizedSearchCV</code> (n_iter=30). All use <code>TimeSeriesSplit(n_splits=3)</code>
      scored on neg-RMSE. Fixed across all models: n_estimators=300, learning_rate=0.05
      (GB/XGB), subsample=0.8.<br><br>
      <strong>Feature importance note:</strong> All three models use impurity-based
      (Gini/variance) importance, which can overstate importance of high-cardinality
      or continuous features. Treat as directional, not definitive — use permutation
      importance for rigorous feature ranking.
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Non-Linear Models Report — Experiment 3 Sub-2 (RF vs GB vs XGB) | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Tree Ensemble Models — Experiment 3 Sub-2
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025</div>

  {intro}
  {legend}

  <h2>Cross-Experiment Summary</h2>
  {_top_summary_cards(df, run)}
  {_cross_exp_table(df)}

  {exp_sections}

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    df, run = _load_results()
    print(f"Building report for run {run}  ({len(df)} datasets)...")

    html = build_html(df, run)

    out_path = os.path.join(SCRIPT_DIR, f"report_nonlinear_exp3_s2_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
