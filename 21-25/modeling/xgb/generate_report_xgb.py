"""
generate_report_xgb.py — RF vs GB vs XGBoost comparison report across all stages.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/xgb/generate_report_xgb.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
XGB_RESULTS  = os.path.join(SCRIPT_DIR, "results.xlsx")
GB_RESULTS   = os.path.join(MODELING_DIR, "gb", "results.xlsx")
RF_S1        = os.path.join(MODELING_DIR, "results.xlsx")
RF_S2P1      = os.path.join(MODELING_DIR, "stage2_phase1", "results.xlsx")
RF_S2P2      = os.path.join(MODELING_DIR, "stage2_phase2", "results.xlsx")

TARGETS      = ["BOD", "COD", "TSS", "pH"]
SAMPLE_TYPES = ["Grab", "Composite"]
STAGES       = ["Stage 1", "Stage 2 P1", "Stage 2 P2"]

ALGO_COLOURS = {"RF": "#2171B5", "GB": "#D94801", "XGB": "#238B45"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rf() -> pd.DataFrame:
    frames = []
    for path, stage in [(RF_S1, "Stage 1"), (RF_S2P1, "Stage 2 P1"),
                        (RF_S2P2, "Stage 2 P2")]:
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path)
        df = df[df["run"] == df["run"].max()].copy()
        df["stage"] = stage
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_gb() -> pd.DataFrame:
    df = pd.read_excel(GB_RESULTS)
    return df[df["run"] == df["run"].max()].copy()


def load_xgb() -> pd.DataFrame:
    df = pd.read_excel(XGB_RESULTS)
    return df[df["run"] == df["run"].max()].copy()


def get_xgb_run() -> int:
    return int(pd.read_excel(XGB_RESULTS)["run"].max())


def _lookup(df: pd.DataFrame, stage: str, suffix: str,
            col: str) -> float | None:
    match = df[(df["stage"] == stage) & (df["model"].str.endswith(suffix))]
    return float(match[col].values[0]) if len(match) else None


# ── Chart builders ─────────────────────────────────────────────────────────────

def r2_heatmap(df_rf, df_gb, df_xgb) -> go.Figure:
    """
    Heatmap: rows = model shortnames, cols = Stage × Algorithm (3×3 = 9 cols).
    """
    model_names = [f"{st}_{t}" for t in TARGETS for st in ["grab", "comp"]]

    col_labels, r2_matrix = [], []
    for stage in STAGES:
        for algo, df, col in [("RF",  df_rf,  "RF_R2_test"),
                               ("GB",  df_gb,  "GB_R2_test"),
                               ("XGB", df_xgb, "XGB_R2_test")]:
            col_labels.append(f"{stage}<br>{algo}")
            col_vals = []
            for mshort in model_names:
                val = _lookup(df, stage, mshort, col)
                col_vals.append(val if val is not None else float("nan"))
            r2_matrix.append(col_vals)

    z = np.array(r2_matrix).T   # (n_models, n_cols)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=model_names,
        colorscale=[
            [0.0,  "#9C0006"],
            [0.35, "#FFC7CE"],
            [0.5,  "#FFEB9C"],
            [0.65, "#C6EFCE"],
            [1.0,  "#276221"],
        ],
        zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Config: %{x}<br>R²: %{z:.3f}<extra></extra>",
        colorbar=dict(title="R²", thickness=14),
    ))
    fig.update_layout(
        title="R² Test Set — All Models · RF vs GB vs XGB",
        height=540,
        xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=20, t=110, b=20),
    )
    return fig


def delta_chart(df_rf, df_gb, df_xgb) -> go.Figure:
    """
    Grouped horizontal bars: ΔR² for (GB−RF) and (XGB−RF) per model.
    Positive = better than RF.
    """
    labels, gb_deltas, xgb_deltas = [], [], []

    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_r2  = _lookup(df_rf,  stage, suffix, "RF_R2_test")
                gb_r2  = _lookup(df_gb,  stage, suffix, "GB_R2_test")
                xgb_r2 = _lookup(df_xgb, stage, suffix, "XGB_R2_test")
                if rf_r2 is None or gb_r2 is None or xgb_r2 is None:
                    continue
                labels.append(f"{stage} · {suffix}")
                gb_deltas.append(gb_r2  - rf_r2)
                xgb_deltas.append(xgb_r2 - rf_r2)

    # Sort by XGB delta descending
    order = np.argsort(xgb_deltas)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[gb_deltas[i]  for i in order],
        y=[labels[i]     for i in order],
        name="GB − RF",
        orientation="h",
        marker_color=ALGO_COLOURS["GB"],
        opacity=0.85,
        hovertemplate="%{y}<br>GB−RF: %{x:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[xgb_deltas[i] for i in order],
        y=[labels[i]     for i in order],
        name="XGB − RF",
        orientation="h",
        marker_color=ALGO_COLOURS["XGB"],
        opacity=0.85,
        hovertemplate="%{y}<br>XGB−RF: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="ΔR² vs RF baseline  (positive = beats RF)",
        barmode="group",
        height=700,
        xaxis_title="ΔR²",
        legend=dict(orientation="h", y=1.04),
        margin=dict(l=220, r=20, t=70, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def r2_bar_by_target(df_rf, df_gb, df_xgb, target: str,
                     sample_type: str) -> go.Figure:
    """
    Grouped bars: x = stage, 3 bars per stage (RF / GB / XGB).
    """
    suffix = f"{'grab' if sample_type == 'Grab' else 'comp'}_{target}"

    rf_vals, gb_vals, xgb_vals = [], [], []
    for stage in STAGES:
        rf_vals.append( _lookup(df_rf,  stage, suffix, "RF_R2_test"))
        gb_vals.append( _lookup(df_gb,  stage, suffix, "GB_R2_test"))
        xgb_vals.append(_lookup(df_xgb, stage, suffix, "XGB_R2_test"))

    fig = go.Figure()
    for name, vals, colour in [
        ("RF",  rf_vals,  ALGO_COLOURS["RF"]),
        ("GB",  gb_vals,  ALGO_COLOURS["GB"]),
        ("XGB", xgb_vals, ALGO_COLOURS["XGB"]),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=STAGES, y=vals,
            marker_color=colour,
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=f"{target} — {sample_type}",
        barmode="group", height=310,
        yaxis_title="R² (test)", showlegend=True,
        legend=dict(orientation="h", y=1.18),
        margin=dict(l=50, r=10, t=65, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def head_to_head_scatter(df_rf, df_gb, df_xgb) -> go.Figure:
    """
    Scatter: XGB R² (x) vs RF R² (y) per model, coloured by stage.
    Points above y=x line → RF better; below → XGB better.
    """
    stage_colours = {"Stage 1": "#6BAED6", "Stage 2 P1": "#74C476",
                     "Stage 2 P2": "#FDAE6B"}

    fig = go.Figure()
    for stage in STAGES:
        xs, ys, texts = [], [], []
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf  = _lookup(df_rf,  stage, suffix, "RF_R2_test")
                xgb = _lookup(df_xgb, stage, suffix, "XGB_R2_test")
                if rf is not None and xgb is not None:
                    xs.append(xgb)
                    ys.append(rf)
                    texts.append(f"{stage} · {suffix}")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=stage,
            marker=dict(color=stage_colours[stage], size=9,
                        line=dict(width=1, color="white")),
            text=texts,
            hovertemplate="%{text}<br>XGB R²: %{x:.3f}<br>RF R²: %{y:.3f}<extra></extra>",
        ))

    lo = min(df_rf["RF_R2_test"].min(), df_xgb["XGB_R2_test"].min()) - 0.05
    hi = max(df_rf["RF_R2_test"].max(), df_xgb["XGB_R2_test"].max()) + 0.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color="black", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title="XGB vs RF — Head-to-Head R² (below diagonal = XGB better)",
        xaxis_title="XGB R² (test)",
        yaxis_title="RF R² (test)",
        height=440,
        legend=dict(title="Stage"),
        plot_bgcolor="#FAFAFA",
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_table_html(df_rf, df_gb, df_xgb) -> str:
    def cell(val, is_r2=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '<td style="text-align:center;color:#aaa">—</td>'
        if is_r2:
            if val >= 0.7:   bg, fg = "#C6EFCE", "#276221"
            elif val >= 0.4: bg, fg = "#FFEB9C", "#9C5700"
            else:            bg, fg = "#FFC7CE", "#9C0006"
            return (f'<td style="background:{bg};color:{fg};'
                    f'font-weight:bold;text-align:center">{val:.3f}</td>')
        return f'<td style="text-align:center">{val:.3f}</td>'

    def winner(rf_r2, gb_r2, xgb_r2):
        vals = {k: v for k, v in [("RF", rf_r2), ("GB", gb_r2), ("XGB", xgb_r2)]
                if v is not None}
        if not vals:
            return '<td style="text-align:center">—</td>'
        best_k = max(vals, key=vals.get)
        best_v = vals[best_k]
        # only declare winner if clearly ahead
        others = [v for k, v in vals.items() if k != best_k]
        margin = best_v - max(others) if others else 0
        if margin > 0.01:
            colour = ALGO_COLOURS[best_k]
            return (f'<td style="text-align:center;color:{colour};'
                    f'font-weight:bold">{best_k} ▲</td>')
        return '<td style="text-align:center;color:#888">≈ Tie</td>'

    rows_html = ""
    for stage in STAGES:
        first = True
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_rmse  = _lookup(df_rf,  stage, suffix, "RF_RMSE_test")
                rf_r2    = _lookup(df_rf,  stage, suffix, "RF_R2_test")
                gb_rmse  = _lookup(df_gb,  stage, suffix, "GB_RMSE_test")
                gb_r2    = _lookup(df_gb,  stage, suffix, "GB_R2_test")
                xgb_rmse = _lookup(df_xgb, stage, suffix, "XGB_RMSE_test")
                xgb_r2   = _lookup(df_xgb, stage, suffix, "XGB_R2_test")

                stage_cell = (f'<td rowspan="8" style="font-weight:bold;'
                              f'vertical-align:middle;text-align:center;'
                              f'background:#F0F4F8">{stage}</td>'
                              if first else "")
                first = False
                rows_html += f"""
                <tr>
                  {stage_cell}
                  <td>{st}_{t}</td>
                  {cell(rf_rmse)}{cell(rf_r2, True)}
                  {cell(gb_rmse)}{cell(gb_r2, True)}
                  {cell(xgb_rmse)}{cell(xgb_r2, True)}
                  {winner(rf_r2, gb_r2, xgb_r2)}
                </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>Stage</th><th>Model</th>
          <th colspan="2" style="background:#2171B5;color:white">RF (test)</th>
          <th colspan="2" style="background:#D94801;color:white">GB (test)</th>
          <th colspan="2" style="background:#238B45;color:white">XGB (test)</th>
          <th style="background:#555;color:white">Winner</th>
        </tr>
        <tr>
          <th></th><th></th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th></th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="table-note">
      ▲ = meaningfully best (ΔR² &gt; 0.01 over next-best).
      R²: <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥0.70</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">&lt;0.40</span>
    </p>"""


# ── Observations ───────────────────────────────────────────────────────────────

def observations_html(df_rf, df_gb, df_xgb) -> str:
    wins = {"RF": 0, "GB": 0, "XGB": 0, "Tie": 0}
    best = {"RF": (-999, ""), "GB": (-999, ""), "XGB": (-999, "")}

    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                vals = {
                    "RF":  _lookup(df_rf,  stage, suffix, "RF_R2_test"),
                    "GB":  _lookup(df_gb,  stage, suffix, "GB_R2_test"),
                    "XGB": _lookup(df_xgb, stage, suffix, "XGB_R2_test"),
                }
                vals = {k: v for k, v in vals.items() if v is not None}
                if not vals:
                    continue
                best_k = max(vals, key=vals.get)
                best_v = vals[best_k]
                others = [v for k, v in vals.items() if k != best_k]
                margin = best_v - max(others) if others else 0
                if margin > 0.01:
                    wins[best_k] += 1
                else:
                    wins["Tie"] += 1
                label = f"{stage} · {suffix}"
                for algo in ("RF", "GB", "XGB"):
                    if algo in vals and vals[algo] > best[algo][0]:
                        best[algo] = (vals[algo], label)

    li = lambda k, v: (f'<li><b>{k}</b> wins: <b>{v}</b> models</li>')

    return f"""
    <ul>
      {li("RF",  wins["RF"])}
      {li("GB",  wins["GB"])}
      {li("XGB", wins["XGB"])}
      <li>Approximately tied: <b>{wins["Tie"]}</b></li>
      <li>Best RF model:  <b>{best["RF"][1]}</b>  — R² = {best["RF"][0]:.3f}</li>
      <li>Best GB model:  <b>{best["GB"][1]}</b>  — R² = {best["GB"][0]:.3f}</li>
      <li>Best XGB model: <b>{best["XGB"][1]}</b> — R² = {best["XGB"][0]:.3f}</li>
    </ul>"""


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
  body { font-family:Calibri,Arial,sans-serif; margin:0; background:#F5F5F5; color:#222; }
  .container { max-width:1200px; margin:0 auto; padding:30px 24px; }
  h1 { color:#333; border-bottom:3px solid #333; padding-bottom:10px; }
  h2 { color:#333; margin-top:48px; border-left:5px solid #555; padding-left:12px; }
  h3 { color:#555; margin-top:24px; }
  .card { background:white; border-radius:8px; padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,0.12); margin-bottom:24px; }
  .grid-4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; }
  .summary-table { border-collapse:collapse; width:100%; font-size:13px; }
  .summary-table th, .summary-table td { border:1px solid #ddd; padding:6px 10px; }
  .summary-table thead th { color:white; text-align:center; }
  .summary-table tbody tr:nth-child(even) { background:#F9F9F9; }
  .table-note { font-size:12px; color:#555; margin-top:8px; }
  .obs-card { background:#F0F4F8; border-left:5px solid #555;
              border-radius:4px; padding:14px 20px; margin-bottom:20px; }
  .obs-card ul { margin:0; padding-left:20px; line-height:1.9; }
  .legend-strip { display:flex; gap:20px; margin-bottom:8px; font-size:13px; }
  .legend-dot { display:inline-block; width:14px; height:14px;
                border-radius:50%; margin-right:5px; vertical-align:middle; }
  .meta { font-size:12px; color:#888; margin-top:6px; }
"""


def chart_div(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def build_report(run: int) -> str:
    df_rf  = load_rf()
    df_gb  = load_gb()
    df_xgb = load_xgb()

    target_charts = ""
    for sample_type in SAMPLE_TYPES:
        target_charts += f"<h3>{sample_type} models</h3><div class='grid-4'>"
        for t in TARGETS:
            target_charts += (f"<div>"
                              f"{chart_div(r2_bar_by_target(df_rf, df_gb, df_xgb, t, sample_type))}"
                              f"</div>")
        target_charts += "</div>"

    legend_strip = "".join(
        f'<span><span class="legend-dot" style="background:{ALGO_COLOURS[k]}"></span>{k}</span>'
        for k in ("RF", "GB", "XGB")
    )

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RF vs GB vs XGB — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>RF vs Gradient Boosting vs XGBoost — Comparison Report</h1>
  <p class="meta">
    All stages (Stage 1, Stage 2 P1, Stage 2 P2) &nbsp;|&nbsp;
    Test year: 2025 &nbsp;|&nbsp; XGB run: <b>{run}</b>
  </p>
  <div class="legend-strip">{legend_strip}</div>

  <h2>Overview Heatmap</h2>
  <div class="card">{chart_div(r2_heatmap(df_rf, df_gb, df_xgb))}</div>

  <h2>ΔR² vs RF Baseline</h2>
  <div class="card">{chart_div(delta_chart(df_rf, df_gb, df_xgb))}</div>

  <h2>XGB vs RF — Head-to-Head</h2>
  <div class="card">{chart_div(head_to_head_scatter(df_rf, df_gb, df_xgb))}</div>

  <h2>R² by Target</h2>
  <div class="card">{target_charts}</div>

  <h2>Full Summary Table</h2>
  <div class="card">{summary_table_html(df_rf, df_gb, df_xgb)}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{observations_html(df_rf, df_gb, df_xgb)}</div>
</div>
</body>
</html>"""


def main():
    run = get_xgb_run()
    print(f"Generating RF vs GB vs XGB report for run {run}...")
    html     = build_report(run)
    out_path = os.path.join(SCRIPT_DIR, f"report_xgb_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
