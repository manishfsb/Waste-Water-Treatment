"""
generate_report_ridge.py — RF vs Ridge comparison report across all stages.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/ridge/generate_report_ridge.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
RIDGE_RESULTS = os.path.join(SCRIPT_DIR, "results.xlsx")
RF_S1        = os.path.join(MODELING_DIR, "results.xlsx")
RF_S2P1      = os.path.join(MODELING_DIR, "stage2_phase1", "results.xlsx")
RF_S2P2      = os.path.join(MODELING_DIR, "stage2_phase2", "results.xlsx")

TARGETS      = ["BOD", "COD", "TSS", "pH"]
STAGES       = ["Stage 1", "Stage 2 P1", "Stage 2 P2"]

STAGE_COLOURS = {
    "Stage 1":    ("#6BAED6", "#4292C6"),
    "Stage 2 P1": ("#74C476", "#238B45"),
    "Stage 2 P2": ("#FDAE6B", "#D94801"),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rf_results() -> pd.DataFrame:
    frames = []
    for path, stage in [(RF_S1, "Stage 1"), (RF_S2P1, "Stage 2 P1"), (RF_S2P2, "Stage 2 P2")]:
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path)
        df = df[df["run"] == df["run"].max()].copy()
        df["stage"] = stage
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ridge_results() -> pd.DataFrame:
    df = pd.read_excel(RIDGE_RESULTS)
    return df[df["run"] == df["run"].max()].copy()


def get_ridge_run() -> int:
    return int(pd.read_excel(RIDGE_RESULTS)["run"].max())


# ── Chart builders ─────────────────────────────────────────────────────────────

def r2_heatmap(df_rf: pd.DataFrame, df_ridge: pd.DataFrame) -> go.Figure:
    model_names = [f"{st}_{mt}" for mt in TARGETS for st in ["grab", "comp"]]
    col_labels  = []
    r2_matrix   = []

    for stage in STAGES:
        for algo, df in [("RF", df_rf), ("Ridge", df_ridge)]:
            col_labels.append(f"{stage}<br>{algo}")
            col_vals = []
            for mshort in model_names:
                if algo == "RF":
                    match = df[(df["stage"] == stage) & (df["model"].str.endswith(mshort))]
                    val = float(match["RF_R2_test"].values[0]) if len(match) else None
                else:
                    match = df[(df["stage"] == stage) & (df["model"].str.endswith(mshort))]
                    val = float(match["Ridge_R2_test"].values[0]) if len(match) else None
                col_vals.append(val if val is not None else float("nan"))
            r2_matrix.append(col_vals)

    z = np.array(r2_matrix).T
    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=model_names,
        colorscale=[[0.0, "#9C0006"], [0.35, "#FFC7CE"], [0.5, "#FFEB9C"],
                    [0.65, "#C6EFCE"], [1.0, "#276221"]],
        zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Config: %{x}<br>R²: %{z:.3f}<extra></extra>",
        colorbar=dict(title="R²", thickness=14),
    ))
    fig.update_layout(
        title="R² Test Set — All Models, All Stages, RF vs Ridge",
        height=520, xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=20, t=100, b=20),
    )
    return fig


def delta_chart(df_rf: pd.DataFrame, df_ridge: pd.DataFrame) -> go.Figure:
    deltas, labels, colours = [], [], []
    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) & (df_rf["model"].str.endswith(suffix))]
                ridge_row = df_ridge[(df_ridge["stage"] == stage) & (df_ridge["model"].str.endswith(suffix))]
                if len(rf_row) and len(ridge_row):
                    d = (float(ridge_row["Ridge_R2_test"].values[0]) -
                         float(rf_row["RF_R2_test"].values[0]))
                    deltas.append(d)
                    labels.append(f"{stage} · {st}_{t}")
                    colours.append("#238B45" if d > 0 else "#CB181D")

    order = np.argsort(deltas)
    fig = go.Figure(go.Bar(
        x=[deltas[i] for i in order],
        y=[labels[i] for i in order],
        orientation="h",
        marker_color=[colours[i] for i in order],
        hovertemplate="%{y}<br>ΔR² (Ridge−RF): %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="ΔR² = Ridge minus RF  (positive = Ridge better)",
        height=650, xaxis_title="ΔR²",
        margin=dict(l=200, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def summary_table_html(df_rf: pd.DataFrame, df_ridge: pd.DataFrame) -> str:
    def cell(val, is_r2=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '<td style="text-align:center;color:#aaa">—</td>'
        if is_r2:
            if val >= 0.7:   bg, fg = "#C6EFCE", "#276221"
            elif val >= 0.4: bg, fg = "#FFEB9C", "#9C5700"
            else:            bg, fg = "#FFC7CE", "#9C0006"
            return f'<td style="background:{bg};color:{fg};font-weight:bold;text-align:center">{val:.3f}</td>'
        return f'<td style="text-align:center">{val:.3f}</td>'

    def winner(rf_r2, ridge_r2):
        if rf_r2 is None or ridge_r2 is None:
            return '<td style="text-align:center">—</td>'
        if ridge_r2 > rf_r2 + 0.01:
            return '<td style="text-align:center;color:#238B45;font-weight:bold">Ridge ▲</td>'
        elif rf_r2 > ridge_r2 + 0.01:
            return '<td style="text-align:center;color:#2171B5;font-weight:bold">RF ▲</td>'
        return '<td style="text-align:center;color:#888">≈ Tie</td>'

    rows_html = ""
    for stage in STAGES:
        first = True
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) & (df_rf["model"].str.endswith(suffix))]
                ridge_row = df_ridge[(df_ridge["stage"] == stage) & (df_ridge["model"].str.endswith(suffix))]

                rf_rmse_tr = float(rf_row["RF_RMSE_train"].values[0]) if len(rf_row) else None
                rf_r2_tr   = float(rf_row["RF_R2_train"].values[0])   if len(rf_row) else None
                rf_rmse    = float(rf_row["RF_RMSE_test"].values[0])  if len(rf_row) else None
                rf_r2      = float(rf_row["RF_R2_test"].values[0])    if len(rf_row) else None
                ridge_rmse_tr = float(ridge_row["Ridge_RMSE_train"].values[0]) if len(ridge_row) else None
                ridge_r2_tr   = float(ridge_row["Ridge_R2_train"].values[0])   if len(ridge_row) else None
                ridge_rmse = float(ridge_row["Ridge_RMSE_test"].values[0])  if len(ridge_row) else None
                ridge_r2   = float(ridge_row["Ridge_R2_test"].values[0])    if len(ridge_row) else None

                stage_cell = (f'<td rowspan="8" style="font-weight:bold;vertical-align:middle;'
                              f'text-align:center;background:#F0F4F8">{stage}</td>'
                              if first else "")
                first = False
                rows_html += f"""
                <tr>
                  {stage_cell}
                  <td>{st}_{t}</td>
                  {cell(rf_rmse_tr)}{cell(rf_r2_tr, True)}
                  {cell(rf_rmse)}{cell(rf_r2, True)}
                  {cell(ridge_rmse_tr)}{cell(ridge_r2_tr, True)}
                  {cell(ridge_rmse)}{cell(ridge_r2, True)}
                  {winner(rf_r2, ridge_r2)}
                </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>Stage</th><th>Model</th>
          <th colspan="2" style="background:#1F4E79">RF (train)</th>
          <th colspan="2" style="background:#2171B5">RF (test)</th>
          <th colspan="2" style="background:#4292C6">Ridge (train)</th>
          <th colspan="2" style="background:#2171B5">Ridge (test)</th>
          <th style="background:#555">Winner</th>
        </tr>
        <tr>
          <th></th><th></th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th></th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="table-note">
      ▲ = meaningfully better (ΔR² > 0.01).
      R²: <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥0.70</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">&lt;0.40</span>
    </p>"""


def observations_html(df_rf: pd.DataFrame, df_ridge: pd.DataFrame) -> str:
    ridge_wins, rf_wins, ties = 0, 0, 0
    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) & (df_rf["model"].str.endswith(suffix))]
                ridge_row = df_ridge[(df_ridge["stage"] == stage) & (df_ridge["model"].str.endswith(suffix))]
                if not len(rf_row) or not len(ridge_row):
                    continue
                rf_r2 = float(rf_row["RF_R2_test"].values[0])
                ridge_r2 = float(ridge_row["Ridge_R2_test"].values[0])
                d = ridge_r2 - rf_r2
                if d > 0.01:   ridge_wins += 1
                elif d < -0.01: rf_wins += 1
                else:          ties += 1

    best_ridge = df_ridge.loc[df_ridge["Ridge_R2_test"].idxmax()]
    best_rf = df_rf.loc[df_rf["RF_R2_test"].idxmax()]

    return f"""
    <ul>
      <li>Ridge outperforms RF in <b>{ridge_wins}</b> models,
          RF outperforms Ridge in <b>{rf_wins}</b>,
          approximately tied in <b>{ties}</b>.</li>
      <li>Best overall Ridge model: <b>{best_ridge['model']}</b>
          [{best_ridge['stage']}] — R² = {best_ridge['Ridge_R2_test']:.3f}</li>
      <li>Best overall RF model: <b>{best_rf['model']}</b>
          [{best_rf['stage']}] — R² = {best_rf['RF_R2_test']:.3f}</li>
    </ul>"""


CSS = """
  body { font-family:Calibri,Arial,sans-serif; margin:0; background:#F5F5F5; color:#222; }
  .container { max-width:1200px; margin:0 auto; padding:30px 24px; }
  h1 { color:#333; border-bottom:3px solid #333; padding-bottom:10px; }
  h2 { color:#333; margin-top:48px; border-left:5px solid #555; padding-left:12px; }
  .card { background:white; border-radius:8px; padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,0.12); margin-bottom:24px; }
  .summary-table { border-collapse:collapse; width:100%; font-size:13px; }
  .summary-table th, .summary-table td { border:1px solid #ddd; padding:6px 10px; }
  .summary-table thead th { background:#333; color:white; text-align:center; }
  .summary-table tbody tr:nth-child(even) { background:#F9F9F9; }
  .table-note { font-size:12px; color:#555; margin-top:8px; }
  .obs-card { background:#F0F4F8; border-left:5px solid #555;
              border-radius:4px; padding:14px 20px; margin-bottom:20px; }
  .obs-card ul { margin:0; padding-left:20px; line-height:1.9; }
  .meta { font-size:12px; color:#888; margin-top:6px; }
"""


def chart_div(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def build_report(run: int) -> str:
    df_rf = load_rf_results()
    df_ridge = load_ridge_results()

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RF vs Ridge Comparison Report — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>RF vs Ridge Regression — Comparison Report</h1>
  <p class="meta">
    All stages (Stage 1, Stage 2 P1, Stage 2 P2) &nbsp;|&nbsp;
    Test year: 2025 &nbsp;|&nbsp; Ridge run: <b>{run}</b>
  </p>

  <h2>Overview Heatmap</h2>
  <div class="card">{chart_div(r2_heatmap(df_rf, df_ridge))}</div>

  <h2>ΔR² — Where Ridge beats RF (and where it doesn't)</h2>
  <div class="card">{chart_div(delta_chart(df_rf, df_ridge))}</div>

  <h2>Full Summary Table</h2>
  <div class="card">{summary_table_html(df_rf, df_ridge)}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{observations_html(df_rf, df_ridge)}</div>
</div>
</body>
</html>"""


def main():
    run = get_ridge_run()
    print(f"Generating RF vs Ridge report for Ridge run {run}...")
    html = build_report(run)
    out_path = os.path.join(SCRIPT_DIR, f"report_ridge_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
