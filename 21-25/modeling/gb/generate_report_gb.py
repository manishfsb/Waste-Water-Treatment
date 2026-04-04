"""
generate_report_gb.py — Combined RF vs GB comparison report across all stages.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/gb/generate_report_gb.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from report_theme import dark_mode_css, DARK_MODE_JS

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
GB_RESULTS   = os.path.join(SCRIPT_DIR, "results.xlsx")
RF_S1        = os.path.join(MODELING_DIR, "results.xlsx")
RF_S2P1      = os.path.join(MODELING_DIR, "stage2_phase1", "results.xlsx")
RF_S2P2      = os.path.join(MODELING_DIR, "stage2_phase2", "results.xlsx")

# Targets and display labels
TARGETS      = ["BOD", "COD", "TSS", "pH"]
SAMPLE_TYPES = ["Grab", "Composite"]
STAGES       = ["Stage 1", "Stage 2 P1", "Stage 2 P2"]

STAGE_COLOURS = {
    "Stage 1":    ("#6BAED6", "#2171B5"),   # (RF, GB)
    "Stage 2 P1": ("#74C476", "#238B45"),
    "Stage 2 P2": ("#FDAE6B", "#D94801"),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rf_results() -> pd.DataFrame:
    """Merge RF results from all three stage result files into one DataFrame."""
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


def load_gb_results() -> pd.DataFrame:
    df = pd.read_excel(GB_RESULTS)
    return df[df["run"] == df["run"].max()].copy()


def get_gb_run() -> int:
    return int(pd.read_excel(GB_RESULTS)["run"].max())


def model_short(name: str) -> str:
    """e.g. 'stage1_grab_BOD' → 'grab_BOD'"""
    parts = name.split("_")
    # remove stage prefix (stage1, stage2, p1, p2)
    for i, p in enumerate(parts):
        if p in ("grab", "comp"):
            return "_".join(parts[i:])
    return name


# ── Chart builders ─────────────────────────────────────────────────────────────

def r2_heatmap(df_rf: pd.DataFrame, df_gb: pd.DataFrame) -> go.Figure:
    """
    Heatmap grid: rows = models, cols = (Stage, Algorithm).
    Cell value = R² test. Colour = performance level.
    """
    model_names = [
        f"{st}_{mt}" for mt in TARGETS for st in ["grab", "comp"]
    ]

    col_labels = []
    r2_matrix  = []

    for stage in STAGES:
        for algo, df in [("RF", df_rf), ("GB", df_gb)]:
            col_labels.append(f"{stage}<br>{algo}")
            col_vals = []
            for mshort in model_names:
                # find matching row
                if algo == "RF":
                    match = df[(df["stage"] == stage) &
                               (df["model"].str.endswith(mshort))]
                    val = float(match["RF_R2_test"].values[0]) if len(match) else None
                else:
                    match = df[(df["stage"] == stage) &
                               (df["model"].str.endswith(mshort))]
                    val = float(match["GB_R2_test"].values[0]) if len(match) else None
                col_vals.append(val if val is not None else float("nan"))
            r2_matrix.append(col_vals)

    z = np.array(r2_matrix).T   # shape: (n_models, n_cols)

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
        text=[[f"{v:.2f}" if not np.isnan(v) else "—"
               for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Config: %{x}<br>R²: %{z:.3f}<extra></extra>",
        colorbar=dict(title="R²", thickness=14),
    ))
    fig.update_layout(
        title="R² Test Set — All Models, All Stages, RF vs GB",
        height=520,
        xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=140, r=20, t=100, b=20),
    )
    return fig


def r2_bar_by_target(df_rf: pd.DataFrame, df_gb: pd.DataFrame,
                     target: str, sample_type: str) -> go.Figure:
    """
    Grouped bar chart: x = stage, bars = RF and GB, for one target + sample type.
    """
    suffix = f"{'grab' if sample_type == 'Grab' else 'comp'}_{target}"

    fig = go.Figure()
    for stage in STAGES:
        rf_row = df_rf[(df_rf["stage"] == stage) &
                       (df_rf["model"].str.endswith(suffix))]
        gb_row = df_gb[(df_gb["stage"] == stage) &
                       (df_gb["model"].str.endswith(suffix))]

        rf_val = float(rf_row["RF_R2_test"].values[0]) if len(rf_row) else None
        gb_val = float(gb_row["GB_R2_test"].values[0]) if len(gb_row) else None

        rf_c, gb_c = STAGE_COLOURS[stage]
        if rf_val is not None:
            fig.add_trace(go.Bar(
                name=f"{stage} RF", x=["RF"], y=[rf_val],
                marker_color=rf_c, legendgroup=stage,
                showlegend=True,
                hovertemplate=f"{stage} RF: {rf_val:.3f}<extra></extra>",
                offsetgroup=stage,
            ))
        if gb_val is not None:
            fig.add_trace(go.Bar(
                name=f"{stage} GB", x=["GB"], y=[gb_val],
                marker_color=gb_c, legendgroup=stage,
                showlegend=True,
                hovertemplate=f"{stage} GB: {gb_val:.3f}<extra></extra>",
                offsetgroup=stage + "_gb",
            ))

    # Simpler: just use x = stage names, paired bars
    fig2 = go.Figure()
    x_labels = STAGES
    rf_vals = []
    gb_vals = []
    for stage in STAGES:
        rf_row = df_rf[(df_rf["stage"] == stage) &
                       (df_rf["model"].str.endswith(suffix))]
        gb_row = df_gb[(df_gb["stage"] == stage) &
                       (df_gb["model"].str.endswith(suffix))]
        rf_vals.append(float(rf_row["RF_R2_test"].values[0])
                       if len(rf_row) else None)
        gb_vals.append(float(gb_row["GB_R2_test"].values[0])
                       if len(gb_row) else None)

    fig2.add_trace(go.Bar(name="RF", x=x_labels, y=rf_vals,
                          marker_color="#2171B5",
                          hovertemplate="%{x} RF: %{y:.3f}<extra></extra>"))
    fig2.add_trace(go.Bar(name="GB", x=x_labels, y=gb_vals,
                          marker_color="#D94801",
                          hovertemplate="%{x} GB: %{y:.3f}<extra></extra>"))
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig2.update_layout(
        title=f"{target} — {sample_type}",
        barmode="group", height=300,
        yaxis_title="R² (test)", showlegend=True,
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=50, r=10, t=60, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig2


def delta_chart(df_rf: pd.DataFrame, df_gb: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of GB − RF R² for every model.
    Positive = GB better, negative = RF better.
    """
    deltas, labels, colours = [], [], []

    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) &
                               (df_rf["model"].str.endswith(suffix))]
                gb_row = df_gb[(df_gb["stage"] == stage) &
                               (df_gb["model"].str.endswith(suffix))]
                if len(rf_row) and len(gb_row):
                    d = (float(gb_row["GB_R2_test"].values[0]) -
                         float(rf_row["RF_R2_test"].values[0]))
                    deltas.append(d)
                    labels.append(f"{stage} · {st}_{t}")
                    colours.append("#238B45" if d > 0 else "#CB181D")

    order = np.argsort(deltas)
    fig   = go.Figure(go.Bar(
        x=[deltas[i] for i in order],
        y=[labels[i] for i in order],
        orientation="h",
        marker_color=[colours[i] for i in order],
        hovertemplate="%{y}<br>ΔR² (GB−RF): %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="ΔR² = GB minus RF  (positive = GB better)",
        height=650,
        xaxis_title="ΔR²",
        margin=dict(l=200, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def summary_table_html(df_rf: pd.DataFrame, df_gb: pd.DataFrame) -> str:
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

    def winner(rf_r2, gb_r2):
        if rf_r2 is None or gb_r2 is None:
            return '<td style="text-align:center">—</td>'
        if gb_r2 > rf_r2 + 0.01:
            return '<td style="text-align:center;color:#238B45;font-weight:bold">GB ▲</td>'
        elif rf_r2 > gb_r2 + 0.01:
            return '<td style="text-align:center;color:#2171B5;font-weight:bold">RF ▲</td>'
        return '<td style="text-align:center;color:#888">≈ Tie</td>'

    rows_html = ""
    for stage in STAGES:
        first = True
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) &
                               (df_rf["model"].str.endswith(suffix))]
                gb_row = df_gb[(df_gb["stage"] == stage) &
                               (df_gb["model"].str.endswith(suffix))]

                rf_rmse_tr = float(rf_row["RF_RMSE_train"].values[0]) if len(rf_row) else None
                rf_r2_tr   = float(rf_row["RF_R2_train"].values[0])   if len(rf_row) else None
                rf_rmse = float(rf_row["RF_RMSE_test"].values[0]) if len(rf_row) else None
                rf_r2   = float(rf_row["RF_R2_test"].values[0])   if len(rf_row) else None
                gb_rmse_tr = float(gb_row["GB_RMSE_train"].values[0]) if len(gb_row) else None
                gb_r2_tr   = float(gb_row["GB_R2_train"].values[0])   if len(gb_row) else None
                gb_rmse = float(gb_row["GB_RMSE_test"].values[0]) if len(gb_row) else None
                gb_r2   = float(gb_row["GB_R2_test"].values[0])   if len(gb_row) else None

                stage_cell = (f'<td rowspan="8" style="font-weight:bold;'
                              f'vertical-align:middle;text-align:center;'
                              f'background:#F0F4F8">{stage}</td>'
                              if first else "")
                first = False
                rows_html += f"""
                <tr>
                  {stage_cell}
                  <td>{st}_{t}</td>
                  {cell(rf_rmse_tr)}{cell(rf_r2_tr, True)}
                  {cell(rf_rmse)}{cell(rf_r2, True)}
                  {cell(gb_rmse_tr)}{cell(gb_r2_tr, True)}
                  {cell(gb_rmse)}{cell(gb_r2, True)}
                  {winner(rf_r2, gb_r2)}
                </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>Stage</th><th>Model</th>
          <th colspan="2" style="background:#1F4E79">RF (train)</th>
          <th colspan="2" style="background:#2171B5">RF (test)</th>
          <th colspan="2" style="background:#833C00">GB (train)</th>
          <th colspan="2" style="background:#D94801">GB (test)</th>
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
      ▲ = meaningfully better (ΔR² &gt; 0.01).
      R²: <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥0.70</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">&lt;0.40</span>
    </p>"""


# ── Observations ───────────────────────────────────────────────────────────────

def observations_html(df_rf: pd.DataFrame, df_gb: pd.DataFrame) -> str:
    gb_wins, rf_wins, ties = 0, 0, 0
    best_gb_delta, best_gb_model = -999, ""

    for stage in STAGES:
        for st in ["grab", "comp"]:
            for t in TARGETS:
                suffix = f"{st}_{t}"
                rf_row = df_rf[(df_rf["stage"] == stage) &
                               (df_rf["model"].str.endswith(suffix))]
                gb_row = df_gb[(df_gb["stage"] == stage) &
                               (df_gb["model"].str.endswith(suffix))]
                if not len(rf_row) or not len(gb_row):
                    continue
                rf_r2 = float(rf_row["RF_R2_test"].values[0])
                gb_r2 = float(gb_row["GB_R2_test"].values[0])
                d     = gb_r2 - rf_r2
                if d > 0.01:   gb_wins += 1
                elif d < -0.01: rf_wins += 1
                else:           ties    += 1
                if d > best_gb_delta:
                    best_gb_delta = d
                    best_gb_model = f"{stage} · {st}_{t}"

    best_overall_gb = df_gb.loc[df_gb["GB_R2_test"].idxmax()]
    best_overall_rf = df_rf.loc[df_rf["RF_R2_test"].idxmax()]

    return f"""
    <ul>
      <li>GB outperforms RF in <b>{gb_wins}</b> models,
          RF outperforms GB in <b>{rf_wins}</b>,
          approximately tied in <b>{ties}</b>.</li>
      <li>Largest GB gain: <b>{best_gb_model}</b>
          (ΔR² = {best_gb_delta:+.3f})</li>
      <li>Best overall GB model:
          <b>{best_overall_gb['model']}</b>
          [{best_overall_gb['stage']}]
          — R² = {best_overall_gb['GB_R2_test']:.3f}</li>
      <li>Best overall RF model:
          <b>{best_overall_rf['model']}</b>
          [{best_overall_rf['stage']}]
          — R² = {best_overall_rf['RF_R2_test']:.3f}</li>
    </ul>"""


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = dark_mode_css("""
  body { font-family:Calibri,Arial,sans-serif; margin:0; background:#F5F5F5; color:#222; }
  .container { max-width:1200px; margin:0 auto; padding:30px 24px; }
  h1 { color:#333; border-bottom:3px solid #333; padding-bottom:10px; }
  h2 { color:#333; margin-top:48px; border-left:5px solid #555; padding-left:12px; }
  h3 { color:#555; margin-top:24px; }
  .card { background:white; border-radius:8px; padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,0.12); margin-bottom:24px; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .grid-4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; }
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
    df_gb = load_gb_results()

    # Per-target bar charts (2×4 grid)
    target_charts = ""
    for sample_type in SAMPLE_TYPES:
        target_charts += f"<h3>{sample_type} models</h3><div class='grid-4'>"
        for t in TARGETS:
            target_charts += f"<div>{chart_div(r2_bar_by_target(df_rf, df_gb, t, sample_type))}</div>"
        target_charts += "</div>"

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RF vs GB Comparison Report — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
<div class="container">
  <h1>RF vs Gradient Boosting — Comparison Report</h1>
  <p class="meta">
    All stages (Stage 1, Stage 2 P1, Stage 2 P2) &nbsp;|&nbsp;
    Test year: 2025 &nbsp;|&nbsp; GB run: <b>{run}</b>
  </p>

  <h2>Overview Heatmap</h2>
  <div class="card">{chart_div(r2_heatmap(df_rf, df_gb))}</div>

  <h2>ΔR² — Where GB beats RF (and where it doesn't)</h2>
  <div class="card">{chart_div(delta_chart(df_rf, df_gb))}</div>

  <h2>R² by Target</h2>
  <div class="card">{target_charts}</div>

  <h2>Full Summary Table</h2>
  <div class="card">{summary_table_html(df_rf, df_gb)}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{observations_html(df_rf, df_gb)}</div>
</div>
</body>
</html>"""


def main():
    run = get_gb_run()
    print(f"Generating RF vs GB report for GB run {run}...")
    html     = build_report(run)
    out_path = os.path.join(SCRIPT_DIR, f"report_gb_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
