"""
generate_report_stage2_phase2.py
Compile Stage 2 Phase 2 results into a self-contained HTML report.
Includes a 3-way comparison: Stage 1 vs Stage 2 Phase 1 vs Stage 2 Phase 2.

Usage (from project root New_Project/):
    .venv/bin/python3 21-25/modeling/stage2_phase2/generate_report_stage2_phase2.py

Usage (from stage2_phase2/ directory):
    ../../../.venv/bin/python3 generate_report_stage2_phase2.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR    = os.path.dirname(SCRIPT_DIR)
DATA_DIR        = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR      = os.path.join(SCRIPT_DIR, "models")
RESULTS_FILE    = os.path.join(SCRIPT_DIR, "results.xlsx")
STAGE1_RESULTS  = os.path.join(MODELING_DIR, "results.xlsx")
STAGE2P1_RESULTS= os.path.join(MODELING_DIR, "stage2_phase1", "results.xlsx")

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

YEAR_COLOURS = {
    2021: "#2171B5", 2022: "#74C476",
    2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801",
}

GRAB_INLET = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
    "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)",
]
COMP_INLET = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
]
SEC_COLS = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)", "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
]
COMMON = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]

GRAB_FEATURES = GRAB_INLET + SEC_COLS + COMMON
COMP_FEATURES = COMP_INLET + SEC_COLS + COMMON

MODELS = [
    ("stage2_p2_grab_BOD", "Effluent BOD (mg/L, Grab)",      "BOD", "Grab",      GRAB_FEATURES),
    ("stage2_p2_grab_COD", "Effluent COD (mg/L, Grab)",      "COD", "Grab",      GRAB_FEATURES),
    ("stage2_p2_grab_TSS", "Effluent TSS (mg/L, Grab)",      "TSS", "Grab",      GRAB_FEATURES),
    ("stage2_p2_grab_pH",  "Effluent pH (Grab)",             "pH",  "Grab",      GRAB_FEATURES),
    ("stage2_p2_comp_BOD", "Effluent BOD (mg/L, Composite)", "BOD", "Composite", COMP_FEATURES),
    ("stage2_p2_comp_COD", "Effluent COD (mg/L, Composite)", "COD", "Composite", COMP_FEATURES),
    ("stage2_p2_comp_TSS", "Effluent TSS (mg/L, Composite)", "TSS", "Composite", COMP_FEATURES),
    ("stage2_p2_comp_pH",  "Effluent pH (Composite)",        "pH",  "Composite", COMP_FEATURES),
]

# Cross-stage equivalents
EQUIV = {
    "stage2_p2_grab_BOD": ("stage1_grab_BOD", "stage2_p1_grab_BOD"),
    "stage2_p2_grab_COD": ("stage1_grab_COD", "stage2_p1_grab_COD"),
    "stage2_p2_grab_TSS": ("stage1_grab_TSS", "stage2_p1_grab_TSS"),
    "stage2_p2_grab_pH":  ("stage1_grab_pH",  "stage2_p1_grab_pH"),
    "stage2_p2_comp_BOD": ("stage1_comp_BOD", "stage2_p1_comp_BOD"),
    "stage2_p2_comp_COD": ("stage1_comp_COD", "stage2_p1_comp_COD"),
    "stage2_p2_comp_TSS": ("stage1_comp_TSS", "stage2_p1_comp_TSS"),
    "stage2_p2_comp_pH":  ("stage1_comp_pH",  "stage2_p1_comp_pH"),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def get_latest_run() -> int:
    return int(pd.read_excel(RESULTS_FILE)["run"].max())


def load_subset(name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(DATA_DIR, f"{name}.xlsx"),
                         parse_dates=["Date"])


def load_results(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    return df[df["run"] == df["run"].max()]


def r2_for(df: pd.DataFrame | None, model_name: str) -> float | None:
    if df is None:
        return None
    rows = df[df["model"] == model_name]["RF_R2_test"].values
    return float(rows[0]) if len(rows) else None


# ── Charts ─────────────────────────────────────────────────────────────────────

def scatter_chart(df: pd.DataFrame, target: str, pred_col: str,
                  title: str) -> go.Figure:
    test = df[df["year"] == TEST_YEAR].copy()
    fig  = go.Figure()
    for yr in sorted(test["year"].unique()):
        sub = test[test["year"] == yr]
        fig.add_trace(go.Scatter(
            x=sub[target], y=sub[pred_col], mode="markers", name=str(yr),
            marker=dict(color=YEAR_COLOURS.get(yr, "#999"), size=6, opacity=0.75),
            text=sub["Date"].dt.strftime("%d %b %Y"),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
    lo = min(test[target].min(), test[pred_col].min())
    hi = max(test[target].max(), test[pred_col].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="black", dash="dash", width=1),
                             name="Perfect fit", hoverinfo="skip"))
    fig.update_layout(title=title, height=420,
                      xaxis_title="Actual", yaxis_title="Predicted",
                      legend_title="Year", margin=dict(l=50, r=20, t=50, b=40),
                      plot_bgcolor="#FAFAFA")
    return fig


def timeseries_chart(df: pd.DataFrame, target: str, pred_col: str,
                     title: str) -> go.Figure:
    df_plot   = df.sort_values("Date")
    test_rows = df_plot[df_plot["year"] == TEST_YEAR]
    fig = go.Figure()
    if len(test_rows):
        fig.add_vrect(x0=str(test_rows["Date"].min().date()),
                      x1=str(test_rows["Date"].max().date()),
                      fillcolor="orange", opacity=0.10, line_width=0,
                      annotation_text=f"Test ({TEST_YEAR})",
                      annotation_position="top left")
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot[target],
                             mode="lines", name="Actual",
                             line=dict(color="#2171B5", width=1),
                             hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot[pred_col],
                             mode="lines", name="RF Predicted",
                             line=dict(color="#D94801", width=1),
                             hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:.2f}<extra></extra>"))
    fig.update_layout(title=title, height=320,
                      xaxis_title="Date", yaxis_title="Value",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=50, r=20, t=60, b=40),
                      plot_bgcolor="#FAFAFA")
    return fig


def importance_chart(rf, features: list, title: str) -> go.Figure:
    imps  = rf.feature_importances_
    order = np.argsort(imps)

    # Colour bars by feature group
    colours = []
    for i in order:
        f = features[i]
        if f in GRAB_INLET or f in COMP_INLET:
            colours.append("#375623")   # green — inlet
        elif f in SEC_COLS:
            colours.append("#006B6B")   # teal  — secondary
        else:
            colours.append("#2171B5")   # blue  — operational/temporal

    fig = go.Figure(go.Bar(
        x=imps[order], y=[features[i] for i in order],
        orientation="h", marker_color=colours,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    # Legend annotation
    fig.add_annotation(x=0.98, y=0.05, xref="paper", yref="paper",
                       text="<span style='color:#375623'>■</span> Inlet &nbsp;"
                            "<span style='color:#006B6B'>■</span> Secondary &nbsp;"
                            "<span style='color:#2171B5'>■</span> Operational/Temporal",
                       showarrow=False, xanchor="right", font=dict(size=10))
    fig.update_layout(title=title, height=500,
                      xaxis_title="Importance (mean decrease in impurity)",
                      margin=dict(l=210, r=20, t=50, b=40),
                      plot_bgcolor="#FAFAFA")
    return fig


def comparison_chart(df_s1, df_p1, df_p2, run) -> go.Figure:
    """3-way bar chart: Stage 1 vs Stage 2 P1 vs Stage 2 P2."""
    labels = [m[0].replace("stage2_p2_", "") for m in MODELS]

    fig = go.Figure()
    if df_s1 is not None:
        fig.add_trace(go.Bar(
            name="Stage 1 (Inlet only)",
            x=labels,
            y=[r2_for(df_s1, EQUIV[m[0]][0]) for m in MODELS],
            marker_color="#6BAED6",
            hovertemplate="%{x}<br>Stage 1 R²: %{y:.3f}<extra></extra>",
        ))
    if df_p1 is not None:
        fig.add_trace(go.Bar(
            name="Stage 2 P1 (Secondary only)",
            x=labels,
            y=[r2_for(df_p1, EQUIV[m[0]][1]) for m in MODELS],
            marker_color="#006B6B",
            hovertemplate="%{x}<br>Stage 2 P1 R²: %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Bar(
        name="Stage 2 P2 (Inlet + Secondary)",
        x=labels,
        y=[df_p2[(df_p2["model"] == m[0]) & (df_p2["run"] == run)]["RF_R2_test"].values[0]
           for m in MODELS],
        marker_color="#2E4057",
        hovertemplate="%{x}<br>Stage 2 P2 R²: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="RF Test R² — 3-Way Comparison: Stage 1 vs Stage 2 P1 vs Stage 2 P2",
        barmode="group", height=420,
        xaxis_title="Model", yaxis_title="R² (test set)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=60, r=20, t=70, b=80),
        plot_bgcolor="#FAFAFA",
    )
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def results_table_html(df_p2, df_s1, df_p1, run) -> str:
    run_data = df_p2[df_p2["run"] == run]

    def cell_r2(val):
        if val >= 0.7:   bg, fg = "#C6EFCE", "#276221"
        elif val >= 0.4: bg, fg = "#FFEB9C", "#9C5700"
        else:            bg, fg = "#FFC7CE", "#9C0006"
        return f'style="background:{bg};color:{fg};font-weight:bold;text-align:center"'

    def delta_cell(new_val, ref_val):
        if ref_val is None:
            return '<td style="text-align:center">—</td>'
        d = new_val - ref_val
        c = "#276221" if d > 0 else "#9C0006"
        s = "+" if d > 0 else ""
        return f'<td style="text-align:center;color:{c};font-weight:bold">{s}{d:.3f}</td>'

    rows_html = ""
    for _, row in run_data.iterrows():
        s1_r2 = r2_for(df_s1, EQUIV[row["model"]][0])
        p1_r2 = r2_for(df_p1, EQUIV[row["model"]][1])
        rows_html += f"""
        <tr>
          <td>{row['model']}</td>
          <td style="text-align:center">{row['RF_RMSE_train']:.3f}</td>
          <td {cell_r2(row['RF_R2_train'])}>{row['RF_R2_train']:.3f}</td>
          <td style="text-align:center">{row['RF_RMSE_test']:.3f}</td>
          <td {cell_r2(row['RF_R2_test'])}>{row['RF_R2_test']:.3f}</td>
          {delta_cell(row['RF_R2_test'], s1_r2)}
          {delta_cell(row['RF_R2_test'], p1_r2)}
          <td style="text-align:center">{row['LR_RMSE_test']:.3f}</td>
          <td {cell_r2(row['LR_R2_test'])}>{row['LR_R2_test']:.3f}</td>
        </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="2" style="background:#1F4E79">RF — Train</th>
          <th colspan="4" style="background:#2E4057">RF — Test (Stage 2 P2)</th>
          <th colspan="2" style="background:#375623">LR Baseline — Test</th>
        </tr>
        <tr>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
          <th>ΔR² vs Stage 1</th><th>ΔR² vs Stage 2 P1</th>
          <th>RMSE</th><th>R²</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="table-note">
      ΔR² positive = improvement over that stage. &nbsp;
      R²: <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥0.70</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">&lt;0.40</span>
    </p>"""


# ── Observations ───────────────────────────────────────────────────────────────

def observations_html(df_p2, df_s1, df_p1, run) -> str:
    run_data = df_p2[df_p2["run"] == run]
    best  = run_data.loc[run_data["RF_R2_test"].idxmax()]
    worst = run_data.loc[run_data["RF_R2_test"].idxmin()]
    rf_beats_lr = (run_data["RF_RMSE_test"] < run_data["LR_RMSE_test"]).sum()

    vs_p1_gains = []
    for _, row in run_data.iterrows():
        p1_r2 = r2_for(df_p1, EQUIV[row["model"]][1])
        if p1_r2 is not None:
            vs_p1_gains.append(row["RF_R2_test"] - p1_r2)

    p1_note = ""
    if vs_p1_gains:
        improved = sum(1 for g in vs_p1_gains if g > 0)
        p1_note  = (f"<li>Adding inlet features to secondary clarifier features "
                    f"improved R² in <b>{improved} of {len(vs_p1_gains)}</b> models "
                    f"vs Stage 2 Phase 1 (Secondary only). "
                    f"Average ΔR²: <b>{np.mean(vs_p1_gains):+.3f}</b>.</li>")

    return f"""
    <ul>
      {p1_note}
      <li>Best R² on test set: <b>{best['model']}</b>
          (RF R² = {best['RF_R2_test']:.3f})</li>
      <li>Worst R² on test set: <b>{worst['model']}</b>
          (RF R² = {worst['RF_R2_test']:.3f})</li>
      <li>RF outperforms LR baseline in
          <b>{rf_beats_lr} of {len(run_data)}</b> models.</li>
      <li>Feature importance charts use colour coding:
          <span style="color:#375623;font-weight:bold">green = inlet</span>,
          <span style="color:#006B6B;font-weight:bold">teal = secondary</span>,
          <span style="color:#2171B5;font-weight:bold">blue = operational/temporal</span>.
          This reveals whether the model leans on inlet or secondary features when both are available.
      </li>
    </ul>"""


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
  body { font-family:Calibri,Arial,sans-serif; margin:0; background:#F5F5F5; color:#222; }
  .container { max-width:1200px; margin:0 auto; padding:30px 24px; }
  h1 { color:#2E4057; border-bottom:3px solid #2E4057; padding-bottom:10px; }
  h2 { color:#2E4057; margin-top:48px; border-left:5px solid #2E4057; padding-left:12px; }
  h3 { color:#375623; margin-top:28px; }
  .card { background:white; border-radius:8px; padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,0.12); margin-bottom:24px; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .summary-table { border-collapse:collapse; width:100%; font-size:13px; }
  .summary-table th, .summary-table td { border:1px solid #ddd; padding:7px 12px; }
  .summary-table thead th { background:#2E4057; color:white; text-align:center; }
  .summary-table tbody tr:nth-child(even) { background:#F9F9F9; }
  .table-note { font-size:12px; color:#555; margin-top:8px; }
  .obs-card { background:#EEF0F4; border-left:5px solid #2E4057;
              border-radius:4px; padding:14px 20px; margin-bottom:20px; }
  .obs-card ul { margin:0; padding-left:20px; line-height:1.9; }
  .tag-grab { background:#375623; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .tag-comp { background:#833C00; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .meta { font-size:12px; color:#888; margin-top:6px; }
  .badge { background:#2E4057; color:white; font-size:11px;
           padding:2px 8px; border-radius:10px; margin-left:6px; }
"""


def chart_div(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def build_report(run: int) -> str:
    df_p2 = pd.read_excel(RESULTS_FILE)
    df_s1 = load_results(STAGE1_RESULTS)
    df_p1 = load_results(STAGE2P1_RESULTS)

    comp_html    = chart_div(comparison_chart(df_s1, df_p1, df_p2, run))
    summary_html = results_table_html(df_p2, df_s1, df_p1, run)
    obs_html     = observations_html(df_p2, df_s1, df_p1, run)

    sections_html = ""
    for name, target_col, tgt_name, sample_type, features in MODELS:
        pred_col  = f"predicted_RF_run_{run}"
        tag_class = "tag-grab" if sample_type == "Grab" else "tag-comp"

        df_sub = load_subset(name)
        if pred_col not in df_sub.columns:
            print(f"  WARNING: {pred_col} not found in {name}.xlsx — skipping")
            continue

        rf_path = os.path.join(MODELS_DIR, f"{name}_run_{run}.pkl")
        rf      = joblib.load(rf_path)
        row     = df_p2[(df_p2["model"] == name) & (df_p2["run"] == run)].iloc[0]

        # ΔR² vs P1 callout
        p1_r2  = r2_for(df_p1, EQUIV[name][1])
        p1_note = ""
        if p1_r2 is not None:
            d = row["RF_R2_test"] - p1_r2
            c = "#276221" if d > 0 else "#9C0006"
            s = "+" if d > 0 else ""
            p1_note = (f'&nbsp;|&nbsp; ΔR² vs P1: '
                       f'<b style="color:{c}">{s}{d:.3f}</b>')

        sections_html += f"""
        <div class="card">
          <h3>{tgt_name} — {sample_type}
            <span class="{tag_class}">{sample_type}</span>
            <span class="badge">Stage 2 Phase 2</span>
          </h3>
          <p class="meta">
            Train rows: {int(df_sub['year'].isin(TRAIN_YEARS).sum())} &nbsp;|&nbsp;
            Test rows: {int((df_sub['year'] == TEST_YEAR).sum())} &nbsp;|&nbsp;
            RF train RMSE: {row['RF_RMSE_train']:.3f} &nbsp;|&nbsp;
            RF train R²: {row['RF_R2_train']:.3f} &nbsp;|&nbsp;
            RF test RMSE: <b>{row['RF_RMSE_test']:.3f}</b> &nbsp;|&nbsp;
            RF test R²: <b>{row['RF_R2_test']:.3f}</b>{p1_note}
          </p>
          <div class="grid-2">
            <div>{chart_div(scatter_chart(df_sub, target_col, pred_col,
                 f"Actual vs Predicted — {tgt_name} {sample_type} (Test {TEST_YEAR})"))}</div>
            <div>{chart_div(importance_chart(rf, features,
                 f"Feature Importances — {tgt_name} {sample_type}"))}</div>
          </div>
          {chart_div(timeseries_chart(df_sub, target_col, pred_col,
               f"Time Series — {tgt_name} {sample_type}"))}
        </div>"""

        sections_html += f'\n        <div style="margin-top:4px"></div>'

        if tgt_name == "pH" and sample_type == "Composite":
            pass  # last model, no divider needed
        elif sample_type == "Composite" and tgt_name != "pH":
            sections_html += ""
        elif sample_type == "Grab":
            pass

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 2 Phase 2 Report — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>Stage 2 — Phase 2 Report
    <span style="font-size:14px;font-weight:normal;color:#555">
      Inlet + Secondary clarifier features → Effluent targets
    </span>
  </h1>
  <p class="meta">
    Features: Inlet (grab/composite) + Secondary Clarifier + Sec Sed &nbsp;|&nbsp;
    Train: {', '.join(str(y) for y in TRAIN_YEARS)} &nbsp;|&nbsp;
    Test: {TEST_YEAR} &nbsp;|&nbsp; Run: <b>{run}</b>
  </p>

  <h2>3-Way Comparison: Stage 1 vs Stage 2 P1 vs Stage 2 P2</h2>
  <div class="card">{comp_html}</div>

  <h2>Summary Table</h2>
  <div class="card">{summary_html}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{obs_html}</div>

  <h2>Model Detail</h2>
  {sections_html}
</div>
</body>
</html>"""


def main():
    run = get_latest_run()
    print(f"Generating Stage 2 Phase 2 report for run {run}...")
    html     = build_report(run)
    out_path = os.path.join(SCRIPT_DIR, f"report_stage2_phase2_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
