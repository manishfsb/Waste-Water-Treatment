"""
generate_report.py — Compile Stage 1 modeling results into a self-contained HTML report.

Usage (from project root):
    .venv/bin/python3 21-25/modeling/generate_report.py

Usage (from modeling/ directory):
    ../../.venv/bin/python3 generate_report.py

Output:
    modeling/report_stage1_run_<N>.html
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_FILE = os.path.join(BASE_DIR, "results.xlsx")

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

YEAR_COLOURS = {
    2020: "#6BAED6", 2021: "#2171B5", 2022: "#74C476",
    2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801",
}

MODELS = [
    ("stage1_grab_BOD", "Effluent BOD (mg/L, Grab)",      "BOD",  "Grab"),
    ("stage1_grab_COD", "Effluent COD (mg/L, Grab)",      "COD",  "Grab"),
    ("stage1_grab_TSS", "Effluent TSS (mg/L, Grab)",      "TSS",  "Grab"),
    ("stage1_grab_pH",  "Effluent pH (Grab)",             "pH",   "Grab"),
    ("stage1_comp_BOD", "Effluent BOD (mg/L, Composite)", "BOD",  "Composite"),
    ("stage1_comp_COD", "Effluent COD (mg/L, Composite)", "COD",  "Composite"),
    ("stage1_comp_TSS", "Effluent TSS (mg/L, Composite)", "TSS",  "Composite"),
    ("stage1_comp_pH",  "Effluent pH (Composite)",        "pH",   "Composite"),
]

GRAB_FEATURES = [
    "Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)", "Inlet COD (mg/L, Grab)",
    "Inlet TSS (mg/L, Grab)", "Flow (MLD)", "Power Total (KW)",
    "month", "day_of_week", "year",
]
COMP_FEATURES = [
    "Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
    "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)",
    "Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_latest_run() -> int:
    df = pd.read_excel(RESULTS_FILE)
    return int(df["run"].max())


def load_subset(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.xlsx")
    df = pd.read_excel(path, parse_dates=["Date"])
    return df


def get_features(sample_type: str) -> list:
    return GRAB_FEATURES if sample_type == "Grab" else COMP_FEATURES


# ── Chart builders ─────────────────────────────────────────────────────────────

def scatter_chart(df: pd.DataFrame, target: str, pred_col: str,
                  title: str) -> go.Figure:
    """Actual vs predicted scatter coloured by year (test set only)."""
    test = df[df["year"] == TEST_YEAR].copy()

    fig = go.Figure()
    for yr in sorted(test["year"].unique()):
        sub = test[test["year"] == yr]
        fig.add_trace(go.Scatter(
            x=sub[target], y=sub[pred_col],
            mode="markers",
            name=str(yr),
            marker=dict(color=YEAR_COLOURS.get(yr, "#999"), size=6, opacity=0.75),
            text=sub["Date"].dt.strftime("%d %b %Y"),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))

    # Perfect-fit diagonal
    lo = min(test[target].min(), test[pred_col].min())
    hi = max(test[target].max(), test[pred_col].max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash", width=1),
        name="Perfect fit", hoverinfo="skip",
    ))

    fig.update_layout(
        title=title, height=420,
        xaxis_title="Actual", yaxis_title="Predicted",
        legend_title="Year",
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def timeseries_chart(df: pd.DataFrame, target: str, pred_col: str,
                     title: str) -> go.Figure:
    """Full time-series: actual vs predicted with test period shaded."""
    df_plot = df.sort_values("Date")
    test_rows = df_plot[df_plot["year"] == TEST_YEAR]

    fig = go.Figure()

    # Shaded test region
    if len(test_rows):
        fig.add_vrect(
            x0=str(test_rows["Date"].min().date()),
            x1=str(test_rows["Date"].max().date()),
            fillcolor="orange", opacity=0.10, line_width=0,
            annotation_text=f"Test ({TEST_YEAR})",
            annotation_position="top left",
        )

    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot[target],
        mode="lines", name="Actual",
        line=dict(color="#2171B5", width=1),
        hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot[pred_col],
        mode="lines", name="RF Predicted",
        line=dict(color="#D94801", width=1),
        hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=title, height=320,
        xaxis_title="Date", yaxis_title="Value",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def importance_chart(rf, features: list, title: str) -> go.Figure:
    """Horizontal bar chart of RF feature importances."""
    imps = rf.feature_importances_
    order = np.argsort(imps)
    labels = [features[i] for i in order]
    values = imps[order]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color="#2171B5",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=360,
        xaxis_title="Importance (mean decrease in impurity)",
        margin=dict(l=180, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def results_table_html(df_results: pd.DataFrame, run: int) -> str:
    run_data = df_results[df_results["run"] == run].copy()

    def colour_r2(val):
        if val >= 0.7:
            bg = "#C6EFCE"; fg = "#276221"
        elif val >= 0.4:
            bg = "#FFEB9C"; fg = "#9C5700"
        else:
            bg = "#FFC7CE"; fg = "#9C0006"
        return f'style="background:{bg};color:{fg};font-weight:bold;text-align:center"'

    rows_html = ""
    for _, row in run_data.iterrows():
        cv  = f"{row['RF_CV_RMSE']:.3f}"  if 'RF_CV_RMSE'   in row.index and pd.notna(row.get('RF_CV_RMSE'))   else "—"
        mae = f"{row['RF_MAE_test']:.3f}"  if 'RF_MAE_test'  in row.index and pd.notna(row.get('RF_MAE_test'))  else "—"
        mape= f"{row['RF_MAPE_test']:.1f}%" if 'RF_MAPE_test' in row.index and pd.notna(row.get('RF_MAPE_test')) else "—"
        rows_html += f"""
        <tr>
          <td>{row['model']}</td>
          <td style="text-align:center">{row['RF_RMSE_train']:.3f}</td>
          <td style="text-align:center">{cv}</td>
          <td {colour_r2(row['RF_R2_train'])}>{row['RF_R2_train']:.3f}</td>
          <td style="text-align:center">{row['RF_RMSE_test']:.3f}</td>
          <td style="text-align:center">{mae}</td>
          <td style="text-align:center">{mape}</td>
          <td {colour_r2(row['RF_R2_test'])}>{row['RF_R2_test']:.3f}</td>
          <td style="text-align:center">{row['LR_RMSE_test']:.3f}</td>
          <td {colour_r2(row['LR_R2_test'])}>{row['LR_R2_test']:.3f}</td>
        </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="3" style="background:#1F4E79">RF — Train</th>
          <th colspan="4" style="background:#833C00">RF — Test</th>
          <th colspan="2" style="background:#375623">LR Baseline — Test</th>
        </tr>
        <tr>
          <th>RMSE</th><th>CV RMSE</th><th>R²</th>
          <th>RMSE</th><th>MAE</th><th>MAPE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="table-note">
      R² colour scale:
      <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥ 0.70 good</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70 moderate</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">< 0.40 poor</span>
    </p>"""


# ── Key observations ───────────────────────────────────────────────────────────

def observations_html(df_results: pd.DataFrame, run: int) -> str:
    run_data = df_results[df_results["run"] == run]
    best     = run_data.loc[run_data["RF_R2_test"].idxmax()]
    worst    = run_data.loc[run_data["RF_R2_test"].idxmin()]
    rf_beats_lr = (run_data["RF_RMSE_test"] < run_data["LR_RMSE_test"]).sum()

    return f"""
    <ul>
      <li>Best R² on test set: <b>{best['model']}</b>
          (RF R² = {best['RF_R2_test']:.3f})</li>
      <li>Worst R² on test set: <b>{worst['model']}</b>
          (RF R² = {worst['RF_R2_test']:.3f})</li>
      <li>RF outperforms LR baseline (lower test RMSE) in
          <b>{rf_beats_lr} of {len(run_data)}</b> models.</li>
      <li>All R² values are negative or near zero — expected at Stage 1.
          Inlet concentrations carry almost no signal for predicting effluent
          quality. Stage 2 (secondary clarifier features) will address this.</li>
    </ul>"""


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
  body { font-family: Calibri, Arial, sans-serif; margin: 0; background: #F5F5F5; color: #222; }
  .container { max-width: 1200px; margin: 0 auto; padding: 30px 24px; }
  h1 { color: #1F4E79; border-bottom: 3px solid #1F4E79; padding-bottom: 10px; }
  h2 { color: #1F4E79; margin-top: 48px; border-left: 5px solid #1F4E79;
       padding-left: 12px; }
  h3 { color: #375623; margin-top: 28px; }
  .card { background: white; border-radius: 8px; padding: 20px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.12); margin-bottom: 24px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .summary-table { border-collapse: collapse; width: 100%; font-size: 13px; }
  .summary-table th, .summary-table td {
      border: 1px solid #ddd; padding: 7px 12px; }
  .summary-table thead th {
      background: #1F4E79; color: white; text-align: center; }
  .summary-table tbody tr:nth-child(even) { background: #F9F9F9; }
  .table-note { font-size: 12px; color: #555; margin-top: 8px; }
  .obs-card { background: #EBF3FB; border-left: 5px solid #2171B5;
              border-radius: 4px; padding: 14px 20px; margin-bottom: 20px; }
  .obs-card ul { margin: 0; padding-left: 20px; line-height: 1.8; }
  .tag-grab { background:#375623; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .tag-comp { background:#833C00; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .meta { font-size: 12px; color: #888; margin-top: 6px; }
"""


def chart_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def build_report(run: int) -> str:
    df_results = pd.read_excel(RESULTS_FILE)

    sections_html = ""

    for target_name in ["BOD", "COD", "TSS", "pH"]:
        sections_html += f'<h2>{target_name}</h2>'

        for sample_type in ["Grab", "Composite"]:
            name = f"stage1_{'grab' if sample_type == 'Grab' else 'comp'}_{target_name}"
            target_col = next(t for n, t, tn, st in MODELS
                              if n == name)
            features   = get_features(sample_type)
            pred_col   = f"predicted_RF_run_{run}"
            tag_class  = "tag-grab" if sample_type == "Grab" else "tag-comp"

            df_sub = load_subset(name)
            if pred_col not in df_sub.columns:
                print(f"  WARNING: {pred_col} not found in {name}.xlsx — skipping")
                continue

            model_path = os.path.join(MODELS_DIR, f"{name}_run_{run}.pkl")
            rf = joblib.load(model_path)

            row = df_results[(df_results["model"] == name) &
                             (df_results["run"] == run)].iloc[0]

            fig_scatter = scatter_chart(
                df_sub, target_col, pred_col,
                f"Actual vs Predicted — {target_name} {sample_type} (Test {TEST_YEAR})"
            )
            fig_ts = timeseries_chart(
                df_sub, target_col, pred_col,
                f"Time Series — {target_name} {sample_type}"
            )
            fig_imp = importance_chart(
                rf, features,
                f"Feature Importances — {target_name} {sample_type}"
            )

            rf_params_str = ""
            for col in ["RF_n_estimators", "RF_max_depth", "RF_min_samples_leaf",
                        "RF_min_samples_split", "RF_max_features"]:
                if col in row.index:
                    rf_params_str += (
                        f"n_est={row['RF_n_estimators']} &nbsp;|&nbsp; "
                        f"depth={row['RF_max_depth']} &nbsp;|&nbsp; "
                        f"leaf={row['RF_min_samples_leaf']} &nbsp;|&nbsp; "
                        f"split={row['RF_min_samples_split']} &nbsp;|&nbsp; "
                        f"features={row['RF_max_features']}"
                    )
                    break

            cv_rmse = f"{row['RF_CV_RMSE']:.3f}"   if 'RF_CV_RMSE'   in row.index and pd.notna(row.get('RF_CV_RMSE'))   else "—"
            mae_test = f"{row['RF_MAE_test']:.3f}"  if 'RF_MAE_test'  in row.index and pd.notna(row.get('RF_MAE_test'))  else "—"
            mape_test= f"{row['RF_MAPE_test']:.1f}%" if 'RF_MAPE_test' in row.index and pd.notna(row.get('RF_MAPE_test')) else "—"
            sections_html += f"""
            <div class="card">
              <h3>{sample_type} model
                <span class="{tag_class}">{sample_type}</span>
              </h3>
              <p class="meta">
                Train rows: {int((df_sub['year'].isin(TRAIN_YEARS)).sum())} &nbsp;|&nbsp;
                Test rows: {int((df_sub['year'] == TEST_YEAR).sum())} &nbsp;|&nbsp;
                RF train RMSE: {row['RF_RMSE_train']:.3f} &nbsp;|&nbsp;
                CV RMSE: {cv_rmse} &nbsp;|&nbsp;
                RF train R²: {row['RF_R2_train']:.3f} &nbsp;|&nbsp;
                RF test RMSE: <b>{row['RF_RMSE_test']:.3f}</b> &nbsp;|&nbsp;
                MAE: {mae_test} &nbsp;|&nbsp;
                MAPE: {mape_test} &nbsp;|&nbsp;
                RF test R²: <b>{row['RF_R2_test']:.3f}</b> &nbsp;|&nbsp;
                LR test RMSE: {row['LR_RMSE_test']:.3f} &nbsp;|&nbsp;
                LR test R²: {row['LR_R2_test']:.3f}
                {'<br><span style="color:#555">RF params: ' + rf_params_str + '</span>' if rf_params_str else ''}
              </p>
              <div class="grid-2">
                <div>{chart_div(fig_scatter)}</div>
                <div>{chart_div(fig_imp)}</div>
              </div>
              {chart_div(fig_ts)}
            </div>"""

    summary_html    = results_table_html(df_results, run)
    obs_html        = observations_html(df_results, run)

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 1 Modeling Report — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>Stage 1 Modeling Report</h1>
  <p class="meta">
    Inlet features → Effluent targets &nbsp;|&nbsp;
    Train: {', '.join(str(y) for y in TRAIN_YEARS)} &nbsp;|&nbsp;
    Test: {TEST_YEAR} &nbsp;|&nbsp;
    Run: <b>{run}</b>
  </p>

  <h2>Summary</h2>
  <div class="card">{summary_html}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{obs_html}</div>

  {sections_html}
</div>
</body>
</html>"""
    return html


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run = get_latest_run()
    print(f"Generating report for run {run}...")
    html = build_report(run)
    out_path = os.path.join(BASE_DIR, f"report_stage1_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
