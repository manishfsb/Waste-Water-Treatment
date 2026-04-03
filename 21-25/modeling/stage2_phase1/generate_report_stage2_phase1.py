"""
generate_report_stage2_phase1.py
Compile Stage 2 Phase 1 results into a self-contained HTML report.

Usage (from project root New_Project/):
    .venv/bin/python3 21-25/modeling/stage2_phase1/generate_report_stage2_phase1.py

Usage (from stage2_phase1/ directory):
    ../../../.venv/bin/python3 generate_report_stage2_phase1.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR    = os.path.join(SCRIPT_DIR, "models")
RESULTS_FILE  = os.path.join(SCRIPT_DIR, "results.xlsx")

# Stage 1 results for comparison
STAGE1_RESULTS = os.path.join(os.path.dirname(SCRIPT_DIR), "results.xlsx")

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

YEAR_COLOURS = {
    2021: "#2171B5", 2022: "#74C476",
    2023: "#238B45", 2024: "#FD8D3C", 2025: "#D94801",
}

SEC_FEATURES = [
    "Sec Clarifier pH", "Sec Clarifier TSS (mg/L)", "Sec Clarifier BOD (mg/L)",
    "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS",
    "Sec Sed pH", "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)",
    "Sec Sed COD (mg/L)", "Sec Sed RAS (New)",
    "Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year",
]

MODELS = [
    ("stage2_p1_grab_BOD", "Effluent BOD (mg/L, Grab)",      "BOD", "Grab"),
    ("stage2_p1_grab_COD", "Effluent COD (mg/L, Grab)",      "COD", "Grab"),
    ("stage2_p1_grab_TSS", "Effluent TSS (mg/L, Grab)",      "TSS", "Grab"),
    ("stage2_p1_grab_pH",  "Effluent pH (Grab)",             "pH",  "Grab"),
    ("stage2_p1_comp_BOD", "Effluent BOD (mg/L, Composite)", "BOD", "Composite"),
    ("stage2_p1_comp_COD", "Effluent COD (mg/L, Composite)", "COD", "Composite"),
    ("stage2_p1_comp_TSS", "Effluent TSS (mg/L, Composite)", "TSS", "Composite"),
    ("stage2_p1_comp_pH",  "Effluent pH (Composite)",        "pH",  "Composite"),
]

# Mapping Stage 2 model names to their Stage 1 counterparts for comparison
STAGE1_EQUIV = {
    "stage2_p1_grab_BOD": "stage1_grab_BOD",
    "stage2_p1_grab_COD": "stage1_grab_COD",
    "stage2_p1_grab_TSS": "stage1_grab_TSS",
    "stage2_p1_grab_pH":  "stage1_grab_pH",
    "stage2_p1_comp_BOD": "stage1_comp_BOD",
    "stage2_p1_comp_COD": "stage1_comp_COD",
    "stage2_p1_comp_TSS": "stage1_comp_TSS",
    "stage2_p1_comp_pH":  "stage1_comp_pH",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_latest_run() -> int:
    return int(pd.read_excel(RESULTS_FILE)["run"].max())


def load_subset(name: str) -> pd.DataFrame:
    return pd.read_excel(os.path.join(DATA_DIR, f"{name}.xlsx"), parse_dates=["Date"])


def load_stage1_results() -> pd.DataFrame | None:
    if os.path.exists(STAGE1_RESULTS):
        df = pd.read_excel(STAGE1_RESULTS)
        return df[df["run"] == df["run"].max()]
    return None


# ── Chart builders ─────────────────────────────────────────────────────────────

def scatter_chart(df: pd.DataFrame, target: str, pred_col: str,
                  title: str) -> go.Figure:
    test = df[df["year"] == TEST_YEAR].copy()
    fig  = go.Figure()
    for yr in sorted(test["year"].unique()):
        sub = test[test["year"] == yr]
        fig.add_trace(go.Scatter(
            x=sub[target], y=sub[pred_col], mode="markers",
            name=str(yr),
            marker=dict(color=YEAR_COLOURS.get(yr, "#999"), size=6, opacity=0.75),
            text=sub["Date"].dt.strftime("%d %b %Y"),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
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
        legend_title="Year", margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def timeseries_chart(df: pd.DataFrame, target: str, pred_col: str,
                     title: str) -> go.Figure:
    df_plot   = df.sort_values("Date")
    test_rows = df_plot[df_plot["year"] == TEST_YEAR]
    fig = go.Figure()
    if len(test_rows):
        fig.add_vrect(
            x0=str(test_rows["Date"].min().date()),
            x1=str(test_rows["Date"].max().date()),
            fillcolor="orange", opacity=0.10, line_width=0,
            annotation_text=f"Test ({TEST_YEAR})", annotation_position="top left",
        )
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot[target], mode="lines",
        name="Actual", line=dict(color="#2171B5", width=1),
        hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot[pred_col], mode="lines",
        name="RF Predicted", line=dict(color="#D94801", width=1),
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


def importance_chart(rf, title: str) -> go.Figure:
    imps  = rf.feature_importances_
    order = np.argsort(imps)
    fig   = go.Figure(go.Bar(
        x=imps[order], y=[SEC_FEATURES[i] for i in order],
        orientation="h", marker_color="#006B6B",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=420,
        xaxis_title="Importance (mean decrease in impurity)",
        margin=dict(l=200, r=20, t=50, b=40),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def comparison_chart(df_s1: pd.DataFrame | None,
                     df_s2: pd.DataFrame, run: int) -> go.Figure:
    """Bar chart comparing RF test R² across Stage 1 and Stage 2 Phase 1."""
    model_labels = [m[0].replace("stage2_p1_", "") for m in MODELS]
    s2_r2 = [
        df_s2[(df_s2["model"] == m[0]) & (df_s2["run"] == run)]["RF_R2_test"].values[0]
        for m in MODELS
    ]

    fig = go.Figure()
    if df_s1 is not None:
        s1_r2 = []
        for m in MODELS:
            equiv = STAGE1_EQUIV[m[0]]
            rows  = df_s1[df_s1["model"] == equiv]["RF_R2_test"].values
            s1_r2.append(rows[0] if len(rows) else None)

        fig.add_trace(go.Bar(
            name="Stage 1 (Inlet)", x=model_labels, y=s1_r2,
            marker_color="#6BAED6",
            hovertemplate="%{x}<br>Stage 1 R²: %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Bar(
        name="Stage 2 P1 (Secondary)", x=model_labels, y=s2_r2,
        marker_color="#006B6B",
        hovertemplate="%{x}<br>Stage 2 P1 R²: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="RF Test R² — Stage 1 vs Stage 2 Phase 1",
        barmode="group", height=400,
        xaxis_title="Model", yaxis_title="R² (test set)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=60, r=20, t=60, b=80),
        plot_bgcolor="#FAFAFA",
    )
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def results_table_html(df_s2: pd.DataFrame, df_s1: pd.DataFrame | None,
                       run: int) -> str:
    run_data = df_s2[df_s2["run"] == run]

    def cell_r2(val):
        if val >= 0.7:   bg, fg = "#C6EFCE", "#276221"
        elif val >= 0.4: bg, fg = "#FFEB9C", "#9C5700"
        else:            bg, fg = "#FFC7CE", "#9C0006"
        return f'style="background:{bg};color:{fg};font-weight:bold;text-align:center"'

    def delta_cell(s2_val, s1_val):
        if s1_val is None:
            return '<td style="text-align:center">—</td>'
        delta = s2_val - s1_val
        colour = "#276221" if delta > 0 else "#9C0006"
        sign   = "+" if delta > 0 else ""
        return (f'<td style="text-align:center;color:{colour};font-weight:bold">'
                f'{sign}{delta:.3f}</td>')

    rows_html = ""
    for _, row in run_data.iterrows():
        s1_r2 = None
        if df_s1 is not None:
            equiv = STAGE1_EQUIV.get(row["model"])
            match = df_s1[df_s1["model"] == equiv]["RF_R2_test"].values
            s1_r2 = float(match[0]) if len(match) else None

        cv   = f"{row['RF_CV_RMSE']:.3f}"   if 'RF_CV_RMSE'   in row.index and pd.notna(row.get('RF_CV_RMSE'))   else "—"
        mae  = f"{row['RF_MAE_test']:.3f}"   if 'RF_MAE_test'  in row.index and pd.notna(row.get('RF_MAE_test'))  else "—"
        mape = f"{row['RF_MAPE_test']:.1f}%" if 'RF_MAPE_test' in row.index and pd.notna(row.get('RF_MAPE_test')) else "—"

        ridge_rmse = f"{row['Ridge_RMSE_test']:.3f}" if 'Ridge_RMSE_test' in row.index and pd.notna(row.get('Ridge_RMSE_test')) else "—"
        ridge_r2_v = row['Ridge_R2_test'] if 'Ridge_R2_test' in row.index and pd.notna(row.get('Ridge_R2_test')) else None
        ridge_r2   = cell_r2(ridge_r2_v) if ridge_r2_v is not None else 'style="text-align:center;color:#aaa"'
        ridge_r2_s = f"{ridge_r2_v:.3f}" if ridge_r2_v is not None else "—"
        rows_html += f"""
        <tr>
          <td>{row['model']}</td>
          <td style="text-align:center">{row['RF_RMSE_train']:.3f}</td>
          <td style="text-align:center">{cv}</td>
          <td {cell_r2(row['RF_R2_train'])}>{row['RF_R2_train']:.3f}</td>
          <td style="text-align:center">{row['RF_RMSE_test']:.3f}</td>
          <td style="text-align:center">{mae}</td>
          <td style="text-align:center">{mape}</td>
          <td {cell_r2(row['RF_R2_test'])}>{row['RF_R2_test']:.3f}</td>
          {delta_cell(row['RF_R2_test'], s1_r2)}
          <td style="text-align:center">{row['LR_RMSE_test']:.3f}</td>
          <td {cell_r2(row['LR_R2_test'])}>{row['LR_R2_test']:.3f}</td>
          <td style="text-align:center">{ridge_rmse}</td>
          <td {ridge_r2}>{ridge_r2_s}</td>
        </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="3" style="background:#1F4E79">RF — Train</th>
          <th colspan="5" style="background:#006B6B">RF — Test (Stage 2 P1)</th>
          <th colspan="2" style="background:#375623">LR Baseline — Test</th>
          <th colspan="2" style="background:#4292C6">Ridge — Test</th>
        </tr>
        <tr>
          <th>RMSE</th><th>CV RMSE</th><th>R²</th>
          <th>RMSE</th><th>MAE</th><th>MAPE</th><th>R²</th><th>ΔR² vs Stage 1</th>
          <th>RMSE</th><th>R²</th>
          <th>RMSE</th><th>R²</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="table-note">
      ΔR² = Stage 2 R² minus Stage 1 R² — positive means improvement.
      R² colour: <span style="background:#C6EFCE;color:#276221;padding:2px 6px;border-radius:3px">≥0.70</span>
      <span style="background:#FFEB9C;color:#9C5700;padding:2px 6px;border-radius:3px;margin:0 4px">0.40–0.70</span>
      <span style="background:#FFC7CE;color:#9C0006;padding:2px 6px;border-radius:3px">&lt;0.40</span>
    </p>"""


# ── Key observations ───────────────────────────────────────────────────────────

def observations_html(df_s2: pd.DataFrame, df_s1: pd.DataFrame | None,
                      run: int) -> str:
    run_data  = df_s2[df_s2["run"] == run]
    best      = run_data.loc[run_data["RF_R2_test"].idxmax()]
    worst     = run_data.loc[run_data["RF_R2_test"].idxmin()]
    rf_beats_lr     = (run_data["RF_RMSE_test"] < run_data["LR_RMSE_test"]).sum()
    rf_beats_ridge  = (run_data["RF_RMSE_test"] < run_data["Ridge_RMSE_test"]).sum() if "Ridge_RMSE_test" in run_data.columns else None

    improvement = ""
    if df_s1 is not None:
        gains = []
        for _, row in run_data.iterrows():
            equiv = STAGE1_EQUIV.get(row["model"])
            match = df_s1[df_s1["model"] == equiv]["RF_R2_test"].values
            if len(match):
                gains.append(row["RF_R2_test"] - float(match[0]))
        if gains:
            avg_gain = np.mean(gains)
            improved = sum(1 for g in gains if g > 0)
            improvement = (f"<li>Replacing inlet with secondary clarifier features improved "
                           f"RF test R² in <b>{improved} of {len(gains)}</b> models. "
                           f"Average ΔR²: <b>{avg_gain:+.3f}</b>.</li>")

    return f"""
    <ul>
      {improvement}
      <li>Best R² on test set: <b>{best['model']}</b>
          (RF R² = {best['RF_R2_test']:.3f})</li>
      <li>Worst R² on test set: <b>{worst['model']}</b>
          (RF R² = {worst['RF_R2_test']:.3f})</li>
      <li>RF outperforms LR baseline in
          <b>{rf_beats_lr} of {len(run_data)}</b> models.</li>
      {f"<li>RF outperforms Ridge (regularised linear) in <b>{rf_beats_ridge} of {len(run_data)}</b> models.</li>" if rf_beats_ridge is not None else ""}
    </ul>"""


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
  body { font-family: Calibri, Arial, sans-serif; margin:0; background:#F5F5F5; color:#222; }
  .container { max-width:1200px; margin:0 auto; padding:30px 24px; }
  h1 { color:#006B6B; border-bottom:3px solid #006B6B; padding-bottom:10px; }
  h2 { color:#006B6B; margin-top:48px; border-left:5px solid #006B6B; padding-left:12px; }
  h3 { color:#375623; margin-top:28px; }
  .card { background:white; border-radius:8px; padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,0.12); margin-bottom:24px; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .summary-table { border-collapse:collapse; width:100%; font-size:13px; }
  .summary-table th, .summary-table td { border:1px solid #ddd; padding:7px 12px; }
  .summary-table thead th { background:#006B6B; color:white; text-align:center; }
  .summary-table tbody tr:nth-child(even) { background:#F9F9F9; }
  .table-note { font-size:12px; color:#555; margin-top:8px; }
  .obs-card { background:#E8F5F5; border-left:5px solid #006B6B;
              border-radius:4px; padding:14px 20px; margin-bottom:20px; }
  .obs-card ul { margin:0; padding-left:20px; line-height:1.8; }
  .tag-grab { background:#375623; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .tag-comp { background:#833C00; color:white; font-size:11px;
              padding:2px 8px; border-radius:10px; margin-left:8px; }
  .meta { font-size:12px; color:#888; margin-top:6px; }
  .badge { background:#006B6B; color:white; font-size:11px;
           padding:2px 8px; border-radius:10px; margin-left:6px; }
"""


def chart_div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def build_report(run: int) -> str:
    df_s2 = pd.read_excel(RESULTS_FILE)
    df_s1 = load_stage1_results()

    # Stage comparison chart
    comp_chart_html = chart_div(comparison_chart(df_s1, df_s2, run))

    sections_html = ""
    for target_name in ["BOD", "COD", "TSS", "pH"]:
        sections_html += f'<h2>{target_name}</h2>'

        for sample_type in ["Grab", "Composite"]:
            pfx  = "grab" if sample_type == "Grab" else "comp"
            name = f"stage2_p1_{pfx}_{target_name}"
            target_col = next(t for n, t, tn, st in MODELS if n == name)
            pred_col   = f"predicted_RF_run_{run}"
            tag_class  = "tag-grab" if sample_type == "Grab" else "tag-comp"

            df_sub = load_subset(name)
            if pred_col not in df_sub.columns:
                print(f"  WARNING: {pred_col} not found in {name}.xlsx — skipping")
                continue

            rf_path = os.path.join(MODELS_DIR, f"{name}_run_{run}.pkl")
            rf      = joblib.load(rf_path)
            row     = df_s2[(df_s2["model"] == name) & (df_s2["run"] == run)].iloc[0]

            # Stage 1 comparison for this model
            s1_note = ""
            if df_s1 is not None:
                equiv = STAGE1_EQUIV.get(name)
                s1_match = df_s1[df_s1["model"] == equiv]["RF_R2_test"].values
                if len(s1_match):
                    delta = row["RF_R2_test"] - float(s1_match[0])
                    colour = "#276221" if delta > 0 else "#9C0006"
                    sign   = "+" if delta > 0 else ""
                    s1_note = (f'&nbsp;|&nbsp; ΔR² vs Stage 1: '
                               f'<b style="color:{colour}">{sign}{delta:.3f}</b>')

            fig_scatter = scatter_chart(
                df_sub, target_col, pred_col,
                f"Actual vs Predicted — {target_name} {sample_type} (Test {TEST_YEAR})"
            )
            fig_ts  = timeseries_chart(
                df_sub, target_col, pred_col,
                f"Time Series — {target_name} {sample_type}"
            )
            fig_imp = importance_chart(
                rf, f"Feature Importances — {target_name} {sample_type}"
            )

            rf_params_str = ""
            for col in ["RF_n_estimators", "RF_max_depth", "RF_min_samples_leaf",
                        "RF_min_samples_split", "RF_max_features"]:
                if col in row.index:
                    rf_params_str = (
                        f"n_est={row['RF_n_estimators']} &nbsp;|&nbsp; "
                        f"depth={row['RF_max_depth']} &nbsp;|&nbsp; "
                        f"leaf={row['RF_min_samples_leaf']} &nbsp;|&nbsp; "
                        f"split={row['RF_min_samples_split']} &nbsp;|&nbsp; "
                        f"features={row['RF_max_features']}"
                    )
                    break

            cv_rmse    = f"{row['RF_CV_RMSE']:.3f}"    if 'RF_CV_RMSE'     in row.index and pd.notna(row.get('RF_CV_RMSE'))     else "—"
            mae_test   = f"{row['RF_MAE_test']:.3f}"   if 'RF_MAE_test'    in row.index and pd.notna(row.get('RF_MAE_test'))    else "—"
            mape_test  = f"{row['RF_MAPE_test']:.1f}%" if 'RF_MAPE_test'   in row.index and pd.notna(row.get('RF_MAPE_test'))   else "—"
            ridge_rmse = f"{row['Ridge_RMSE_test']:.3f}" if 'Ridge_RMSE_test' in row.index and pd.notna(row.get('Ridge_RMSE_test')) else "—"
            ridge_r2   = f"{row['Ridge_R2_test']:.3f}"   if 'Ridge_R2_test'   in row.index and pd.notna(row.get('Ridge_R2_test'))   else "—"
            sections_html += f"""
            <div class="card">
              <h3>{sample_type} model
                <span class="{tag_class}">{sample_type}</span>
                <span class="badge">Stage 2 Phase 1</span>
              </h3>
              <p class="meta">
                Train rows: {int(df_sub['year'].isin(TRAIN_YEARS).sum())} &nbsp;|&nbsp;
                Test rows: {int((df_sub['year'] == TEST_YEAR).sum())} &nbsp;|&nbsp;
                RF train RMSE: {row['RF_RMSE_train']:.3f} &nbsp;|&nbsp;
                CV RMSE: {cv_rmse} &nbsp;|&nbsp;
                RF train R²: {row['RF_R2_train']:.3f} &nbsp;|&nbsp;
                RF test RMSE: <b>{row['RF_RMSE_test']:.3f}</b> &nbsp;|&nbsp;
                MAE: {mae_test} &nbsp;|&nbsp;
                MAPE: {mape_test} &nbsp;|&nbsp;
                RF test R²: <b>{row['RF_R2_test']:.3f}</b>
                {s1_note}
                &nbsp;|&nbsp; Ridge test RMSE: {ridge_rmse} &nbsp;|&nbsp; Ridge test R²: {ridge_r2}
                {'<br><span style="color:#555">RF params: ' + rf_params_str + '</span>' if rf_params_str else ''}
              </p>
              <div class="grid-2">
                <div>{chart_div(fig_scatter)}</div>
                <div>{chart_div(fig_imp)}</div>
              </div>
              {chart_div(fig_ts)}
            </div>"""

    summary_html = results_table_html(df_s2, df_s1, run)
    obs_html     = observations_html(df_s2, df_s1, run)
    plotly_cdn   = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 2 Phase 1 Report — Run {run}</title>
  {plotly_cdn}
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>Stage 2 — Phase 1 Report
    <span style="font-size:14px;font-weight:normal;color:#555">
      Secondary clarifier features → Effluent targets
    </span>
  </h1>
  <p class="meta">
    Features: Secondary Clarifier + Sec Sed (replaces inlet) &nbsp;|&nbsp;
    Train: {', '.join(str(y) for y in TRAIN_YEARS)} &nbsp;|&nbsp;
    Test: {TEST_YEAR} &nbsp;|&nbsp;
    Run: <b>{run}</b>
  </p>

  <h2>Stage 1 vs Stage 2 Phase 1 — R² Comparison</h2>
  <div class="card">{comp_chart_html}</div>

  <h2>Summary Table</h2>
  <div class="card">{summary_html}</div>

  <h2>Key Observations</h2>
  <div class="obs-card">{obs_html}</div>

  {sections_html}
</div>
</body>
</html>"""


def main():
    run = get_latest_run()
    print(f"Generating Stage 2 Phase 1 report for run {run}...")
    html     = build_report(run)
    out_path = os.path.join(SCRIPT_DIR, f"report_stage2_phase1_run_{run}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
