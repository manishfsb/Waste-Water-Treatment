"""
error_decomposition.py — Decompose 2025 residuals by operational regime.

For each target we take the stored Phase 9 Voting predictions from the
Experiment 3 Sub-2 dataset files (column `predicted_Voting_run_1`) and
decompose 2025 residuals along four regime axes:

  1. Flow quartile      (quartiles of Flow (MLD) on the TRAINING set)
  2. Weekday vs weekend (day_of_week)
  3. Season             (DJF / MAM / JJA / SON)
  4. Inlet load         (quartiles of upstream Inlet BOD on TRAINING set)

The regime thresholds are fit ONLY on training data and applied to 2025 —
this prevents regime definitions from being biased by 2025's own distribution.

For each cell we report n, MAE, RMSE, mean bias, and naive-R²
(1 − SSE/SST where SST uses the per-cell mean of y_true).
The aim is operational: identify whether model failure concentrates in
a handful of regimes (e.g. high-flow storm days, weekends, specific months)
rather than being uniform — which reframes "Comp COD is broken" as
"Comp COD is broken specifically under regime X".

Outputs:
  reports/error_decomposition.xlsx
  reports/error_decomposition.html

Run:  .venv/bin/python3 21-25/modeling/scripts/error_decomposition.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODELING_DIR)
DS_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

TARGETS = [
    ("s2_stage3_grab_BOD", "Effluent BOD (mg/L, Grab)",      "Inlet BOD (mg/L, Grab)"),
    ("s2_stage3_grab_COD", "Effluent COD (mg/L, Grab)",      "Inlet COD (mg/L, Grab)"),
    ("s2_stage3_grab_TSS", "Effluent TSS (mg/L, Grab)",      "Inlet TSS (mg/L, Grab)"),
    ("s2_stage3_grab_pH",  "Effluent pH (Grab)",             "Inlet pH (Grab)"),
    ("s2_stage3_comp_BOD", "Effluent BOD (mg/L, Composite)", "Inlet BOD (mg/L, Composite)"),
    ("s2_stage3_comp_COD", "Effluent COD (mg/L, Composite)", "Inlet COD (mg/L, Composite)"),
    ("s2_stage3_comp_TSS", "Effluent TSS (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"),
    ("s2_stage3_comp_pH",  "Effluent pH (Composite)",        "Inlet pH (Composite)"),
]

MODEL_COL = "predicted_Voting_run_1"     # Phase 9 ensemble artefact — present in all s2 files
SEASON_MAP = {12:"DJF",1:"DJF",2:"DJF", 3:"MAM",4:"MAM",5:"MAM",
              6:"JJA",7:"JJA",8:"JJA",  9:"SON",10:"SON",11:"SON"}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-regime metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _cell_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, r2=np.nan)
    resid = y_true - y_pred
    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    if np.var(y_true) > 0:
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        r2 = np.nan
    return dict(n=int(len(y_true)), mae=mae, rmse=rmse, bias=bias, r2=float(r2))


def _quartile_edges(train_series: pd.Series) -> list:
    s = train_series.dropna()
    return list(np.quantile(s, [0.0, 0.25, 0.5, 0.75, 1.0]))


def _bucket_by_quartile(value, edges, labels=("Q1", "Q2", "Q3", "Q4")):
    if pd.isna(value):
        return "NA"
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        if i == 3:
            if value >= lo and value <= hi:
                return labels[i]
        else:
            if value >= lo and value < hi:
                return labels[i]
    # Out-of-range
    return labels[0] if value < edges[0] else labels[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-target decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_target(name: str, target: str, inlet_col: str) -> pd.DataFrame:
    """
    Return long-format DataFrame with columns:
      target, axis, bucket, n, mae, rmse, bias, r2
    """
    path = os.path.join(DS_DIR, f"{name}.xlsx")
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if MODEL_COL not in df.columns:
        print(f"  WARN: {MODEL_COL} missing in {name}; skipped")
        return pd.DataFrame()

    # Restrict to rows with non-null (y_true, y_pred, flow, inlet)
    flow_col   = "Flow (MLD)"
    dow_col    = "day_of_week"
    needed = [target, MODEL_COL, flow_col, dow_col, inlet_col]
    df = df.dropna(subset=needed).copy()
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    train = df[df["year"] < 2025]
    test  = df[df["year"] == 2025]
    if len(test) == 0:
        return pd.DataFrame()

    # Quartile edges from training
    flow_edges  = _quartile_edges(train[flow_col])
    inlet_edges = _quartile_edges(train[inlet_col])

    # Apply to 2025
    y_true = test[target].values
    y_pred = test[MODEL_COL].values

    rows = []

    # Overall
    rows.append({"target": target, "axis": "OVERALL", "bucket": "2025",
                 **_cell_metrics(y_true, y_pred)})

    # 1. Flow quartile
    buckets = test[flow_col].apply(lambda v: _bucket_by_quartile(v, flow_edges))
    for b in ("Q1", "Q2", "Q3", "Q4"):
        mask = (buckets == b).values
        rows.append({"target": target, "axis": "Flow",
                     "bucket": f"{b} (train [{flow_edges[int(b[1])-1]:.1f},{flow_edges[int(b[1])]:.1f}])",
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    # 2. Weekday vs weekend
    is_weekend = test[dow_col].isin([5, 6]).values
    rows.append({"target": target, "axis": "Day",
                 "bucket": "Weekday", **_cell_metrics(y_true[~is_weekend], y_pred[~is_weekend])})
    rows.append({"target": target, "axis": "Day",
                 "bucket": "Weekend", **_cell_metrics(y_true[is_weekend], y_pred[is_weekend])})

    # 3. Season
    seasons = test["month"].map(SEASON_MAP).values
    for s in ("DJF", "MAM", "JJA", "SON"):
        mask = seasons == s
        rows.append({"target": target, "axis": "Season", "bucket": s,
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    # 4. Inlet load (upstream feed quartile)
    load_buckets = test[inlet_col].apply(lambda v: _bucket_by_quartile(v, inlet_edges))
    for b in ("Q1", "Q2", "Q3", "Q4"):
        mask = (load_buckets == b).values
        rows.append({"target": target, "axis": "InletLoad",
                     "bucket": f"{b} (train [{inlet_edges[int(b[1])-1]:.1f},{inlet_edges[int(b[1])]:.1f}])",
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML render
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, d=3):
    if pd.isna(v): return "—"
    if isinstance(v, (int, np.integer)): return str(int(v))
    return f"{v:.{d}f}"


def _bias_color(bias, overall_mae):
    """Red if |bias| exceeds half the overall MAE — systematic shift."""
    if pd.isna(bias) or pd.isna(overall_mae) or overall_mae == 0:
        return ""
    return "color:#e74c3c;font-weight:600" if abs(bias) > 0.5 * overall_mae else ""


def _mae_ratio_color(mae_cell, mae_overall):
    if pd.isna(mae_cell) or pd.isna(mae_overall) or mae_overall == 0:
        return ""
    r = mae_cell / mae_overall
    if r > 1.75: return "color:#e74c3c;font-weight:600"
    if r > 1.25: return "color:#f1c40f"
    if r < 0.60: return "color:#2ecc71"
    return ""


def render_html(all_results: pd.DataFrame) -> str:
    css = dark_mode_css("""
    body { font-family:-apple-system, Segoe UI, sans-serif; padding:24px; }
    .container { max-width:1400px; margin:0 auto; }
    table.decomp { width:100%; border-collapse:collapse; font-size:12px; margin:8px 0 24px; }
    table.decomp th, table.decomp td { padding:5px 8px; border-bottom:1px solid var(--border); }
    table.decomp th { background:var(--bg-soft); font-weight:600; text-align:left; }
    table.decomp td.num { text-align:right; font-variant-numeric:tabular-nums; }
    .target-header { font-size:16px; font-weight:600; margin:20px 0 6px;
                     padding:6px 10px; background:var(--bg-soft); border-left:3px solid #4A90D9; }
    .axis-row td { background:var(--bg-soft); font-weight:600; font-size:11px;
                   text-transform:uppercase; letter-spacing:.4px; color:#4A90D9; }
    .note { color:var(--text-muted); font-size:12px; margin:8px 0 16px; }
    """)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>2025 Residuals — Regime Decomposition</title>",
        f"<style>{css}</style></head><body><div class='container'>",
        "<h1>2025 Residual Decomposition by Operational Regime</h1>",
        f"<p class='note'>Generated {ts}. Model: <code>{MODEL_COL}</code> "
        "(Phase 9 Voting = ElNet+RF+XGB). Quartile thresholds fitted on "
        "training years (2021–2024) and applied to 2025.</p>",
        "<div class='rule-card note'>",
        "<strong>How to read:</strong> The OVERALL row is the baseline. "
        "Cells where MAE is &gt; 1.75× baseline are red — error concentrates "
        "in that regime. Bias coloured when |bias| &gt; 0.5 × overall MAE "
        "(systematic over/under-prediction).</div>",
    ]

    for target, df_tgt in all_results.groupby("target", sort=False):
        overall = df_tgt[df_tgt["axis"] == "OVERALL"].iloc[0]
        mae_overall = overall["mae"]

        parts.append(f"<div class='target-header'>{target} "
                     f"(overall n={int(overall['n'])}, MAE={_fmt(overall['mae'])}, "
                     f"RMSE={_fmt(overall['rmse'])}, bias={_fmt(overall['bias'])}, "
                     f"R²={_fmt(overall['r2'])})</div>")
        parts.append("<table class='decomp'>")
        parts.append(
            "<thead><tr>"
            "<th>Axis</th><th>Bucket</th><th>n</th>"
            "<th>MAE</th><th>RMSE</th><th>Bias</th><th>R²</th>"
            "</tr></thead><tbody>"
        )
        current_axis = None
        for _, r in df_tgt.iterrows():
            if r["axis"] == "OVERALL":
                continue
            if r["axis"] != current_axis:
                parts.append(
                    f"<tr class='axis-row'><td colspan='7'>{r['axis']}</td></tr>"
                )
                current_axis = r["axis"]
            mae_style  = _mae_ratio_color(r["mae"], mae_overall)
            bias_style = _bias_color(r["bias"], mae_overall)
            parts.append(
                "<tr>"
                f"<td></td><td>{r['bucket']}</td>"
                f"<td class='num'>{_fmt(r['n'])}</td>"
                f"<td class='num' style='{mae_style}'>{_fmt(r['mae'])}</td>"
                f"<td class='num'>{_fmt(r['rmse'])}</td>"
                f"<td class='num' style='{bias_style}'>{_fmt(r['bias'])}</td>"
                f"<td class='num'>{_fmt(r['r2'])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    parts.append(f"<script>{DARK_MODE_JS}</script></div></body></html>")
    return "\n".join(parts)


def section_html(all_results: pd.DataFrame) -> str:
    """Return inner section HTML for embedding in the unified report (no html/head/body)."""
    if all_results.empty:
        return "<p style='color:var(--text-muted)'>No error decomposition data available.</p>"

    css = """
    <style>
    table.ed-decomp { width:100%; border-collapse:collapse; font-size:12px; margin:8px 0 24px; }
    table.ed-decomp th, table.ed-decomp td { padding:5px 8px; border-bottom:1px solid var(--border); }
    table.ed-decomp th { background:var(--bg-soft); font-weight:600; text-align:left; }
    table.ed-decomp td.num { text-align:right; font-variant-numeric:tabular-nums; }
    .ed-target-header { font-size:15px; font-weight:600; margin:20px 0 6px;
                        padding:6px 10px; background:var(--bg-soft);
                        border-left:3px solid #4A90D9; border-radius:0 4px 4px 0; }
    .ed-axis-row td { background:var(--bg-soft); font-weight:600; font-size:11px;
                      text-transform:uppercase; letter-spacing:.4px; color:#4A90D9; }
    </style>
    """

    note = (
        "<p style='color:var(--text-muted);font-size:12px;margin:0 0 12px'>"
        f"Model: <code>{MODEL_COL}</code> (Phase 9 Voting = ElNet+RF+XGB). "
        "Quartile thresholds fitted on training years (2021–2024) and applied to 2025. "
        "Cells where MAE is &gt;1.75× baseline are red — error concentrates in that regime. "
        "Bias coloured when |bias| &gt; 0.5 × overall MAE (systematic over/under-prediction).</p>"
    )

    parts = [css, note]

    for target, df_tgt in all_results.groupby("target", sort=False):
        overall = df_tgt[df_tgt["axis"] == "OVERALL"].iloc[0]
        mae_overall = overall["mae"]

        parts.append(
            f"<div class='ed-target-header'>{target} "
            f"(overall n={int(overall['n'])}, MAE={_fmt(overall['mae'])}, "
            f"RMSE={_fmt(overall['rmse'])}, bias={_fmt(overall['bias'])}, "
            f"R²={_fmt(overall['r2'])})</div>"
        )
        parts.append("<table class='ed-decomp'>")
        parts.append(
            "<thead><tr>"
            "<th>Axis</th><th>Bucket</th><th>n</th>"
            "<th>MAE</th><th>RMSE</th><th>Bias</th><th>R²</th>"
            "</tr></thead><tbody>"
        )
        current_axis = None
        for _, r in df_tgt.iterrows():
            if r["axis"] == "OVERALL":
                continue
            if r["axis"] != current_axis:
                parts.append(
                    f"<tr class='ed-axis-row'><td colspan='7'>{r['axis']}</td></tr>"
                )
                current_axis = r["axis"]
            mae_style  = _mae_ratio_color(r["mae"], mae_overall)
            bias_style = _bias_color(r["bias"], mae_overall)
            parts.append(
                "<tr>"
                f"<td></td><td>{r['bucket']}</td>"
                f"<td class='num'>{_fmt(r['n'])}</td>"
                f"<td class='num' style='{mae_style}'>{_fmt(r['mae'])}</td>"
                f"<td class='num'>{_fmt(r['rmse'])}</td>"
                f"<td class='num' style='{bias_style}'>{_fmt(r['bias'])}</td>"
                f"<td class='num'>{_fmt(r['r2'])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== 2025 Residual Decomposition ===")
    frames = []
    for name, tgt, inlet in TARGETS:
        print(f"\n[{name}]")
        df = decompose_target(name, tgt, inlet)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data decomposed.")
        return

    combined = pd.concat(frames, ignore_index=True)

    xlsx_path = os.path.join(REPORTS_DIR, "error_decomposition.xlsx")
    combined.to_excel(xlsx_path, index=False)
    print(f"\nxlsx → {xlsx_path}")

    html = render_html(combined)
    html_path = os.path.join(REPORTS_DIR, "error_decomposition.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"html → {html_path}")

    # CLI summary: flag axes where worst cell is much worse than overall
    print("\n=== Regime concentration (axes where worst MAE > 1.75 × overall MAE) ===")
    for target, df_tgt in combined.groupby("target", sort=False):
        overall = df_tgt[df_tgt["axis"] == "OVERALL"].iloc[0]
        mae_ov = overall["mae"]
        for axis, df_ax in df_tgt.groupby("axis"):
            if axis == "OVERALL":
                continue
            worst = df_ax.loc[df_ax["mae"].idxmax()]
            if pd.isna(worst["mae"]) or pd.isna(mae_ov) or mae_ov == 0:
                continue
            ratio = worst["mae"] / mae_ov
            if ratio > 1.75:
                print(f"  {target:45s} {axis:10s} worst bucket={worst['bucket']:35s} "
                      f"MAE={worst['mae']:.3f} ({ratio:.2f}× overall)")


if __name__ == "__main__":
    main()
