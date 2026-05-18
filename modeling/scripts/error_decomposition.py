"""
error_decomposition.py - Decompose 2025 residuals by operational regime,
using the per-target gap-adjusted winner identified by the Overfit-Aware
Selection table (best_models_selection.py).

For each target we:
  1. Look up the canonical gap-adjusted winner (with SCORE_NOISE_BAND tiebreak).
  2. Find that model's stored row-level predictions in the relevant dataset
     directory under modeling/datasets/.
  3. Decompose 2025 residuals along four regime axes:
       - Flow quartile      (quartiles of Flow (MLD) on the TRAINING set)
       - Weekday vs weekend (day_of_week)
       - Season             (DJF / MAM / JJA / SON)
       - Inlet load         (quartiles of upstream Inlet on TRAINING set)
  4. If predictions are not stored to disk for the winner (Phase 9 ANN,
     Phase 10/11, several FS variants), the target is skipped with an
     explicit reason and the report shows a "predictions not available" note.

Regime thresholds are fit on TRAINING data only (year < 2025) and applied to
2025, so regime definitions are not biased by 2025's own distribution.

Outputs:
  reports/error_decomposition.xlsx  (long-format metrics + winner manifest)
  reports/error_decomposition.html  (standalone dark-mode report)

Run:  .venv/bin/python3 modeling/scripts/error_decomposition.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODELING_DIR)
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
DATASETS_DIR = os.path.join(MODELING_DIR, "datasets")
MASTER_DATA  = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

TARGETS = [
    ("grab_BOD", "Effluent BOD (mg/L, Grab)",      "Inlet BOD (mg/L, Grab)"),
    ("grab_COD", "Effluent COD (mg/L, Grab)",      "Inlet COD (mg/L, Grab)"),
    ("grab_TSS", "Effluent TSS (mg/L, Grab)",      "Inlet TSS (mg/L, Grab)"),
    ("grab_pH",  "Effluent pH (Grab)",             "Inlet pH (Grab)"),
    ("comp_BOD", "Effluent BOD (mg/L, Composite)", "Inlet BOD (mg/L, Composite)"),
    ("comp_COD", "Effluent COD (mg/L, Composite)", "Inlet COD (mg/L, Composite)"),
    ("comp_TSS", "Effluent TSS (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"),
    ("comp_pH",  "Effluent pH (Composite)",        "Inlet pH (Composite)"),
]

# Reverse of _DS_EXP_MAP in generate_unified_report.py: exp_key → dataset directory.
# Stays in sync with _DS_EXP_MAP by importing the source-of-truth list.
def _build_exp_to_dir():
    from generate_unified_report import _DS_EXP_MAP  # noqa: E402
    out = {}
    # Iterate in order so the more-specific entries (FS, sec_clarifier, sec_sed)
    # override the general ones; that matches the runtime behaviour of the
    # forward mapping (longest-prefix wins on directory lookup).
    for fragment, key in _DS_EXP_MAP:
        out[key] = fragment
    return out

# Exp 6 (Phase 9 ensemble) predictions live INSIDE the Exp3-S2 dataset directory
# under their own column tag. Same with Exp 7/8 in principle, but they don't
# currently write predictions. Listed here so the lookup can find them.
_EMBEDDED_IN_EXP3_S2 = {"Exp6-Voting", "Exp6-Stacking", "Exp6-ANN"}

SEASON_MAP = {12:"DJF",1:"DJF",2:"DJF", 3:"MAM",4:"MAM",5:"MAM",
              6:"JJA",7:"JJA",8:"JJA",  9:"SON",10:"SON",11:"SON"}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-target winner lookup
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_target_winners():
    """For every target, identify the gap-adjusted winner (with noise-band
    tiebreak) and the on-disk location of its row-level predictions.

    Returns: dict[target_long] = {
        'exp_key': str, 'model': str, 'r2': float, 'gap': float, 'score': float,
        'xlsx_path': str | None, 'pred_col': str | None, 'skip_reason': str | None
    }
    """
    from generate_unified_report import load_all_data, compute_all_mdae  # noqa: E402
    from best_models_selection import build_per_experiment  # noqa: E402

    df_all = load_all_data()
    mdae_df = compute_all_mdae()
    if not mdae_df.empty:
        df_all = df_all.merge(mdae_df, on=["exp_key", "model", "target"], how="left")

    per_exp = build_per_experiment(df_all)
    exp_to_dir = _build_exp_to_dir()

    out = {}
    for ds_name, target, inlet_col in TARGETS:
        sub = per_exp[per_exp["target"] == target].copy()
        if sub.empty:
            out[target] = {"skip_reason": "no per-experiment rows for this target"}
            continue
        # Match the Analytics-table sort: gap-adj score (with noise band),
        # then smaller |gap|, then higher R².
        from best_models_selection import SCORE_NOISE_BAND  # noqa: E402
        sub["_abs_gap"] = sub["gadj_gap"].abs()
        sub["_bucket"] = (sub["gadj_score"] / SCORE_NOISE_BAND).round()
        sub = sub.sort_values(["_bucket", "_abs_gap", "naive_R2"],
                              ascending=[False, True, False])
        winner = sub.iloc[0]
        exp_key = str(winner["exp_key"])
        model   = str(winner["gadj_model"])

        # Locate the dataset xlsx that carries this model's predictions.
        # Most winners live in their experiment's directory; Exp 6 (P9 ensemble)
        # predictions live inside Exp3-S2's directory.
        lookup_key = "Exp3-S2" if exp_key in _EMBEDDED_IN_EXP3_S2 else exp_key
        ds_subdir = exp_to_dir.get(lookup_key)
        if ds_subdir is None:
            out[target] = {
                "exp_key": exp_key, "model": model,
                "r2": float(winner["gadj_R2"]), "gap": float(winner["gadj_gap"]),
                "score": float(winner["gadj_score"]),
                "xlsx_path": None, "pred_col": None,
                "skip_reason": f"exp_key {exp_key} has no dataset directory on disk",
            }
            continue

        xlsx_path = os.path.join(DATASETS_DIR, ds_subdir, f"{ds_name}.xlsx")
        if not os.path.exists(xlsx_path):
            out[target] = {
                "exp_key": exp_key, "model": model,
                "r2": float(winner["gadj_R2"]), "gap": float(winner["gadj_gap"]),
                "score": float(winner["gadj_score"]),
                "xlsx_path": None, "pred_col": None,
                "skip_reason": f"dataset file missing: {xlsx_path}",
            }
            continue

        # Find the highest-run predicted_<model>_run_N column.
        import re
        cols = pd.read_excel(xlsx_path, nrows=0).columns.tolist()
        pat = re.compile(rf"^predicted_{re.escape(model)}_run_(\d+)$")
        runs = [(int(m.group(1)), c) for c in cols if (m := pat.match(c))]
        if not runs:
            out[target] = {
                "exp_key": exp_key, "model": model,
                "r2": float(winner["gadj_R2"]), "gap": float(winner["gadj_gap"]),
                "score": float(winner["gadj_score"]),
                "xlsx_path": xlsx_path, "pred_col": None,
                "skip_reason": (f"predictions not stored to disk for "
                                f"{model} in {ds_subdir}; re-run the relevant "
                                "training script with prediction-dump enabled."),
            }
            continue

        pred_col = max(runs)[1]
        out[target] = {
            "exp_key": exp_key, "model": model,
            "r2": float(winner["gadj_R2"]), "gap": float(winner["gadj_gap"]),
            "score": float(winner["gadj_score"]),
            "xlsx_path": xlsx_path, "pred_col": pred_col,
            "skip_reason": None,
        }
    return out


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

def decompose_target(name: str, target: str, inlet_col: str,
                     winner_info: dict | None = None) -> pd.DataFrame:
    """
    Return long-format DataFrame with columns:
      target, axis, bucket, n, mae, rmse, bias, r2,
      plus winner metadata (exp_key, model, pred_col) on every row.

    winner_info is the dict returned by resolve_target_winners()[target].
    If predictions aren't available on disk, returns an empty DataFrame
    after printing the skip reason.
    """
    if winner_info is None or winner_info.get("skip_reason"):
        reason = (winner_info or {}).get("skip_reason", "no winner info")
        print(f"  [{name}] SKIPPED - {reason}")
        return pd.DataFrame()

    xlsx_path = winner_info["xlsx_path"]
    pred_col  = winner_info["pred_col"]
    exp_key   = winner_info["exp_key"]
    model     = winner_info["model"]

    df = pd.read_excel(xlsx_path)
    df["Date"] = pd.to_datetime(df["Date"])
    if pred_col not in df.columns:
        print(f"  [{name}] SKIPPED - {pred_col} missing in {xlsx_path}")
        return pd.DataFrame()

    # Pull regime variables (Flow, Inlet) from the master dataset by Date.
    # Some experiment datasets (e.g. VIF-pruned Exp4-S2) have dropped these
    # columns as predictors, but the regime definitions are physical facts
    # tied to the date - so they should come from the master, not the slice.
    flow_col = "Flow (MLD)"
    master_cols = ["Date", flow_col, inlet_col]
    master = pd.read_excel(MASTER_DATA, usecols=lambda c: c in master_cols)
    master["Date"] = pd.to_datetime(master["Date"])
    # Keep only target + pred + Date from the experiment slice
    df = df[["Date", target, pred_col]].merge(master, on="Date", how="left")

    needed = [target, pred_col, flow_col, inlet_col]
    df = df.dropna(subset=needed).copy()
    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    train = df[df["year"] < 2025]
    test  = df[df["year"] == 2025]
    if len(test) == 0:
        print(f"  [{name}] SKIPPED - no 2025 rows for this combo")
        return pd.DataFrame()

    flow_edges  = _quartile_edges(train[flow_col])
    inlet_edges = _quartile_edges(train[inlet_col])

    y_true = test[target].values
    y_pred = test[pred_col].values

    rows = []
    meta = {"target": target, "winner_exp_key": exp_key, "winner_model": model,
            "winner_pred_col": pred_col}

    rows.append({**meta, "axis": "OVERALL", "bucket": "2025",
                 **_cell_metrics(y_true, y_pred)})

    buckets = test[flow_col].apply(lambda v: _bucket_by_quartile(v, flow_edges))
    for b in ("Q1", "Q2", "Q3", "Q4"):
        mask = (buckets == b).values
        rows.append({**meta, "axis": "Flow",
                     "bucket": f"{b} (train [{flow_edges[int(b[1])-1]:.1f},{flow_edges[int(b[1])]:.1f}])",
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    is_weekend = test["day_of_week"].isin([5, 6]).values
    rows.append({**meta, "axis": "Day", "bucket": "Weekday",
                 **_cell_metrics(y_true[~is_weekend], y_pred[~is_weekend])})
    rows.append({**meta, "axis": "Day", "bucket": "Weekend",
                 **_cell_metrics(y_true[is_weekend], y_pred[is_weekend])})

    seasons = test["month"].map(SEASON_MAP).values
    for s in ("DJF", "MAM", "JJA", "SON"):
        mask = seasons == s
        rows.append({**meta, "axis": "Season", "bucket": s,
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    load_buckets = test[inlet_col].apply(lambda v: _bucket_by_quartile(v, inlet_edges))
    for b in ("Q1", "Q2", "Q3", "Q4"):
        mask = (load_buckets == b).values
        rows.append({**meta, "axis": "InletLoad",
                     "bucket": f"{b} (train [{inlet_edges[int(b[1])-1]:.1f},{inlet_edges[int(b[1])]:.1f}])",
                     **_cell_metrics(y_true[mask], y_pred[mask])})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML render
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, d=3):
    if pd.isna(v): return "-"
    if isinstance(v, (int, np.integer)): return str(int(v))
    return f"{v:.{d}f}"


def _bias_color(bias, overall_mae):
    """Red if |bias| exceeds half the overall MAE - systematic shift."""
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
        "<title>2025 Residuals - Regime Decomposition</title>",
        f"<style>{css}</style></head><body><div class='container'>",
        "<h1>2025 Residual Decomposition by Operational Regime</h1>",
        f"<p class='note'>Generated {ts}. Each target is decomposed using its "
        "<strong>gap-adjusted winner</strong> (per the Overfit-Aware Selection table). "
        "Quartile thresholds fitted on training years (2021-2024) and applied to 2025.</p>",
        "<div class='rule-card note'>",
        "<strong>How to read:</strong> The OVERALL row is the baseline. "
        "Cells where MAE is &gt; 1.75× baseline are red - error concentrates "
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
            "<th>Axis</th><th>Bucket</th><th class='num'>n</th>"
            "<th class='num'>MAE</th><th class='num'>RMSE</th><th class='num'>Bias</th><th class='num'>R²</th>"
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


def section_html(all_results: pd.DataFrame, winners: dict | None = None) -> str:
    """Return inner section HTML for embedding in the unified report (no html/head/body).

    winners is the dict from resolve_target_winners() - used to render
    per-target winner badges and to surface skipped targets (those whose
    gap-adjusted winner has no row-level predictions on disk yet).
    """
    if all_results.empty and not winners:
        return "<p style='color:#888888'>No error decomposition data available.</p>"

    css = """
    <style>
    table.ed-decomp { width:100%; border-collapse:collapse; font-size:0.81rem;
                      margin:8px 0 24px; background:#ffffff; color:#1a1a1a; }
    table.ed-decomp th, table.ed-decomp td {
        padding:5px 10px; border-bottom:1px solid #e0e0e0; }
    table.ed-decomp th { background:#eeeeee; font-weight:600; text-align:left;
                         font-size:0.82rem; color:#333333; }
    table.ed-decomp thead tr { border-bottom:2px solid #cccccc; }
    table.ed-decomp th.num { text-align:right; }
    table.ed-decomp td.num { text-align:right; font-variant-numeric:tabular-nums; }
    .ed-tbl-wrap { overflow-x:auto; border:1px solid #cccccc; border-radius:4px;
                   margin-bottom:24px; }
    .ed-target-header { font-size:15px; font-weight:600; margin:20px 0 6px;
                        padding:6px 10px; background:#f5f5f5; color:#1a1a1a;
                        border-left:3px solid #4A90D9; border-radius:0 4px 4px 0; }
    .ed-winner-badge { display:inline-block; background:#E3F0FB; color:#1a1a1a;
                       padding:1px 8px; border:1px solid #4A90D9; border-radius:3px;
                       font-size:0.78em; font-weight:600; margin-left:8px; }
    .ed-axis-row td { background:#e8e8e8; font-weight:700; font-size:0.75rem;
                      text-transform:uppercase; letter-spacing:.05em; color:#555555;
                      border-bottom:1px solid #d0d0d0; }
    .ed-skipped { background:#FFF3CD; border:1px solid #F0C36D; padding:8px 12px;
                  border-radius:4px; margin:12px 0; color:#5A4500; font-size:12px; }
    </style>
    """

    note = (
        "<p style='color:#555555;font-size:12px;margin:0 0 12px'>"
        "Each target is decomposed using its <strong>gap-adjusted winner</strong> "
        "from the Overfit-Aware Selection table. "
        "Quartile thresholds fitted on training years (2021-2024) and applied to 2025. "
        "Cells where MAE is &gt;1.75× baseline are red - error concentrates in that regime. "
        "Bias coloured when |bias| &gt; 0.5 × overall MAE (systematic over/under-prediction).</p>"
    )

    parts = [css, note]

    # Surface targets where the gap-adjusted winner's predictions aren't on disk.
    if winners:
        skipped = [(tgt, info) for tgt, info in winners.items()
                   if info.get("skip_reason")]
        if skipped:
            parts.append("<div class='ed-skipped'>")
            parts.append("<strong>Pending re-runs</strong> - the gap-adjusted winner "
                         "for these targets does not yet have row-level predictions "
                         "on disk, so decomposition was skipped. Re-run the relevant "
                         "training script with prediction-dump enabled to populate.")
            parts.append("<ul style='margin:6px 0 0 18px;padding:0'>")
            for tgt, info in skipped:
                exp_model = (f"{info.get('exp_key','?')} · {info.get('model','?')}"
                             if info.get("exp_key") else "unresolved")
                parts.append(f"<li><strong>{tgt}</strong> - winner {exp_model}. "
                             f"<em>{info.get('skip_reason','no reason')}</em></li>")
            parts.append("</ul></div>")

    winners = winners or {}
    for target, df_tgt in all_results.groupby("target", sort=False):
        overall = df_tgt[df_tgt["axis"] == "OVERALL"].iloc[0]
        mae_overall = overall["mae"]
        win = winners.get(target, {})
        winner_badge = ""
        if win.get("exp_key") and win.get("model"):
            w_score = win.get("score", 0.0)
            w_r2    = win.get("r2",    0.0)
            w_gap   = win.get("gap",   0.0)
            badge_title = (f"Gap-adj score {w_score:.4f}, "
                           f"R² {w_r2:+.3f}, gap {w_gap:+.3f}")
            winner_badge = (f"<span class='ed-winner-badge' title='{badge_title}'>"
                            f"{win['exp_key']} · {win['model']}</span>")

        parts.append(
            f"<div class='ed-target-header'>{target}{winner_badge} "
            f"<span style='font-weight:400;color:#555555;font-size:0.85em'>"
            f"(overall n={int(overall['n'])}, MAE={_fmt(overall['mae'])}, "
            f"RMSE={_fmt(overall['rmse'])}, bias={_fmt(overall['bias'])}, "
            f"R²={_fmt(overall['r2'])})</span></div>"
        )
        parts.append("<div class='ed-tbl-wrap'><table class='ed-decomp'>")
        parts.append(
            "<thead><tr>"
            "<th>Axis</th><th>Bucket</th><th class='num'>n</th>"
            "<th class='num'>MAE</th><th class='num'>RMSE</th><th class='num'>Bias</th><th class='num'>R²</th>"
            "</tr></thead><tbody>"
        )
        current_axis = None
        for i_row, (_, r) in enumerate(df_tgt.iterrows()):
            if r["axis"] == "OVERALL":
                continue
            if r["axis"] != current_axis:
                parts.append(
                    f"<tr class='ed-axis-row'><td colspan='7'>{r['axis']}</td></tr>"
                )
                current_axis = r["axis"]
            row_bg = "#ffffff" if i_row % 2 == 0 else "#f7f7f7"
            mae_style  = _mae_ratio_color(r["mae"], mae_overall)
            bias_style = _bias_color(r["bias"], mae_overall)
            parts.append(
                f"<tr style='background:{row_bg}'>"
                f"<td></td><td>{r['bucket']}</td>"
                f"<td class='num'>{_fmt(r['n'])}</td>"
                f"<td class='num' style='{mae_style}'>{_fmt(r['mae'])}</td>"
                f"<td class='num'>{_fmt(r['rmse'])}</td>"
                f"<td class='num' style='{bias_style}'>{_fmt(r['bias'])}</td>"
                f"<td class='num'>{_fmt(r['r2'])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== 2025 Residual Decomposition - per-target gap-adjusted winners ===")
    winners = resolve_target_winners()
    print("\nResolved winners:")
    for tgt, info in winners.items():
        if info.get("skip_reason"):
            print(f"  [SKIP] {tgt}: winner = "
                  f"{info.get('exp_key','?')} · {info.get('model','?')} - "
                  f"{info['skip_reason']}")
        else:
            print(f"  [ OK ] {tgt}: winner = {info['exp_key']} · {info['model']}  "
                  f"(score {info['score']:.4f}, R² {info['r2']:+.3f}, gap {info['gap']:+.3f})  "
                  f"<- {info['pred_col']}")

    frames = []
    for name, tgt, inlet in TARGETS:
        print(f"\n[{name}]")
        df = decompose_target(name, tgt, inlet, winner_info=winners.get(tgt))
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nNo data decomposed (all targets skipped).")
        # Still write the winner manifest for diagnostic purposes.
        manifest = pd.DataFrame([{"target": k, **v} for k, v in winners.items()])
        manifest.to_excel(os.path.join(REPORTS_DIR, "error_decomposition.xlsx"),
                          index=False)
        return

    combined = pd.concat(frames, ignore_index=True)

    xlsx_path = os.path.join(REPORTS_DIR, "error_decomposition.xlsx")
    with pd.ExcelWriter(xlsx_path) as xw:
        combined.to_excel(xw, sheet_name="decomposition", index=False)
        pd.DataFrame([{"target": k, **v} for k, v in winners.items()]).to_excel(
            xw, sheet_name="winners", index=False)
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
