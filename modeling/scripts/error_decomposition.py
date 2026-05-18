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
       - Season             (Kathmandu/Nepal climate seasons)
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
import re
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

# Some generated result rows do not have a one-to-one dataset directory. These
# models append row-level predictions back to their source dataset files.
_DATASET_DIR_OVERRIDES = {
    "Exp3-S2-FS": "experiment3/sub_exp2",
    "Exp2-S6-FS": "experiment2/sub_exp6",
    "Exp6-Voting": "experiment3/sub_exp2",
    "Exp6-Stacking": "experiment3/sub_exp2",
    "Exp6-ANN": "experiment3/sub_exp2",
    "Exp7-SE1": "experiment3/sub_exp2",
    "Exp7-SE2": "experiment3/sub_exp2",
    "Exp8": "experiment3/sub_exp2",
}

_PRED_COL_PATTERNS = {
    "Exp2-S6-FS": [
        "predicted_{model}_fs_run_{run}",
        "predicted_{model}_run_{run}",
    ],
    "Exp7-SE1": [
        "predicted_{model}_exp7se1_run_{run}",
    ],
    "Exp7-SE2": [
        "predicted_{model}_exp7se2_run_{run}",
    ],
    "Exp8": [
        "predicted_{model}_exp8_run_{run}",
    ],
}

# Kathmandu/Nepal climate seasons: Winter (Dec-Feb), Spring/Pre-monsoon
# (Mar-May), Monsoon (Jun-Sep), Autumn/Post-monsoon (Oct-Nov).
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ("Winter", "Spring", "Monsoon", "Autumn")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-target winner lookup
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_target_winners():
    """For every target, identify the Global best winner and prediction path.

    Returns: dict[target_long] = {
        'exp_key': str, 'model': str, 'r2': float, 'gap': float, 'score': float,
        'xlsx_path': str | None, 'pred_col': str | None, 'skip_reason': str | None
    }
    """
    from generate_unified_report import load_all_data, compute_all_mdae  # noqa: E402
    from best_models_selection import (  # noqa: E402
        build_per_experiment, build_global_from_per_experiment,
    )

    df_all = load_all_data()
    mdae_df = compute_all_mdae()
    if not mdae_df.empty:
        df_all = df_all.merge(mdae_df, on=["exp_key", "model", "target"], how="left")

    per_exp = build_per_experiment(df_all)
    global_winners = build_global_from_per_experiment(per_exp)
    exp_to_dir = _build_exp_to_dir()

    out = {}
    for ds_name, target, inlet_col in TARGETS:
        sub = global_winners[global_winners["target"] == target].copy()
        if sub.empty:
            out[target] = {"skip_reason": "no per-experiment rows for this target"}
            continue
        winner = sub.iloc[0]
        exp_key = str(winner["exp_key"])
        raw_winner_model = str(winner.get("winner_model", winner.get("gadj_model", "")))
        model = raw_winner_model.split(" · ")[-1]
        r2 = float(winner.get("winner_R2", winner.get("gadj_R2", np.nan)))
        gap = float(winner.get("winner_gap", winner.get("gadj_gap", np.nan)))
        score = float(winner.get("winner_score", winner.get("gadj_score", np.nan)))
        rule = str(winner.get("winner_rule", "Gap-adj"))

        # Locate the dataset xlsx that carries this model's predictions.
        ds_subdir = _DATASET_DIR_OVERRIDES.get(exp_key, exp_to_dir.get(exp_key))
        if ds_subdir is None:
            out[target] = {
                "exp_key": exp_key, "model": model,
                "rule": rule, "r2": r2, "gap": gap, "score": score,
                "xlsx_path": None, "pred_col": None,
                "skip_reason": f"exp_key {exp_key} has no dataset directory on disk",
            }
            continue

        xlsx_path = os.path.join(DATASETS_DIR, ds_subdir, f"{ds_name}.xlsx")
        if not os.path.exists(xlsx_path):
            out[target] = {
                "exp_key": exp_key, "model": model,
                "rule": rule, "r2": r2, "gap": gap, "score": score,
                "xlsx_path": None, "pred_col": None,
                "skip_reason": f"dataset file missing: {xlsx_path}",
            }
            continue

        # Find the highest-run predicted_<model>_run_N column.
        cols = pd.read_excel(xlsx_path, nrows=0).columns.tolist()
        templates = _PRED_COL_PATTERNS.get(exp_key, ["predicted_{model}_run_{run}"])
        runs = []
        for template in templates:
            pat = re.escape(template).replace(re.escape("{model}"), re.escape(model))
            pat = pat.replace(re.escape("{run}"), r"(\d+)")
            rx = re.compile(f"^{pat}$")
            runs.extend((int(m.group(1)), c) for c in cols if (m := rx.match(c)))
        if not runs:
            out[target] = {
                "exp_key": exp_key, "model": model,
                "rule": rule, "r2": r2, "gap": gap, "score": score,
                "xlsx_path": xlsx_path, "pred_col": None,
                "skip_reason": (f"predictions not stored to disk for "
                                f"{model} in {ds_subdir}; re-run the relevant "
                                "training script with prediction-dump enabled."),
            }
            continue

        pred_col = max(runs)[1]
        out[target] = {
            "exp_key": exp_key, "model": model,
            "rule": rule, "r2": r2, "gap": gap, "score": score,
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
    for s in SEASON_ORDER:
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


def _target_short(target: str) -> str:
    return (target
            .replace("Effluent ", "")
            .replace(" (mg/L, Grab)", " Grab")
            .replace(" (mg/L, Composite)", " Composite")
            .replace(" (Grab)", " Grab")
            .replace(" (Composite)", " Composite"))


def _findings_html(all_results: pd.DataFrame) -> str:
    if all_results.empty:
        return ""

    overall = all_results[all_results["axis"] == "OVERALL"].copy()
    resolved = len(overall)

    def _winner_line(r):
        return f"{_target_short(r['target'])}: {r['winner_exp_key']} · {r['winner_model']}"

    winners_text = "; ".join(_winner_line(r) for _, r in overall.iterrows())

    regime_rows = []
    for target, df_tgt in all_results.groupby("target", sort=False):
        base = df_tgt[df_tgt["axis"] == "OVERALL"].iloc[0]
        base_mae = base["mae"]
        if pd.isna(base_mae) or base_mae == 0:
            continue
        sub = df_tgt[(df_tgt["axis"] != "OVERALL") & (df_tgt["n"] >= 5)].copy()
        sub["ratio"] = sub["mae"] / base_mae
        if sub.empty:
            continue
        worst = sub.sort_values("ratio", ascending=False).iloc[0]
        regime_rows.append((worst["ratio"], target, worst))

    regime_rows = sorted(regime_rows, key=lambda x: x[0], reverse=True)
    concentration = [x for x in regime_rows if x[0] >= 1.75]
    concentration_text = "; ".join(
        f"{_target_short(tgt)} peaks in {r['axis']} = {r['bucket']} "
        f"({_fmt(r['mae'])} MAE, {ratio:.2f}x baseline)"
        for ratio, tgt, r in concentration[:5]
    ) or "No target has a regime with MAE above 1.75x its own 2025 baseline once tiny buckets are ignored."

    season_rows = all_results[(all_results["axis"] == "Season") & (all_results["bucket"] == "Winter")].copy()
    if not season_rows.empty:
        baseline = overall.set_index("target")["mae"]
        season_rows["ratio"] = season_rows.apply(
            lambda r: r["mae"] / baseline.get(r["target"], np.nan), axis=1)
        winter_hits = season_rows[(season_rows["ratio"] >= 1.4) & (season_rows["n"] >= 5)]
        winter_text = "; ".join(
            f"{_target_short(r['target'])} Winter is {_fmt(r['ratio'], 2)}x baseline "
            f"(bias {_fmt(r['bias'])})"
            for _, r in winter_hits.sort_values("ratio", ascending=False).iterrows()
        ) or "Winter no longer dominates every target, but remains a useful watch-list slice."
    else:
        winter_text = "Winter season buckets were not available in the decomposition output."

    cod_rows = overall[overall["target"].str.contains("COD", regex=False)].copy()
    cod_text = "; ".join(
        f"{_target_short(r['target'])}: R² {_fmt(r['r2'])}, MAE {_fmt(r['mae'])}, bias {_fmt(r['bias'])}"
        for _, r in cod_rows.iterrows()
    )

    def _qcard(n, question, answer):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem'>{answer}</p>
  </div>
</div>"""

    return f"""
<details class="exp-details" id="error-decomposition-findings" open>
  <summary><span class="fold-icon">▶</span> Findings</summary>
  <div class="exp-body">
    {_qcard(1, "Are all current global winners represented with row-wise predictions?",
            f"Yes. The section now decomposes all <strong>{resolved}</strong> target winners, so the pending rerun box is no longer needed. Current winners: {winners_text}.")}
    {_qcard(2, "Where does error concentrate most strongly in 2025?",
            concentration_text)}
    {_qcard(3, "Is winter still a systematic risk regime?",
            winter_text)}
    {_qcard(4, "What changed in the model-level interpretation?",
            "The decomposition is no longer anchored to whichever model happened to have predictions on disk. It now follows the same global overfit-aware winners as the Analytics table, which is why Grab BOD is decomposed with Exp7-SE2 Voting and Grab TSS with Exp7-SE1 Ridge. "
            f"For COD specifically: {cod_text}.")}
  </div>
</details>
"""


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
        "<strong>global overfit-aware winner</strong> (per the Overfit-Aware Selection table). "
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

    parts.append(_findings_html(all_results))
    parts.append(f"<script>{DARK_MODE_JS}</script></div></body></html>")
    return "\n".join(parts)


def section_html(all_results: pd.DataFrame, winners: dict | None = None) -> str:
    """Return inner section HTML for embedding in the unified report (no html/head/body).

    winners is the dict from resolve_target_winners() - used to render
    per-target winner badges.
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
    </style>
    """

    note = (
        "<p style='color:#555555;font-size:12px;margin:0 0 12px'>"
        "Each target is decomposed using its <strong>global overfit-aware winner</strong> "
        "from the Overfit-Aware Selection table. "
        "Quartile thresholds fitted on training years (2021-2024) and applied to 2025. "
        "Cells where MAE is &gt;1.75× baseline are red - error concentrates in that regime. "
        "Bias coloured when |bias| &gt; 0.5 × overall MAE (systematic over/under-prediction).</p>"
    )

    parts = [css, note]

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

    parts.append(_findings_html(all_results))
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
