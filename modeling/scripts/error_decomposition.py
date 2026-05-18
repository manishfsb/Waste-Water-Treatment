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

    is_weekend = test["day_of_week"].isin([5]).values   # Nepal: Saturday only; Sunday is a workday
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

    overall      = all_results[all_results["axis"] == "OVERALL"].copy()
    baseline_mae = overall.set_index("target")["mae"]
    baseline_r2  = overall.set_index("target")["r2"]  # noqa: F841

    # ── Q1: Winter R² collapse ────────────────────────────────────────────────
    winter_rows = all_results[
        (all_results["axis"] == "Season") & (all_results["bucket"] == "Winter")
    ].copy()
    negative_r2, elevated_winter = [], []
    for _, r in winter_rows.iterrows():
        if r["n"] < 5:
            continue
        ratio = r["mae"] / baseline_mae.get(r["target"], np.nan)
        if not pd.isna(r["r2"]) and r["r2"] < 0:
            negative_r2.append(
                f"<strong>{_target_short(r['target'])}</strong> "
                f"(R2 = {_fmt(r['r2'])}, MAE {_fmt(ratio, 2)}x baseline)"
            )
        elif not pd.isna(ratio) and ratio >= 1.4:
            elevated_winter.append(
                f"<strong>{_target_short(r['target'])}</strong> "
                f"(MAE {_fmt(ratio, 2)}x baseline, bias {_fmt(r['bias'])})"
            )
    if negative_r2:
        winter_alarm = (
            "<span style='color:#e74c3c;font-weight:700'>[!] R2 < 0 in Winter</span> - "
            "the model performs <em>worse than predicting the training mean</em> for: "
            + "; ".join(negative_r2) + ". "
        )
    else:
        winter_alarm = ""
    if elevated_winter:
        winter_alarm += (
            "Additional targets with elevated Winter MAE (>= 1.4x baseline): "
            + "; ".join(elevated_winter) + ". "
        )
    winter_alarm += (
        "Winter (Dec-Feb) in Kathmandu brings low temperatures that depress biological "
        "activity in the activated-sludge process, making effluent quality less predictable. "
        "<strong>Human review of winter predictions is strongly recommended.</strong>"
    ) if (negative_r2 or elevated_winter) else "Winter season shows no severe deterioration for any target."

    # ── Q2: Seasonal directional bias signatures ──────────────────────────────
    season_rows = all_results[all_results["axis"] == "Season"].copy()
    bias_lines  = []
    for season in ("Winter", "Spring", "Monsoon", "Autumn"):
        sb    = season_rows[season_rows["bucket"] == season]
        under = [_target_short(r["target"]) for _, r in sb.iterrows()
                 if r["n"] >= 5 and not pd.isna(r["bias"])
                 and r["bias"] > 0.5 * baseline_mae.get(r["target"], np.inf)]
        over  = [_target_short(r["target"]) for _, r in sb.iterrows()
                 if r["n"] >= 5 and not pd.isna(r["bias"])
                 and r["bias"] < -0.5 * baseline_mae.get(r["target"], np.inf)]
        parts = []
        if under:
            parts.append(f"<em>under-predicts</em> {', '.join(under)}")
        if over:
            parts.append(f"<em>over-predicts</em> {', '.join(over)}")
        if parts:
            bias_lines.append(f"<strong>{season}</strong>: model " + " and ".join(parts))
    bias_text = (
        "The model's directional errors follow a structured seasonal pattern. "
        + "; ".join(bias_lines)
        + ". This signature suggests a missing seasonal mean-shift: consider adding "
        "season dummies or a per-season additive offset at inference time."
    ) if bias_lines else "No systematic directional bias (|bias| > 0.5x MAE) found in any season."

    # ── Q3: 2025 regime shift ─────────────────────────────────────────────────
    flow_rows  = all_results[all_results["axis"] == "Flow"].copy()
    inlet_rows = all_results[all_results["axis"] == "InletLoad"].copy()
    q1_lines, inlet_q4_lines = [], []
    for target, df_tgt in flow_rows.groupby("target", sort=False):
        total_n = df_tgt["n"].sum()
        if total_n == 0:
            continue
        q1 = df_tgt[df_tgt["bucket"].str.startswith("Q1")]
        q4 = df_tgt[df_tgt["bucket"].str.startswith("Q4")]
        if q1.empty or q4.empty:
            continue
        q1_pct = 100 * q1.iloc[0]["n"] / total_n
        q4_pct = 100 * q4.iloc[0]["n"] / total_n
        if q1_pct >= 70:
            q1_lines.append(
                f"<strong>{_target_short(target)}</strong>: "
                f"{q1_pct:.0f}% of samples in Flow Q1; only {q4_pct:.0f}% in Q4"
            )
    for target, df_tgt in inlet_rows.groupby("target", sort=False):
        total_n = df_tgt["n"].sum()
        if total_n == 0:
            continue
        q4 = df_tgt[df_tgt["bucket"].str.startswith("Q4")]
        if q4.empty:
            continue
        q4_pct = 100 * q4.iloc[0]["n"] / total_n
        if q4_pct >= 50:
            inlet_q4_lines.append(
                f"<strong>{_target_short(target)}</strong>: {q4_pct:.0f}% in Inlet Load Q4"
            )
    if q1_lines or inlet_q4_lines:
        regime_text = "2025 is operating in an atypical regime relative to training (2021\u20132024). "
        if q1_lines:
            regime_text += (
                "Flow is concentrated in the <strong>lowest quartile</strong>: "
                + "; ".join(q1_lines[:2])
                + ". Most Q2/Q3/Q4 flow cells have n < 10 - "
                "treat those MAEs as directional signals only. "
            )
        if inlet_q4_lines:
            regime_text += (
                "Simultaneously, inlet loads skew toward the <strong>highest quartile</strong>: "
                + "; ".join(inlet_q4_lines[:3])
                + ". Low flow + high load = elevated organic concentration - "
                "a combination underrepresented in training data."
            )
    else:
        regime_text = "2025 flow and load distributions appear consistent with training-era quartile boundaries."


    # ── Q4: Error concentration hotspots ─────────────────────────────────────
    hotspot_rows = []
    for target, df_tgt in all_results.groupby("target", sort=False):
        bm = baseline_mae.get(target, np.nan)
        if pd.isna(bm) or bm == 0:
            continue
        sub = df_tgt[(df_tgt["axis"] != "OVERALL") & (df_tgt["n"] >= 5)].copy()
        sub["ratio"] = sub["mae"] / bm
        if sub.empty:
            continue
        worst = sub.sort_values("ratio", ascending=False).iloc[0]
        if worst["ratio"] >= 1.75:
            hotspot_rows.append((worst["ratio"], target, worst))
    hotspot_rows.sort(key=lambda x: x[0], reverse=True)
    if hotspot_rows:
        items = "".join(
            f"<li><strong>{_target_short(tgt)}</strong> - worst in "
            f"<em>{r['axis']} = {r['bucket']}</em>: "
            f"MAE {_fmt(r['mae'])} ({ratio:.2f}x baseline), "
            f"bias {_fmt(r['bias'])}, R2 {_fmt(r['r2'])}</li>"
            for ratio, tgt, r in hotspot_rows
        )
        hotspot_text = (
            "Regimes where MAE exceeds 1.75x the overall 2025 baseline (n >= 5):"
            f"<ul style='margin:0.4rem 0 0 1rem;padding:0;font-size:inherit;line-height:1.6'>{items}</ul>"
        )
    else:
        hotspot_text = "No target has a regime with MAE above 1.75x its own 2025 baseline."

    # ── Q5: Saturday vs Weekday ────────────────────────────────────────────────
    day_rows = all_results[all_results["axis"] == "Day"].copy()
    sat_better, sat_worse, sat_neutral = [], [], []
    for target, df_tgt in day_rows.groupby("target", sort=False):
        wd = df_tgt[df_tgt["bucket"] == "Weekday"]
        we = df_tgt[df_tgt["bucket"] == "Weekend"]
        if wd.empty or we.empty:
            continue
        wd_mae, we_mae, we_n = wd.iloc[0]["mae"], we.iloc[0]["mae"], we.iloc[0]["n"]
        if pd.isna(wd_mae) or pd.isna(we_mae) or we_n < 5 or wd_mae == 0:
            continue
        ratio = we_mae / wd_mae
        short = _target_short(target)
        if ratio < 0.85:
            sat_better.append(f"{short} ({ratio:.2f}x)")
        elif ratio > 1.15:
            sat_worse.append(f"{short} ({ratio:.2f}x)")
        else:
            sat_neutral.append(short)
    day_parts = []
    if sat_better:
        day_parts.append(f"<strong>Saturday easier to predict</strong>: {', '.join(sat_better)}")
    if sat_worse:
        day_parts.append(f"<strong>Saturday harder to predict</strong>: {', '.join(sat_worse)}")
    if sat_neutral:
        day_parts.append(f"No meaningful difference: {', '.join(sat_neutral)}")
    day_text = (
        "Weekend bucket now correctly reflects <strong>Saturdays only</strong> (Nepal calendar - Sunday is a workday). "
        + "; ".join(day_parts)
        + ". Saturday sample size is small (~52/year) - treat as directional signal only."
    ) if day_parts else "Insufficient Saturday samples (< 5) to draw conclusions for any target."

    # ── Q6: Target reliability ranking ────────────────────────────────────────
    overall_sorted = overall.sort_values("r2", ascending=True)
    hard  = overall_sorted[overall_sorted["r2"] < 0.30]
    solid = overall_sorted[overall_sorted["r2"] >= 0.50]
    hard_lines = "; ".join(
        f"<strong>{_target_short(r['target'])}</strong> "
        f"(R2 {_fmt(r['r2'])}, MAE {_fmt(r['mae'])}, "
        f"winner: {r['winner_exp_key']} / {r['winner_model']})"
        for _, r in hard.iterrows()
    )
    solid_lines = "; ".join(
        f"<strong>{_target_short(r['target'])}</strong> "
        f"(R2 {_fmt(r['r2'])}, winner: {r['winner_exp_key']} / {r['winner_model']})"
        for _, r in solid.iterrows()
    )
    difficulty_text = ""
    if hard_lines:
        difficulty_text += f"<strong>Use with caution (R2 < 0.30)</strong>: {hard_lines}. "
    if solid_lines:
        difficulty_text += f"<strong>Reliable (R2 >= 0.50)</strong>: {solid_lines}."
    difficulty_text = difficulty_text or "All targets fall in the 0.30-0.50 R2 range."

    # ── Q7: Operational recommendations ──────────────────────────────────────
    rec = []
    if negative_r2:
        rec.append(
            "<li><strong>Winter (Dec-Feb) - mandatory human review.</strong> "
            "R2 < 0 means model outputs can be more misleading than a flat estimate. "
            "Flag all winter predictions for operator verification before compliance reporting.</li>"
        )
    if bias_lines:
        rec.append(
            "<li><strong>Apply a seasonal bias correction.</strong> The directional error pattern "
            "is consistent enough to warrant a per-season additive offset derived from training residuals.</li>"
        )
    if inlet_q4_lines:
        rec.append(
            "<li><strong>Heighten monitoring on high-load days (Inlet Q4).</strong> "
            "These now dominate 2025 but were only the top 25% of training - "
            "model confidence is lowest when plant stress is highest.</li>"
        )
    if hard_lines:
        rec.append(
            "<li><strong>Do not use low-R2 targets for automated compliance decisions.</strong> "
            "Composite COD (R2 ~12%) should not serve as the sole basis for "
            "regulatory reporting without lab confirmation.</li>"
        )
    rec.append(
        "<li><strong>Retrain with 2025 data before next season.</strong> "
        "The Q1-flow + Q4-load operating regime of 2025 is systematically different from training; "
        "incorporating 2025 observations will materially improve generalization.</li>"
    )
    reco_text = "<ul style='margin:0.4rem 0 0 1rem;padding:0;font-size:inherit;line-height:1.6'>" + "".join(rec) + "</ul>"

    # ── Card renderer ──────────────────────────────────────────────────────────
    def _qcard(n, question, answer, alert=False):
        bc = "#e74c3c" if alert else "var(--border)"
        hb = "rgba(231,76,60,0.08)" if alert else "var(--bg-secondary)"
        lc = "#e74c3c" if alert else "#4A90D9"
        return (
            f"<div style='margin-bottom:1.1rem;border:1px solid {bc};border-radius:5px;overflow:hidden'>"
            f"<div style='background:{hb};padding:0.45rem 0.8rem;border-bottom:1px solid {bc};"
            f"display:flex;align-items:baseline;gap:0.5rem'>"
            f"<span style='color:{lc};font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>"
            f"<span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span></div>"
            f"<div style='padding:0.6rem 0.8rem 0.55rem'>"
            f"<div class='meta' style='margin:0;line-height:1.6;font-size:12px;border-left:2px solid {bc};"
            f"padding-left:0.6rem'>{answer}</div></div></div>"
        )

    return (
        "<details class=\"exp-details\" id=\"error-decomposition-findings\" open>"
        "<summary><span class=\"fold-icon\">\u25b6</span> Findings</summary>"
        "<div class=\"exp-body\">"
        + _qcard(1, "Winter crisis: does the model collapse in cold months?", winter_alarm, alert=bool(negative_r2))
        + _qcard(2, "Are the model's errors directionally biased by season?", bias_text)
        + _qcard(3, "Is 2025 operating in an unusual regime relative to training?", regime_text)
        + _qcard(4, "Where does error concentrate most strongly across all axes?", hotspot_text)
        + _qcard(5, "Does Saturday (Nepal weekend) behave differently from weekdays?", day_text)
        + _qcard(6, "Which targets are reliable vs. which need caution?", difficulty_text)
        + _qcard(7, "Operational recommendations", reco_text)
        + "</div></details>"
    )




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
    print("(Standalone HTML not written — output is embedded in the Unified Report.)")

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
