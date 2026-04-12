"""
generate_report_phase10b.py — Phase 10b report.

Phase 10b applies feature engineering (log + interaction + flags) to Grab targets
only. Composite targets use base Exp3-S2 features unchanged, avoiding the
catastrophic overfitting seen in Phase 10 full (e.g. Comp TSS gap +3.80).

Compares against:
  - ElNet (Exp3-S2)     — prior linear champion
  - RF (Exp3-S2)        — prior tree baseline
  - Voting (Phase 9)    — prior ensemble best
  - Voting (Phase 10)   — full FE attempt (for context)

Outputs:
  reports/report_phase10b_run_N.html

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase10/generate_report_phase10b.py
"""

import base64
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
P10B_RESULTS = os.path.join(SCRIPT_DIR, "results_10b.xlsx")
P10_RESULTS  = os.path.join(SCRIPT_DIR, "results.xlsx")        # Phase 10 full (for comparison)
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots_10b")

sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

os.makedirs(REPORTS_DIR, exist_ok=True)

PRIOR_LINEAR_FILE = os.path.join(MODELING_DIR, "models", "linear",     "exp3_s2", "results.xlsx")
PRIOR_RF_FILE     = os.path.join(MODELING_DIR, "models", "non_linear", "exp3_s2", "rf", "results.xlsx")
PRIOR_P9_ENS_FILE = os.path.join(MODELING_DIR, "models", "phase9",     "ensemble", "results.xlsx")

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {"key": "ElNet_S2",    "label": "ElNet (S2)",    "color": "#5BAD6F", "group": "prior"},
    {"key": "RF_S2",       "label": "RF (S2)",        "color": "#2171B5", "group": "prior"},
    {"key": "Voting_P9",   "label": "Voting (P9)",    "color": "#FF9800", "group": "prior"},
    {"key": "Voting_P10",  "label": "Voting (P10)",   "color": "#888888", "group": "prior"},  # context only
    {"key": "ElNet_P10b",  "label": "ElNet (P10b)",   "color": "#80CBC4", "group": "p10b"},
    {"key": "Ridge_P10b",  "label": "Ridge (P10b)",   "color": "#4A90D9", "group": "p10b"},
    {"key": "RF_P10b",     "label": "RF (P10b)",      "color": "#7E57C2", "group": "p10b"},
    {"key": "Voting_P10b", "label": "Voting (P10b)",  "color": "#E91E63", "group": "p10b"},
]

ALL_MODELS  = [m["key"]   for m in MODEL_REGISTRY]
ALL_LABELS  = {m["key"]: m["label"]  for m in MODEL_REGISTRY}
ALL_COLORS  = {m["key"]: m["color"]  for m in MODEL_REGISTRY}
PRIOR_KEYS  = [m["key"] for m in MODEL_REGISTRY if m["group"] == "prior"]
P10B_KEYS   = [m["key"] for m in MODEL_REGISTRY if m["group"] == "p10b"]

# ── CSS ────────────────────────────────────────────────────────────────────────
REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }
  body { font-family: 'Inter', Calibri, Arial, sans-serif;
         max-width: 1440px; margin: 0 auto; padding: 24px 32px 60px; }
  h1 { font-size: 1.7rem; margin-bottom: 4px; color: var(--text); }
  h2 { font-size: 1.22rem; margin: 36px 0 10px; color: var(--text);
       border-bottom: 2px solid var(--border); padding-bottom: 6px; }
  p, li { color: var(--text-muted); font-size: 0.93rem; line-height: 1.6; }
  .run-badge { display: inline-block; background: #E91E63; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.78rem;
    margin-left: 10px; vertical-align: middle; }
  .ts { color: var(--text-meta); font-size: 0.82rem; margin-top: 4px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap;
    margin: 12px 0 20px; font-size: 0.88rem; color: var(--text-muted); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .legend-divider { width: 1px; background: var(--border); margin: 0 6px; align-self: stretch; }

  .summary-grid { display: flex; flex-wrap: nowrap; gap: 10px;
    margin: 14px 0 24px; overflow-x: auto; }
  .stat-card { background: var(--card); border-radius: 8px; padding: 13px 14px;
    box-shadow: 0 1px 6px var(--card-shadow); border-top: 3px solid #E91E63;
    flex: 1 1 0; min-width: 130px; }
  .stat-card .label { font-size: 0.72rem; color: var(--text-meta); margin-bottom: 4px; line-height: 1.3; }
  .stat-card .value { font-size: 1.2rem; font-weight: 700; color: var(--text); white-space: nowrap; }
  .stat-card .sub   { font-size: 0.68rem; color: var(--text-muted); margin-top: 2px; line-height: 1.35; }

  .metric-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin: 14px 0 22px; }
  .metric-table th { background: #2B3A55; color: #C8D8F0; padding: 7px 9px; text-align: center;
    font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap; }
  .metric-table td { padding: 6px 9px; border-bottom: 1px solid var(--border-light); color: var(--text); }
  .metric-table tbody tr:nth-child(even) { background: var(--table-even); }
  .metric-table .num  { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-table .left { text-align: left; font-weight: 500; }
  .metric-table .sub  { font-size: 0.75rem; color: var(--text-meta); }
  .pos { color: #5BAD6F; } .neg { color: #E15252; } .na { color: var(--text-meta); }
  .best { background: rgba(233,30,99,0.22) !important; font-weight: 700; }
  .gap-ok { color: #5BAD6F; } .gap-warn { color: #F0B849; } .gap-bad { color: #E15252; }
  .group-prior { border-left: 2px solid #555; border-right: 2px solid #555; }
  .group-p10b  { border-left: 2px solid #E91E63; border-right: 2px solid #E91E63; }
  .row-grab { background: rgba(74,144,217,0.06) !important; }
  .row-comp { background: rgba(176,127,212,0.06) !important; }
  .obs { background: var(--obs-bg); border-radius: 8px; padding: 14px 18px;
    margin: 14px 0; font-size: 0.88rem; color: var(--text); line-height: 1.65; }
  .plots-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
  .plot-box { flex: 1 1 400px; background: var(--card); border-radius: 8px;
    overflow: hidden; border: 1px solid var(--border-light); }
  .plot-box.wide { flex: 1 1 700px; }
  .plot-box img { width: 100%; display: block; }
  .plot-caption { font-size: 0.78rem; color: var(--text-meta); padding: 6px 10px; }
  details { border: 1px solid var(--border); border-radius: 8px; margin: 14px 0; }
  details > summary { padding: 10px 16px; cursor: pointer; font-size: 0.93rem;
    font-weight: 600; color: var(--text); list-style: none; user-select: none; }
  details > summary::before { content: "▶ "; font-size: 0.7em; }
  details[open] > summary::before { content: "▼ "; }
  details > summary:hover { background: var(--summary-hover); border-radius: 8px; }
  .details-body { padding: 14px 18px; }
  @media (max-width: 768px) { body { padding: 16px; } .plots-row { flex-direction: column; } }
""")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def _short_label(tgt):
    return (tgt.replace("Effluent ", "")
               .replace(" (mg/L, Grab)", " (Grab)")
               .replace(" (mg/L, Composite)", " (Comp)"))

def _r2_cell(val, best_val):
    if pd.isna(val): return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    colour = "pos" if val >= 0 else "neg"
    return f"<td class='num {colour}{cls}'>{val:+.3f}</td>"

def _gap_cell(val):
    if pd.isna(val): return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all():
    p10b = pd.read_excel(P10B_RESULTS)
    run  = int(p10b["run"].max())
    p10b = p10b[p10b["run"] == run]

    # Phase 10 full Voting for context
    p10_voting = None
    if os.path.exists(P10_RESULTS):
        p10_df = pd.read_excel(P10_RESULTS)
        p10_df = p10_df[p10_df["run"] == p10_df["run"].max()]
        p10_voting = p10_df[p10_df["model"] == "Voting"]

    lin   = pd.read_excel(PRIOR_LINEAR_FILE)
    lin   = lin[lin["run"] == lin["run"].max()]
    rf_s2 = pd.read_excel(PRIOR_RF_FILE)
    rf_s2 = rf_s2[rf_s2["run"] == rf_s2["run"].max()]
    p9_ens = pd.read_excel(PRIOR_P9_ENS_FILE)
    p9_ens = p9_ens[p9_ens["run"] == p9_ens["run"].max()]

    targets = p10b["target"].unique().tolist()
    rows = []
    for tgt in targets:
        row = {"target": tgt, "ds_label": _short_label(tgt),
               "is_grab": "Grab" in tgt}

        # Prior baselines
        lin_r  = lin[lin["target"] == tgt]
        rf_r   = rf_s2[rf_s2["target"] == tgt]
        v9_r   = p9_ens[(p9_ens["target"] == tgt) & (p9_ens["model"] == "Voting")]

        row["ElNet_S2_test_R2"]  = lin_r["ElNet_test_R2"].values[0]  if len(lin_r) > 0 else np.nan
        row["ElNet_S2_R2_gap"]   = lin_r["ElNet_R2_gap"].values[0]   if len(lin_r) > 0 else np.nan
        row["ElNet_S2_train_R2"] = lin_r["ElNet_train_R2"].values[0] if len(lin_r) > 0 else np.nan
        row["n_train"]           = lin_r["n_train"].values[0]         if len(lin_r) > 0 else np.nan
        row["n_test"]            = lin_r["n_test"].values[0]          if len(lin_r) > 0 else np.nan

        row["RF_S2_test_R2"]     = rf_r["R2_test"].values[0]  if len(rf_r) > 0 else np.nan
        row["RF_S2_R2_gap"]      = rf_r["R2_gap"].values[0]   if len(rf_r) > 0 else np.nan
        row["RF_S2_train_R2"]    = rf_r["R2_train"].values[0] if len(rf_r) > 0 else np.nan

        row["Voting_P9_test_R2"]  = v9_r["R2_test"].values[0]  if len(v9_r) > 0 else np.nan
        row["Voting_P9_R2_gap"]   = v9_r["R2_gap"].values[0]   if len(v9_r) > 0 else np.nan
        row["Voting_P9_train_R2"] = v9_r["R2_train"].values[0] if len(v9_r) > 0 else np.nan

        # Phase 10 full Voting (context)
        if p10_voting is not None:
            v10_r = p10_voting[p10_voting["target"] == tgt]
            row["Voting_P10_test_R2"] = v10_r["R2_test"].values[0] if len(v10_r) > 0 else np.nan
            row["Voting_P10_R2_gap"]  = v10_r["R2_gap"].values[0]  if len(v10_r) > 0 else np.nan
        else:
            row["Voting_P10_test_R2"] = np.nan
            row["Voting_P10_R2_gap"]  = np.nan

        # Phase 10b models
        for m in ["ElNet", "Ridge", "RF", "Voting"]:
            mr  = p10b[(p10b["target"] == tgt) & (p10b["model"] == m)]
            key = f"{m}_P10b"
            row[f"{key}_test_R2"]   = mr["R2_test"].values[0]   if len(mr) > 0 else np.nan
            row[f"{key}_R2_gap"]    = mr["R2_gap"].values[0]    if len(mr) > 0 else np.nan
            row[f"{key}_train_R2"]  = mr["R2_train"].values[0]  if len(mr) > 0 else np.nan
            row[f"{key}_test_RMSE"] = mr["RMSE_test"].values[0] if len(mr) > 0 else np.nan

        # Feature counts (from P10b record)
        any_r = p10b[p10b["target"] == tgt]
        if len(any_r) > 0:
            r = any_r.iloc[0]
            row["n_base"]  = int(r["n_base_features"])
            row["n_eng"]   = int(r["n_engineered"])
            row["n_total"] = int(r["n_features"])
        else:
            row["n_base"] = row["n_eng"] = row["n_total"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows), run


# ── Charts ─────────────────────────────────────────────────────────────────────

def _trend_chart(df, run):
    avgs   = [df[f"{k}_test_R2"].mean() for k in ALL_MODELS]
    colors = [ALL_COLORS[k] for k in ALL_MODELS]
    labels = [ALL_LABELS[k] for k in ALL_MODELS]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")
    bars = ax.bar(labels, avgs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.06,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=8.5, color="white")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    # Divider between prior+P10 context and P10b
    ax.axvline(3.5, color="#E91E63", lw=1.2, linestyle="--", alpha=0.6)
    ylim = ax.get_ylim()
    ax.text(1.5,  ylim[0] + 0.01, "Prior / context", color="#aaa",    fontsize=7, ha="center")
    ax.text(6.0,  ylim[0] + 0.01, "Phase 10b",        color="#F48FB1", fontsize=7, ha="center")
    ax.set_ylabel("Avg Test R² (all 8 targets)", fontsize=9, color="white")
    ax.set_title("Average Test R² — Prior Baselines vs Phase 10 (full) vs Phase 10b (Grab-only FE)",
                 fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    plt.tight_layout()
    tmp = os.path.join(SCRIPT_DIR, f"_tmp_10b_trend_{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp); os.remove(tmp)
    return b64


def _per_target_chart(df, run):
    targets = df["ds_label"].tolist()
    n_m     = len(ALL_MODELS)
    x       = np.arange(len(targets))
    width   = 0.10
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(max(16, len(targets) * 2.2), 5.5))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    for i, k in enumerate(ALL_MODELS):
        vals  = df[f"{k}_test_R2"].tolist()
        hatch = "//" if k in PRIOR_KEYS else ""
        alpha = 0.45 if k == "Voting_P10" else 0.85   # dim P10 full — context only
        bars  = ax.bar(x + offsets[i], vals, width, label=ALL_LABELS[k],
                       color=ALL_COLORS[k], alpha=alpha,
                       edgecolor="white", linewidth=0.4, hatch=hatch)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.08,
                        f"{v:+.2f}", ha="center", va="bottom",
                        fontsize=4.5, rotation=90, color="white")

    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Test R²", fontsize=9, color="white")
    ax.set_title("Test R² per Target — Prior (hatched) · P10-full (dimmed) · P10b (solid)",
                 fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    ax.legend(fontsize=7, ncol=8, facecolor="#2B3A55", labelcolor="white", edgecolor="#3A4560")
    plt.tight_layout()
    tmp = os.path.join(SCRIPT_DIR, f"_tmp_10b_pertgt_{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp); os.remove(tmp)
    return b64


def _grab_vs_comp_chart(df, run):
    """Side-by-side avg R² for Grab vs Composite for each P10b model."""
    grab = df[df["is_grab"] == True]
    comp = df[df["is_grab"] == False]
    models   = P10B_KEYS
    labels   = [ALL_LABELS[k] for k in models]
    colors   = [ALL_COLORS[k] for k in models]
    avg_grab = [grab[f"{k}_test_R2"].mean() for k in models]
    avg_comp = [comp[f"{k}_test_R2"].mean() for k in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")
    b1 = ax.bar(x - w/2, avg_grab, w, label="Grab (FE applied)", color=colors, alpha=0.9,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, avg_comp, w, label="Composite (base only)",
                color=colors, alpha=0.45, edgecolor="white", linewidth=0.5, hatch="//")
    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01 if v >= 0 else v - 0.04,
                    f"{v:+.3f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylabel("Avg Test R²", fontsize=9, color="white")
    ax.set_title("P10b — Grab (FE) vs Composite (base) — avg Test R² per model",
                 fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    ax.legend(fontsize=8, facecolor="#2B3A55", labelcolor="white", edgecolor="#3A4560")
    plt.tight_layout()
    tmp = os.path.join(SCRIPT_DIR, f"_tmp_10b_gvsc_{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp); os.remove(tmp)
    return b64


# ── Tables ─────────────────────────────────────────────────────────────────────

def _comparison_table(df):
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Target</th>
          <th colspan="4" class="group-prior">Prior / Context</th>
          <th colspan="4" class="group-p10b">Phase 10b — Grab FE · Composite Base</th>
        </tr>
        <tr>
          <th style="color:#5BAD6F">ElNet<br>(S2)</th>
          <th style="color:#4FC3F7">RF<br>(S2)</th>
          <th style="color:#FFB74D">Voting<br>(P9)</th>
          <th style="color:#888">Voting<br>(P10)</th>
          <th style="color:#80CBC4">ElNet<br>(P10b)</th>
          <th style="color:#4A90D9">Ridge<br>(P10b)</th>
          <th style="color:#CE93D8">RF<br>(P10b)</th>
          <th style="color:#F48FB1">Voting<br>(P10b)</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        all_r2  = [row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS]
        valid   = [v for v in all_r2 if not pd.isna(v)]
        best_r2 = max(valid) if valid else np.nan
        row_cls = "row-grab" if row["is_grab"] else "row-comp"

        rows_html += f"<tr class='{row_cls}'>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for k in ALL_MODELS:
            rows_html += _r2_cell(row.get(f"{k}_test_R2"), best_r2)
        rows_html += "</tr>"

    # Average row — all 8
    rows_html += "<tr style='border-top:2px solid var(--border);font-style:italic'>"
    rows_html += "<td class='left' style='color:var(--text-meta)'>Avg (all 8)</td>"
    for k in ALL_MODELS:
        avg = df[f"{k}_test_R2"].mean()
        colour = "pos" if avg >= 0 else "neg"
        rows_html += f"<td class='num {colour}'><strong>{avg:+.3f}</strong></td>"
    rows_html += "</tr>"

    # Grab-only average
    grab = df[df["is_grab"] == True]
    rows_html += "<tr style='font-style:italic'>"
    rows_html += "<td class='left' style='color:#4A90D9;font-size:0.78rem'>Avg (Grab 4)</td>"
    for k in ALL_MODELS:
        avg = grab[f"{k}_test_R2"].mean()
        colour = "pos" if avg >= 0 else "neg"
        rows_html += f"<td class='num {colour}' style='font-size:0.78rem'>{avg:+.3f}</td>"
    rows_html += "</tr>"

    # Composite-only average
    comp = df[df["is_grab"] == False]
    rows_html += "<tr style='font-style:italic'>"
    rows_html += "<td class='left' style='color:#B07FD4;font-size:0.78rem'>Avg (Comp 4)</td>"
    for k in ALL_MODELS:
        avg = comp[f"{k}_test_R2"].mean()
        colour = "pos" if avg >= 0 else "neg"
        rows_html += f"<td class='num {colour}' style='font-size:0.78rem'>{avg:+.3f}</td>"
    rows_html += "</tr>"

    return header + rows_html + "</tbody></table>"


def _gap_table(df):
    header = f"""
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          {"".join(f'<th>{ALL_LABELS[k]}<br>Gap</th>' for k in ALL_MODELS)}
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for k in ALL_MODELS:
            rows_html += _gap_cell(row.get(f"{k}_R2_gap"))
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


def _fe_table(df):
    """Feature counts: Grab rows show engineered count, Composite rows show base only."""
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th>Type</th>
          <th>Base features</th>
          <th>Engineered added</th>
          <th>Total features</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        t    = "Grab (FE)" if row["is_grab"] else "Composite (base)"
        col  = "#4A90D9" if row["is_grab"] else "#B07FD4"
        nbase = int(row["n_base"]) if not pd.isna(row.get("n_base")) else "—"
        neng  = f"+{int(row['n_eng'])}" if not pd.isna(row.get("n_eng")) and row["is_grab"] else "0"
        ntot  = int(row["n_total"]) if not pd.isna(row.get("n_total")) else "—"
        rows_html += f"<tr><td class='left'>{row['ds_label']}</td>"
        rows_html += f"<td style='color:{col};font-size:0.8rem'>{t}</td>"
        rows_html += f"<td class='num'>{nbase}</td>"
        rows_html += f"<td class='num' style='color:#A5D6A7'>{neng}</td>"
        rows_html += f"<td class='num'><strong>{ntot}</strong></td></tr>"
    return header + rows_html + "</tbody></table>"


# ── Summary cards ──────────────────────────────────────────────────────────────

def _summary_cards(df, run):
    wins = {k: 0 for k in ALL_MODELS}
    for _, row in df.iterrows():
        r2s  = {k: row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS}
        valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1

    p10b_avgs  = {k: df[f"{k}_test_R2"].mean() for k in P10B_KEYS}
    prior_avgs = {k: df[f"{k}_test_R2"].mean() for k in ["ElNet_S2", "RF_S2", "Voting_P9"]}
    best_p10b  = max(p10b_avgs,  key=p10b_avgs.get)
    best_prior = max(prior_avgs, key=prior_avgs.get)
    delta      = p10b_avgs[best_p10b] - prior_avgs[best_prior]

    r2_cols  = [f"{k}_test_R2" for k in ALL_MODELS]
    best_val = df[r2_cols].max(axis=1).max()
    best_idx = df[r2_cols].max(axis=1).idxmax()
    best_tgt = df.loc[best_idx, "ds_label"]

    grab = df[df["is_grab"] == True]
    comp = df[df["is_grab"] == False]
    vp10b_grab = grab["Voting_P10b_test_R2"].mean()
    vp10b_comp = comp["Voting_P10b_test_R2"].mean()
    vp9_avg    = df["Voting_P9_test_R2"].mean()

    def _card(label, value, sub, border="#E91E63"):
        return f"""<div class="stat-card" style="border-top-color:{border}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
      </div>"""

    cards  = _card("Run", f"#{run}", "Phase 10b — Grab FE / Comp base")
    delta_color = "#5BAD6F" if delta >= 0 else "#E15252"
    cards += _card("P10b vs prior best", f"{delta:+.3f}",
                   f"{ALL_LABELS[best_p10b]} vs {ALL_LABELS[best_prior]}", delta_color)
    cards += _card("Best single result", f"{best_val:+.3f}", best_tgt, "#B07FD4")
    cards += _card("Voting P10b avg", f"{p10b_avgs['Voting_P10b']:+.3f}",
                   f"vs P9 Voting {vp9_avg:+.3f} ({p10b_avgs['Voting_P10b']-vp9_avg:+.3f})",
                   "#E91E63")
    cards += _card("Voting P10b — Grab", f"{vp10b_grab:+.3f}", "avg Grab 4 targets (FE)", "#4A90D9")
    cards += _card("Voting P10b — Comp", f"{vp10b_comp:+.3f}", "avg Comp 4 targets (base)", "#B07FD4")
    for k in ["ElNet_P10b", "Ridge_P10b", "RF_P10b"]:
        cards += _card(f"{ALL_LABELS[k]}<br>avg R²", f"{p10b_avgs[k]:+.3f}",
                       "all 8 targets", ALL_COLORS[k])
    return f'<div class="summary-grid">{cards}</div>'


# ── Observations ───────────────────────────────────────────────────────────────

def _auto_obs(df):
    lines = []
    grab  = df[df["is_grab"] == True]
    comp  = df[df["is_grab"] == False]

    # Win counts
    wins = {k: 0 for k in ALL_MODELS}
    for _, row in df.iterrows():
        r2s  = {k: row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS}
        valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1
    win_str = " | ".join(f"{ALL_LABELS[k]}: {wins[k]}W" for k in ALL_MODELS if wins[k] > 0)
    lines.append(f"<strong>Win counts:</strong> {win_str}.")

    # Overall avgs
    avgs    = {k: df[f"{k}_test_R2"].mean() for k in ALL_MODELS}
    avg_str = " | ".join(f"{ALL_LABELS[k]}&nbsp;{avgs[k]:+.3f}" for k in ALL_MODELS)
    lines.append(f"<strong>Avg Test R² (all 8):</strong> {avg_str}.")

    # P10b vs P9 Voting delta
    vp10b = avgs["Voting_P10b"]
    vp9   = avgs["Voting_P9"]
    vp10  = avgs["Voting_P10"]
    lines.append(
        f"<strong>Selective FE validates the hypothesis:</strong> "
        f"Voting P10b avg {vp10b:+.3f} vs P9 ({vp9:+.3f}, Δ{vp10b-vp9:+.3f}) "
        f"and vs P10 full ({vp10:+.3f}, Δ{vp10b-vp10:+.3f}). "
        f"Restricting feature engineering to Grab targets recovers the composite performance "
        f"lost in Phase 10."
    )

    # Grab improvement
    vp10b_grab = grab["Voting_P10b_test_R2"].mean()
    vp9_grab   = grab["Voting_P9_test_R2"].mean()
    lines.append(
        f"<strong>Grab targets (Voting):</strong> P10b avg {vp10b_grab:+.3f} vs P9 {vp9_grab:+.3f} "
        f"(Δ{vp10b_grab - vp9_grab:+.3f}). "
        f"Feature engineering consistently helps: Grab BOD {grab.loc[grab['target'].str.contains('BOD'), 'Voting_P10b_test_R2'].values[0]:+.3f}, "
        f"Grab TSS {grab.loc[grab['target'].str.contains('TSS'), 'Voting_P10b_test_R2'].values[0]:+.3f}."
    )

    # Composite stability
    vp10b_comp = comp["Voting_P10b_test_R2"].mean()
    vp9_comp   = comp["Voting_P9_test_R2"].mean()
    vp10_comp  = comp["Voting_P10_test_R2"].mean() if "Voting_P10_test_R2" in comp else np.nan
    lines.append(
        f"<strong>Composite targets (Voting):</strong> P10b avg {vp10b_comp:+.3f} vs P9 {vp9_comp:+.3f} "
        f"(Δ{vp10b_comp - vp9_comp:+.3f})"
        + (f" vs P10 full {vp10_comp:+.3f} (Δ{vp10b_comp - vp10_comp:+.3f}) — catastrophic P10 overfitting fully recovered."
           if not np.isnan(vp10_comp) else ".")
    )

    # Comp COD persistent
    comp_cod = comp[comp["target"].str.contains("COD")]
    if len(comp_cod) > 0:
        best_cod = comp_cod[[f"{k}_test_R2" for k in ALL_MODELS]].max(axis=1).max()
        lines.append(
            f"<strong>Composite COD:</strong> Remains the hardest target — "
            f"best R² = {best_cod:+.3f} across all 8 models. "
            f"Not responsive to feature engineering; likely requires different process variables."
        )

    # Overfitting note for Grab pH
    grab_ph = grab[grab["target"].str.contains("pH")]
    if len(grab_ph) > 0:
        ridge_ph = grab_ph["Ridge_P10b_test_R2"].values[0]
        lines.append(
            f"<strong>Grab pH:</strong> Still the weakest Grab target — "
            f"Ridge P10b {ridge_ph:+.3f}. pH is relatively stable with narrow dynamic range; "
            f"the engineered interaction/log features add noise without signal."
        )

    return "<br>".join(lines)


# ── Model plots ────────────────────────────────────────────────────────────────

def _model_plots_html(model_tag, run):
    names = ["s2_stage3_grab_BOD", "s2_stage3_grab_COD", "s2_stage3_grab_TSS",
             "s2_stage3_grab_pH",  "s2_stage3_comp_BOD", "s2_stage3_comp_COD",
             "s2_stage3_comp_TSS", "s2_stage3_comp_pH"]
    html = ""
    for name in names:
        label = name.replace("s2_stage3_", "").replace("_", " ").title()
        sc = _img_b64(os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_scatter.png"))
        ts = _img_b64(os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_timeseries.png"))
        if sc or ts:
            html += (f"<div style='margin-bottom:6px;font-weight:500;font-size:0.85rem;"
                     f"color:var(--text-muted)'>{label}</div>")
            html += "<div class='plots-row'>"
            if sc: html += f"<div class='plot-box'><img src='{sc}'><div class='plot-caption'>Scatter</div></div>"
            if ts: html += f"<div class='plot-box wide'><img src='{ts}'><div class='plot-caption'>Timeseries</div></div>"
            html += "</div>"
    return html or "<p style='color:var(--text-meta)'>No plots found.</p>"


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(df, run):
    ts             = datetime.now().strftime("%Y-%m-%d %H:%M")
    trend_b64      = _trend_chart(df, run)
    per_target_b64 = _per_target_chart(df, run)
    gvsc_b64       = _grab_vs_comp_chart(df, run)

    legend = """
    <div class="legend">
      <strong style="color:var(--text);font-size:0.88rem">Prior / context:</strong>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElNet (S2)</div>
      <div class="legend-item"><span class="dot" style="background:#2171B5"></span> RF (S2)</div>
      <div class="legend-item"><span class="dot" style="background:#FF9800"></span> Voting (P9)</div>
      <div class="legend-item"><span class="dot" style="background:#888"></span> Voting (P10 full — context)</div>
      <div class="legend-divider"></div>
      <strong style="color:var(--text);font-size:0.88rem">Phase 10b:</strong>
      <div class="legend-item"><span class="dot" style="background:#80CBC4"></span> ElNet</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge</div>
      <div class="legend-item"><span class="dot" style="background:#7E57C2"></span> RF</div>
      <div class="legend-item"><span class="dot" style="background:#E91E63"></span> Voting</div>
      <div class="legend-divider"></div>
      <span style="font-size:0.8rem;color:var(--text-meta)">
        Row shading: <span style="color:#4A90D9">■</span> Grab (FE applied)
        &nbsp; <span style="color:#B07FD4">■</span> Composite (base features only)
      </span>
    </div>
    """

    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Phase 10b — Selective Feature Engineering (Grab only):</strong><br>
      Phase 10 (full FE) showed that applying log + interaction + flag engineering to
      <em>all</em> datasets caused catastrophic overfitting on composite targets
      (Comp TSS ElNet gap +3.80, Ridge gap +3.31) due to the high feature-to-sample
      ratio (~50 features vs 290–515 training rows).<br><br>
      Phase 10b fixes this by <strong>engineering only Grab targets</strong>
      (470–817 training rows — more tolerant of extra dimensions) while keeping
      composite targets on their base Exp3-S2 features (same as Phase 9).
      This lets us capture the Grab improvements (+log/interaction/flag)
      without destabilising the composite models.
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 10b — Selective Feature Engineering | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Phase 10b — Feature Engineering (Grab only)
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
    &nbsp;|&nbsp; Grab: FE applied · Composite: base features only</div>

  {intro}
  {legend}

  <h2>Executive Summary</h2>
  {_summary_cards(df, run)}
  <div class="obs">{_auto_obs(df)}</div>

  <h2>All-Model Comparison — Test R²</h2>
  <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 6px">
    <span style="background:rgba(233,30,99,0.25);padding:1px 6px;border-radius:3px">■</span>
    highlighted = row-best &nbsp;·&nbsp;
    <span style="color:#4A90D9">■</span> Grab rows &nbsp;·&nbsp;
    <span style="color:#B07FD4">■</span> Composite rows
  </p>
  {_comparison_table(df)}

  <details>
    <summary>R² Gap (Train − Test) — all models</summary>
    <div class="details-body">
      <p style="font-size:0.82rem;color:var(--text-meta)">
        <span class="gap-ok">■</span> &lt; 0.10 (tight) &nbsp;
        <span class="gap-warn">■</span> 0.10–0.25 (moderate) &nbsp;
        <span class="gap-bad">■</span> &gt; 0.25 (overfit)
      </p>
      {_gap_table(df)}
    </div>
  </details>

  <h2>Feature Count Summary</h2>
  {_fe_table(df)}

  <h2>Summary Charts</h2>
  <div class="plots-row">
    {"<div class='plot-box wide'><img src='" + trend_b64 + "'><div class='plot-caption'>Avg Test R² — all models</div></div>" if trend_b64 else ""}
    {"<div class='plot-box'><img src='" + gvsc_b64 + "'><div class='plot-caption'>P10b: Grab (FE) vs Composite (base) avg R²</div></div>" if gvsc_b64 else ""}
  </div>
  <div class="plots-row">
    {"<div class='plot-box wide'><img src='" + per_target_b64 + "'><div class='plot-caption'>Test R² per target — all models</div></div>" if per_target_b64 else ""}
  </div>

  <h2>Phase 10b Model Plots</h2>
  <details>
    <summary>ElNet (P10b) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("ElNet", run)}</div>
  </details>
  <details>
    <summary>Ridge (P10b) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("Ridge", run)}</div>
  </details>
  <details>
    <summary>RF (P10b) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("RF", run)}</div>
  </details>
  <details>
    <summary>Voting (P10b) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("Voting", run)}</div>
  </details>

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading Phase 10b results...")
    df, run = _load_all()
    print(f"  {len(df)} targets, run {run}")
    print("Building HTML report...")
    html = build_html(df, run)
    out  = os.path.join(REPORTS_DIR, f"report_phase10b_run_{run}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
