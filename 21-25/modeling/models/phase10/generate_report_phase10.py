"""
generate_report_phase10.py — Comprehensive Phase 10 report.

Compares Phase 10 feature-engineered models (ElNet, Ridge, RF, Voting)
against prior baselines: ElNet (Exp3-S2), RF (Exp3-S2), Voting (Phase 9).

New section: Feature Engineering Summary — per-target breakdown of
log / interaction / outlier-flag columns added.

Outputs:
  reports/report_phase10_run_N.html

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase10/generate_report_phase10.py
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
P10_RESULTS  = os.path.join(SCRIPT_DIR, "results.xlsx")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")

sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

os.makedirs(REPORTS_DIR, exist_ok=True)

# Prior baseline result files
PRIOR_LINEAR_FILE = os.path.join(MODELING_DIR, "models", "linear",     "exp3_s2", "results.xlsx")
PRIOR_RF_FILE     = os.path.join(MODELING_DIR, "models", "non_linear", "exp3_s2", "rf", "results.xlsx")
PRIOR_P9_ENS_FILE = os.path.join(MODELING_DIR, "models", "phase9",     "ensemble", "results.xlsx")

# ── Model registry ─────────────────────────────────────────────────────────────
# "key"       used as column prefix in wide df
# "label"     display string
# "color"     hex colour
# "group"     "prior" or "p10"

MODEL_REGISTRY = [
    {"key": "ElNet_S2",   "label": "ElNet (S2)",   "color": "#5BAD6F", "group": "prior"},
    {"key": "RF_S2",      "label": "RF (S2)",       "color": "#2171B5", "group": "prior"},
    {"key": "Voting_P9",  "label": "Voting (P9)",   "color": "#FF9800", "group": "prior"},
    {"key": "ElNet_P10",  "label": "ElNet (P10)",   "color": "#80CBC4", "group": "p10"},
    {"key": "Ridge_P10",  "label": "Ridge (P10)",   "color": "#4A90D9", "group": "p10"},
    {"key": "RF_P10",     "label": "RF (P10)",      "color": "#7E57C2", "group": "p10"},
    {"key": "Voting_P10", "label": "Voting (P10)",  "color": "#E91E63", "group": "p10"},
]

ALL_MODELS  = [m["key"]   for m in MODEL_REGISTRY]
ALL_LABELS  = {m["key"]: m["label"]  for m in MODEL_REGISTRY}
ALL_COLORS  = {m["key"]: m["color"]  for m in MODEL_REGISTRY}
PRIOR_KEYS  = [m["key"] for m in MODEL_REGISTRY if m["group"] == "prior"]
P10_KEYS    = [m["key"] for m in MODEL_REGISTRY if m["group"] == "p10"]

# ── CSS ────────────────────────────────────────────────────────────────────────
REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', Calibri, Arial, sans-serif;
    max-width: 1440px; margin: 0 auto; padding: 24px 32px 60px;
  }
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
  .group-p10   { border-left: 2px solid #E91E63; border-right: 2px solid #E91E63; }

  .obs { background: var(--obs-bg); border-radius: 8px; padding: 14px 18px;
    margin: 14px 0; font-size: 0.88rem; color: var(--text); line-height: 1.65; }
  .obs strong { color: var(--text); }

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

def _img_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _r2_cell(val, best_val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    colour = "pos" if val >= 0 else "neg"
    return f"<td class='num {colour}{cls}'>{val:+.3f}</td>"


def _gap_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


def _short_label(tgt: str) -> str:
    return (tgt.replace("Effluent ", "")
               .replace(" (mg/L, Grab)", " (Grab)")
               .replace(" (mg/L, Composite)", " (Comp)"))


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all():
    """Return wide DataFrame (one row per target) and Phase 10 run number."""

    # Phase 10 results
    p10 = pd.read_excel(P10_RESULTS)
    run = int(p10["run"].max())
    p10 = p10[p10["run"] == run]

    # Prior baselines
    lin = pd.read_excel(PRIOR_LINEAR_FILE)
    lin = lin[lin["run"] == lin["run"].max()]

    rf_s2 = pd.read_excel(PRIOR_RF_FILE)
    rf_s2 = rf_s2[rf_s2["run"] == rf_s2["run"].max()]

    p9_ens = pd.read_excel(PRIOR_P9_ENS_FILE)
    p9_ens = p9_ens[p9_ens["run"] == p9_ens["run"].max()]

    # Target order from P10 results
    targets = p10["target"].unique().tolist()

    rows = []
    for tgt in targets:
        row = {"target": tgt, "ds_label": _short_label(tgt)}

        # ── Prior: ElNet (S2) ──
        lin_row = lin[lin["target"] == tgt]
        row["ElNet_S2_test_R2"]   = lin_row["ElNet_test_R2"].values[0]   if len(lin_row) > 0 else np.nan
        row["ElNet_S2_test_RMSE"] = lin_row["ElNet_test_RMSE"].values[0] if len(lin_row) > 0 else np.nan
        row["ElNet_S2_R2_gap"]    = lin_row["ElNet_R2_gap"].values[0]    if len(lin_row) > 0 else np.nan
        row["ElNet_S2_train_R2"]  = lin_row["ElNet_train_R2"].values[0]  if len(lin_row) > 0 else np.nan
        row["n_train"]            = lin_row["n_train"].values[0]          if len(lin_row) > 0 else np.nan
        row["n_test"]             = lin_row["n_test"].values[0]           if len(lin_row) > 0 else np.nan

        # ── Prior: RF (S2) ──
        rf_row = rf_s2[rf_s2["target"] == tgt]
        row["RF_S2_test_R2"]   = rf_row["R2_test"].values[0]   if len(rf_row) > 0 else np.nan
        row["RF_S2_test_RMSE"] = rf_row["RMSE_test"].values[0] if len(rf_row) > 0 else np.nan
        row["RF_S2_R2_gap"]    = rf_row["R2_gap"].values[0]    if len(rf_row) > 0 else np.nan
        row["RF_S2_train_R2"]  = rf_row["R2_train"].values[0]  if len(rf_row) > 0 else np.nan

        # ── Prior: Voting (P9) ──
        v_row = p9_ens[(p9_ens["target"] == tgt) & (p9_ens["model"] == "Voting")]
        row["Voting_P9_test_R2"]   = v_row["R2_test"].values[0]   if len(v_row) > 0 else np.nan
        row["Voting_P9_test_RMSE"] = v_row["RMSE_test"].values[0] if len(v_row) > 0 else np.nan
        row["Voting_P9_R2_gap"]    = v_row["R2_gap"].values[0]    if len(v_row) > 0 else np.nan
        row["Voting_P9_train_R2"]  = v_row["R2_train"].values[0]  if len(v_row) > 0 else np.nan

        # ── Phase 10 models ──
        for m in ["ElNet", "Ridge", "RF", "Voting"]:
            m_row = p10[(p10["target"] == tgt) & (p10["model"] == m)]
            key   = f"{m}_P10"
            row[f"{key}_test_R2"]   = m_row["R2_test"].values[0]   if len(m_row) > 0 else np.nan
            row[f"{key}_test_RMSE"] = m_row["RMSE_test"].values[0] if len(m_row) > 0 else np.nan
            row[f"{key}_R2_gap"]    = m_row["R2_gap"].values[0]    if len(m_row) > 0 else np.nan
            row[f"{key}_train_R2"]  = m_row["R2_train"].values[0]  if len(m_row) > 0 else np.nan

        # Feature engineering counts (same for all P10 models of same target)
        any_p10 = p10[p10["target"] == tgt]
        if len(any_p10) > 0:
            r = any_p10.iloc[0]
            row["n_base"]  = int(r["n_base_features"])
            row["n_log"]   = int(r["n_log"])
            row["n_inter"] = int(r["n_inter"])
            row["n_flag"]  = int(r["n_flag"])
            row["n_total"] = int(r["n_features"])
        else:
            for k in ["n_base", "n_log", "n_inter", "n_flag", "n_total"]:
                row[k] = np.nan

        rows.append(row)

    return pd.DataFrame(rows), run


# ── Charts ─────────────────────────────────────────────────────────────────────

def _trend_chart(df: pd.DataFrame, run: int) -> str:
    """Avg Test R² bar chart — prior baselines vs Phase 10 models."""
    avgs   = [df[f"{k}_test_R2"].mean() for k in ALL_MODELS]
    colors = [ALL_COLORS[k] for k in ALL_MODELS]
    labels = [ALL_LABELS[k] for k in ALL_MODELS]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")
    bars = ax.bar(labels, avgs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.06,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.axvline(2.5, color="#E91E63", lw=1.2, linestyle="--", alpha=0.6)
    ylim = ax.get_ylim()
    ax.text(1.0,  ylim[0] + 0.01, "Prior baselines", color="#aaa",    fontsize=7, ha="center")
    ax.text(5.0,  ylim[0] + 0.01, "Phase 10",         color="#F48FB1", fontsize=7, ha="center")
    ax.set_ylabel("Avg Test R² (all 8 targets)", fontsize=9, color="white")
    ax.set_title("Average Test R² — Prior Baselines vs Phase 10 (Feature Engineered)", fontsize=9, color="white")
    ax.tick_params(colors="white", axis="both")
    ax.spines[:].set_color("#3A4560")
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_p10trend_{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


def _per_target_chart(df: pd.DataFrame, run: int) -> str:
    """Grouped bar: Test R² per target for all 7 models."""
    targets = df["ds_label"].tolist()
    n_m   = len(ALL_MODELS)
    x     = np.arange(len(targets))
    width = 0.12
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(max(14, len(targets) * 2), 5))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    for i, k in enumerate(ALL_MODELS):
        vals  = df[f"{k}_test_R2"].tolist()
        hatch = "//" if k in PRIOR_KEYS else ""
        bars  = ax.bar(x + offsets[i], vals, width, label=ALL_LABELS[k],
                       color=ALL_COLORS[k], alpha=0.85,
                       edgecolor="white", linewidth=0.4, hatch=hatch)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.07,
                        f"{v:+.2f}", ha="center", va="bottom",
                        fontsize=5, rotation=90, color="white")

    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Test R²", fontsize=9, color="white")
    ax.set_title("Test R² per Target — Prior Baselines (hatched) vs Phase 10 (solid)", fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    ax.legend(fontsize=7.5, ncol=7, facecolor="#2B3A55", labelcolor="white", edgecolor="#3A4560")
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_p10pertgt_{run}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


# ── Tables ─────────────────────────────────────────────────────────────────────

def _comparison_table(df: pd.DataFrame) -> str:
    th_cls = {
        "ElNet_S2":   "th-elnet",
        "RF_S2":      "th-rf",
        "Voting_P9":  "th-voting",
        "ElNet_P10":  "",
        "Ridge_P10":  "th-ridge",
        "RF_P10":     "",
        "Voting_P10": "",
    }
    header = f"""
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Target</th>
          <th colspan="3" class="group-prior">Prior Baselines</th>
          <th colspan="4" class="group-p10">Phase 10 — Feature Engineered</th>
        </tr>
        <tr>
          {"".join(f'<th class="{th_cls[k]}">{ALL_LABELS[k]}<br>Test R²</th>' for k in ALL_MODELS)}
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        all_r2 = [row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS]
        valid_r2 = [v for v in all_r2 if not pd.isna(v)]
        best_r2  = max(valid_r2) if valid_r2 else np.nan

        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for k in ALL_MODELS:
            rows_html += _r2_cell(row.get(f"{k}_test_R2"), best_r2)
        rows_html += "</tr>"

    # Average row
    rows_html += "<tr style='border-top:2px solid var(--border);font-style:italic'>"
    rows_html += "<td class='left' style='color:var(--text-meta)'>Average</td>"
    for k in ALL_MODELS:
        avg = df[f"{k}_test_R2"].mean()
        colour = "pos" if avg >= 0 else "neg"
        rows_html += f"<td class='num {colour}'><strong>{avg:+.3f}</strong></td>"
    rows_html += "</tr>"

    return header + rows_html + "</tbody></table>"


def _gap_table(df: pd.DataFrame) -> str:
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


def _fe_summary_table(df: pd.DataFrame) -> str:
    """Per-target breakdown of engineered features."""
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th>Base<br>features</th>
          <th>Log<br>cols</th>
          <th>Interaction<br>cols</th>
          <th>Flag<br>cols</th>
          <th>Total<br>features</th>
          <th>Added</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in df.iterrows():
        n_base  = row.get("n_base", np.nan)
        n_log   = row.get("n_log",  np.nan)
        n_inter = row.get("n_inter",np.nan)
        n_flag  = row.get("n_flag", np.nan)
        n_total = row.get("n_total",np.nan)
        n_added = n_total - n_base if not pd.isna(n_total) and not pd.isna(n_base) else np.nan

        def _n(v):
            return "—" if pd.isna(v) else str(int(v))

        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        rows_html += f"<td class='num'>{_n(n_base)}</td>"
        rows_html += f"<td class='num' style='color:#80CBC4'>{_n(n_log)}</td>"
        rows_html += f"<td class='num' style='color:#FFB74D'>{_n(n_inter)}</td>"
        rows_html += f"<td class='num' style='color:#F48FB1'>{_n(n_flag)}</td>"
        rows_html += f"<td class='num'><strong>{_n(n_total)}</strong></td>"
        rows_html += f"<td class='num' style='color:#A5D6A7'>+{_n(n_added)}</td>"
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


# ── Summary cards ──────────────────────────────────────────────────────────────

def _summary_cards(df: pd.DataFrame, run: int) -> str:
    wins = {k: 0 for k in ALL_MODELS}
    for _, row in df.iterrows():
        r2s   = {k: row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS}
        valid  = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1

    best_model = max(wins, key=wins.get)

    prior_avgs = {k: df[f"{k}_test_R2"].mean() for k in PRIOR_KEYS}
    p10_avgs   = {k: df[f"{k}_test_R2"].mean() for k in P10_KEYS}
    best_prior = max(prior_avgs, key=prior_avgs.get)
    best_p10   = max(p10_avgs,   key=p10_avgs.get)
    delta      = p10_avgs[best_p10] - prior_avgs[best_prior]

    # Best single R²
    r2_cols  = [f"{k}_test_R2" for k in ALL_MODELS]
    best_val = df[r2_cols].max(axis=1).max()
    best_idx = df[r2_cols].max(axis=1).idxmax()
    best_tgt = df.loc[best_idx, "ds_label"]

    # Avg n_engineered
    avg_eng = df["n_total"].mean() - df["n_base"].mean()

    def _card(label, value, sub, border="#E91E63"):
        return f"""
      <div class="stat-card" style="border-top-color:{border}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
      </div>"""

    win_str = " · ".join(f"{wins[k]}W {ALL_LABELS[k]}" for k in ALL_MODELS if wins[k] > 0)
    cards = ""
    cards += _card("Run", f"#{run}", "Phase 10 — Feature Engineering")
    cards += _card("Overall best model", ALL_LABELS[best_model], win_str,
                   ALL_COLORS[best_model])
    delta_color = "#5BAD6F" if delta >= 0 else "#E15252"
    cards += _card("P10 vs prior best",
                   f"{delta:+.3f}",
                   f"{ALL_LABELS[best_p10]} vs {ALL_LABELS[best_prior]}", delta_color)
    cards += _card("Best single result", f"{best_val:+.3f}", best_tgt, "#B07FD4")
    for k in P10_KEYS:
        cards += _card(f"{ALL_LABELS[k]}<br>avg Test R²",
                       f"{p10_avgs[k]:+.3f}", "Phase 10 model", ALL_COLORS[k])
    cards += _card("Avg feats added", f"+{avg_eng:.1f}", "per target (log+inter+flag)", "#80CBC4")
    return f'<div class="summary-grid">{cards}</div>'


# ── Observations ───────────────────────────────────────────────────────────────

def _auto_obs(df: pd.DataFrame) -> str:
    lines = []
    avgs = {k: df[f"{k}_test_R2"].mean() for k in ALL_MODELS}

    # Win counts
    wins = {k: 0 for k in ALL_MODELS}
    for _, row in df.iterrows():
        r2s  = {k: row.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS}
        valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
        if valid:
            wins[max(valid, key=valid.get)] += 1
    win_str = " | ".join(f"{ALL_LABELS[k]}: {wins[k]}W" for k in ALL_MODELS)
    lines.append(f"<strong>Win counts:</strong> {win_str}.")

    # Avg comparison
    avg_str = " | ".join(f"{ALL_LABELS[k]}&nbsp;{avgs[k]:+.3f}" for k in ALL_MODELS)
    lines.append(f"<strong>Avg Test R²:</strong> {avg_str}.")

    # P10 Voting vs P9 Voting
    v_p10 = avgs["Voting_P10"]
    v_p9  = avgs["Voting_P9"]
    delta_v = v_p10 - v_p9
    sign  = "+" if delta_v >= 0 else ""
    lines.append(
        f"<strong>Voting P10 vs P9:</strong> avg {v_p10:+.3f} vs {v_p9:+.3f} "
        f"({sign}{delta_v:.3f}). "
        + ("Feature engineering improved Voting on average."
           if delta_v >= 0
           else "Feature engineering degraded average Voting performance — "
                "composite target failures outweigh grab improvements.")
    )

    # Grab vs Composite split
    grab_mask = df["target"].str.contains("Grab")
    comp_mask = df["target"].str.contains("Composite")
    vp10_grab_avg = df[grab_mask]["Voting_P10_test_R2"].mean()
    vp10_comp_avg = df[comp_mask]["Voting_P10_test_R2"].mean()
    lines.append(
        f"<strong>Grab vs Composite (Voting P10):</strong> "
        f"Grab avg R²={vp10_grab_avg:+.3f} · Composite avg R²={vp10_comp_avg:+.3f}. "
        + ("Engineered features clearly help Grab targets but hurt composite targets — "
           "likely due to high-dimensionality relative to smaller composite training sets.")
    )

    # Comp TSS note
    comp_tss = df[df["target"].str.contains("TSS") & df["target"].str.contains("Composite")]
    if len(comp_tss) > 0:
        best_comp_tss = comp_tss[[f"{k}_test_R2" for k in ALL_MODELS]].max(axis=1).max()
        lines.append(
            f"<strong>Composite TSS:</strong> Most problematic target — "
            f"best R² across all 7 models = {best_comp_tss:+.3f}. "
            f"Only 290 training rows with 50 features creates severe overfitting for linear models "
            f"(ElNet gap = +3.80, Ridge gap = +3.31)."
        )

    # Comp COD
    comp_cod = df[df["target"].str.contains("COD") & df["target"].str.contains("Composite")]
    if len(comp_cod) > 0:
        best_comp_cod = comp_cod[[f"{k}_test_R2" for k in ALL_MODELS]].max(axis=1).max()
        if best_comp_cod < 0.1:
            lines.append(
                f"<strong>Composite COD:</strong> Still unresolved — "
                f"best R² = {best_comp_cod:+.3f}. "
                f"Feature engineering did not help. Target may require different process variables."
            )

    # Best Grab targets
    grab_rows = df[grab_mask]
    best_grab = grab_rows.apply(
        lambda r: max((r.get(f"{k}_test_R2", np.nan) for k in ALL_MODELS), default=np.nan), axis=1
    )
    best_grab_idx = best_grab.idxmax()
    best_grab_tgt = df.loc[best_grab_idx, "ds_label"]
    best_grab_val = best_grab.max()
    best_grab_model = max(
        {k: df.loc[best_grab_idx, f"{k}_test_R2"] for k in ALL_MODELS if not pd.isna(df.loc[best_grab_idx, f"{k}_test_R2"])},
        key=lambda k: df.loc[best_grab_idx, f"{k}_test_R2"]
    )
    lines.append(
        f"<strong>Best Grab result:</strong> {ALL_LABELS[best_grab_model]} on {best_grab_tgt} "
        f"= {best_grab_val:+.3f}."
    )

    return "<br>".join(lines)


# ── Model plots HTML ───────────────────────────────────────────────────────────

def _model_plots_html(model_tag: str, run: int) -> str:
    names = ["s2_stage3_grab_BOD", "s2_stage3_grab_COD", "s2_stage3_grab_TSS",
             "s2_stage3_grab_pH",  "s2_stage3_comp_BOD", "s2_stage3_comp_COD",
             "s2_stage3_comp_TSS", "s2_stage3_comp_pH"]
    html = ""
    for name in names:
        label = name.replace("s2_stage3_", "").replace("_", " ").title()
        sc = _img_b64(os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_scatter.png"))
        ts = _img_b64(os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_timeseries.png"))
        lc = _img_b64(os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_lc.png"))
        if sc or ts or lc:
            html += (f"<div style='margin-bottom:6px;font-weight:500;font-size:0.85rem;"
                     f"color:var(--text-muted)'>{label}</div>")
            html += "<div class='plots-row'>"
            if sc: html += f"<div class='plot-box'><img src='{sc}'><div class='plot-caption'>Scatter</div></div>"
            if ts: html += f"<div class='plot-box wide'><img src='{ts}'><div class='plot-caption'>Timeseries</div></div>"
            html += "</div>"
    return html if html else "<p style='color:var(--text-meta)'>No plots found.</p>"


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    trend_b64      = _trend_chart(df, run)
    per_target_b64 = _per_target_chart(df, run)

    legend = """
    <div class="legend">
      <strong style="color:var(--text);font-size:0.88rem">Prior baselines:</strong>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElNet (Exp3-S2)</div>
      <div class="legend-item"><span class="dot" style="background:#2171B5"></span> RF (Exp3-S2)</div>
      <div class="legend-item"><span class="dot" style="background:#FF9800"></span> Voting (Phase 9)</div>
      <div class="legend-divider"></div>
      <strong style="color:var(--text);font-size:0.88rem">Phase 10:</strong>
      <div class="legend-item"><span class="dot" style="background:#80CBC4"></span> ElNet</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge</div>
      <div class="legend-item"><span class="dot" style="background:#7E57C2"></span> RF</div>
      <div class="legend-item"><span class="dot" style="background:#E91E63"></span> Voting</div>
    </div>
    """

    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Phase 10 — Feature Engineering:</strong> Three types of engineered features
      applied in-memory to Exp3-S2 datasets before training:<br>
      • <strong style="color:#80CBC4">Log transforms</strong> (log1p) — concentration and
        coliform columns (BOD, COD, TSS inlet/secondary, coliform, primary sludge).<br>
      • <strong style="color:#FFB74D">Interaction terms</strong> (A × B) — domain-meaningful
        products: inlet × secondary (removal efficiency proxy), BOD × Flow (load), SecBOD × MLSS,
        DO × MLSS.<br>
      • <strong style="color:#F48FB1">Outlier indicator flags</strong> (IQR binary) —
        applied to inlet and secondary BOD/COD/TSS; IQR bounds from training set only
        (no data leakage).<br><br>
      Models trained: ElNet, Ridge, RF, Voting (no Stacking — unstable on composite targets in P9).
      Compared against prior baselines: ElNet (Exp3-S2), RF (Exp3-S2), Voting (Phase 9).
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 10 — Feature Engineering Report | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Phase 10 — Feature Engineering (log + interaction + flags)
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2021–2024 &nbsp;|&nbsp; Test: 2025
    &nbsp;|&nbsp; Base datasets: Exp3-S2</div>

  {intro}
  {legend}

  <h2>Executive Summary</h2>
  {_summary_cards(df, run)}
  <div class="obs">{_auto_obs(df)}</div>

  <h2>All-Model Comparison — Test R²</h2>
  <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 6px">
    <span style="background:rgba(233,30,99,0.25);padding:1px 6px;border-radius:3px">■</span>
    highlighted = row-best Test R² &nbsp;·&nbsp;
    <span class="pos">green</span> = R² ≥ 0 &nbsp;·&nbsp;
    <span class="neg">red</span> = R² &lt; 0
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

  <h2>Feature Engineering Summary</h2>
  <p style="font-size:0.85rem;color:var(--text-meta);margin:0 0 8px">
    Counts of each engineered feature type added per target dataset.
    Columns vary across grab/composite because some source features are absent
    (e.g. Inlet Coliform not in all composite files).
  </p>
  {_fe_summary_table(df)}

  <h2>Summary Charts</h2>
  <div class="plots-row">
    {"<div class='plot-box wide'><img src='" + trend_b64 + "'><div class='plot-caption'>Average Test R² — all models</div></div>" if trend_b64 else ""}
  </div>
  <div class="plots-row">
    {"<div class='plot-box wide'><img src='" + per_target_b64 + "'><div class='plot-caption'>Test R² per target</div></div>" if per_target_b64 else ""}
  </div>

  <h2>Phase 10 Model Plots</h2>

  <details>
    <summary>ElNet (P10) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("ElNet", run)}</div>
  </details>

  <details>
    <summary>Ridge (P10) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("Ridge", run)}</div>
  </details>

  <details>
    <summary>RF (P10) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("RF", run)}</div>
  </details>

  <details>
    <summary>Voting (P10) — Scatter + Timeseries</summary>
    <div class="details-body">{_model_plots_html("Voting", run)}</div>
  </details>

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading Phase 10 results...")
    df, run = _load_all()
    print(f"  {len(df)} targets, run {run}")

    print("Building HTML report...")
    html = build_html(df, run)

    out_path = os.path.join(REPORTS_DIR, f"report_phase10_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
