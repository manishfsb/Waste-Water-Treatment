"""
generate_comparison_report.py — Phase 4: Linear vs Tree Ensemble comparison report.

Merges results from:
  linear_modeling/results.xlsx         (OLS, Ridge, ElasticNet)
  non_linear_modeling/{rf,gb,xgb}/results.xlsx  (RF, GB, XGB)

Produces:
  modeling/report_comparison_run_N.html

Sections:
  • Cross-experiment summary — stat cards + avg metric table (all 6 models)
  • Per-experiment cards:
      - Auto-generated observations (winner, avg R², overfit check)
      - Primary metric table (Test R² | Test RMSE | MAE/MAPE | R² Gap — 6 models)
      - Train-set metrics (collapsed)
      - Side-by-side Test R² bar chart (6 models × 8 targets)
      - Side-by-side Test RMSE bar chart

Usage (from project root):
    .venv/bin/python3 21-25/modeling/generate_comparison_report.py
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
LIN_DIR      = os.path.join(SCRIPT_DIR, "linear_modeling_baseline")
NL_DIR       = os.path.join(SCRIPT_DIR, "non_linear_modeling_baseline")

sys.path.insert(0, SCRIPT_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Design tokens ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "OLS":   "#E15252",
    "Ridge": "#4A90D9",
    "ElNet": "#5BAD6F",
    "RF":    "#2171B5",
    "GB":    "#238B45",
    "XGB":   "#D94801",
}

LINEAR_MODELS = ["OLS", "Ridge", "ElNet"]
NL_MODELS     = ["RF",  "GB",   "XGB"]
ALL_MODELS    = LINEAR_MODELS + NL_MODELS

EXPERIMENT_MAP = {
    "Exp1":      "Experiment 1",
    "Exp2-Sub1": "Experiment 2 Sub-1",
    "Exp2-Sub2": "Experiment 2 Sub-2",
}
EXPERIMENT_LABELS = {
    "Experiment 1":       "Experiment 1 — Inlet Features Only",
    "Experiment 2 Sub-1": "Experiment 2 Sub-1 — Secondary Features Only",
    "Experiment 2 Sub-2": "Experiment 2 Sub-2 — Inlet + Secondary Features",
}
EXPERIMENT_NFEAT = {
    "Experiment 1": 9,
    "Experiment 2 Sub-1": 15,
    "Experiment 2 Sub-2": 19,
}

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, int]:
    """
    Load and merge linear + NL results into one wide DataFrame with columns:
      experiment, target, n_train, n_test,
      <MODEL>_test_R2, <MODEL>_test_RMSE, <MODEL>_test_MAE,
      <MODEL>_train_R2, <MODEL>_train_RMSE, <MODEL>_R2_gap
    for MODEL in ALL_MODELS.
    Returns (df, run).
    """
    # ── Linear ────────────────────────────────────────────────────────────────
    lin = pd.read_excel(os.path.join(LIN_DIR, "results.xlsx"))
    run_lin = int(lin["run"].max())
    lin = lin[lin["run"] == run_lin].copy()
    lin["experiment"] = lin["experiment"].map(EXPERIMENT_MAP)

    # Normalise dataset → target label
    def _lin_label(ds):
        return (ds.replace("Exp1_","").replace("Exp2S1_","").replace("Exp2S2_",""))
    lin["ds_label"] = lin["dataset"].apply(_lin_label)

    # ── Non-linear ────────────────────────────────────────────────────────────
    nl_frames = {}
    for tag in NL_MODELS:
        fp = os.path.join(NL_DIR, tag.lower(), "results.xlsx")
        df = pd.read_excel(fp)
        run = int(df["run"].max())
        nl_frames[tag] = df[df["run"] == run].copy()

    run_nl = int(nl_frames["RF"]["run"].max())

    # ── Build wide table ──────────────────────────────────────────────────────
    # Base: one row per (experiment, target) from linear
    base = lin[["experiment", "target", "n_train", "n_test", "ds_label"]].copy()

    # Add linear metrics
    for m in LINEAR_MODELS:
        mlo = m.lower().replace("elnet", "elnet")   # already fine
        prefix = m  # OLS, Ridge, ElNet
        base[f"{m}_test_R2"]   = lin[f"{prefix}_test_R2"].values
        base[f"{m}_test_RMSE"] = lin[f"{prefix}_test_RMSE"].values
        base[f"{m}_test_MAE"]  = lin[f"{prefix}_test_MAE"].values
        base[f"{m}_test_MAPE"] = lin[f"{prefix}_test_MAPE"].values
        base[f"{m}_train_R2"]  = lin[f"{prefix}_train_R2"].values
        base[f"{m}_train_RMSE"]= lin[f"{prefix}_train_RMSE"].values
        base[f"{m}_R2_gap"]    = lin[f"{prefix}_R2_gap"].values

    # Add NL metrics by matching on experiment + target
    for tag in NL_MODELS:
        nl = nl_frames[tag]
        nl_map = {}
        for _, row in nl.iterrows():
            nl_map[(row["experiment"], row["target"])] = row

        r2_te, rmse_te, mae_te, r2_tr, rmse_tr, gap = [], [], [], [], [], []
        for _, row in base.iterrows():
            key = (row["experiment"], row["target"])
            if key in nl_map:
                r = nl_map[key]
                r2_te.append(r["R2_test"]); rmse_te.append(r["RMSE_test"])
                mae_te.append(r["MAE_test"]); r2_tr.append(r["R2_train"])
                rmse_tr.append(r["RMSE_train"]); gap.append(r["R2_gap"])
            else:
                r2_te.append(np.nan); rmse_te.append(np.nan)
                mae_te.append(np.nan); r2_tr.append(np.nan)
                rmse_tr.append(np.nan); gap.append(np.nan)

        base[f"{tag}_test_R2"]    = r2_te
        base[f"{tag}_test_RMSE"]  = rmse_te
        base[f"{tag}_test_MAE"]   = mae_te
        base[f"{tag}_train_R2"]   = r2_tr
        base[f"{tag}_train_RMSE"] = rmse_tr
        base[f"{tag}_R2_gap"]     = gap

    return base, max(run_lin, run_nl)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _img_b64(path):
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


def _rmse_cell(val, best_val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = " best" if abs(val - best_val) < 1e-9 else ""
    return f"<td class='num{cls}'>{val:.3f}</td>"


def _gap_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "gap-ok" if val < 0.1 else ("gap-warn" if val < 0.25 else "gap-bad")
    return f"<td class='num {cls}'>{val:+.3f}</td>"


def _mape_cell(val):
    if pd.isna(val):
        return "<td class='num na'>—</td>"
    cls = "mape-ok" if val < 20 else ("mape-warn" if val < 40 else "mape-bad")
    return f"<td class='num {cls}'>{val:.1f}%</td>"


def _winner(row):
    """Return the model name with the highest Test R²."""
    r2s = {m: row.get(f"{m}_test_R2", np.nan) for m in ALL_MODELS}
    valid = {k: v for k, v in r2s.items() if not pd.isna(v)}
    if not valid:
        return None
    return max(valid, key=valid.get)


# ── Chart generation ───────────────────────────────────────────────────────────

def _bar_chart_b64(sub: pd.DataFrame, metric: str, ylabel: str, exp: str) -> str:
    """
    Grouped bar chart: one group per target, bars = 6 models.
    Linear models use hatched bars; NL solid. Returns base64 PNG string.
    """
    targets = sub["ds_label"].tolist()
    x = np.arange(len(targets))
    n = len(ALL_MODELS)
    width = 0.13
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(max(10, len(targets) * 1.6), 5))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    for i, m in enumerate(ALL_MODELS):
        col = f"{m}_{metric}"
        vals = sub[col].tolist()
        hatch = "//" if m in LINEAR_MODELS else ""
        bars = ax.bar(x + offsets[i], vals, width,
                      label=m, color=MODEL_COLORS[m],
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      hatch=hatch)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if metric == "test_R2" else 0),
                        f"{v:+.2f}" if metric == "test_R2" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90,
                        color="white")

    if metric == "test_R2":
        ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=35, ha="right", fontsize=8, color="white")
    ax.set_ylabel(ylabel, fontsize=9, color="white")
    ax.set_title(f"{ylabel} — {EXPERIMENT_LABELS.get(exp, exp)}\n"
                 f"(hatched = linear,  solid = tree ensemble)",
                 fontsize=9, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#3A4560")
    leg = ax.legend(fontsize=8, ncol=6, loc="upper right",
                    facecolor="#2B3A55", labelcolor="white",
                    edgecolor="#3A4560")
    plt.tight_layout()

    tmp = os.path.join(SCRIPT_DIR, f"_tmp_comp_{exp.replace(' ','')}_{metric}.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = _img_b64(tmp)
    os.remove(tmp)
    return b64


# ── Metric tables ──────────────────────────────────────────────────────────────

def _primary_table(sub: pd.DataFrame) -> str:
    # Two header rows: model groups + metric sub-columns
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th rowspan="2" class="left">Target</th>
          <th rowspan="2">n<br><span class='sub'>train</span></th>
          <th rowspan="2">n<br><span class='sub'>test</span></th>
          <th colspan="3" class="group-lin">Linear Models</th>
          <th colspan="3" class="group-nl">Tree Ensembles</th>
        </tr>
        <tr>
          <th class="th-ols">OLS<br>Test R²</th>
          <th class="th-ridge">Ridge<br>Test R²</th>
          <th class="th-elnet">ElNet<br>Test R²</th>
          <th class="th-rf">RF<br>Test R²</th>
          <th class="th-gb">GB<br>Test R²</th>
          <th class="th-xgb">XGB<br>Test R²</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in sub.iterrows():
        best_r2 = max((row.get(f"{m}_test_R2", np.nan) for m in ALL_MODELS
                       if not pd.isna(row.get(f"{m}_test_R2", np.nan))),
                      default=np.nan)
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        rows_html += f"<td class='num'>{row['n_train']:,}</td>"
        rows_html += f"<td class='num'>{row['n_test']:,}</td>"
        for m in ALL_MODELS:
            rows_html += _r2_cell(row.get(f"{m}_test_R2"), best_r2)
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


def _rmse_table(sub: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th colspan="3" class="group-lin">Linear — Test RMSE</th>
          <th colspan="3" class="group-nl">Tree — Test RMSE</th>
          <th colspan="3" class="group-lin">Linear — R² Gap</th>
          <th colspan="3" class="group-nl">Tree — R² Gap</th>
        </tr>
        <tr>
          <th></th>
          <th class="th-ols">OLS</th>
          <th class="th-ridge">Ridge</th>
          <th class="th-elnet">ElNet</th>
          <th class="th-rf">RF</th>
          <th class="th-gb">GB</th>
          <th class="th-xgb">XGB</th>
          <th class="th-ols">OLS</th>
          <th class="th-ridge">Ridge</th>
          <th class="th-elnet">ElNet</th>
          <th class="th-rf">RF</th>
          <th class="th-gb">GB</th>
          <th class="th-xgb">XGB</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in sub.iterrows():
        best_rmse_vals = [row.get(f"{m}_test_RMSE") for m in ALL_MODELS
                          if not pd.isna(row.get(f"{m}_test_RMSE", np.nan))]
        best_rmse = min(best_rmse_vals) if best_rmse_vals else np.nan
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for m in ALL_MODELS:
            rows_html += _rmse_cell(row.get(f"{m}_test_RMSE"), best_rmse)
        for m in ALL_MODELS:
            rows_html += _gap_cell(row.get(f"{m}_R2_gap"))
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


def _train_table(sub: pd.DataFrame) -> str:
    header = """
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Target</th>
          <th colspan="3" class="group-lin">Linear — Train R²</th>
          <th colspan="3" class="group-nl">Tree — Train R²</th>
        </tr>
        <tr>
          <th></th>
          <th class="th-ols">OLS</th>
          <th class="th-ridge">Ridge</th>
          <th class="th-elnet">ElNet</th>
          <th class="th-rf">RF</th>
          <th class="th-gb">GB</th>
          <th class="th-xgb">XGB</th>
        </tr>
      </thead>
      <tbody>
    """
    rows_html = ""
    for _, row in sub.iterrows():
        rows_html += "<tr>"
        rows_html += f"<td class='left'>{row['ds_label']}</td>"
        for m in ALL_MODELS:
            v = row.get(f"{m}_train_R2")
            colour = "pos" if (not pd.isna(v) and v >= 0) else "neg"
            val_str = f"{v:+.3f}" if not pd.isna(v) else "—"
            rows_html += f"<td class='num {colour}'>{val_str}</td>"
        rows_html += "</tr>"
    return header + rows_html + "</tbody></table>"


# ── Auto-observation ───────────────────────────────────────────────────────────

def _auto_obs(sub: pd.DataFrame, exp: str) -> str:
    lines = []

    # Win counts
    wins = {m: 0 for m in ALL_MODELS}
    lin_wins  = {m: 0 for m in LINEAR_MODELS}
    nl_wins   = {m: 0 for m in NL_MODELS}
    for _, row in sub.iterrows():
        w = _winner(row)
        if w:
            wins[w] += 1
            if w in LINEAR_MODELS: lin_wins[w] += 1
            else: nl_wins[w] += 1

    best_overall = max(wins, key=wins.get)
    total_lin = sum(lin_wins.values())
    total_nl  = sum(nl_wins.values())
    n = len(sub)

    group_winner = "Tree Ensembles" if total_nl > total_lin else (
        "Linear Models" if total_lin > total_nl else "Tie")
    win_str = " | ".join(f"{m}: {wins[m]}W" for m in ALL_MODELS)
    lines.append(
        f"<strong>Best model group:</strong> <strong>{group_winner}</strong> "
        f"(Linear {total_lin}W vs Tree {total_nl}W across {n} datasets). "
        f"Individual: {win_str}."
    )

    # Avg Test R²
    avg = {m: sub[f"{m}_test_R2"].mean() for m in ALL_MODELS}
    avg_lin = np.mean([avg[m] for m in LINEAR_MODELS])
    avg_nl  = np.mean([avg[m] for m in NL_MODELS])
    r2_str  = " | ".join(f"{m}&nbsp;{avg[m]:+.3f}" for m in ALL_MODELS)
    lines.append(
        f"<strong>Average Test R²:</strong> {r2_str}. "
        f"Group avg — Linear: <strong>{avg_lin:+.3f}</strong>, "
        f"Tree: <strong>{avg_nl:+.3f}</strong>."
    )

    # Overfit check (R² gap)
    gap_notes = []
    for m in ALL_MODELS:
        g = sub[f"{m}_R2_gap"].mean()
        if g > 0.35:
            gap_notes.append(f"<span class='gap-bad'>{m} heavily overfit (avg gap {g:+.2f})</span>")
        elif g > 0.15:
            gap_notes.append(f"<span class='gap-warn'>{m} moderately overfit (avg gap {g:+.2f})</span>")
    if gap_notes:
        lines.append("<strong>Overfitting:</strong> " + "; ".join(gap_notes) + ".")
    else:
        lines.append("<strong>Generalisation:</strong> All models show small train–test gaps.")

    # pH note
    ph_rows = sub[sub["target"].str.contains("pH")]
    if len(ph_rows) > 0:
        best_ph = ph_rows[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).mean()
        if best_ph < 0.15:
            lines.append(
                f"<strong>pH targets:</strong> All 6 models struggle (best avg Test R² "
                f"{best_ph:+.3f}) — effluent pH is tightly buffered with very low variance."
            )

    # Best single result
    best_val = sub[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).max()
    best_idx = sub[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).idxmax()
    best_tgt = sub.loc[best_idx, "ds_label"]
    best_col = sub.loc[best_idx, [f"{m}_test_R2" for m in ALL_MODELS]].idxmax()
    best_m   = best_col.replace("_test_R2", "")
    lines.append(
        f"<strong>Best single result:</strong> {best_m} on <em>{best_tgt}</em> "
        f"— Test R² = {best_val:+.3f}."
    )

    return "<br>".join(lines)


# ── Cross-experiment table ─────────────────────────────────────────────────────

def _cross_exp_table(df: pd.DataFrame) -> str:
    rows = ""
    for exp in ["Experiment 1", "Experiment 2 Sub-1", "Experiment 2 Sub-2"]:
        grp = df[df["experiment"] == exp]
        if grp.empty:
            continue
        label  = EXPERIMENT_LABELS.get(exp, exp)
        n_feat = EXPERIMENT_NFEAT.get(exp, "?")
        rows += "<tr>"
        rows += f"<td class='left'>{label}</td>"
        rows += f"<td class='num'>{n_feat}</td>"
        for m in ALL_MODELS:
            rows += f"<td class='num'>{grp[f'{m}_test_R2'].mean():+.3f}</td>"
        rows += "</tr>"

    return f"""
    <table class="metric-table">
      <thead>
        <tr>
          <th class="left">Experiment</th>
          <th># feats</th>
          <th colspan="3" class="group-lin">Linear — avg Test R²</th>
          <th colspan="3" class="group-nl">Tree — avg Test R²</th>
        </tr>
        <tr>
          <th></th><th></th>
          <th class="th-ols">OLS</th>
          <th class="th-ridge">Ridge</th>
          <th class="th-elnet">ElNet</th>
          <th class="th-rf">RF</th>
          <th class="th-gb">GB</th>
          <th class="th-xgb">XGB</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


# ── Summary stat cards ─────────────────────────────────────────────────────────

def _top_summary_cards(df: pd.DataFrame) -> str:
    n_ds = len(df)

    # Win counts
    wins = {m: 0 for m in ALL_MODELS}
    for _, row in df.iterrows():
        w = _winner(row)
        if w:
            wins[w] += 1

    total_lin = sum(wins[m] for m in LINEAR_MODELS)
    total_nl  = sum(wins[m] for m in NL_MODELS)
    group_winner = "Tree Ensembles" if total_nl > total_lin else (
        "Linear Models" if total_lin > total_nl else "Tie")
    group_sub = f"Linear {total_lin}W vs Tree {total_nl}W ({n_ds} datasets)"

    best_overall = max(wins, key=wins.get)
    best_sub = " · ".join(f"{wins[m]}W {m}" for m in ALL_MODELS)

    # Best single
    best_val = df[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).max()
    best_idx = df[[f"{m}_test_R2" for m in ALL_MODELS]].max(axis=1).idxmax()
    best_tgt = df.loc[best_idx, "ds_label"]

    # Avg R² per model
    avg = {m: df[f"{m}_test_R2"].mean() for m in ALL_MODELS}

    def _card(label, value, sub, border="#3A7BD5"):
        return f"""
      <div class="stat-card" style="border-top-color:{border}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
      </div>"""

    cards = ""
    cards += _card("Datasets", n_ds, "3 experiments × 8 targets")
    cards += _card("Best model group", group_winner, group_sub, "#F0B849")
    cards += _card("Best individual model", best_overall,
                   best_sub, MODEL_COLORS[best_overall])
    cards += _card("Best single result", f"{best_val:+.3f}",
                   best_tgt, "#B07FD4")
    for m in ALL_MODELS:
        cards += _card(f"{m} avg Test R²", f"{avg[m]:+.3f}",
                       "linear" if m in LINEAR_MODELS else "tree ensemble",
                       MODEL_COLORS[m])

    return f'<div class="summary-grid">{cards}</div>'


# ── Per-experiment section ─────────────────────────────────────────────────────

def _exp_section(sub: pd.DataFrame, exp: str) -> str:
    label  = EXPERIMENT_LABELS.get(exp, exp)
    n_feat = EXPERIMENT_NFEAT.get(exp, "?")
    obs    = _auto_obs(sub, exp)

    r2_b64   = _bar_chart_b64(sub, "test_R2",   "Test R²",   exp)
    rmse_b64 = _bar_chart_b64(sub, "test_RMSE", "Test RMSE", exp)

    charts_html = ""
    for b64, caption in [(r2_b64, "Test R² — all 6 models"), (rmse_b64, "Test RMSE — all 6 models")]:
        if b64:
            charts_html += f"""
            <div class="plot-box wide">
              <img src="{b64}" alt="{caption}">
              <div class="plot-caption">{caption}</div>
            </div>"""

    exp_id = exp.replace(" ", "-")
    return f"""
    <div class="exp-card" id="{exp_id}">
      <h2>{label}</h2>
      <p style="margin:0 0 12px;font-size:0.85rem;color:var(--text-meta)">
        Features: <strong>{n_feat}</strong> &nbsp;|&nbsp;
        Datasets: <strong>{len(sub)}</strong> &nbsp;|&nbsp;
        Train: 2020–2024 &nbsp;|&nbsp; Test: 2025
      </p>

      <div class="obs">{obs}</div>

      <h3>Test R² — All Models</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 4px">
        <span style="background:rgba(91,173,111,0.3);padding:1px 6px;border-radius:3px">■</span>
        highlighted cell = best Test R² in row &nbsp;·&nbsp;
        <span class="pos">green</span> = R² ≥ 0 &nbsp;·&nbsp;
        <span class="neg">red</span> = R² &lt; 0
      </p>
      {_primary_table(sub)}

      <details>
        <summary>Test RMSE &amp; R² Gap — all models</summary>
        <div class="details-body">
          {_rmse_table(sub)}
        </div>
      </details>

      <details>
        <summary>Train R² (overfitting reference)</summary>
        <div class="details-body">
          {_train_table(sub)}
        </div>
      </details>

      <h3>Summary Charts</h3>
      <div class="plots-row">{charts_html}</div>
    </div>
    """


# ── CSS ────────────────────────────────────────────────────────────────────────

REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }

  body {
    font-family: 'Inter', Calibri, Arial, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px 60px;
  }

  h1 { font-size: 1.7rem; margin-bottom: 4px; color: var(--text); }
  h2 { font-size: 1.22rem; margin: 36px 0 10px; color: var(--text);
       border-bottom: 2px solid var(--border); padding-bottom: 6px; }
  h3 { font-size: 1.05rem; margin: 22px 0 8px; color: var(--text-muted); }
  p, li { color: var(--text-muted); font-size: 0.93rem; line-height: 1.6; }

  .run-badge {
    display: inline-block; background: #3A7BD5; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.78rem;
    margin-left: 10px; vertical-align: middle;
  }
  .ts { color: var(--text-meta); font-size: 0.82rem; margin-top: 4px; }

  /* ── Legend ── */
  .legend { display: flex; gap: 14px; flex-wrap: wrap;
            margin: 12px 0 20px; font-size: 0.88rem; color: var(--text-muted); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .dot-hatch {
    width: 12px; height: 12px; border-radius: 50%;
    background-image: repeating-linear-gradient(
      45deg, currentColor 0, currentColor 2px, transparent 0, transparent 50%
    );
    background-size: 6px 6px;
    display: inline-block;
  }
  .legend-divider {
    width: 1px; background: var(--border); margin: 0 6px; align-self: stretch;
  }

  /* ── Summary stat cards ── */
  .summary-grid {
    display: flex; flex-wrap: nowrap; gap: 10px;
    margin: 14px 0 24px; overflow-x: auto;
  }
  .stat-card {
    background: var(--card); border-radius: 8px; padding: 13px 14px;
    box-shadow: 0 1px 6px var(--card-shadow); border-top: 3px solid #3A7BD5;
    flex: 1 1 0; min-width: 120px;
  }
  .stat-card .label { font-size: 0.72rem; color: var(--text-meta); margin-bottom: 4px; line-height: 1.3; }
  .stat-card .value { font-size: 1.2rem; font-weight: 700; color: var(--text); white-space: nowrap; }
  .stat-card .sub   { font-size: 0.68rem; color: var(--text-muted); margin-top: 2px; line-height: 1.35; }

  /* ── Experiment card ── */
  .exp-card {
    background: var(--card); border-radius: 10px; padding: 24px 28px;
    margin-bottom: 36px; box-shadow: 0 2px 10px var(--card-shadow);
    border-left: 4px solid #3A7BD5;
  }
  .exp-card h2 { margin-top: 0; border: none; padding: 0; font-size: 1.15rem; }

  /* ── Metric table ── */
  .metric-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin: 14px 0 22px; }
  .metric-table th {
    background: #2B3A55; color: #C8D8F0; padding: 7px 8px; text-align: center;
    font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap;
  }
  .metric-table td { padding: 6px 8px; border-bottom: 1px solid var(--border-light); color: var(--text); }
  .metric-table tbody tr:nth-child(even) { background: var(--table-even); }
  .metric-table .num  { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-table .left { text-align: left; font-weight: 500; }
  .metric-table .sub  { font-size: 0.75rem; color: var(--text-meta); }

  /* Group header spans */
  .group-lin { border-left: 3px solid #555; border-right: 3px solid #555; }
  .group-nl  { border-left: 3px solid #555; border-right: 3px solid #555; }

  /* Model colour headers */
  .th-ols   { color: #E15252; }
  .th-ridge { color: #4A90D9; }
  .th-elnet { color: #5BAD6F; }
  .th-rf    { color: #4FC3F7; }
  .th-gb    { color: #81C784; }
  .th-xgb   { color: #FF8A65; }

  /* R² colours */
  .pos { color: #5BAD6F; }
  .neg { color: #E15252; }
  .na  { color: var(--text-meta); }

  /* Best-model highlight */
  .best { background: rgba(74,144,217,0.2) !important; font-weight: 700; }

  /* R² Gap */
  .gap-ok   { color: #5BAD6F; }
  .gap-warn { color: #F0B849; }
  .gap-bad  { color: #E15252; }

  /* MAPE */
  .mape-ok   { color: #5BAD6F; }
  .mape-warn { color: #F0B849; }
  .mape-bad  { color: #E15252; }

  /* Obs callout */
  .obs {
    background: var(--obs-bg); border-radius: 8px; padding: 14px 18px;
    margin: 14px 0; font-size: 0.88rem; color: var(--text); line-height: 1.65;
  }
  .obs strong { color: var(--text); }

  /* Plot containers */
  .plots-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
  .plot-box       { flex: 1 1 400px; background: var(--card); border-radius: 8px;
                    overflow: hidden; border: 1px solid var(--border-light); }
  .plot-box.wide  { flex: 1 1 700px; }
  .plot-box img   { width: 100%; display: block; }
  .plot-caption   { font-size: 0.78rem; color: var(--text-meta); padding: 6px 10px; }

  details { border: 1px solid var(--border); border-radius: 8px; margin: 14px 0; }
  details > summary {
    padding: 10px 16px; cursor: pointer; font-size: 0.93rem; font-weight: 600;
    color: var(--text); list-style: none; user-select: none;
  }
  details > summary::before { content: "▶ "; font-size: 0.7em; }
  details[open] > summary::before { content: "▼ "; }
  details > summary:hover { background: var(--summary-hover); border-radius: 8px; }
  .details-body { padding: 14px 18px; }

  @media (max-width: 768px) {
    body { padding: 16px; }
    .plots-row { flex-direction: column; }
  }
""")


# ── Full HTML assembly ─────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, run: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    legend = """
    <div class="legend">
      <strong style="color:var(--text);font-size:0.88rem">Linear:</strong>
      <div class="legend-item"><span class="dot" style="background:#E15252"></span> OLS</div>
      <div class="legend-item"><span class="dot" style="background:#4A90D9"></span> Ridge (L2)</div>
      <div class="legend-item"><span class="dot" style="background:#5BAD6F"></span> ElasticNet (L1+L2)</div>
      <div class="legend-divider"></div>
      <strong style="color:var(--text);font-size:0.88rem">Tree:</strong>
      <div class="legend-item"><span class="dot" style="background:#2171B5"></span> Random Forest</div>
      <div class="legend-item"><span class="dot" style="background:#238B45"></span> Gradient Boosting</div>
      <div class="legend-item"><span class="dot" style="background:#D94801"></span> XGBoost</div>
    </div>
    """

    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Phase 4 — Model Comparison:</strong> Head-to-head comparison of three linear
      models (OLS, Ridge, ElasticNet) against three tree ensembles (RF, GB, XGB) across
      Experiments 1 and 2. All models trained on 2020–2024 data (2020 rows included where
      available), tested on 2025. <strong>All six models are hyperparameter-tuned</strong>
      using identical CV protocol: <code>GridSearchCV</code> (RF, GB) and
      <code>RandomizedSearchCV</code> (XGB) with <code>TimeSeriesSplit(n_splits=3)</code>,
      scored on <code>neg_root_mean_squared_error</code>. Linear models additionally use
      <code>StandardScaler</code>.<br><br>
      <strong>Highlighted cell</strong> = best Test R² in that row (across all 6 models).
      <strong>R² Gap</strong> = Train R² − Test R² (lower = less overfitting).
      Bar charts use <em>hatched bars for linear</em> and <em>solid bars for tree models</em>.
    </div>
    """

    exp_sections = ""
    for exp in ["Experiment 1", "Experiment 2 Sub-1", "Experiment 2 Sub-2"]:
        sub = df[df["experiment"] == exp].copy()
        if sub.empty:
            continue
        exp_sections += _exp_section(sub, exp)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 4 — Linear vs Tree Model Comparison | Run {run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>

  <h1>Phase 4 — Linear vs Tree Ensemble Comparison
    <span class="run-badge">Run {run}</span>
  </h1>
  <div class="ts">Generated {ts} &nbsp;|&nbsp; Train: 2020–2024 &nbsp;|&nbsp; Test: 2025</div>

  {intro}
  {legend}

  <h2>Cross-Experiment Summary</h2>
  {_top_summary_cards(df)}
  {_cross_exp_table(df)}

  {exp_sections}

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    df, run = _load_data()
    print(f"  {len(df)} datasets loaded across {df['experiment'].nunique()} experiments.")
    print(f"  Run: {run}")

    print("Building HTML report...")
    html = build_html(df, run)

    out_path = os.path.join(SCRIPT_DIR, f"report_comparison_baseline_run_{run}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
