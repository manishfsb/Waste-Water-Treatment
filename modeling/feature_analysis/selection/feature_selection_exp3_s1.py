"""
feature_selection_exp3_s1.py - Feature importance ranking for Experiment 3 Sub-1.

For each of the 8 Exp3-S1 targets:
  - Loads the tuned RF best_estimator_ from models/non_linear/exp3_s1/rf/models/
  - Computes RF permutation importance on TRAINING data (n_repeats=20)
  - Records impurity importance, MI score, Pearson r, Spearman ρ
  - Applies Core / Useful / Weak tiering (same thresholds as Phase 5)

Outputs (all in feature_analysis/selection/):
  plots_exp3_s1/   - heatmap + per-target bar charts
  feature_importance_exp3_s1.xlsx
  report_feature_selection_exp3_s1_run_N.html

Usage (from project root):
  .venv/bin/python3 21-25/modeling/feature_analysis/selection/feature_selection_exp3_s1.py
"""

import os
import sys
import base64
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROJECT_ROOT  = os.path.dirname(MODELING_DIR)
RF_MODELS_DIR = os.path.join(MODELING_DIR, "models", "non_linear", "exp3_s1", "rf", "models")
PLOTS_DIR     = os.path.join(SCRIPT_DIR, "plots_exp3_s1")

sys.path.insert(0, PROJECT_ROOT)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEMPORAL    = {"month", "day_of_week", "year"}

CORE_THRESH   = 0.08
USEFUL_THRESH = 0.03

# ── Dataset helper ─────────────────────────────────────────────────────────────
def _e3s1(name):
    return os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1", f"{name}.xlsx")

# ── Registry ──────────────────────────────────────────────────────────────────
# (exp_label, variant_label, model_name, subset_path, target)
# Features are inferred dynamically from each subset file's columns.
REGISTRY = [
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_BOD",
     _e3s1("s1_stage3_grab_BOD"), "Effluent BOD (mg/L, Grab)"),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_COD",
     _e3s1("s1_stage3_grab_COD"), "Effluent COD (mg/L, Grab)"),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_TSS",
     _e3s1("s1_stage3_grab_TSS"), "Effluent TSS (mg/L, Grab)"),
    ("Experiment 3 Sub-1", "Grab",      "s1_stage3_grab_pH",
     _e3s1("s1_stage3_grab_pH"),  "Effluent pH (Grab)"),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_BOD",
     _e3s1("s1_stage3_comp_BOD"), "Effluent BOD (mg/L, Composite)"),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_COD",
     _e3s1("s1_stage3_comp_COD"), "Effluent COD (mg/L, Composite)"),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_TSS",
     _e3s1("s1_stage3_comp_TSS"), "Effluent TSS (mg/L, Composite)"),
    ("Experiment 3 Sub-1", "Composite", "s1_stage3_comp_pH",
     _e3s1("s1_stage3_comp_pH"),  "Effluent pH (Composite)"),
]

# ── Feature inference ──────────────────────────────────────────────────────────
_EXCLUDE_COLS     = {"Date", "year", "month", "day_of_week"}
_EXCLUDE_PREFIXES = ("predicted_",)

def infer_features(df: pd.DataFrame, target: str) -> list:
    return [
        c for c in df.columns
        if c != target
        and c not in _EXCLUDE_COLS
        and not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
    ]

# ── Label helpers ──────────────────────────────────────────────────────────────
def _tgt_short(tgt):
    return (tgt.replace("Effluent ","").replace(" (mg/L, Grab)","_G")
               .replace(" (mg/L, Composite)","_C")
               .replace(" (Grab)","_G").replace(" (Composite)","_C"))

def _feat_short(f):
    return (f.replace("Inlet ","In ").replace("Effluent ","Eff ")
             .replace("(mg/L, Grab)","(G)").replace("(mg/L, Composite)","(C)")
             .replace("(mg/L)","").replace("Sec Clarifier ","SecCl ")
             .replace("Sec Sed ","SecSd ").replace("Power Total (KW)","Power")
             .replace("Flow (MLD)","Flow").strip())

# ── Detect RF run number ───────────────────────────────────────────────────────
def _rf_run_number() -> int:
    rfile = os.path.join(os.path.dirname(RF_MODELS_DIR), "results.xlsx")
    if not os.path.exists(rfile):
        return 1
    df = pd.read_excel(rfile)
    return int(df["run"].max()) if "run" in df.columns else 1

# ── Compute metrics for one dataset ───────────────────────────────────────────
def compute_metrics(exp, variant, name, subset_path, target, run):
    pkl = os.path.join(RF_MODELS_DIR, f"{name}_RF_run_{run}.pkl")
    if not os.path.exists(pkl):
        print(f"  SKIP (no model): {pkl}")
        return []

    model = joblib.load(pkl)
    df    = pd.read_excel(subset_path, parse_dates=["Date"])
    features = infer_features(df, target)

    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    X_tr = train[features].values
    y_tr = train[target].values

    # Permutation importance (n_repeats=20, clipped to 0, normalised)
    perm      = permutation_importance(model, X_tr, y_tr,
                                       n_repeats=20, random_state=42, n_jobs=-1)
    perm_mean = perm.importances_mean.clip(min=0)
    perm_std  = perm.importances_std
    total     = perm_mean.sum()
    perm_norm = perm_mean / total if total > 0 else perm_mean

    # Impurity importance
    imp_imp = model.feature_importances_

    # Mutual Information
    mi = mutual_info_regression(X_tr, y_tr, random_state=42)

    # Pearson & Spearman
    pearson  = [scipy_stats.pearsonr(X_tr[:, i], y_tr)[0]  for i in range(len(features))]
    spearman = [scipy_stats.spearmanr(X_tr[:, i], y_tr)[0] for i in range(len(features))]

    rows = []
    for i, feat in enumerate(features):
        rows.append({
            "experiment":    exp,
            "variant":       variant,
            "model_name":    name,
            "target":        target,
            "target_short":  _tgt_short(target),
            "feature":       feat,
            "feat_short":    _feat_short(feat),
            "is_temporal":   feat in TEMPORAL,
            "perm_imp_mean": round(float(perm_mean[i]), 6),
            "perm_imp_std":  round(float(perm_std[i]),  6),
            "perm_imp_norm": round(float(perm_norm[i]), 6),
            "impurity_imp":  round(float(imp_imp[i]),   6),
            "mi_score":      round(float(mi[i]),        6),
            "pearson_r":     round(float(pearson[i]),   4),
            "spearman_r":    round(float(spearman[i]),  4),
        })
    return rows

# ── Tiering ───────────────────────────────────────────────────────────────────
def _tier_single(v: float) -> tuple:
    if v >= CORE_THRESH:   return "Core ✓",  "tier-core"
    if v >= USEFUL_THRESH: return "Useful",   "tier-useful"
    return "Weak ✗",  "tier-weak"

# ── Image helper ──────────────────────────────────────────────────────────────
def _b64(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ── Heatmap: perm_imp_norm ────────────────────────────────────────────────────
def make_heatmap(sub: pd.DataFrame, title: str, fname: str) -> str:
    pivot = sub.pivot_table(index="feat_short", columns="target_short",
                            values="perm_imp_norm", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4, len(pivot) * 0.45)
    fig_w = max(6, len(pivot.columns) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "imp", ["#1C2333", "#1A4A6A", "#2171B5", "#6BAED6", "#FFF5EB"], N=256)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8, color="white")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8, color="white")

    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            v = pivot.values[r, c]
            if not np.isnan(v) and v > 0.01:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v < 0.4 else "#1C2333")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(labelcolor="white", labelsize=7)
    cbar.set_label("Norm. Perm. Importance", color="white", fontsize=8)

    ax.set_title(title, fontsize=10, color="white", pad=10)
    ax.spines[:].set_color("#3A4560")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path

# ── Per-target bar chart ──────────────────────────────────────────────────────
def make_bar_chart(sub_target: pd.DataFrame, title: str, fname: str) -> str:
    sub = sub_target.sort_values("perm_imp_norm", ascending=True)
    colors = ["#FD8D3C" if row["is_temporal"] else "#2171B5"
              for _, row in sub.iterrows()]

    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.38)))
    fig.patch.set_facecolor("#1C2333")
    ax.set_facecolor("#1C2333")

    bars = ax.barh(sub["feat_short"], sub["perm_imp_norm"],
                   color=colors, edgecolor="#3A4560", linewidth=0.5)
    for bar, val in zip(bars, sub["perm_imp_norm"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5, color="white")

    ax.set_xlabel("Normalised Permutation Importance", fontsize=9, color="white")
    ax.set_title(title, fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#3A4560")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2171B5", label="Process feature"),
                        Patch(color="#FD8D3C", label="Temporal (month/dow/year)")],
              fontsize=7.5, facecolor="#2B3A55", labelcolor="white",
              edgecolor="#3A4560", loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _target_details(sub_target: pd.DataFrame, target: str, bar_path: str) -> str:
    sub_s = sub_target.sort_values("perm_imp_norm", ascending=False)

    core_f   = [r["feat_short"] for _, r in sub_s.iterrows() if r["perm_imp_norm"] >= CORE_THRESH]
    useful_f = [r["feat_short"] for _, r in sub_s.iterrows() if USEFUL_THRESH <= r["perm_imp_norm"] < CORE_THRESH]
    weak_f   = [r["feat_short"] for _, r in sub_s.iterrows() if r["perm_imp_norm"] < USEFUL_THRESH]
    rec_line = (
        f"<span class='tier-core'>Core ({len(core_f)}): {', '.join(core_f) or '-'}</span> &nbsp;|&nbsp; "
        f"<span class='tier-useful'>Useful ({len(useful_f)})</span> &nbsp;|&nbsp; "
        f"<span class='tier-weak'>Weak ({len(weak_f)}): {', '.join(weak_f) or '-'}</span>"
    )

    rows = ""
    for _, r in sub_s.iterrows():
        tier, tcls = _tier_single(r["perm_imp_norm"])
        temp = " 🕐" if r["is_temporal"] else ""
        rows += f"""<tr>
          <td class='left'>{r['feat_short']}{temp}</td>
          <td class='num'>{r['perm_imp_norm']:.4f}</td>
          <td class='num'>{r['perm_imp_mean']:.6f} ± {r['perm_imp_std']:.6f}</td>
          <td class='num'>{r['impurity_imp']:.4f}</td>
          <td class='num'>{r['mi_score']:.4f}</td>
          <td class='num {"pos" if r["pearson_r"] >= 0 else "neg"}'>{r['pearson_r']:+.3f}</td>
          <td class='num {"pos" if r["spearman_r"] >= 0 else "neg"}'>{r['spearman_r']:+.3f}</td>
          <td class='{tcls}'>{tier}</td>
        </tr>"""

    img_b64 = _b64(bar_path)
    img_html = f'<img src="{img_b64}" alt="bar">' if img_b64 else ""
    return f"""
    <details>
      <summary>{target}</summary>
      <div class="details-body">
        <div class="obs" style="padding:8px 14px;margin-bottom:10px;font-size:0.85rem">
          {rec_line}
        </div>
        <div class="plot-box" style="max-width:680px;margin-bottom:12px">{img_html}</div>
        <table class="metric-table">
          <thead>
            <tr>
              <th class='left'>Feature <span style='font-weight:400;font-size:0.76rem'>🕐 = temporal</span></th>
              <th>Norm Perm Imp</th>
              <th>Raw Perm Imp (mean ± std)</th>
              <th>Impurity Imp</th>
              <th>MI Score</th>
              <th>Pearson r</th>
              <th>Spearman ρ</th>
              <th>Recommendation</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""


def _variant_section(exp: str, variant: str, sub: pd.DataFrame,
                     heatmap_path: str, bar_paths: dict) -> str:
    n_feat = sub["feature"].nunique()
    obs = f"""
    <strong>Feature set:</strong> {n_feat} features (ADD-tier + Exp2 Sub-2 baseline) &nbsp;|&nbsp;
    Thresholds: <span class='tier-core'>Core ✓</span> norm perm imp ≥ {CORE_THRESH} &nbsp;|&nbsp;
    <span class='tier-useful'>Useful</span> ≥ {USEFUL_THRESH} &nbsp;|&nbsp;
    <span class='tier-weak'>Weak ✗</span> &lt; {USEFUL_THRESH}.<br>
    Models used are the tuned RF estimators from Exp3 Sub-1 (run {sub.get('run', pd.Series([1])).max() if 'run' in sub.columns else 1}).
    Temporal features 🕐 (month, day_of_week, year) are flagged separately.
    """

    hm_b64 = _b64(heatmap_path)
    hm_img = f'<img src="{hm_b64}" alt="heatmap">' if hm_b64 else ""

    targets_html = ""
    for tgt in sub["target"].unique():
        sub_t = sub[sub["target"] == tgt].copy()
        ts    = _tgt_short(tgt)
        bp    = bar_paths.get(ts, "")
        targets_html += _target_details(sub_t, tgt, bp)

    sec_id = f"{exp.replace(' ','-')}-{variant}"
    return f"""
    <div class="exp-card" id="{sec_id}">
      <h2>{exp} - {variant} Effluent</h2>
      <div class="obs">{obs}</div>

      <h3>Feature × Target Importance Heatmap</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 8px">
        Normalised permutation importance (sums to 1.0 per target column).
        Computed on training data (2021-2024).
      </p>
      <div class="plot-box wide" style="max-width:900px">{hm_img}</div>

      <h3>Per-Target Recommendations &amp; Detail</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 10px">
        Core + Useful features (norm perm imp ≥ {USEFUL_THRESH}) will be kept
        in the feature-selected Exp3-S1 subsets. Click to expand.
      </p>
      {targets_html}
    </div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
REPORT_CSS = dark_mode_css("""
  * { box-sizing: border-box; }
  body { font-family:'Inter',Calibri,Arial,sans-serif;
         max-width:1400px; margin:0 auto; padding:24px 32px 60px; }
  h1 { font-size:1.7rem; margin-bottom:4px; color:var(--text); }
  h2 { font-size:1.18rem; margin:36px 0 10px; color:var(--text);
       border-bottom:2px solid var(--border); padding-bottom:5px; }
  h3 { font-size:1.0rem; margin:20px 0 8px; color:var(--text-muted); }
  p, li { color:var(--text-muted); font-size:0.88rem; line-height:1.6; }
  .ts { color:var(--text-meta); font-size:0.82rem; margin-top:4px; }
  .run-badge { display:inline-block; background:#3A7BD5; color:white;
               padding:2px 10px; border-radius:12px; font-size:0.78rem;
               margin-left:10px; vertical-align:middle; }
  .exp-card { background:var(--card); border-radius:10px; padding:24px 28px;
              margin-bottom:36px; box-shadow:0 2px 10px var(--card-shadow);
              border-left:4px solid #3A7BD5; }
  .exp-card h2 { margin-top:0; border:none; padding:0; }
  .obs { background:var(--obs-bg); border-radius:8px; padding:12px 16px;
         margin:12px 0 18px; font-size:0.88rem; color:var(--text); line-height:1.65; }
  .metric-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin:12px 0 20px; }
  .metric-table th { background:#2B3A55; color:#C8D8F0; padding:7px 8px;
                     text-align:center; font-weight:600; border-bottom:2px solid var(--border);
                     white-space:nowrap; }
  .metric-table td { padding:6px 8px; border-bottom:1px solid var(--border-light); color:var(--text); }
  .metric-table tbody tr:nth-child(even) { background:var(--table-even); }
  .metric-table .num  { text-align:right; font-variant-numeric:tabular-nums; }
  .metric-table .left { text-align:left; font-weight:500; }
  .pos { color:#5BAD6F; }
  .neg { color:#E15252; }
  .tier-core  { color:#5BAD6F; font-weight:700; }
  .tier-useful{ color:#F0B849; font-weight:600; }
  .tier-weak  { color:#E15252; }
  .plot-box       { background:var(--card); border-radius:8px; overflow:hidden;
                    border:1px solid var(--border-light); margin-bottom:12px; }
  .plot-box.wide  { max-width:100%; }
  .plot-box img   { width:100%; display:block; }
  details { border:1px solid var(--border); border-radius:8px; margin:10px 0; }
  details > summary { padding:10px 16px; cursor:pointer; font-size:0.92rem;
                      font-weight:600; color:var(--text); list-style:none; user-select:none; }
  details > summary::before { content:"▶ "; font-size:0.7em; }
  details[open] > summary::before { content:"▼ "; }
  details > summary:hover { background:var(--summary-hover); border-radius:8px; }
  .details-body { padding:14px 18px; }
""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Feature Selection - Experiment 3 Sub-1")
    print("=" * 60)

    run = _rf_run_number()
    print(f"Using RF run {run} models from: {RF_MODELS_DIR}")
    print()

    all_rows = []

    for exp, variant, name, subset_path, target in REGISTRY:
        print(f"  [{variant}]  {name}")
        if not os.path.exists(subset_path):
            print(f"    SKIP - file not found: {subset_path}")
            continue
        rows = compute_metrics(exp, variant, name, subset_path, target, run)
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no metrics computed. Check that RF models and subset files exist.")
        return

    df = pd.DataFrame(all_rows)

    xlsx_path = os.path.join(SCRIPT_DIR, "feature_importance_exp3_s1.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"\nSaved: {xlsx_path}")

    # Generate plots and HTML sections
    groups = [
        ("Experiment 3 Sub-1", "Grab"),
        ("Experiment 3 Sub-1", "Composite"),
    ]

    print("\nGenerating plots...")
    sections_html = ""

    for exp, variant in groups:
        sub = df[(df["experiment"] == exp) & (df["variant"] == variant)].copy()
        if sub.empty:
            continue

        tag = f"Exp3S1_{variant}"

        hm_path = make_heatmap(sub,
            f"Normalised Permutation Importance - Exp3 Sub-1 ({variant})",
            f"{tag}_heatmap.png")

        bar_paths = {}
        for tgt in sub["target"].unique():
            ts    = _tgt_short(tgt)
            bpath = make_bar_chart(
                sub[sub["target"] == tgt].copy(),
                f"Feature Importance - {ts}",
                f"{tag}_{ts}_bar.png")
            bar_paths[ts] = bpath

        sections_html += _variant_section(exp, variant, sub, hm_path, bar_paths)
        print(f"  ✓ {exp} / {variant}")

    # Determine report run number
    existing = [f for f in os.listdir(SCRIPT_DIR)
                if f.startswith("report_feature_selection_exp3_s1_run_") and f.endswith(".html")]
    report_run = len(existing) + 1

    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    intro = f"""
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Experiment 3 Sub-1 - Feature Selection:</strong> Feature importance computed
      on the Exp3 Sub-1 subsets (Exp2 Sub-2 baseline + ADD-tier candidate features).
      RF permutation importance (n_repeats=20) is the primary ranking signal, supported by
      mutual information and Pearson/Spearman correlations.<br><br>
      <strong>Goal:</strong> Identify which ADD-tier features genuinely contribute signal vs.
      which ones add noise and restrict rows. Core + Useful features (norm perm imp ≥ {USEFUL_THRESH})
      will be retained in the feature-selected Exp3-S1 subsets.
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Feature Selection - Experiment 3 Sub-1 | Run {report_run}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
  <h1>Feature Selection - Experiment 3 Sub-1
    <span class="run-badge">Run {report_run}</span>
  </h1>
  <div class="ts">Generated {ts_str} &nbsp;|&nbsp; Based on tuned RF models (Exp3-S1 run {run})</div>
  {intro}
  {sections_html}
</body>
</html>"""

    report_path = os.path.join(SCRIPT_DIR, f"report_feature_selection_exp3_s1_run_{report_run}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
