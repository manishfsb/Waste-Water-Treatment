"""
feature_selection.py — Phase 5: Feature importance ranking and selection.

For each experiment × target:
  - Loads the tuned RF best_estimator_ (from non_linear_modeling/rf/models/)
  - Computes RF permutation importance on TRAINING data (n_repeats=20, stable estimate)
  - Records RF impurity importance, MI score, Pearson r, Spearman ρ
  - Ranks features and generates per-experiment recommendations

Recommendation logic (per experiment, averaged across all targets in that experiment):
  - "Core"   : mean normalised perm-imp ≥ 0.08  → keep without question
  - "Useful" : mean normalised perm-imp 0.03–0.07 → context-dependent
  - "Weak"   : mean normalised perm-imp < 0.03  → candidate for removal

Outputs (all in modeling/feature_selection/):
  plots/  — heatmaps per experiment variant
  feature_importance.xlsx — full flat table of all metrics
  report_feature_selection_run_1.html

Usage (from project root):
  .venv/bin/python3 21-25/modeling/feature_selection/feature_selection.py
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROJECT_ROOT = os.path.dirname(MODELING_DIR)
RF_MODELS    = os.path.join(MODELING_DIR, "models", "non_linear", "baseline", "rf", "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")

sys.path.insert(0, PROJECT_ROOT)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Feature sets (must match non_linear_modeling.py exactly) ──────────────────
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)",
              "Sec Clarifier BOD (mg/L)", "Sec Clarifier COD (mg/L)",
              "Sec Clarifier RAS", "Sec Sed pH", "Sec Sed TSS (mg/L)",
              "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)", "Sec Sed RAS (New)"]
COMMON     = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]
TEMPORAL   = {"month", "day_of_week", "year"}

S1_GRAB   = GRAB_INLET + COMMON
S1_COMP   = COMP_INLET + COMMON
S2P1      = SEC_COLS   + COMMON
S2P2_GRAB = GRAB_INLET + SEC_COLS + COMMON
S2P2_COMP = COMP_INLET + SEC_COLS + COMMON

TRAIN_YEARS = [2021, 2022, 2023, 2024]

_STAGE_DIRS = {
    "experiment1":    os.path.join("datasets", "experiment1"),
    "experiment2_s1": os.path.join("datasets", "experiment2", "sub_exp1"),
    "experiment2_s2": os.path.join("datasets", "experiment2", "sub_exp2"),
}

def _sub(stage_dir, name):
    return os.path.join(MODELING_DIR, _STAGE_DIRS[stage_dir], f"{name}.xlsx")

# ── Registry ──────────────────────────────────────────────────────────────────
# (exp_label, variant_label, model_name, subset_path, features, target)
REGISTRY = [
    # Experiment 1 — grab
    ("Experiment 1", "Grab", "stage1_grab_BOD",
     _sub("experiment1","stage1_grab_BOD"), S1_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 1", "Grab", "stage1_grab_COD",
     _sub("experiment1","stage1_grab_COD"), S1_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Experiment 1", "Grab", "stage1_grab_TSS",
     _sub("experiment1","stage1_grab_TSS"), S1_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 1", "Grab", "stage1_grab_pH",
     _sub("experiment1","stage1_grab_pH"),  S1_GRAB, "Effluent pH (Grab)"),
    # Experiment 1 — composite
    ("Experiment 1", "Composite", "stage1_comp_BOD",
     _sub("experiment1","stage1_comp_BOD"), S1_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 1", "Composite", "stage1_comp_COD",
     _sub("experiment1","stage1_comp_COD"), S1_COMP, "Effluent COD (mg/L, Composite)"),
    ("Experiment 1", "Composite", "stage1_comp_TSS",
     _sub("experiment1","stage1_comp_TSS"), S1_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 1", "Composite", "stage1_comp_pH",
     _sub("experiment1","stage1_comp_pH"),  S1_COMP, "Effluent pH (Composite)"),
    # Experiment 2 Sub-1 — grab
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_BOD",
     _sub("experiment2_s1","stage2_p1_grab_BOD"), S2P1, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_COD",
     _sub("experiment2_s1","stage2_p1_grab_COD"), S2P1, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_TSS",
     _sub("experiment2_s1","stage2_p1_grab_TSS"), S2P1, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-1", "Grab", "stage2_p1_grab_pH",
     _sub("experiment2_s1","stage2_p1_grab_pH"),  S2P1, "Effluent pH (Grab)"),
    # Experiment 2 Sub-1 — composite
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_BOD",
     _sub("experiment2_s1","stage2_p1_comp_BOD"), S2P1, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_COD",
     _sub("experiment2_s1","stage2_p1_comp_COD"), S2P1, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_TSS",
     _sub("experiment2_s1","stage2_p1_comp_TSS"), S2P1, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-1", "Composite", "stage2_p1_comp_pH",
     _sub("experiment2_s1","stage2_p1_comp_pH"),  S2P1, "Effluent pH (Composite)"),
    # Experiment 2 Sub-2 — grab
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_BOD",
     _sub("experiment2_s2","stage2_p2_grab_BOD"), S2P2_GRAB, "Effluent BOD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_COD",
     _sub("experiment2_s2","stage2_p2_grab_COD"), S2P2_GRAB, "Effluent COD (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_TSS",
     _sub("experiment2_s2","stage2_p2_grab_TSS"), S2P2_GRAB, "Effluent TSS (mg/L, Grab)"),
    ("Experiment 2 Sub-2", "Grab", "stage2_p2_grab_pH",
     _sub("experiment2_s2","stage2_p2_grab_pH"),  S2P2_GRAB, "Effluent pH (Grab)"),
    # Experiment 2 Sub-2 — composite
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_BOD",
     _sub("experiment2_s2","stage2_p2_comp_BOD"), S2P2_COMP, "Effluent BOD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_COD",
     _sub("experiment2_s2","stage2_p2_comp_COD"), S2P2_COMP, "Effluent COD (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_TSS",
     _sub("experiment2_s2","stage2_p2_comp_TSS"), S2P2_COMP, "Effluent TSS (mg/L, Composite)"),
    ("Experiment 2 Sub-2", "Composite", "stage2_p2_comp_pH",
     _sub("experiment2_s2","stage2_p2_comp_pH"),  S2P2_COMP, "Effluent pH (Composite)"),
]

# ── Short target labels ────────────────────────────────────────────────────────
def _tgt_short(tgt):
    return (tgt.replace("Effluent ","").replace(" (mg/L, Grab)","_G")
               .replace(" (mg/L, Composite)","_C")
               .replace(" (Grab)","_G").replace(" (Composite)","_C"))

# ── Short feature labels ───────────────────────────────────────────────────────
def _feat_short(f):
    return (f.replace("Inlet ","In ").replace("Effluent ","Eff ")
             .replace("(mg/L, Grab)","(G)").replace("(mg/L, Composite)","(C)")
             .replace("(mg/L)","").replace("Sec Clarifier ","SecCl ")
             .replace("Sec Sed ","SecSd ").replace("Power Total (KW)","Power")
             .replace("Flow (MLD)","Flow").strip())

# ── Compute all metrics for one dataset ───────────────────────────────────────
def compute_metrics(exp, variant, name, subset_path, features, target):
    pkl = os.path.join(RF_MODELS, f"{name}_RF_run_1.pkl")
    if not os.path.exists(pkl):
        print(f"  SKIP (no model): {name}"); return []

    model = joblib.load(pkl)
    df    = pd.read_excel(subset_path, parse_dates=["Date"])

    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    extra = df[df["year"] == 2020]
    if len(extra) > 0:
        train = pd.concat([extra, train]).drop_duplicates()

    X_tr = train[features].values
    y_tr = train[target].values

    # ── Permutation importance (training set, n_repeats=20) ───────────────────
    perm = permutation_importance(model, X_tr, y_tr,
                                  n_repeats=20, random_state=42, n_jobs=-1)
    perm_mean = perm.importances_mean.clip(min=0)   # clip negatives to 0
    perm_std  = perm.importances_std

    # Normalise so importances sum to 1 per target
    perm_total   = perm_mean.sum()
    perm_norm    = perm_mean / perm_total if perm_total > 0 else perm_mean

    # ── Impurity importance ───────────────────────────────────────────────────
    imp_imp = model.feature_importances_

    # ── MI score ──────────────────────────────────────────────────────────────
    mi = mutual_info_regression(X_tr, y_tr, random_state=42)

    # ── Pearson & Spearman r ──────────────────────────────────────────────────
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

# ── Image helper ──────────────────────────────────────────────────────────────
def _b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ── Heatmap: perm_imp_norm, features × targets ────────────────────────────────
def make_heatmap(sub: pd.DataFrame, title: str, fname: str) -> str:
    pivot = sub.pivot_table(index="feat_short", columns="target_short",
                            values="perm_imp_norm", aggfunc="mean")
    # sort features by mean importance descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4, len(pivot) * 0.45)
    fig_w = max(6, len(pivot.columns) * 1.2)
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

    # Annotate cells
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

# ── Per-target ranked bar chart ───────────────────────────────────────────────
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
    # Legend for temporal
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

# ── Recommendation logic ──────────────────────────────────────────────────────
CORE_THRESH   = 0.08
USEFUL_THRESH = 0.03

def _tier_single(v: float) -> tuple[str, str]:
    """Return (label, css_class) for a single normalised perm-imp value."""
    if v >= CORE_THRESH:   return "Core ✓",  "tier-core"
    if v >= USEFUL_THRESH: return "Useful",   "tier-useful"
    return "Weak ✗",  "tier-weak"

# ── HTML helpers ──────────────────────────────────────────────────────────────



def _target_details(sub_target: pd.DataFrame, target: str, bar_path: str) -> str:
    sub_s = sub_target.sort_values("perm_imp_norm", ascending=False)

    # Build per-target recommendation summary line
    core_f   = [r["feat_short"] for _, r in sub_s.iterrows() if r["perm_imp_norm"] >= CORE_THRESH]
    useful_f = [r["feat_short"] for _, r in sub_s.iterrows() if USEFUL_THRESH <= r["perm_imp_norm"] < CORE_THRESH]
    weak_f   = [r["feat_short"] for _, r in sub_s.iterrows() if r["perm_imp_norm"] < USEFUL_THRESH]
    rec_line = (
        f"<span class='tier-core'>Core ({len(core_f)}): {', '.join(core_f) or '—'}</span> &nbsp;|&nbsp; "
        f"<span class='tier-useful'>Useful ({len(useful_f)})</span> &nbsp;|&nbsp; "
        f"<span class='tier-weak'>Weak ({len(weak_f)}): {', '.join(weak_f) or '—'}</span>"
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

    img = f'<img src="{_b64(bar_path)}" alt="bar">' if bar_path and os.path.exists(bar_path) else ""
    return f"""
    <details>
      <summary>{target}</summary>
      <div class="details-body">
        <div class="obs" style="padding:8px 14px;margin-bottom:10px;font-size:0.85rem">
          {rec_line}
        </div>
        <div class="plot-box" style="max-width:680px;margin-bottom:12px">{img}</div>
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
    <strong>Feature set:</strong> {n_feat} features &nbsp;|&nbsp;
    Thresholds: <span class='tier-core'>Core ✓</span> norm perm imp ≥ {CORE_THRESH} &nbsp;|&nbsp;
    <span class='tier-useful'>Useful</span> ≥ {USEFUL_THRESH} &nbsp;|&nbsp;
    <span class='tier-weak'>Weak ✗</span> &lt; {USEFUL_THRESH}.<br>
    Recommendations are shown per target below. Temporal features 🕐 (month, day_of_week, year)
    reflect seasonality — not a process-controllable variable.
    """

    hm_img = f'<img src="{_b64(heatmap_path)}" alt="heatmap">' if heatmap_path and os.path.exists(heatmap_path) else ""

    # Per-target collapsible sections
    targets_html = ""
    for tgt in sub["target"].unique():
        sub_t = sub[sub["target"] == tgt].copy()
        ts    = _tgt_short(tgt)
        bp    = bar_paths.get(ts, "")
        targets_html += _target_details(sub_t, tgt, bp)

    sec_id = f"{exp.replace(' ','-')}-{variant}"
    return f"""
    <div class="exp-card" id="{sec_id}">
      <h2>{exp} — {variant} Effluent</h2>
      <div class="obs">{obs}</div>

      <h3>Feature × Target Importance Heatmap</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 8px">
        Normalised permutation importance (sums to 1.0 per target column).
        Computed on training data (2020–2024). Higher = more impact on model predictions.
      </p>
      <div class="plot-box wide" style="max-width:900px">{hm_img}</div>

      <h3>Per-Target Recommendations &amp; Detail</h3>
      <p style="font-size:0.8rem;color:var(--text-meta);margin:0 0 10px">
        Each section shows the Core / Useful / Weak classification for that specific target,
        alongside the full ranked metric table. Click to expand.
      </p>
      {targets_html}
    </div>"""

# ── CSS ────────────────────────────────────────────────────────────────────────
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

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Phase 5 — Feature Selection")
    print("=" * 60)

    all_rows = []

    # Step 1: compute metrics for all 24 datasets
    for exp, variant, name, subset_path, features, target in REGISTRY:
        print(f"  [{exp} / {variant}]  {name}")
        rows = compute_metrics(exp, variant, name, subset_path, features, target)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Save flat table
    xlsx_path = os.path.join(SCRIPT_DIR, "feature_importance.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"\nSaved: {xlsx_path}")

    # Step 2: group into report sections
    # Group key: (experiment, variant)  — but Exp2S1 grab+comp share same features → one section
    groups = [
        ("Experiment 1",       "Grab"),
        ("Experiment 1",       "Composite"),
        ("Experiment 2 Sub-1", "Grab"),
        ("Experiment 2 Sub-1", "Composite"),
        ("Experiment 2 Sub-2", "Grab"),
        ("Experiment 2 Sub-2", "Composite"),
    ]

    print("\nGenerating plots...")
    sections_html = ""

    for exp, variant in groups:
        sub = df[(df["experiment"] == exp) & (df["variant"] == variant)].copy()
        if sub.empty:
            continue

        tag = f"{exp.replace(' ','')}_{variant}"

        # Heatmap
        hm_path = make_heatmap(sub,
            f"Normalised Permutation Importance — {exp} ({variant})",
            f"{tag}_heatmap.png")

        # Per-target bar charts
        bar_paths = {}
        for tgt in sub["target"].unique():
            ts    = _tgt_short(tgt)
            bpath = make_bar_chart(
                sub[sub["target"] == tgt].copy(),
                f"Feature Importance — {_tgt_short(tgt)}",
                f"{tag}_{ts}_bar.png")
            bar_paths[ts] = bpath

        sections_html += _variant_section(exp, variant, sub, hm_path, bar_paths)
        print(f"  ✓ {exp} / {variant}")

    # Step 3: build HTML
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    intro = """
    <div style="margin:10px 0 20px;padding:12px 16px;background:var(--obs-bg);
                border-radius:8px;font-size:0.88rem;color:var(--text)">
      <strong>Phase 5 — Feature Selection:</strong> Feature importance computed using
      three complementary metrics: (1) <strong>RF Permutation Importance</strong> — how much
      model accuracy drops when a feature is randomly shuffled (computed on training data,
      n_repeats=20, more reliable than impurity importance); (2) <strong>Mutual Information</strong>
      — detects both linear and non-linear feature–target associations; (3)
      <strong>Pearson / Spearman r</strong> — linear and monotonic correlation.
      Models used are the tuned RF best_estimators from Phase 3.<br><br>
      <strong>Interpretation:</strong> Permutation importance is the primary ranking signal.
      A feature scoring near zero has little causal or predictive value for the trained model.
      Temporal features (month, day_of_week, year) capture seasonal patterns but are not
      process-controllable — they are flagged separately (🕐) for interpretation.
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 5 — Feature Selection | Run 1</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>{REPORT_CSS}</style>
  {DARK_MODE_JS}
</head>
<body>
  <h1>Phase 5 — Feature Importance &amp; Selection
    <span class="run-badge">Run 1</span>
  </h1>
  <div class="ts">Generated {ts_str} &nbsp;|&nbsp; Based on tuned RF models (Phase 3)</div>
  {intro}
  {sections_html}
</body>
</html>"""

    report_path = os.path.join(SCRIPT_DIR, "report_feature_selection_run_1.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
