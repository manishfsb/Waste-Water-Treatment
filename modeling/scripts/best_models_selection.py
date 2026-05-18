"""
best_models_selection.py - Reinterpret leaderboard under overfitting-aware rules.

Re-ranks every (target, experiment) row in the unified results using three
complementary rules instead of raw max(Test R²):

  1. Naive:            argmax(R2_test)                                  - current rule
  2. Gap-adjusted:     argmax(R2_test − λ · max(0, R2_gap − τ))         - penalise overfit
                         where τ=0.10 (tolerated gap), λ=0.5
  3. One-SE rule:      among models within δ=0.05 R² of the top,
                         pick the smallest |R2_gap|
  4. Pareto frontier:  models not dominated in (R2_test↑, |R2_gap|↓)

Outputs:
  reports/best_models_selection.xlsx   - per-target comparison table
  reports/best_models_selection.html   - dark-mode HTML panel

Run:  .venv/bin/python3 21-25/modeling/scripts/best_models_selection.py
"""

import html as _html_lib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODELING_DIR)
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Reuse the unified loader - single source of truth for schema harmonisation.
sys.path.insert(0, SCRIPT_DIR)
from generate_unified_report import (           # noqa: E402
    load_all_data, TARGETS_ORDERED, TARGET_SHORT, EXP_CHART_ORDER,
)
sys.path.insert(0, PROJECT_ROOT)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Selection-rule parameters ──────────────────────────────────────────────────
GAP_TOLERANCE  = 0.10   # τ - gaps below this are not penalised
GAP_PENALTY    = 0.50   # λ - weight on excess gap
ONE_SE_MARGIN  = 0.05   # δ - R² margin counted as "equivalent"
                        #     0.03 was too tight: SE(R²) ≈ 0.05-0.07 at n_test≈200,
                        #     so a 0.03 gap is well within measurement noise.
SCORE_NOISE_BAND = 0.005  # ε - retained as metadata for historical notes and
                          # display context. Winner selection is score-first:
                          # smaller |gap| breaks only true score ties.


# ═══════════════════════════════════════════════════════════════════════════════
# Selection rules
# ═══════════════════════════════════════════════════════════════════════════════

def _gap_adjusted_score(r2, gap):
    if pd.isna(r2) or pd.isna(gap):
        return np.nan
    return r2 - GAP_PENALTY * max(0.0, abs(gap) - GAP_TOLERANCE)


def _pareto_front(sub: pd.DataFrame) -> list:
    """Return model names on the (R2_test ↑, |R2_gap| ↓) Pareto frontier."""
    pts = []
    for _, r in sub.iterrows():
        if pd.isna(r["R2_test"]) or pd.isna(r["R2_gap"]):
            continue
        pts.append((r["model"], r["R2_test"], abs(r["R2_gap"])))
    front = []
    for name, r2, g in pts:
        dominated = any(
            (r2b >= r2 and gb <= g) and (r2b > r2 or gb < g)
            for nb, r2b, gb in pts if nb != name
        )
        if not dominated:
            front.append(name)
    return front


def _pick_rules(sub: pd.DataFrame) -> dict:
    """Apply every rule to one (target, experiment) slice. Returns a row-dict.

    Tie-breaking: when the primary criterion (gap-adjusted score for gadj_row,
    |gap| for onese_row) is equal across candidates, prefer the row with the
    smaller |gap|; if that also ties, prefer the higher R²_test. This makes
    the ranking deterministic and aligns with the regularisation intent of
    the gap-adjusted formula (within the 0.10 dead band, the score collapses
    to raw R² and the gap signal is otherwise discarded - the tiebreaker
    restores it).
    """
    sub = sub.dropna(subset=["R2_test"]).copy()
    if sub.empty:
        return {}
    sub["gap_adj"] = sub.apply(
        lambda r: _gap_adjusted_score(r["R2_test"], r["R2_gap"]), axis=1
    )
    sub["abs_gap"] = sub["R2_gap"].abs()

    # 1. Naive
    naive_row = sub.loc[sub["R2_test"].idxmax()]
    # 2. Gap-adjusted (score first; tiebreak: smaller |gap|, then higher R²).
    # The score already encodes the overfit penalty, so a lower-score row should
    # not outrank a higher-score row solely because the score gap is small.
    if sub["gap_adj"].notna().any():
        gadj_cands = sub.sort_values(
            ["gap_adj", "abs_gap", "R2_test"], ascending=[False, True, False]
        )
        gadj_row = gadj_cands.iloc[0]
    else:
        gadj_row = naive_row
    # 3. One-SE (smallest |gap| within ONE_SE_MARGIN of top R²; tiebreak: higher R²)
    top_r2 = sub["R2_test"].max()
    within = sub[sub["R2_test"] >= top_r2 - ONE_SE_MARGIN].copy()
    onese_cands = within.sort_values(["abs_gap", "R2_test"], ascending=[True, False])
    onese_row = onese_cands.iloc[0]
    # 4. Pareto
    front = _pareto_front(sub)

    return dict(
        naive_model   = naive_row["model"],
        naive_R2      = naive_row["R2_test"],
        naive_gap     = naive_row["R2_gap"],
        naive_score   = naive_row["gap_adj"],
        naive_RMSE    = naive_row.get("RMSE_test",  float("nan")),
        naive_MAE     = naive_row.get("MAE_test",   float("nan")),
        naive_MdAE    = naive_row.get("MdAE_test",  float("nan")),
        gadj_model    = gadj_row["model"],
        gadj_R2       = gadj_row["R2_test"],
        gadj_gap      = gadj_row["R2_gap"],
        gadj_score    = gadj_row["gap_adj"],
        gadj_RMSE     = gadj_row.get("RMSE_test",   float("nan")),
        gadj_MAE      = gadj_row.get("MAE_test",    float("nan")),
        gadj_MdAE     = gadj_row.get("MdAE_test",   float("nan")),
        onese_model   = onese_row["model"],
        onese_R2      = onese_row["R2_test"],
        onese_gap     = onese_row["R2_gap"],
        onese_score   = onese_row["gap_adj"],
        onese_RMSE    = onese_row.get("RMSE_test",  float("nan")),
        onese_MAE     = onese_row.get("MAE_test",   float("nan")),
        onese_MdAE    = onese_row.get("MdAE_test",  float("nan")),
        pareto        = ", ".join(front),
        n_pareto      = len(front),
        changed       = (naive_row["model"] != gadj_row["model"])
                         or (naive_row["model"] != onese_row["model"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Build leaderboards
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_experiment(df_all: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, target).

    Only canonical exp_keys (those in EXP_CHART_ORDER) are included.
    Rows are ordered by EXP_CHART_ORDER position so the table follows the
    same canonical experiment sequence used throughout the report.
    """
    allowed   = set(EXP_CHART_ORDER)
    exp_rank  = {k: i for i, k in enumerate(EXP_CHART_ORDER)}
    df_all    = df_all[df_all["exp_key"].isin(allowed)]

    recs = []
    for exp, sub_exp in df_all.groupby("exp_key", sort=False):
        for tgt in TARGETS_ORDERED:
            sub = sub_exp[sub_exp["target"] == tgt]
            if sub.empty:
                continue
            picks = _pick_rules(sub)
            if not picks:
                continue
            recs.append({"exp_key": exp, "target": tgt,
                         "target_short": TARGET_SHORT.get(tgt, tgt),
                         "exp_order": exp_rank.get(exp, 999), **picks})
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values(["target", "exp_order"]).reset_index(drop=True)
    return df


def build_global(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Global winner per target across all canonical experiments/phases.
    Only exp_keys in EXP_CHART_ORDER are considered.
    """
    allowed = set(EXP_CHART_ORDER)
    df_all  = df_all[df_all["exp_key"].isin(allowed)]

    recs = []
    for tgt in TARGETS_ORDERED:
        sub = df_all[df_all["target"] == tgt].copy()
        if sub.empty:
            continue
        # Label entries with a compound (exp|model) identity so rule outputs
        # carry both fields.
        sub = sub.assign(model=sub["exp_key"].astype(str) + " · " + sub["model"].astype(str))
        picks = _pick_rules(sub)
        recs.append({"target": tgt, "target_short": TARGET_SHORT.get(tgt, tgt), **picks})
    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML render helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cell(val, d=3):
    if pd.isna(val):
        return "-"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return f"{val:.{d}f}"

def _r2_color(v):
    if pd.isna(v): return "#888888"
    if v >= 0.6:  return "#2ecc71"
    if v >= 0.4:  return "#52c98a"
    if v >= 0.2:  return "#f1c40f"
    if v >= 0.0:  return "#e67e22"
    return "#e74c3c"

def _gap_cls(v):
    if pd.isna(v): return ""
    a = abs(v)
    return "gap-good" if a < 0.10 else ("gap-warn" if a < 0.25 else "gap-bad")


def _safe_float(v):
    try:
        f = float(v)
        return "nan" if np.isnan(f) else str(f)
    except (TypeError, ValueError):
        return "nan"


# ── Row-state classification ──────────────────────────────────────────────────

def _row_state(r):
    """Classify a result row and return (state_str, info_html).

    state_str:
      'agree_ok'  - gap-adj == naive AND |gap| < 0.25  (green or yellow gap)
      'agree_bad' - gap-adj == naive AND |gap| >= 0.25 (red gap, no better alt)
      'disagree'  - gap-adj picks a different model
    """
    gadj   = str(r.get("gadj_model", ""))
    naive  = str(r.get("naive_model", ""))
    gap    = r.get("naive_gap", float("nan"))
    gap_abs = abs(gap) if not pd.isna(gap) else 0.0

    if gadj == naive:
        if gap_abs >= 0.25:
            state = "agree_bad"
            info_html = (
                "<strong>All rules agree - but the overfitting gap is large.</strong><br><br>"
                "Naive, gap-adjusted, and one-SE rules all select the same model, "
                "because no alternative achieves comparable accuracy with a smaller gap. "
                f"However, the train/test gap is "
                f"<span style='color:#e74c3c;font-weight:600'>{gap:+.3f}</span> "
                f"(|gap| ≥ 0.25 - red zone). "
                "The model is likely memorising training patterns. "
                "Monitor performance on fresh data, especially during high-flow or "
                "winter regimes where distribution shift is most pronounced."
            )
        else:
            state = "agree_ok"
            gap_str   = f"{gap:+.3f}" if not pd.isna(gap) else "-"
            gap_color = "#2ecc71" if gap_abs < 0.10 else "#f1c40f"
            thresh    = "well within tolerance (green)" if gap_abs < 0.10 else "within tolerance (yellow)"
            info_html = (
                "<strong>All selection rules agree - safe pick.</strong><br><br>"
                "The naive winner also passes the gap-adjusted test: gap = "
                f"<span style='color:{gap_color};font-weight:600'>{gap_str}</span> "
                f"({thresh}). "
                f"The one-SE rule confirms the same model (within the "
                f"{ONE_SE_MARGIN} R² equivalence margin). "
                "No overfitting concern for this experiment × target combination."
            )
    else:
        state = "disagree"
        gadj_r2  = r.get("gadj_R2",  float("nan"))
        gadj_gap = r.get("gadj_gap", float("nan"))
        gap_str  = f"{gap:+.3f}"     if not pd.isna(gap)      else "-"
        gr2_str  = f"{gadj_r2:+.3f}" if not pd.isna(gadj_r2)  else "-"
        gg_str   = f"{gadj_gap:+.3f}" if not pd.isna(gadj_gap) else "-"
        gg_color = ("#2ecc71" if (not pd.isna(gadj_gap) and abs(gadj_gap) < 0.10) else
                    "#f1c40f" if (not pd.isna(gadj_gap) and abs(gadj_gap) < 0.25) else "#e74c3c")
        info_html = (
            "<strong>Rules disagree - gap-adjusted recommends a different model.</strong>"
            "<br><br>"
            f"<em>Naive winner:</em> {naive} "
            f"(R² = {_cell(r.get('naive_R2'))}, "
            f"gap = <span style='color:#e74c3c;font-weight:600'>{gap_str}</span>) "
            "- overfit flag.<br>"
            f"<em>Gap-adjusted:</em> {gadj} "
            f"(R² = {gr2_str}, "
            f"gap = <span style='color:{gg_color};font-weight:600'>{gg_str}</span>) "
            "- preferred for deployment.<br><br>"
            "The gap-adjusted model trades a small R² reduction for a much smaller "
            "overfitting gap, suggesting better generalisation on out-of-distribution data."
        )

    return state, info_html


# State cell-background constants
# Applied to One-SE + Gap-adj cell groups (state agree_ok/agree_bad).
# For 'disagree': naive cells get a mild warning tint, gadj cells get a blue recommendation tint.
_STATE_ALT_BG = {
    "agree_ok":  "background:rgba(46,204,113,0.12);",
    "agree_bad": "background:rgba(231,76,60,0.10);",
    "disagree":  "",
}
_DISAGREE_NAIVE_BG = "background:rgba(241,196,15,0.07);"
_DISAGREE_ALT_BG   = "background:rgba(74,144,217,0.12);"


def _popup_btn(info_html):
    """Return HTML for a ⓘ button that triggers the shared overlay popup."""
    escaped = _html_lib.escape(info_html, quote=True)
    return (f"<span class='sel-info-btn' data-popup=\"{escaped}\" "
            f"title='Why this selection?'>ⓘ</span>")


# Shared popup overlay (rendered once per page / section).
_SEL_OVERLAY_HTML = (
    "<div id='sel-popup-overlay' style='"
    "display:none;position:fixed;z-index:1000;"
    "background:#ffffff;"
    "border:1px solid #cccccc;"
    "border-radius:6px;padding:14px 16px 12px;"
    "max-width:420px;min-width:260px;"
    "font-size:12px;line-height:1.7;"
    "color:#1a1a1a;"
    "box-shadow:0 8px 24px rgba(0,0,0,0.18);"
    "white-space:normal'>"
    "<button onclick=\"document.getElementById('sel-popup-overlay').style.display='none'\" "
    "style='float:right;background:none;border:none;cursor:pointer;"
    "font-size:16px;color:#888888;line-height:1;"
    "margin:-4px -4px 6px 10px'>×</button>"
    "<div id='sel-popup-content'></div>"
    "</div>"
)

_SEL_POPUP_JS = """
<script>
(function(){
  var overlay = document.getElementById('sel-popup-overlay');
  if (!overlay) return;
  document.querySelectorAll('.sel-info-btn').forEach(function(btn){
    btn.addEventListener('click', function(e){
      document.getElementById('sel-popup-content').innerHTML = btn.dataset.popup || '';
      var rect = btn.getBoundingClientRect();
      var top  = rect.bottom + 8;
      var left = rect.left;
      if (left + 430 > window.innerWidth) left = Math.max(0, window.innerWidth - 440);
      if (top  + 220 > window.innerHeight) top  = Math.max(0, rect.top - 228);
      overlay.style.top  = top  + 'px';
      overlay.style.left = left + 'px';
      overlay.style.display = 'block';
      e.stopPropagation();
    });
  });
  document.addEventListener('click', function(){ overlay.style.display = 'none'; });
  overlay.addEventListener('click', function(e){ e.stopPropagation(); });
})();
</script>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Table renderers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_global_table(df: pd.DataFrame) -> str:
    """8-row global table.

    If ``winner_*`` columns are present, they are used as the displayed global
    winner. This lets the embedded report align the global summary with the
    highlighted winner rows in the per-experiment table, including cases where
    the final pick comes from the One-SE column rather than the naive winner.
    Row background depends on state (agree_ok → green, agree_bad → red, disagree → split).
    ⓘ button in winner cell links to shared overlay popup.
    """
    head = [
        "Target",
        "Overall winner", "Rule", "R²", "Gap", "Score", "RMSE", "MAE",
        "Pareto frontier",
    ]
    rows = []
    for _, r in df.iterrows():
        state, info_html = _row_state(r)

        def _td(content, extra_style="", cls=""):
            cls_attr = f" class='{cls}'" if cls else ""
            return f"<td{cls_attr} style='{extra_style}'>{content}</td>"

        winner_model = r.get("winner_model", r["gadj_model"])
        winner_rule  = r.get("winner_rule", "Gap-adj")
        winner_R2    = r.get("winner_R2", r["gadj_R2"])
        winner_gap   = r.get("winner_gap", r["gadj_gap"])
        winner_score = r.get("winner_score", r["gadj_score"])
        winner_RMSE  = r.get("winner_RMSE", r.get("gadj_RMSE", float("nan")))
        winner_MAE   = r.get("winner_MAE",  r.get("gadj_MAE",  float("nan")))
        if "winner_model" in r.index:
            naive = str(r.get("naive_model", ""))
            info_html = (
                f"<strong>Global winner follows the lower table.</strong><br><br>"
                f"The highlighted per-target row underneath is {r.get('exp_key')}. "
                f"The selected global model is the <em>{winner_rule}</em> pick: "
                f"{winner_model} (R² = {_cell(winner_R2)}, "
                f"gap = {_cell(winner_gap)}).<br><br>"
                f"The row's naive winner is {naive} "
                f"(R² = {_cell(r.get('naive_R2'))}, gap = {_cell(r.get('naive_gap'))}). "
                "When the One-SE rule selects a smaller-gap model inside the competitive "
                "R² window, that conservative pick is carried into the global summary."
            )
        btn = _popup_btn(info_html)

        winner_R2_color = f"color:{_r2_color(winner_R2)}"
        winner_gap_cls  = _gap_cls(winner_gap)

        tds = [
            f"<td><strong>{r['target_short']}</strong></td>",
            # Overall winner block
            f"<td>{winner_model}{btn}</td>",
            _td(winner_rule, "font-size:11px;color:#555555"),
            _td(_cell(winner_R2),                winner_R2_color),
            _td(_cell(winner_gap),               "", winner_gap_cls),
            _td(_cell(winner_score, d=5),        ""),
            _td(_cell(winner_RMSE),              ""),
            _td(_cell(winner_MAE),               ""),
            # Meta
            f"<td style='font-size:11px'>{r['pareto']}</td>",
        ]
        rows.append("<tr>" + "".join(tds) + "</tr>")

    thead = "".join(f"<th>{h}</th>" for h in head)
    return ("<table class='sel-table'><thead><tr>"
            + thead + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _render_perexp_table(df: pd.DataFrame) -> str:
    """Per-(experiment × target) table.

    Columns (logical indices; 0-based, excluding the target rowspan col):
      0  Experiment
      1  Naive model    (no sort header)
      2  Naive R²
      3  Naive Gap
      4  Naive Score
      5  Naive RMSE
      6  Naive MAE
      7  One-SE model   (no sort header)
      8  One-SE R²
      9  One-SE Gap
      10 One-SE Score
      11 One-SE RMSE
      12 One-SE MAE
      13 Gap-adj model  (no sort header; ⓘ embedded)
      14 Gap-adj R²
      15 Gap-adj Gap
      16 Gap-adj Score
      17 Gap-adj RMSE
      18 Gap-adj MAE
      19 Gap-adj MdAE
      20 Pareto         (no sort)

    N_DATA_COLS = 21 (+ 1 target col = 22 total)
    """
    N_DATA_COLS = 21

    def _sh(label, col_idx, is_num=False):
        arr = f" <span class='sort-arr' data-col='{col_idx}'>↕</span>"
        return (f"<th class='perexp-sort-th' data-logical-col='{col_idx}'"
                f" data-num='{1 if is_num else 0}'>{label}{arr}</th>")

    head_ths = (
        "<th class='target-th'>Target</th>"
        + _sh("Experiment", 0)
        + "<th>Naive winner</th>"
        + _sh("R²",   2, True) + _sh("Gap", 3, True) + _sh("Score", 4, True)
        + _sh("RMSE", 5, True) + _sh("MAE", 6, True)
        + "<th>One-SE winner</th>"
        + _sh("R²",   8, True) + _sh("Gap", 9, True) + _sh("Score", 10, True)
        + _sh("RMSE", 11, True) + _sh("MAE", 12, True)
        + "<th>Gap-adj winner</th>"
        + _sh("R²",  14, True) + _sh("Gap",  15, True) + _sh("Score", 16, True)
        + _sh("RMSE", 17, True) + _sh("MAE",  18, True) + _sh("MdAE", 19, True)
        + "<th>Pareto</th>"
    )

    target_order = [t for t in TARGETS_ORDERED if t in df["target"].values]
    rows_html = []

    for i_tgt, tgt in enumerate(target_order):
        # Sort target groups strictly by the displayed gap-adjusted winner's
        # score. The score already contains the overfit penalty, so a lower-gap
        # row should only outrank a higher-score row when the score truly ties.
        grp = df[df["target"] == tgt].copy()
        grp["_abs_gadj_gap"] = grp["gadj_gap"].abs()
        grp = grp.sort_values(
            ["gadj_score", "_abs_gadj_gap", "naive_R2"],
            ascending=[False, True, False],
        )
        n   = len(grp)
        slug = TARGET_SHORT.get(tgt, tgt).lower().replace(" ", "-")

        # Compute best values per metric within this target group. A compact
        # blue star is appended to the winning cells instead of changing the
        # cell background, so the table remains readable and stable in width.
        def _group_max(col):
            vals = grp[col].dropna()
            return float(vals.max()) if not vals.empty else None

        def _group_min(col):
            vals = grp[col].dropna()
            return float(vals.min()) if not vals.empty else None
        best_r2   = _group_max("gadj_R2")
        best_score= _group_max("gadj_score")
        best_rmse = _group_min("gadj_RMSE")
        best_mae  = _group_min("gadj_MAE")
        best_mdae = _group_min("gadj_MdAE") if "gadj_MdAE" in grp.columns else None

        def _is_best(val, best):
            """True when val ties the group min within float tolerance."""
            if val is None or pd.isna(val) or best is None:
                return False
            return abs(float(val) - best) < 1e-9

        def _best_cls(is_best):
            return " class='metric-best-cell' title='Best value in this target group'" if is_best else ""

        for i_row, (_, r) in enumerate(grp.iterrows()):
            is_first = (i_row == 0)
            is_winner = is_first  # post-tiebreak winner is row 1 of the group
            state, info_html = _row_state(r)
            btn = _popup_btn(info_html)
            chg = "✓" if r["changed"] else ""
            dv  = lambda v: f"data-val='{_safe_float(v)}'"  # noqa: E731

            tds = []
            if is_first:
                tds.append(
                    f"<td class='target-cell' rowspan='{n}' "
                    f"data-val='{r['target_short']}'>"
                    f"{r['target_short']}</td>"
                )

            onese_same = r['onese_model'] == r['naive_model']
            gadj_same  = r['gadj_model'] == r['naive_model']

            gadj_gap_val = r.get('gadj_gap', float('nan'))
            is_severe = not pd.isna(gadj_gap_val) and abs(gadj_gap_val) >= 0.25

            if is_severe:
                gadj_bg = "background:rgba(231,76,60,0.12);"
            elif gadj_same:
                gadj_bg = "background:rgba(46,204,113,0.12);"
            else:
                gadj_bg = "background:rgba(74,144,217,0.12);"

            def _fmt_st(val_str, is_same):
                if is_same and val_str and val_str != "-":
                    return f"<s style='color:#888888;opacity:0.6'>{val_str}</s>"
                return val_str

            onese_R2   = _fmt_st(_cell(r['onese_R2']), onese_same)
            onese_gap  = _fmt_st(_cell(r['onese_gap']), onese_same)
            onese_score= _fmt_st(_cell(r['onese_score'], d=5), onese_same)
            onese_RMSE = _fmt_st(_cell(r.get('onese_RMSE', float('nan'))), onese_same)
            onese_MAE  = _fmt_st(_cell(r.get('onese_MAE', float('nan'))), onese_same)
            
            gadj_R2_val = r.get('gadj_R2', float('nan'))
            gadj_score_val = r.get('gadj_score', float('nan'))
            r2_best = _is_best(gadj_R2_val, best_r2)
            score_best = _is_best(gadj_score_val, best_score)

            gadj_R2    = (_cell(gadj_R2_val) if r2_best
                          else _fmt_st(_cell(gadj_R2_val), gadj_same))
            gadj_gap   = _fmt_st(_cell(r['gadj_gap']), gadj_same)
            
            gadj_score = (_cell(gadj_score_val, d=5) if score_best
                          else _fmt_st(_cell(gadj_score_val, d=5), gadj_same))
            
            gadj_RMSE_val  = r.get('gadj_RMSE',  float('nan'))
            gadj_MAE_val   = r.get('gadj_MAE',   float('nan'))
            gadj_MdAE_val  = r.get('gadj_MdAE',  float('nan'))
            rmse_best = _is_best(gadj_RMSE_val, best_rmse)
            mae_best  = _is_best(gadj_MAE_val,  best_mae)
            mdae_best = _is_best(gadj_MdAE_val, best_mdae)
            gadj_RMSE  = (_cell(gadj_RMSE_val) if rmse_best
                          else _fmt_st(_cell(gadj_RMSE_val), gadj_same))
            gadj_MAE   = (_cell(gadj_MAE_val)  if mae_best
                          else _fmt_st(_cell(gadj_MAE_val),  gadj_same))
            gadj_MdAE  = (_cell(gadj_MdAE_val) if mdae_best
                          else _fmt_st(_cell(gadj_MdAE_val), gadj_same))
            onese_R2_color = "" if onese_same else f"color:{_r2_color(r['onese_R2'])}"
            onese_gap_cls  = "" if onese_same else _gap_cls(r['onese_gap'])
            gadj_R2_color  = "" if gadj_same else f"color:{_r2_color(r['gadj_R2'])}"
            gadj_gap_cls   = "" if gadj_same else _gap_cls(r['gadj_gap'])

            tds += [
                # Experiment (col 0)
                f"<td {dv(r['exp_key'])}>{r['exp_key']}</td>",
                # Naive block (cols 1-6)
                f"<td {dv(r['naive_model'])}>"
                    f"<strong>{r['naive_model']}</strong></td>",
                f"<td {dv(r['naive_R2'])} style='color:{_r2_color(r['naive_R2'])}'>"
                    f"{_cell(r['naive_R2'])}</td>",
                f"<td {dv(r['naive_gap'])} class='{_gap_cls(r['naive_gap'])}'>"
                    f"{_cell(r['naive_gap'])}</td>",
                f"<td {dv(r['naive_score'])}>"
                    f"{_cell(r['naive_score'], d=5)}</td>",
                f"<td {dv(r.get('naive_RMSE', float('nan')))}>"
                    f"{_cell(r.get('naive_RMSE', float('nan')))}</td>",
                f"<td {dv(r.get('naive_MAE', float('nan')))}>"
                    f"{_cell(r.get('naive_MAE', float('nan')))}</td>",
                # One-SE block (cols 7-12)
                f"<td {dv(r['onese_model'])}>{r['onese_model']}</td>",
                f"<td {dv(r['onese_R2'])} style='{onese_R2_color}'>{onese_R2}</td>",
                f"<td {dv(r['onese_gap'])} class='{onese_gap_cls}'>{onese_gap}</td>",
                f"<td {dv(r['onese_score'])}>{onese_score}</td>",
                f"<td {dv(r.get('onese_RMSE', float('nan')))}>{onese_RMSE}</td>",
                f"<td {dv(r.get('onese_MAE', float('nan')))}>{onese_MAE}</td>",
                # Gap-adj block (cols 13-19)
                f"<td {dv(r['gadj_model'])} style='{gadj_bg}'>"
                    f"{r['gadj_model']}{btn}</td>",
                f"<td{_best_cls(r2_best)} {dv(r['gadj_R2'])} style='{gadj_bg}{gadj_R2_color}'>{gadj_R2}</td>",
                f"<td {dv(r['gadj_gap'])} class='{gadj_gap_cls}' style='{gadj_bg}'>{gadj_gap}</td>",
                f"<td{_best_cls(score_best)} {dv(r['gadj_score'])} style='{gadj_bg}'>{gadj_score}</td>",
                f"<td{_best_cls(rmse_best)} {dv(gadj_RMSE_val)} style='{gadj_bg}'>{gadj_RMSE}</td>",
                f"<td{_best_cls(mae_best)} {dv(gadj_MAE_val)}  style='{gadj_bg}'>{gadj_MAE}</td>",
                f"<td{_best_cls(mdae_best)} {dv(gadj_MdAE_val)} style='{gadj_bg}'>{gadj_MdAE}</td>",
                # Meta (col 20)
                f"<td style='font-size:10px'>{r['pareto']}</td>",
            ]

            cls_parts = []
            if is_winner:
                cls_parts.append("winner-row")
            cls_attr = f" class='{' '.join(cls_parts)}'" if cls_parts else ""
            attrs = f"data-group='{slug}'"
            if is_first:
                attrs += " data-is-first='true'"
            rows_html.append(f"<tr{cls_attr} {attrs}>{''.join(tds)}</tr>")

        if i_tgt < len(target_order) - 1:
            rows_html.append(
                f"<tr class='group-gap'>"
                f"<td colspan='{N_DATA_COLS + 1}' "
                f"style='padding:6px;background:transparent;border:none'></td></tr>"
            )

    table_html = (
        f"<table id='perexp-tbl' class='sel-table sel-perexp-tbl'>"
        f"<thead><tr>{head_ths}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
    )
    return table_html + _perexp_sort_js()


def _perexp_sort_js() -> str:
    exp_order_json = json.dumps(EXP_CHART_ORDER)
    return f"""
<script>
(function(){{
  var EXP_ORDER = {exp_order_json};
  var sortState = null;  // {{col: int, asc: bool}}

  document.querySelectorAll('#perexp-tbl .perexp-sort-th').forEach(function(th){{
    th.style.cursor = 'pointer';
    th.addEventListener('click', function(){{
      var col   = parseInt(th.dataset.logicalCol);
      var isNum = th.dataset.num === '1';
      var asc   = !(sortState && sortState.col === col && sortState.asc);
      sortState = {{col: col, asc: asc}};
      document.querySelectorAll('#perexp-tbl .sort-arr').forEach(function(a){{
        a.textContent = (parseInt(a.dataset.col) === col) ? (asc ? '↑' : '↓') : '↕';
      }});
      sortPerexp(col, isNum, asc);
    }});
  }});

  function getVal(tr, logicalCol) {{
    var offset = tr.dataset.isFirst === 'true' ? 1 : 0;
    var cell = tr.cells[logicalCol + offset];
    if (!cell) return null;
    var v = cell.getAttribute('data-val');
    return (v === null || v === 'nan' || v === '') ? null : v;
  }}

  function sortPerexp(col, isNum, asc) {{
    var tbody = document.querySelector('#perexp-tbl tbody');
    var allTrs = Array.from(tbody.querySelectorAll('tr'));

    var groups = {{}}, groupOrder = [], gapRows = [];
    allTrs.forEach(function(tr){{
      if (tr.classList.contains('group-gap')){{ gapRows.push(tr); return; }}
      var g = tr.dataset.group;
      if (!g) return;
      if (!groups[g]){{ groups[g] = []; groupOrder.push(g); }}
      groups[g].push(tr);
    }});

    groupOrder.forEach(function(g){{
      var rows = groups[g];

      rows.sort(function(a, b){{
        var av = getVal(a, col), bv = getVal(b, col);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        if (isNum) {{
          var an = parseFloat(av), bn = parseFloat(bv);
          return asc ? an - bn : bn - an;
        }}
        // Experiment column (col 0): sort by canonical EXP_ORDER index
        if (col === 0) {{
          var ai = EXP_ORDER.indexOf(av), bi = EXP_ORDER.indexOf(bv);
          if (ai < 0) ai = 999;
          if (bi < 0) bi = 999;
          return asc ? ai - bi : bi - ai;
        }}
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});

      // Move the target-cell td to whichever row is now first
      var oldFirst = rows.find(function(r){{ return r.dataset.isFirst === 'true'; }});
      var newFirst = rows[0];
      if (oldFirst && newFirst !== oldFirst) {{
        var targetTd = oldFirst.querySelector('.target-cell');
        if (targetTd) {{
          oldFirst.removeChild(targetTd);
          delete oldFirst.dataset.isFirst;
          newFirst.insertBefore(targetTd, newFirst.firstChild);
          newFirst.dataset.isFirst = 'true';
        }}
      }}
    }});

    tbody.innerHTML = '';
    groupOrder.forEach(function(g, i){{
      groups[g].forEach(function(tr){{ tbody.appendChild(tr); }});
      if (gapRows[i]) tbody.appendChild(gapRows[i]);
    }});
  }}
}})();
</script>
"""


def _sort_per_target_winners(per_exp_df: pd.DataFrame, tgt: str) -> pd.DataFrame:
    """Return one target group in the same order used by the per-experiment table."""
    grp = per_exp_df[per_exp_df["target"] == tgt].copy()
    if grp.empty:
        return grp
    grp["_abs_gadj_gap"] = grp["gadj_gap"].abs()
    return grp.sort_values(
        ["gadj_score", "_abs_gadj_gap", "naive_R2"],
        ascending=[False, True, False],
    )


def build_global_from_per_experiment(per_exp_df: pd.DataFrame) -> pd.DataFrame:
    """Build global display rows from the highlighted per-target rows below.

    The global table is a synopsis of the per-experiment table. For each target,
    it takes the row that appears first in the lower table after the canonical
    score-first sort. The displayed winner then comes from the One-SE column when
    One-SE differs from the naive winner; otherwise it comes from the gap-adjusted
    column. That captures the conservative "overall winner" cases where One-SE is
    the intended pick even though the row's naive model has a slightly higher R².
    """
    rows = []
    for tgt in TARGETS_ORDERED:
        grp = _sort_per_target_winners(per_exp_df, tgt)
        if grp.empty:
            continue
        r = grp.iloc[0].copy()
        use_onese = str(r["onese_model"]) != str(r["naive_model"])
        prefix = "onese" if use_onese else "gadj"
        if use_onese:
            rule = (
                "One-SE / Gap-adj"
                if str(r["onese_model"]) == str(r["gadj_model"])
                else "One-SE"
            )
        else:
            rule = "Gap-adj" if str(r["gadj_model"]) != str(r["naive_model"]) else "All agree"

        out = r.to_dict()
        exp = str(r["exp_key"])
        out.update({
            "winner_model": f"{exp} · {r[f'{prefix}_model']}",
            "winner_rule": rule,
            "winner_R2": r[f"{prefix}_R2"],
            "winner_gap": r[f"{prefix}_gap"],
            "winner_score": r[f"{prefix}_score"],
            "winner_RMSE": r.get(f"{prefix}_RMSE", float("nan")),
            "winner_MAE": r.get(f"{prefix}_MAE", float("nan")),
            "winner_MdAE": r.get(f"{prefix}_MdAE", float("nan")),
            "changed": str(r[f"{prefix}_model"]) != str(r["naive_model"]),
        })
        rows.append(out)
    return pd.DataFrame(rows)


SEL_CSS = """
    .sel-table { width:100%; border-collapse:collapse; font-size:0.81rem; margin:10px 0 20px;
                 background:#ffffff; color:#1a1a1a; }
    .sel-table th, .sel-table td {
        padding:5px 10px; border-bottom:1px solid #e0e0e0;
        text-align:left; white-space:nowrap; }
    .sel-table th { background:#eeeeee; font-weight:600; color:#333333;
                    font-size:0.82rem; }
    .sel-table thead tr { border-bottom:2px solid #cccccc; }
    .sel-perexp-tbl .target-cell {
        font-weight:700; background:#eeeeee; color:#555555;
        border-right:2px solid #cccccc;
        text-align:center; vertical-align:middle;
        font-size:0.75rem; padding:6px 10px;
        letter-spacing:0.05em; text-transform:uppercase; }
    .sel-perexp-tbl .group-gap td { border:none; }
    /* Gap-adjusted winner row per target: light-blue highlight matching project
       accent colour. The target-cell (rowspan column) is explicitly excluded
       so its uniform group header styling is preserved. */
    .sel-perexp-tbl .winner-row td:not(.target-cell) {
        background:#E3F0FB; font-weight:600;
    }
    .sel-perexp-tbl td.metric-best-cell {
        position:relative;
        font-weight:700;
    }
    .sel-perexp-tbl td.metric-best-cell::after {
        content:'★';
        position:absolute;
        right:2px;
        top:50%;
        transform:translateY(-52%);
        color:#005FCC;
        font-size:11px;
        line-height:1;
        pointer-events:none;
        text-shadow:0 0 2px #ffffff, 0 0 2px #ffffff;
    }
    .perexp-sort-th:hover { background:#e0e0e0; }
    .sort-arr { font-size:10px; color:#888888; margin-left:2px; }
    .rule-card-sel { background:#f5f5f5; border-left:3px solid #4A90D9;
                     padding:12px 16px; margin:12px 0; border-radius:4px; }
    .rule-card-sel h4 { margin:0 0 6px 0; font-size:13px; color:#4A90D9; }
    .rule-card-sel p  { margin:2px 0; font-size:12px; color:#555555; }
    .sel-qcard { margin-bottom:1.1rem; border:1px solid var(--border);
                 border-radius:5px; overflow:hidden; background:#ffffff; }
    .sel-qhead { background:var(--bg-secondary); padding:0.45rem 0.8rem;
                 border-bottom:1px solid var(--border); display:flex;
                 align-items:baseline; gap:0.5rem; }
    .sel-qnum { color:#4A90D9; font-size:0.78em; font-weight:bold;
                letter-spacing:0.06em; flex-shrink:0; }
    .sel-qtitle { font-weight:bold; font-size:0.93em; line-height:1.4; color:#1a1a1a; }
    .sel-qbody { padding:0.6rem 0.8rem 0.55rem; }
    .sel-qbody p { margin:0; line-height:1.6; border-left:2px solid var(--border);
                   padding-left:0.6rem; color:#555555; font-size:12px; }
    .sel-tbl-wrap { overflow-x:auto; border:1px solid #cccccc; border-radius:4px; }
    /* ⓘ info button */
    .sel-info-btn {
        cursor:pointer; color:#4A90D9; font-size:12px;
        margin-left:5px; user-select:none; vertical-align:middle;
    }
    .sel-info-btn:hover { color:#74b3f0; }
    /* State legend */
    .sel-state-legend {
        display:flex; gap:16px; flex-wrap:wrap;
        margin:8px 0 12px; font-size:11px; color:#555555;
    }
    .sel-state-legend span { display:flex; align-items:center; gap:5px; }
    .sel-state-swatch {
        display:inline-block; width:12px; height:12px;
        border-radius:2px; flex-shrink:0;
    }
"""


def section_html(global_df: pd.DataFrame, per_exp_df: pd.DataFrame,
                 df_all: pd.DataFrame) -> str:
    """Return inner section HTML for embedding in the unified report (no html/head/body)."""
    global_display_df = build_global_from_per_experiment(per_exp_df)
    n_changed_global  = int(global_display_df["changed"].sum())
    n_changed_per_exp = int(per_exp_df["changed"].sum())

    legend = (
        "<div class='sel-state-legend'>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(46,204,113,0.35)'></span>"
        "All rules agree, gap ≤ 0.25</span>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(231,76,60,0.35)'></span>"
        "All rules agree, gap &gt; 0.25 - caution</span>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(74,144,217,0.35)'></span>"
        "Gap-adj picks a different model (blue = recommended)</span>"
        "</div>"
    )

    parts = [
        _SEL_OVERLAY_HTML,
        f"<style>{SEL_CSS}</style>",
        "<div class='sel-qcard'>",
        "  <div class='sel-qhead'><span class='sel-qnum'>Q1</span>"
        "  <span class='sel-qtitle'>Why re-rank?</span></div>",
        "  <div class='sel-qbody'><p>On this dataset (n_test ≈ 160-280 for the held-out 2025 year), "
        "Test R² alone is noisy. A model with R²=0.43 and gap=0.88 is not "
        "equivalent to one with R²=0.40 and gap=0.02 - the former is memorising.</p></div>",
        "</div>",
        "<div class='sel-qcard'>",
        "  <div class='sel-qhead'><span class='sel-qnum'>Q2</span>"
        "  <span class='sel-qtitle'>Which rules are applied?</span></div>",
        "  <div class='sel-qbody'><p>"
        f"<strong>Gap-adjusted:</strong> score = R²_test − {GAP_PENALTY} · max(0, |gap| − {GAP_TOLERANCE}). "
        f"<strong>One-SE:</strong> among models within {ONE_SE_MARGIN} R² of the top, pick the smallest |gap|. "
        "<strong>Pareto frontier:</strong> models not dominated on (R²_test ↑, |gap| ↓). "
        "<strong>RMSE and MAE</strong> are shown alongside each rule's winner because they carry "
        "operational units (mg/L for BOD/COD/TSS, pH units for pH).</p></div>",
        "</div>",
        "<h3 style='margin:20px 0 8px;font-size:15px'>Global best per target (across all experiments)</h3>",
        "<p style='color:#555555;font-size:12px;margin:0 0 10px'>"
        f"Naive winner differs from the selected overfit-aware winner in "
        f"<strong>{n_changed_global} / {len(global_display_df)}</strong> global target rows and "
        f"<strong>{n_changed_per_exp} / {len(per_exp_df)}</strong> per-experiment rows. "
        "The global rows below are taken from the highlighted winners in the Best per "
        "(experiment × target) table; when the lower-table winner is a One-SE pick, "
        "the global table now shows that One-SE model instead of the row's naive model.</p>",
        legend,
        "<p style='color:#555555;font-size:11px;margin:0 0 6px'>"
        "Click ⓘ on any overall winner for the reasoning behind that selection.</p>",
        "<div class='sel-tbl-wrap'>",
        _render_global_table(global_display_df),
        "</div>",
        "<h3 style='margin:20px 0 8px;font-size:15px'>Best per (experiment × target)</h3>",
        legend,
        "<p style='color:#555555;font-size:12px'>",
        "Initially sorted within each target group by <strong>gap-adjusted score</strong> (descending), "
        "tie-broken by smaller |gap| then higher R². The per-target winner is highlighted in "
        "<span style='background:#E3F0FB;padding:1px 4px;border-left:3px solid #4A90D9'>light blue</span>. "
        "Scores are shown to 5 decimal places so close ties are visible. "
        "A small <span style='color:#1F6FD1;font-size:10px;font-weight:700'>★</span> "
        "marks the target-group best in the Gap-adj R², Score, RMSE, MAE, and MdAE columns. "
        "Click column headers to re-sort. "
        "Click ⓘ on any Gap-adj winner cell for selection reasoning.</p>",
        "<div class='sel-tbl-wrap'>",
        _render_perexp_table(per_exp_df),
        "</div>",
        # Findings / interpretation
        _findings_section_html(),
        f"<p style='color:#555555;font-size:11px;margin-top:20px'>"
        f"Source: {len(df_all)} model evaluations across "
        f"{df_all['exp_key'].nunique()} experiments.</p>",
        _SEL_POPUP_JS,
    ]
    return "\n".join(parts)


def _findings_section_html() -> str:
    """Findings panel below the per-experiment table.

    Uses the same <details class='exp-details'> styling as the Findings sections
    in every experiment block so the look is consistent across the report.
    """
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

    q1 = (
        "The score subtracts a penalty equal to 0.5 x (|gap| - 0.10) when |gap| "
        "exceeds the 10-point tolerance. This discounts models that fit the "
        "training years well but show a large train-to-test drop. A slightly "
        "lower-R² model with a much smaller gap can therefore be the more "
        "deployable pick."
    )
    q2 = (
        "Gap-adjusted score is the primary ranking key because it already includes "
        "the overfit penalty. Smaller |gap| is used only when scores are equal. "
        "For Grab COD, Exp3-S2-FS XGB wins over Exp2-SE2-Comb-FS RF because its "
        "penalised score is higher while still remaining in the low-gap zone."
    )
    q3 = (
        "Each experiment keeps a different test population after dropna(), so "
        "RMSE, MAE, and MdAE across experiments are useful flags, not final "
        "verdicts. A lower error on a different population is not automatically "
        "better generalisation."
    )
    q4 = (
        "For BOD and TSS, MdAE is preferred because those targets have spike-heavy "
        "error distributions and RMSE can be dominated by a few unusual days. "
        "When RMSE and MdAE disagree for those targets, MdAE is the better typical-day "
        "reliability signal."
    )
    q5 = (
        "When Naive, One-SE, and Gap-adj disagree, the Gap-adj cells are tinted blue "
        "and the info button explains the choice. When all rules agree but |gap| is "
        "at least 0.25, the row is tinted red: it is still the best available row, "
        "but should be treated as lower-confidence in an operational dashboard."
    )
    return f"""
<details class="exp-details" id="overfit-aware-findings" open>
  <summary><span class="fold-icon">▶</span> Findings</summary>
  <div class="exp-body">
    {_qcard(1, "Why is the gap-adjusted winner the production pick?", q1)}
    {_qcard(2, "How are close score ties handled?", q2)}
    {_qcard(3, "How should RMSE / MAE / MdAE highlights be interpreted?", q3)}
    {_qcard(4, "Why does MdAE matter for BOD and TSS?", q4)}
    {_qcard(5, "What do blue and red row states mean?", q5)}
  </div>
</details>
"""


def render_html(global_df: pd.DataFrame, per_exp_df: pd.DataFrame,
                df_all: pd.DataFrame) -> str:
    css = dark_mode_css("""
    .sel-table { width:100%; border-collapse:collapse; font-size:12px; margin:10px 0 20px; }
    .sel-table th, .sel-table td { padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; }
    .sel-table th { background:var(--bg-soft); font-weight:600; }
    .gap-good { color:#2ecc71; font-weight:600; }
    .gap-warn { color:#f1c40f; }
    .gap-bad  { color:#e74c3c; font-weight:600; }
    .rule-card { background:var(--bg-soft); border-left:3px solid #4A90D9;
                 padding:12px 16px; margin:12px 0; border-radius:4px; }
    .rule-card h4 { margin:0 0 6px 0; font-size:13px; color:#4A90D9; }
    .rule-card p { margin:2px 0; font-size:12px; color:var(--text-muted); }
    .sel-info-btn { cursor:pointer; color:#4A90D9; font-size:12px;
                    margin-left:5px; user-select:none; vertical-align:middle; }
    .sel-info-btn:hover { color:#74b3f0; }
    .sel-state-legend { display:flex; gap:16px; flex-wrap:wrap;
        margin:8px 0 12px; font-size:11px; color:var(--text-muted); }
    .sel-state-legend span { display:flex; align-items:center; gap:5px; }
    .sel-state-swatch { display:inline-block; width:12px; height:12px;
        border-radius:2px; flex-shrink:0; }
    h2 { border-bottom:1px solid var(--border); padding-bottom:4px; }
    body { font-family: -apple-system, Segoe UI, sans-serif; padding:24px; }
    .container { max-width:1400px; margin:0 auto; }
    """)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    global_display_df = build_global_from_per_experiment(per_exp_df)
    n_changed_global  = int(global_display_df["changed"].sum())
    n_changed_per_exp = int(per_exp_df["changed"].sum())

    legend = (
        "<div class='sel-state-legend'>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(46,204,113,0.35)'></span>"
        "All rules agree, gap ≤ 0.25</span>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(231,76,60,0.35)'></span>"
        "All rules agree, gap &gt; 0.25 - caution</span>"
        "<span><span class='sel-state-swatch' "
        "style='background:rgba(74,144,217,0.35)'></span>"
        "Gap-adj picks a different model (blue = recommended)</span>"
        "</div>"
    )

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Best Models - Overfitting-Aware Selection</title>",
        f"<style>{css}</style></head><body>",
        _SEL_OVERLAY_HTML,
        "<div class='container'>",
        f"<h1>Overfitting-Aware Best-Model Selection</h1>",
        f"<p style='color:var(--text-muted)'>Generated {ts}</p>",
        "<div class='rule-card'><h4>Q1 Why re-rank?</h4>"
        "<p>On this dataset (n_test ≈ 160-280 for one held-out year), "
        "Test R² alone is noisy and treats a 0.43 / 0.88 model as equivalent "
        "to a 0.40 / 0.42 model. The former is memorising.</p></div>",
        "<div class='rule-card'><h4>Q2 Which rules are applied?</h4>"
        f"<p><strong>Gap-adjusted:</strong> score = R²_test − {GAP_PENALTY} · max(0, |gap| − {GAP_TOLERANCE}). "
        f"<strong>One-SE:</strong> among models within {ONE_SE_MARGIN} R² of the top, pick the smallest |gap|. "
        "<strong>Pareto frontier:</strong> models not dominated on (R²_test ↑, |gap| ↓).</p></div>",
        "<h2>Global best per target (across all experiments)</h2>",
        "<p style='color:var(--text-muted);font-size:12px'>"
        f"Naive winner differs from the selected overfit-aware winner in "
        f"<strong>{n_changed_global} / {len(global_display_df)}</strong> global target rows and "
        f"<strong>{n_changed_per_exp} / {len(per_exp_df)}</strong> per-experiment rows. "
        "The global rows are taken from the highlighted winners in the Best per "
        "(experiment × target) table.</p>",
        legend,
        "<p style='color:#555555;font-size:11px;margin:0 0 6px'>"
        "Click ⓘ on any overall winner for the reasoning behind that selection.</p>",
        "<div style='overflow-x:auto'>",
        _render_global_table(global_display_df),
        "</div>",
        "<h2>Best per (experiment × target)</h2>",
        legend,
        "<p style='color:var(--text-muted); font-size:12px'>"
        "Sorted within each target group by naive Test R² (descending). "
        "Click column headers to re-sort within groups - Experiment column follows "
        "canonical order (Exp1 → Exp8). "
        "Click ⓘ on any Gap-adj winner for selection reasoning.</p>",
        "<div style='overflow-x:auto'>",
        _render_perexp_table(per_exp_df),
        "</div>",
        f"<p style='color:var(--text-muted); font-size:11px; margin-top:30px'>"
        f"Source: {len(df_all)} model evaluations across "
        f"{df_all['exp_key'].nunique()} experiments.</p>",
        _SEL_POPUP_JS,
        f"<script>{DARK_MODE_JS}</script></div></body></html>",
    ]
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading unified results...")
    df_all = load_all_data()
    print(f"  {len(df_all)} rows from {df_all['exp_key'].nunique()} experiments")

    print("Applying selection rules...")
    global_df  = build_global(df_all)
    per_exp_df = build_per_experiment(df_all)

    xlsx_path = os.path.join(REPORTS_DIR, "best_models_selection.xlsx")
    with pd.ExcelWriter(xlsx_path) as xw:
        global_df.to_excel(xw, sheet_name="global", index=False)
        per_exp_df.to_excel(xw, sheet_name="per_experiment", index=False)
    print(f"  xlsx → {xlsx_path}")

    html = render_html(global_df, per_exp_df, df_all)
    html_path = os.path.join(REPORTS_DIR, "best_models_selection.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  html → {html_path}")

    # CLI summary
    print("\n=== Global (across experiments) ===")
    for _, r in global_df.iterrows():
        flag = " ← CHANGED" if r["changed"] else ""
        print(f"  {r['target_short']:9s}  naive={r['naive_model']:28s} "
              f"R²={r['naive_R2']:+.3f} gap={r['naive_gap']:+.3f}{flag}")
        if r["changed"]:
            print(f"             gadj ={r['gadj_model']:28s} "
                  f"R²={r['gadj_R2']:+.3f} gap={r['gadj_gap']:+.3f}")
            print(f"             1-SE ={r['onese_model']:28s} "
                  f"R²={r['onese_R2']:+.3f} gap={r['onese_gap']:+.3f}")

    n_changed = int(per_exp_df["changed"].sum())
    print(f"\nPer-experiment rows where rule disagrees with naive: "
          f"{n_changed}/{len(per_exp_df)}")


if __name__ == "__main__":
    main()
