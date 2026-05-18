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
SCORE_NOISE_BAND = 0.005  # ε - gap-adjusted scores within this band are
                          # treated as a tie for winner selection. Differences
                          # below SE(R²) noise floor are not meaningful;
                          # smaller |gap| breaks the tie.


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
    # 2. Gap-adjusted (tiebreak: smaller |gap|, then higher R²).
    # Within SCORE_NOISE_BAND of the top score, treat rows as effectively tied -
    # score differences below SE(R²) ≈ 0.05 are not meaningful, so a row with
    # 0.0004 less score but a much smaller |gap| is the more defensible pick.
    if sub["gap_adj"].notna().any():
        gadj_max = sub["gap_adj"].max()
        gadj_cands = sub[sub["gap_adj"] >= gadj_max - SCORE_NOISE_BAND]
        gadj_cands = gadj_cands.sort_values(
            ["abs_gap", "R2_test"], ascending=[True, False]
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

    Columns (no Gap-adj R²):
      Target | Naive winner R² Gap RMSE MAE |
      One-SE winner R² Gap RMSE MAE |
      Gap-adj winner Gap Score RMSE MAE | Pareto | Changed?
    Row background depends on state (agree_ok → green, agree_bad → red, disagree → split).
    ⓘ button in Gap-adj winner cell links to shared overlay popup.
    """
    head = [
        "Target",
        "Gap-adj winner", "R²", "Gap", "Score", "RMSE", "MAE",
        "Pareto frontier",
    ]
    rows = []
    for _, r in df.iterrows():
        state, info_html = _row_state(r)
        btn = _popup_btn(info_html)

        def _td(content, extra_style="", cls=""):
            cls_attr = f" class='{cls}'" if cls else ""
            return f"<td{cls_attr} style='{extra_style}'>{content}</td>"

        gadj_R2    = _cell(r["gadj_R2"])
        gadj_gap   = _cell(r["gadj_gap"])
        gadj_score = _cell(r["gadj_score"], d=5)
        gadj_RMSE  = _cell(r.get("gadj_RMSE", float("nan")))
        gadj_MAE   = _cell(r.get("gadj_MAE",  float("nan")))

        gadj_R2_color  = f"color:{_r2_color(r['gadj_R2'])}"
        gadj_gap_cls   = _gap_cls(r["gadj_gap"])

        tds = [
            f"<td><strong>{r['target_short']}</strong></td>",
            # Gap-adj block
            f"<td>{r['gadj_model']}{btn}</td>",
            _td(gadj_R2,                gadj_R2_color),
            _td(gadj_gap,               "", gadj_gap_cls),
            _td(gadj_score,             ""),
            _td(gadj_RMSE,              ""),
            _td(gadj_MAE,               ""),
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
        # Sort by gap-adjusted score (descending), but bucket scores by the
        # SCORE_NOISE_BAND ε so rows within ε share a sort key - within a bucket,
        # smaller |gap| (then higher R²) wins. The canonical winner therefore
        # appears at the top of each target group consistently with _pick_rules.
        grp = df[df["target"] == tgt].copy()
        grp["_abs_gadj_gap"] = grp["gadj_gap"].abs()
        grp["_score_bucket"] = (grp["gadj_score"] / SCORE_NOISE_BAND).round().fillna(-1e9)
        grp = grp.sort_values(
            ["_score_bucket", "_abs_gadj_gap", "naive_R2"],
            ascending=[False, True, False],
        )
        n   = len(grp)
        slug = TARGET_SHORT.get(tgt, tgt).lower().replace(" ", "-")

        # Compute best (minimum) value per operational metric within this target
        # group; the row whose cell equals the group-min gets a green ● prefix.
        def _group_min(col):
            vals = grp[col].dropna()
            return float(vals.min()) if not vals.empty else None
        best_rmse = _group_min("gadj_RMSE")
        best_mae  = _group_min("gadj_MAE")
        best_mdae = _group_min("gadj_MdAE") if "gadj_MdAE" in grp.columns else None

        def _is_best(val, best):
            """True when val ties the group min within float tolerance."""
            if val is None or pd.isna(val) or best is None:
                return False
            return abs(float(val) - best) < 1e-9

        # CSS for the best-cell highlight is injected via class .best-cell.
        # Applied to the <td> itself (not an inner span) so it overrides
        # the green/blue/red agree-state tints of the gap-adj block.

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
            
            gadj_R2    = _fmt_st(_cell(r['gadj_R2']), gadj_same)
            gadj_gap   = _fmt_st(_cell(r['gadj_gap']), gadj_same)
            
            gadj_score = _fmt_st(_cell(r['gadj_score'], d=5), gadj_same)
            
            gadj_RMSE_val  = r.get('gadj_RMSE',  float('nan'))
            gadj_MAE_val   = r.get('gadj_MAE',   float('nan'))
            gadj_MdAE_val  = r.get('gadj_MdAE',  float('nan'))
            # If cell is the per-target column-best, override gadj_bg with a bright
            # yellow inline style (no CSS class needed). The strikethrough on the
            # value is stripped here because the marker should read as positive.
            _BEST_BG = ("background:#FFD54F;border:2px solid #E0A800;"
                        "color:#1A1A1A;font-weight:700;")
            rmse_best = _is_best(gadj_RMSE_val, best_rmse)
            mae_best  = _is_best(gadj_MAE_val,  best_mae)
            mdae_best = _is_best(gadj_MdAE_val, best_mdae)
            gadj_RMSE  = (_cell(gadj_RMSE_val) if rmse_best
                          else _fmt_st(_cell(gadj_RMSE_val), gadj_same))
            gadj_MAE   = (_cell(gadj_MAE_val)  if mae_best
                          else _fmt_st(_cell(gadj_MAE_val),  gadj_same))
            gadj_MdAE  = (_cell(gadj_MdAE_val) if mdae_best
                          else _fmt_st(_cell(gadj_MdAE_val), gadj_same))
            rmse_bg = _BEST_BG if rmse_best else gadj_bg
            mae_bg  = _BEST_BG if mae_best  else gadj_bg
            mdae_bg = _BEST_BG if mdae_best else gadj_bg
            
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
                f"<td {dv(r['gadj_R2'])} style='{gadj_bg}{gadj_R2_color}'>{gadj_R2}</td>",
                f"<td {dv(r['gadj_gap'])} class='{gadj_gap_cls}' style='{gadj_bg}'>{gadj_gap}</td>",
                f"<td {dv(r['gadj_score'])} style='{gadj_bg}'>{gadj_score}</td>",
                f"<td {dv(gadj_RMSE_val)} style='{rmse_bg}'>{gadj_RMSE}</td>",
                f"<td {dv(gadj_MAE_val)}  style='{mae_bg}'>{gadj_MAE}</td>",
                f"<td {dv(gadj_MdAE_val)} style='{mdae_bg}'>{gadj_MdAE}</td>",
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
    /* Column-best highlight on RMSE/MAE/MdAE cells in the gap-adj block.
       !important is needed because the cell has an inline rgba tint from
       the agree/disagree state colouring. */
    .sel-perexp-tbl td.best-cell {
        background:#FFD54F !important;
        font-weight:700;
        border:2px solid #E0A800 !important;
        color:#1A1A1A;
        position:relative;
    }
    .sel-perexp-tbl td.best-cell s { color:#1A1A1A !important; opacity:1 !important; text-decoration:none; }
    .perexp-sort-th:hover { background:#e0e0e0; }
    .sort-arr { font-size:10px; color:#888888; margin-left:2px; }
    .rule-card-sel { background:#f5f5f5; border-left:3px solid #4A90D9;
                     padding:12px 16px; margin:12px 0; border-radius:4px; }
    .rule-card-sel h4 { margin:0 0 6px 0; font-size:13px; color:#4A90D9; }
    .rule-card-sel p  { margin:2px 0; font-size:12px; color:#555555; }
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
    n_changed_global  = int(global_df["changed"].sum())
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
        "<div class='rule-card-sel'>",
        "  <h4>Why re-rank?</h4>",
        "  <p>On this dataset (n_test ≈ 160-280 for the held-out 2025 year), "
        "Test R² alone is noisy. A model with R²=0.43 and gap=0.88 is not "
        "equivalent to one with R²=0.40 and gap=0.02 - the former is memorising.</p>",
        "  <h4>Rules applied</h4>",
        f"  <p><strong>Gap-adjusted:</strong> score = R²_test − {GAP_PENALTY} · max(0, |gap| − {GAP_TOLERANCE}). "
        "Penalises overfit beyond a 10-pt tolerance.</p>",
        f"  <p><strong>One-SE rule:</strong> among models within {ONE_SE_MARGIN} R² of the top, pick the smallest |gap|.</p>",
        "  <p><strong>Pareto frontier:</strong> models not dominated on (R²_test ↑, |gap| ↓).</p>",
        "  <p><strong>RMSE and MAE</strong> are shown alongside R² for each rule's winner - "
        "they carry the same ranking as R² within a target but are operationally interpretable "
        "(mg/L for BOD/COD/TSS, pH units for pH).</p>",
        "</div>",
        "<div class='rule-card-sel'>",
        f"  <h4>Headline</h4>",
        f"  <p>Naive winner differs from gap-adjusted / one-SE winner in "
        f"<strong>{n_changed_global} / {len(global_df)}</strong> global target rows and "
        f"<strong>{n_changed_per_exp} / {len(per_exp_df)}</strong> per-experiment rows.</p>",
        "</div>",
        "<h3 style='margin:20px 0 8px;font-size:15px'>Global best per target (across all experiments)</h3>",
        legend,
        "<p style='color:#555555;font-size:11px;margin:0 0 6px'>"
        "Click ⓘ on any Gap-adj winner for the reasoning behind that selection.</p>",
        "<div class='sel-tbl-wrap'>",
        _render_global_table(global_df),
        "</div>",
        "<h3 style='margin:20px 0 8px;font-size:15px'>Best per (experiment × target)</h3>",
        legend,
        "<p style='color:#555555;font-size:12px'>",
        "Initially sorted within each target group by <strong>gap-adjusted score</strong> (descending), "
        "tie-broken by smaller |gap| then higher R². The per-target winner is highlighted in "
        "<span style='background:#E3F0FB;padding:1px 4px;border-left:3px solid #4A90D9'>light blue</span>. "
        "Scores are shown to 5 decimal places so close ties are visible. "
        "Cells highlighted "
        "<span style='background:#FFD54F;color:#1A1A1A;padding:2px 6px;"
        "font-weight:700;border:2px solid #E0A800'>in yellow</span> "
        "(in the RMSE / MAE / MdAE columns) hold the lowest value in that column for the target group. "
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
    return """
<details class="exp-details" id="overfit-aware-findings" open>
  <summary><span class="fold-icon">▶</span> Findings  -  Overfit-Aware Model Selection</summary>
  <div class="exp-body">
    <div class="obs-card">
      <h4>Why the gap-adjusted winner is the production pick</h4>
      <p class="meta">The score subtracts a penalty equal to 0.5 × (|gap| − 0.10) when |gap| exceeds the 10-point
      tolerance. The intent is to discount models that fit 2024 well but show a large train-to-test
      drop - they are likely memorising patterns specific to a single year. A model with R²=0.51 and
      gap=0.15 is treated as worse than a model with R²=0.49 and gap=0.00, even though the former
      scored higher in 2025. This bets that the latter generalises better when 2026 data looks
      different.</p>
    </div>
    <div class="obs-card">
      <h4>Score noise band tiebreaker</h4>
      <p class="meta">Gap-adjusted scores that fall within 0.005 of the top score are treated as
      tied for selection. Differences below the standard error of R² (~0.04-0.06 at n_test ≈ 200)
      are not statistically meaningful, so within that band the row with the smaller |gap| wins.
      This is why for Grab BOD, Exp8 Ridge (R²=0.656, gap=-0.095) is preferred over Exp7-SE2 Voting
      (R²=0.674, gap=-0.136) - same score within noise, much lower overfit.</p>
    </div>
    <div class="obs-card">
      <h4>Cross-experiment RMSE / MAE / MdAE comparisons are NOT apples-to-apples</h4>
      <p class="meta">Each experiment requires a different feature set, so <code>dropna()</code>
      retains a different population at test time. A 168-row test set and a 223-row test set may
      simply be measuring different things. Lower RMSE on a different population is not the same as
      a better-generalising model. Treat the operational-metric highlights as "worth a second look",
      not as a verdict.</p>
    </div>
    <div class="obs-card">
      <h4>When to consider an alternative to the canonical winner</h4>
      <p class="meta">Look at a non-winner row when (a) its gap is meaningfully smaller, (b) it
      ranks well on RMSE / MAE / MdAE within the target group, and (c) its experiment uses a feature
      set you can realistically maintain in production. A row that wins on RMSE but is built on an
      experiment with sparse inputs (e.g. coliform measurements that arrive 3 days late) is usually
      not deployable.</p>
    </div>
    <div class="obs-card">
      <h4>MdAE is the project's primary metric for BOD and TSS</h4>
      <p class="meta">BOD and TSS have spike-dominated error distributions. RMSE is sensitive to
      those spikes and can rank models that simply got lucky on a quiet 2025. MdAE (median absolute
      error) reflects typical-day performance and is more robust. Where MdAE differs from the RMSE
      ranking for those targets, prefer MdAE.</p>
    </div>
    <div class="obs-card">
      <h4>Why some MdAE cells show "-"</h4>
      <p class="meta">MdAE is computed from row-level prediction columns stored alongside the raw
      datasets. Three groups of models do not write row-level predictions to disk, so their MdAE is
      unavailable: <strong>(1)</strong> all ANN runs, <strong>(2)</strong> Exp 7 (Feature
      Engineering) and Exp 8 (Temporal Features) which only export summary metrics, and
      <strong>(3)</strong> several feature-selected variants where the FS dataset directory was not
      materialised. The summary R² / Gap / Score columns are still reliable for those rows.</p>
    </div>
    <div class="obs-card">
      <h4>The three rule columns (Naive / One-SE / Gap-adj) usually agree</h4>
      <p class="meta">When they disagree, the row's Gap-adj cells are tinted in
      <span style='background:rgba(74,144,217,0.20);padding:1px 4px'>blue</span>
      (a different model is recommended) and the ⓘ button explains why. When all three rules agree
      but the gap is wide (|gap| ≥ 0.25), the row is tinted
      <span style='background:rgba(231,76,60,0.20);padding:1px 4px'>red</span>
      - it means no better-generalising alternative exists, so the model is still picked, but
      it should be flagged as low-confidence in any operational dashboard.</p>
    </div>
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
    n_changed_global  = int(global_df["changed"].sum())
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
        "<div class='rule-card'>",
        "  <h4>Why re-rank?</h4>",
        "  <p>On this dataset (n_test ≈ 160-280 for one held-out year), "
        "Test R² alone is noisy and treats a 0.43 / 0.88 model as equivalent "
        "to a 0.40 / 0.42 model. The former is memorising.</p>",
        "  <h4>Rules applied</h4>",
        f"  <p><strong>Gap-adjusted:</strong> score = R²_test − {GAP_PENALTY} · max(0, |gap| − {GAP_TOLERANCE}). "
        "Penalises overfit beyond a 10-pt tolerance.</p>",
        f"  <p><strong>One-SE rule:</strong> among models within {ONE_SE_MARGIN} R² of the top, pick the smallest |gap|.</p>",
        "  <p><strong>Pareto frontier:</strong> models not dominated on (R²_test ↑, |gap| ↓). All listed are defensible.</p>",
        "</div>",
        "<div class='rule-card'>",
        f"  <h4>Headline</h4>",
        f"  <p>Naive winner differs from gap-adjusted / one-SE winner in "
        f"<strong>{n_changed_global} / {len(global_df)}</strong> global target rows and "
        f"<strong>{n_changed_per_exp} / {len(per_exp_df)}</strong> per-experiment rows.</p>",
        "</div>",
        "<h2>Global best per target (across all experiments)</h2>",
        legend,
        "<p style='color:#555555;font-size:11px;margin:0 0 6px'>"
        "Click ⓘ on any Gap-adj winner for the reasoning behind that selection.</p>",
        "<div style='overflow-x:auto'>",
        _render_global_table(global_df),
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
