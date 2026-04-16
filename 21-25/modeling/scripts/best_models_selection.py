"""
best_models_selection.py — Reinterpret leaderboard under overfitting-aware rules.

Re-ranks every (target, experiment) row in the unified results using three
complementary rules instead of raw max(Test R²):

  1. Naive:            argmax(R2_test)                                  — current rule
  2. Gap-adjusted:     argmax(R2_test − λ · max(0, R2_gap − τ))         — penalise overfit
                         where τ=0.10 (tolerated gap), λ=0.5
  3. One-SE rule:      among models within δ=0.03 R² of the top,
                         pick the smallest |R2_gap|
  4. Pareto frontier:  models not dominated in (R2_test↑, |R2_gap|↓)

Outputs:
  reports/best_models_selection.xlsx   — per-target comparison table
  reports/best_models_selection.html   — dark-mode HTML panel

Run:  .venv/bin/python3 21-25/modeling/scripts/best_models_selection.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Reuse the unified loader — single source of truth for schema harmonisation.
sys.path.insert(0, SCRIPT_DIR)
from generate_unified_report import (           # noqa: E402
    load_all_data, TARGETS_ORDERED, TARGET_SHORT,
)
sys.path.insert(0, MODELING_DIR)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

# ── Selection-rule parameters ──────────────────────────────────────────────────
GAP_TOLERANCE  = 0.10   # τ — gaps below this are not penalised
GAP_PENALTY    = 0.50   # λ — weight on excess gap
ONE_SE_MARGIN  = 0.03   # δ — R² margin counted as "equivalent"


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
    """Apply every rule to one (target, experiment) slice. Returns a row-dict."""
    sub = sub.dropna(subset=["R2_test"]).copy()
    if sub.empty:
        return {}
    sub["gap_adj"] = sub.apply(
        lambda r: _gap_adjusted_score(r["R2_test"], r["R2_gap"]), axis=1
    )
    # 1. Naive
    naive_row = sub.loc[sub["R2_test"].idxmax()]
    # 2. Gap-adjusted
    gadj_row  = sub.loc[sub["gap_adj"].idxmax()] if sub["gap_adj"].notna().any() else naive_row
    # 3. One-SE
    top_r2 = sub["R2_test"].max()
    within = sub[sub["R2_test"] >= top_r2 - ONE_SE_MARGIN].copy()
    within["abs_gap"] = within["R2_gap"].abs()
    onese_row = within.loc[within["abs_gap"].idxmin()]
    # 4. Pareto
    front = _pareto_front(sub)

    return dict(
        naive_model   = naive_row["model"],
        naive_R2      = naive_row["R2_test"],
        naive_gap     = naive_row["R2_gap"],
        gadj_model    = gadj_row["model"],
        gadj_R2       = gadj_row["R2_test"],
        gadj_gap      = gadj_row["R2_gap"],
        gadj_score    = gadj_row["gap_adj"],
        onese_model   = onese_row["model"],
        onese_R2      = onese_row["R2_test"],
        onese_gap     = onese_row["R2_gap"],
        pareto        = ", ".join(front),
        n_pareto      = len(front),
        changed       = (naive_row["model"] != gadj_row["model"])
                         or (naive_row["model"] != onese_row["model"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Build leaderboards
# ═══════════════════════════════════════════════════════════════════════════════

def build_per_experiment(df_all: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, target)."""
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
                         "target_short": TARGET_SHORT.get(tgt, tgt), **picks})
    return pd.DataFrame(recs)


def build_global(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Global winner per target across ALL experiments/phases — the real
    'production pick' question.
    """
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
# HTML render
# ═══════════════════════════════════════════════════════════════════════════════

def _cell(val, d=3):
    if pd.isna(val):
        return "—"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return f"{val:.{d}f}"

def _r2_color(v):
    if pd.isna(v): return "var(--text-muted)"
    if v >= 0.6:  return "#2ecc71"
    if v >= 0.4:  return "#52c98a"
    if v >= 0.2:  return "#f1c40f"
    if v >= 0.0:  return "#e67e22"
    return "#e74c3c"

def _gap_cls(v):
    if pd.isna(v): return ""
    a = abs(v)
    return "gap-good" if a < 0.10 else ("gap-warn" if a < 0.25 else "gap-bad")


def _render_table(df: pd.DataFrame, include_exp: bool) -> str:
    head = []
    if include_exp:
        head.append("Experiment")
    head += [
        "Target",
        "Naive winner", "R²", "Gap",
        "Gap-adjusted winner", "R²", "Gap", "Score",
        "One-SE winner", "R²", "Gap",
        "Pareto frontier", "Changed?",
    ]
    rows = []
    for _, r in df.iterrows():
        tds = []
        if include_exp:
            tds.append(f"<td>{r['exp_key']}</td>")
        chg = "✓" if r["changed"] else ""
        tds += [
            f"<td><strong>{r['target_short']}</strong></td>",
            f"<td>{r['naive_model']}</td>",
            f"<td style='color:{_r2_color(r['naive_R2'])}'>{_cell(r['naive_R2'])}</td>",
            f"<td class='{_gap_cls(r['naive_gap'])}'>{_cell(r['naive_gap'])}</td>",
            f"<td><strong>{r['gadj_model']}</strong></td>",
            f"<td style='color:{_r2_color(r['gadj_R2'])}'>{_cell(r['gadj_R2'])}</td>",
            f"<td class='{_gap_cls(r['gadj_gap'])}'>{_cell(r['gadj_gap'])}</td>",
            f"<td>{_cell(r['gadj_score'])}</td>",
            f"<td>{r['onese_model']}</td>",
            f"<td style='color:{_r2_color(r['onese_R2'])}'>{_cell(r['onese_R2'])}</td>",
            f"<td class='{_gap_cls(r['onese_gap'])}'>{_cell(r['onese_gap'])}</td>",
            f"<td style='font-size:11px'>{r['pareto']}</td>",
            f"<td style='text-align:center'>{chg}</td>",
        ]
        rows.append("<tr>" + "".join(tds) + "</tr>")
    thead = "".join(f"<th>{h}</th>" for h in head)
    return ("<table class='sel-table'><thead><tr>"
            + thead + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


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
    h2 { border-bottom:1px solid var(--border); padding-bottom:4px; }
    body { font-family: -apple-system, Segoe UI, sans-serif; padding:24px; }
    .container { max-width:1400px; margin:0 auto; }
    """)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_changed_global = int(global_df["changed"].sum())
    n_changed_per_exp = int(per_exp_df["changed"].sum())

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Best Models — Overfitting-Aware Selection</title>",
        f"<style>{css}</style></head><body><div class='container'>",
        f"<h1>Overfitting-Aware Best-Model Selection</h1>",
        f"<p style='color:var(--text-muted)'>Generated {ts}</p>",
        "<div class='rule-card'>",
        "  <h4>Why re-rank?</h4>",
        "  <p>On this dataset (n_test ≈ 160–280 for one held-out year), "
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
        _render_table(global_df, include_exp=False),
        "<h2>Best per (experiment × target)</h2>",
        "<p style='color:var(--text-muted); font-size:12px'>"
        "Rows where the gap-adjusted or one-SE rule disagrees with naive Test R² "
        "are flagged in the rightmost column.</p>",
        _render_table(per_exp_df, include_exp=True),
        f"<p style='color:var(--text-muted); font-size:11px; margin-top:30px'>"
        f"Source: {len(df_all)} model evaluations across "
        f"{df_all['exp_key'].nunique()} experiments.</p>",
        f"<script>{DARK_MODE_JS}</script></div></body></html>",
    ]
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading unified results…")
    df_all = load_all_data()
    print(f"  {len(df_all)} rows from {df_all['exp_key'].nunique()} experiments")

    print("Applying selection rules…")
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
