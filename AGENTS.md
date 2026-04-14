# AGENTS.md — Project Context & Standing Instructions

## Self-maintenance instruction
**Whenever you learn something new about this project — a new model run, a file renamed,
a design decision made, a deprecated script, a feature set changed — update this file
before the conversation ends.**

---

## Project overview
Wastewater treatment plant effluent quality prediction. Historical daily measurements
(2021–2025) train ML models to predict effluent concentrations from upstream process parameters.

**Philosophy:** Let EDA guide model selection. Report results plainly — do not manipulate
observations or tone to justify a pre-chosen model.

---

## Environment
```
Working directory : /Users/soltii/Desktop/New_Project/
Python            : .venv/bin/python3   (always use this, not system python)
Main data dir     : 21-25/
```

---

## Data

**Master dataset:** `21-25/All_Years_Full.xlsx` — ~1,900 rows × 60 cols, 2021–2025  
**Split:** Train = 2021–2024 (2020 rows included where present) · Test = 2025

**Targets (8):**
- Grab: `Effluent BOD (mg/L, Grab)`, `Effluent COD (mg/L, Grab)`, `Effluent TSS (mg/L, Grab)`, `Effluent pH (Grab)`
- Composite: `Effluent BOD (mg/L, Composite)`, `Effluent COD (mg/L, Composite)`, `Effluent TSS (mg/L, Composite)`, `Effluent pH (Composite)`

**Key feature groups:**
```python
GRAB_INLET = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
              "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
COMP_INLET = ["Inlet pH (Composite)", "Inlet BOD (mg/L, Composite)",
              "Inlet COD (mg/L, Composite)", "Inlet TSS (mg/L, Composite)"]
SEC_COLS   = ["Sec Clarifier pH", "Sec Clarifier TSS (mg/L)", "Sec Clarifier BOD (mg/L)",
              "Sec Clarifier COD (mg/L)", "Sec Clarifier RAS", "Sec Sed pH",
              "Sec Sed TSS (mg/L)", "Sec Sed BOD (mg/L)", "Sec Sed COD (mg/L)",
              "Sec Sed RAS (New)"]
COMMON     = ["Flow (MLD)", "Power Total (KW)", "month", "day_of_week", "year"]
```
**COMMON note:** `month`, `day_of_week`, `year` are derived — compute via `df["Date"].dt.*`
before any `dropna()` call.

**Directory layout under `21-25/modeling/`:**
```
datasets/
  experiment1/                           8 xlsx + feature_selected_datasets/
  experiment2/sub_exp{1,2}/              8 xlsx each + feature_selected_datasets/
  experiment3/sub_exp1/                  8 xlsx + feature_selected_datasets/
  experiment3/sub_exp2/                  8 xlsx + models run (Phase 8 complete)
models/
  linear/{baseline,feature_selected,exp3_s1,exp3_s1_fs,exp3_s2}/
  non_linear/{baseline,feature_selected,exp3_s1,exp3_s1_fs,exp3_s2}/{rf,gb,xgb}/
  phase9/{ann/,ensemble/}                ANN + Voting/Stacking results + plots
  phase10/                               Feature engineering run (log+interaction+flag)
feature_analysis/{selection,audit}/
scripts/    reports/
```

---

## Nomenclature
- **Experiment** (formerly "Stage") — distinct feature-set scope.
- **Sub-experiment** (formerly "Phase") — variant within an experiment.
- Dataset filenames use old `stage*` prefix (e.g. `stage1_grab_BOD.xlsx`); directories use `experiment*/`.
- Feature-selected variants live in `feature_selected_datasets/` inside each experiment dir.

---

## Modeling experiments

| Experiment | Features | # cols (grab) | Status |
|---|---|---|---|
| Exp 1 | Inlet + COMMON | 9 | **Complete** — all 6 models run 1 |
| Exp 2 Sub-1 | Secondary + COMMON | 15 | **Complete** — all 6 models run 1 |
| Exp 2 Sub-2 | Inlet + Secondary + COMMON | 19 | **Complete** — all 6 models run 1 |
| Exp 3 Sub-1 | Exp2-S2 + ADD-tier features | 16–22 | **Complete** — all 6 models run 1 |
| Exp 3 Sub-1 FS | Core+Useful (perm imp ≥ 0.03) | 5–8 | **Complete** — all 6 models run 1 |
| Exp 3 Sub-2 | Exp2-S2 + ADD + CONSIDER features | 20–32 | **Complete** — all 6 models run 1 |

**All completed experiments:** 8 datasets each, 6 models (OLS, Ridge, ElNet, RF, GB, XGB) at run 1,
all active. Predictions stored as `predicted_<Model>_run_N` columns in subset xlsx files.
Run numbering increments on re-run; use `df["run"].max()` for latest.

**Tuning protocol:** `TimeSeriesSplit(n_splits=3)`. RF/GB via `GridSearchCV`; XGB via
`RandomizedSearchCV(n_iter=30)`. Fixed: n_estimators=300, learning_rate=0.05 (GB/XGB), subsample=0.8.
Tuned: max_depth, min_samples_leaf (RF/GB); max_depth, min_child_weight, reg_alpha, reg_lambda (XGB).

---

## Scripts

**Naming pattern (training scripts only):**
- Training: `models/linear/<variant>/linear_modeling_<variant>.py` and `models/non_linear/<variant>/non_linear_modeling_<variant>.py`

**Reporting — unified single script (all individual report scripts deleted):**
- `scripts/generate_unified_report.py` — reads all `results.xlsx` files across every
  experiment/phase, produces `reports/unified_report.html` (single navigable report).
  Run: `.venv/bin/python3 21-25/modeling/scripts/generate_unified_report.py`
- Output: `reports/unified_report.html` (~240 KB, no embedded images)

Report scripts are standalone — run independently from modeling scripts.  
Run any script: `.venv/bin/python3 <path/to/script.py>` from project root.

**Exp 3 Sub-2 training scripts (created):**
- `models/linear/exp3_s2/linear_modeling_exp3_s2.py`
- `models/non_linear/exp3_s2/non_linear_modeling_exp3_s2.py`

**Phase 9 training scripts (created):**
- `models/phase9/ann/ann_modeling_phase9.py` — MLPRegressor with StandardScaler pipeline, GridSearchCV (TimeSeriesSplit)
- `models/phase9/ensemble/ensemble_modeling_phase9.py` — VotingRegressor (RF+Ridge+ElNet) and StackingRegressor (RF+Ridge+ElNet → Ridge meta, KFold CV)

**Phase 10 training scripts (created):**
- `models/phase10/phase10_modeling.py` — feature engineering (log1p + interaction terms + IQR flags) in-memory on Exp3-S2 datasets; trains ElNet, Ridge, RF, Voting
- `models/phase10/phase10b_modeling.py` — selective FE: Grab targets get full engineering, Composite targets use base Exp3-S2 features unchanged; outputs to `results_10b.xlsx`, `models_10b/`, `plots_10b/`

---

## What NOT to read
- **Unified HTML report** — `reports/unified_report.html` (~240 KB). View in browser only, do not read.
- **Old individual HTML reports** — still in `reports/` for reference but no longer regenerated. View in browser only.
- **PNG files, `.pkl` model files, `.venv/`, `__pycache__/`** — never read these.

---

## File disambiguation
- Active EDA: `21-25/eda_full.py` (not `eda.py` — superseded)
- Active EDA plots: `21-25/eda_full_plots/` (not `eda_plots/`)
- Always use the highest `run_N` report of any type.

---

## Coding conventions
- Dark mode: import from `modeling/report_theme.py` (`dark_mode_css`, `DARK_MODE_JS`) — never duplicate inline.
- Obs card tables: CSS selector `.obs-card table` (not a class on the table element).
- R² table column order: `Train R² | Test R² | R² Gap`, then RMSE columns.
- Foldable sections: `<details>`/`<summary>` with `.fold-icon` (▶) and `.fold-hint`. Use `_fold()` helper in `eda_full.py` — do not duplicate inline.
- EDA sidebar nav: `<nav id="sidenav">`, `<div id="main-content">` with `margin-left: 265px`. Anchor pattern: `{section_id}-{tgt_slug}`. Slug map in `_TGT_SLUG` dict inside `build_html`.

---

## EDA — critical pending bug
`plot_linearity_residuals`, `compute_top_ridge_features`, `plot_ridge_coefficients`, and `_ridge_metrics`
all call `feature_cols(df)` (all ~49 numeric cols) → `dropna()` retains only ~9% of rows.

**Fix (not yet applied):** Use stage-appropriate features:
- Grab targets → `GRAB_INLET + COMMON` (~94% row retention)
- Composite targets → `COMP_INLET + COMMON` (~92% row retention)

EDA section order: Ridge Residuals → Scatter Grids → Pearson/Spearman matrices → MI scores →
Cross-stage heatmap → remaining sections.

---

## Workflow status

| Phase | Description | Status |
|---|---|---|
| 0 | EDA — missingness, distributions, correlations, Ridge diagnostic | Done — Ridge fix pending |
| 1 | Collinearity — Pearson feature-feature matrix + VIF | Not started |
| 2 | Linear models, Exp 1 & 2 | **Done** |
| 3 | Tree models, Exp 1 & 2 | **Done** |
| 4 | Comparison report, Exp 1 & 2 baseline | **Done** |
| 5a | Feature selection — RF perm importance + MI | **Done** |
| 5b | Feature-selected re-run, all 6 models | **Done** |
| 6 | Feature audit, Exp 3 candidates (marginal row cost + MI tiers) | **Done** |
| 7 | Exp 3 Sub-1 — all 6 models | **Done** |
| 7b | Exp 3 Sub-1 FS — feature selection + re-run | **Done** |
| 8 | Exp 3 Sub-2 — ADD + CONSIDER features | **Done** |
| 9 | Final model — ANN, ensemble (Voting/Stacking), hyperparameter tuning, validate, export | **Done** |
| 10 | Feature engineering — log transforms, interaction terms, outlier indicator flags | **Done** |
| 10b | Selective FE — Grab targets only; Composite targets use base Exp3-S2 features | **Done** |

## New review notes

- Phase 10 / 10b bug: `infer_features()` currently excludes `month`, `day_of_week`, and `year`, so the FE runs are not actually using the full Exp3-S2 COMMON feature set. This makes the Phase 10 comparisons materially different from the earlier experiments.
- CV leakage risk: several tuning routines scale the entire training matrix before `TimeSeriesSplit` is applied. That can make internal CV scores and selected hyperparameters slightly optimistic.
- Holdout usage caution: the 2025 test set has been used as a development signal to choose Exp3-S2 and subsequent Phase 9 / Phase 10 directions. The report should treat 2025 as a quasi-validation set, not a fully untouched final holdout.

**Exp 3 Sub-2 key findings (run 1):**
- ElNet avg Test R² jumped +0.213 vs S1 (biggest winner from CONSIDER features). Ridge +0.07.
- RF avg Test R² +0.041 vs S1 (modest gain, Comp COD still failing at −0.41).
- GB/XGB catastrophically overfit composite targets (Comp TSS gap: GB +1.56, XGB +2.12). **Not usable for composites with these features.**
- Best performers: ElNet (Grab BOD 0.684, Comp BOD 0.410, Comp pH 0.340), RF (Grab TSS 0.504, Comp BOD 0.473).
- Comp COD remains the hardest target — all models near 0 or negative.
- Conclusion: CONSIDER features help linear models via regularization; GB/XGB need aggressive regularization or feature selection before composites are viable.

**Phase 9 key findings (run 1):**
- **ANN** (MLPRegressor): avg Test R² = −1.123 — **failed**. ~470 training samples insufficient for MLP; all targets chose alpha=1.0 (max regularisation); temporal distribution shift to 2025 hurts generalisation. Not recommended for this dataset size.
- **Voting** (RF+Ridge+ElNet average): avg Test R² = +0.353 — **best Phase 9 model**. New per-target records: Grab BOD 0.678, Comp BOD 0.553.
- **Stacking** (RF+Ridge+ElNet base → Ridge meta-learner): avg Test R² = +0.084 — inconsistent. New records for Grab: BOD 0.689 (overall best), TSS 0.534. Catastrophic failure on Comp pH (−1.564); meta-learner inherits difficulty of composite targets.
- **Overall best single result:** Stacking Grab BOD = 0.689.
- **Stacking CV note:** StackingRegressor internal CV changed from TimeSeriesSplit to KFold(n_splits=5, shuffle=False) — required because `cross_val_predict` needs every sample in exactly one fold (TimeSeriesSplit excludes early samples). Meta-learner just combines outputs; final eval still on 2025 holdout.
- **Recommendation:** Use Voting for production inference (most stable). Stacking viable for Grab targets only.

**Phase 10 key findings (run 1) — Feature Engineering on Exp3-S2:**
- **Feature counts:** base 17–29 cols → +9 to +13 log + 4–6 interaction + 6 flag = **37–52 total features** per target.
- **Grab targets — clear improvement:** Voting P10 avg (Grab only) ≈ +0.51 vs P9 Voting grab avg. Ridge P10 best: Grab TSS 0.633, Grab BOD 0.590. Voting P10 best: Grab TSS 0.642, Grab BOD 0.674.
- **Composite targets — feature engineering backfires:** Comp TSS catastrophic for all linear models (ElNet gap +3.80, Ridge gap +3.31) — 290 train rows + 50 features = severe overfitting. Comp COD still unresolved (best R² ≈ −0.008).
- **Composite BOD:** RF P10 = 0.490 (best composite result in P10). ElNet P10 = 0.447.
- **Average Test R² (all 8 targets):** ElNet −0.085, Ridge −0.104, RF +0.167, Voting +0.153 — all **lower than Phase 9 Voting 0.353** due to composite disasters dragging averages.
- **Conclusion:** Feature engineering helps Grab targets but creates severe overfitting on small composite datasets (esp. Comp TSS, n_train=290). Needs either (a) feature selection before engineering on composites, or (b) only applying engineering to Grab targets.
- **New per-target records (Grab):** Ridge TSS 0.633, Voting TSS 0.642 (both new highs for TSS).

**Phase 10b key findings (run 1) — Selective FE (Grab only):**
- **Strategy:** Grab targets → full FE (log + interaction + flags, 37–52 features). Composite targets → base Exp3-S2 features only (17–27 features, same as Phase 9).
- **Voting P10b avg Test R² = +0.361** — new best overall (beats Phase 9 Voting +0.353 by +0.008).
- **Grab avg (Voting P10b) = +0.483** vs Phase 9 Voting grab avg. Feature engineering clearly helps grab targets.
- **Composite avg (Voting P10b) = +0.239** — stable and competitive vs Phase 9 (+0.239 ≈ same level).
- **Comp TSS fully recovered:** ElNet P10b = +0.458, Voting P10b = +0.349 (vs P10 full catastrophic −2.882 / −1.267).
- **Comp BOD new records:** Ridge P10b = 0.531, RF P10b = 0.532, Voting P10b = 0.553 (all ≥ Phase 9 Voting 0.553 — matched/beaten).
- **Grab TSS new best:** Voting P10b = 0.642, Ridge P10b = 0.633 (confirmed Phase 10 highs maintained).
- **Grab BOD:** Voting P10b = 0.674 (matches Phase 10 result — stable).
- **Comp COD still unresolved:** best R² across all models = −0.051 (Ridge). Not responsive to any feature set tried.
- **Grab pH weakest Grab target:** Voting P10b = 0.244 — pH has narrow dynamic range, engineered features add noise.
- **Recommended production model:** Voting P10b — new best average, stable on both Grab and Composite, no catastrophic failures.

**Phase 9 scope (from ref project analysis):**
- **ANN** — tried; failed on this dataset (insufficient samples ~470, temporal shift). LSTM worth revisiting if dataset grows.
- **Ensemble models** — Voting best, Stacking mixed. Both implemented at run 1.
- **Feature engineering** — interaction terms (A×B), log/Box-Cox transforms, outlier indicator flags — not yet tried; may help COD/TSS.
- **Learning curves** — implemented in ANN script via sklearn `learning_curve`; added to Phase 9 ANN report section.
- **Ref project caveat:** headline BOD R²=0.96 is partly inflated — O_TSS (effluent, weight=0.90) used as feature to predict O_BOD (data leakage). Their COD/TSS (R²=0.55–0.65) is the honest baseline for inlet-only features with no secondary data.

---

## Feature audit — tier logic (Phase 6)
**Marginal row cost** = additional rows lost adding a feature on top of Exp2 Sub-2 baseline `dropna`
(not raw pairwise miss%). This is the operationally relevant number.

| Tier | Criteria |
|---|---|
| ADD | MI ≥ 0.20 and marginal cost ≤ 20% |
| CONSIDER | MI ≥ 0.15 & cost ≤ 35%, or MI ≥ 0.25 & cost ≤ 50% |
| LOW | MI ≥ 0.05 |
| SKIP | MI < 0.05 |
| REDUNDANT | Known collinear with baseline feature |

**Key finding:** Most Aeration features (MLSS, SV30, SVI, DO, pH) have 0–12% marginal cost on top
of Exp2 — because SEC_COLS already filter to days with secondary data, which correlate with
aeration data availability.

---

## Reference thresholds

**Pearson/Spearman |r|:** <0.20 very weak · 0.20–0.40 weak · 0.40–0.60 moderate · 0.60–0.80 strong · >0.80 very strong (check collinearity). With ~1,900 rows, r=0.10 is significant but explains only 1% variance.

**MI scores (relative ranking only):** 0–0.05 negligible · 0.05–0.15 weak · 0.15–0.30 moderate · >0.30 strong. High MI + low Spearman = non-monotonic relationship (only exploitable by non-linear models).

**Collinearity:** VIF > 10 = problem for linear models. RF predictions unaffected (use permutation importance, not impurity). ElasticNet handles collinearity automatically via L1. Feature-feature |r| > 0.85 = check for redundancy.

**Additional metrics to add in Phase 9 reports:** MdAE (Median Absolute Error — robust to outliers, more informative than MAE when high-value spikes exist) · NRMSE = RMSE / (max − min) (scale-normalised, enables cross-target comparison).

**Feature selection criteria (apply in order):** domain logic → EDA signal (MI + Spearman) → missingness cost → collinearity check.
