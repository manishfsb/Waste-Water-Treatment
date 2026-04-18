# CLAUDE.md — Project Context & Standing Instructions

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
Main data dir     : raw_data/
```

---

## Data

**Master dataset:** `raw_data/All_Years_Full.xlsx` — ~1,900 rows × 60 cols, 2021–2025  
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

**Directory layout under `modeling/`:**
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
  Run: `.venv/bin/python3 modeling/scripts/generate_unified_report.py`
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

**Phase 11 training script (created):**
- `models/phase11/phase11_modeling.py` — temporal features (lag 1/3/7 + calendar `7D` rolling mean with `closed='left'`) applied only to inlet + Flow + Power columns (~50–58 total features). Targets on log1p scale with Duan smearing back-transform (pH left untransformed). Trains Ridge, ElNet, RF, XGB, Voting. Tuning uses `TimeSeriesSplit(n_splits=5)`. Results include `CV_R2` and `Gap_gen = CV_R² − Test_R²` for distribution-shift diagnosis (separate from overfit gap).

**Diagnostic & selection scripts (created):**
- `scripts/best_models_selection.py` — re-ranks every (target × experiment) using three overfit-aware rules on top of the unified leaderboard: (a) gap-adjusted score = R²_test − 0.5·max(0, |gap|−0.10), (b) one-SE rule (within 0.03 R² of max, smallest |gap|), (c) Pareto frontier on (R²_test ↑, |gap| ↓). Outputs `reports/best_models_selection.{xlsx,html}`. Uses `load_all_data()` from `generate_unified_report.py` as single source of truth for schema normalisation.
- `scripts/error_decomposition.py` — decomposes 2025 residuals by Flow quartile / weekday-weekend / season / Inlet-load quartile using `predicted_Voting_run_1` from each s2 xlsx. Quartile thresholds fit on training set only. Outputs `reports/error_decomposition.{xlsx,html}`.

---

## What NOT to read
- **Unified HTML report** — `reports/unified_report.html` (~240 KB). View in browser only, do not read.
- **Old individual HTML reports** — still in `reports/` for reference but no longer regenerated. View in browser only.
- **PNG files, `.pkl` model files, `.venv/`, `__pycache__/`** — never read these.

---

## File disambiguation
- Active EDA: `eda/eda_full.py`
- Active EDA plots: `eda/plots/`
- Raw year data: `raw_data/20XX/`
- Extracted JSON data: `raw_data/20XX/extracted_data/`
- Dashboard: `dashboard/`
- Data prep scripts: `data_prep/` (check_missing, extract_full_dataset, merge_all_years, process_years, update_2021_first_half)
- Always use the highest `run_N` report of any type.

---

## Coding conventions
- Dark mode: import from `modeling/report_theme.py` (`dark_mode_css`, `DARK_MODE_JS`) — never duplicate inline.
- Obs card tables: CSS selector `.obs-card table` (not a class on the table element).
- R² table column order: `Train R² | Test R² | R² Gap`, then RMSE columns.
- Foldable sections: `<details>`/`<summary>` with `.fold-icon` (▶) and `.fold-hint`. Use `_fold()` helper in `eda/eda_full.py` — do not duplicate inline.
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
| 11 | Temporal features (lag 1/3/7 + 7-day rolling mean on inlets+Flow+Power) + log1p targets with Duan smearing | **Done** |
| Sel | Overfitting-aware model selection (Pareto / gap-adjusted / one-SE rules) | **Done** |
| Reg | 2025 residual decomposition by Flow quartile / season / weekday / inlet-load | **Done** |

**Exp 3 Sub-2 key findings (run 1):**
- ElNet avg Test R² jumped +0.213 vs S1 (biggest winner from CONSIDER features). Ridge +0.07.
- RF avg Test R² +0.041 vs S1 (modest gain, Comp COD still failing at −0.41).
- GB/XGB catastrophically overfit composite targets (Comp TSS gap: GB +1.56, XGB +2.12). **Not usable for composites with these features.**
- Best performers: ElNet (Grab BOD 0.684, Comp BOD 0.410, Comp pH 0.340), RF (Grab TSS 0.504, Comp BOD 0.473).
- Comp COD remains the hardest target — all models near 0 or negative.
- Conclusion: CONSIDER features help linear models via regularization; GB/XGB need aggressive regularization or feature selection before composites are viable.

**Phase 9 key findings (run 1 — CORRECTED; old run wiped):**

Methodology corrections applied (old run deleted, fresh run 1):
- **Voting composition changed:** Ridge removed (L2 redundancy with ElNet); XGBoost added for structural diversity. New composition: ElNet + RF + XGB.
- **Stacking CV corrected:** KFold(n_splits=5) had look-ahead bias (2021 samples predicted by models trained on 2022–2024 data). Replaced with manual walk-forward OOF using TimeSeriesSplit(n_splits=5). ~83% OOF coverage per target; first ~17% excluded from meta-learner training. No look-ahead bias.
- **Diagnostic metrics added to results.xlsx:** MAE_2024, MAE_2025 (=MAE_test), y_train_std, y_test_std, NRMSE_test for variance-collapse diagnosis.

Results (corrected run 1):
- **ANN** (MLPRegressor): avg Test R² = −1.123 — **failed** (not rerun; confirmed failed).
- **Voting** (ElNet+RF+XGB): avg Test R² = **+0.287**. Grab BOD 0.692, Grab COD 0.433, Comp BOD 0.454. Note: Comp BOD down from old +0.553 because Ridge was removed (Ridge helped composites).
- **Stacking** (ElNet+RF+XGB → Ridge meta, walk-forward OOF): avg Test R² = **+0.098**. Comp pH fixed from −1.564 (KFold leak) to −0.326 (genuine difficulty). Stacking is now methodologically clean.
- **Recommendation:** Voting (ElNet+RF+XGB) is the recommended Phase 9 model. Stacking now unbiased but weaker on average.

Variance diagnosis (corrected run, negative R² targets):
- **Comp COD (Voting −0.305):** σ-ratio=0.78 — variance NOT collapsed. MAE 2024→2025: 8.7→18.1 (doubled). Genuine model failure. Root cause: unknown 2025 process change.
- **Comp TSS (Stacking −0.171):** σ-ratio=0.34 — variance collapse. Plant stabilised TSS in 2025. MAE 2024→2025: 1.7→4.2 — real degradation too, but R² overstates severity.
- **Comp pH (Stacking −0.326):** σ-ratio=0.72, MAE improved 2024→2025 (0.256→0.189). R² negative purely due to variance collapse — absolute predictions actually improved. Not a model failure.
- **Key takeaway:** R² alone is misleading for composite targets on 2025 holdout. MAE/RMSE are primary quality metrics.

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

**Phase 11 key findings (run 1) — Temporal features + log1p targets:**
- **Feature expansion is restricted** to inlet + Flow + Power columns only (not every continuous col). Total feats per target = 50–58 (base 24–30 + lag 18–21 + roll 6–7). Rationale: full expansion (128 feats on ~400 rows) overfits composites catastrophically.
- **log1p target transform** applied to BOD/COD/TSS; pH untransformed. Back-transform uses **Duan smearing** (`expm1(ŷ_log) · mean(exp(training residuals))`).
- **CV_R2 and Gap_gen = CV_R² − Test_R²** recorded in results.xlsx for distribution-shift diagnosis separate from overfit gap.
- **Wins on Grab targets (cleaner gaps than Phase 10b per-row):**
  - Grab BOD **Ridge P11 = 0.656, gap = −0.095** — honest generalisation (vs P10b Voting 0.674 gap ~−0). Near-tie on R², lower model complexity.
  - Grab COD **XGB P11 = 0.464, gap = +0.057** — new Grab COD high with an honest gap (prior best XGB 0.526 had gap +0.378; selection-rule demotes it).
  - Grab COD **RF P11 = 0.305, CV_R² = 0.291, Gap_gen = −0.014** — the most consistent Grab COD model by CV.
- **Losses on Composite targets — log1p+lags overfits at n≈290:**
  - Comp TSS Ridge P11 = −1.04, gap = +1.82 — catastrophic; Phase 10b (base features, no log) remains better.
  - Comp BOD P11 avg ≈ 0.25 (worse than P10b Voting 0.553).
- **Average Test R² (all 8 targets):** Ridge +0.10, ElNet +0.16, RF +0.16, XGB +0.19, Voting +0.22 — all lower than P10b Voting 0.361 because composite regressions drag the average.
- **Conclusion:** log1p + temporal-lag features are **the right answer for Grab BOD/COD** (especially via Ridge + XGB). They **should not** be applied to composites — the sample size does not support the feature count. A full "best per target" production stack now reads:
  - Grab BOD → **Phase 11 Ridge** (0.656, gap −0.095)
  - Grab COD → **Phase 11 XGB** (0.464, gap +0.057) or **Phase 11 RF** (0.305, CV 0.291) for robustness
  - Grab TSS → **Phase 10b Voting** (0.642)
  - Grab pH  → Phase 10b Voting (0.244) — still weak; consider quantile regression
  - Comp BOD → **Phase 10b Voting** (0.553) or Ridge (0.531)
  - Comp COD → no model generalises (MAE 2024→2025 doubles from 9 → 17)
  - Comp TSS → **Phase 10b ElNet** (0.458)
  - Comp pH  → Phase 10b Voting (stable; R² misleading due to σ-collapse)
- **CV R² on early TimeSeriesSplit folds is very noisy** for Ridge (fold 1 trains on ~80 rows with ~55 features); interpret `CV_R2` per-fold rather than as a scalar mean.

**Overfit-aware selection rule (meta-finding):**
- 23 of 112 (target × experiment) rows change winner under the gap-adjusted or one-SE rule.
- Headline global reshuffles: Grab COD naive=XGB (R²=0.526, gap=+0.378) → gap-adjusted=RF (R²=0.489, gap=−0.004); Comp COD naive=XGB (R²=0.063, gap=+0.611) → gap-adjusted=ElNet (R²=−0.024, gap=+0.409).
- Rule parameters (`scripts/best_models_selection.py`): τ (gap tolerance) = 0.10, λ (penalty) = 0.50, δ (one-SE margin) = 0.03.

**2025 residual regime decomposition (meta-finding):**
Run `scripts/error_decomposition.py`. Uses Phase 9 Voting predictions. Headline:
- Error is **NOT uniform** — concentrates in high-Flow (Q4) and winter (DJF) regimes for every BOD/TSS target.
- Grab TSS: Q4 Flow MAE = 12.3 (**3.0× overall**), DJF MAE = 13.3 (**3.2× overall**).
- Comp BOD: Q4 Flow MAE = 5.2 (**3.1×**), DJF MAE = 4.3 (**2.6×**).
- Comp pH:  Q4 Flow MAE = 0.44 (**3.4× overall**) — very asymmetric.
- Grab BOD: DJF MAE = 3.4 (**2.2× overall**).
- **Operational implication:** flag predictions made during Q4 Flow or DJF as low-confidence; the models are fit for normal operation and fail under hydraulic/thermal stress. Collect more storm/winter samples or build a specialist model for those regimes before deploying.

**Phase 9 scope (from ref project analysis):**
- **ANN** — tried; failed on this dataset (insufficient samples ~470, temporal shift). LSTM worth revisiting if dataset grows.
- **Ensemble models** — Voting (ElNet+RF+XGB) best at avg R²=+0.287. Stacking (walk-forward OOF) avg R²=+0.098, methodologically clean but weaker. Both at run 1.
- **Feature engineering** — interaction terms (A×B), log/Box-Cox transforms, outlier indicator flags — tried in Phase 10/10b.
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

---

## Outlier audit (targets, April 2026)

Analysed from `raw_data/All_Years_Full.xlsx` using IQR rule (Q3 + 1.5×IQR threshold fit on full training set).

**No outlier treatment has been applied in any experiment (Exp 1–3, Phase 9–11).** Phase 10/10b added binary IQR *flag* features as model inputs but never removed or capped values. Phase 11 log1p-transforms the targets (BOD/COD/TSS), which partially mitigates target-side leverage.

| Target | Train n | Severe outliers (>Q3+3×IQR) | % severe | Train max | 2025 max |
|---|---|---|---|---|---|
| BOD Grab | 1245 | 72 | 5.8% | 167.1 | 22.5 |
| TSS Grab | 1423 | 140 | **9.8%** | **1266.0** | 46.7 |
| COD Grab | 1410 | 17 | 1.2% | 867.0 | 130.1 |
| BOD Comp | 868 | 40 | 4.6% | 191.3 | 21.5 |
| TSS Comp | 979 | 70 | 7.2% | 812.0 | 36.0 |
| COD Comp | 973 | 20 | 2.1% | 957.0 | 110.0 |

**Key findings:**
- **2022 is the outlier year.** BOD Grab: 29% of 2022 rows are above IQR threshold. TSS Grab: 31% in 2022. 2020 and 2021 have zero IQR outliers — likely the plant ran cleanly or sensor calibration differed.
- **TSS Grab max = 1266 mg/L** (median = 8 mg/L, 158× median). Almost certainly a plant upset or sensor error. This single value has high leverage on OLS/Ridge coefficients and RMSE.
- **2025 test set is calmer than training.** Test maxima are all well below training maxima — no test value exceeds the training P99 for any target. This means reported Test RMSE is *understated* relative to real operational conditions (models never saw 2022-level spikes at test time).
- **RMSE is unreliable as a primary metric** for TSS and BOD targets given the spike distribution. MdAE should be the primary metric for these targets; RMSE is supplementary.
- **Linear models (OLS, Ridge) are most vulnerable** — a single TSS=1266 row can shift coefficients significantly. Tree models (RF, XGB) are less affected (splits are ordinal). ElNet L1 penalty provides partial robustness.

**Implication for future phases:** Before training, consider winsorizing features at training P99 for linear models. For target outliers, log1p (already in Phase 11) is the preferred treatment over removal. Any new phase reporting RMSE on TSS/BOD should also report MdAE.
