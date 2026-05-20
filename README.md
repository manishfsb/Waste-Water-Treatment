# Wastewater Effluent Quality Prediction

Machine learning models to predict effluent water quality at a municipal wastewater treatment plant from upstream process measurements.

## What this is

Daily process records (2021-2025, ~1,900 rows, 60 columns) are used to train models that map
same-day operational inputs to effluent concentrations. Eight targets are modelled: BOD, COD,
TSS, and pH in both grab-sample and composite-sample forms. The 2025 calendar year is held out
as a blind test set.

Six model classes are evaluated across every experiment: OLS, Ridge, ElasticNet, Random Forest,
Gradient Boosting, and XGBoost. Hyperparameters are tuned with `TimeSeriesSplit(n_splits=3)`.

## Experiment structure

Nine systematic experiments explore what actually drives predictive performance:

| Experiment | Focus |
|---|---|
| Exp 1 | Inlet-only features (BOD/COD/TSS/pH inlet, with and without calendar and extended inlet columns) |
| Exp 2 | Process-stage ablation - primary, secondary, and combined feature scopes |
| Exp 3 | Extended operational features (aeration, SVI, DO, coliform) with tier-based feature selection |
| Exp 4 | VIF-pruned feature set for collinearity-safe linear modelling |
| Exp 5 | Cross-type inlet hypothesis - grab inlet in composite models and vice versa |
| Exp 6 | Advanced methods - ANN, Voting ensemble, Stacking ensemble |
| Exp 7 | Feature engineering - log transforms, interaction terms |
| Exp 8 | Temporal features - lag-1/3/7 and 7-day rolling means; same-day operational baseline |
| Exp 9 | Recency hypothesis - 2024-only training window with log-feature and log-target variants |

## Key findings

- **Inlet alone is not predictive.** Inlet BOD/COD/TSS give negative R² on the test set.
  Secondary clarifier data is necessary for positive performance.
- **Best regularised linear models** reach Grab BOD/COD/TSS R² of 0.42-0.58.
- **2024-only training (Exp 9)** produced the largest single jump: Grab BOD OLS improved from
  -0.209 (full window) to +0.616, indicating non-stationarity in plant operation.
- **Voting ensembles** improved tree performance; Stacking and ANN overfitted on composite data.
- **Lag features** (Exp 8) helped linear models on BOD; trees showed no consistent gain.
- **Operationally:** Grab BOD RMSE ~2 mg/L (20% of discharge limit) is plausibly useful.
  Grab TSS RMSE ~6 mg/L (60% of limit) should be treated as directional only.

## Limitations

- **Not a forecast model.** All models map same-day inputs to same-day effluent. No future
  prediction is implied.
- **Sample timing is unclear.** BOD/COD lab results may carry a ~5-day lag relative to the
  collection date. The exact workflow has not been confirmed with plant operators.
- **Feature availability is unknown.** It is unclear which inputs are continuous online sensors
  vs. periodic grab samples. A feature-by-feature audit with operators is needed before any
  deployment discussion.
- **Non-stationarity.** The 2024-only training result implies the plant operating regime shifted
  materially. Any operational use would require periodic retraining.

These models are best interpreted as retrospective process diagnostics and research benchmarks
rather than ready-to-deploy tools.

## Repository layout

```
raw_data/               Master dataset (All_Years_Full.xlsx) and per-year source files
eda/                    Exploratory data analysis scripts and HTML report
data_prep/              Data extraction, merging, and one-off fix utilities
modeling/
  datasets/             Per-experiment training datasets (xlsx, organised by experiment/)
  models/               Training scripts and saved results (linear/, non_linear/, phase9/, ...)
  feature_analysis/     Feature selection and audit scripts
  scripts/              Report generation (unified report, model selection, error decomposition)
  reports/              Generated HTML and xlsx reports
dashboard/              Interactive dashboard (separate from main report)
```

## Running the report

```bash
.venv/bin/python3 modeling/scripts/generate_unified_report.py
# output: modeling/reports/unified_report.html
```

## Environment

Python 3 via `.venv/`. All scripts must be invoked with `.venv/bin/python3`.
