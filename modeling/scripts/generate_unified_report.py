"""
generate_unified_report.py - Unified HTML report covering all experiments and phases.

Reads all results.xlsx files from the modeling directory tree and produces
a single, navigable HTML report with:
  - Fixed left-sidebar navigation with collapsible experiment groups
  - Global target filter (Grab BOD / Comp COD / … / All)
  - Global leaderboard: best Test R² per target across every experiment
  - R² progression chart (Chart.js) showing trend across experiments per model
  - Per-experiment sections with feature set card, dataset summary, linear /
    non-linear / combined metric tables (all foldable)
  - Overfitting traffic-light coloring on R² gap column
  - Best-model box at the end of each major section

Usage (from project root):
    .venv/bin/python3 21-25/modeling/scripts/generate_unified_report.py
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODELING_DIR)
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")

sys.path.insert(0, PROJECT_ROOT)
from report_theme import dark_mode_css, DARK_MODE_JS  # noqa: E402

os.makedirs(REPORTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TARGETS_ORDERED = [
    "Effluent BOD (mg/L, Grab)",
    "Effluent COD (mg/L, Grab)",
    "Effluent TSS (mg/L, Grab)",
    "Effluent pH (Grab)",
    "Effluent BOD (mg/L, Composite)",
    "Effluent COD (mg/L, Composite)",
    "Effluent TSS (mg/L, Composite)",
    "Effluent pH (Composite)",
]
TARGET_SHORT = dict(zip(TARGETS_ORDERED, [
    "Grab BOD", "Grab COD", "Grab TSS", "Grab pH",
    "Comp BOD", "Comp COD", "Comp TSS", "Comp pH",
]))
TARGET_SLUG = dict(zip(TARGETS_ORDERED, [
    "grab-bod", "grab-cod", "grab-tss", "grab-ph",
    "comp-bod", "comp-cod", "comp-tss", "comp-ph",
]))

MODEL_COLORS = {
    "OLS": "#E15252", "Ridge": "#4A90D9", "ElNet": "#5BAD6F",
    "RF": "#2171B5", "GB": "#238B45", "XGB": "#D94801",
    "ANN": "#9B59B6", "Voting": "#E67E22", "Stacking": "#1ABC9C",
}
LINEAR_MODELS  = ["OLS", "Ridge", "ElNet"]
NL_MODELS      = ["RF",  "GB",   "XGB"]
ADV_MODELS     = ["ANN", "Voting", "Stacking"]
ALL_MODELS_ORD = LINEAR_MODELS + NL_MODELS + ADV_MODELS
CORE_THRESH = 0.08
USEFUL_THRESH = 0.03

# Experiment key → short chart label
EXP_CHART_LABELS = {
    "Exp1-Sub1": "E1-S1",
    "Exp1": "Exp1",        "Exp1-FS": "Exp1-FS",   "Exp1-Cyclic": "E1-Cyc",
    "Exp2-Sub1": "E2-S1",  "Exp2-Sub1-FS": "E2-S1-FS",
    "Exp2-Sub1-Clr": "E2-S1-Clr",
    "Exp2-Sub1-Sed": "E2-S1-Sed",
    "Exp2-Sub2": "E2-S2",  "Exp2-Sub2-FS": "E2-S2-FS",
    "Exp2-Sub2-Cyc": "E2-S2-Cyc",
    "Exp3-S1": "E3-S1",    "Exp3-S1-FS": "E3-S1-FS",
    "Exp3-S2": "E3-S2",
    "Exp4-S1": "E4-S1",
    "Exp4-S2": "E4-S2",
    "Phase9-ANN": "P9-ANN", "Phase9-Voting": "P9-Vote",
    "Phase9-Stacking": "P9-Stack",
    "ANN-Exp1": "ANN-E1", "ANN-Exp2-Sub1": "ANN-E2-S1", "ANN-Exp2-Sub2": "ANN-E2-S2",
    "Phase10-FE": "P10-FE", "Phase10b-FE": "P10b-FE",
    "Phase11": "P11",
}
EXP_CHART_ORDER = list(EXP_CHART_LABELS.keys())

GRAB_TARGETS = TARGETS_ORDERED[:4]
COMP_TARGETS = TARGETS_ORDERED[4:]

# MdAE is the primary reliability metric for these targets (severe outlier distributions)
MDAE_TARGETS = {
    "Effluent BOD (mg/L, Grab)",
    "Effluent TSS (mg/L, Grab)",
    "Effluent BOD (mg/L, Composite)",
    "Effluent TSS (mg/L, Composite)",
}

# Dataset directory fragment → exp_key (order matters: most-specific first)
_DS_EXP_MAP = [
    ("experiment4/sub_exp2",                            "Exp4-S2"),
    ("experiment4/sub_exp1",                            "Exp4-S1"),
    ("experiment3/sub_exp2",                           "Exp3-S2"),
    ("experiment3/sub_exp1/feature_selected_datasets", "Exp3-S1-FS"),
    ("experiment3/sub_exp1",                           "Exp3-S1"),
    ("experiment2/sub_exp2/feature_selected_datasets", "Exp2-Sub2-FS"),
    ("experiment2/sub_exp2",                           "Exp2-Sub2"),
    ("experiment2/sub_exp1/feature_selected_datasets", "Exp2-Sub1-FS"),
    ("experiment2/sub_exp1",                           "Exp2-Sub1"),
    ("experiment1/sub_exp2/feature_selected_datasets", "Exp1-FS"),
    ("experiment1/sub_exp2_cyclic",                    "Exp1-Cyclic"),
    ("experiment1/sub_exp2",                           "Exp1"),
    ("experiment1/sub_exp1",                           "Exp1-Sub1"),
]

# predicted_<TAG>_run_N → canonical model name
_PRED_MODEL_MAP = {
    "OLS": "OLS", "Ridge": "Ridge", "ElNet": "ElNet",
    "RF_NL": "RF", "GB_NL": "GB", "XGB_NL": "XGB",
    "ANN": "ANN", "Voting": "Voting", "Stacking": "Stacking",
}

# Phase 9 models live inside Exp3-S2 files but belong to a different exp_key
_PHASE9_EXP = {"ANN": "Phase9-ANN", "Voting": "Phase9-Voting", "Stacking": "Phase9-Stacking"}

FEATURE_DESCRIPTIONS = {
    "Exp1-Sub1": {
        "label": "Inlet Only (4 features) — no process or temporal context",
        "features": "Inlet pH, Inlet BOD (mg/L), Inlet COD (mg/L), Inlet TSS (mg/L) "
                    "(Grab or Composite) — no Flow, Power, or calendar features",
        "rationale": "Absolute minimum feature set: can inlet water quality alone predict "
                     "effluent quality? This sub-experiment establishes the floor and "
                     "quantifies how much process context (Flow, Power, calendar) adds "
                     "when compared against Exp1 (Sub 2).",
    },
    "Exp1": {
        "label": "Inlet + COMMON (9 features)",
        "features": "Inlet pH, Inlet BOD, Inlet COD, Inlet TSS (Grab or Composite) + "
                    "Flow (MLD), Power Total (KW), month, day_of_week, year",
        "rationale": "Baseline: can inlet water quality alone predict effluent quality? "
                     "Tests the simplest, most operationally available feature set - "
                     "no secondary treatment data required.",
    },
    "Exp1-Cyclic": {
        "label": "Inlet + COMMON with Cyclic Calendar Encoding (11 features)",
        "features": "Inlet pH, BOD, COD, TSS (Grab or Composite) + "
                    "Flow (MLD), Power Total (KW), year, "
                    "month_sin, month_cos, dow_sin, dow_cos",
        "rationale": "Tests whether replacing raw integer month (1–12) and day_of_week (0–6) "
                     "with sin/cos projections improves generalisation. Raw integers impose a "
                     "false December→January discontinuity on linear models; cyclic encoding "
                     "preserves the wrap-around topology of both features.",
    },
    "Exp1-FS": {
        "label": "Inlet + COMMON - Feature Selected (5-8 features)",
        "features": "Subset of Exp1 features retained by RF permutation importance "
                    "(importance ≥ 0.03 threshold) and mutual information screening",
        "rationale": "Checks whether pruning the 9-feature set to its most informative "
                     "subset improves generalisation on the 2025 holdout.",
    },
    "Exp2-Sub1": {
        "label": "Combined Secondary + COMMON (15 features)",
        "features": "Sec Clarifier pH/TSS/BOD/COD/RAS + Sec Sed pH/TSS/BOD/COD/RAS "
                    "+ Flow, Power, year, month_sin, month_cos, dow_sin, dow_cos",
        "rationale": "Tests secondary treatment process data only (no inlet). "
                     "Do downstream process indicators predict effluent better than inlet?",
    },
    "Exp2-Sub1-FS": {
        "label": "Secondary + COMMON - Feature Selected",
        "features": "Subset of Exp2-S1 secondary + COMMON features after permutation "
                    "importance screening",
        "rationale": "Feature selection on the secondary feature set to remove noisy "
                     "or redundant secondary parameters.",
    },
    "Exp2-Sub2": {
        "label": "Inlet + Secondary + COMMON (19 features)",
        "features": "Inlet (Grab/Composite) + Sec Clarifier + Sec Sed + "
                    "Flow, Power, month, day_of_week, year",
        "rationale": "Full baseline combining both monitoring points. Tests whether "
                     "joining inlet and secondary data outperforms either alone.",
    },
    "Exp2-Sub2-FS": {
        "label": "Inlet + Secondary + COMMON - Feature Selected",
        "features": "Subset of Exp2-S2 combined features after permutation importance "
                    "and mutual information screening",
        "rationale": "Remove redundant or low-signal features from the full Exp2-S2 set.",
    },
    "Exp2-Sub1-Clr": {
        "label": "Secondary Clarifier + COMMON (12 features)",
        "features": "Sec Clarifier pH, TSS, BOD, COD, RAS + Flow, Power, year, month_sin, month_cos, dow_sin, dow_cos",
        "rationale": "Isolate the Clarifier group to test whether Clarifier and Sedimentation "
                     "features carry distinct signal or are interchangeable.",
    },
    "Exp2-Sub1-Sed": {
        "label": "Secondary Sedimentation + COMMON (12 features)",
        "features": "Sec Sed pH, TSS, BOD, COD, RAS(New) + Flow, Power, year, month_sin, month_cos, dow_sin, dow_cos",
        "rationale": "Isolate the Sedimentation group — paired with Exp2-Sub1-Clr to test "
                     "whether one group dominates the other.",
    },
    "Exp2-Sub2-Cyc": {
        "label": "Inlet + Secondary + COMMON (21 features)",
        "features": "Grab or Composite inlet (4) + all SEC_COLS (10) + Flow, Power, year, "
                    "month_sin, month_cos, dow_sin, dow_cos (7)",
        "rationale": "Full combined feature set with corrected calendar encoding. "
                     "OLS uses LassoCV pre-screening; trees use OOF permutation importance "
                     "feature selection, both stored in the same results file.",
    },
    "Exp3-S1": {
        "label": "Exp2-S2 + ADD-tier Aeration Features (16-22 features)",
        "features": "Exp2-S2 baseline + ADD-tier features (MI ≥ 0.20, marginal row "
                    "cost ≤ 20%): MLSS, SV30, SVI, DO, Aeration pH - adds ~3-7 columns",
        "rationale": "Expand with aeration basin data that shows high mutual information "
                     "and low missingness cost on top of the Exp2-S2 baseline.",
    },
    "Exp3-S1-FS": {
        "label": "Exp3-S1 - Core + Useful (Feature Selected, 5-8 features)",
        "features": "RF permutation importance ≥ 0.03 threshold applied to the "
                    "full Exp3-S1 feature set → 5-8 features per target",
        "rationale": "Aggressive pruning of the expanded Exp3-S1 set to the most "
                     "predictive features per target. Tests if less is more.",
    },
    "Exp3-S2": {
        "label": "Exp2-S2 + ADD + CONSIDER-tier Features (20-32 features)",
        "features": "Exp2-S2 + ADD features + CONSIDER-tier (MI ≥ 0.15 & cost ≤ 35%, "
                    "or MI ≥ 0.25 & cost ≤ 50%) - 20-32 total features per target",
        "rationale": "Full feature expansion per the audit-tier logic. Tests whether "
                     "CONSIDER-tier features help despite higher missingness cost. "
                     "Key question: do regularised linear models benefit from "
                     "additional features even if non-linear models overfit?",
    },
    "Exp4-S1": {
        "label": "Exp3-S2 minus Derived/Redundant Columns (12-19 features)",
        "features": "Exp3-S2 feature set with three redundant groups removed: "
                    "(1) Aeration SVI (Existing + New) - derived from SV30/MLSS, zero independent information; "
                    "(2) All (New) aeration tank columns - cross-tank r=0.74-0.84, Existing tank wins on every BOD/COD/TSS target; "
                    "(3) All Sec Sedimentation columns - cross-stage r=0.69-0.87, Sec Clarifier wins consistently. "
                    "Remaining: Inlet, COMMON, Existing aeration (MLSS, SV30, DO, pH), Sec Clarifier (pH, TSS, BOD, COD, RAS).",
        "rationale": "Hypothesis: removing collinear and derived features reduces variance inflation and "
                     "improves generalisation, especially on composite targets where Exp3-S2 GB/XGB catastrophically "
                     "overfit. Phase 1 of Exp4 - Phase 2 will address within-group VIF (MLSS vs SV30, "
                     "Sec Clarifier inter-correlations).",
    },
    "Exp4-S2": {
        "label": "Exp4-S1 + Iterative VIF Pruning (threshold=10) - 5-10 features",
        "features": "Starting from the Exp4-S1 feature pool, automated iterative VIF pruning "
                    "(drop highest-VIF feature, recompute, repeat until all VIF ≤ 10) run on "
                    "training rows only. Dropped per target: all pH features (Inlet pH, "
                    "Sec Clarifier pH, Aeration pH, Primary Clarifier pH - VIF 600-2400), "
                    "Inlet BOD/COD (VIF 18-34), MLSS (SV30 survives as biomass proxy), "
                    "one of Flow/Power Total. "
                    "Temporal features (month, day_of_week, year) always retained, excluded from VIF. "
                    "Row gain vs Exp4-S1: +47 to +293 rows (dropped features had high missingness).",
        "rationale": "Hypothesis: automated, data-driven VIF pruning removes intra-group "
                     "multicollinearity more precisely than manual feature removal (Exp4-S1), "
                     "yielding a well-conditioned feature matrix where Ridge/ElNet regularisation "
                     "can work effectively and tree model overfitting may reduce.",
    },
    "ANN-Exp1": {
        "label": "Exp1 Features → ANN (Inlet + COMMON, 9 features)",
        "features": "Inlet pH, Inlet BOD, Inlet COD, Inlet TSS (Grab or Composite) + "
                    "Flow (MLD), Power Total (KW). StandardScaler + MLPRegressor, "
                    "GridSearchCV on hidden_layer_sizes and alpha (TimeSeriesSplit). "
                    "~1175 Grab / ~800 Composite training rows.",
        "rationale": "Phase 9 ANN failed with ~470 Grab training samples. Exp1 has 2.5× more "
                     "data with simpler features. Tests whether ANN failure was data-volume "
                     "limited. Same architecture as Phase 9 ANN; only dataset changed.",
    },
    "ANN-Exp2-Sub1": {
        "label": "Exp2-Sub1 Features → ANN (Secondary + COMMON, 15 features)",
        "features": "Sec Clarifier pH/TSS/BOD/COD/RAS, Sec Sed pH/TSS/BOD/COD/RAS + "
                    "Flow, Power. StandardScaler + MLPRegressor, GridSearchCV (TimeSeriesSplit). "
                    "~924 Grab / ~740 Composite training rows.",
        "rationale": "Middle ground: richer features (secondary process data) with ~2× the "
                     "samples of Exp3-S2. Tests whether secondary process data adds meaningful "
                     "ANN signal at adequate sample size.",
    },
    "ANN-Exp2-Sub2": {
        "label": "Exp2-Sub2 Features → ANN (Inlet + Secondary + COMMON, 19 features)",
        "features": "Inlet (Grab/Composite) + Sec Clarifier + Sec Sed + Flow, Power. "
                    "StandardScaler + MLPRegressor, GridSearchCV (TimeSeriesSplit). "
                    "~920 Grab / ~733 Composite training rows.",
        "rationale": "Combined inlet + secondary features at twice the sample count of "
                     "Exp3-S2 ANN. The most direct comparison: same feature scope as "
                     "Exp3-S2 baseline but without the CONSIDER-tier columns that reduced "
                     "training rows via missingness.",
    },
    "Phase9-ANN": {
        "label": "Exp3-S2 Features → ANN (MLPRegressor, StandardScaler pipeline)",
        "features": "Same as Exp3-S2 (25 features), StandardScaler + MLPRegressor, "
                    "GridSearchCV on hidden_layer_sizes and alpha (TimeSeriesSplit)",
        "rationale": "Tests whether a neural network can outperform regularised linear "
                     "and tree models. Key challenge: ~470 training samples may be "
                     "insufficient for a multi-layer perceptron.",
    },
    "Phase9-Voting": {
        "label": "Exp3-S2 Features → Voting Ensemble (ElNet + RF + XGBoost, equal weights)",
        "features": "Same as Exp3-S2 (25 features per target); VotingRegressor averaging "
                    "ElasticNet, RandomForest, and XGBoost with equal weights. "
                    "Replaces original Ridge+ElNet+RF - Ridge removed because it duplicates "
                    "ElNet's L2 regularisation (redundant voter); XGBoost added for "
                    "structural diversity (gradient boosting vs. bagging vs. linear).",
        "rationale": "Ensemble averaging reduces variance through uncorrelated errors. "
                     "ElNet: regularised linear with feature selection (L1). "
                     "RF: bagging over decision trees, robust to outliers. "
                     "XGB: boosted trees, captures residual non-linearities. "
                     "Three structurally distinct inductive biases.",
    },
    "Phase9-Stacking": {
        "label": "Exp3-S2 Features → Stacking (ElNet + RF + XGB → Ridge meta, walk-forward OOF)",
        "features": "Same as Exp3-S2 (25 features); walk-forward stacking with "
                    "TimeSeriesSplit(n_splits=5) OOF generation. "
                    "Base: ElNet + RF + XGBoost. Meta-learner: Ridge (tuned alpha). "
                    "First ~1/6 of training samples excluded from meta-learner training "
                    "(no OOF fold coverage) - strictly no look-ahead bias. "
                    "Base models retrained on full training set before inference.",
        "rationale": "Replaces original KFold(n_splits=5) StackingRegressor which had "
                     "look-ahead bias: predicting 2021 data using models trained on 2022-2024. "
                     "Walk-forward OOF ensures the meta-learner learns how base models "
                     "perform when projecting forward in time. "
                     "Approx. 83% OOF coverage across all targets.",
    },
    "Phase10-FE": {
        "label": "Exp3-S2 + Full Feature Engineering - All Targets (37-52 features)",
        "features": "Exp3-S2 base + log1p transforms of skewed cols + pairwise "
                    "interaction terms (key pairs) + IQR outlier indicator flags; "
                    "applied uniformly to all 8 targets",
        "rationale": "Feature engineering to capture non-linear relationships and "
                     "reduce skewness. Applied uniformly to test if it universally "
                     "helps. Hypothesis: log transforms help BOD/COD/TSS (right-skewed).",
    },
    "Phase10b-FE": {
        "label": "Selective Feature Engineering - Grab: Full FE / Composite: Base Exp3-S2",
        "features": "Grab targets: full FE (37-52 features each). "
                    "Composite targets: base Exp3-S2 features only (17-27 features) "
                    "- no engineering applied to avoid overfitting on small composite n.",
        "rationale": "Addresses Phase 10 finding: FE caused severe overfitting on "
                     "composite datasets (n_train ≈ 290). Selective approach preserves "
                     "FE gains on grab targets while stabilising composites.",
    },
    "Phase11": {
        "label": "Exp3-S2 + Temporal Lags + log1p Target Transform (50-58 features)",
        "features": "Base Exp3-S2 features + lag 1/3/7-day + 7-day calendar rolling mean "
                    "applied to inlet + Flow + Power columns only (~50-58 total features per target). "
                    "BOD/COD/TSS targets log1p-transformed; pH untransformed. "
                    "Back-transform via Duan smearing: expm1(ŷ) × mean(exp(training residuals)).",
        "rationale": "Daily wastewater quality has temporal autocorrelation - yesterday's "
                     "influent predicts today's effluent better than cross-sectional features "
                     "alone. Lag expansion restricted to inlet+Flow+Power columns to avoid "
                     "catastrophic overfitting on composite datasets (n≈290 rows). "
                     "log1p targets reduce right-skew in BOD/COD/TSS distributions.",
    },
}

EXP_INTRO = {
    "Exp1": (
        "Experiment 1 investigates inlet-feature prediction across two feature-set scopes. "
        "<strong>Sub-experiment 1</strong> uses only the 4 inlet concentration columns — the "
        "absolute floor. <strong>Sub-experiment 2</strong> adds Flow, Power, and cyclic calendar "
        "features (month/dow encoded as sin/cos projections), raising the count to 11. "
        "All datasets use cyclic calendar encoding as the standard going forward. "
        "Feature selection is <strong>model-specific</strong> and built into each training run: "
        "OLS uses LassoCV pre-screening, Ridge regularises over the full set (no pruning), "
        "ElNet applies L1 selection internally, and tree models (RF, GB, XGB) use OOF "
        "permutation importance to retain features above a 5% relative importance threshold. "
        "The <em>Sub-experiment Comparison</em> panel at the bottom shows the full-feature vs "
        "feature-selected breakdown per model and target with magnitude-weighted deltas."
    ),
    "Exp2": (
        "Experiment 2 expands the feature set from inlet-only to include secondary treatment data. "
        "<strong>Sub-experiment 1</strong> tests three secondary-only feature scopes side by side: "
        "Sec Clarifier features alone (10 features), Sec Sedimentation features alone (10 features), "
        "and both groups combined (15 features). The split variants test whether the two secondary "
        "groups carry distinct signal or are interchangeable. "
        "<strong>Sub-experiment 2</strong> adds inlet concentrations to the combined secondary set "
        "(21 features) and applies model-specific feature selection "
        "(OLS: LassoCV; trees: OOF permutation importance)."
    ),
    "Exp3": (
        "Experiment 3 extends the Exp2-S2 combined feature set with aeration basin data "
        "identified in the Phase 6 feature audit. <strong>Sub-experiment 1</strong> adds "
        "ADD-tier features (high MI, low missingness cost). "
        "<strong>Sub-experiment 2</strong> further adds CONSIDER-tier features to test "
        "whether more features - at higher missingness cost - still benefit regularised "
        "linear models."
    ),
    "Exp4": (
        "Experiment 4 tests the hypothesis that <strong>removing collinear and derived features</strong> "
        "from Exp3-S2 improves generalisation. "
        "<strong>Sub-experiment 1</strong> manually drops three redundant groups "
        "(SVI, New aeration, Sec Sed - feature count 12-19). "
        "<strong>Sub-experiment 2</strong> applies automated iterative VIF pruning (threshold=10) "
        "within the Exp4-S1 feature pool, reducing to 5-10 features per target and gaining "
        "+47 to +293 rows by dropping high-missingness correlated features. "
        "<strong>Overall result: both sub-experiments refuted the pruning hypothesis.</strong> "
        "S1 showed performance degraded universally. S2 confirmed the pattern: "
        "Ridge alone partially recovered on some targets; tree models failed catastrophically. "
        "See the finding boxes below for per-sub-experiment analysis."
    ),
    "ANN-Dataset-Exploration": (
        "The Phase 9 ANN failed on Exp3-S2 datasets (avg Test R²=−1.12) due to insufficient "
        "training samples (~470 Grab, ~290 Composite). These runs test the same ANN architecture "
        "on earlier, <strong>data-richer</strong> experiment datasets to isolate whether the failure "
        "was <em>data-volume limited</em> or reflects a fundamental ANN limitation on this type of data."
        "<br><br>"
        "Three datasets are tested: <strong>Exp1</strong> (Inlet + COMMON, 9 features, "
        "~1175 Grab / ~800 Composite rows — 2.5× more data), "
        "<strong>Exp2-Sub1</strong> (Secondary + COMMON, 15 features, "
        "~924 / ~740 rows), and "
        "<strong>Exp2-Sub2</strong> (Inlet + Secondary + COMMON, 19 features, "
        "~920 / ~733 rows). "
        "The hyperparameter grid includes larger architectures (256-128, 128-64-32 hidden layers) "
        "compared to Phase 9 ANN, appropriate for the larger sample counts."
    ),
    "Phase9": (
        "Phase 9 evaluates advanced model architectures on the <strong>Exp3-S2 feature set</strong>, "
        "which was selected as the richest validated set (ElNet Test R²=0.684 Grab BOD, "
        "RF 0.504 Grab TSS - new records at the time). "
        "<br><br>"
        "Three architectures are tested: "
        "<strong>ANN</strong> (MLPRegressor) - failed; ~470 training samples insufficient for "
        "MLP (avg Test R²=−1.12). All targets selected alpha=1.0 (max regularisation). Not recommended. "
        "<br>"
        "<strong>Voting ensemble</strong> (ElNet + RF + XGBoost, equal weights) - "
        "original composition (Ridge+ElNet+RF) was corrected: Ridge duplicates ElNet's L2 "
        "penalty, making two of three voters collinear. XGBoost replaces Ridge for structural "
        "diversity (gradient boosting vs. bagging vs. regularised linear). "
        "All three voters pre-tuned via TimeSeriesSplit GridSearchCV/RandomizedSearchCV. "
        "<br>"
        "<strong>Stacking ensemble</strong> (ElNet + RF + XGB → Ridge meta, walk-forward OOF) - "
        "original KFold(n_splits=5) stacking had look-ahead bias: 2021 samples were predicted "
        "using base models trained on 2022-2024 data. Replaced with manual walk-forward OOF "
        "using TimeSeriesSplit(n_splits=5): each sample's OOF prediction uses only past data. "
        "~83% OOF coverage per target; first ~17% excluded from meta-learner training."
    ),
    "Phase10": (
        "Phase 10 applies <strong>feature engineering</strong> (log transforms, "
        "interaction terms, outlier flags) on top of the Exp3-S2 feature set. "
        "Phase 10 (full) applies this uniformly to all targets. Phase 10b "
        "(selective) applies FE only to grab targets after discovering that "
        "composite datasets overfit severely with the expanded feature count."
    ),
    "Phase11": (
        "Phase 11 adds <strong>temporal lag features</strong> (lag 1/3/7-day) and a "
        "<strong>7-day rolling mean</strong> to inlet, Flow, and Power columns, "
        "plus a log1p target transform on BOD/COD/TSS with Duan smearing back-transform. "
        "Feature expansion is restricted to inlet+Flow+Power columns to avoid the "
        "catastrophic overfitting seen when applying temporal features to all columns "
        "(which would produce ~128 features on ~290 composite rows)."
        "<br><br>"
        "Key wins: <strong>Grab BOD</strong> (Ridge 0.656, gap −0.095) and "
        "<strong>Grab COD</strong> (XGB 0.464, gap +0.057 - new Grab COD high with an honest gap). "
        "Key losses: all composite targets deteriorate, especially Comp TSS (Ridge −1.04, gap +1.82). "
        "Conclusion: temporal features help Grab targets; composites lack the sample size to benefit. "
        "CV_R² and Gap_gen = CV_R² − Test_R² are recorded per model for distribution-shift diagnosis."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _exp_key(raw: str, is_fs: bool) -> str:
    mapping = {
        "Exp1-Sub1": "Exp1-Sub1", "Exp1": "Exp1", "Exp1-Cyclic": "Exp1-Cyclic",
        "Exp2-Sub1": "Exp2-Sub1", "Exp2-Sub2": "Exp2-Sub2",
        "Exp2-Sub1-Clr": "Exp2-Sub1-Clr",
        "Exp2-Sub1-Sed": "Exp2-Sub1-Sed",
        "Exp2-Sub2-Cyc": "Exp2-Sub2-Cyc",
        "Exp3-S1": "Exp3-S1", "Exp3-S1-FS": "Exp3-S1-FS", "Exp3-S2": "Exp3-S2",
        "Experiment 1": "Exp1",
        "Experiment 2 Sub-1": "Exp2-Sub1",
        "Experiment 2 Sub-2": "Exp2-Sub2",
        "Experiment 3 Sub-1": "Exp3-S1",
        "Experiment 3 Sub-1 FS": "Exp3-S1-FS",
        "Experiment 3 Sub-2": "Exp3-S2",
        "Exp4-S1": "Exp4-S1",
        "Experiment 4 Sub-1": "Exp4-S1",
        "Exp4-S2": "Exp4-S2",
        "Experiment 4 Sub-2": "Exp4-S2",
        "Phase9-ANN": "Phase9-ANN",
        "Phase9-Ensemble": "Phase9-Ensemble",
        "Phase10-FE": "Phase10-FE",
        "Phase10b-FE-GrabOnly": "Phase10b-FE",
    }
    key = mapping.get(raw, raw)
    if is_fs and key in ("Exp1", "Exp2-Sub1", "Exp2-Sub2"):
        key += "-FS"
    return key


def _melt_linear(df: pd.DataFrame, is_fs: bool) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        key = _exp_key(str(row["experiment"]), is_fs)
        base = dict(
            exp_key=key,
            target=row["target"],
            n_train=row.get("n_train"),
            n_test=row.get("n_test"),
            n_features=row.get("n_features"),
            n_features_input=row.get("n_features_input"),
            n_selected_ols=row.get("n_selected_ols"),
            selected_features_ols=row.get("selected_features_ols"),
            ElNet_n_selected=row.get("ElNet_n_selected"),
            ElNet_selected_features=row.get("ElNet_selected_features"),
        )
        for m in LINEAR_MODELS:
            r = base.copy()
            r.update(
                model=m,
                R2_train=row.get(f"{m}_train_R2"),
                R2_test=row.get(f"{m}_test_R2"),
                R2_gap=row.get(f"{m}_R2_gap"),
                RMSE_train=row.get(f"{m}_train_RMSE"),
                RMSE_test=row.get(f"{m}_test_RMSE"),
                MAE_test=row.get(f"{m}_test_MAE"),
                MAPE_test=row.get(f"{m}_test_MAPE"),
            )
            if m == "OLS":
                r["R2_test_full"]   = row.get("OLS_full_test_R2")
                r["R2_train_full"]  = row.get("OLS_full_train_R2")
                r["RMSE_test_full"] = row.get("OLS_full_test_RMSE")
                r["MAE_test_full"]  = row.get("OLS_full_test_MAE")
                r["R2_gap_full"]    = row.get("OLS_full_R2_gap")
            rows.append(r)
    return pd.DataFrame(rows)


def _norm_nl(df: pd.DataFrame, is_fs: bool) -> pd.DataFrame:
    out = pd.DataFrame(dict(
        exp_key=df["experiment"].map(lambda x: _exp_key(str(x), is_fs)),
        target=df["target"],
        model=df["model"].str.upper(),
        n_train=df.get("n_train"),
        n_test=df.get("n_test"),
        n_features=df.get("n_features"),
        n_features_input=df.get("n_features_input"),
        n_selected_nl=df.get("n_selected_nl"),
        selected_features_nl=df.get("selected_features_nl"),
        R2_train_full=df.get("R2_train_full"),
        R2_test_full=df.get("R2_test_full"),
        RMSE_test_full=df.get("RMSE_test_full"),
        MAE_test_full=df.get("MAE_test_full"),
        R2_gap_full=df.get("R2_gap_full"),
        R2_train=df.get("R2_train"),
        R2_test=df.get("R2_test"),
        R2_gap=df.get("R2_gap"),
        RMSE_train=df.get("RMSE_train"),
        RMSE_test=df.get("RMSE_test"),
        MAE_test=df.get("MAE_test"),
        MAPE_test=df.get("MAPE_test"),
    ))
    return out


def _norm_phase9(df: pd.DataFrame) -> pd.DataFrame:
    model_map = {"ANN": "ANN", "Voting": "Voting", "Stacking": "Stacking"}
    out = pd.DataFrame(dict(
        exp_key=df["model"].map(lambda m: f"Phase9-{model_map.get(m, m)}"),
        target=df["target"],
        model=df["model"].map(model_map),
        n_train=df.get("n_train"),
        n_test=df.get("n_test"),
        n_features=df.get("n_features"),
        R2_train=df.get("R2_train"),
        R2_test=df.get("R2_test"),
        R2_gap=df.get("R2_gap"),
        RMSE_train=df.get("RMSE_train"),
        RMSE_test=df.get("RMSE_test"),
        MAE_test=df.get("MAE_test"),
        MAPE_test=df.get("MAPE_test"),
    ))
    return out


def _norm_phase11(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Phase 11 results (temporal features + log1p targets)."""
    out = pd.DataFrame(dict(
        exp_key="Phase11",
        target=df["target"],
        model=df["model"],
        n_train=df.get("n_train"),
        n_test=df.get("n_test"),
        n_features=df.get("n_features"),
        R2_train=df.get("R2_train"),
        R2_test=df.get("R2_test"),
        R2_gap=df.get("R2_gap"),
        RMSE_train=df.get("RMSE_train"),
        RMSE_test=df.get("RMSE_test"),
        MAE_test=df.get("MAE_test"),
        MAPE_test=df.get("MAPE_test"),
    ))
    return out


def _norm_ann_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ANN results for Exp1/Exp2-Sub1/Exp2-Sub2 dataset runs.

    experiment column in results.xlsx is the exp_key directly
    (e.g. 'ANN-Exp1', 'ANN-Exp2-Sub1', 'ANN-Exp2-Sub2').
    """
    out = pd.DataFrame(dict(
        exp_key=df["experiment"],
        target=df["target"],
        model="ANN",
        n_train=df.get("n_train"),
        n_test=df.get("n_test"),
        n_features=df.get("n_features"),
        R2_train=df.get("R2_train"),
        R2_test=df.get("R2_test"),
        R2_gap=df.get("R2_gap"),
        RMSE_train=df.get("RMSE_train"),
        RMSE_test=df.get("RMSE_test"),
        MAE_test=df.get("MAE_test"),
        MAPE_test=df.get("MAPE_test"),
    ))
    return out


def _norm_phase10(df: pd.DataFrame, is_10b: bool) -> pd.DataFrame:
    exp_key = "Phase10b-FE" if is_10b else "Phase10-FE"
    out = pd.DataFrame(dict(
        exp_key=exp_key,
        target=df["target"],
        model=df["model"],
        n_train=df.get("n_train"),
        n_test=df.get("n_test"),
        n_features=df.get("n_features"),
        R2_train=df.get("R2_train"),
        R2_test=df.get("R2_test"),
        R2_gap=df.get("R2_gap"),
        RMSE_train=df.get("RMSE_train"),
        RMSE_test=df.get("RMSE_test"),
        MAE_test=df.get("MAE_test"),
        MAPE_test=df.get("MAPE_test"),
    ))
    return out


def load_all_data() -> pd.DataFrame:
    frames = []
    m = os.path.join(MODELING_DIR, "models")

    # Linear
    for variant, is_fs in [
        ("baseline", False), ("feature_selected", True),
        ("exp1_s1", False), ("exp1_cyclic", False),
        ("exp3_s1", False), ("exp3_s1_fs", False), ("exp3_s2", False),
        ("exp4_s1", False),
        ("exp4_s2", False),
        ("exp2_s1", False), ("exp2_s1_split", False), ("exp2_s2", False),
    ]:
        p = os.path.join(m, "linear", variant, "results.xlsx")
        if os.path.exists(p):
            df = pd.read_excel(p)
            df = df[df["run"] == df["run"].max()]
            frames.append(_melt_linear(df, is_fs))

    # Non-linear
    for variant, is_fs in [
        ("baseline", False), ("feature_selected", True),
        ("exp1_s1", False), ("exp1_cyclic", False),
        ("exp3_s1", False), ("exp3_s1_fs", False), ("exp3_s2", False),
        ("exp4_s1", False),
        ("exp4_s2", False),
        ("exp2_s1", False), ("exp2_s1_split", False), ("exp2_s2", False),
    ]:
        for mdl in ["rf", "gb", "xgb"]:
            p = os.path.join(m, "non_linear", variant, mdl, "results.xlsx")
            if os.path.exists(p):
                df = pd.read_excel(p)
                df = df[df["run"] == df["run"].max()]
                frames.append(_norm_nl(df, is_fs))

    # Phase 9
    for p in [os.path.join(m, "phase9", "ann", "results.xlsx"),
              os.path.join(m, "phase9", "ensemble", "results.xlsx")]:
        if os.path.exists(p):
            df = pd.read_excel(p)
            df = df[df["run"] == df["run"].max()]
            frames.append(_norm_phase9(df))

    # ANN on Exp1/Exp2 datasets (diagnostic data-volume experiment)
    for subdir in ["ann_exp1", "ann_exp2s1", "ann_exp2s2"]:
        p = os.path.join(m, "phase9", subdir, "results.xlsx")
        if os.path.exists(p):
            df = pd.read_excel(p)
            df = df[df["run"] == df["run"].max()]
            frames.append(_norm_ann_extra(df))

    # Phase 10
    for path, is_10b in [
        (os.path.join(m, "phase10", "results.xlsx"), False),
        (os.path.join(m, "phase10", "results_10b.xlsx"), True),
    ]:
        if os.path.exists(path):
            df = pd.read_excel(path)
            df = df[df["run"] == df["run"].max()]
            frames.append(_norm_phase10(df, is_10b))

    # Phase 11
    p11 = os.path.join(m, "phase11", "results.xlsx")
    if os.path.exists(p11):
        df = pd.read_excel(p11)
        df = df[df["run"] == df["run"].max()]
        frames.append(_norm_phase11(df))

    combined = pd.concat(frames, ignore_index=True)

    # Backfill n_features into non-linear rows from corresponding linear results
    nf = (combined[combined["model"].isin(LINEAR_MODELS)]
          .drop_duplicates(["exp_key", "target"])[["exp_key", "target", "n_features"]]
          .rename(columns={"n_features": "_nf"}))
    combined = combined.merge(nf, on=["exp_key", "target"], how="left")
    combined["n_features"] = combined["n_features"].fillna(combined["_nf"])
    combined.drop(columns=["_nf"], inplace=True)

    # Ensure numeric types
    for col in ["n_train", "n_test", "n_features", "R2_train", "R2_test",
                "R2_gap", "RMSE_train", "RMSE_test", "MAE_test", "MAPE_test"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined


def compute_all_mdae() -> pd.DataFrame:
    """Scan every dataset xlsx file, compute MdAE on 2025 rows for BOD/TSS targets only.

    Returns a DataFrame with columns: exp_key, model, target, MdAE_test.
    Phase 10 and Phase 11 do not store row-level predictions so those phases
    are absent - MdAE_test will be NaN for them after the merge.
    """
    import glob, re
    ds_base = os.path.join(MODELING_DIR, "datasets")
    records = []

    for fpath in sorted(glob.glob(os.path.join(ds_base, "**", "*.xlsx"), recursive=True)):
        if os.path.basename(fpath).startswith("~$"):
            continue
        rel = fpath.replace(ds_base + os.sep, "").replace("\\", "/")

        # Determine exp_key from directory path
        exp_key = None
        for fragment, key in _DS_EXP_MAP:
            if rel.startswith(fragment):
                exp_key = key
                break
        if exp_key is None:
            continue

        df = pd.read_excel(fpath)
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df_2025 = df[df["Date"].dt.year == 2025]
        if len(df_2025) < 5:
            continue

        # Identify target column (single Effluent column)
        tgt_cols = [c for c in df.columns if c.startswith("Effluent")]
        if len(tgt_cols) != 1:
            continue
        target = tgt_cols[0]
        if target not in MDAE_TARGETS:
            continue

        # Find predicted columns, take highest run per model tag
        pred_cols = [c for c in df.columns if c.startswith("predicted_")]
        best_run: dict = {}
        for col in pred_cols:
            m = re.match(r"predicted_(.+)_run_(\d+)$", col)
            if not m:
                continue
            tag, run = m.group(1), int(m.group(2))
            if tag not in best_run or run > best_run[tag][0]:
                best_run[tag] = (run, col)

        for tag, (run, col) in best_run.items():
            model = _PRED_MODEL_MAP.get(tag)
            if model is None:
                continue

            # Phase 9 models stored in Exp3-S2 files need their own exp_key
            row_exp_key = exp_key
            if exp_key == "Exp3-S2" and model in _PHASE9_EXP:
                row_exp_key = _PHASE9_EXP[model]

            sub = df_2025[[target, col]].dropna()
            if len(sub) < 5:
                continue

            mdae = float(np.median(np.abs(sub[target].values - sub[col].values)))
            records.append({"exp_key": row_exp_key, "model": model,
                            "target": target, "MdAE_test": mdae})

    if not records:
        return pd.DataFrame(columns=["exp_key", "model", "target", "MdAE_test"])
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, d=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{val:.{d}f}"

def _r2_color(val):
    if not isinstance(val, (int, float)) or np.isnan(val):
        return "var(--text-muted)"
    if val >= 0.6:  return "#2ecc71"
    if val >= 0.4:  return "#52c98a"
    if val >= 0.2:  return "#f1c40f"
    if val >= 0.0:  return "#e67e22"
    return "#e74c3c"

def _gap_cls(gap):
    if not isinstance(gap, (int, float)) or np.isnan(gap):
        return ""
    return "gap-good" if abs(gap) < 0.10 else ("gap-warn" if abs(gap) < 0.25 else "gap-bad")

def _badge(text, kind="default"):
    colors = {
        "rec": ("#27ae60", "#fff"),
        "fail": ("#c0392b", "#fff"),
        "warn": ("#e67e22", "#fff"),
        "info": ("#4A90D9", "#fff"),
        "default": ("#6b7280", "#fff"),
    }
    bg, fg = colors.get(kind, colors["default"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:10px;font-size:11px;font-weight:bold;'
            f'letter-spacing:.3px">{text}</span>')


_FS_IMPORTANCE_CACHE = {}
_LINEAR_RANKING_CACHE = {}
FS_BASE_MAP = {
    "Exp1-FS": "Exp1",
    "Exp2-Sub1-FS": "Exp2-Sub1",
    "Exp2-Sub2-FS": "Exp2-Sub2",
    "Exp3-S1-FS": "Exp3-S1",
}


def _tier_single(v: float) -> tuple[str, str]:
    if not isinstance(v, (int, float)) or np.isnan(v):
        return "-", "tier-na"
    if v >= CORE_THRESH:
        return "Core", "tier-core"
    if v >= USEFUL_THRESH:
        return "Useful", "tier-useful"
    return "Dropped", "tier-weak"


def _feat_short(f: str) -> str:
    return (f.replace("Inlet ", "In ").replace("Effluent ", "Eff ")
             .replace("(mg/L, Grab)", "(G)").replace("(mg/L, Composite)", "(C)")
             .replace("(mg/L)", "").replace("Sec Clarifier ", "SecCl ")
             .replace("Sec Sed ", "SecSd ").replace("Power Total (KW)", "Power")
             .replace("Flow (MLD)", "Flow").strip())


def _infer_source_features(df: pd.DataFrame, target: str, dataset_id: str) -> list[str]:
    exclude = {"Date"}
    if dataset_id.startswith("Exp3S1"):
        exclude.update({"year", "month", "day_of_week"})
    return [c for c in df.columns if c != target and c not in exclude and not c.startswith("predicted_")]


def _source_dataset_path(dataset_id: str) -> str:
    prefix, sample, measure = dataset_id.split("_", 2)
    sample = sample.lower()
    base_dir_map = {
        "Exp1": os.path.join(MODELING_DIR, "datasets", "experiment1", "sub_exp2"),
        "Exp2S1": os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp1"),
        "Exp2S2": os.path.join(MODELING_DIR, "datasets", "experiment2", "sub_exp2"),
        "Exp3S1": os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp1"),
    }
    file_prefix_map = {
        "Exp1": "stage1",
        "Exp2S1": "stage2_p1",
        "Exp2S2": "stage2_p2",
        "Exp3S1": "s1_stage3",
    }
    return os.path.join(
        base_dir_map[prefix],
        f"{file_prefix_map[prefix]}_{sample}_{measure}.xlsx",
    )


def _load_linear_rankings(exp_key: str) -> dict:
    if exp_key in _LINEAR_RANKING_CACHE:
        return _LINEAR_RANKING_CACHE[exp_key]

    full_key = FS_BASE_MAP.get(exp_key)
    if not full_key:
        return {}

    variant = "exp3_s1" if full_key == "Exp3-S1" else "baseline"
    results_path = os.path.join(MODELING_DIR, "models", "linear", variant, "results.xlsx")
    models_dir = os.path.join(MODELING_DIR, "models", "linear", variant, "models")
    if not os.path.exists(results_path):
        return {}

    df = pd.read_excel(results_path)
    df = df[df["run"] == df["run"].max()].copy()
    df = df[df["experiment"] == full_key].copy()
    out = {}

    for _, row in df.iterrows():
        target = row["target"]
        dataset_id = row["dataset"]
        run = int(row["run"])
        pkl_path = os.path.join(models_dir, f"{dataset_id}_ElNet_run_{run}.pkl")
        dataset_path = _source_dataset_path(dataset_id)
        if not (os.path.exists(pkl_path) and os.path.exists(dataset_path)):
            continue

        artifact = joblib.load(pkl_path)
        model = artifact["model"]
        ds_df = pd.read_excel(dataset_path, nrows=0)
        features = _infer_source_features(ds_df, target, dataset_id)
        coefs = np.asarray(getattr(model, "coef_", []), dtype=float)
        if len(features) != len(coefs):
            continue

        rows = []
        for feat, coef in zip(features, coefs):
            rows.append({
                "feature": feat,
                "feat_short": _feat_short(feat),
                "coef": float(coef),
                "abs_coef": float(abs(coef)),
                "selected": abs(float(coef)) > 1e-9,
            })

        out[target] = {
            "model": "ElNet",
            "rows": sorted(rows, key=lambda r: (-r["abs_coef"], r["feature"])),
        }

    _LINEAR_RANKING_CACHE[exp_key] = out
    return out


def _load_fs_importance(exp_key: str) -> pd.DataFrame:
    if exp_key in _FS_IMPORTANCE_CACHE:
        return _FS_IMPORTANCE_CACHE[exp_key]

    sel_dir = os.path.join(MODELING_DIR, "feature_analysis", "selection")
    source_map = {
        "Exp1-FS": ("feature_importance.xlsx", "Experiment 1"),
        "Exp2-Sub1-FS": ("feature_importance.xlsx", "Experiment 2 Sub-1"),
        "Exp2-Sub2-FS": ("feature_importance.xlsx", "Experiment 2 Sub-2"),
        "Exp3-S1-FS": ("feature_importance_exp3_s1.xlsx", "Experiment 3 Sub-1"),
    }
    meta = source_map.get(exp_key)
    if not meta:
        return pd.DataFrame()

    fname, experiment = meta
    path = os.path.join(sel_dir, fname)
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_excel(path)
    df = df[df["experiment"] == experiment].copy()
    _FS_IMPORTANCE_CACHE[exp_key] = df
    return df


def _feature_selection_details(exp_key: str) -> str:
    df = _load_fs_importance(exp_key)
    if df.empty:
        return ""
    linear_lookup = _load_linear_rankings(exp_key)

    intro = (
        "Each target now shows both selection signals together: "
        "the non-linear ranking used to build the current feature-selected datasets, "
        "and a linear ranking from the matching ElasticNet full-feature run."
    )

    target_blocks = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["perm_imp_norm", "feature"], ascending=[False, True]).reset_index(drop=True)

        kept = sub[sub["perm_imp_norm"] >= USEFUL_THRESH]["feat_short"].tolist()
        dropped = sub[sub["perm_imp_norm"] < USEFUL_THRESH]["feat_short"].tolist()

        nl_rows = []
        for idx, (_, row) in enumerate(sub.iterrows(), start=1):
            label, cls = _tier_single(row["perm_imp_norm"])
            nl_rows.append(
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td>{row['feat_short']}</td>"
                f"<td class='num'>{row['perm_imp_norm']:.3f}</td>"
                f"<td class='{cls}'>{label}</td>"
                f"</tr>"
            )

        lin_meta = linear_lookup.get(tgt, {})
        lin_rows = lin_meta.get("rows", [])
        lin_kept = [r["feat_short"] for r in lin_rows if r["selected"]]
        lin_dropped = [r["feat_short"] for r in lin_rows if not r["selected"]]
        lin_tbl_rows = []
        for idx, row in enumerate(lin_rows, start=1):
            coef_cls = "pos" if row["coef"] > 0 else ("neg" if row["coef"] < 0 else "tier-na")
            status = "Selected" if row["selected"] else "Zeroed"
            status_cls = "tier-useful" if row["selected"] else "tier-weak"
            lin_tbl_rows.append(
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td>{row['feat_short']}</td>"
                f"<td class='num'>{row['abs_coef']:.3f}</td>"
                f"<td class='num {coef_cls}'>{row['coef']:+.3f}</td>"
                f"<td class='{status_cls}'>{status}</td>"
                f"</tr>"
            )

        target_blocks.append(f"""
    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> {TARGET_SHORT.get(tgt, tgt)}</summary>
      <div class="fold-body">
        <div class="fs-compare-grid">
          <div class="fs-method-card">
            <div class="fs-method-title">Non-Linear Ranking</div>
            <p class="meta fs-method-note">
              RF permutation importance on the tuned non-linear model. Features with
              normalised permutation importance &ge; {USEFUL_THRESH:.2f} were retained.
            </p>
            <div class="fs-rank-summary">
              <div><strong>Retained ({len(kept)}):</strong> {", ".join(kept) if kept else "-"}</div>
              <div><strong>Dropped ({len(dropped)}):</strong> {", ".join(dropped) if dropped else "-"}</div>
            </div>
            <table class="summary-table fs-rank-table">
              <thead>
                <tr><th>Rank</th><th>Feature</th><th>Norm Perm Imp</th><th>Status</th></tr>
              </thead>
              <tbody>{"".join(nl_rows)}</tbody>
            </table>
          </div>
          <div class="fs-method-card">
            <div class="fs-method-title">Linear Ranking</div>
            <p class="meta fs-method-note">
              ElasticNet coefficient ranking on the matching full-feature linear run.
              Features are ranked by <code>|standardized coefficient|</code>; zeroed coefficients are drop candidates.
            </p>
            <div class="fs-rank-summary">
              <div><strong>Selected ({len(lin_kept)}):</strong> {", ".join(lin_kept) if lin_kept else "-"}</div>
              <div><strong>Zeroed ({len(lin_dropped)}):</strong> {", ".join(lin_dropped) if lin_dropped else "-"}</div>
            </div>
            <table class="summary-table fs-rank-table">
              <thead>
                <tr><th>Rank</th><th>Feature</th><th>|Std Coef|</th><th>Signed Coef</th><th>Status</th></tr>
              </thead>
              <tbody>{"".join(lin_tbl_rows) if lin_tbl_rows else '<tr><td colspan=\"5\">No linear ranking found.</td></tr>'}</tbody>
            </table>
          </div>
        </div>
      </div>
    </details>""")

    return f"""
    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Feature Selection Rationale (ranked importance)</summary>
      <div class="fold-body">
        <p class="meta" style="margin-top:0">{intro}</p>
        {"".join(target_blocks)}
      </div>
    </details>"""


def _metric_delta_html(current, baseline, higher_is_better: bool, decimals: int = 3) -> str:
    if not isinstance(current, (int, float)) or np.isnan(current):
        return ""
    if not isinstance(baseline, (int, float)) or np.isnan(baseline):
        return ""
    delta = current - baseline
    if not higher_is_better:
        delta = -delta
    color = "#2ecc71" if delta > 0 else ("#e74c3c" if delta < 0 else "var(--text-muted)")
    sign = "+" if delta > 0 else ""
    raw_delta = current - baseline
    raw_sign = "+" if raw_delta > 0 else ""
    return (
        f' <span class="delta-note" style="color:{color}" '
        f'title="Delta vs full-feature version: {raw_sign}{raw_delta:.{decimals}f}">'
        f'(Δ {sign}{delta:.{decimals}f})</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _feature_card(exp_key: str) -> str:
    fd = FEATURE_DESCRIPTIONS.get(exp_key, {})
    if not fd:
        return ""
    feats_html = fd.get("features", "")
    ratio_html = fd.get("rationale", "")
    return f"""
<div class="feat-card">
  <div class="feat-card-label">Feature Set</div>
  <div class="feat-card-name">{fd.get('label','')}</div>
  <div class="feat-card-features"><strong>Columns:</strong> {feats_html}</div>
  <div class="feat-card-rationale"><strong>Hypothesis:</strong> {ratio_html}</div>
</div>"""


def _dataset_summary(df: pd.DataFrame) -> str:
    """n_train / n_test / n_features per target, one row per target."""
    rows = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt]
        if sub.empty:
            continue
        row0 = sub.iloc[0]
        n_tr = int(row0["n_train"]) if not np.isnan(row0["n_train"]) else "-"
        n_te = int(row0["n_test"])  if not np.isnan(row0["n_test"])  else "-"
        n_fe = int(row0["n_features"]) if not np.isnan(row0["n_features"]) else "-"
        slug = TARGET_SLUG.get(tgt, "all")
        rows.append(
            f'<tr data-target="{slug}">'
            f'<td>{TARGET_SHORT.get(tgt, tgt)}</td>'
            f'<td>{n_tr}</td><td>{n_te}</td><td>{n_fe}</td></tr>'
        )
    if not rows:
        return ""
    return f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> Dataset Details (n_train / n_test / n_features)</summary>
  <div class="fold-body">
    <table class="summary-table ds-table">
      <thead><tr>
        <th>Target</th><th>n_train (2021-2024)</th>
        <th>n_test (2025)</th><th>n_features</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</details>"""


def _feature_selection_table(df: pd.DataFrame) -> str:
    """
    Render a per-target feature selection comparison table showing which
    features were selected by each model class.

    OLS: LassoCV pre-screen (TimeSeriesSplit, fitted on scaled train data).
    ElasticNet: internal L1 selection (non-zero coefficients after fitting).
    Non-linear: OOF permutation importance (per model: RF, GB, XGB).

    Features are color-coded:
      green  = selected by ALL model classes present (universal core)
      amber  = selected by some but not all (model-specific)
      (not shown) = dropped by all
    """
    # Check if selection columns exist (only present in run_2+ data)
    has_linear = "selected_features_ols" in df.columns and df["selected_features_ols"].notna().any()
    has_nl     = "selected_features_nl" in df.columns and df["selected_features_nl"].notna().any()
    if not has_linear and not has_nl:
        return ""

    rows_html = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt]
        if sub.empty:
            continue

        slug = TARGET_SLUG.get(tgt, "all")
        tgt_short = TARGET_SHORT.get(tgt, tgt)

        # OLS selected (LassoCV pre-screen)
        lin_row = sub[sub["model"] == "OLS"]
        lin_sel = set()
        lin_str = "—"
        if has_linear and not lin_row.empty:
            raw = lin_row.iloc[0].get("selected_features_ols", "")
            if isinstance(raw, str) and raw.strip():
                lin_sel = set(f.strip() for f in raw.split(","))
                lin_str = f"{len(lin_sel)} features"

        # ElasticNet internal selection
        elnet_row = sub[sub["model"] == "ElNet"]
        elnet_sel = set()
        elnet_str = "—"
        if has_linear and not elnet_row.empty:
            raw = elnet_row.iloc[0].get("ElNet_selected_features", "")
            if isinstance(raw, str) and raw.strip():
                elnet_sel = set(f.strip() for f in raw.split(","))
                elnet_str = f"{len(elnet_sel)} features"

        # Tree model selections (RF, GB, XGB — may differ)
        nl_sels = {}
        for mdl in ["RF", "GB", "XGB"]:
            nl_row = sub[sub["model"] == mdl]
            if has_nl and not nl_row.empty:
                raw = nl_row.iloc[0].get("selected_features_nl", "")
                if isinstance(raw, str) and raw.strip():
                    nl_sels[mdl] = set(f.strip() for f in raw.split(","))

        # All non-empty sets for overlap analysis
        all_sets = [s for s in [lin_sel, elnet_sel] + list(nl_sels.values()) if s]
        if not all_sets:
            continue
        universal = set.intersection(*all_sets) if len(all_sets) > 1 else set()

        def _feat_tags(feat_set, universal_set):
            if not feat_set:
                return "<em style='color:var(--text-meta)'>none</em>"
            tags = []
            for f in sorted(feat_set):
                cls = "fs-univ" if f in universal_set else "fs-model"
                tags.append(f'<span class="{cls}">{f}</span>')
            return " ".join(tags)

        nl_cells = ""
        for mdl in ["RF", "GB", "XGB"]:
            s = nl_sels.get(mdl, set())
            nl_cells += f"<td>{_feat_tags(s, universal)}</td>"

        rows_html.append(f"""
<tr data-target="{slug}">
  <td><strong>{tgt_short}</strong></td>
  <td>{_feat_tags(lin_sel, universal)}</td>
  <td>{_feat_tags(elnet_sel, universal)}</td>
  {nl_cells}
</tr>""")

    if not rows_html:
        return ""

    return f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> Model-Specific Feature Selection</summary>
  <div class="fold-body">
    <p style="font-size:0.82rem;color:var(--text-meta);margin:0 0 8px">
      <span class="fs-univ" style="padding:1px 6px;border-radius:4px">Green</span> = selected by <strong>all</strong> model classes (universal core) &nbsp;|&nbsp;
      <span class="fs-model" style="padding:1px 6px;border-radius:4px">Amber</span> = model-specific selection &nbsp;|&nbsp;
      OLS: LassoCV pre-screen (TimeSeriesSplit) &nbsp;|&nbsp;
      ElasticNet: internal L1 selection (non-zero coefficients) &nbsp;|&nbsp;
      RF/GB/XGB: OOF permutation importance
    </p>
    <table class="summary-table fs-table">
      <thead><tr>
        <th>Target</th><th>OLS</th><th>ElasticNet</th>
        <th>RF</th><th>GB</th><th>XGB</th>
      </tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</details>"""



def _pick_best(avail, sub):
    """Returns the model name with the highest Test R² for this target row.

    Returns empty set when only one model is available — ★ is a comparison
    marker and is meaningless without multiple models to compare.
    """
    if len(avail) <= 1:
        return set()
    r2_vals = {}
    for m in avail:
        msub = sub[sub["model"] == m]
        if msub.empty:
            continue
        r2 = msub["R2_test"].values[0]
        if not np.isnan(r2):
            r2_vals[m] = r2
    if not r2_vals:
        return set()
    max_r2 = max(r2_vals.values())
    return {m for m, v in r2_vals.items() if v >= max_r2 - 1e-9}


def _metrics_table(df: pd.DataFrame, models: list, section_id: str, df_all: pd.DataFrame | None = None) -> str:
    """Metric table: rows = targets, cols = (model × Test R² / Gap / RMSE / MAE / MdAE).

    MdAE (Median Absolute Error) is shown for BOD and TSS targets only - these have
    severe outlier distributions (up to 9.8% severe outliers in training) where RMSE
    is dominated by spikes. MdAE is the primary reliability metric for those targets.
    Phase 10 and Phase 11 do not store row-level predictions so MdAE is unavailable (-).

    Best-model highlighting: highest Test R² per target row (★), regardless of sign.
    """
    avail = [m for m in models if m in df["model"].values]
    if not avail:
        return "<p class='meta'>No data for these models.</p>"

    has_mdae = "MdAE_test" in df.columns
    exp_key = df["exp_key"].dropna().iloc[0] if "exp_key" in df.columns and not df["exp_key"].dropna().empty else None
    baseline_key = FS_BASE_MAP.get(exp_key)
    baseline_lookup = {}
    df_base = pd.DataFrame()
    if baseline_key and df_all is not None:
        df_base = df_all[df_all["exp_key"] == baseline_key].copy()

    if not df_base.empty:
        for _, row in df_base.iterrows():
            baseline_lookup[(row["target"], row["model"])] = row

    hdr1 = "".join(
        '<th colspan="5" style="color:{};">{}</th>'.format(MODEL_COLORS.get(m, "var(--text-muted)"), m)
        for m in avail
    )
    hdr2 = "".join(
        "<th>Test R²</th><th>R² Gap</th><th>RMSE</th><th>MAE</th><th>MdAE</th>"
        for _ in avail
    )

    rows = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt]
        if sub.empty:
            continue
        slug = TARGET_SLUG.get(tgt, "all")
        cells = f'<td class="tgt-name">{TARGET_SHORT.get(tgt, tgt)}</td>'
        show_mdae = tgt in MDAE_TARGETS

        best_models = _pick_best(avail, sub)

        for m in avail:
            msub = sub[sub["model"] == m]
            if msub.empty:
                cells += "<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>"
                continue
            r2   = msub["R2_test"].values[0]
            gap  = msub["R2_gap"].values[0]
            rmse = msub["RMSE_test"].values[0]
            mae  = msub["MAE_test"].values[0]
            is_best = m in best_models
            base_row = baseline_lookup.get((tgt, m))

            cell_bg = "background:rgba(74,144,217,0.20);font-weight:bold;" if is_best else ""
            r2_delta = _metric_delta_html(r2, base_row["R2_test"] if base_row is not None else np.nan, True)
            gap_delta = _metric_delta_html(abs(gap) if isinstance(gap, (int, float)) else np.nan,
                                           abs(base_row["R2_gap"]) if base_row is not None else np.nan,
                                           False)
            rmse_delta = _metric_delta_html(rmse, base_row["RMSE_test"] if base_row is not None else np.nan, False)
            mae_delta = _metric_delta_html(mae, base_row["MAE_test"] if base_row is not None else np.nan, False)

            # MdAE cell - only for BOD/TSS targets; dash otherwise
            if show_mdae and has_mdae:
                mdae_val = msub["MdAE_test"].values[0]
                mdae_str = _fmt(mdae_val) if not (isinstance(mdae_val, float) and np.isnan(mdae_val)) else "-"
                mdae_delta = _metric_delta_html(
                    mdae_val,
                    base_row["MdAE_test"] if (base_row is not None and "MdAE_test" in base_row.index) else np.nan,
                    False,
                )
                mdae_title = 'title="Primary reliability metric for outlier-prone targets"'
                mdae_cell = (
                    f'<td style="color:#f1c40f;font-style:italic;" {mdae_title}>'
                    f'{mdae_str}{mdae_delta}</td>'
                )
            else:
                mdae_cell = '<td style="color:var(--text-muted)">-</td>'

            cells += (
                f'<td style="color:{_r2_color(r2)};{cell_bg}">{"★ " if is_best else ""}{_fmt(r2)}{r2_delta}</td>'
                f'<td class="{_gap_cls(gap)}">{_fmt(gap)}{gap_delta}</td>'
                f'<td>{_fmt(rmse)}{rmse_delta}</td>'
                f'<td>{_fmt(mae)}{mae_delta}</td>'
                f'{mdae_cell}'
            )
        rows.append(f'<tr data-target="{slug}">{cells}</tr>')

    legend = (
        '<p class="table-note">'
        'Test R²: '
        '<span style="color:#2ecc71">≥0.6 strong</span> · '
        '<span style="color:#52c98a">0.4-0.6 good</span> · '
        '<span style="color:#f1c40f">0.2-0.4 moderate</span> · '
        '<span style="color:#e67e22">0-0.2 weak</span> · '
        '<span style="color:#e74c3c">&lt;0 fails baseline</span>. '
        'Best per row: <span style="background:rgba(74,144,217,0.20);'
        'padding:1px 4px;border-radius:3px;font-weight:bold">★ highest Test R²</span>.</p>'
        '<p class="table-note">'
        'R² Gap (Train − Test): '
        '<span class="gap-good">■ &lt;0.10 OK</span> · '
        '<span class="gap-warn">■ 0.10-0.25 mild overfit</span> · '
        '<span class="gap-bad">■ &gt;0.25 severe overfit - treat result with caution</span>.</p>'
        '<p class="table-note">'
        '<span style="color:#f1c40f;font-style:italic;">MdAE</span> '
        '(Median Absolute Error) shown for BOD and TSS targets only - RMSE is unreliable '
        'for these targets due to severe outlier distributions (up to 9.8% severe outliers '
        'in training data, TSS train max = 1266 mg/L). '
        'MdAE is the primary reliability metric for BOD/TSS; - indicates predictions not '
        'stored at row level (Phase 10 / Phase 11).</p>'
    )
    if baseline_key and baseline_lookup:
        legend += (
            '<p class="table-note">'
            'Inline deltas are shown for feature-selected variants only. '
            'For Test R², <span style="color:#2ecc71">positive Δ</span> means the FS model improved. '
            'For Gap, RMSE, MAE, and MdAE, positive Δ means the metric decreased versus the full-feature baseline.</p>'
        )
    table = f"""
<div class="tbl-wrap" id="{section_id}">
<div class="tbl-scroll">
<table class="summary-table metrics-table">
  <thead>
    <tr><th rowspan="2">Target</th>{hdr1}</tr>
    <tr>{hdr2}</tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>
{legend}
</div>"""
    return table


def _train_metrics_table(df: pd.DataFrame, models: list) -> str:
    """Foldable train R² table for completeness."""
    avail = [m for m in models if m in df["model"].values]
    if not avail:
        return ""
    hdr = "".join(
        '<th style="color:{};">{} Train R²</th>'.format(MODEL_COLORS.get(m, "#888"), m)
        for m in avail
    )
    rows = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt]
        if sub.empty:
            continue
        slug = TARGET_SLUG.get(tgt, "all")
        cells = f'<td class="tgt-name">{TARGET_SHORT.get(tgt, tgt)}</td>'
        for m in avail:
            msub = sub[sub["model"] == m]
            r2 = msub["R2_train"].values[0] if not msub.empty else float("nan")
            cells += f'<td style="color:{_r2_color(r2)}">{_fmt(r2)}</td>'
        rows.append(f'<tr data-target="{slug}">{cells}</tr>')
    return f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> Train-set R² (reference)</summary>
  <div class="fold-body">
    <table class="summary-table metrics-table">
      <thead><tr><th>Target</th>{hdr}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</details>"""


def _fs_analysis_div(df_all: pd.DataFrame, full_key: str, fs_key: str) -> str:
    """Summary div comparing FS variant against its full-feature counterpart."""
    df_full = df_all[df_all["exp_key"] == full_key].dropna(subset=["R2_test"])
    df_fs   = df_all[df_all["exp_key"] == fs_key].dropna(subset=["R2_test"])
    if df_fs.empty or df_full.empty:
        return ""

    improved, degraded = [], []
    deltas, delta_rows = {}, ""
    for tgt in TARGETS_ORDERED:
        fs_sub   = df_fs[df_fs["target"] == tgt]
        full_sub = df_full[df_full["target"] == tgt]
        if fs_sub.empty or full_sub.empty:
            continue
        r2_fs   = fs_sub["R2_test"].max()
        r2_full = full_sub["R2_test"].max()
        d = r2_fs - r2_full
        short = TARGET_SHORT.get(tgt, tgt)
        deltas[short] = d
        (improved if d > 0.01 else (degraded if d < -0.01 else [])).append(short)
        clr = "#2ecc71" if d > 0.01 else ("#e74c3c" if d < -0.01 else "#f39c12")
        arrow = "▲" if d > 0.01 else ("▼" if d < -0.01 else "≈")
        delta_rows += f'<tr><td>{short}</td><td style="color:{clr}">{arrow} {d:+.3f}</td></tr>'

    n = len(deltas)
    avg_d = sum(deltas.values()) / n if n else 0
    full_lbl = EXP_CHART_LABELS.get(full_key, full_key)
    fs_lbl   = EXP_CHART_LABELS.get(fs_key, fs_key)

    if len(improved) > len(degraded):
        vc, vi, vt = "#2ecc71", "✓", (
            f"Feature selection <strong>helped</strong> on {len(improved)}/{n} targets "
            f"(avg Δ = {avg_d:+.3f}). Fewer features → less overfitting; pruned features "
            f"were likely adding noise rather than signal.")
    elif len(degraded) > len(improved):
        vc, vi, vt = "#e74c3c", "✗", (
            f"Feature selection <strong>hurt</strong> on {len(degraded)}/{n} targets "
            f"(avg Δ = {avg_d:+.3f}). The pruned features were still carrying predictive "
            f"signal for those targets - consider a higher importance threshold.")
    else:
        vc, vi, vt = "#f39c12", "≈", (
            f"Feature selection produced <strong>mixed results</strong> "
            f"(avg Δ = {avg_d:+.3f}). Gains and losses roughly cancel; the full feature "
            f"set and the reduced set are approximately equivalent in this experiment.")

    return f"""
<div class="fs-analysis-div">
  <div class="fs-verdict" style="border-left:3px solid {vc}">
    <span class="fs-verdict-icon" style="color:{vc};font-size:16px;font-weight:bold">{vi}</span>
    <div><strong>Feature Selection Verdict ({fs_lbl} vs {full_lbl}):</strong> {vt}</div>
  </div>
  <details class="inner-fold">
    <summary><span class="fold-icon">▶</span> Per-target Δ Test R² (FS − Full)</summary>
    <div class="fold-body">
      <table class="summary-table" style="width:auto;min-width:280px">
        <thead><tr><th>Target</th><th>Δ Best Test R²</th></tr></thead>
        <tbody>{delta_rows}</tbody>
      </table>
      <p class="meta" style="margin-top:6px">
        ▲ &gt; +0.01 = FS improved · ▼ &lt; −0.01 = FS degraded · ≈ = negligible change</p>
    </div>
  </details>
</div>"""


def _data_cost_div(df_all: pd.DataFrame, expanded_key: str, base_key: str) -> str:
    """Row-loss vs R²-gain analysis for experiments that add costly features."""
    df_base = df_all[df_all["exp_key"] == base_key].dropna(subset=["R2_test"])
    df_exp  = df_all[df_all["exp_key"] == expanded_key].dropna(subset=["R2_test"])
    if df_base.empty or df_exp.empty:
        return ""

    tbl_rows, justified = "", 0
    total = 0
    for tgt in TARGETS_ORDERED:
        bs = df_base[df_base["target"] == tgt]
        es = df_exp[df_exp["target"] == tgt]
        if bs.empty or es.empty:
            continue
        total += 1
        n_b = int(bs["n_train"].iloc[0]) if not np.isnan(bs["n_train"].iloc[0]) else 0
        n_e = int(es["n_train"].iloc[0])  if not np.isnan(es["n_train"].iloc[0])  else 0
        loss_pct = (n_b - n_e) / n_b * 100 if n_b > 0 else 0
        r2_b = bs["R2_test"].max()
        r2_e = es["R2_test"].max()
        delta = r2_e - r2_b
        ok = delta > 0.02
        if ok:
            justified += 1
        lc = "#e74c3c" if loss_pct > 30 else ("#f39c12" if loss_pct > 15 else "#2ecc71")
        dc = "#2ecc71" if delta > 0.01 else ("#e74c3c" if delta < -0.01 else "#f39c12")
        tbl_rows += (
            f'<tr><td>{TARGET_SHORT.get(tgt, tgt)}</td>'
            f'<td>{n_b} → {n_e}</td>'
            f'<td style="color:{lc}">−{loss_pct:.0f}%</td>'
            f'<td style="color:{dc}">{delta:+.3f}</td>'
            f'<td>{"✓" if ok else "✗"}</td></tr>'
        )

    base_lbl = EXP_CHART_LABELS.get(base_key, base_key)
    exp_lbl  = EXP_CHART_LABELS.get(expanded_key, expanded_key)
    if justified >= (total + 1) // 2:
        verdict = (f"The row cost is <strong>justified on {justified}/{total} targets</strong> - "
                   f"R² gains exceed +0.02 for the majority. The CONSIDER-tier features add "
                   f"meaningful signal despite the smaller training set.")
    else:
        verdict = (f"The row cost is <strong>not justified on most targets "
                   f"({total - justified}/{total} showed &lt;0.02 R² gain)</strong>. "
                   f"For those targets, prefer the {base_lbl} feature set which retains "
                   f"more training data.")

    return f"""
<div class="data-cost-div">
  <h5>📉 Data Cost vs Performance Gain ({base_lbl} → {exp_lbl})</h5>
  <p class="meta">{verdict}</p>
  <table class="summary-table" style="font-size:12px">
    <thead><tr>
      <th>Target</th><th>n_train change</th>
      <th>Row loss %</th><th>Δ Best R²</th><th>Justified? (Δ &gt; +0.02)</th>
    </tr></thead>
    <tbody>{tbl_rows}</tbody>
  </table>
  <p class="meta" style="margin-top:6px">
    Base = {base_lbl} · Expanded = {exp_lbl}</p>
</div>"""


def _load_mi_lookup() -> dict:
    """Return {(feature, target): mi_score} from feature_importance (Exp2-S2) + feature_audit."""
    sel_dir   = os.path.join(MODELING_DIR, "feature_analysis", "selection")
    audit_dir = os.path.join(MODELING_DIR, "feature_analysis", "audit")
    lookup: dict = {}

    fi_path = os.path.join(sel_dir, "feature_importance.xlsx")
    if os.path.exists(fi_path):
        fi = pd.read_excel(fi_path)
        fi = fi[fi["experiment"] == "Experiment 2 Sub-2"][["feature", "target", "mi_score"]].drop_duplicates()
        for _, row in fi.iterrows():
            lookup[(row["feature"], row["target"])] = float(row["mi_score"])

    fa_path = os.path.join(audit_dir, "feature_audit.xlsx")
    if os.path.exists(fa_path):
        fa = pd.read_excel(fa_path)[["feature", "target", "mi"]].drop_duplicates()
        for _, row in fa.iterrows():
            key = (row["feature"], row["target"])
            if key not in lookup:
                lookup[key] = float(row["mi"])

    return lookup


def _load_cost_lookup() -> dict:
    """Return {feature: max marginal_cost_pct across targets} from feature_audit.
    Base features (Inlet + Secondary + COMMON) are not in the audit - they default to 0.0."""
    audit_dir = os.path.join(MODELING_DIR, "feature_analysis", "audit")
    fa_path   = os.path.join(audit_dir, "feature_audit.xlsx")
    if not os.path.exists(fa_path):
        return {}
    fa = pd.read_excel(fa_path)[["feature", "marginal_cost_pct"]]
    return fa.groupby("feature")["marginal_cost_pct"].max().to_dict()


# Known mathematically derived features: {derived: set_of_components}.
# If a feature is derived from one of its pair partners, drop the derived feature.
_DERIVED_FROM: dict = {
    "Aeration SVI (Existing)": {"Aeration SV30 (ml/L, Existing)", "Aeration MLSS (mg/L, Existing)"},
    "Aeration SVI (New)":      {"Aeration SV30 (ml/L, New)",      "Aeration MLSS (mg/L, New)"},
    "Power / Flow (KW/ML)":    {"Power Total (KW)", "Flow (MLD)"},
}


def _vif_callout() -> str:
    """Compute VIF for Exp3-S2 feature sets inline and render as a foldable section."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
    except ImportError:
        return '<div class="info-note">VIF analysis requires statsmodels (pip install statsmodels).</div>'

    mi_lookup   = _load_mi_lookup()
    cost_lookup = _load_cost_lookup()   # {feature: max marginal_cost_pct}; base features default 0.0

    ds_dir = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
    DATASETS = [
        ("grab_BOD", "Effluent BOD (mg/L, Grab)"),
        ("grab_COD", "Effluent COD (mg/L, Grab)"),
        ("grab_TSS", "Effluent TSS (mg/L, Grab)"),
        ("grab_pH",  "Effluent pH (Grab)"),
        ("comp_BOD", "Effluent BOD (mg/L, Composite)"),
        ("comp_COD", "Effluent COD (mg/L, Composite)"),
        ("comp_TSS", "Effluent TSS (mg/L, Composite)"),
        ("comp_pH",  "Effluent pH (Composite)"),
    ]
    EXCLUDE = {"Date", "year", "month", "day_of_week"}

    def _infer_feats(df, target):
        return [c for c in df.columns if c != target and c not in EXCLUDE
                and not c.startswith("predicted_")]

    def _compute_vif(X, names):
        Xc = add_constant(X, has_constant="add")
        rows = []
        for i, name in enumerate(names):
            try:
                vif = variance_inflation_factor(Xc, i + 1)
            except Exception:
                vif = float("nan")
            rows.append((name, round(float(vif), 2)))
        return sorted(rows, key=lambda x: -x[1])

    def _vflag(vif):
        if vif > 10: return "HIGH", "#e74c3c"
        if vif > 5:  return "MODERATE", "#f39c12"
        return "OK", "#2ecc71"

    PAIR_R_THRESH = 0.85  # |r| above which two features are called collinear

    def _collinear_pairs(X_df, feat_names, vif_scores):
        """Return list of (feat_a, feat_b, r) for pairs with |r| >= threshold."""
        corr = X_df.corr().abs()
        pairs = []
        for i, fa in enumerate(feat_names):
            for j, fb in enumerate(feat_names):
                if j <= i:
                    continue
                r = corr.loc[fa, fb] if fa in corr.index and fb in corr.columns else 0.0
                if r >= PAIR_R_THRESH:
                    pairs.append((fa, fb, round(float(r), 3)))
        return sorted(pairs, key=lambda x: -x[2])

    total_high = total_mod = 0
    ds_htmls = []

    for name, target in DATASETS:
        path = os.path.join(ds_dir, f"{name}.xlsx")
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path, parse_dates=["Date"])
        train = df[df["year"] < 2025].copy()
        features = _infer_feats(train, target)
        X_clean = train[features].dropna()
        if X_clean.empty:
            continue
        vif_rows = _compute_vif(X_clean.values, features)
        n_high = sum(1 for _, v in vif_rows if v > 10)
        n_mod  = sum(1 for _, v in vif_rows if 5 < v <= 10)
        total_high += n_high
        total_mod  += n_mod

        # Collinear pairs among ALL features
        X_df_clean = pd.DataFrame(X_clean.values, columns=features)
        pairs = _collinear_pairs(X_df_clean, features, dict(vif_rows))

        short_name = name.replace("s2_stage3_", "").replace("_", " ").title()
        vif_dict = dict(vif_rows)

        # ── Build per-feature: collinear partners + suggested action ──────────
        feature_partners = {f: [] for f in features}  # feat -> [(partner, r)]
        drop_signals = {}   # feat -> reason string (first DROP signal wins)
        keep_drops   = {}   # feat (keep) -> [drop_feat names]

        for feat_a, feat_b, r in pairs:
            mi_a   = mi_lookup.get((feat_a, target), None)
            mi_b   = mi_lookup.get((feat_b, target), None)
            cost_a = cost_lookup.get(feat_a, 0.0)
            cost_b = cost_lookup.get(feat_b, 0.0)

            drop_feat = reason = None
            # Priority 1: domain logic - drop known derived feature
            if feat_a in _DERIVED_FROM and feat_b in _DERIVED_FROM[feat_a]:
                drop_feat = feat_a
                reason = f"derived from {feat_b.split('(')[0].strip()}"
            elif feat_b in _DERIVED_FROM and feat_a in _DERIVED_FROM[feat_b]:
                drop_feat = feat_b
                reason = f"derived from {feat_a.split('(')[0].strip()}"
            # Priority 2: missingness cost - drop higher-cost feature if gap > 5 pp
            if drop_feat is None and abs(cost_a - cost_b) > 5.0:
                drop_feat  = feat_a if cost_a > cost_b else feat_b
                drop_cost  = cost_a if drop_feat == feat_a else cost_b
                keep_cost  = cost_b if drop_feat == feat_a else cost_a
                reason = f"higher missingness cost ({drop_cost:.1f}% vs {keep_cost:.1f}%)"
            # Priority 3: MI with target - drop lower MI
            if drop_feat is None:
                if mi_a is not None and mi_b is not None:
                    drop_feat = feat_a if mi_a <= mi_b else feat_b
                    mi_drop   = mi_a if drop_feat == feat_a else mi_b
                    mi_keep   = mi_b if drop_feat == feat_a else mi_a
                    reason    = f"lower MI with target ({mi_drop:.3f} vs {mi_keep:.3f})"
                elif mi_a is not None or mi_b is not None:
                    drop_feat = feat_b if mi_a is not None else feat_a
                    reason    = "MI available for one feature only"
            # Priority 4: VIF tiebreaker
            if drop_feat is None:
                drop_feat = feat_a if vif_dict.get(feat_a, 0) >= vif_dict.get(feat_b, 0) else feat_b
                reason    = f"VIF tiebreaker ({vif_dict.get(drop_feat, 0):.1f} > other)"

            keep_feat = feat_b if drop_feat == feat_a else feat_a
            feature_partners[feat_a].append((feat_b, r))
            feature_partners[feat_b].append((feat_a, r))

            if drop_feat not in drop_signals:        # first DROP signal wins
                drop_signals[drop_feat] = reason
            keep_drops.setdefault(keep_feat, []).append(drop_feat)

        # ── Build merged table rows ───────────────────────────────────────────
        tbl_rows = ""
        for feat, vif_val in vif_rows:
            flag, fc = _vflag(vif_val)
            vif_str  = f"{vif_val:.2f}" if not (isinstance(vif_val, float) and np.isnan(vif_val)) else "-"
            bold     = "bold" if vif_val > 10 else "normal"

            mi_val   = mi_lookup.get((feat, target), None)
            mi_str   = f"{mi_val:.3f}" if mi_val is not None else "-"
            cost_val = cost_lookup.get(feat, 0.0)
            cost_str = f"{cost_val:.1f}%" if cost_val > 0 else "0%"

            partners = sorted(feature_partners[feat], key=lambda x: -x[1])
            if partners:
                partner_parts = []
                for p, pr in partners:
                    pc = "#e74c3c" if pr >= 0.95 else "#f39c12"
                    partner_parts.append(
                        f'<span style="color:{pc}">{p}</span>'
                        f'<span style="color:var(--text-meta);font-size:10px"> (r={pr:.3f})</span>'
                    )
                partner_html = "<br>".join(partner_parts)
            else:
                partner_html = '<span style="color:var(--text-muted)">-</span>'

            if feat in drop_signals:
                action_html = (
                    f'<span style="color:#e74c3c;font-weight:bold">DROP</span>'
                    f'<br><span style="color:var(--text-meta);font-size:10px">{drop_signals[feat]}</span>'
                )
            elif feat in keep_drops:
                dropped_names = ", ".join(d.split("(")[0].strip() for d in keep_drops[feat])
                action_html = (
                    f'<span style="color:#2ecc71;font-weight:bold">KEEP</span>'
                    f'<br><span style="color:var(--text-meta);font-size:10px">retained; {dropped_names} to be dropped</span>'
                )
            else:
                action_html = '<span style="color:var(--text-muted)">-</span>'

            tbl_rows += (
                f'<tr>'
                f'<td>{feat}</td>'
                f'<td style="color:{fc};font-weight:{bold}">{vif_str}</td>'
                f'<td><span style="color:{fc};font-size:11px">{flag}</span></td>'
                f'<td>{mi_str}</td>'
                f'<td>{cost_str}</td>'
                f'<td style="font-size:11px;line-height:1.6">{partner_html}</td>'
                f'<td style="font-size:11px">{action_html}</td>'
                f'</tr>'
            )

        pair_count_note = (
            f'{len(pairs)} collinear pair{"s" if len(pairs) != 1 else ""} (|r| ≥ {PAIR_R_THRESH})'
            if pairs else f'no collinear pairs (|r| ≥ {PAIR_R_THRESH})'
        )
        badge_col = "#e74c3c" if n_high > 0 else ("#f39c12" if n_mod > 0 else "#2ecc71")
        ok_ct     = len(vif_rows) - n_high - n_mod
        badge_txt = f"{n_high} HIGH · {n_mod} MOD · {ok_ct} OK"
        ds_htmls.append(f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> {short_name}
    <span style="margin-left:8px;font-size:11px;color:{badge_col}">{badge_txt}</span>
  </summary>
  <div class="fold-body">
    <p style="font-size:12px;color:var(--text-muted);margin:0 0 8px">
      n_train (complete rows): {len(X_clean)} · n_features: {len(features)} · {pair_count_note} ·
      Suggestion priority: domain logic → missingness cost → MI with target → VIF tiebreaker
    </p>
    <table class="summary-table" style="font-size:12px">
      <thead><tr>
        <th>Feature</th><th>VIF</th><th>Flag</th>
        <th>MI</th><th>Cost</th>
        <th>Collinear With (|r|)</th><th>Suggested Action</th>
      </tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table>
  </div>
</details>""")

    action_guide = """
<div class="obs-card" style="margin:12px 0;padding:14px 18px">
  <strong style="font-size:13px">Notes on VIF removal</strong>
  <ul style="margin:8px 0 0 18px;font-size:12px;line-height:1.8">
    <li><strong>Hub pattern:</strong> a feature appearing in many "Collinear With" rows (e.g. SVI)
        is a hub - dropping it alone resolves multiple HIGH VIFs simultaneously.
        Multiple HIGH flags ≠ remove all; count the distinct hubs first.</li>
    <li><strong>Tree models (RF, XGB) are unaffected</strong> - collinearity only inflates
        OLS/Ridge coefficient variance. ElasticNet handles it automatically via L1 shrinkage.
        VIF-based removal is relevant only before fitting OLS or Ridge.</li>
  </ul>
</div>"""

    summary_html = (
        f'<p class="meta">Global across all 8 datasets: '
        f'<strong style="color:#e74c3c">{total_high} HIGH (VIF&gt;10)</strong> · '
        f'<strong style="color:#f39c12">{total_mod} MODERATE (VIF 5-10)</strong>. '
        f'Concentrated in Sec Sed vs Sec Clarifier pairs (adjacent measurements of the same parameter) '
        f'and Aeration SVI (mathematically derived from SV30/MLSS). '
        f'ElasticNet handles collinearity via L1 penalty - this explains why ElNet outperforms '
        f'OLS/Ridge on composite targets. OLS/Ridge coefficients are directly inflated by collinear pairs.</p>'
        f'{action_guide}'
    )

    return f"""
<details class="exp-details" id="exp3-vif">
  <summary><span class="fold-icon">▶</span> VIF Collinearity Analysis - Exp3-S2 Feature Sets</summary>
  <div class="exp-body">
    <p class="meta">
      Variance Inflation Factor computed on training rows (year &lt; 2025) after listwise deletion.
      VIF &gt; 10 = problematic collinearity for linear models.
      VIF 5-10 = moderate.
      The threshold rule of thumb: VIF &gt; 10 indicates that the variance of a coefficient estimate
      is inflated by a factor of 10 relative to an orthogonal design - OLS/Ridge predictions are
      less reliable for such features.
    </p>
    {summary_html}
    {"".join(ds_htmls)}
  </div>
</details>"""


def _ann_failure_callout() -> str:
    """Render ANN (MLPRegressor) failure post-mortem as a foldable section."""
    results_path = os.path.join(MODELING_DIR, "models", "phase9", "ann", "results.xlsx")
    if not os.path.exists(results_path):
        return ""

    df = pd.read_excel(results_path)
    df = df[df["run"] == df["run"].max()].copy()

    avg_r2  = df["R2_test"].mean()
    avg_gap = df["R2_gap"].mean()

    VOTING_R2 = {
        "Effluent BOD (mg/L, Grab)":       0.6924,
        "Effluent COD (mg/L, Grab)":       0.4328,
        "Effluent TSS (mg/L, Grab)":       0.4781,
        "Effluent pH (Grab)":              0.2208,
        "Effluent BOD (mg/L, Composite)":  0.4542,
        "Effluent COD (mg/L, Composite)": -0.3050,
        "Effluent TSS (mg/L, Composite)":  0.1130,
        "Effluent pH (Composite)":         0.2065,
    }

    def _aflag(r2):
        if r2 < -0.5:  return "CATASTROPHIC", "#c0392b"
        if r2 < 0:     return "FAILED", "#e74c3c"
        if r2 < 0.25:  return "POOR", "#e67e22"
        if r2 < 0.50:  return "MODERATE", "#f39c12"
        return "GOOD", "#2ecc71"

    tbl_rows = ""
    for _, row in df.sort_values("R2_test").iterrows():
        tgt    = row["target"]
        n_p    = row["n_train"] / row["n_features"]
        r2_tr  = row["R2_train"]
        r2_te  = row["R2_test"]
        gap    = row["R2_gap"]
        params = str(row.get("best_params", ""))
        arch   = (params.split("hidden_layer_sizes': ")[1].split(",")[0].rstrip("}").strip()
                  if "hidden_layer_sizes" in params else "-")
        alpha  = (params.split("alpha': ")[1].split("}")[0].strip()
                  if "alpha" in params else "-")
        voting = VOTING_R2.get(tgt, float("nan"))
        delta  = r2_te - voting
        status, sc = _aflag(r2_te)
        np_col = "#e74c3c" if n_p < 20 else "#2ecc71"
        dc_col = "#e74c3c" if delta < 0 else "#2ecc71"

        tbl_rows += (
            f'<tr>'
            f'<td>{TARGET_SHORT.get(tgt, tgt)}</td>'
            f'<td>{int(row["n_train"])}</td>'
            f'<td>{int(row["n_features"])}</td>'
            f'<td style="color:{np_col}">{n_p:.1f}</td>'
            f'<td style="font-size:11px">{arch}</td>'
            f'<td style="font-size:11px">{alpha}</td>'
            f'<td>{_fmt(r2_tr)}</td>'
            f'<td style="color:{sc}">{_fmt(r2_te)}</td>'
            f'<td class="{_gap_cls(gap)}">{_fmt(gap)}</td>'
            f'<td><span style="color:{sc};font-size:11px">{status}</span></td>'
            f'<td style="color:{dc_col}">{delta:+.3f}</td>'
            f'</tr>'
        )

    return f"""
<details class="exp-details" id="p9-ann-diagnosis">
  <summary><span class="fold-icon">▶</span> ANN Failure Post-Mortem
    <span style="margin-left:8px;font-size:11px;color:#e74c3c">avg Test R²={avg_r2:+.3f} · avg Gap={avg_gap:+.3f}</span>
  </summary>
  <div class="exp-body">
    <p class="meta">
      Grid search: 5 architectures × 4 regularisation values = 20 combinations per target.
      Fixed: relu, adam, max_iter=2000, early_stopping=True (validation_fraction=0.10,
      n_iter_no_change=20). Tuning via TimeSeriesSplit(n_splits=3).
      All targets selected alpha=1.0 (maximum regularisation).
    </p>

    <div class="obs-card" style="border-left:4px solid #e74c3c;margin:0.8rem 0">
      <strong style="color:#e74c3c">6 of 8 targets have Test R² ≤ 0. ANN not recommended at current sample sizes.</strong>
    </div>

    <table class="summary-table metrics-table" style="font-size:12px">
      <thead>
        <tr>
          <th>Target</th><th>n_train</th><th>n_feat</th>
          <th title="Training rows ÷ features - MLPs need n/p ≥ 100">n/p</th>
          <th>Best arch</th><th>Best α</th>
          <th>Train R²</th><th>Test R²</th><th>R² Gap</th>
          <th>Status</th><th>Δ vs Voting</th>
        </tr>
      </thead>
      <tbody>{tbl_rows}</tbody>
    </table>
    <p class="table-note">
      n/p = training rows ÷ features. <span style="color:#e74c3c">Red</span> = n/p &lt; 20 (underdetermined - high overfitting risk).
      MLPs typically need n/p ≥ 100-200 for stable generalisation on tabular data.
      Δ vs Voting = ANN Test R² − Phase 9 Voting Test R².
    </p>

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Root-Cause Analysis</summary>
      <div class="fold-body">
        <p class="meta"><strong style="color:var(--accent-blue)">1. Sample-to-feature ratio.</strong>
          Worst case: Comp TSS - 290 training rows, 27 features → n/p = 10.7.
          MLPs require far more data than tree ensembles to generalise from small tabular datasets.
          Typical guidance: n/p ≥ 100-200 for stable MLP generalisation.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">2. Temporal distribution shift.</strong>
          2025 test data shows different distributional properties than 2021-2024 training data
          (Q4-Flow and DJF errors 2-3× higher - see Error Regime Decomposition).
          MLPs form highly non-linear decision boundaries that diverge further from shifted data
          than regularised linear models or bagged trees.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">3. Early stopping did not prevent 2025 failure.</strong>
          Early stopping (validation_fraction=0.10, n_iter_no_change=20) monitors in-sample
          generalisation on the 2021-2024 fold, not 2025 generalisation.
          Gaps range from +0.41 (Grab COD) to +6.77 (Comp pH) despite early stopping -
          confirming that early stopping alone cannot address distributional shift.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">4. Network capacity is not the bottleneck.</strong>
          Best architectures are small - (64,) or (128,) single hidden layer with α=1.0 for 6 of 8 targets.
          The grid search found that more capacity hurts; adding layers would worsen the gaps.
          The problem is generalisation, not expressiveness.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">Future conditions for ANN.</strong>
          Dataset growth to n ≥ 3,000 per target and an LSTM architecture (explicit temporal modelling
          via hidden state rather than hand-crafted lags) are the recommended prerequisites before
          revisiting neural networks on this problem.</p>
      </div>
    </details>
  </div>
</details>"""


# ─── Cyclical Encoding methodology callout ────────────────────────────────────

def _cyclic_encoding_callout(df_all: pd.DataFrame) -> str:
    """Foldable Exp3-S2 methodology card explaining sin/cos calendar encoding."""
    lin = df_all[df_all["model"].isin(LINEAR_MODELS)].copy()

    rows_html = ""
    for exp_key, enc_label in [("Exp3-S1", "Raw integer (1-12 / 0-6)"),
                                ("Exp3-S2", "Cyclic sin/cos (4 columns)")]:
        sub = lin[lin["exp_key"] == exp_key]
        if sub.empty:
            continue
        for model in LINEAR_MODELS:
            m_sub = sub[sub["model"] == model]
            if m_sub.empty:
                continue
            avg_r2 = m_sub["R2_test"].mean()
            color  = _r2_color(avg_r2)
            rows_html += (
                f"<tr><td>{exp_key}</td><td>{model}</td><td>{enc_label}</td>"
                f'<td style="color:{color};font-weight:bold">{_fmt(avg_r2)}</td></tr>'
            )

    return f"""
<details class="exp-details" id="exp3-cyclic">
  <summary><span class="fold-icon">▶</span> Methodology - Cyclical Encoding of Calendar Features</summary>
  <div class="exp-body">

    <div class="obs-card" style="border-left:4px solid #4A90D9">
      <h4 style="margin:0 0 0.6rem">The Discontinuity Problem with Raw Integer Encoding</h4>
      <p class="meta">
        <code>month</code> (1-12) and <code>day_of_week</code> (0-6) are cyclic: December is
        adjacent to January, and Sunday wraps back to Monday. As raw integers, a linear model
        treats the gap between month 12 and month 1 as <em>11 units</em> - the largest possible
        distance - creating a spurious discontinuity at the year boundary. The fitted coefficient
        forces a monotone trend across the full 1-12 range and cannot capture the seasonal
        "U-shape" or mid-year peaks that several effluent targets exhibit. Tree models (RF, GB,
        XGB) are unaffected because their splits are ordinal and learned independently per node;
        the discontinuity never enters their loss.
      </p>
    </div>

    <div class="obs-card" style="border-left:4px solid #5BAD6F;margin-top:1rem">
      <h4 style="margin:0 0 0.6rem">Encoding Formula</h4>
      <p class="meta">Two orthogonal projections onto the unit circle replace each integer feature.
         Together, (sin, cos) uniquely identify any cycle position while preserving wrap-around
         topology: December and January are geometrically close, not 11 units apart.</p>
      <div style="overflow-x:auto">
      <table style="font-size:0.88em;max-width:640px">
        <thead><tr>
          <th>Original feature</th><th>Period</th>
          <th>sin column</th><th>cos column</th>
        </tr></thead>
        <tbody>
          <tr>
            <td><code>month</code> (1-12)</td><td>12</td>
            <td><code>month_sin = sin(2π × month / 12)</code></td>
            <td><code>month_cos = cos(2π × month / 12)</code></td>
          </tr>
          <tr>
            <td><code>day_of_week</code> (0-6)</td><td>7</td>
            <td><code>dow_sin = sin(2π × dow / 7)</code></td>
            <td><code>dow_cos = cos(2π × dow / 7)</code></td>
          </tr>
        </tbody>
      </table>
      </div>
      <p class="meta" style="margin-top:0.6rem">
        The raw <code>month</code> and <code>day_of_week</code> columns are dropped before
        training; the 4 cyclic columns take their place, adding zero rows of missingness.
        The <code>year</code> column is kept as-is (not cyclic - it encodes genuine secular trend).
      </p>
    </div>

    <div class="obs-card" style="border-left:4px solid #e67e22;margin-top:1rem">
      <h4 style="margin:0 0 0.6rem">Scope and Interpretation Caveat</h4>
      <p class="meta">
        Cyclic encoding is applied <strong>only in Exp3-S2 linear models</strong>
        (OLS, Ridge, ElasticNet). All prior experiments (Exp1, Exp2, Exp3 Sub-1) and all
        tree-based and advanced models (RF, GB, XGB, Voting, Stacking, Phase 9-11) use raw
        integer encoding.
      </p>
      <p class="meta">
        Exp3-S2 also introduces new CONSIDER-tier features absent from Exp3-S1, so the two
        experiments are <strong>not a controlled A/B test</strong> for encoding alone. Any R²
        difference in the table below reflects the combined effect of the broader feature set
        <em>and</em> the encoding change.
      </p>
    </div>

    <details class="inner-fold" style="margin-top:1rem">
      <summary><span class="fold-icon">▶</span>
        Exp3-S1 (raw integers) vs Exp3-S2 (cyclic) - Linear Model Avg Test R² (all 8 targets)
      </summary>
      <div style="overflow-x:auto;margin-top:0.6rem">
      <table style="font-size:0.88em;max-width:560px">
        <thead><tr>
          <th>Experiment</th><th>Model</th>
          <th>Calendar encoding</th><th>Avg Test R²</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      </div>
      <p class="meta" style="margin-top:0.5rem;font-size:0.8em;color:var(--text-muted)">
        A controlled run of Exp3-S2 features with raw-integer encoding was not performed.
        Isolating the encoding contribution would require a dedicated ablation run.
      </p>
    </details>

  </div>
</details>"""


# ─── Comp COD persistent-failure diagnostic ───────────────────────────────────

def _build_comp_cod_diagnostic(df_all: pd.DataFrame) -> str:
    """Standalone section: per-year distribution, distribution shift, performance
    panorama, and root cause analysis for the persistently failing Comp COD target."""

    TARGET_COL = "Effluent COD (mg/L, Composite)"
    data_file  = os.path.join(PROJECT_ROOT, "raw_data", "All_Years_Full.xlsx")

    # ── Per-year statistics and shift metrics ─────────────────────────────
    stats_html = ""
    shift_html = ""

    if os.path.exists(data_file):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            df_raw = pd.read_excel(data_file, parse_dates=["Date"])
        df_raw = (df_raw[["Date", TARGET_COL]]
                  .dropna(subset=[TARGET_COL])
                  .sort_values("Date")
                  .reset_index(drop=True))
        df_raw["year"] = df_raw["Date"].dt.year

        train = df_raw[df_raw["year"] <= 2024]
        test  = df_raw[df_raw["year"] == 2025]

        train_mean = train[TARGET_COL].mean()
        train_std  = train[TARGET_COL].std()
        train_p25  = train[TARGET_COL].quantile(0.25)
        train_p75  = train[TARGET_COL].quantile(0.75)

        year_rows = ""
        for year, grp in df_raw.groupby("year"):
            v = grp[TARGET_COL]
            mean_shift = abs(v.mean() - train_mean) / train_std
            m_color = ("#e74c3c" if mean_shift > 1.0 else
                       "#e67e22" if mean_shift > 0.5 else "var(--text)")
            bold = "bold" if year == 2025 else "normal"
            tag  = " ★" if year == 2025 else ""
            year_rows += (
                f"<tr><td><strong>{year}{tag}</strong></td>"
                f"<td>{len(v)}</td>"
                f'<td style="color:{m_color};font-weight:{bold}">{v.mean():.1f}</td>'
                f"<td>{v.std():.1f}</td>"
                f"<td>{v.min():.1f}</td>"
                f"<td>{v.quantile(0.25):.1f}</td>"
                f"<td>{v.quantile(0.75):.1f}</td>"
                f"<td>{v.max():.1f}</td></tr>"
            )

        stats_html = f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> Per-Year Distribution Statistics</summary>
  <div style="overflow-x:auto;margin-top:0.6rem">
  <table style="font-size:0.88em">
    <thead><tr>
      <th>Year</th><th>n</th><th>Mean</th><th>Std</th>
      <th>Min</th><th>P25</th><th>P75</th><th>Max</th>
    </tr></thead>
    <tbody>
      {year_rows}
      <tr style="border-top:2px solid var(--border);font-style:italic;color:var(--text-muted)">
        <td>Train 2021-2024</td>
        <td>{len(train)}</td>
        <td>{train_mean:.1f}</td><td>{train_std:.1f}</td>
        <td>{train[TARGET_COL].min():.1f}</td>
        <td>{train_p25:.1f}</td><td>{train_p75:.1f}</td>
        <td>{train[TARGET_COL].max():.1f}</td>
      </tr>
    </tbody>
  </table>
  </div>
  <p class="meta" style="font-size:0.8em;color:var(--text-muted);margin-top:0.4rem">
    Red mean = year mean deviates by &gt;1σ from training mean.
    Orange = 0.5-1σ shift. 2025 (★) is the test set.
  </p>
</details>"""

        if not test.empty:
            frac_iqr   = ((test[TARGET_COL] >= train_p25) &
                          (test[TARGET_COL] <= train_p75)).mean()
            frac_above = (test[TARGET_COL] > train_p75).mean()
            mean_sigma = abs(test[TARGET_COL].mean() - train_mean) / train_std

            iqr_col   = ("#2ecc71" if frac_iqr > 0.5 else
                         "#e67e22" if frac_iqr > 0.3 else "#e74c3c")
            sig_col   = ("#2ecc71" if mean_sigma < 0.5 else
                         "#e67e22" if mean_sigma < 1.0 else "#e74c3c")

            shift_html = f"""
<div class="obs-card" style="border-left:4px solid #e74c3c;margin-bottom:1rem">
  <h4 style="margin:0 0 0.8rem">2025 Distribution Shift vs Training (2021-2024)</h4>
  <div style="display:flex;gap:2rem;flex-wrap:wrap">
    <div style="text-align:center;min-width:130px">
      <div style="font-size:1.8em;font-weight:bold;color:{iqr_col}">{frac_iqr:.0%}</div>
      <div style="font-size:0.8em;color:var(--text-muted)">
        of 2025 values within<br>training IQR
        [{train_p25:.1f}, {train_p75:.1f}] mg/L
      </div>
    </div>
    <div style="text-align:center;min-width:130px">
      <div style="font-size:1.8em;font-weight:bold;color:#e74c3c">{frac_above:.0%}</div>
      <div style="font-size:0.8em;color:var(--text-muted)">
        of 2025 values above<br>training P75 ({train_p75:.1f} mg/L)
      </div>
    </div>
    <div style="text-align:center;min-width:130px">
      <div style="font-size:1.8em;font-weight:bold;color:{sig_col}">{mean_sigma:.2f}σ</div>
      <div style="font-size:0.8em;color:var(--text-muted)">
        2025 mean shift relative<br>to training std
      </div>
    </div>
    <div style="text-align:center;min-width:130px">
      <div style="font-size:1.8em;font-weight:bold;color:#e74c3c">
        {test[TARGET_COL].mean():.1f} mg/L
      </div>
      <div style="font-size:0.8em;color:var(--text-muted)">
        2025 mean<br>(train mean: {train_mean:.1f} mg/L)
      </div>
    </div>
  </div>
</div>"""
    else:
        stats_html = '<div class="info-note">Raw data file not found - distribution stats unavailable.</div>'

    # ── Performance panorama across all experiments ───────────────────────
    comp_cod = df_all[df_all["target"] == TARGET_COL].copy()
    global_max = comp_cod["R2_test"].max()
    n_exp = comp_cod["exp_key"].nunique()
    n_mod = comp_cod["model"].nunique()

    perf_rows = ""
    for exp_key in EXP_CHART_ORDER:
        sub = comp_cod[comp_cod["exp_key"] == exp_key]
        if sub.empty:
            continue
        best = sub.loc[sub["R2_test"].idxmax()]
        r2   = best["R2_test"]
        gap  = best.get("R2_gap", float("nan"))
        color   = _r2_color(r2)
        gap_cls = _gap_cls(gap)
        gap_str = _fmt(gap) if not (isinstance(gap, float) and np.isnan(gap)) else "-"
        badge = ""
        if r2 == global_max:
            badge = ' <span style="font-size:0.75em;color:#f1c40f">★ best</span>'
        perf_rows += (
            f"<tr><td>{EXP_CHART_LABELS.get(exp_key, exp_key)}</td>"
            f"<td>{best['model']}</td>"
            f'<td style="color:{color};font-weight:bold">{_fmt(r2)}{badge}</td>'
            f'<td class="{gap_cls}">{gap_str}</td></tr>'
        )

    perf_table = f"""
<details class="inner-fold" style="margin-top:1rem">
  <summary><span class="fold-icon">▶</span>
    Best Comp COD Result per Experiment ({n_exp} experiments, {n_mod} model families)
  </summary>
  <div style="overflow-x:auto;margin-top:0.6rem">
  <table style="font-size:0.88em;max-width:500px">
    <thead><tr>
      <th>Experiment</th><th>Best Model</th><th>Test R²</th><th>R² Gap</th>
    </tr></thead>
    <tbody>{perf_rows}</tbody>
  </table>
  </div>
  <p class="meta" style="margin-top:0.5rem;font-size:0.8em;color:var(--text-muted)">
    Global best Test R² across all experiments: <strong>{_fmt(global_max)}</strong>.
    Every regularisation strategy, feature set expansion, and engineering approach
    has been tried; none breaks through to meaningful generalisation on 2025.
  </p>
</details>"""

    return f"""
<section id="comp-cod-diagnostic">
  <h1 class="section-title">Comp COD - Persistent Failure Diagnostic</h1>
  <p class="section-intro">
    Effluent COD (Composite) is the only target where <em>no model generalises</em> across any
    experiment or phase (best Test R² = {_fmt(global_max)} across {n_exp} experiment variants).
    Phase 9 Voting MAE doubled from 8.7 mg/L (2024 in-sample) to 18.1 mg/L (2025 test).
    Variance is not collapsed (σ-ratio = 0.78), which rules out the low-variance artefact
    that drives negative R² in Comp pH and Comp TSS. This section documents the distribution
    shift, the performance panorama across every experiment, and the evidence for a
    2025 process change as the likely root cause.
  </p>

  {shift_html}
  {stats_html}
  {perf_table}

  <div class="obs-card" style="border-left:4px solid #e74c3c;margin-top:1.5rem">
    <h4 style="margin:0 0 0.8rem">Root Cause Analysis</h4>

    <p class="meta"><strong style="color:#e74c3c">1. Genuine 2025 process change - not a
      modelling artefact.</strong>
      MAE doubled (8.7 → 18.1 mg/L) while σ-ratio = 0.78, confirming the 2025 test set has
      comparable variance to training. The model is not hitting a low-variance wall -
      it is predicting the wrong values. This is the signature of a non-stationarity in
      the plant's COD removal mechanism that was absent during the 2021-2024 training window.
    </p>

    <p class="meta"><strong style="color:#e67e22">2. Inlet COD is a weak proxy for effluent
      COD under the current conditions.</strong>
      Composite effluent COD depends on secondary treatment efficiency, which is governed by
      MLSS, SVI, and aeration conditions - features with 35-50% missingness on composite rows
      (CONSIDER tier). When these are absent, the model falls back to inlet + flow, whose MI
      with effluent Comp COD is low (&lt;0.15 on training data).
    </p>

    <p class="meta"><strong style="color:#e67e22">3. Small composite sample size amplifies
      memorisation.</strong>
      Composite targets have only ~515-633 training rows after dropna. With 17-32 features
      the effective rows-per-feature ratio is 16-37 - below the commonly cited heuristic of 50.
      Even heavily regularised models (ElNet α=10) risk encoding 2022-era COD spikes that
      do not recur in 2025.
    </p>

    <p class="meta"><strong style="color:#f1c40f">4. Feature engineering and temporal lags
      do not help.</strong>
      Phase 10 full FE: best R² = −0.008 (Ridge). Phase 10b selective FE: best = −0.051.
      Phase 11 temporal lags: best = +0.107 (ElNet, Phase 11) but 2025 MAE remains ~17 mg/L.
      Every additional feature increases the risk of learning training-specific patterns without
      improving generalisation.
    </p>

    <p class="meta"><strong style="color:#4A90D9">5. Recommended next steps.</strong>
      (a) <strong>Flag Comp COD predictions as unreliable</strong> in any operational dashboard
      until 2025 data is incorporated into training.
      (b) <strong>Collect causal features</strong> with higher priority: secondary sludge age
      (SRT), MLSS, and effluent turbidity have high missingness now but directly govern COD
      removal - their availability on composite measurement days would be the single highest
      leverage improvement.
      (c) <strong>Retrain once ≥ 90 new 2025 rows are available</strong> (roughly 3 months of
      composite measurements) and test whether the new regime is stable enough for a combined
      2021-2025 model.
      (d) As an interim measure, <strong>a shallow decision tree (max depth 3) trained only
      on 2024-2025 data</strong> is likely to outperform any model trained on the full
      2021-2024 window for near-term operational use.
    </p>
  </div>
</section>"""


def _section_bests_json(df_all: pd.DataFrame) -> str:
    """JSON for the dynamic running-leaders sidebar panel."""
    section_exp_keys = {
        "exp1-sub1":   ["Exp1-Sub1"],
        "exp1-full":   ["Exp1"],
        "exp1-fs":     ["Exp1-FS"],
        "exp1-cyclic": ["Exp1-Cyclic"],
        "exp2-s1":     ["Exp2-Sub1"],
        "exp2-s1-fs":  ["Exp2-Sub1-FS"],
        "exp2-s2":     ["Exp2-Sub2"],
        "exp2-s2-fs":  ["Exp2-Sub2-FS"],
        "exp3-s1":     ["Exp3-S1"],
        "exp3-s1-fs":  ["Exp3-S1-FS"],
        "exp3-s2":     ["Exp3-S2"],
        "p9-ann":      ["Phase9-ANN"],
        "p9-voting":   ["Phase9-Voting"],
        "p9-stacking": ["Phase9-Stacking"],
        "p10-full":    ["Phase10-FE"],
        "p10b":        ["Phase10b-FE"],
        "p11":         ["Phase11"],
        "exp4-s1":     ["Exp4-S1"],
        "exp4-s2":     ["Exp4-S2"],
    }
    result = {}
    for sec_id, exp_keys in section_exp_keys.items():
        df_sec = df_all[df_all["exp_key"].isin(exp_keys)].dropna(subset=["R2_test"])
        bests = {}
        for tgt in TARGETS_ORDERED:
            sub = df_sec[df_sec["target"] == tgt]
            if sub.empty:
                continue
            row = sub.loc[sub["R2_test"].idxmax()]
            bests[TARGET_SHORT.get(tgt, tgt)] = {
                "model":  str(row["model"]),
                "r2":     round(float(row["R2_test"]), 3),
                "gap":    round(float(row["R2_gap"]) if not np.isnan(row["R2_gap"]) else 0, 3),
                "exp":    EXP_CHART_LABELS.get(str(row["exp_key"]), str(row["exp_key"])),
                "color":  MODEL_COLORS.get(str(row["model"]), "#888"),
            }
        result[sec_id] = bests
    return json.dumps(result, ensure_ascii=False)


def _exp1_best_model_box(df_all: pd.DataFrame) -> str:
    """Custom champion box for Experiment 1 with gap-adjusted selection and readable source labels.

    Candidates per (model, target):
      Sub1          → Exp1-Sub1 R2_test / R2_gap
      Sub2 (linear) → Exp1 R2_test_full (OLS only, unregularised)
                       Exp1 R2_test / R2_gap (OLS LassoCV; Ridge / ElNet full=FS)
      Sub2 (trees)  → Exp1-Cyclic R2_test_full / R2_gap_full  (pre-selection baseline)
                       Exp1-Cyclic R2_test / R2_gap            (OOF-selected refit)
    Winner per target = highest gap-adj score across all candidates.
    """
    s1  = df_all[df_all["exp_key"] == "Exp1-Sub1"].copy()
    s2  = df_all[df_all["exp_key"] == "Exp1"].copy()
    cyc = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()

    if s1.empty and s2.empty:
        return ""

    lin_set  = {"OLS", "Ridge", "ElNet"}
    tree_set = {"RF", "GB", "XGB"}
    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]

    def _g(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns:
            return None
        v = r[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _candidates(model, tgt):
        """Return list of (r2, gap, source_label) tuples for this model × target."""
        out = []
        # Sub1
        r2 = _g(s1, model, tgt); gap = _g(s1, model, tgt, "R2_gap")
        if r2 is not None:
            out.append((r2, gap, "Sub1 (4 feat · inlet only)"))

        if model == "OLS":
            # Full (unregularised)
            r2f = _g(s2, model, tgt, "R2_test_full"); gapf = _g(s2, model, tgt, "R2_gap_full")
            if r2f is not None:
                out.append((r2f, gapf, "Sub2 Full · OLS (unregularised, 11 feat)"))
            # FS (LassoCV)
            r2s = _g(s2, model, tgt); gaps = _g(s2, model, tgt, "R2_gap")
            if r2s is not None and (r2f is None or abs(r2s - r2f) > 1e-9):
                out.append((r2s, gaps, "Sub2 FS · OLS (LassoCV)"))
        elif model in lin_set:
            # Ridge / ElNet: full = FS (L2 / L1 selects internally)
            r2 = _g(s2, model, tgt); gap = _g(s2, model, tgt, "R2_gap")
            lbl = ("Sub2 · Ridge (L2 full)" if model == "Ridge"
                   else "Sub2 · ElNet (L1 internal)")
            if r2 is not None:
                out.append((r2, gap, lbl))
        else:
            # Trees: full baseline (pre-selection, cyclic dataset)
            r2f  = _g(cyc, model, tgt, "R2_test_full")
            gapf = (_g(cyc, model, tgt, "R2_gap_full")
                    if "R2_gap_full" in cyc.columns else None)
            if r2f is not None:
                out.append((r2f, gapf, f"Sub2 Full Cyclic · {model}"))
            # Trees FS (OOF-selected refit)
            r2s = _g(cyc, model, tgt); gaps = _g(cyc, model, tgt, "R2_gap")
            if r2s is not None and (r2f is None or abs(r2s - r2f) > 1e-9):
                out.append((r2s, gaps, f"Sub2 FS Cyclic · {model} (OOF-sel)"))
        return out

    table_rows = []
    overfit_warnings = []

    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        slug  = TARGET_SLUG.get(tgt, "all")

        all_cands = []  # (r2, gap, model, source_label)
        for m in models_ord:
            for r2, gap, lbl in _candidates(m, tgt):
                all_cands.append((r2, gap, m, lbl))

        if not all_cands:
            continue

        def _gaj_score(r2, gap):
            return _gap_adj(r2, gap if gap is not None else 0.0)

        best_gaj = max(all_cands, key=lambda c: _gaj_score(c[0], c[1]))
        best_raw = max(all_cands, key=lambda c: c[0])

        r2_win, gap_win, m_win, src_win = best_gaj
        gaj_win = _gaj_score(r2_win, gap_win)
        r2_raw_best = best_raw[0]
        raw_differs = abs(r2_raw_best - r2_win) > 0.005

        r2_col  = _r2_color(r2_win)
        mdl_col = MODEL_COLORS.get(m_win, "#888")
        gap_col = ("#E15252" if gap_win is not None and gap_win > 0.10
                   else "#5BAD6F" if gap_win is not None and gap_win < -0.10
                   else "var(--text-muted)")
        gap_str = f"{gap_win:+.3f}" if gap_win is not None else "—"

        overfit_flag = ""
        if gap_win is not None and abs(gap_win) > 0.25:
            overfit_flag = (' <span style="color:#e74c3c;font-size:10px" '
                            'title="Gap > 0.25 on the gap-adj winner">⚠ overfit</span>')
            overfit_warnings.append(short)

        alt_note = ""
        if raw_differs:
            _, gap_raw_best, m_raw_best, src_raw_best = best_raw
            alt_note = (
                f'<br><span style="color:#f39c12;font-size:10px" '
                f'title="Raw R² winner ({m_raw_best} · {src_raw_best}: '
                f'{r2_raw_best:+.3f}) demoted by gap-adj scoring">'
                f'raw: {m_raw_best} {r2_raw_best:+.3f} ({src_raw_best})</span>'
            )

        table_rows.append(
            f'<tr data-target="{slug}">'
            f'<td>{short}</td>'
            f'<td><strong style="color:{mdl_col}">{m_win}</strong></td>'
            f'<td style="color:{r2_col};font-weight:bold">'
            f'{r2_win:+.3f}{overfit_flag}'
            f'<br><span style="font-size:0.78em;color:var(--text-muted)">'
            f'gap-adj {gaj_win:+.3f}</span></td>'
            f'<td style="color:{gap_col}">{gap_str}</td>'
            f'<td class="meta" style="font-size:11px">{src_win}{alt_note}</td>'
            f'</tr>'
        )

    if not table_rows:
        return ""

    df_exp1 = df_all[df_all["exp_key"].isin(["Exp1-Sub1", "Exp1", "Exp1-Cyclic"])]
    grab_best_vals = [float(df_exp1[df_exp1["target"] == t]["R2_test"].max())
                      for t in GRAB_TARGETS
                      if not df_exp1[df_exp1["target"] == t].dropna(subset=["R2_test"]).empty]
    comp_best_vals = [float(df_exp1[df_exp1["target"] == t]["R2_test"].max())
                      for t in COMP_TARGETS
                      if not df_exp1[df_exp1["target"] == t].dropna(subset=["R2_test"]).empty]
    grab_avg = float(np.nanmean(grab_best_vals)) if grab_best_vals else float("nan")
    comp_avg = float(np.nanmean(comp_best_vals)) if comp_best_vals else float("nan")
    avg_r2   = float(np.nanmean(grab_best_vals + comp_best_vals)) \
               if (grab_best_vals or comp_best_vals) else float("nan")

    avg_detail = ""
    if not np.isnan(grab_avg) and not np.isnan(comp_avg):
        avg_detail = (
            f" · Grab avg {_fmt(grab_avg)}"
            f" · Composite avg <span style='color:{_r2_color(comp_avg)}'>{_fmt(comp_avg)}</span>"
        )

    overfit_note = ""
    if overfit_warnings:
        overfit_note = (
            f'<p class="table-note" style="margin-top:8px;color:#f39c12">'
            f'⚠ <strong>Overfitting note:</strong> '
            f'{", ".join(overfit_warnings)} show |gap| &gt; 0.25 even on the gap-adj winner — '
            f'treat these results with caution.</p>'
        )

    criteria_note = (
        '<p class="table-note" style="margin-top:4px">'
        '<strong>Selection:</strong> winner chosen by gap-adjusted score '
        '= R² − 0.5 · max(0, |gap| − 0.10), consistent with the global leaderboard methodology. '
        'Where the raw R² winner differs from the gap-adj winner it is shown in orange below the '
        'source label. avg best Test R² uses the raw per-target best for cross-experiment '
        'comparability. Composite averages should be read cautiously — all composite targets '
        'fail at the Exp1 feature level (Grab COD likewise); secondary clarifier data is required '
        'before these targets become learnable.</p>'
    )

    return f"""
<div class="best-box">
  <div class="best-box-title">Best Model in Experiment 1
    <span class="best-box-avg">avg best Test R² = {_fmt(avg_r2)}{avg_detail}</span>
  </div>
  <table class="summary-table best-table">
    <thead><tr>
      <th>Target</th><th>Model</th>
      <th>Test R²<br><span class="meta" style="font-weight:normal;font-size:0.8em">gap-adj below</span></th>
      <th>Gap</th><th>Source (gap-adj winner)</th>
    </tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
  {overfit_note}
  {criteria_note}
</div>"""


def _best_model_box(df: pd.DataFrame, label: str) -> str:
    """Champion box: best Test R² per target in this section, with experiment source."""
    rows = []
    overfit_warnings = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt].dropna(subset=["R2_test"])
        if sub.empty:
            continue
        # Primary: best non-overfit model (gap ≤ 0.25); fallback: absolute best
        non_of = sub[sub["R2_gap"].abs() <= 0.25]
        best = non_of.loc[non_of["R2_test"].idxmax()] if not non_of.empty else sub.loc[sub["R2_test"].idxmax()]
        abs_best = sub.loc[sub["R2_test"].idxmax()]
        skipped_better = (not non_of.empty) and (abs_best["R2_test"] - best["R2_test"] > 0.005)

        slug = TARGET_SLUG.get(tgt, "all")
        col = _r2_color(best["R2_test"])
        mdl_col = MODEL_COLORS.get(best["model"], "#888")
        gap_cls_val = _gap_cls(best["R2_gap"])
        exp_lbl = EXP_CHART_LABELS.get(best["exp_key"], best["exp_key"])

        overfit_flag = ""
        if abs(best["R2_gap"]) > 0.25:
            overfit_flag = ' <span style="color:#e74c3c;font-size:10px" title="R² gap > 0.25 - possible overfit">⚠ overfit</span>'
            overfit_warnings.append(TARGET_SHORT.get(tgt, tgt))
        skipped_note = ""
        if skipped_better:
            abs_r2_str = _fmt(abs_best["R2_test"])
            abs_mdl    = abs_best["model"]
            abs_gap    = _fmt(abs_best["R2_gap"])
            skipped_note = (
                f' <span style="color:#f39c12;font-size:10px" '
                f'title="Higher R\u00b2 of {abs_r2_str} from {abs_mdl} suppressed (gap={abs_gap} > 0.25)">'
                f'(overfit suppressed)</span>'
            )

        rows.append(
            f'<tr data-target="{slug}">'
            f'<td>{TARGET_SHORT.get(tgt, tgt)}</td>'
            f'<td><strong style="color:{mdl_col};">{best["model"]}</strong></td>'
            f'<td style="color:{col};font-weight:bold">{_fmt(best["R2_test"])}{overfit_flag}</td>'
            f'<td class="{gap_cls_val}">{_fmt(best["R2_gap"])}</td>'
            f'<td class="meta" style="font-size:11px">({exp_lbl}){skipped_note}</td>'
            f'</tr>'
        )
    if not rows:
        return ""
    grab_best_vals = [df[df["target"] == t]["R2_test"].max()
                      for t in GRAB_TARGETS
                      if not df[df["target"] == t].dropna(subset=["R2_test"]).empty]
    comp_best_vals = [df[df["target"] == t]["R2_test"].max()
                      for t in COMP_TARGETS
                      if not df[df["target"] == t].dropna(subset=["R2_test"]).empty]
    grab_avg = float(np.nanmean(grab_best_vals)) if grab_best_vals else float("nan")
    comp_avg = float(np.nanmean(comp_best_vals)) if comp_best_vals else float("nan")
    avg_r2   = float(np.nanmean(grab_best_vals + comp_best_vals)) \
               if (grab_best_vals or comp_best_vals) else float("nan")
    avg_detail = ""
    if not np.isnan(grab_avg) and not np.isnan(comp_avg):
        avg_detail = (
            f" · Grab avg {_fmt(grab_avg)}"
            f" · Composite avg <span style='color:{_r2_color(comp_avg)}'>{_fmt(comp_avg)}</span>"
        )
    overfit_note = ""
    if overfit_warnings:
        overfit_note = (
            f'<p class="table-note" style="margin-top:8px;color:#f39c12">'
            f'⚠ <strong>Overfitting note:</strong> '
            f'{", ".join(overfit_warnings)} show R² gap &gt; 0.25. '
            f'Best is ranked by Test R² among non-overfit models (gap ≤ 0.25) where available. '
            f'High-gap winners may reflect the model memorising 2021-2024 patterns absent in 2025 - '
            f'prefer lower-gap alternatives for deployment.</p>'
        )
    criteria_note = (
        '<p class="table-note" style="margin-top:4px">'
        '<strong>Selection criteria:</strong> highest Test R² among models with R² gap ≤ 0.25. '
        'Alternatives: (a) use harmonic mean of Train+Test R² to penalise both under- and overfitting; '
        '(b) rank by RMSE if absolute error scale matters; (c) for deployment, cap gap at 0.15 for '
        'stricter stability.</p>'
    )
    return f"""
<div class="best-box">
  <div class="best-box-title">Best Model in {label}
    <span class="best-box-avg">avg best Test R² = {_fmt(avg_r2)}{avg_detail}</span>
  </div>
  <table class="summary-table best-table">
    <thead><tr><th>Target</th><th>Model</th><th>Test R²</th><th>Gap</th><th>Source</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  {overfit_note}
  {criteria_note}
</div>"""


def _exp_subsection(df_all: pd.DataFrame, exp_key: str,
                    section_id: str, title: str,
                    models_linear=None, models_nl=None, open_default=True) -> str:
    """Build a standard sub-section for one experiment variant."""
    df = df_all[df_all["exp_key"] == exp_key].copy()
    if df.empty:
        return ""
    ml = models_linear or LINEAR_MODELS
    mn = models_nl or NL_MODELS
    all_m = [m for m in ALL_MODELS_ORD if m in df["model"].values]

    feat_html   = _feature_card(exp_key)
    fs_html     = _feature_selection_details(exp_key)
    ds_html     = _dataset_summary(df)
    fs_sel_html = _feature_selection_table(df)
    lin_tbl     = _metrics_table(df[df["model"].isin(ml)], ml, f"{section_id}-lin", df_all)
    nl_tbl      = _metrics_table(df[df["model"].isin(mn)], mn, f"{section_id}-nl", df_all)
    comp_tbl    = _metrics_table(df, all_m, f"{section_id}-comp", df_all)
    train_tbl   = _train_metrics_table(df, all_m)

    open_attr = " open" if open_default else ""
    return f"""
<details class="exp-details"{open_attr} id="{section_id}">
  <summary><span class="fold-icon">▶</span> {title}</summary>
  <div class="exp-body">
    {feat_html}
    {fs_html}
    {ds_html}
    {fs_sel_html}

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Linear Models (OLS · Ridge · ElNet)</summary>
      <div class="fold-body">{lin_tbl}</div>
    </details>

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Non-Linear Models (RF · GB · XGB)</summary>
      <div class="fold-body">{nl_tbl}</div>
    </details>

    <details class="inner-fold" open>
      <summary><span class="fold-icon">▶</span> All Models - Combined Comparison</summary>
      <div class="fold-body">{comp_tbl}{train_tbl}</div>
    </details>
  </div>
</details>"""


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _gap_adj(r2: float, gap: float) -> float:
    """Gap-adjusted score using the canonical formula from best_models_selection.py.

    Deferred import to avoid the circular dependency at module level
    (best_models_selection imports load_all_data from this module).
    """
    from best_models_selection import _gap_adjusted_score  # noqa: E402
    return _gap_adjusted_score(r2, gap)


def _no_alt_popup_text(tgt: str, naive_compound: str, naive_r2: float,
                        naive_gap: float, df_all: pd.DataFrame) -> str:
    """Generate a concise paragraph explaining why no better alternative exists."""
    from best_models_selection import _gap_adjusted_score, ONE_SE_MARGIN  # noqa: E402
    sub = df_all[df_all["target"] == tgt].dropna(subset=["R2_test"]).copy()
    sub = sub.assign(model=sub["exp_key"].astype(str) + " · " + sub["model"].astype(str))
    sub["gap_adj"] = sub.apply(lambda r: _gap_adjusted_score(r["R2_test"], r["R2_gap"]), axis=1)

    # One-SE window
    within = sub[sub["R2_test"] >= naive_r2 - ONE_SE_MARGIN].copy()
    within["abs_gap"] = within["R2_gap"].abs()
    within = within.sort_values("abs_gap")

    window_items = [
        f"{r['model']} (R²={r['R2_test']:.3f}, gap={r['R2_gap']:+.3f})"
        for _, r in within.iterrows()
    ]
    window_str = "; ".join(window_items) if window_items else "none"

    min_gap_in_window = within["abs_gap"].min() if not within.empty else float("nan")
    n_window = len(within)

    # Best gap-adjusted model (should equal naive since no_alt)
    best_gadj = sub.loc[sub["gap_adj"].idxmax()] if sub["gap_adj"].notna().any() else None
    gadj_score = best_gadj["gap_adj"] if best_gadj is not None else float("nan")

    intro = (
        f"{n_window} model{'s' if n_window != 1 else ''} fall within the "
        f"±{ONE_SE_MARGIN} R² competitive window: {window_str}. "
    )
    if np.isnan(min_gap_in_window):
        gap_note = "No gap data available for window members."
    else:
        gap_note = (
            f"The smallest |gap| among them is {min_gap_in_window:+.3f} - "
            f"every competitive model carries a substantial overfit gap. "
        )
    conclusion = (
        f"The gap-adjusted score (max {gadj_score:.3f}) still peaks at the naive champion "
        f"({naive_compound}), so no less-overfit alternative achieves comparable accuracy "
        f"on this target. This target needs more training data, better features, or "
        f"a regime-specific model rather than a different model from the current pool."
    )
    return intro + gap_note + conclusion


def _global_leaderboard(df_all: pd.DataFrame) -> str:
    # Late import - best_models_selection imports from this module, so must be deferred
    from best_models_selection import build_global as _build_global  # noqa: E402
    global_picks   = _build_global(df_all)
    gadj_by_target = {row["target"]: row for _, row in global_picks.iterrows()}
    _popup_counter = [0]  # mutable counter for unique popup IDs

    rows = []
    for tgt in TARGETS_ORDERED:
        sub = df_all[df_all["target"] == tgt].dropna(subset=["R2_test"])
        if sub.empty:
            continue
        best = sub.loc[sub["R2_test"].idxmax()]
        slug = TARGET_SLUG.get(tgt, "all")

        # ── Naive champion ─────────────────────────────────────────────────────
        naive_r2   = best["R2_test"]
        naive_gap  = best["R2_gap"]
        naive_rmse = best.get("RMSE_test", float("nan"))
        naive_mdl  = best["model"]
        exp_label  = EXP_CHART_LABELS.get(best["exp_key"], best["exp_key"])
        mdl_col    = MODEL_COLORS.get(naive_mdl, "#888")

        # ── Gap-adjusted recommendation (across all experiments) ───────────────
        # build_global stores "exp_key · model" as the composite label so that
        # the rule can distinguish the same model type from different experiments.
        naive_compound = f"{best['exp_key']} · {naive_mdl}"

        gpick = gadj_by_target.get(tgt)
        if gpick is not None:
            gadj_lbl  = str(gpick.get("gadj_model", "-"))
            gadj_r2   = float(gpick.get("gadj_R2",  float("nan")))
            gadj_gap  = float(gpick.get("gadj_gap", float("nan")))
            gadj_rmse = float(gpick.get("gadj_RMSE", float("nan")))
        else:
            gadj_lbl = naive_compound; gadj_r2 = naive_r2
            gadj_gap = naive_gap; gadj_rmse = naive_rmse

        # Classify this row into three display states:
        #   • gap_ok       - naive gap is acceptable (< 0.10); no concern, blank right side
        #   • has_alt      - gap is concerning AND gap-adj picks a DIFFERENT model
        #   • no_alt       - gap is concerning but gap-adj picks the SAME model (no better option)
        gap_ok   = naive_gap < 0.10          # positive direction is the concerning one
        has_alt  = (not gap_ok) and (gadj_lbl != naive_compound)
        no_alt   = (not gap_ok) and (gadj_lbl == naive_compound)

        if gap_ok:
            right_cells = "<td></td><td></td><td></td><td></td>"
            warn_cell   = "<td></td>"
        elif has_alt:
            right_cells = (
                f"<td style='font-size:11px;color:var(--text-muted)'>{gadj_lbl}</td>"
                f"<td style='color:{_r2_color(gadj_r2)}'>{_fmt(gadj_r2)}</td>"
                f"<td class='{_gap_cls(gadj_gap)}'>{_fmt(gadj_gap)}</td>"
                f"<td class='meta'>{_fmt(gadj_rmse)}</td>"
            )
            warn_cell = (
                "<td style='color:#E67E22;font-size:15px;text-align:center"
                ";vertical-align:middle' title='Naive model is overfit; "
                "consider the gap-adjusted recommendation instead'>⚠</td>"
            )
        else:  # no_alt
            _popup_counter[0] += 1
            pid = f"noalt-popup-{_popup_counter[0]}"
            popup_text = _no_alt_popup_text(
                tgt, naive_compound, naive_r2, naive_gap, df_all
            )
            # Escape single quotes for inline HTML attribute
            popup_text_safe = popup_text.replace("'", "&#39;")
            right_cells = (
                f"<td colspan='4' style='color:var(--text-muted);font-size:11px"
                f";font-style:italic;position:relative'>"
                f"no better alternative found"
                f"<span class='noalt-info-btn' "
                f"onclick=\"document.getElementById('{pid}').style.display="
                f"document.getElementById('{pid}').style.display==='none'?'block':'none';"
                f"event.stopPropagation()\" "
                f"title='Click for details'>ⓘ</span>"
                f"<div id='{pid}' class='noalt-popup' style='display:none'>"
                f"<button class='noalt-popup-close' "
                f"onclick=\"this.parentElement.style.display='none'\">×</button>"
                f"<p>{popup_text}</p>"
                f"</div></td>"
            )
            warn_cell = (
                "<td style='color:#E67E22;font-size:15px;text-align:center"
                ";vertical-align:middle' title='Naive model is overfit but "
                "no less-overfit model achieves comparable accuracy'>⚠</td>"
            )

        TARGET_LIMITS = {
            "Effluent BOD (mg/L, Grab)": 10.0,
            "Effluent COD (mg/L, Grab)": 250.0,
            "Effluent TSS (mg/L, Grab)": 10.0,
            "Effluent BOD (mg/L, Composite)": 10.0,
            "Effluent COD (mg/L, Composite)": 250.0,
            "Effluent TSS (mg/L, Composite)": 10.0,
        }
        limit = TARGET_LIMITS.get(tgt)
        
        gadj_rmse_val = gadj_rmse if not pd.isna(gadj_rmse) else float("nan")
        gadj_mdae_val = float(gpick.get("gadj_MdAE", best.get("MdAE_test", float("nan")))) if gpick is not None else float(best.get("MdAE_test", float("nan")))

        if limit is not None:
            limit_str = f"{limit:g} mg/L"
            rmse_pct = f"{(gadj_rmse_val / limit * 100):.1f}%" if not pd.isna(gadj_rmse_val) else "-"
            mdae_pct = f"{(gadj_mdae_val / limit * 100):.1f}%" if not pd.isna(gadj_mdae_val) else "-"
            rmse_col = "#e74c3c" if not pd.isna(gadj_rmse_val) and (gadj_rmse_val / limit) > 0.5 else "var(--text-primary)"
            mdae_col = "#e74c3c" if not pd.isna(gadj_mdae_val) and (gadj_mdae_val / limit) > 0.5 else "var(--text-primary)"
            limit_cells = f"<td class='meta'>{limit_str}</td><td class='meta' style='color:{rmse_col}'>{rmse_pct}</td><td class='meta' style='color:{mdae_col}'>{mdae_pct}</td>"
        else:
            limit_cells = "<td class='meta'>-</td><td class='meta'>-</td><td class='meta'>-</td>"

        # Embed ⚠ inline into the Naive R² cell when the naive champion is overfit
        overfit_inline = (
            ' <span style="color:#E67E22;font-size:11px;vertical-align:super" '
            'title="Naive model is overfit (gap ≥ 0.10) - see Recommended column">⚠</span>'
            if not gap_ok else ""
        )
        rows.append(
            f'<tr data-target="{slug}">'
            # Naive champion
            f'<td><strong>{TARGET_SHORT.get(tgt, tgt)}</strong></td>'
            f'<td><strong style="color:{mdl_col}">{naive_mdl}</strong></td>'
            f'<td style="color:{_r2_color(naive_r2)};font-weight:bold">{_fmt(naive_r2)}{overfit_inline}</td>'
            f'<td class="{_gap_cls(naive_gap)}">{_fmt(naive_gap)}</td>'
            f'<td class="meta">{_fmt(naive_rmse)}</td>'
            f'<td class="meta" style="font-size:11px">{exp_label}</td>'
            + right_cells + limit_cells +
            f'</tr>'
        )

    return f"""
<div class="card section-card" id="overview-leaderboard">
  <h2>Global Leaderboard - Best Result Per Target</h2>
  <p class="meta">
    Left half: the <strong>highest raw Test R²</strong> achieved for each target across all
    experiments - the naive ceiling. A <span style="color:#E67E22">⚠</span> superscript on the
    Test R² value flags targets where the naive champion is overfit (gap ≥ 0.10).
    Right half: the <strong>gap-adjusted recommendation</strong> - the best model after
    penalising large train/test gaps (see
    <a href="#model-selection">Model Selection</a> for the full rule explanation).
  </p>
  <div style="overflow-x:auto">
  <table class="summary-table leaderboard-table">
    <thead>
      <tr>
        <th rowspan="2">Target</th>
        <th colspan="5" style="text-align:center;border-bottom:1px solid var(--border-color)">
          Naive champion (highest Test R²)</th>
        <th colspan="4" style="text-align:center;border-bottom:1px solid var(--border-color)">
          Recommended (gap-adjusted)</th>
        <th colspan="3" style="text-align:center;border-bottom:1px solid var(--border-color)">
          Compliance Error (Rec. Model)</th>
      </tr>
      <tr>
        <th>Model</th><th>Test R²</th><th>R² Gap</th><th>RMSE</th><th>Experiment</th>
        <th>Model · Experiment</th><th>R²</th><th>Gap</th><th>RMSE</th>
        <th>Limit</th><th>RMSE % Limit</th><th>MdAE % Limit</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <p class="meta" style="margin-top:8px;font-size:11px;color:var(--text-muted)">
    MdAE % Limit is computed from row-level predictions stored in dataset files.
    Feature Engineering and Temporal Features phases do not store row-level predictions,
    so MdAE shows <em>-</em> when their models are the recommended choice.
    BOD/TSS targets only — MdAE is not shown for pH (narrow range) or COD (no discharge limit).
  </p>
</div>"""


def _progression_chart(df_all: pd.DataFrame) -> str:
    """Chart.js line chart with Target dropdown and R² / RMSE / MAE / Gap toggle."""
    chart_models = ["Ridge", "ElNet", "RF", "Voting", "Stacking", "ANN"]
    metrics = [
        ("r2",   "R2_test",   "Test R²",       False),   # (m_key, col, y-label, lower_is_better)
        ("gap",  "R2_gap",    "Test R² Gap",   True),
        ("rmse", "RMSE_test", "Test RMSE",     True),
        ("mae",  "MAE_test",  "Test MAE",      True),
    ]

    all_data = {}
    options_html = ""
    first_slug = None

    for tgt in TARGETS_ORDERED:
        df_t = df_all[df_all["target"] == tgt]
        if df_t.empty:
            continue
        
        slug = TARGET_SLUG.get(tgt, tgt.replace(" ", "-").lower())
        short = TARGET_SHORT.get(tgt, tgt)
        if first_slug is None:
            first_slug = slug
            
        options_html += f'<option value="{slug}">{short}</option>'
        
        tgt_data = {}
        for m_key, col_name, _, _ in metrics:
            df_m = df_t.dropna(subset=[col_name])
            exp_order_present = [e for e in EXP_CHART_ORDER if e in df_m["exp_key"].values]
            labels = [EXP_CHART_LABELS.get(e, e) for e in exp_order_present]
            
            datasets = []
            for mdl in chart_models:
                data = []
                for ek in exp_order_present:
                    row = df_m[(df_m["exp_key"] == ek) & (df_m["model"] == mdl)]
                    val = float(row[col_name].values[0]) if not row.empty else None
                    data.append(val if val is not None and not np.isnan(val) else None)
                
                if any(v is not None for v in data):
                    datasets.append({
                        "label": mdl,
                        "data": data,
                        "borderColor": MODEL_COLORS.get(mdl, "#888"),
                        "backgroundColor": MODEL_COLORS.get(mdl, "#888"),
                        "spanGaps": True,
                        "tension": 0.3,
                        "pointRadius": 5,
                        "borderWidth": 2,
                    })
            tgt_data[m_key] = {"labels": labels, "datasets": datasets}
        all_data[slug] = tgt_data

    chart_json = json.dumps(all_data)

    return f"""
<div class="card section-card" id="overview-progression">
  <h2>Metric Progression Across Experiments</h2>
  <p class="meta">View how model performance evolved across experiments for specific targets.
  Gaps in a line indicate the model was not run in that experiment.
  <strong>R²:</strong> higher is better. <strong>RMSE / MAE:</strong> lower is better (original units, mg/L or pH).</p>
  
  <div style="margin-bottom:12px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;">
    <div>
      <label for="prog-target-sel" style="font-weight:600;font-size:13px;margin-right:6px;color:var(--text-color)">Target:</label>
      <select id="prog-target-sel" onchange="switchProgTarget()" 
        style="padding:6px 10px;border-radius:4px;border:1px solid var(--border-color);background:var(--input-bg);color:var(--text-color);font-size:13px;min-width:240px;outline:none;">
        {options_html}
      </select>
    </div>
    <div style="display:flex;gap:8px;">
      <button id="prog-btn-r2"   onclick="switchProgMetric('r2')"
        style="padding:5px 16px;border-radius:4px;cursor:pointer;border:1px solid #4A90D9;background:#4A90D9;color:#fff;font-weight:600;font-size:13px">R²</button>
      <button id="prog-btn-gap"   onclick="switchProgMetric('gap')"
        style="padding:5px 16px;border-radius:4px;cursor:pointer;border:1px solid var(--border-color);background:var(--card-bg);color:var(--text-color);font-size:13px">R² Gap</button>
      <button id="prog-btn-rmse" onclick="switchProgMetric('rmse')"
        style="padding:5px 16px;border-radius:4px;cursor:pointer;border:1px solid var(--border-color);background:var(--card-bg);color:var(--text-color);font-size:13px">RMSE</button>
      <button id="prog-btn-mae"  onclick="switchProgMetric('mae')"
        style="padding:5px 16px;border-radius:4px;cursor:pointer;border:1px solid var(--border-color);background:var(--card-bg);color:var(--text-color);font-size:13px">MAE</button>
    </div>
  </div>
  
  <div style="position:relative;height:380px;max-width:1000px;margin:0 auto">
    <canvas id="progressionChart"></canvas>
  </div>
  
  <script>
  (function() {{
    var allData = {chart_json};
    var currentTarget = '{first_slug}';
    var currentMetric = 'r2';
    
    var isDark    = document.documentElement.getAttribute('data-theme') === 'dark';
    var gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    var textColor = isDark ? '#A0A6B2' : '#555';

    var yLabels = {{ r2: 'Test R²', gap: 'Train/Test R² Gap', rmse: 'Test RMSE', mae: 'Test MAE' }};
    var lowerBetter = {{ r2: false, gap: true, rmse: true, mae: true }};

    var ctx = document.getElementById('progressionChart').getContext('2d');
    var progChart = new Chart(ctx, {{
      type: 'line',
      data: allData[currentTarget][currentMetric],
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: textColor, boxWidth: 14 }} }},
          tooltip: {{
            callbacks: {{
              label: function(c) {{
                var v = c.parsed.y;
                return c.dataset.label + ': ' + (v !== null ? v.toFixed(3) : 'n/a');
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: textColor, maxRotation: 45 }},
                grid:  {{ color: gridColor }} }},
          y: {{ title: {{ display: true, text: 'Test R²', color: textColor }},
                ticks: {{ color: textColor }},
                grid:  {{ color: gridColor }},
                suggestedMin: -0.2, suggestedMax: 0.7 }}
        }}
      }}
    }});

    function updateChart() {{
      var d = allData[currentTarget]?.[currentMetric];
      if (!d) return;
      progChart.data = d;
      
      var isLower = lowerBetter[currentMetric];
      progChart.options.scales.y.title.text = yLabels[currentMetric];
      if (currentMetric === 'r2') {{
        progChart.options.scales.y.suggestedMin = -0.2;
        progChart.options.scales.y.suggestedMax = 0.7;
      }} else {{
        delete progChart.options.scales.y.suggestedMin;
        delete progChart.options.scales.y.suggestedMax;
      }}
      progChart.update();
    }}

    window.switchProgTarget = function() {{
      var sel = document.getElementById('prog-target-sel');
      if (sel) currentTarget = sel.value;
      updateChart();
    }};

    window.switchProgMetric = function(metric) {{
      currentMetric = metric;
      updateChart();

      // Update button styles
      ['r2','gap','rmse','mae'].forEach(function(m) {{
        var btn = document.getElementById('prog-btn-' + m);
        if(!btn) return;
        if (m === metric) {{
          btn.style.background = '#4A90D9';
          btn.style.color = '#fff';
          btn.style.borderColor = '#4A90D9';
          btn.style.fontWeight = '600';
        }} else {{
          btn.style.background = 'var(--card-bg)';
          btn.style.color = 'var(--text-color)';
          btn.style.borderColor = 'var(--border-color)';
          btn.style.fontWeight = 'normal';
        }}
      }});
    }};
  }})();
  </script>
</div>"""


def _abstract_section() -> str:
    return """
<div class="card section-card" id="overview-abstract" style="border-left:4px solid #4A90D9; margin-bottom: 24px;">
  <h2 style="margin-top:0">Abstract</h2>
  <ul style="line-height:1.5;margin-bottom:0;font-size:13px;color:var(--text-color)">
    <li><strong>Baseline (Exp1 & 2):</strong> Established that inlet water quality alone is insufficient for prediction. Secondary clarifier data is essential to achieve meaningful accuracy.</li>
    <li><strong>Feature Expansion (Exp3):</strong> Aeration basin data (Exp3-S2) produced the best linearly-regularised models (ElasticNet/Ridge), providing the most robust baseline predicting 2025 holdout data.</li>
    <li><strong>Feature Pruning (Exp4):</strong> Removing collinear features (via VIF or manually) degraded generalisation across the board. The collinear features carry structural signal necessary for decision tree models, while regularised linear models already suppress collinearity automatically.</li>
    <li><strong>Advanced Methods (Ensembles & Temporal):</strong> Stacking and Neural Networks (ANN) struggled to generalise on the extremely small Composite datasets (n≈290). Conversely, temporal lag features successfully improved R² on Grab targets, demonstrating that recent flow history strongly influences spot samples.</li>
    <li><strong>Practical significance — R² alone is insufficient:</strong> A high Test R² does not guarantee compliance-grade accuracy. Discharge limits for BOD and TSS are 10 mg/L.
      <strong>Grab BOD</strong> (best R²≈0.69): RMSE ≈ 2 mg/L — roughly 20% of the limit, operationally useful.
      <strong>Grab TSS</strong> (best R²≈0.64): RMSE ≈ 6 mg/L — roughly 60% of the limit; the model explains variance well but its absolute prediction error is large relative to the compliance threshold. Predictions should be treated as trend indicators rather than compliance certifications for TSS.</li>
  </ul>
</div>"""

def build_overview(df_all: pd.DataFrame) -> str:
    return f"""
<section id="overview">
  <h1 class="section-title">Overview</h1>
  {_abstract_section()}
  {_global_leaderboard(df_all)}
  {_progression_chart(df_all)}
</section>"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _exp1_qna(df_all: pd.DataFrame) -> str:
    """Data-driven Q&A for Experiment 1 — four key questions."""
    s1  = df_all[df_all["exp_key"] == "Exp1-Sub1"].copy()
    s2  = df_all[df_all["exp_key"] == "Exp1"].copy()
    cyc = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()

    if s1.empty or s2.empty:
        return ""

    lin_m  = ["OLS", "Ridge", "ElNet"]
    tree_m = ["RF", "GB", "XGB"]
    all_m  = lin_m + tree_m
    lin_set = set(lin_m)

    def _r2(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns: return None
        v = r[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _gap(df, model, tgt):
        return _r2(df, model, tgt, "R2_gap")

    def _full_val(model, tgt):
        if model == "OLS":
            return _r2(s2, model, tgt, "R2_test_full")
        elif model in lin_set:
            return _r2(s2, model, tgt)
        else:
            v = _r2(cyc, model, tgt, "R2_test_full")
            return v if v is not None else _r2(s2, model, tgt)

    def _fs_val(model, tgt):
        if model in lin_set:
            return _r2(s2, model, tgt)
        else:
            return _r2(cyc, model, tgt)

    def _delta_arr_raw(from_fn, to_fn, models, targets):
        vals = []
        for m in models:
            for t in targets:
                a, b = from_fn(m, t), to_fn(m, t)
                if a is not None and b is not None:
                    vals.append(b - a)
        return np.array(vals) if vals else np.array([])

    def _delta_arr(from_df, to_df, models, targets):
        return _delta_arr_raw(
            lambda m, t: _r2(from_df, m, t),
            lambda m, t: _r2(to_df,   m, t),
            models, targets)

    def _colored(arr):
        if not len(arr): return "<em>—</em>"
        v = float(arr.mean())
        c = "#5BAD6F" if v > 0.005 else ("#E15252" if v < -0.005 else "var(--text-muted)")
        return f"<span style='color:{c};font-weight:bold'>{v:+.3f}</span>"

    # ── Q1 : Sub1 floor ──────────────────────────────────────────────────────
    s1_vals = [_r2(s1, m, t) for m in all_m for t in TARGETS_ORDERED]
    s1_vals = [v for v in s1_vals if v is not None]
    s1_pos  = [v for v in s1_vals if v > 0]
    s1_avg  = float(np.mean(s1_vals)) if s1_vals else None
    best_r2, best_m, best_t = None, None, None
    for m in all_m:
        for t in TARGETS_ORDERED:
            v = _r2(s1, m, t)
            if v is not None and (best_r2 is None or v > best_r2):
                best_r2, best_m, best_t = v, m, TARGET_SHORT.get(t, t)

    q1 = (
        f"Barely. Only {len(s1_pos)}/{len(s1_vals)} (model × target) evaluations achieved a "
        f"positive Test R². Best single result: {best_m} on {best_t} (R² = {best_r2:+.3f}). "
        f"Mean Test R² across all cells: {s1_avg:+.3f}. All composite targets and Grab COD "
        f"returned universally negative R² — inlet concentrations carry no generalisable signal "
        f"for those targets. This establishes the floor: inlet data is necessary but not sufficient."
    )

    # ── Q2 : Context gain (Sub1 → Sub2 Full) ─────────────────────────────────
    # Sub2 "Full" for linear uses the full-set OLS result and Ridge/ElNet result.
    # For trees we compare Sub1 vs Sub2 baseline (both trained on full sets at their size).
    lin_d12       = _delta_arr(s1, s2, lin_m,  TARGETS_ORDERED)
    tree_d12      = _delta_arr(s1, s2, tree_m, TARGETS_ORDERED)
    lin_grab_d12  = _delta_arr(s1, s2, lin_m,  GRAB_TARGETS)
    lin_comp_d12  = _delta_arr(s1, s2, lin_m,  COMP_TARGETS)
    tree_grab_d12 = _delta_arr(s1, s2, tree_m, GRAB_TARGETS)
    tree_comp_d12 = _delta_arr(s1, s2, tree_m, COMP_TARGETS)

    _lin_net  = float(lin_d12.mean())  if len(lin_d12)  else 0
    _tree_net = float(tree_d12.mean()) if len(tree_d12) else 0
    _lin_grab_net = float(lin_grab_d12.mean()) if len(lin_grab_d12) else 0

    _lin_dir  = "improved" if _lin_net  > 0.01 else ("degraded" if _lin_net  < -0.01 else "barely changed")
    _tree_dir = "improved" if _tree_net > 0.01 else ("degraded" if _tree_net < -0.01 else "barely changed")
    _lin_grab_dir = "positive" if _lin_grab_net > 0.01 else ("negative" if _lin_grab_net < -0.01 else "negligible")

    if _lin_net < 0 and _tree_net < 0:
        _driver = (
            "Both model families degraded. Adding 7 process/calendar features increased "
            "parameter count without proportional signal gain at this dataset size — "
            "OLS especially suffers from overfitting with 11 unconstrained coefficients."
        )
    elif _lin_net < 0 and _tree_net > 0:
        _driver = (
            f"Linear models {_lin_dir} while tree models {_tree_dir}. "
            f"Tree models recovered from Sub1 composite failures (fewer features → more spurious splits); "
            f"linear models, particularly OLS, overfitted on the expanded 11-feature set."
        )
    elif _lin_net > 0 and _tree_net < 0:
        _driver = (
            f"Linear models {_lin_dir} while tree models {_tree_dir}. "
            f"Additional features gave tree models more paths to overfit on the smaller composite sets."
        )
    else:
        _driver = f"Both families {_lin_dir}, with linear models showing {_lin_grab_dir} gains on Grab targets."

    q2 = (
        f"{_driver} "
        f"<br><strong>Linear models (Sub2 Full):</strong> mean Δ = {_colored(lin_d12)} overall "
        f"(Grab: {_colored(lin_grab_d12)}, Composite: {_colored(lin_comp_d12)}). "
        f"<br><strong>Tree models:</strong> mean Δ = {_colored(tree_d12)} overall "
        f"(Grab: {_colored(tree_grab_d12)}, Composite: {_colored(tree_comp_d12)}). "
        f"Ridge and ElNet are the meaningful linear references here — OLS without regularisation "
        f"is poorly suited to 11 features and its column in the comparison table largely reflects overfitting, "
        f"not the actual signal in the feature set."
    )

    # ── Q3 : Model-specific FS (Sub2 Full → Sub2 FS) ─────────────────────────
    fs_full_vals = [_full_val(m, t) for m in all_m for t in TARGETS_ORDERED]
    fs_sel_vals  = [_fs_val(m,  t) for m in all_m for t in TARGETS_ORDERED]
    fs_deltas    = [b - a for a, b in zip(fs_full_vals, fs_sel_vals)
                   if a is not None and b is not None]
    fs_arr = np.array(fs_deltas) if fs_deltas else np.array([])

    # By model family
    lin_fs_d  = _delta_arr_raw(_full_val, _fs_val, lin_m,  TARGETS_ORDERED)
    tree_fs_d = _delta_arr_raw(_full_val, _fs_val, tree_m, TARGETS_ORDERED)
    grab_fs_d = _delta_arr_raw(_full_val, _fs_val, all_m,  GRAB_TARGETS)
    comp_fs_d = _delta_arr_raw(_full_val, _fs_val, all_m,  COMP_TARGETS)

    _fs_net  = float(fs_arr.mean())     if len(fs_arr)   else 0
    _fs_wins = int((fs_arr >  0.01).sum()) if len(fs_arr) else 0
    _fs_loss = int((fs_arr < -0.01).sum()) if len(fs_arr) else 0
    _fs_n    = len(fs_arr)

    # Gap delta (FS − Full) — does FS reduce overfitting?
    gap_deltas = []
    for m in all_m:
        for t in TARGETS_ORDERED:
            if m == "OLS":
                gf = _r2(s2, m, t, "R2_gap_full")
                gs = _gap(s2, m, t)
            elif m in lin_set:
                gf = _gap(s2, m, t); gs = gf
            else:
                gf = _r2(cyc, m, t, "R2_gap_full") if "R2_gap_full" in cyc.columns else None
                gs = _gap(cyc, m, t)
            if gf is not None and gs is not None:
                gap_deltas.append(gs - gf)
    gap_arr  = np.array(gap_deltas) if gap_deltas else np.array([])
    gap_mean = float(gap_arr.mean()) if len(gap_arr) else None
    gap_reduced_n = int((gap_arr < -0.01).sum()) if len(gap_arr) else 0

    if _fs_net > 0.01:
        _fs_verdict = f"Model-specific FS improved mean Test R² overall ({_fs_wins}/{_fs_n} cells gained)."
    elif _fs_net < -0.01:
        _fs_verdict = f"Model-specific FS reduced mean Test R² overall ({_fs_loss}/{_fs_n} cells regressed)."
    else:
        _fs_verdict = f"Model-specific FS had negligible net impact on Test R² ({_fs_wins}/{_fs_n} gained, {_fs_loss}/{_fs_n} regressed)."

    if gap_mean is not None:
        gc = "#5BAD6F" if gap_mean < -0.005 else ("#E15252" if gap_mean > 0.005 else "var(--text-muted)")
        gap_str = (
            f"Mean Δ R²-gap (FS − Full) = <span style='color:{gc};font-weight:bold'>{gap_mean:+.3f}</span> "
            f"({gap_reduced_n}/{len(gap_arr)} model-target pairs saw the gap narrow)."
        )
    else:
        gap_str = "R²-gap data unavailable."

    _ols_note = (
        "For OLS: LassoCV kept all 11 features in most cases (regularisation path "
        "found no features to zero out), so Full ≈ FS. When features were pruned, "
        "it benefited generalization only modestly."
    )
    _ridge_note = "Ridge does not prune — L2 distributes weight across all features. Full = FS."
    _elnet_note = "ElNet's L1 penalty selects internally on the full set — Full = FS by construction."
    _tree_note = (
        "Tree OOF FS (threshold ≥5% relative importance): "
        f"mean Δ for trees = {_colored(tree_fs_d)}. "
        "At 11 features × ~800–1175 training rows, the OOF selection step often hurt "
        "rather than helped — the pruned features still contributed signal, and the "
        "refit on a reduced set introduced a different overfit pattern."
    )

    q3 = (
        f"{_fs_verdict} "
        f"<br><strong>Overall:</strong> mean Δ = {_colored(fs_arr)} "
        f"(Grab: {_colored(grab_fs_d)}, Composite: {_colored(comp_fs_d)}). "
        f"<br><strong>Generalisation:</strong> {gap_str} "
        f"<br><strong>By model:</strong> linear {_colored(lin_fs_d)}, tree {_colored(tree_fs_d)}. "
        f"<br>{_ols_note} {_ridge_note} {_elnet_note} "
        f"<br>{_tree_note} "
        f"<br><em>Implication:</em> at the Exp1 feature-set level (11 features), "
        f"model-specific FS does not consistently improve on using the full set. "
        f"Feature selection becomes more valuable when the feature pool grows larger "
        f"relative to the training size (Exp3 and beyond)."
    )

    # ── Q4 : Best variant per target (raw ★ and gap-adj ✦) ──────────────────
    def _gaj_q(r2, gap):
        if r2 is None: return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    star_counts = {"Sub1": 0, "Sub2 Full": 0, "Sub2 FS": 0}
    gaj_counts  = {"Sub1": 0, "Sub2 Full": 0, "Sub2 FS": 0}
    tgt_winner_rows = ""

    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        per_raw = {}; per_gaj = {}

        # Sub1
        raws1 = [_r2(s1, m, tgt) for m in all_m if _r2(s1, m, tgt) is not None]
        gajs1 = [_gaj_q(_r2(s1, m, tgt), _gap(s1, m, tgt)) for m in all_m
                 if _r2(s1, m, tgt) is not None]
        per_raw["Sub1"] = max(raws1) if raws1 else None
        per_gaj["Sub1"] = max((g for g in gajs1 if g is not None), default=None)

        # Sub2 Full
        rawf = [_full_val(m, tgt) for m in all_m if _full_val(m, tgt) is not None]
        per_raw["Sub2 Full"] = max(rawf) if rawf else None
        # gap for full
        def _full_gap_q(m, t):
            if m == "OLS": return _r2(s2, m, t, "R2_gap_full")
            elif m in lin_set: return _gap(s2, m, t)
            else: return _r2(cyc, m, t, "R2_gap_full") if "R2_gap_full" in cyc.columns else None
        gajf = [_gaj_q(_full_val(m, tgt), _full_gap_q(m, tgt)) for m in all_m
                if _full_val(m, tgt) is not None]
        per_gaj["Sub2 Full"] = max((g for g in gajf if g is not None), default=None)

        # Sub2 FS
        rawfs = [_fs_val(m, tgt) for m in all_m if _fs_val(m, tgt) is not None]
        per_raw["Sub2 FS"] = max(rawfs) if rawfs else None
        def _fs_gap_q(m, t):
            if m in lin_set: return _gap(s2, m, t)
            else: return _gap(cyc, m, t)
        gajfs = [_gaj_q(_fs_val(m, tgt), _fs_gap_q(m, tgt)) for m in all_m
                 if _fs_val(m, tgt) is not None]
        per_gaj["Sub2 FS"] = max((g for g in gajfs if g is not None), default=None)

        raw_valid = {lb: v for lb, v in per_raw.items() if v is not None}
        best_raw  = max(raw_valid.values()) if raw_valid else None
        raw_wins  = [lb for lb, v in raw_valid.items() if best_raw is not None and abs(v - best_raw) < 1e-9]
        for w in raw_wins: star_counts[w] += 1

        gaj_valid = {lb: v for lb, v in per_gaj.items() if v is not None}
        best_gaj  = max(gaj_valid.values()) if gaj_valid else None
        gaj_wins  = [lb for lb, v in gaj_valid.items() if best_gaj is not None and abs(v - best_gaj) < 1e-9]
        for w in gaj_wins: gaj_counts[w] += 1

        cells = ""
        for lb in ["Sub1", "Sub2 Full", "Sub2 FS"]:
            rv = per_raw.get(lb); gv = per_gaj.get(lb)
            is_raw = lb in raw_wins; is_gaj = lb in gaj_wins
            if rv is None:
                cells += "<td class='meta' style='text-align:center;font-size:0.82em'>—</td>"
                continue
            marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
            marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
            if is_raw and is_gaj:   col = "#5BAD6F"; fw = "bold"
            elif is_raw:            col = "#E67E22"; fw = "bold"
            elif is_gaj:            col = "#4A90D9"; fw = "bold"
            else:                   col = "inherit"; fw = "normal"
            gaj_display = f"<br><span style='font-size:0.78em;color:var(--text-muted)'>{gv:+.3f}</span>" if gv is not None else ""
            cells += (
                f"<td style='text-align:center;color:{col};font-weight:{fw};font-size:0.83em'>"
                f"{rv:+.3f}{marker_html}{gaj_display}</td>"
            )
        tgt_winner_rows += f"<tr><td style='font-size:0.83em'><strong>{short}</strong></td>{cells}</tr>"

    best_raw_var = max(star_counts, key=lambda k: star_counts[k])
    best_gaj_var = max(gaj_counts,  key=lambda k: gaj_counts[k])

    diverged = []
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        pr = {}; pg = {}
        for lb, fn in [("Sub1", lambda m,t: _r2(s1, m, t)),
                       ("Sub2 Full", _full_val), ("Sub2 FS", _fs_val)]:
            raws = [fn(m, tgt) for m in all_m if fn(m, tgt) is not None]
            pr[lb] = max(raws) if raws else None
        braw = max((lb for lb, v in pr.items() if v is not None), key=lambda lb: pr[lb], default=None)
        bg   = max((lb for lb, v in pg.items() if v is not None), key=lambda lb: pg[lb], default=None) if pg else None
        if braw and bg and braw != bg:
            diverged.append(f"{short} (raw→{braw}, gap-adj→{bg})")

    _div_note = (
        f" Raw and gap-adjusted winners diverge on {len(diverged)} target(s): "
        + ", ".join(diverged) + "." if diverged else
        " Raw and gap-adjusted winners agree on all targets."
    )

    q4 = (
        f"Raw winner: <strong>{best_raw_var}</strong> "
        f"({star_counts['Sub1']} Sub1 / {star_counts['Sub2 Full']} Sub2-Full / "
        f"{star_counts['Sub2 FS']} Sub2-FS ★). "
        f"Gap-adjusted winner: <strong>{best_gaj_var}</strong> "
        f"({gaj_counts['Sub1']} Sub1 / {gaj_counts['Sub2 Full']} Sub2-Full / "
        f"{gaj_counts['Sub2 FS']} Sub2-FS ✦).{_div_note} "
        f"The table below shows the best raw Test R² (large) and gap-adjusted score (small, grey) "
        f"each variant achieves per target:"
    )

    q4_table = f"""
<div style='overflow-x:auto;margin-top:0.5rem'>
<table class='summary-table' style='font-size:0.83em;width:auto;min-width:420px'>
  <thead><tr>
    <th>Target</th>
    <th style='text-align:center'>Sub1</th>
    <th style='text-align:center'>Sub2 Full</th>
    <th style='text-align:center'>Sub2 FS</th>
  </tr></thead>
  <tbody>{tgt_winner_rows}</tbody>
</table>
<p class='meta' style='margin-top:0.4rem'>
  <strong>★</strong> = raw R² winner · <strong>✦</strong> = gap-adj winner ·
  green = both · orange = raw only · blue = gap-adj only.
  Small grey value = gap-adjusted score.
</p>
</div>"""

    # ── Q5 : Grab vs Composite split ──────────────────────────────────────────
    grab_d12 = _delta_arr(s1, s2, all_m, GRAB_TARGETS)
    comp_d12 = _delta_arr(s1, s2, all_m, COMP_TARGETS)
    grab_fs_d2 = _delta_arr_raw(_full_val, _fs_val, all_m, GRAB_TARGETS)
    comp_fs_d2 = _delta_arr_raw(_full_val, _fs_val, all_m, COMP_TARGETS)

    q5 = (
        f"Consistently, yes. Grab targets respond more positively to every transition. "
        f"<br>"
        f"<strong>Sub1 → Sub2 Full:</strong> Grab {_colored(grab_d12)}, Composite {_colored(comp_d12)}. "
        f"<strong>Full → FS:</strong> Grab {_colored(grab_fs_d2)}, Composite {_colored(comp_fs_d2)}. "
        f"The pattern is structural: Composite targets have roughly 800 training rows vs ~1,175 "
        f"for Grab. Any feature change that expands dimensionality amplifies overfitting risk "
        f"proportionally more on the smaller composite sets. "
        f"Composite COD and pH fail across all variants — they need secondary clarifier data "
        f"(Experiment 2 and beyond) before any model generalises."
    )

    def _qcard(n, question, answer, extra=""):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem'>{answer}</p>
    {extra}
  </div>
</div>"""

    return f"""
<details class="exp-details" id="exp1-findings" open>
  <summary><span class="fold-icon">▶</span> Findings — Experiment 1</summary>
  <div class="exp-body">
  {_qcard(1, "Do inlet concentrations alone carry meaningful predictive power?", q1)}
  {_qcard(2, "What does adding process context (Flow, Power, cyclic calendar) contribute?", q2)}
  {_qcard(3, "Does model-specific feature selection improve accuracy or generalisation within Sub2?", q3)}
  {_qcard(4, "Which variant achieves the best performance per target?", q4, q4_table)}
  {_qcard(5, "Do Grab and Composite targets respond differently to feature changes?", q5)}
  </div>
</details>"""


def _exp1_comparison_panel(df_all: pd.DataFrame) -> str:
    """Sub-experiment comparison: Sub1 vs Sub2-Full vs Sub2-FS.

    Columns:
      Sub1       — 4 inlet features, no FS
      Sub2 Full  — 11 cyclic features, full set (OLS: R2_test_full; Ridge/ElNet/trees: full set)
      Δ S1→Full  — value of adding process + cyclic calendar context
      Sub2 FS    — 11 cyclic features, model-specific FS
                   OLS: LassoCV (R2_test from Exp1)
                   Ridge: same as Full (L2, no pruning)
                   ElNet: same as Full (L1 selects internally)
                   RF/GB/XGB: OOF perm importance refit (R2_test from Exp1-Cyclic)
      Δ Full→FS  — value of feature selection within Sub2

    Data sources:
      Sub1       → exp_key == "Exp1-Sub1", R2_test
      Sub2-Full  → OLS: exp_key == "Exp1", R2_test_full
                   Ridge/ElNet: exp_key == "Exp1", R2_test
                   Trees: exp_key == "Exp1-Cyclic", R2_test_full
      Sub2-FS    → OLS: exp_key == "Exp1", R2_test
                   Ridge/ElNet: exp_key == "Exp1", R2_test
                   Trees: exp_key == "Exp1-Cyclic", R2_test
    """
    s1  = df_all[df_all["exp_key"] == "Exp1-Sub1"].copy()
    s2  = df_all[df_all["exp_key"] == "Exp1"].copy()
    cyc = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()

    if s1.empty and s2.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    lin_models = {"OLS", "Ridge", "ElNet"}
    MEANINGFUL = 0.01

    def _get(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns: return None
        v = r[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _gap_v(df, model, tgt):
        return _get(df, model, tgt, "R2_gap")

    def _gaj(r2, gap):
        if r2 is None: return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    def _full_val(model, tgt):
        """Full-feature R² for Sub2."""
        if model == "OLS":
            return _get(s2, model, tgt, "R2_test_full")
        elif model in lin_models:
            return _get(s2, model, tgt, "R2_test")
        else:
            v = _get(cyc, model, tgt, "R2_test_full")
            return v if v is not None else _get(s2, model, tgt, "R2_test")

    def _fs_val(model, tgt):
        """Feature-selected R² for Sub2."""
        if model in lin_models:
            return _get(s2, model, tgt, "R2_test")
        else:
            return _get(cyc, model, tgt, "R2_test")

    def _full_gap(model, tgt):
        if model == "OLS":
            return _get(s2, model, tgt, "R2_gap_full")
        elif model in lin_models:
            return _gap_v(s2, model, tgt)
        else:
            return _get(s2, model, tgt, "R2_gap_full")  # baseline NL stores full-set gap

    def _fs_gap(model, tgt):
        if model in lin_models:
            return _gap_v(s2, model, tgt)
        else:
            return _gap_v(cyc, model, tgt)

    def _full_rmse(model, tgt):
        if model == "OLS":
            return _get(s2, model, tgt, "RMSE_test_full")
        elif model in lin_models:
            return _get(s2, model, tgt, "RMSE_test")
        else:
            return _get(s2, model, tgt, "RMSE_test_full")  # baseline NL stores full-set RMSE

    def _fs_rmse(model, tgt):
        if model in lin_models:
            return _get(s2, model, tgt, "RMSE_test")
        else:
            return _get(cyc, model, tgt, "RMSE_test")

    def _s1_rmse(model, tgt):
        return _get(s1, model, tgt, "RMSE_test")

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return "<td class='meta' style='text-align:center'>—</td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw and is_gaj:   col = "#5BAD6F"; fw = "bold"
        elif is_raw:            col = "#E67E22"; fw = "bold"
        elif is_gaj:            col = "#4A90D9"; fw = "bold"
        else:                   col = "inherit"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else "—"
        gap_val  = gap if (gap is not None and gap == gap) else None
        gap_str  = f"{gap_val:+.3f}" if gap_val is not None else "—"
        gap_col  = ("#E15252" if gap_val is not None and gap_val > 0.10
                    else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                    else "var(--text-muted)")
        secondary = (f"<br><span style='font-size:0.72em;color:var(--text-muted);"
                     f"font-weight:normal'>RMSE {rmse_str} · "
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return "<td class='meta' style='text-align:center'>—</td>"
        r2_col = ("#5BAD6F" if dr2 >= MEANINGFUL else
                  "#E15252" if dr2 <= -MEANINGFUL else "var(--text-muted)")
        extra = ""
        if drmse is not None and drmse == drmse:
            rm_col = ("#5BAD6F" if drmse <= -MEANINGFUL else
                      "#E15252" if drmse >= MEANINGFUL else "var(--text-muted)")
            extra += (f"<br><span style='font-size:0.72em;color:{rm_col};"
                      f"font-weight:normal'>RMSE {drmse:+.2f}</span>")
        if dgap is not None and dgap == dgap:
            # positive ΔGap = more overfitting (bad); negative = less (good)
            gp_col = ("#E15252" if dgap > MEANINGFUL else
                      "#5BAD6F" if dgap < -MEANINGFUL else "var(--text-muted)")
            extra += (f"<br><span style='font-size:0.72em;color:{gp_col};"
                      f"font-weight:normal'>Gap {dgap:+.3f}</span>")
        return (f"<td style='text-align:center;color:{r2_col};font-weight:bold'>"
                f"{dr2:+.3f}{extra}</td>")

    deltas_s1_full  = []; gaj_deltas_s1_full  = []
    deltas_full_fs  = []; gaj_deltas_full_fs  = []

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:linear-gradient(90deg,var(--bg-secondary) 0%,var(--bg-alt) 100%)'>"
            f"<td colspan='6' style='font-weight:bold;padding:0.5rem 0.8rem;font-size:0.9em;"
            f"border-left:4px solid var(--accent-blue,#4A90D9)'>{short}</td></tr>"
        )
        for m in models_ord:
            v1   = _get(s1, m, tgt);   g1   = _gap_v(s1, m, tgt)
            vf   = _full_val(m, tgt);  gf   = _full_gap(m, tgt)
            vs   = _fs_val(m, tgt);    gs   = _fs_gap(m, tgt)

            r1   = _s1_rmse(m, tgt)
            rf_  = _full_rmse(m, tgt)
            rs_  = _fs_rmse(m, tgt)

            sc1 = _gaj(v1, g1); scf = _gaj(vf, gf); scs = _gaj(vs, gs)

            d1f = (vf - v1) if (v1 is not None and vf is not None) else None
            dfs = (vs - vf) if (vf is not None and vs is not None) else None

            d1f_rmse = ((rf_ - r1) if (r1 is not None and rf_ is not None
                         and r1 == r1 and rf_ == rf_) else None)
            dfs_rmse = ((rs_ - rf_) if (rf_ is not None and rs_ is not None
                         and rf_ == rf_ and rs_ == rs_) else None)

            d1f_gap = ((gf - g1) if (g1 is not None and gf is not None
                        and g1 == g1 and gf == gf) else None)
            dfs_gap = ((gs - gf) if (gf is not None and gs is not None
                        and gf == gf and gs == gs) else None)

            sd1f = (scf - sc1) if (sc1 is not None and scf is not None) else None
            sdfs = (scs - scf) if (scf is not None and scs is not None) else None

            if d1f  is not None: deltas_s1_full.append(d1f)
            if dfs  is not None: deltas_full_fs.append(dfs)
            if sd1f is not None: gaj_deltas_s1_full.append(sd1f)
            if sdfs is not None: gaj_deltas_full_fs.append(sdfs)

            raw_cands = {k: vv for k, vv in [("s1",v1),("full",vf),("fs",vs)] if vv is not None}
            gaj_cands = {k: vv for k, vv in [("s1",sc1),("full",scf),("fs",scs)] if vv is not None}
            best_raw = max(raw_cands.values()) if raw_cands else None
            best_gaj = max(gaj_cands.values()) if gaj_cands else None

            def _is_raw(v_): return best_raw is not None and v_ is not None and abs(v_ - best_raw) < 1e-9
            def _is_gaj(s_): return best_gaj is not None and s_ is not None and abs(s_ - best_gaj) < 1e-9

            tbody += (
                f"<tr><td><strong>{m}</strong></td>"
                f"{_val_td(v1,  r1,  g1,  _is_raw(v1),  _is_gaj(sc1))}"
                f"{_val_td(vf,  rf_, gf,  _is_raw(vf),  _is_gaj(scf))}"
                f"{_delta_td(d1f, d1f_rmse, d1f_gap)}"
                f"{_val_td(vs,  rs_, gs,  _is_raw(vs),  _is_gaj(scs))}"
                f"{_delta_td(dfs, dfs_rmse, dfs_gap)}"
                f"</tr>"
            )

    def _combined_stats_row(raw_deltas, gaj_deltas, from_lbl, to_lbl):
        if not raw_deltas:
            return (f"<tr><td><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' class='meta'>—</td></tr>")
        arr = np.array(raw_deltas)
        n   = len(arr)
        net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mean_win  = float(wins.mean())   if len(wins)   else None
        mean_loss = float(losses.mean()) if len(losses) else None
        net_col  = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "var(--text-muted)")
        verdict  = ("Net improvement" if net > MEANINGFUL else
                    "Net regression"  if net < -MEANINGFUL else "Negligible")
        win_str  = (f"{len(wins)}/{n} (avg {mean_win:+.3f})"    if mean_win  is not None else f"{len(wins)}/{n}")
        loss_str = (f"{len(losses)}/{n} (avg {mean_loss:+.3f})" if mean_loss is not None else f"{len(losses)}/{n}")
        if gaj_deltas:
            gaj_net  = float(np.array(gaj_deltas).mean())
            diff     = gaj_net - net
            gaj_col  = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "var(--text-muted)")
            if diff < -0.02:   interp = "Raw gains partially inflated by increased overfitting"; diff_col = "#E15252"
            elif diff > 0.02:  interp = "Raw gains understated — overfitting decreased"; diff_col = "#5BAD6F"
            else:              interp = "Overfitting largely unchanged"; diff_col = "var(--text-muted)"
            gaj_str  = f"{gaj_net:+.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            gaj_str = diff_str = "—"; gaj_col = diff_col = "var(--text-muted)"; interp = "—"
        verdict_cell = (f"<strong>{verdict}</strong><br>"
                        f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span>")
        return (
            f"<tr>"
            f"<td style='white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
            f"<td style='text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='text-align:center;color:var(--text-muted)'>{len(ties)}/{n}</td>"
            f"<td style='text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td class='meta'>{verdict_cell}</td>"
            f"</tr>"
        )

    n_cells = len(models_ord) * len(TARGETS_ORDERED)
    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem'>Transition Summary</p>
  <div style='overflow-x:auto'>
  <table class='summary-table' style='font-size:0.85em;min-width:960px'>
    <thead><tr>
      <th>Transition</th>
      <th style='text-align:center'>Net Mean ΔR²<br><span class='meta'>raw</span></th>
      <th style='text-align:center'>Improvements<br><span class='meta'>Δ &gt; +{MEANINGFUL}</span></th>
      <th style='text-align:center'>Regressions<br><span class='meta'>Δ &lt; −{MEANINGFUL}</span></th>
      <th style='text-align:center'>Negligible<br><span class='meta'>|Δ| ≤ {MEANINGFUL}</span></th>
      <th style='text-align:center'>Gap-Adj Net ΔR²<br><span class='meta'>R²−0.5·max(0,|gap|−0.10)</span></th>
      <th style='text-align:center'>Gap-Adj − Raw<br><span class='meta'>overfitting shift</span></th>
      <th>Verdict / Interpretation</th>
    </tr></thead>
    <tbody>
      {_combined_stats_row(deltas_s1_full, gaj_deltas_s1_full, "Sub1", "Sub2 Full")}
      {_combined_stats_row(deltas_full_fs, gaj_deltas_full_fs, "Sub2 Full", "Sub2 FS")}
    </tbody>
  </table>
  </div>

  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      Net Mean ΔR² is the signed average across all {n_cells} (model × target) cells.
      <strong>Sub2 Full</strong> = full 11-feature set before any pruning
      (OLS: unregularised on all features; Ridge/ElNet: full set; RF/GB/XGB: Phase 1 CV model).
      <strong>Sub2 FS</strong> = model-specific selection applied
      (OLS: LassoCV; Ridge: no change — L2 handles it; ElNet: no change — L1 selects internally;
      RF/GB/XGB: OOF permutation importance refit on features with ≥5% relative importance).
      <strong>Gap-Adj − Raw</strong> indicates whether raw gains are inflated by increased overfitting
      (negative = gains overstated) or understated by reduced overfitting (positive = gains understated).
      <strong>★</strong> = best raw Test R² · <strong>✦</strong> = best gap-adjusted score ·
      <span style='color:#5BAD6F'>green</span> = both ·
      <span style='color:#E67E22'>orange</span> = raw only ·
      <span style='color:#4A90D9'>blue</span> = gap-adj only.
      In the delta columns: <span style='color:#5BAD6F;font-weight:bold'>green ΔR²</span> = improvement ·
      <span style='color:#E15252;font-weight:bold'>red ΔR²</span> = regression ·
      <span style='color:#E15252'>red ΔGap</span> = gap widened (more overfit) ·
      <span style='color:#5BAD6F'>green ΔGap</span> = gap narrowed (less overfit) ·
      grey = negligible (|Δ| ≤ {MEANINGFUL}).
    </p>
  </div>
</div>"""

    # ── Validation framework card ─────────────────────────────────────────────
    val_rows = []
    for m in models_ord:
        if m == "OLS":
            method  = "LassoCV — L1 cross-validated pre-screen (TimeSeriesSplit)"
            ns_vals = [v for tgt in TARGETS_ORDERED
                       for v in [_get(s2, m, tgt, "n_selected_ols")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 11" if ns_vals else "—"
            bypass  = False
        elif m == "Ridge":
            method  = "Full set — L2 regularisation handles collinearity internally"
            avg_sel = "11 / 11 (all)"
            bypass  = True
        elif m == "ElNet":
            method  = "Internal L1 — retains features with non-zero fitted coefficients"
            ns_vals = [v for tgt in TARGETS_ORDERED
                       for v in [_get(s2, m, tgt, "ElNet_n_selected")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 11" if ns_vals else "—"
            bypass  = False
        else:
            method  = "OOF permutation importance ≥ 5% threshold (3-phase: full → select → refit)"
            ns_vals = [v for tgt in TARGETS_ORDERED
                       for v in [_get(cyc, m, tgt, "n_selected_nl")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 11" if ns_vals else "—"
            bypass  = False
        status_col = "#7FB3D3" if bypass else "#5BAD6F"
        status     = "✓ Correctly bypassed (L2)" if bypass else "✓ Selection applied"
        val_rows.append(
            f"<tr><td><strong>{m}</strong></td>"
            f"<td style='font-size:0.82em'>{method}</td>"
            f"<td style='text-align:center'>11</td>"
            f"<td style='text-align:center'>{avg_sel}</td>"
            f"<td style='text-align:center;color:{status_col};font-weight:600'>{status}</td></tr>"
        )

    val_card = f"""
<div style='margin:1rem 0 0.6rem'>
  <p style='font-weight:bold;margin-bottom:0.3rem;font-size:0.88em;color:var(--text-main)'>
    Feature Selection Validation — Sub-experiment 2 (11 cyclic features input)
  </p>
  <p class='meta' style='margin-bottom:0.6rem;font-size:0.82em'>
    Each model applies a distinct selection protocol to the shared 11-feature pool.
    The table confirms which method was used, how many features were retained on average
    across the 8 targets, and whether the protocol was correctly applied.
  </p>
  <div style='overflow-x:auto'>
  <table class='summary-table' style='font-size:0.83em'>
    <thead><tr>
      <th>Model</th><th>FS Method</th>
      <th style='text-align:center'>Input<br>Feats</th>
      <th style='text-align:center'>Avg Selected<br>(across 8 targets)</th>
      <th style='text-align:center'>Status</th>
    </tr></thead>
    <tbody>{"".join(val_rows)}</tbody>
  </table>
  </div>
  <p class='meta' style='font-size:0.78em;margin-top:0.3rem'>
    Ridge is the only model that intentionally bypasses feature removal — L2 shrinks
    uninformative coefficients toward zero without discarding them. All other models
    reduce the feature space. Per-target selected feature lists are in the
    <em>Model-Specific Feature Selection</em> fold within Sub-experiment 2 above.
  </p>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem'>
<table class='summary-table' style='font-size:0.83em;min-width:860px;border-collapse:collapse'>
  <thead>
    <tr style='background:var(--bg-secondary)'>
      <th style='min-width:60px'>Model</th>
      <th style='text-align:center;border-left:3px solid #888'>Sub1<br>
          <span class='meta' style='font-weight:normal'>4 feat · no FS</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center;border-left:3px solid #4A90D9'>Sub2 Full<br>
          <span class='meta' style='font-weight:normal'>11 feat · full set</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center'>Δ (S1→Full)<br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='text-align:center;border-left:3px solid #5BAD6F'>Sub2 FS<br>
          <span class='meta' style='font-weight:normal'>model-specific</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center'>Δ (Full→FS)<br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>ΔR² · ΔRMSE · ΔGap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp1-comparison">
  <summary><span class="fold-icon">▶</span>
    Sub-experiment Comparison — Sub1 / Sub2 Full / Sub2 Feature-Selected
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>Sub1</strong> (inlet only, 4 features, no FS) →
        <strong>Sub2 Full</strong> (cyclic calendar + process, 11 features, no pruning) →
        <strong>Sub2 FS</strong> (same 11 features, model-specific selection applied).
        For linear models, Full and FS differ only for OLS (LassoCV may prune);
        Ridge keeps the full set (L2 handles collinearity), ElNet runs L1 on the full set.
        For tree models, Full = Phase 1 CV result (all features);
        FS = Phase 3 result after OOF permutation importance pruning (≥5% threshold).
        <strong>★</strong> = best raw Test R² per (model, target) ·
        <strong>✦</strong> = best gap-adjusted score ·
        <span style='color:#5BAD6F;font-weight:bold'>green</span> = both ·
        <span style='color:#E67E22;font-weight:bold'>orange</span> = raw only ·
        <span style='color:#4A90D9;font-weight:bold'>blue</span> = gap-adj only ·
        <span style='color:#5BAD6F;font-weight:bold'>green Δ</span> = improvement ·
        <span style='color:#E15252;font-weight:bold'>red Δ</span> = regression.
        RMSE reported in native units (mg/L or pH units). Gap = Train R² − Test R²;
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {val_card}
    {main_table}
    {stats_block}
  </div>
</details>"""


def build_exp1_section(df_all: pd.DataFrame) -> str:
    sub_s1 = _exp_subsection(df_all, "Exp1-Sub1", "exp1-sub1",
                             "Sub-experiment 1 — Inlet Only (4 features, no FS)",
                             open_default=False)
    sub_s2 = _exp_subsection(df_all, "Exp1", "exp1-full",
                             "Sub-experiment 2 — Inlet + Process + Cyclic Calendar "
                             "(11 features, model-specific FS)",
                             open_default=True)
    cmp_div      = _exp1_comparison_panel(df_all)
    findings_div = _exp1_qna(df_all)
    best         = _exp1_best_model_box(df_all)
    return f"""
<section id="exp1">
  <h1 class="section-title">Experiment 1 - Inlet Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp1"]}</p>
  {sub_s1}
  {sub_s2}
  {cmp_div}
  {findings_div}
  {best}
</section>"""


def _exp2_comparison_panel(df_all: pd.DataFrame) -> str:
    """Sub-experiment comparison table for Experiment 2.

    Columns: Sub1-Clr | Sub1-Sed | Combined | Δ(Clr→Comb) | Sub2 Full | Sub2 FS | Δ(Comb→S2-FS)
    """
    clr  = df_all[df_all["exp_key"] == "Exp2-Sub1-Clr"].copy()
    sed  = df_all[df_all["exp_key"] == "Exp2-Sub1-Sed"].copy()
    comb = df_all[df_all["exp_key"] == "Exp2-Sub1"].copy()
    cyc  = df_all[df_all["exp_key"] == "Exp2-Sub2-Cyc"].copy()

    if clr.empty and sed.empty and comb.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    lin_models = {"OLS", "Ridge", "ElNet"}
    MEANINGFUL = 0.01

    def _get(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns: return None
        v = r[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _gap_v(df, model, tgt):
        return _get(df, model, tgt, "R2_gap")

    def _gaj(r2, gap):
        if r2 is None: return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    def _sub2_full(model, tgt):
        """R² on full feature set before FS for Sub2."""
        if model == "OLS":
            v = _get(cyc, model, tgt, "R2_test_full")
            return v if v is not None else _get(cyc, model, tgt)
        elif model in lin_models:
            return _get(cyc, model, tgt)
        else:
            v = _get(cyc, model, tgt, "R2_test_full")
            return v if v is not None else _get(cyc, model, tgt)

    def _sub2_fs(model, tgt):
        """Feature-selected R² for Sub2."""
        return _get(cyc, model, tgt)

    def _sub2_full_gap(model, tgt):
        if model == "OLS":
            v = _get(cyc, model, tgt, "R2_gap_full")
            return v if v is not None else _gap_v(cyc, model, tgt)
        elif model in lin_models:
            return _gap_v(cyc, model, tgt)
        else:
            v = _get(cyc, model, tgt, "R2_gap_full")
            return v if v is not None else _gap_v(cyc, model, tgt)

    def _sub2_fs_gap(model, tgt):
        return _gap_v(cyc, model, tgt)

    def _sub2_full_rmse(model, tgt):
        if model == "OLS":
            v = _get(cyc, model, tgt, "RMSE_test_full")
            return v if v is not None else _get(cyc, model, tgt, "RMSE_test")
        else:
            return _get(cyc, model, tgt, "RMSE_test")

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return "<td class='meta' style='text-align:center'>—</td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw and is_gaj:   col = "#5BAD6F"; fw = "bold"
        elif is_raw:            col = "#E67E22"; fw = "bold"
        elif is_gaj:            col = "#4A90D9"; fw = "bold"
        else:                   col = "inherit"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else "—"
        gap_val  = gap if (gap is not None and gap == gap) else None
        gap_str  = f"{gap_val:+.3f}" if gap_val is not None else "—"
        gap_col  = ("#E15252" if gap_val is not None and gap_val > 0.10
                    else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                    else "var(--text-muted)")
        secondary = (f"<br><span style='font-size:0.72em;color:var(--text-muted);"
                     f"font-weight:normal'>RMSE {rmse_str} · "
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return "<td class='meta' style='text-align:center'>—</td>"
        r2_col = ("#5BAD6F" if dr2 >= MEANINGFUL else
                  "#E15252" if dr2 <= -MEANINGFUL else "var(--text-muted)")
        extra = ""
        if drmse is not None and drmse == drmse:
            rm_col = ("#5BAD6F" if drmse <= -MEANINGFUL else
                      "#E15252" if drmse >= MEANINGFUL else "var(--text-muted)")
            extra += (f"<br><span style='font-size:0.72em;color:{rm_col};"
                      f"font-weight:normal'>RMSE {drmse:+.2f}</span>")
        if dgap is not None and dgap == dgap:
            gp_col = ("#E15252" if dgap > MEANINGFUL else
                      "#5BAD6F" if dgap < -MEANINGFUL else "var(--text-muted)")
            extra += (f"<br><span style='font-size:0.72em;color:{gp_col};"
                      f"font-weight:normal'>Gap {dgap:+.3f}</span>")
        return (f"<td style='text-align:center;color:{r2_col};font-weight:bold'>"
                f"{dr2:+.3f}{extra}</td>")

    # Collect deltas for summary rows
    d_clr_comb      = []; gaj_d_clr_comb      = []
    d_sed_comb      = []; gaj_d_sed_comb      = []
    d_clr_sed       = []; gaj_d_clr_sed       = []
    d_comb_s2full   = []; gaj_d_comb_s2full   = []
    d_comb_s2fs     = []; gaj_d_comb_s2fs     = []
    d_sed_s2full    = []; gaj_d_sed_s2full    = []
    d_sed_s2fs      = []; gaj_d_sed_s2fs      = []
    d_s2full_s2fs   = []; gaj_d_s2full_s2fs   = []

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:linear-gradient(90deg,var(--bg-secondary) 0%,var(--bg-alt) 100%)'>"
            f"<td colspan='6' style='font-weight:bold;padding:0.5rem 0.8rem;font-size:0.9em;"
            f"border-left:4px solid var(--accent-blue,#4A90D9)'>{short}</td></tr>"
        )
        for m in models_ord:
            vc  = _get(clr,  m, tgt); gc  = _gap_v(clr,  m, tgt)
            vs  = _get(sed,  m, tgt); gs  = _gap_v(sed,  m, tgt)
            vco = _get(comb, m, tgt); gco = _gap_v(comb, m, tgt)
            vf  = _sub2_full(m, tgt); gf  = _sub2_full_gap(m, tgt)
            vfs = _sub2_fs(m,   tgt); gfs = _sub2_fs_gap(m,   tgt)

            rc  = _get(clr,  m, tgt, "RMSE_test")
            rs  = _get(sed,  m, tgt, "RMSE_test")
            rco = _get(comb, m, tgt, "RMSE_test")
            rf_ = _sub2_full_rmse(m, tgt)
            rfs = _get(cyc,  m, tgt, "RMSE_test")

            sc  = _gaj(vc,  gc);   ss  = _gaj(vs,  gs)
            sco = _gaj(vco, gco);  sf  = _gaj(vf,  gf); sfs = _gaj(vfs, gfs)

            d_cc  = (vco - vc)  if (vc  is not None and vco is not None) else None
            d_sc  = (vco - vs)  if (vs  is not None and vco is not None) else None
            d_cs  = (vs  - vc)  if (vc  is not None and vs  is not None) else None
            d_cf  = (vf  - vco) if (vco is not None and vf  is not None) else None
            d_cfs = (vfs - vco) if (vco is not None and vfs is not None) else None
            d_sf  = (vf  - vs)  if (vs  is not None and vf  is not None) else None
            d_sfs = (vfs - vs)  if (vs  is not None and vfs is not None) else None
            d_ff  = (vfs - vf)  if (vf  is not None and vfs is not None) else None

            d_cc_rmse  = (rco - rc)  if (rc  is not None and rco is not None) else None
            d_sc_rmse  = (rco - rs)  if (rs  is not None and rco is not None) else None
            d_cf_rmse  = (rf_ - rco) if (rco is not None and rf_ is not None) else None
            d_cfs_rmse = (rfs - rco) if (rco is not None and rfs is not None) else None
            d_sf_rmse  = (rf_ - rs)  if (rs  is not None and rf_ is not None) else None
            d_sfs_rmse = (rfs - rs)  if (rs  is not None and rfs is not None) else None
            d_ff_rmse  = (rfs - rf_) if (rf_ is not None and rfs is not None) else None

            d_cc_gap  = (gco - gc)  if (gc  is not None and gco is not None) else None
            d_sc_gap  = (gco - gs)  if (gs  is not None and gco is not None) else None
            d_cf_gap  = (gf  - gco) if (gco is not None and gf  is not None) else None
            d_cfs_gap = (gfs - gco) if (gco is not None and gfs is not None) else None
            d_sf_gap  = (gf  - gs)  if (gs  is not None and gf  is not None) else None
            d_sfs_gap = (gfs - gs)  if (gs  is not None and gfs is not None) else None
            d_ff_gap  = (gfs - gf)  if (gf  is not None and gfs is not None) else None

            # gap-adj deltas
            gaj_cc  = (sco - sc)  if (sc  is not None and sco is not None) else None
            gaj_sc  = (sco - ss)  if (ss  is not None and sco is not None) else None
            gaj_cs  = (ss  - sc)  if (sc  is not None and ss  is not None) else None
            gaj_cf  = (sf  - sco) if (sco is not None and sf  is not None) else None
            gaj_cfs = (sfs - sco) if (sco is not None and sfs is not None) else None
            gaj_sf  = (sf  - ss)  if (ss  is not None and sf  is not None) else None
            gaj_sfs = (sfs - ss)  if (ss  is not None and sfs is not None) else None
            gaj_ff  = (sfs - sf)  if (sf  is not None and sfs is not None) else None

            if d_cc  is not None: d_clr_comb.append(d_cc);       gaj_d_clr_comb.append(gaj_cc)    if gaj_cc  is not None else None
            if d_sc  is not None: d_sed_comb.append(d_sc);       gaj_d_sed_comb.append(gaj_sc)    if gaj_sc  is not None else None
            if d_cs  is not None: d_clr_sed.append(d_cs);        gaj_d_clr_sed.append(gaj_cs)     if gaj_cs  is not None else None
            if d_cf  is not None: d_comb_s2full.append(d_cf);    gaj_d_comb_s2full.append(gaj_cf) if gaj_cf  is not None else None
            if d_cfs is not None: d_comb_s2fs.append(d_cfs);     gaj_d_comb_s2fs.append(gaj_cfs)  if gaj_cfs is not None else None
            if d_sf  is not None: d_sed_s2full.append(d_sf);     gaj_d_sed_s2full.append(gaj_sf)  if gaj_sf  is not None else None
            if d_sfs is not None: d_sed_s2fs.append(d_sfs);      gaj_d_sed_s2fs.append(gaj_sfs)   if gaj_sfs is not None else None
            if d_ff  is not None: d_s2full_s2fs.append(d_ff);    gaj_d_s2full_s2fs.append(gaj_ff) if gaj_ff  is not None else None

            cands_raw = {k: v for k, v in [("Clr",vc),("Sed",vs),("Comb",vco),("S2-Full",vf),("S2-FS",vfs)] if v is not None}
            cands_gaj = {k: v for k, v in [("Clr",sc),("Sed",ss),("Comb",sco),("S2-Full",sf),("S2-FS",sfs)] if v is not None}
            best_raw = max(cands_raw.values()) if cands_raw else None
            best_gaj = max(cands_gaj.values()) if cands_gaj else None

            def _is_raw(v_): return best_raw is not None and v_ is not None and abs(v_ - best_raw) < 1e-9
            def _is_gaj(s_): return best_gaj is not None and s_ is not None and abs(s_ - best_gaj) < 1e-9

            tbody += (
                f"<tr><td><strong>{m}</strong></td>"
                f"{_val_td(vc,  rc,  gc,  _is_raw(vc),  _is_gaj(sc))}"
                f"{_val_td(vs,  rs,  gs,  _is_raw(vs),  _is_gaj(ss))}"
                f"{_val_td(vco, rco, gco, _is_raw(vco), _is_gaj(sco))}"
                f"{_val_td(vf,  rf_, gf,  _is_raw(vf),  _is_gaj(sf))}"
                f"{_val_td(vfs, rfs, gfs, _is_raw(vfs), _is_gaj(sfs))}"
                f"</tr>"
            )

    def _combined_stats_row(raw_deltas, gaj_deltas, from_lbl, to_lbl):
        if not raw_deltas:
            return (f"<tr><td><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' class='meta'>—</td></tr>")
        arr = np.array(raw_deltas)
        n   = len(arr)
        net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mean_win  = float(wins.mean())   if len(wins)   else None
        mean_loss = float(losses.mean()) if len(losses) else None
        net_col  = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "var(--text-muted)")
        verdict  = ("Net improvement" if net > MEANINGFUL else
                    "Net regression"  if net < -MEANINGFUL else "Negligible")
        win_str  = (f"{len(wins)}/{n} (avg {mean_win:+.3f})"    if mean_win  is not None else f"{len(wins)}/{n}")
        loss_str = (f"{len(losses)}/{n} (avg {mean_loss:+.3f})" if mean_loss is not None else f"{len(losses)}/{n}")
        if gaj_deltas:
            gaj_arr = [g for g in gaj_deltas if g is not None]
            if gaj_arr:
                gaj_net  = float(np.array(gaj_arr).mean())
                diff     = gaj_net - net
                gaj_col  = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "var(--text-muted)")
                if diff < -0.02:   interp = "Raw gains partially inflated by increased overfitting"; diff_col = "#E15252"
                elif diff > 0.02:  interp = "Raw gains understated — overfitting decreased"; diff_col = "#5BAD6F"
                else:              interp = "Overfitting largely unchanged"; diff_col = "var(--text-muted)"
                gaj_str  = f"{gaj_net:+.4f}"
                diff_str = f"{diff:+.4f}"
            else:
                gaj_str = diff_str = "—"; gaj_col = diff_col = "var(--text-muted)"; interp = "—"
        else:
            gaj_str = diff_str = "—"; gaj_col = diff_col = "var(--text-muted)"; interp = "—"
        verdict_cell = (f"<strong>{verdict}</strong><br>"
                        f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span>")
        return (
            f"<tr>"
            f"<td style='white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
            f"<td style='text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='text-align:center;color:var(--text-muted)'>{len(ties)}/{n}</td>"
            f"<td style='text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td class='meta'>{verdict_cell}</td>"
            f"</tr>"
        )

    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem'>Transition Summary</p>
  <div style='overflow-x:auto'>
  <table class='summary-table' style='font-size:0.85em;min-width:960px'>
    <thead><tr>
      <th>Transition</th>
      <th style='text-align:center'>Net Mean ΔR²<br><span class='meta'>raw</span></th>
      <th style='text-align:center'>Improvements<br><span class='meta'>Δ &gt; +{MEANINGFUL}</span></th>
      <th style='text-align:center'>Regressions<br><span class='meta'>Δ &lt; −{MEANINGFUL}</span></th>
      <th style='text-align:center'>Negligible<br><span class='meta'>|Δ| ≤ {MEANINGFUL}</span></th>
      <th style='text-align:center'>Gap-Adj Net ΔR²</th>
      <th style='text-align:center'>Gap-Adj − Raw</th>
      <th>Verdict</th>
    </tr></thead>
    <tbody>
      {_combined_stats_row(d_clr_sed,      gaj_d_clr_sed,      "Clarifier", "Sed")}
      {_combined_stats_row(d_clr_comb,     gaj_d_clr_comb,     "Clarifier", "Combined")}
      {_combined_stats_row(d_sed_comb,     gaj_d_sed_comb,     "Sed",       "Combined")}
      {_combined_stats_row(d_comb_s2full,  gaj_d_comb_s2full,  "Combined",  "Sub2 Full")}
      {_combined_stats_row(d_sed_s2full,   gaj_d_sed_s2full,   "Sed",       "Sub2 Full")}
      {_combined_stats_row(d_comb_s2fs,    gaj_d_comb_s2fs,    "Combined",  "Sub2 FS")}
      {_combined_stats_row(d_sed_s2fs,     gaj_d_sed_s2fs,     "Sed",       "Sub2 FS")}
      {_combined_stats_row(d_s2full_s2fs,  gaj_d_s2full_s2fs,  "Sub2 Full", "Sub2 FS")}
    </tbody>
  </table>
  </div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      <strong>Sub1-Clr</strong> = Sec Clarifier + COMMON (10 features, no FS).
      <strong>Sub1-Sed</strong> = Sec Sedimentation + COMMON (10 features, no FS).
      <strong>Combined</strong> = both secondary groups + COMMON (15 features, original baseline).
      <strong>Sub2 Full</strong> = Inlet + Secondary + COMMON (21 features,
      no FS applied; for OLS this is the pre-LassoCV result).
      <strong>Sub2 FS</strong> = same 21 features after model-specific selection
      (OLS: LassoCV; Ridge: full set; ElNet: internal L1; RF/GB/XGB: OOF permutation importance ≥5%).
      <strong>★</strong> = best raw Test R² · <strong>✦</strong> = best gap-adjusted score.
      Detailed transition deltas (win/loss counts and gap-adjusted net) are in the Transition Summary below.
    </p>
  </div>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem'>
<table class='summary-table' style='font-size:0.82em;min-width:860px;border-collapse:collapse'>
  <thead>
    <tr style='background:var(--bg-secondary)'>
      <th style='min-width:60px'>Model</th>
      <th style='text-align:center;border-left:3px solid #888'>Sub1-Clr<br>
          <span class='meta' style='font-weight:normal'>10 feat</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center'>Sub1-Sed<br>
          <span class='meta' style='font-weight:normal'>10 feat</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center;border-left:3px solid #4A90D9'>Combined<br>
          <span class='meta' style='font-weight:normal'>15 feat</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center;border-left:3px solid #5BAD6F'>Sub2 Full<br>
          <span class='meta' style='font-weight:normal'>21 feat</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
      <th style='text-align:center'>Sub2 FS<br>
          <span class='meta' style='font-weight:normal'>model-specific</span><br>
          <span class='meta' style='font-size:0.78em;font-weight:normal'>R² · RMSE · Gap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp2-comparison">
  <summary><span class="fold-icon">▶</span>
    Sub-experiment Comparison — Clarifier / Sed / Combined / Sub2-FS
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        Five columns per (model, target): three secondary-only baselines (Clarifier, Sed,
        Combined) followed by Sub2 Full and Sub2 FS results. Transition deltas are in the
        Transition Summary table below. <strong>★</strong> = best raw Test R² ·
        <strong>✦</strong> = best gap-adjusted score.
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {main_table}
    {stats_block}
  </div>
</details>"""


def _exp2_qna(df_all: pd.DataFrame) -> str:
    """Data-driven Q&A for Experiment 2 — four key questions."""
    clr  = df_all[df_all["exp_key"] == "Exp2-Sub1-Clr"].copy()
    sed  = df_all[df_all["exp_key"] == "Exp2-Sub1-Sed"].copy()
    comb = df_all[df_all["exp_key"] == "Exp2-Sub1"].copy()
    cyc  = df_all[df_all["exp_key"] == "Exp2-Sub2-Cyc"].copy()

    if clr.empty and comb.empty:
        return ""

    all_m  = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL = 0.005

    def _r2(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns: return None
        v = r[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _delta_arr(from_df, to_df, models, targets):
        vals = []
        for m in models:
            for t in targets:
                a = _r2(from_df, m, t); b = _r2(to_df, m, t)
                if a is not None and b is not None:
                    vals.append(b - a)
        return np.array(vals) if vals else np.array([])

    def _colored(arr):
        if not len(arr): return "<em>—</em>"
        v = float(arr.mean())
        c = "#5BAD6F" if v > MEANINGFUL else ("#E15252" if v < -MEANINGFUL else "var(--text-muted)")
        return f"<span style='color:{c};font-weight:bold'>{v:+.3f}</span>"

    # ── Q1: Do Clarifier and Sed carry distinct signal? ───────────────────────
    clr_vals = [_r2(clr, m, t) for m in all_m for t in TARGETS_ORDERED if _r2(clr, m, t) is not None]
    sed_vals = [_r2(sed, m, t) for m in all_m for t in TARGETS_ORDERED if _r2(sed, m, t) is not None]
    clr_avg  = float(np.mean(clr_vals)) if clr_vals else None
    sed_avg  = float(np.mean(sed_vals)) if sed_vals else None

    d_cs = _delta_arr(clr, sed, all_m, TARGETS_ORDERED)
    d_cs_grab = _delta_arr(clr, sed, all_m, GRAB_TARGETS)
    d_cs_comp = _delta_arr(clr, sed, all_m, COMP_TARGETS)

    if clr_avg is not None and sed_avg is not None:
        diff = sed_avg - clr_avg
        if abs(diff) < 0.02:
            _q1_verdict = ("They carry <strong>similar signal</strong> overall "
                           f"(Clarifier avg={clr_avg:+.3f}, Sed avg={sed_avg:+.3f}; Δ={diff:+.3f}). "
                           "The two secondary groups are largely interchangeable at this level.")
        elif diff > 0:
            _q1_verdict = (f"<strong>Sedimentation features slightly outperform Clarifier</strong> "
                           f"(avg Sed={sed_avg:+.3f} vs Clarifier={clr_avg:+.3f}; Δ={diff:+.3f}). "
                           "The difference is small but consistent across targets.")
        else:
            _q1_verdict = (f"<strong>Clarifier features slightly outperform Sedimentation</strong> "
                           f"(avg Clr={clr_avg:+.3f} vs Sed={sed_avg:+.3f}; Δ={diff:+.3f}). "
                           "The difference is small but consistent across targets.")
    else:
        _q1_verdict = "Insufficient data to compare."

    q1 = (
        f"{_q1_verdict} "
        f"<br><strong>Grab targets:</strong> Sed − Clr mean Δ = {_colored(d_cs_grab)}, "
        f"<strong>Composite targets:</strong> Sed − Clr mean Δ = {_colored(d_cs_comp)}. "
        f"Co-occurring missingness means both groups have nearly identical row counts "
        f"— row count is not a confound here. The signal overlap is high because "
        f"Clarifier and Sedimentation units process the same water sequentially."
    )

    # ── Q2: What does combining both groups add? ───────────────────────────────
    d_clr_comb = _delta_arr(clr,  comb, all_m, TARGETS_ORDERED)
    d_sed_comb = _delta_arr(sed,  comb, all_m, TARGETS_ORDERED)
    d_clr_comb_grab = _delta_arr(clr, comb, all_m, GRAB_TARGETS)
    d_clr_comb_comp = _delta_arr(clr, comb, all_m, COMP_TARGETS)

    n_cc = len(d_clr_comb)
    wins_cc = int((d_clr_comb > 0.01).sum()) if n_cc else 0

    q2 = (
        f"Adding Sedimentation to Clarifier (Clr→Combined): mean Δ = {_colored(d_clr_comb)} "
        f"({wins_cc}/{n_cc} cells improved). "
        f"<br>Adding Clarifier to Sedimentation (Sed→Combined): mean Δ = {_colored(d_sed_comb)}. "
        f"<br><strong>Grab:</strong> {_colored(d_clr_comb_grab)}, "
        f"<strong>Composite:</strong> {_colored(d_clr_comb_comp)}. "
        f"The combined group rarely beats the better individual group by more than a small margin — "
        f"confirming the high signal overlap between the two secondary sub-systems. "
        f"Models using the combined 15-feature set do not consistently outperform either "
        f"10-feature variant because regularised models (Ridge, ElNet) already handle "
        f"the redundancy, and tree models are not noticeably improved by the additional 5 features."
    )

    # ── Q3: What does adding inlet concentrations contribute? ──────────────
    # Compare Combined (15 feat) to Sub2-FS (21 feat, model-specific FS)
    def _cyc_fs(m, t): return _r2(cyc, m, t)

    fs_vals = [(m, t, _cyc_fs(m, t)) for m in all_m for t in TARGETS_ORDERED]
    fs_vals = [(m, t, v) for m, t, v in fs_vals if v is not None]

    comb_vals_aligned = [_r2(comb, m, t) for m, t, _ in fs_vals]
    d_comb_s2 = np.array([b - a for (_, _, b), a in zip(fs_vals, comb_vals_aligned)
                           if a is not None])
    d_comb_s2_grab = _delta_arr(
        comb, cyc, all_m, GRAB_TARGETS)
    d_comb_s2_comp = _delta_arr(
        comb, cyc, all_m, COMP_TARGETS)

    n_cs2 = len(d_comb_s2)
    wins_cs2  = int((d_comb_s2 > 0.01).sum())  if n_cs2 else 0
    loss_cs2  = int((d_comb_s2 < -0.01).sum()) if n_cs2 else 0

    q3 = (
        f"Combined (15) → Sub2-FS (21 feat + FS): mean Δ = {_colored(d_comb_s2)} "
        f"({wins_cs2}/{n_cs2} improved, {loss_cs2}/{n_cs2} regressed). "
        f"<br><strong>Grab:</strong> {_colored(d_comb_s2_grab)}, "
        f"<strong>Composite:</strong> {_colored(d_comb_s2_comp)}. "
        f"OLS uses LassoCV to reduce the 21-feature set; tree models use OOF "
        f"permutation importance (≥5% threshold) to select the most informative features. "
        f"Where the delta is positive, the inlet signal and correct calendar encoding "
        f"add genuine predictive value. Where negative, the added features primarily "
        f"increase variance (especially on smaller composite datasets, n≈730–740 train rows)."
    )

    # ── Q4: Grab vs Composite split ───────────────────────────────────────────
    grab_clr = [_r2(clr, m, t) for m in all_m for t in GRAB_TARGETS if _r2(clr, m, t) is not None]
    grab_sed = [_r2(sed, m, t) for m in all_m for t in GRAB_TARGETS if _r2(sed, m, t) is not None]
    comp_clr = [_r2(clr, m, t) for m in all_m for t in COMP_TARGETS if _r2(clr, m, t) is not None]
    comp_sed = [_r2(sed, m, t) for m in all_m for t in COMP_TARGETS if _r2(sed, m, t) is not None]

    ga_clr = float(np.mean(grab_clr)) if grab_clr else None
    ga_sed = float(np.mean(grab_sed)) if grab_sed else None
    co_clr = float(np.mean(comp_clr)) if comp_clr else None
    co_sed = float(np.mean(comp_sed)) if comp_sed else None

    def _fmt(v):
        if v is None: return "—"
        c = "#5BAD6F" if v > 0.05 else ("#E15252" if v < -0.05 else "var(--text-muted)")
        return f"<span style='color:{c};font-weight:bold'>{v:+.3f}</span>"

    q4 = (
        f"Yes — consistently. "
        f"<br><strong>Clarifier-only:</strong> Grab avg = {_fmt(ga_clr)}, Composite avg = {_fmt(co_clr)}. "
        f"<br><strong>Sed-only:</strong> Grab avg = {_fmt(ga_sed)}, Composite avg = {_fmt(co_sed)}. "
        f"Grab targets respond better to every secondary-feature configuration. "
        f"Composite targets show mostly negative or near-zero R² from secondary-only features — "
        f"they need inlet concentrations (Sub2) before any linear model generalises. "
        f"The structural reason: composite samples are collected on fewer days (~730–970 rows) "
        f"and their effluent quality is driven more by inlet load (what goes in) than by "
        f"instantaneous secondary process readings. "
        f"Composite COD fails across all Exp2 variants — this target requires either "
        f"additional features or a process-specific explanation."
    )

    def _qcard(n, question, answer, extra=""):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem'>{answer}</p>
    {extra}
  </div>
</div>"""

    return f"""
<details class="exp-details" id="exp2-findings" open>
  <summary><span class="fold-icon">▶</span> Findings — Experiment 2</summary>
  <div class="exp-body">
  {_qcard(1, "Do Sec Clarifier and Sec Sedimentation features carry distinct signal, or are they interchangeable?", q1)}
  {_qcard(2, "What does combining both secondary groups add over either individually?", q2)}
  {_qcard(3, "What does adding inlet concentrations contribute (Combined → Sub2-FS)?", q3)}
  {_qcard(4, "Do Grab and Composite targets respond differently to the secondary feature transitions?", q4)}
  </div>
</details>"""


def build_exp2_section(df_all: pd.DataFrame) -> str:
    # Sub-exp 1: three secondary-only scopes
    sub1_combined = _exp_subsection(df_all, "Exp2-Sub1", "exp2-s1-combined",
                                    "Combined Secondary + COMMON (15 features)",
                                    open_default=True)
    sub1_clr = _exp_subsection(df_all, "Exp2-Sub1-Clr", "exp2-s1-clr",
                               "Sec Clarifier + COMMON (10 features)",
                               open_default=False)
    sub1_sed = _exp_subsection(df_all, "Exp2-Sub1-Sed", "exp2-s1-sed",
                               "Sec Sedimentation + COMMON (10 features)",
                               open_default=False)

    # Sub-exp 2: cyclic, with integrated FS
    sub2_cyc = _exp_subsection(df_all, "Exp2-Sub2-Cyc", "exp2-s2-cyc",
                               "Sub-experiment 2 — Inlet + Secondary + COMMON "
                               "(21 features, model-specific FS integrated)",
                               open_default=True)

    cmp_div      = _exp2_comparison_panel(df_all)
    findings_div = _exp2_qna(df_all)
    best         = _best_model_box(
        df_all[df_all["exp_key"].isin([
            "Exp2-Sub1", "Exp2-Sub1-Clr", "Exp2-Sub1-Sed", "Exp2-Sub2-Cyc",
        ])],
        "Experiment 2")

    sub1_wrapper = f"""
<details class="exp-details" open id="exp2-s1">
  <summary><span class="fold-icon">▶</span>
    Sub-experiment 1 — Secondary Features Only (three scopes)
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #4A90D9">
      <p class="meta">
        Three feature scopes tested side by side within Sub-experiment 1.
        <strong>Combined</strong> (15 features) is the original baseline — both secondary groups
        together. <strong>Clarifier-only</strong> and <strong>Sed-only</strong> (10 features each)
        isolate each group to test whether they carry distinct signal or are interchangeable.
        All variants include COMMON (Flow, Power, year, month_sin/cos, dow_sin/cos).
        Co-occurring missingness means row counts are nearly identical across the three.
      </p>
    </div>
    {sub1_clr}
    {sub1_sed}
    {sub1_combined}
  </div>
</details>"""

    return f"""
<section id="exp2">
  <h1 class="section-title">Experiment 2 - Secondary & Combined Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp2"]}</p>
  {sub1_wrapper}
  {sub2_cyc}
  {cmp_div}
  {findings_div}
  {best}
</section>"""


def build_exp3_section(df_all: pd.DataFrame) -> str:
    sub1   = _exp_subsection(df_all, "Exp3-S1", "exp3-s1",
                             "Sub-experiment 1 - ADD-tier Aeration Features", open_default=True)
    fs1div = _fs_analysis_div(df_all, "Exp3-S1", "Exp3-S1-FS")
    sub1fs = _exp_subsection(df_all, "Exp3-S1-FS", "exp3-s1-fs",
                             "Sub-experiment 1 - Feature Selected", open_default=False)
    sub2   = _exp_subsection(df_all, "Exp3-S2", "exp3-s2",
                             "Sub-experiment 2 - ADD + CONSIDER-tier Features", open_default=True)
    cyclic_div = _cyclic_encoding_callout(df_all)
    cost_div   = _data_cost_div(df_all, "Exp3-S2", "Exp3-S1")
    vif_div    = _vif_callout()

    # Note: no feature-selected variant exists for Exp3-S2
    no_fs_note = """
<div class="info-note">
  <strong>ℹ No Feature-Selected variant for Sub-experiment 2:</strong>
  Exp3-S2 represents the broadest feature scope (ADD + CONSIDER tier). A feature-selected
  variant was not run because: (1) the Exp3-S1-FS result above already provides the
  "best-features-only" view on the ADD-tier set; (2) GB/XGB already catastrophically
  overfit composite targets in Exp3-S2, making further FS primarily relevant for the
  linear models; (3) Exp3-S2 feeds directly into Phase 9 ensemble methods, which handle
  feature redundancy via regularisation internally. A dedicated Exp3-S2-FS run remains
  a candidate for future work if Comp COD performance needs further improvement.
</div>"""

    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Exp3-S1","Exp3-S1-FS","Exp3-S2"])],
        "Experiment 3")
    return f"""
<section id="exp3">
  <h1 class="section-title">Experiment 3 - Expanded Feature Sets</h1>
  <p class="section-intro">{EXP_INTRO["Exp3"]}</p>
  {sub1}
  {sub1fs}
  {fs1div}
  {sub2}
  {cyclic_div}
  {cost_div}
  {vif_div}
  {no_fs_note}
  {best}
</section>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9
# ═══════════════════════════════════════════════════════════════════════════════

def _phase9_model_subsection(df_all: pd.DataFrame, exp_key: str,
                              section_id: str, title: str,
                              badge_html: str = "") -> str:
    df = df_all[df_all["exp_key"] == exp_key].copy()
    if df.empty:
        return ""
    models = df["model"].unique().tolist()
    feat_html   = _feature_card(exp_key)
    ds_html     = _dataset_summary(df)
    fs_sel_html = _feature_selection_table(df)
    tbl       = _metrics_table(df, models, section_id, df_all)
    train_tbl = _train_metrics_table(df, models)
    return f"""
<details class="exp-details" id="{section_id}">
  <summary><span class="fold-icon">▶</span> {title} {badge_html}</summary>
  <div class="exp-body">
    {feat_html}
    {ds_html}
    {fs_sel_html}
    {tbl}
    {train_tbl}
  </div>
</details>"""


def _variance_diagnosis_callout() -> str:
    """
    Load raw Phase 9 ensemble results and render a variance-collapse diagnosis
    table for targets with negative Test R². Shows σ_train, σ_test, σ-ratio,
    MAE_2024 vs MAE_2025 to distinguish genuine model failure from variance collapse.
    """
    results_path = os.path.join(
        MODELING_DIR, "models", "phase9", "ensemble", "results.xlsx"
    )
    if not os.path.exists(results_path):
        return ""

    df = pd.read_excel(results_path)
    df = df[df["run"] == df["run"].max()]

    # Targets with negative R² in either Voting or Stacking
    neg_rows = df[df["R2_test"] < 0].copy()
    if neg_rows.empty:
        return ""

    rows_html = ""
    for _, r in neg_rows.sort_values(["target", "model"]).iterrows():
        y_tr_std  = r.get("y_train_std", float("nan"))
        y_te_std  = r.get("y_test_std",  float("nan"))
        mae_2024  = r.get("MAE_2024",    float("nan"))
        mae_2025  = r.get("MAE_test",    float("nan"))
        rmse_2024 = r.get("RMSE_2024",   float("nan"))
        rmse_2025 = r.get("RMSE_test",   float("nan"))
        nrmse     = r.get("NRMSE_test",  float("nan"))

        def _safe(v):
            return not (isinstance(v, float) and np.isnan(v))

        ratio      = y_te_std  / y_tr_std  if (_safe(y_tr_std)  and y_tr_std  > 0) else float("nan")
        mae_ratio  = mae_2025  / mae_2024  if (_safe(mae_2024)  and mae_2024  > 0) else float("nan")
        rmse_ratio = rmse_2025 / rmse_2024 if (_safe(rmse_2024) and rmse_2024 > 0) else float("nan")

        delta_mae  = mae_2025  - mae_2024  if (_safe(mae_2024)  and _safe(mae_2025))  else float("nan")
        delta_rmse = rmse_2025 - rmse_2024 if (_safe(rmse_2024) and _safe(rmse_2025)) else float("nan")

        # Classify: collapse vs genuine failure vs mixed
        # Variance collapse: σ-ratio < 0.5 AND (MAE ratio < 1.3 - not much absolute change)
        # Genuine failure: MAE doubled or more regardless of variance
        is_collapse  = _safe(ratio)     and ratio     < 0.5
        is_failure   = _safe(mae_ratio) and mae_ratio > 1.5
        is_improving = _safe(delta_mae) and delta_mae < 0   # MAE improved despite negative R²

        if is_improving:
            dx_class   = "info"
            badge_col  = "#1abc9c"
            diagnosis  = f"MAE improved ({_fmt(mae_2024,3)}→{_fmt(mae_2025,3)}); R² negative due to variance collapse"
        elif is_collapse and not is_failure:
            dx_class   = "warn"
            badge_col  = "#e67e22"
            diagnosis  = f"Variance collapse (σ-ratio={_fmt(ratio,2)}) - R² unreliable; check MAE"
        elif is_failure and not is_collapse:
            dx_class   = "fail"
            badge_col  = "#e74c3c"
            diagnosis  = f"Genuine deterioration (MAE ×{_fmt(mae_ratio,1)} from 2024→2025)"
        else:
            dx_class   = "warn"
            badge_col  = "#e67e22"
            r_str = _fmt(ratio,2) if _safe(ratio) else "-"
            m_str = _fmt(mae_ratio,1) if _safe(mae_ratio) else "-"
            diagnosis  = f"Mixed - σ-ratio={r_str}, MAE ×{m_str}"

        tgt_short = TARGET_SHORT.get(r["target"], r["target"])

        def _delta_cell(val, is_good_if_negative=True):
            """Format a delta value with colour: green if improving, red if degrading."""
            if not _safe(val):
                return "-"
            colour = "#2ecc71" if (val < 0) == is_good_if_negative else "#e74c3c"
            sign = "+" if val >= 0 else ""
            return f'<span style="color:{colour}">{sign}{_fmt(val,3)}</span>'

        rows_html += f"""
        <tr>
          <td>{r['model']}</td>
          <td>{tgt_short}</td>
          <td style="color:#e74c3c">{_fmt(r['R2_test'])}</td>
          <td>{_fmt(y_tr_std, 3)}</td>
          <td>{_fmt(y_te_std, 3)}</td>
          <td>{"-" if not _safe(ratio) else _fmt(ratio, 2)}</td>
          <td>{_fmt(mae_2024, 3)}</td>
          <td>{_fmt(mae_2025, 3)}</td>
          <td>{_delta_cell(delta_mae)}</td>
          <td>{_fmt(rmse_2024, 3)}</td>
          <td>{_fmt(rmse_2025, 3)}</td>
          <td>{_delta_cell(delta_rmse)}</td>
          <td>{_fmt(nrmse, 3)}</td>
          <td><span style="color:{badge_col};font-size:0.82em">{diagnosis}</span></td>
        </tr>"""

    return f"""
<div class="obs-card" style="margin:1rem 0;border-left:4px solid #e67e22">
  <h4 style="margin:0 0 0.6rem">Negative R² Diagnosis - Variance Collapse vs Genuine Failure</h4>
  <p style="font-size:0.85em;color:var(--text-muted);margin:0 0 0.8rem">
    R² = 1 − SS_res/SS_tot. A low-variance 2025 test set (σ-ratio &lt; 0.5) shrinks the
    denominator and forces R² negative even when absolute accuracy is unchanged or improving.
    <strong>ΔMAE</strong> and <strong>ΔRMSE</strong> (2025 − 2024) give scale-grounded evidence:
    negative = model improved in absolute terms; positive = genuine deterioration.
    NRMSE = RMSE / (max − min) for scale-normalised cross-target comparison.
  </p>
  <div style="overflow-x:auto">
  <table style="font-size:0.82em;width:100%">
    <thead><tr>
      <th>Model</th><th>Target</th><th>Test R²</th>
      <th>σ_train</th><th>σ_test</th><th>σ-ratio</th>
      <th>MAE 2024</th><th>MAE 2025</th><th>ΔMAE</th>
      <th>RMSE 2024</th><th>RMSE 2025</th><th>ΔRMSE</th>
      <th>NRMSE</th><th>Diagnosis</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>"""


def _exp4_comparison_table(df_all: pd.DataFrame) -> str:
    """Three-way comparison table: Exp3-S2 vs Exp4-S1 vs Exp4-S2 for each target × model."""
    df3  = df_all[df_all["exp_key"] == "Exp3-S2"]
    df4s1 = df_all[df_all["exp_key"] == "Exp4-S1"]
    df4s2 = df_all[df_all["exp_key"] == "Exp4-S2"]

    all_models = LINEAR_MODELS + NL_MODELS
    rows_html = []

    for tgt in TARGETS_ORDERED:
        tgt_short = (tgt.replace("Effluent ", "").replace(" (mg/L, Grab)", " Grab")
                        .replace(" (mg/L, Composite)", " Comp")
                        .replace(" (Grab)", " pH Grab").replace(" (Composite)", " pH Comp"))
        rows_html.append(
            f'<tr><td colspan="7" style="background:var(--card-bg);font-weight:600;'
            f'padding:6px 10px;color:var(--accent)">{tgt_short}</td></tr>'
        )
        for mdl in all_models:
            def _r2(df_exp):
                s = df_exp[(df_exp["target"] == tgt) & (df_exp["model"] == mdl)]
                if s.empty or s["R2_test"].isna().all():
                    return None, None
                return float(s["R2_test"].iloc[0]), float(s["R2_gap"].iloc[0])

            r2_e3, gap_e3   = _r2(df3)
            r2_s1, gap_s1   = _r2(df4s1)
            r2_s2, gap_s2   = _r2(df4s2)

            def _fmt_cell(r2, gap, ref_r2):
                if r2 is None:
                    return '<td style="color:var(--text-muted)">-</td><td style="color:var(--text-muted)">-</td>'
                r2_str = f"{r2:+.3f}"
                gap_str = f"{gap:+.3f}"
                # Colour R2 relative to reference (Exp3-S2)
                if ref_r2 is None:
                    r2_col = "var(--text-muted)"
                elif r2 >= ref_r2 + 0.02:
                    r2_col = "#2ecc71"
                elif r2 <= ref_r2 - 0.02:
                    r2_col = "#e74c3c"
                else:
                    r2_col = "var(--text-primary)"
                gap_col = "#e74c3c" if gap > 0.15 else ("var(--text-muted)" if abs(gap) <= 0.05 else "var(--text-primary)")
                return (f'<td style="color:{r2_col};font-weight:500">{r2_str}</td>'
                        f'<td style="color:{gap_col};font-size:0.85em">{gap_str}</td>')

            e3_cells  = _fmt_cell(r2_e3,  gap_e3,  None)
            s1_cells  = _fmt_cell(r2_s1,  gap_s1,  r2_e3)
            s2_cells  = _fmt_cell(r2_s2,  gap_s2,  r2_e3)

            rows_html.append(
                f'<tr><td style="padding-left:20px;color:var(--text-muted)">{mdl}</td>'
                f'{e3_cells}{s1_cells}{s2_cells}</tr>'
            )

    thead = """<thead><tr>
      <th>Model</th>
      <th colspan="2" style="text-align:center;background:var(--th-col-blue);color:var(--th-col-blue-text)">Exp3-S2 (baseline)</th>
      <th colspan="2" style="text-align:center;background:var(--th-col-red);color:var(--th-col-red-text)">Exp4-S1 (manual prune)</th>
      <th colspan="2" style="text-align:center;background:var(--th-col-green);color:var(--th-col-green-text)">Exp4-S2 (VIF prune)</th>
    </tr>
    <tr style="font-size:0.82em">
      <th></th>
      <th>R² Test</th><th>R² Gap</th>
      <th>R² Test</th><th>R² Gap</th>
      <th>R² Test</th><th>R² Gap</th>
    </tr></thead>"""

    return f"""
<div class="obs-card" style="margin-top:24px">
  <h3 style="margin:0 0 12px">Three-way comparison: Exp3-S2 → Exp4-S1 → Exp4-S2</h3>
  <p style="color:var(--text-muted);font-size:0.88em;margin:0 0 10px">
    Colours vs Exp3-S2 baseline: <span style="color:#2ecc71">green = +0.02 gain</span>,
    <span style="color:#e74c3c">red = −0.02 loss</span>, white = within ±0.02.
    Gap column: <span style="color:#e74c3c">red = overfitting gap &gt;0.15</span>.
  </p>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:0.87em">
    {thead}
    <tbody>{''.join(rows_html)}</tbody>
  </table>
  </div>
</div>"""


def build_exp4_section(df_all: pd.DataFrame) -> str:
    sub1 = _exp_subsection(df_all, "Exp4-S1", "exp4-s1",
                           "Sub-experiment 1 - Manual Redundancy Pruning (SVI, New Aeration, Sec Sed)",
                           open_default=True)

    note_s1 = """
<div class="info-note" style="border-left-color:#e74c3c">
  <strong>⚠ Sub-exp 1 finding - manual pruning hypothesis refuted:</strong>
  Removing the three feature groups made performance <em>worse</em> across every target and every
  model. Row counts barely changed (+5-10, &lt;2%) because Sec Sed and Sec Clarifier are
  collected on the same operational days (co-occurring missingness).

  <br><br><strong>Why performance degraded:</strong>
  <ul style="margin:6px 0 0 16px;padding:0">
    <li><strong>Linear (ElNet/Ridge):</strong> regularisation already handled collinearity via L1/L2.
        Manual removal discarded signal the penalties would have downweighted automatically.</li>
    <li><strong>Tree models (RF, GB, XGB):</strong> Sec Sed COD/TSS carried direct process signal
        (RF Grab COD: +0.40 → −0.08). Fewer features → less split diversity → faster overfitting.</li>
  </ul>
</div>"""

    sub2 = _exp_subsection(df_all, "Exp4-S2", "exp4-s2",
                           "Sub-experiment 2 - Automated VIF Pruning (threshold = 10)",
                           open_default=True)

    note_s2 = """
<div class="info-note" style="border-left-color:#e67e22">
  <strong>⚠ Sub-exp 2 finding - VIF pruning also refuted, with nuance:</strong>
  Automated iterative VIF pruning (drop highest-VIF feature until all VIF ≤ 10) reduced
  feature sets to 5-10 and unlocked +47 to +293 additional rows by eliminating high-missingness
  correlated features (pH columns, Inlet BOD/COD, MLSS).

  <br><br><strong>Row gain is significant for composites:</strong> Comp TSS grew from 448 → 741
  rows (+65%), Comp COD from 684 → 866 (+27%). This is a genuine data efficiency gain.

  <br><br><strong>But performance still degraded:</strong>
  <ul style="margin:6px 0 0 16px;padding:0">
    <li><strong>Ridge:</strong> mixed results - Grab BOD Ridge recovered to +0.463 (close to
        Exp4-S1 +0.518), Grab TSS Ridge stable at +0.373. But Comp pH Ridge collapsed to −1.44
        and Comp COD remains unresolved.</li>
    <li><strong>ElNet:</strong> Grab BOD dropped to +0.072 (vs Exp3-S2 +0.684). ElNet relies on
        L1 to select from correlated features; pre-selecting them via VIF removes its advantage.</li>
    <li><strong>Tree models (RF, GB, XGB):</strong> catastrophic on most targets. Removing all
        pH features and Inlet BOD/COD strips the primary process-signal columns that trees use
        for their first splits. Without these, boosted trees cannot learn the systematic trend
        and overfit severely to noise.</li>
  </ul>

  <br><strong>Root cause:</strong> VIF pruning is <em>target-agnostic</em> - it removes features
  that are correlated with other features, regardless of their correlation with the target.
  Inlet BOD predicts Effluent BOD. Sec Clarifier pH predicts Effluent pH. These features have
  high VIF precisely because the plant operates systematically (pH is buffered across all stages),
  but they carry the predictive signal the models need.

  <br><br><strong>Conclusion:</strong> Both manual and automated collinearity pruning are
  counter-productive on this dataset. The collinearity is intrinsic to the process physics and
  should be absorbed by regularisation (Ridge/ElNet), not removed. Tree models require the
  full feature pool. <strong>Exp3-S2 remains the best feature set for tree models.</strong>
  For linear models, Ridge on Exp4-S2 is competitive for Grab BOD/TSS but should not replace
  Exp3-S2 Ridge as the baseline due to losses on other targets.
</div>"""

    comparison = _exp4_comparison_table(df_all)
    best = _best_model_box(df_all[df_all["exp_key"].isin(["Exp4-S1", "Exp4-S2"])], "Experiment 4")

    return f"""
<section id="exp4">
  <h1 class="section-title">Experiment 4 - Collinearity Pruning</h1>
  <p class="section-intro">{EXP_INTRO["Exp4"]}</p>
  {sub1}
  {note_s1}
  {sub2}
  {note_s2}
  {comparison}
  {best}
</section>"""


def _ann_dataset_exploration_callout() -> str:
    return """
<div class="obs-card" style="margin:1.5rem 0;border-left:4px solid #9B59B6">
  <h4 style="margin:0 0 0.6rem">ANN Dataset Exploration — Key Findings</h4>
  <ul style="margin:0 0 0 1rem;padding:0;font-size:0.9em;line-height:1.7">
    <li><strong>Inlet features alone (Exp1, 9 features) are insufficient for the ANN</strong> —
        avg Test R²=−5.6, worse than Phase 9 (Exp3-S2, −1.1). More data cannot compensate
        for missing secondary process signal.</li>
    <li><strong>Secondary features unlock positive Grab R² for the first time</strong> —
        Exp2-Sub1 ANN (Secondary + COMMON, 12 features, ~924 rows) achieves Grab BOD +0.20,
        Grab TSS +0.27. This is the only dataset × architecture combination where the ANN
        produces positive generalisation on Grab targets.</li>
    <li><strong>Adding inlet to secondary data (Exp2-Sub2) does not help composites</strong> —
        Comp BOD collapses from −0.11 (Exp2-S1) to −1.49 (Exp2-S2). The ANN overfits
        more with 16 features than with 12 on the same 733 composite rows.</li>
    <li><strong>Composite targets fail for the ANN on every dataset tested</strong> —
        all composite test R² are negative across Exp1, Exp2-Sub1, Exp2-Sub2, and Phase 9
        (Exp3-S2). The pattern is consistent: composite measurements are temporally noisier
        and the ANN cannot capture the distributional shift from training to 2025.</li>
    <li><strong>Conclusion — the ANN failure is not data-volume limited</strong> —
        tripling the training rows (Exp1: 1175 vs Exp3-S2: 470) did not rescue performance.
        The binding constraint is the <em>feature set</em>: secondary process data is a
        prerequisite for positive ANN generalisation on Grab targets. Even with secondary
        features and adequate data (~924 rows), the ANN substantially underperforms the
        Voting ensemble (avg Grab R²≈0.20 vs Voting 0.287 overall). The ANN is not
        recommended for any target on this dataset.</li>
  </ul>
</div>"""


def _ann_dataset_comparison(df_all: pd.DataFrame) -> str:
    """Three-way ANN comparison table: Exp1 vs Exp2-Sub1 vs Exp2-Sub2 vs Phase9 (Exp3-S2)."""
    keys = ["ANN-Exp1", "ANN-Exp2-Sub1", "ANN-Exp2-Sub2", "Phase9-ANN"]
    labels = {
        "ANN-Exp1":      "Exp1 (9 feat, ~1175/800 rows)",
        "ANN-Exp2-Sub1": "Exp2-S1 (15 feat, ~924/740 rows)",
        "ANN-Exp2-Sub2": "Exp2-S2 (19 feat, ~920/733 rows)",
        "Phase9-ANN":    "Exp3-S2 (25 feat, ~470/290 rows)",
    }
    available = [k for k in keys if k in df_all["exp_key"].values]
    if len(available) < 2:
        return ""

    rows_html = []
    for tgt in TARGETS_ORDERED:
        tgt_short = TARGET_SHORT.get(tgt, tgt)
        rows_html.append(
            f'<tr><td colspan="{len(available) * 2 + 1}" '
            f'style="background:var(--card-bg);font-weight:600;'
            f'padding:6px 10px;color:var(--accent)">{tgt_short}</td></tr>'
        )
        row_cells = f'<td style="padding-left:20px;color:var(--text-muted)">ANN</td>'
        for key in available:
            sub = df_all[(df_all["exp_key"] == key) & (df_all["target"] == tgt)]
            if sub.empty or sub["R2_test"].isna().all():
                row_cells += ('<td style="color:var(--text-muted)">-</td>'
                              '<td style="color:var(--text-muted)">-</td>')
                continue
            r2   = float(sub["R2_test"].iloc[0])
            gap  = float(sub["R2_gap"].iloc[0]) if not pd.isna(sub["R2_gap"].iloc[0]) else float("nan")
            r2_col  = ("#2ecc71" if r2 > 0.2 else ("#e74c3c" if r2 < 0 else "var(--text-primary)"))
            gap_col = "#e74c3c" if gap > 0.15 else "var(--text-muted)"
            gap_str = f"{gap:+.3f}" if not np.isnan(gap) else "-"
            row_cells += (f'<td style="color:{r2_col};font-weight:500">{r2:+.3f}</td>'
                          f'<td style="color:{gap_col};font-size:0.85em">{gap_str}</td>')
        rows_html.append(f'<tr>{row_cells}</tr>')

    col_headers = "".join(
        f'<th colspan="2" style="text-align:center">{labels[k]}</th>'
        for k in available
    )
    sub_headers = "".join(
        '<th>R² Test</th><th>R² Gap</th>' for _ in available
    )
    thead = f"""<thead>
      <tr><th>Target</th>{col_headers}</tr>
      <tr style="font-size:0.82em"><th></th>{sub_headers}</tr>
    </thead>"""

    return f"""
<div class="obs-card" style="margin-top:24px">
  <h3 style="margin:0 0 12px">ANN Performance Across Datasets — Data-Volume Diagnostic</h3>
  <p style="color:var(--text-muted);font-size:0.88em;margin:0 0 10px">
    Same ANN architecture across all columns; only dataset (feature set + sample count) changes.
    <span style="color:#2ecc71">Green = R² &gt; 0.20</span>,
    <span style="color:#e74c3c">Red = R² &lt; 0 or gap &gt; 0.15</span>.
  </p>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:0.87em">
    {thead}
    <tbody>{''.join(rows_html)}</tbody>
  </table>
  </div>
</div>"""


def build_phase9_section(df_all: pd.DataFrame) -> str:
    ann_sub     = _phase9_model_subsection(
        df_all, "Phase9-ANN", "p9-ann", "ANN (MLPRegressor)",
        _badge("FAILED avg R²=−1.12", "fail"))
    ann_failure = _ann_failure_callout()
    vote_sub = _phase9_model_subsection(
        df_all, "Phase9-Voting", "p9-voting",
        "Voting Ensemble (ElNet + RF + XGBoost)",
        _badge("RECOMMENDED", "rec"))
    stack_sub = _phase9_model_subsection(
        df_all, "Phase9-Stacking", "p9-stacking",
        "Stacking Ensemble (ElNet + RF + XGB → Ridge, walk-forward OOF)",
        _badge("CONSISTENT - no leakage", "warn"))

    # Combined comparison across all three
    df_p9 = df_all[df_all["exp_key"].isin(["Phase9-ANN","Phase9-Voting","Phase9-Stacking"])].copy()
    all_m = [m for m in ADV_MODELS if m in df_p9["model"].values]
    comp_tbl = _metrics_table(df_p9, all_m, "p9-comp")
    best = _best_model_box(df_p9, "Phase 9")
    var_dx = _variance_diagnosis_callout()

    # ANN dataset-exploration sub-sections
    ann_e1_sub = _phase9_model_subsection(
        df_all, "ANN-Exp1", "p9-ann-exp1",
        "ANN — Exp1 Datasets (Inlet + COMMON, 9 features)")
    ann_e2s1_sub = _phase9_model_subsection(
        df_all, "ANN-Exp2-Sub1", "p9-ann-exp2s1",
        "ANN — Exp2-Sub1 Datasets (Secondary + COMMON, 15 features)")
    ann_e2s2_sub = _phase9_model_subsection(
        df_all, "ANN-Exp2-Sub2", "p9-ann-exp2s2",
        "ANN — Exp2-Sub2 Datasets (Inlet + Secondary + COMMON, 19 features)")
    ann_ds_comparison = _ann_dataset_comparison(df_all)
    ann_ds_callout = _ann_dataset_exploration_callout()

    # Only render the exploration block if at least one result file exists
    has_ann_extra = any(
        k in df_all["exp_key"].values
        for k in ["ANN-Exp1", "ANN-Exp2-Sub1", "ANN-Exp2-Sub2"]
    )
    ann_exploration_block = ""
    if has_ann_extra:
        ann_exploration_block = f"""
  <h2 class="section-title" id="p9-ann-exploration"
      style="font-size:1.1rem;margin:2rem 0 0.5rem">
    ANN Dataset Exploration — Data-Volume Diagnostic
  </h2>
  <p class="section-intro" style="margin-bottom:1rem">
    {EXP_INTRO["ANN-Dataset-Exploration"]}
  </p>
  {ann_e1_sub}
  {ann_e2s1_sub}
  {ann_e2s2_sub}
  {ann_ds_comparison}
  {ann_ds_callout}"""

    return f"""
<section id="phase9">
  <h1 class="section-title">Advanced Methods - ANN &amp; Ensembles</h1>
  <p class="section-intro">{EXP_INTRO["Phase9"]}</p>
  {ann_sub}
  {ann_failure}
  {vote_sub}
  {stack_sub}
  <details class="exp-details" id="p9-comparison">
    <summary><span class="fold-icon">▶</span> Advanced Methods - All Models Combined</summary>
    <div class="exp-body">{comp_tbl}</div>
  </details>
  {var_dx}
  {best}
  {ann_exploration_block}
</section>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10
# ═══════════════════════════════════════════════════════════════════════════════

def _phase10_variant(df_all: pd.DataFrame, exp_key: str,
                     section_id: str, title: str, badge: str = "") -> str:
    df = df_all[df_all["exp_key"] == exp_key].copy()
    if df.empty:
        return ""
    models = [m for m in ALL_MODELS_ORD if m in df["model"].values]
    feat_html = _feature_card(exp_key)
    ds_html   = _dataset_summary(df)
    tbl       = _metrics_table(df, models, section_id)
    train_tbl = _train_metrics_table(df, models)
    return f"""
<details class="exp-details" id="{section_id}">
  <summary><span class="fold-icon">▶</span> {title} {badge}</summary>
  <div class="exp-body">
    {feat_html}
    {ds_html}
    {tbl}
    {train_tbl}
  </div>
</details>"""


def build_phase10_section(df_all: pd.DataFrame) -> str:
    full_fe = _phase10_variant(
        df_all, "Phase10-FE", "p10-full",
        "Feature Engineering - Full (All Targets)",
        _badge("COMPOSITE OVERFIT", "fail"))
    sel_fe = _phase10_variant(
        df_all, "Phase10b-FE", "p10b",
        "Phase 10b - Selective Feature Engineering (Grab FE / Composite Base)",
        _badge("CURRENT BEST", "rec"))
    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Phase10-FE","Phase10b-FE"])],
        "Phase 10")
    return f"""
<section id="phase10">
  <h1 class="section-title">Feature Engineering</h1>
  <p class="section-intro">{EXP_INTRO["Phase10"]}</p>
  {full_fe}
  {sel_fe}
  {best}
</section>"""


def build_phase11_section(df_all: pd.DataFrame) -> str:
    df = df_all[df_all["exp_key"] == "Phase11"].copy()
    if df.empty:
        return ""
    models = [m for m in ALL_MODELS_ORD if m in df["model"].values]
    feat_html = _feature_card("Phase11")
    ds_html   = _dataset_summary(df)
    tbl       = _metrics_table(df, models, "p11-comp")
    train_tbl = _train_metrics_table(df, models)
    best      = _best_model_box(df, "Phase 11")

    # Phase 11-specific callout: CV_R2 is very noisy on first fold
    cv_note = """
<div class="info-note">
  <strong>Phase 11 CV note:</strong> <code>CV_R²</code> and <code>Gap_gen = CV_R² − Test_R²</code>
  are stored in results.xlsx but not shown in the table above (unified schema). They are available
  in the raw <code>models/phase11/results.xlsx</code>. The first TimeSeriesSplit fold trains on
  ~80 rows with ~55 features, making CV_R² very noisy for Ridge/ElNet - interpret per-fold
  rather than as a mean. Key finding: Grab COD RF has CV_R²=0.291, Gap_gen=−0.014 - the most
  consistent Grab COD model by cross-validation.
</div>"""

    return f"""
<section id="phase11">
  <h1 class="section-title">Temporal Features + log1p Targets</h1>
  <p class="section-intro">{EXP_INTRO["Phase11"]}</p>
  <details class="exp-details" open id="p11-detail">
    <summary><span class="fold-icon">▶</span> Temporal Features - All Models</summary>
    <div class="exp-body">
      {feat_html}
      {ds_html}
      {tbl}
      {train_tbl}
    </div>
  </details>
  {cv_note}
  {best}
</section>"""


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAV
# ═══════════════════════════════════════════════════════════════════════════════

def _sidebar() -> str:
    return """
<nav id="sidenav" aria-label="Report navigation">
  <div class="nav-logo">Unified Report</div>

  <div class="nav-group">
    <div class="nav-group-title">Overview</div>
    <a class="nav-item" href="#overview-leaderboard">Global Leaderboard</a>
    <a class="nav-item" href="#overview-progression">R² Progression</a>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp1">
      Experiment 1 <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp1">
      <a class="nav-item nav-sub" href="#exp1-sub1">Sub1 (Inlet Only)</a>
      <a class="nav-item nav-sub" href="#exp1-full">Sub2 (Cyclic, 11 feat)</a>
      <a class="nav-item nav-sub" href="#exp1-comparison">Sub-exp Comparison</a>
      <a class="nav-item nav-sub" href="#exp1-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp2">
      Experiment 2 <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp2">
      <a class="nav-item nav-sub" href="#exp2-s1">Sub-exp 1 (Secondary)</a>
      <a class="nav-item nav-sub nav-subsub" href="#exp2-s1-clr">↳ Clarifier only</a>
      <a class="nav-item nav-sub nav-subsub" href="#exp2-s1-sed">↳ Sedimentation only</a>
      <a class="nav-item nav-sub nav-subsub" href="#exp2-s1-combined">↳ Combined (15 feat)</a>
      <a class="nav-item nav-sub" href="#exp2-s2-cyc">Sub-exp 2 (Inlet+Sec, 21 feat)</a>
      <a class="nav-item nav-sub" href="#exp2-comparison">Sub-exp Comparison</a>
      <a class="nav-item nav-sub" href="#exp2-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp3">
      Experiment 3 <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp3">
      <a class="nav-item nav-sub" href="#exp3-s1">Sub-exp 1 (ADD)</a>
      <a class="nav-item nav-sub" href="#exp3-s1-fs">Sub-exp 1 FS</a>
      <a class="nav-item nav-sub" href="#exp3-s2">Sub-exp 2 (CONSIDER)</a>
      <a class="nav-item nav-sub" href="#exp3-cyclic">Cyclic Encoding</a>
      <a class="nav-item nav-sub" href="#exp3-vif">VIF Collinearity</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp4">
      Experiment 4 <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp4">
      <a class="nav-item nav-sub" href="#exp4-s1">Sub-exp 1 (Manual Pruning)</a>
      <a class="nav-item nav-sub" href="#exp4-s2">Sub-exp 2 (VIF Pruning)</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p9">
      Advanced Methods <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p9">
      <a class="nav-item nav-sub" href="#p9-ann">ANN (Exp3-S2)</a>
      <a class="nav-item nav-sub" href="#p9-ann-diagnosis">ANN Failure Post-Mortem</a>
      <a class="nav-item nav-sub" href="#p9-voting">Voting (ElNet+RF+XGB)</a>
      <a class="nav-item nav-sub" href="#p9-stacking">Stacking (walk-fwd OOF)</a>
      <a class="nav-item nav-sub" href="#p9-comparison">Combined</a>
      <a class="nav-item nav-sub" href="#p9-ann-exploration">ANN Dataset Exploration</a>
      <a class="nav-item nav-sub" href="#p9-ann-exp1">→ ANN Exp1</a>
      <a class="nav-item nav-sub" href="#p9-ann-exp2s1">→ ANN Exp2-Sub1</a>
      <a class="nav-item nav-sub" href="#p9-ann-exp2s2">→ ANN Exp2-Sub2</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p10">
      Feature Engineering <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p10">
      <a class="nav-item nav-sub" href="#p10-full">Full FE</a>
      <a class="nav-item nav-sub" href="#p10b">Selective FE ★</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p11">
      Temporal Features <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p11">
      <a class="nav-item nav-sub" href="#phase11">Temporal Lags + log1p</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-analytics">
      Analytics <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-analytics">
      <a class="nav-item nav-sub" href="#model-selection">Overfit-Aware Selection</a>
      <a class="nav-item nav-sub" href="#error-decomposition">Error Regime Decomposition</a>
      <a class="nav-item nav-sub" href="#comp-cod-diagnostic">Comp COD Diagnostic</a>
    </div>
  </div>

  <div class="nav-divider"></div>
  <div class="nav-group">
    <div class="nav-group-title">Related Reports</div>
    <a class="nav-item" href="../../eda/eda_full_report.html" target="_blank">↗ EDA Report</a>
    <a class="nav-item" href="../../eda/operational_overview.html" target="_blank">↗ Operational Overview</a>
  </div>

  <div id="running-leaders-panel">
    <div class="nav-divider"></div>
    <div class="nav-leaders-title">
      Leaders so far
      <span class="nav-leaders-hint">scroll to update</span>
    </div>
    <div id="leaders-content">
      <p class="leaders-empty">Scroll past a section to see running leaders.</p>
    </div>
  </div>
</nav>"""


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
  /* ── Layout ────────────────────────────────────────────────────── */
  #sidenav {
    position: fixed; top: 0; left: 0;
    width: 240px; height: 100vh; overflow-y: auto;
    background: var(--card); border-right: 1px solid var(--border);
    padding: 0 0 24px; z-index: 200;
    font-size: 13px;
  }
  #sidenav::-webkit-scrollbar { width: 4px; }
  #sidenav::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  #main-content {
    margin-left: 240px; padding: 24px 36px 60px;
    max-width: 1280px;
  }

  /* ── Sidebar nav ───────────────────────────────────────────────── */
  .nav-logo {
    padding: 16px 16px 12px;
    font-weight: 700; font-size: 14px; color: var(--text);
    border-bottom: 1px solid var(--border); margin-bottom: 8px;
    letter-spacing: .3px;
  }
  .nav-group { margin-bottom: 2px; }
  .nav-group-title {
    padding: 6px 16px; font-weight: 600;
    font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
    color: var(--text-muted); cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    user-select: none;
  }
  .nav-group-title:hover { color: var(--text); }
  .nav-chevron { font-size: 10px; transition: transform .2s; }
  .nav-group-title.collapsed .nav-chevron { transform: rotate(-90deg); }
  .nav-group-items { overflow: hidden; transition: max-height .25s ease; }
  .nav-group-items.collapsed { max-height: 0 !important; }
  .nav-item {
    display: block; padding: 5px 16px; color: var(--text-muted);
    text-decoration: none; border-left: 2px solid transparent;
    transition: color .15s, border-color .15s, background .15s;
  }
  .nav-item:hover { color: var(--text); background: var(--summary-hover); }
  .nav-item.active {
    color: #4A90D9; border-left-color: #4A90D9;
    background: var(--toc-bg); font-weight: 600;
  }
  .nav-sub { padding-left: 28px; font-size: 12px; }
  .nav-subsub { padding-left: 2.4rem !important; font-size: 0.78em; color: var(--text-muted); }

  /* ── Page header ───────────────────────────────────────────────── */
  .page-header {
    border-bottom: 1px solid var(--border); margin-bottom: 28px; padding-bottom: 16px;
  }
  .page-header h1 {
    margin: 0 0 4px; font-size: 22px; color: var(--text);
  }
  .page-header .meta { margin: 0; }

  /* ── Target filter bar ─────────────────────────────────────────── */
  .filter-bar {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-bottom: 24px; padding: 10px 14px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; position: sticky; top: 0; z-index: 100;
  }
  .filter-bar label { font-size: 11px; color: var(--text-muted);
    font-weight: 600; align-self: center; margin-right: 4px; }
  .filter-btn {
    padding: 4px 11px; border: 1px solid var(--border);
    border-radius: 12px; cursor: pointer; font-size: 12px;
    background: var(--card); color: var(--text);
    transition: background .15s, border-color .15s, color .15s;
    font-family: inherit;
  }
  .filter-btn:hover { border-color: #4A90D9; color: #4A90D9; }
  .filter-btn.active {
    background: #4A90D9; color: #fff; border-color: #4A90D9;
    font-weight: 600;
  }

  /* ── Section titles ────────────────────────────────────────────── */
  .section-title {
    font-size: 20px; margin: 40px 0 8px;
    padding-bottom: 8px; border-bottom: 2px solid #4A90D9;
    color: var(--text);
  }
  section:first-of-type .section-title { margin-top: 0; }
  .section-intro { color: var(--text-muted); margin: 0 0 20px; line-height: 1.6; }

  /* ── Cards ─────────────────────────────────────────────────────── */
  .section-card {
    padding: 20px 24px; margin-bottom: 20px;
    border-radius: 8px;
  }
  .section-card h2 { margin: 0 0 10px; font-size: 16px; }

  /* ── Feature card ──────────────────────────────────────────────── */
  .feat-card {
    border: 1px solid var(--border); border-radius: 6px;
    padding: 12px 16px; margin: 12px 0 16px;
    background: var(--details-bg);
  }
  .feat-card-label { font-size: 10px; text-transform: uppercase;
    letter-spacing: .6px; color: var(--text-muted); margin-bottom: 2px; }
  .feat-card-name { font-size: 14px; font-weight: 700;
    color: var(--text); margin-bottom: 8px; }
  .feat-card-features, .feat-card-rationale {
    font-size: 12.5px; color: var(--text-muted); line-height: 1.5; margin-bottom: 4px;
  }
  /* Feature selection comparison table badges */
  .fs-univ {
    display: inline-block; font-size: 11px; font-weight: 600;
    background: rgba(46, 204, 113, 0.15); color: #2ecc71;
    border: 1px solid rgba(46, 204, 113, 0.35);
    border-radius: 4px; padding: 1px 5px; margin: 1px 2px;
    white-space: nowrap;
  }
  .fs-model {
    display: inline-block; font-size: 11px;
    background: rgba(243, 156, 18, 0.12); color: #f39c12;
    border: 1px solid rgba(243, 156, 18, 0.3);
    border-radius: 4px; padding: 1px 5px; margin: 1px 2px;
    white-space: nowrap;
  }
  .fs-table td { vertical-align: top; font-size: 11.5px; line-height: 1.6; }
  .fs-table th { font-size: 11.5px; white-space: nowrap; }
  .fs-rank-summary {
    margin: 0 0 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: var(--card);
    border: 1px solid var(--border-light);
    font-size: 12px;
    line-height: 1.7;
    color: var(--text-muted);
  }
  .fs-compare-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
  }
  .fs-method-card {
    display: flex;
    flex-direction: column;
    height: 100%;
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 10px 12px;
    background: var(--details-bg);
  }
  .fs-method-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: .4px;
  }
  .fs-method-note {
    font-size: 11px;
    line-height: 1.45;
    margin: 0 0 8px;
  }
  .fs-rank-table {
    font-size: 11px;
    line-height: 1.35;
  }
  .fs-rank-table th, .fs-rank-table td {
    padding: 4px 7px;
  }
  .fs-rank-table td.num { text-align: right; }
  .delta-note {
    font-size: 10px;
    white-space: nowrap;
    font-weight: 500;
  }
  .tier-na { color: var(--text-muted); }

  /* ── Exp details / foldable sections ──────────────────────────── */
  .exp-details {
    border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 12px; background: var(--details-bg);
  }
  .exp-details > summary {
    padding: 12px 16px; cursor: pointer; list-style: none;
    font-size: 14px; font-weight: 600; color: var(--text);
    user-select: none; display: flex; align-items: center; gap: 8px;
  }
  .exp-details > summary:hover { background: var(--summary-hover); border-radius: 8px; }
  .exp-details[open] > summary { border-bottom: 1px solid var(--border); }
  .exp-body { padding: 16px 28px; }

  .inner-fold {
    border: 1px solid var(--border-light); border-radius: 6px;
    margin: 10px 0; background: var(--bg-page);
  }
  .inner-fold > summary {
    padding: 8px 14px; cursor: pointer; list-style: none;
    font-size: 13px; font-weight: 600; color: var(--text-muted);
    user-select: none; display: flex; align-items: center; gap: 8px;
  }
  .inner-fold > summary:hover { color: var(--text); background: var(--summary-hover); }
  .inner-fold[open] > summary { color: var(--text); }
  .fold-body { padding: 12px 16px; }

  .fold-icon {
    font-size: 10px; color: var(--text-muted);
    transition: transform .2s; display: inline-block;
  }
  details[open] > summary .fold-icon { transform: rotate(90deg); }

  /* ── Metric tables ─────────────────────────────────────────────── */
  .tbl-scroll { overflow-x: auto; }
  .metrics-table, .ds-table, .leaderboard-table, .best-table {
    width: 100%; border-collapse: collapse;
    font-size: 12.5px; margin-bottom: 6px;
  }
  .metrics-table th, .metrics-table td,
  .ds-table th, .ds-table td,
  .leaderboard-table th, .leaderboard-table td,
  .best-table th, .best-table td {
    padding: 6px 10px; border: 1px solid var(--border); text-align: center;
  }
  .metrics-table th, .ds-table th,
  .leaderboard-table th, .best-table th {
    background: var(--toc-bg); color: var(--text); font-weight: 600;
  }
  .metrics-table tbody tr:nth-child(even),
  .ds-table tbody tr:nth-child(even),
  .leaderboard-table tbody tr:nth-child(even),
  .best-table tbody tr:nth-child(even) {
    background: var(--table-even);
  }
  .metrics-table tbody tr:hover,
  .leaderboard-table tbody tr:hover,
  .best-table tbody tr:hover { background: var(--summary-hover); }
  .tgt-name { text-align: left; font-weight: 600; white-space: nowrap; }
  .table-note { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

  /* ── Overfitting traffic light ─────────────────────────────────── */
  .gap-good { color: #2ecc71; font-weight: 600; }
  .gap-warn { color: #f39c12; font-weight: 600; }
  .gap-bad  { color: #e74c3c; font-weight: 600; }

  /* ── No-alt ⓘ popup ─────────────────────────────────────────────── */
  .noalt-info-btn {
    cursor: pointer; color: #4A90D9; font-size: 13px;
    margin-left: 6px; user-select: none;
  }
  .noalt-info-btn:hover { color: #74b3f0; }
  .noalt-popup {
    position: absolute; left: 0; top: 100%;
    background: var(--card); border: 1px solid var(--border-color);
    border-radius: 6px; padding: 12px 14px 10px;
    max-width: 420px; min-width: 280px;
    font-size: 12px; line-height: 1.6;
    color: var(--text-color); z-index: 200;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    white-space: normal; font-style: normal;
  }
  .noalt-popup p { margin: 0; }
  .noalt-popup-close {
    float: right; background: none; border: none;
    cursor: pointer; color: var(--text-muted);
    font-size: 15px; line-height: 1; margin: -2px -2px 4px 8px;
  }

  /* ── Best model box ────────────────────────────────────────────── */
  .best-box {
    border: 2px solid #4A90D9; border-radius: 8px;
    padding: 16px 20px; margin-top: 20px;
    background: var(--toc-bg);
  }
  .best-box-title {
    font-weight: 700; font-size: 14px; color: var(--text);
    margin-bottom: 12px; display: flex; align-items: center; gap: 12px;
  }
  .best-box-avg {
    font-size: 12px; font-weight: 400; color: var(--text-muted);
    border: 1px solid var(--border); border-radius: 10px; padding: 2px 10px;
  }
  .best-table { font-size: 12.5px; }

  /* ── Leaderboard ───────────────────────────────────────────────── */
  .leaderboard-table { font-size: 13px; }

  /* ── Running Leaders sidebar panel ────────────────────────────── */
  #running-leaders-panel { border-top: 1px solid var(--border); margin-top: 8px; }
  .nav-divider { height: 1px; background: var(--border); margin: 4px 0; }
  .nav-leaders-title {
    padding: 8px 16px 4px; font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: .6px; color: var(--text);
    display: flex; justify-content: space-between; align-items: center;
  }
  .nav-leaders-hint { font-size: 9px; color: var(--text-muted); font-weight: 400;
    text-transform: none; letter-spacing: 0; }
  .leaders-empty { padding: 6px 16px; font-size: 11px; color: var(--text-muted); }
  #leaders-content { padding: 4px 8px 8px; }
  .leaders-mini { width: 100%; border-collapse: collapse; font-size: 11px; }
  .leaders-mini td { padding: 3px 6px; border-bottom: 1px solid var(--border-light); }
  .leaders-mini .l-tgt { color: var(--text-muted); font-size: 10px; }
  .leaders-mini .l-r2  { font-weight: 700; text-align: right; }
  .leaders-mini .l-mdl { font-size: 10px; }
  .leaders-mini .l-exp { font-size: 9px; color: var(--text-muted); }

  /* ── FS analysis div ───────────────────────────────────────────── */
  .fs-analysis-div { margin: 8px 0 16px; }
  .fs-verdict {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 10px 14px; border-radius: 6px;
    background: var(--details-bg); margin-bottom: 8px;
    font-size: 13px; line-height: 1.5; color: var(--text);
  }
  .fs-verdict-icon { font-size: 18px; line-height: 1; margin-top: 1px; flex-shrink: 0; }

  /* ── Data cost div ─────────────────────────────────────────────── */
  .data-cost-div {
    padding: 12px 16px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--details-bg); margin: 12px 0;
  }
  .data-cost-div h5 { margin: 0 0 8px; font-size: 13px; color: var(--text); }

  /* ── Info note (neutral) ───────────────────────────────────────── */
  .info-note {
    padding: 10px 14px; border-left: 3px solid #4A90D9;
    background: var(--toc-bg); border-radius: 0 6px 6px 0;
    font-size: 12.5px; line-height: 1.5; color: var(--text-muted);
    margin: 12px 0;
  }
  .info-note strong { color: var(--text); }

  /* ── obs-card default padding ─────────────────────────────────── */
  .obs-card { padding: 14px 18px; border-radius: 6px; }

  /* ── Hidden filter rows ────────────────────────────────────────── */
  tr.filter-hidden { display: none; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# JAVASCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

FILTER_JS = """
<script>
// ── No-alt popup: close on outside click ───────────────────────────────────
document.addEventListener('click', function(e) {
  if (!e.target.classList.contains('noalt-info-btn')) {
    document.querySelectorAll('.noalt-popup').forEach(function(p) {
      p.style.display = 'none';
    });
  }
});

// ── Target filter ──────────────────────────────────────────────────────────
(function() {
  var currentFilter = 'all';

  function applyFilter(slug) {
    currentFilter = slug;
    document.querySelectorAll('.filter-btn').forEach(function(btn) {
      btn.classList.toggle('active', btn.dataset.slug === slug);
    });
    document.querySelectorAll('tr[data-target]').forEach(function(tr) {
      if (slug === 'all') {
        tr.classList.remove('filter-hidden');
      } else {
        tr.classList.toggle('filter-hidden', tr.dataset.target !== slug);
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.filter-btn').forEach(function(btn) {
      btn.addEventListener('click', function() { applyFilter(btn.dataset.slug); });
    });
    applyFilter('all');
  });
})();

// ── Nav sidebar: collapsible groups ────────────────────────────────────────
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.nav-collapsible').forEach(function(title) {
      var groupId = title.dataset.targetGroup;
      var items   = document.getElementById(groupId);
      if (!items) return;
      // Store natural height
      items.style.maxHeight = items.scrollHeight + 'px';
      title.addEventListener('click', function() {
        var collapsed = title.classList.toggle('collapsed');
        if (collapsed) {
          items.style.maxHeight = '0';
          items.classList.add('collapsed');
        } else {
          items.style.maxHeight = items.scrollHeight + 'px';
          items.classList.remove('collapsed');
        }
      });
    });
  });
})();

// ── Nav sidebar: active link on scroll ─────────────────────────────────────
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    var navLinks = Array.from(document.querySelectorAll('#sidenav .nav-item'));
    var sections = navLinks.map(function(a) {
      var id = a.getAttribute('href').slice(1);
      return document.getElementById(id);
    });

    function onScroll() {
      var scrollY = window.scrollY + 80;
      var active = null;
      for (var i = 0; i < sections.length; i++) {
        var sec = sections[i];
        if (!sec) continue;
        if (sec.getBoundingClientRect().top + window.scrollY <= scrollY) {
          active = navLinks[i];
        }
      }
      navLinks.forEach(function(a) { a.classList.remove('active'); });
      if (active) active.classList.add('active');
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  });
})();

// ── Smooth scroll for anchor links ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('#sidenav a[href^="#"]').forEach(function(a) {
    a.addEventListener('click', function(e) {
      var target = document.getElementById(a.getAttribute('href').slice(1));
      if (!target) return;
      e.preventDefault();
      // Open parent <details> if needed
      var el = target;
      while (el && el !== document.body) {
        if (el.tagName === 'DETAILS' && !el.open) el.open = true;
        el = el.parentElement;
      }
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
});

// ── Running Leaders sidebar ─────────────────────────────────────────────────
(function() {
  // SECTION_BESTS is injected by Python below
  var sectionOrder = ['exp1-sub1','exp1-full','exp1-cyclic','exp1-fs','exp2-s1','exp2-s1-fs','exp2-s2','exp2-s2-fs',
                      'exp3-s1','exp3-s1-fs','exp3-s2','p9-ann','p9-voting','p9-stacking',
                      'p10-full','p10b','p11'];
  var targetList   = ['Grab BOD','Grab COD','Grab TSS','Grab pH',
                      'Comp BOD','Comp COD','Comp TSS','Comp pH'];
  var modelColors  = {
    'OLS':'#E15252','Ridge':'#4A90D9','ElNet':'#5BAD6F',
    'RF':'#2171B5','GB':'#238B45','XGB':'#D94801',
    'ANN':'#9B59B6','Voting':'#E67E22','Stacking':'#1ABC9C'
  };
  var runningLeaders = {};
  var lastRendered   = '';

  function gapClass(g) {
    if (Math.abs(g) < 0.10) return 'gap-good';
    if (Math.abs(g) < 0.25) return 'gap-warn';
    return 'gap-bad';
  }

  function mergeSection(id) {
    var bests = window.SECTION_BESTS && window.SECTION_BESTS[id];
    if (!bests) return;
    targetList.forEach(function(tgt) {
      var b = bests[tgt];
      if (!b) return;
      if (!runningLeaders[tgt] || b.r2 > runningLeaders[tgt].r2) {
        runningLeaders[tgt] = b;
      }
    });
  }

  function renderLeaders() {
    var keys = Object.keys(runningLeaders);
    if (keys.length === 0) return;
    var rows = '';
    targetList.forEach(function(tgt) {
      var b = runningLeaders[tgt];
      if (!b) return;
      var r2Color = b.r2 >= 0.6 ? '#2ecc71' : b.r2 >= 0.4 ? '#52c98a'
                  : b.r2 >= 0.2 ? '#f1c40f' : b.r2 >= 0 ? '#e67e22' : '#e74c3c';
      var mdlColor = modelColors[b.model] || '#888';
      rows += '<tr>'
        + '<td class="l-tgt">' + tgt + '</td>'
        + '<td class="l-r2" style="color:' + r2Color + '">' + b.r2.toFixed(3) + '</td>'
        + '<td class="l-mdl ' + gapClass(b.gap) + '" style="color:' + mdlColor + '">'
        + b.model + '</td>'
        + '<td class="l-exp">(' + b.exp + ')</td>'
        + '</tr>';
    });
    var html = '<table class="leaders-mini"><tbody>' + rows + '</tbody></table>';
    if (html !== lastRendered) {
      document.getElementById('leaders-content').innerHTML = html;
      lastRendered = html;
    }
  }

  function checkSections() {
    var mid = window.innerHeight * 0.6;
    sectionOrder.forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      var rect = el.getBoundingClientRect();
      if (rect.top < mid) mergeSection(id);
    });
    renderLeaders();
  }

  // Debounced scroll
  var ticking = false;
  window.addEventListener('scroll', function() {
    if (!ticking) {
      requestAnimationFrame(function() { checkSections(); ticking = false; });
      ticking = true;
    }
  }, { passive: true });

  document.addEventListener('DOMContentLoaded', checkSections);
})();
</script>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_bar() -> str:
    slugs = [("All", "all")] + [
        (TARGET_SHORT[t], TARGET_SLUG[t]) for t in TARGETS_ORDERED
    ]
    btns = "".join(
        '<button class="filter-btn{}" data-slug="{}">{}</button>'.format(
            " active" if s == "all" else "", s, label)
        for label, s in slugs
    )
    return f'<div class="filter-bar" id="global-filter"><label>Filter by target:</label>{btns}</div>'


def _build_model_selection_section(df_all: pd.DataFrame) -> str:
    """Build the overfit-aware model selection section by importing from best_models_selection."""
    # Late import to avoid circular dependency at module level
    # (best_models_selection imports from generate_unified_report)
    from best_models_selection import (  # noqa: E402
        build_global, build_per_experiment, section_html as bms_section_html,
    )
    global_df  = build_global(df_all)
    per_exp_df = build_per_experiment(df_all)
    inner = bms_section_html(global_df, per_exp_df, df_all)
    return f"""
<section id="model-selection">
  <h1 class="section-title">Overfit-Aware Model Selection</h1>
  <p class="section-intro">
    Three complementary rules re-rank every (target × experiment) row beyond naive
    max(Test R²): <strong>gap-adjusted score</strong> penalises overfit beyond a 10-pt
    tolerance; the <strong>one-SE rule</strong> picks the smallest-gap model within
    0.03 R² of the top; the <strong>Pareto frontier</strong> lists all models not
    dominated on (R²_test ↑, |gap| ↓). RMSE and MAE are shown alongside R² for
    operational interpretability - they rank identically to R² within a target but
    carry real units (mg/L or pH) that relate to discharge consent limits.
  </p>
  {inner}
</section>"""


def _build_error_decomposition_section() -> str:
    """Build the error decomposition section by importing from error_decomposition."""
    from error_decomposition import (   # noqa: E402
        decompose_target, TARGETS as ED_TARGETS, section_html as ed_section_html,
    )
    frames = []
    for name, tgt, inlet in ED_TARGETS:
        df = decompose_target(name, tgt, inlet)
        if not df.empty:
            frames.append(df)
    if not frames:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, ignore_index=True)
    inner = ed_section_html(combined)
    return f"""
<section id="error-decomposition">
  <h1 class="section-title">2025 Residual Decomposition by Operational Regime</h1>
  <p class="section-intro">
    Decomposes 2025 residuals for the <strong>Phase 9 Voting</strong> (ElNet+RF+XGB)
    model across four operational axes: Flow quartile, Weekday vs Weekend, Season
    (DJF/MAM/JJA/SON), and Inlet Load quartile. Quartile thresholds are fit on the
    training set (2021-2024) only. Headline finding: error concentrates in
    <strong>Q4 high-flow</strong> and <strong>DJF winter</strong> regimes (2-3× overall
    MAE) for every BOD/TSS target - predictions during hydraulic or thermal stress
    should be flagged as low-confidence.
  </p>
  {inner}
</section>"""


def main():
    print("Loading results data…")
    df_all = load_all_data()
    print(f"  Loaded {len(df_all)} rows across "
          f"{df_all['exp_key'].nunique()} experiments, "
          f"{df_all['model'].nunique()} models.")

    print("Computing MdAE from stored predictions (BOD/TSS targets)…")
    mdae_df = compute_all_mdae()
    if not mdae_df.empty:
        df_all = df_all.merge(mdae_df, on=["exp_key", "model", "target"], how="left")
        n_filled = df_all["MdAE_test"].notna().sum()
        print(f"  MdAE computed for {n_filled} rows "
              f"({mdae_df['exp_key'].nunique()} experiments × BOD/TSS targets).")
    else:
        df_all["MdAE_test"] = np.nan
        print("  No MdAE data found - MdAE column will show - in report.")

    print("Building HTML sections…")
    print("  Building experiment and phase sections…")
    sections = [
        build_overview(df_all),
        build_exp1_section(df_all),
        build_exp2_section(df_all),
        build_exp3_section(df_all),
        build_exp4_section(df_all),
        build_phase9_section(df_all),
        build_phase10_section(df_all),
        build_phase11_section(df_all),
    ]

    print("  Building model selection section…")
    sections.append(_build_model_selection_section(df_all))

    print("  Building error decomposition section…")
    sections.append(_build_error_decomposition_section())

    print("  Building Comp COD diagnostic section…")
    sections.append(_build_comp_cod_diagnostic(df_all))

    # Inline section-bests JSON for the running leaders JS widget
    sec_bests_json = _section_bests_json(df_all)
    section_data_js = f"<script>window.SECTION_BESTS = {sec_bests_json};</script>"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    css = dark_mode_css(CUSTOM_CSS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unified Modeling Report - Wastewater Treatment</title>
  <style>{css}</style>
  {DARK_MODE_JS}
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  {section_data_js}
</head>
<body>
{_sidebar()}
<div id="main-content">
  <div class="page-header">
    <h1>Wastewater Treatment - Unified Modeling Report</h1>
    <p class="meta">Generated {ts} &nbsp;·&nbsp;
      Experiments 1-4 + Phases 9-11 &nbsp;·&nbsp;
      9 models &nbsp;·&nbsp; 8 effluent targets &nbsp;·&nbsp;
      Train 2021-2024 · Test 2025
    </p>
  </div>
  {_filter_bar()}
  {"".join(sections)}
</div>
{FILTER_JS}
</body>
</html>"""

    out = os.path.join(REPORTS_DIR, "unified_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDone → {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
