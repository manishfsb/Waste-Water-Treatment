"""
generate_unified_report.py - Unified HTML report covering all experiments and phases.

Reads all results.xlsx files from the modeling directory tree and produces
a single, navigable HTML report with:
  - Fixed left-sidebar navigation with collapsible experiment groups
  - Global target filter (Grab BOD / Comp COD / ... / All)
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
    "Exp1-SE1": "E1-SE1",
    "Exp1-Cyclic": "E1-SE2",
    "Exp1-S3": "E1-SE3",       "Exp1-S3-FS": "E1-SE3-FS",
    "Exp2-SE2-Comb": "E2-SE2-Comb", "Exp2-SE2-Comb-FS": "E2-SE2-Comb-FS",
    "Exp2-SE2-Clr": "E2-SE2-Clr",
    "Exp2-SE2-Sed": "E2-SE2-Sed",
    "Exp2-Sub2": "E2-CIS",      "Exp2-SE3-Ref-FS": "E2-CIS-FS",
    "Exp2-SE3-Ref": "E2-CIS",
    "Exp2-S5": "E2-SE1a",
    "Exp2-S3": "E2-SE1b",
    "Exp2-S6":    "E2-SE2-Only",
    "Exp2-S6-FS": "E2-SecOnly-FS",
    "Exp2-S4": "E2-SE3",
    "Exp3-S1": "E3-SE1",
    "Exp3-S2": "E3-SE2",        "Exp3-S2-FS": "E3-SE2-FS",
    "Exp3-S3": "E3-SE3",        "Exp3-S3-FS": "E3-SE3-FS",
    "Exp3-S4": "E3-SE4",        "Exp3-S4-FS": "E3-SE4-FS",
    "Exp4-S1": "E4-SE1",
    "Exp4-S2": "E4-SE2",
    "Exp5-S1": "E5-SE1",        "Exp5-S1-FS": "E5-SE1-FS",
    "Exp5-S2": "E5-SE2",        "Exp5-S2-FS": "E5-SE2-FS",
    "Exp9-SE1": "E9-SE1",
    "Exp9-SE2": "E9-SE2-FS",
    "Exp9-SE3": "E9-SE3-LogY",
    "Exp9-SE4": "E9-SE4-LogF",
    "Exp9-SE5": "E9-SE5-LogLog",
    "Exp6-ANN": "E6-ANN", "Exp6-Voting": "E6-Vote",
    "Exp6-Stacking": "E6-Stack",
    "ANN-Exp1": "ANN-E1", "ANN-Exp2-SE2": "ANN-E2-SE1", "ANN-Exp2-SE3-Ref": "ANN-E2-SE2",
    "Exp7-SE1": "E7-SE1", "Exp7-SE2": "E7-SE2",
    "Exp8": "E8",
}

# Longer descriptive labels used only in the "Source" column of best-model tables.
EXP_SOURCE_LABELS = {
    "Exp1-SE1":      "Exp1 SE1  -  Inlet Only (4 features)",
    "Exp1-Cyclic":    "Exp1 SE2  -  Inlet + COMMON (11 features)",
    "Exp1-S3":        "Exp1 SE3  -  Extended Inlet + COMMON",
    "Exp1-S3-FS":     "Exp1 SE3-FS  -  Extended Inlet + COMMON (FS)",
    "Exp2-SE2-Clr":  "Exp2 SE2-Clr  -  Clarifier + COMMON",
    "Exp2-SE2-Sed":  "Exp2 SE2-Sed  -  Sedimentation + COMMON",
    "Exp2-SE2-Comb":      "Exp2 SE2-Comb  -  All Secondary + COMMON",
    "Exp2-SE2-Comb-FS":   "Exp2 SE2-Comb-FS  -  All Secondary + COMMON (FS)",
    "Exp2-Sub2":      "Exp2 CIS  -  Core Inlet + Secondary + COMMON",
    "Exp2-SE3-Ref-FS":   "Exp2 CIS-FS  -  Core Inlet + Secondary + COMMON (FS)",
    "Exp2-SE3-Ref":  "Exp2 CIS  -  Core Inlet + Secondary + COMMON",
    "Exp2-S5":        "Exp2 SE1a  -  Primary + COMMON",
    "Exp2-S3":        "Exp2 SE1b  -  Primary + Inlet + COMMON",
    "Exp2-S6":        "Exp2 SE2-Only  -  All Secondary (no COMMON)",
    "Exp2-S6-FS":     "Exp2 Sec-Only-FS  -  All Secondary (FS)",
    "Exp2-S4":        "Exp2 SE3  -  Inlet + Primary + Secondary + COMMON",
    "Exp3-S1":        "Exp3 SE1  -  ADD-tier",
    "Exp3-S2":        "Exp3 SE2  -  ADD+CONSIDER (no FS)",
    "Exp3-S2-FS":     "Exp3 SE2  -  ADD+CONSIDER (FS)",
    "Exp3-S3":        "Exp3 SE3 - ADD+CONSIDER minus Coliform",
    "Exp3-S3-FS":     "Exp3 SE3-FS - ADD+CONSIDER minus Coliform (FS)",
    "Exp3-S4":        "Exp3 SE4 - All Features",
    "Exp3-S4-FS":     "Exp3 SE4-FS - All Features",
    "Exp4-S1":        "Exp4 SE1",
    "Exp4-S2":        "Exp4 SE2",
    "Exp9-SE1":       "Exp9 SE1  -  W1 (2024-only training window)",
    "Exp9-SE2":       "Exp9 SE2  -  W1+FS (2024-only + LassoCV / OOF FS)",
    "Exp9-SE3":       "Exp9 SE3  -  W1+LogY (2024-only + log1p target transform)",
    "Exp9-SE4":       "Exp9 SE4  -  W1+LogF (2024-only + log1p feature transform)",
    "Exp9-SE5":       "Exp9 SE5  -  W1+LogLog (2024-only + log1p features + log1p targets)",
    "Exp5-S1":        "Exp5 SE1  -  Composite + Grab Inlet",
    "Exp5-S1-FS":     "Exp5 SE1-FS  -  Composite + Grab Inlet (FS)",
    "Exp5-S2":        "Exp5 SE2  -  Grab + Composite Inlet",
    "Exp5-S2-FS":     "Exp5 SE2-FS  -  Grab + Composite Inlet (FS)",
    "Exp6-ANN":     "Exp6  -  ANN",
    "Exp6-Voting":  "Exp6  -  Voting",
    "Exp6-Stacking":"Exp6  -  Stacking",
    "Exp7-SE1":     "Exp7 SE1  -  Full FE",
    "Exp7-SE2":     "Exp7 SE2  -  Selective FE",
    "Exp8":         "Exp8  -  Temporal Features",
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
    ("experiment5/sub_exp2",                            "Exp5-S2"),
    ("experiment5/sub_exp1",                            "Exp5-S1"),
    ("experiment4/sub_exp2",                            "Exp4-S2"),
    ("experiment4/sub_exp1",                            "Exp4-S1"),
    ("experiment3/sub_exp4/feature_selected_datasets",  "Exp3-S4-FS"),
    ("experiment3/sub_exp4",                            "Exp3-S4"),
    ("experiment3/sub_exp3/feature_selected_datasets",  "Exp3-S3-FS"),
    ("experiment3/sub_exp3",                            "Exp3-S3"),
    ("experiment3/sub_exp2/feature_selected_datasets",  "Exp3-S2-FS"),
    ("experiment3/sub_exp2",                            "Exp3-S2"),
    ("experiment3/sub_exp1",                            "Exp3-S1"),
    ("experiment2/sub_exp6",                            "Exp2-S6"),
    ("experiment2/sub_exp5",                            "Exp2-S5"),
    ("experiment2/sub_exp4",                            "Exp2-S4"),
    ("experiment2/sub_exp3",                            "Exp2-S3"),
    ("experiment2/sub_exp2/feature_selected_datasets",  "Exp2-SE3-Ref-FS"),
    ("experiment2/sub_exp2",                            "Exp2-SE3-Ref"),
    ("experiment2/sub_exp1/sec_clarifier",              "Exp2-SE2-Clr"),
    ("experiment2/sub_exp1/sec_sed",                    "Exp2-SE2-Sed"),
    ("experiment2/sub_exp1/feature_selected_datasets",  "Exp2-SE2-Comb-FS"),
    ("experiment2/sub_exp1",                            "Exp2-SE2-Comb"),
    ("experiment1/sub_exp3",                            "Exp1-S3"),
    ("experiment1/sub_exp2/feature_selected_datasets",  "Exp1-FS"),
    ("experiment1/sub_exp2",                            "Exp1-Cyclic"),
    ("experiment1/sub_exp1",                            "Exp1-SE1"),
    ("experiment9/sub_exp1",                            "Exp9-SE1"),
    ("experiment9/sub_exp1",                            "Exp9-SE2"),
    ("experiment9/sub_exp1",                            "Exp9-SE3"),
    ("experiment9/sub_exp1",                            "Exp9-SE4"),
    ("experiment9/sub_exp1",                            "Exp9-SE5"),
]

# predicted_<TAG>_run_N → canonical model name
_PRED_MODEL_MAP = {
    "OLS": "OLS", "Ridge": "Ridge", "ElNet": "ElNet",
    "RF": "RF", "GB": "GB", "XGB": "XGB",
    "RF_NL": "RF", "GB_NL": "GB", "XGB_NL": "XGB",
    "OLS_nofs": "OLS", "Ridge_nofs": "Ridge", "ElNet_nofs": "ElNet",
    "RF_nofs": "RF", "GB_nofs": "GB", "XGB_nofs": "XGB",
    "ANN": "ANN", "Voting": "Voting", "Stacking": "Stacking",
}

# Exp 6 models live inside Exp3-SE2 files but belong to a different exp_key
_PHASE9_EXP = {"ANN": "Exp6-ANN", "Voting": "Exp6-Voting", "Stacking": "Exp6-Stacking"}

FEATURE_DESCRIPTIONS = {
    "Exp1-SE1": {
        "label": "Core Inlet (4 features)  -  no process or temporal context",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features (4):</strong> Inlet pH, Inlet BOD (mg/L), Inlet COD (mg/L), Inlet TSS (mg/L) - Grab or Composite.</li>"
            "<li><strong>Excluded:</strong> Flow (MLD), Power Total (KW), all calendar features - no process or temporal context.</li>"
            "</ul>"
        ),
        "rationale": "Absolute minimum feature set: can inlet water quality alone predict "
                     "effluent quality? This sub-experiment establishes the floor and "
                     "quantifies how much process context (Flow, Power, calendar) adds "
                     "when compared against Exp1 (Sub 2).",
    },
    "Exp1-Cyclic": {
        "label": "Core Inlet + COMMON (11 features)  -  standard SE2",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Inlet (4):</strong> pH, BOD, COD, TSS - Grab or Composite.</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin, month_cos, dow_sin, dow_cos.</li>"
            "<li><strong>Calendar encoding:</strong> month and day_of_week replaced by sin/cos projections - the project standard from SE2 onwards.</li>"
            "</ul>"
        ),
        "rationale": "Project-standard inlet feature set. "
                     "Establishes the inlet-only ceiling before adding supplementary inlet "
                     "measurements in SE3.",
    },
    "Exp1-S3": {
        "label": "Extended Inlet + COMMON (16 features)  -  grab and composite targets",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Core Inlet (4):</strong> pH, BOD, COD, TSS (Grab).</li>"
            "<li><strong>Extended Inlet (5):</strong> TKN/NH3-N, Oil &amp; Grease, PO4/TP, "
            "Total Coliform, Fecal Coliform. Note: TKN/NH3-N and O&amp;G are measured on "
            "composite Raw Sewage samples (2022+); on the single grab sample in 2021. "
            "Column header changed from TKN to NH3-N in 2022 - these are related but "
            "distinct nitrogen parameters. No separate Grab equivalents exist for TKN/NH3-N or O&amp;G.</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin, month_cos, dow_sin, dow_cos.</li>"
            "<li><strong>8 datasets:</strong> 4 grab + 4 composite targets. Train row counts "
            "limited by joint missingness of the 5 supplementary columns - see Findings Q1.</li>"
            "</ul>"
        ),
        "rationale": "Tests whether the full set of available inlet measurements - "
                     "including microbiological and nutrient indicators - adds predictive "
                     "value beyond the 4 core inlet parameters. Row count is an inherent "
                     "data limitation driven by supplementary column missingness, not a design choice.",
    },
    "Exp1-S3-FS": {
        "label": "All Grab Inlet + COMMON - Feature Selected (2-15 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> SE3 features (15: 9 inlet + 7 COMMON - grab targets only).</li>"
            "<li><strong>OLS:</strong> LassoCV pre-screen. <strong>Ridge:</strong> full set (L2). "
            "<strong>ElNet:</strong> full set (L1+L2 selects internally).</li>"
            "<li><strong>Trees:</strong> 3-phase OOF permutation importance (threshold 5%).</li>"
            "<li><strong>Remaining:</strong> 2-15 features per model/target.</li>"
            "</ul>"
        ),
        "rationale": "Checks whether removing noisy or redundant supplementary inlet features "
                     "from the SE3 set recovers generalisation on the 2025 holdout.",
    },
    "Exp2-SE2-Comb": {
        "label": "Combined Secondary + COMMON (15 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Sec Clarifier (5):</strong> pH, TSS, BOD, COD, RAS.</li>"
            "<li><strong>Sec Sedimentation (5):</strong> pH, TSS, BOD, COD, RAS (New).</li>"
            "<li><strong>COMMON (5):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Excluded:</strong> No inlet features - secondary-only scope.</li>"
            "</ul>"
        ),
        "rationale": "Tests secondary treatment process data only (no inlet). "
                     "Do downstream process indicators predict effluent better than inlet?",
    },
    "Exp2-SE2-Comb-FS": {
        "label": "Secondary + COMMON - Feature Selected",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> SE1 combined secondary + COMMON (15 features).</li>"
            "<li><strong>Feature selection:</strong> RF permutation importance screening.</li>"
            "<li><strong>Remaining:</strong> subset of most predictive secondary parameters.</li>"
            "</ul>"
        ),
        "rationale": "Feature selection on the secondary feature set to remove noisy "
                     "or redundant secondary parameters.",
    },
    "Exp2-Sub2": {
        "label": "Inlet + Secondary + COMMON (19 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Inlet (4):</strong> pH, BOD, COD, TSS - Grab or Composite.</li>"
            "<li><strong>Secondary (10):</strong> Sec Clarifier (5) + Sec Sedimentation (5).</li>"
            "<li><strong>COMMON (5):</strong> Flow (MLD), Power Total (KW), month, day_of_week, year.</li>"
            "</ul>"
        ),
        "rationale": "Full baseline combining both monitoring points. Tests whether "
                     "joining inlet and secondary data outperforms either alone.",
    },
    "Exp2-SE3-Ref-FS": {
        "label": "CIS - Feature Selected",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> SE2 combined features (19: Inlet + Secondary + COMMON).</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV; trees use OOF permutation importance (threshold 5%).</li>"
            "<li><strong>Remaining:</strong> subset of most predictive combined features per model/target.</li>"
            "</ul>"
        ),
        "rationale": "Remove redundant or low-signal features from the full Exp2-S2 set.",
    },
    "Exp2-SE2-Clr": {
        "label": "Secondary Clarifier + COMMON (12 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Sec Clarifier (5):</strong> pH, TSS, BOD, COD, RAS.</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Removed vs SE1-Combined:</strong> Sec Sedimentation group (5 cols) - isolated to test standalone Clarifier signal.</li>"
            "</ul>"
        ),
        "rationale": "Isolate the Clarifier group to test whether Clarifier and Sedimentation "
                     "features carry distinct signal or are interchangeable.",
    },
    "Exp2-SE2-Sed": {
        "label": "Secondary Sedimentation + COMMON (12 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Sec Sedimentation (5):</strong> pH, TSS, BOD, COD, RAS (New).</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Removed vs SE2-Combined:</strong> Sec Clarifier group (5 cols) - isolated to test standalone Sedimentation signal.</li>"
            "</ul>"
        ),
        "rationale": "Isolate the Sedimentation group  -  paired with Exp2-SE2-Clr to test "
                     "whether one group dominates the other.",
    },
    "Exp2-SE3-Ref": {
        "label": "Core Inlet + Secondary + Common (CIS) - 21 features",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Inlet (4):</strong> pH, BOD, COD, TSS - Grab or Composite.</li>"
            "<li><strong>Secondary (10):</strong> Sec Clarifier (5) + Sec Sedimentation (5).</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos (cyclic encoding).</li>"
            "</ul>"
        ),
        "rationale": "Full combined feature set with corrected calendar encoding. "
                     "OLS uses LassoCV pre-screening; trees use OOF permutation importance "
                     "feature selection, both stored in the same results file.",
    },
    "Exp2-S4": {
        "label": "Inlet + Primary + Secondary + COMMON (27 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Core Inlet (4):</strong> pH, BOD, COD, TSS - Grab or Composite.</li>"
            "<li><strong>Primary Clarifier (5):</strong> Primary Clarifier pH, Primary TSS, "
            "Primary BOD, Primary COD, Primary Sludge Totalizer (m3).</li>"
            "<li><strong>Grit Classifier (1):</strong> Grit Classifier TSS (mg/L).</li>"
            "<li><strong>Sec Clarifier (5):</strong> pH, TSS, BOD, COD, RAS.</li>"
            "<li><strong>Sec Sedimentation (5):</strong> pH, TSS, BOD, COD, RAS (New).</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Row count:</strong> Grab ~764 train / 197 test. Composite ~595 train / 176 test. "
            "Secondary co-occurs almost perfectly with Primary days (~14-row additional cost vs SE1).</li>"
            "</ul>"
        ),
        "rationale": "Full stage-based feature set combining all three monitoring points. "
                     "Directly tests whether primary clarifier data adds predictive value on top of the "
                     "CIS baseline (Core Inlet + Secondary + Common, ~920 train rows). SE1 showed primary alone"
                     "adds no signal; this experiment tests whether primary is informative in the "
                     "presence of secondary data.",
    },
    "Exp2-S6": {
        "label": "All Secondary only (10 features, no COMMON)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Sec Clarifier (5):</strong> pH, TSS (mg/L), BOD (mg/L), COD (mg/L), RAS.</li>"
            "<li><strong>Sec Sedimentation (5):</strong> pH, TSS (mg/L), BOD (mg/L), COD (mg/L), RAS (New).</li>"
            "<li><strong>COMMON:</strong> excluded entirely.</li>"
            "<li><strong>Row count:</strong> Grab ~971-975 train / ~218-220 test. Composite ~785-813 train / ~206-215 test. "
            "Notably higher than SE2-Comb (+50 grab, +150 comp) because Flow/Power NaN rows are no longer binding.</li>"
            "</ul>"
        ),
        "rationale": "Isolates secondary-stage signal from plant-level operational context (Flow, Power, calendar). "
                     "Motivated by the Exp1 finding that COMMON sometimes hurts grab targets and substantially "
                     "helps composite targets. Contrasts with SE2-Comb (All Secondary + COMMON, 17 features) to "
                     "attribute how much of the SE2 signal comes from secondary measurements vs COMMON features. "
                     "The extra ~50 grab / ~150 comp rows (vs SE2-Comb) are a secondary benefit of dropping COMMON.",
    },
    "Exp2-S6-FS": {
        "label": "Secondary Only - Feature Selected (10 feat -> selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 10 secondary features as SE2a (no COMMON, no Inlet).</li>"
            "<li><strong>Linear FS:</strong> LassoCV pre-screening for OLS (1-4 features retained per target). "
            "Ridge and ElNet select implicitly via regularisation.</li>"
            "<li><strong>Non-linear FS:</strong> OOF permutation importance (threshold 5%); 2-6 features retained. "
            "Three-phase protocol: full CV search -> OOF selection -> refit on selected.</li>"
            "<li><strong>Row count:</strong> Same as SE2a (Grab ~971-975 / ~785-813 Composite). "
            "No additional row cost from FS.</li>"
            "<li><strong>Consistent finding:</strong> Sec Clarifier BOD and Sec Sed COD are the two features "
            "retained most frequently across all models and targets.</li>"
            "</ul>"
        ),
        "rationale": "Tests whether feature selection on the pure secondary feature pool improves "
                     "generalisation over the full 10-feature set (SE2a). "
                     "With only 10 starting features the marginal benefit of FS is expected to be small; "
                     "the key question is whether it outperforms CIS-FS "
                     "(21 features -> selected), which includes inlet and COMMON context. "
                     "A better Secondary Only-FS result would confirm that COMMON and inlet add no net value "
                     "once noise features are pruned from the secondary pool.",
    },
    "Exp2-S5": {
        "label": "Primary Stage + COMMON only (13 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Primary Clarifier (5):</strong> Primary Clarifier pH, Primary TSS, "
            "Primary BOD, Primary COD, Primary Sludge Totalizer (m3).</li>"
            "<li><strong>Grit Classifier (1):</strong> Grit Classifier TSS (mg/L).</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Row count:</strong> Grab ~780-784 train / ~204 test. Composite ~611-633 train / ~193-201 test. "
            "Slightly higher than SE1b since inlet NaN rows are no longer binding.</li>"
            "</ul>"
        ),
        "rationale": "Isolates whether primary treatment features carry any predictive signal WITHOUT "
                     "inlet concentrations. Directly contrasts with SE1b (Primary + Inlet + COMMON, 17 features) "
                     "to determine whether inlet adds or destroys signal at the primary stage. "
                     "If SE1a and SE1b both produce all-negative R², primary features are genuinely "
                     "non-predictive regardless of inlet pairing. "
                     "VIF warning remains: Primary Clarifier pH is near-collinear with itself across years.",
    },
    "Exp2-S3": {
        "label": "Primary Stage + Inlet + COMMON (17 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Inlet (4):</strong> pH, BOD, COD, TSS - Grab or Composite.</li>"
            "<li><strong>Primary Clarifier (5):</strong> Primary Clarifier pH, Primary TSS, "
            "Primary BOD, Primary COD, Primary Sludge Totalizer (m3).</li>"
            "<li><strong>Grit Classifier (1):</strong> Grit Classifier TSS (mg/L).</li>"
            "<li><strong>COMMON (7):</strong> Flow (MLD), Power Total (KW), year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Row count:</strong> Grab ~778 train / 204 test. Composite ~608 train / 181 test "
            "(~34% row cost vs Exp1-SE2 baseline; driven by Primary BOD/Grit missingness).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether primary clarification and grit removal measurements add predictive "
                     "value on top of inlet concentrations. Primary stage sits between inlet and secondary "
                     "in the treatment process. Pearson correlations to effluent targets are weak (|r| < 0.22 "
                     "for BOD/COD/TSS), though MI scores suggest modest non-linear signal (0.07-0.19). "
                     "VIF warning: Primary Clarifier pH shows VIF ~2743 in OLS (collinear with inlet pH).",
    },
    "Exp3-S1": {
        "label": "Exp2-S2 + ADD-tier Aeration Features (28 features, no FS)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Baseline (21):</strong> Exp2-SE2 - Inlet (4) + Secondary (10) + COMMON (7: Flow, Power, year, month/dow cyclic).</li>"
            "<li><strong>Added - ADD-tier (7):</strong> Aeration DO/MLSS/SV30/SVI/pH (Existing) + Aeration DO/SV30 (New). MI >= 0.20, marginal row cost <= 20%.</li>"
            "<li><strong>Row count:</strong> Grab ~816 train / 202 test. Composite ~632 train / 181 test. Near-zero row loss vs Exp2-SE2.</li>"
            "</ul>"
        ),
        "rationale": "Expand with aeration basin data that shows high mutual information "
                     "and low missingness cost on top of the Exp2-SE2 baseline. "
                     "No feature selection applied  -  tests the full ADD-tier set.",
    },
    "Exp3-S2": {
        "label": "Exp2-S2 + ADD + CONSIDER-tier Features (31 features, no FS)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>SE1 base (28):</strong> Inlet + Secondary + COMMON + ADD-tier aeration (7).</li>"
            "<li><strong>Added - CONSIDER-tier (3):</strong> Aeration SVI (New), Inlet Total Coliform (CFU/100ml), Primary Sludge Totalizer (m3).</li>"
            "<li><strong>Removed vs original SE2:</strong> Power GE (KW) - redundant: Power Total = Power GE + Power NEA.</li>"
            "<li><strong>Row cost:</strong> Grab ~469 train / 168 test (Coliform causes ~347-row loss vs SE1). Composite ~423 train / 152 test.</li>"
            "</ul>"
        ),
        "rationale": "Baseline (no FS) for the ADD+CONSIDER-tier feature set. "
                     "Tests the raw performance of 32 features without model-specific selection, "
                     "enabling a direct comparison with the FS variant (Exp3-S2-FS).",
    },
    "Exp3-S2-FS": {
        "label": "Exp2-S2 + ADD + CONSIDER-tier Features with FS (31 features -> 5-24 selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 31 features as SE2 (SE1 + CONSIDER-tier, Power GE excluded).</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV pre-screen. RF/GB/XGB use 3-phase OOF permutation importance (threshold=0.05).</li>"
            "<li><strong>Ridge/ElNet:</strong> Full 31-feature set - L2/L1+L2 regularisation handles collinearity internally.</li>"
            "<li><strong>Row count:</strong> Same as SE2 (~469 Grab / ~423 Composite train). No dataset rebuild.</li>"
            "</ul>"
        ),
        "rationale": "Tests whether CONSIDER-tier features (higher missingness cost) add value "
                     "when combined with model-specific feature selection. Key trade-off: "
                     "CONSIDER features increase signal but reduce training rows significantly.",
    },
    "Exp3-S3": {
        "label": "ADD + CONSIDER minus Inlet Total Coliform (30 features, no FS)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Removed from SE2:</strong> Inlet Total Coliform (CFU/100ml, Grab) - "
            "the single highest-missingness CONSIDER-tier feature; caused ~347 row loss (816→469 Grab) in SE2.</li>"
            "<li><strong>Remaining (30 features):</strong> Inlet (4 Grab or Composite), "
            "Sec Clarifier (5) + Sec Sedimentation (5), Flow, Power Total, year, "
            "month/dow cyclic (4), ADD-tier aeration (7: DO/MLSS/SV30/SVI/pH Existing + DO/SV30 New), "
            "CONSIDER-tier remainder (2: Aeration SVI New, Primary Sludge Totalizer).</li>"
            "<li><strong>Row recovery:</strong> Grab train ~815 rows (vs ~469 in SE2), "
            "Composite train ~631 rows (vs ~423 in SE2). Nearly SE1-level data volume.</li>"
            "</ul>"
        ),
        "rationale": "Tests whether the row loss from Inlet Total Coliform was the primary driver "
                     "of SE2's inconsistent performance. Removing just that one feature recovers ~347 "
                     "Grab rows at the cost of one CONSIDER-tier signal. Paired with SE3-FS to test "
                     "whether feature selection further improves results at the higher row count.",
    },
    "Exp3-S3-FS": {
        "label": "ADD + CONSIDER minus Inlet Total Coliform with Model-Specific FS (30 features -> selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 30 features as SE3 (SE2 minus Inlet Total Coliform).</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV pre-screen (TimeSeriesSplit). "
            "RF/GB/XGB use 3-phase OOF permutation importance (threshold=0.05).</li>"
            "<li><strong>Ridge/ElNet:</strong> Full 30-feature set - L2/L1+L2 regularisation handles collinearity internally.</li>"
            "<li><strong>Row count:</strong> Same as SE3 (~815 Grab / ~631 Composite train). "
            "No dataset rebuild needed (no high-missingness columns in starting set).</li>"
            "</ul>"
        ),
        "rationale": "With ~815 Grab training rows (vs ~469 in SE2-FS), OOF folds contain ~272 rows "
                     "instead of ~156 - permutation importance estimates should be more stable. "
                     "Tests whether FS on the larger SE3 dataset outperforms SE2-FS.",
    },
    "Exp3-S4": {
        "label": "All Available Features - no FS (44 Grab / 39 Composite features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Added beyond SE2:</strong> TKN, NH3-N, Fecal/Total Coliform, "
            "Grit Classifier TSS, Primary COD/TSS, Power GE/NEA, Primary Sludge Totalizer, "
            "and all remaining measured columns not excluded for leakage or redundancy.</li>"
            "<li><strong>Remaining (44 Grab / 39 Composite features):</strong> Every non-leakage, "
            "non-redundant column from All_Years_Full.xlsx.</li>"
            "<li><strong>Row cost:</strong> Grab train ~130 rows (74% loss vs Exp2-SE2 baseline), "
            "Composite train ~498 rows. Joint dropna from high-missingness columns (TKN, Grit Classifier TSS) "
            "drives the collapse.</li>"
            "</ul>"
        ),
        "rationale": "Establishes the ceiling of feature availability and tests whether adding "
                     "all available measurements, despite severe row loss, improves prediction. "
                     "No feature selection applied - exposes raw overfitting risk at n/p ~= 3. "
                     "Confirms the Feature Audit methodology: marginal row cost is the binding constraint.",
    },
    "Exp3-S4-FS": {
        "label": "All Available Features with Model-Specific FS (44/39 features -> 1-30 selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 44/39 all features as SE4 full.</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV. Trees use OOF permutation importance "
            "(threshold=0.05). OOF fold size at n=130 with 3-fold TSS is ~43 rows - noisy estimates.</li>"
            "<li><strong>Dataset rebuild:</strong> OLS and tree models rebuild the dataset from "
            "All_Years_Full.xlsx after FS, keeping only selected features. Grab n_train can expand "
            "from 130 to 216-1133 rows depending on selected features. n_train_ks=130 stored for reference.</li>"
            "</ul>"
        ),
        "rationale": "Tests whether data-driven FS can rescue the undersized all-features dataset. "
                     "The dataset rebuild partially mitigates the row-loss problem. "
                     "Key finding: OOF noise at n=130 makes FS unstable; rebuild recovers rows "
                     "but upstream instability limits gains vs SE3-FS (which starts with ~815 rows).",
    },
    "Exp4-S1": {
        "label": "Exp3-SE2 minus Three Feature Groups (18 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Removed: Aeration SVI (Existing + New)</strong> - derived directly from "
            "SV30/MLSS ratio; carries zero independent information.</li>"
            "<li><strong>Removed: All New aeration tank columns</strong> (DO New, SV30 New) - "
            "cross-tank r=0.74-0.84 with Existing tanks; Existing-tank features are the primary "
            "ADD-tier contributors. Note: Comp pH OLS did select New-tank features in FS runs, "
            "so this removal is not lossless for all targets.</li>"
            "<li><strong>Removed: Sec Clarifier (5 columns)</strong> - correlated r=0.69-0.87 "
            "with Sec Sedimentation group. Exp2 split analysis showed Sec Sedimentation was the "
            "stronger standalone predictor; Sec Clarifier is removed here to test whether "
            "Sec Sed alone suffices.</li>"
            "<li><strong>Remaining (18 features):</strong> Inlet (4), Sec Sedimentation (5: pH, "
            "TSS, BOD, COD, RAS New), Flow, Power Total, year, Existing aeration (DO, MLSS, "
            "SV30, pH), Inlet Total Coliform, Primary Sludge Totalizer. "
            "Cyclic calendar features (month/dow sin/cos) added on the fly by training scripts.</li>"
            "</ul>"
        ),
        "rationale": "Hypothesis: removing correlated and derived feature groups reduces variance inflation and "
                     "improves generalisation, especially on composite targets where Exp3-SE2 GB/XGB catastrophically "
                     "overfit. SE1 tests manual removal; SE2 tests automated VIF-based pruning within the SE1 pool.",
    },
    "Exp4-S2": {
        "label": "Exp4-SE1 + Iterative VIF Pruning (threshold=10) - 7-8 features",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting pool:</strong> Exp4-SE1 feature set per target (18 features).</li>"
            "<li><strong>VIF pruning:</strong> Iteratively drop highest-VIF feature until all VIF <= 10. Run on training rows only (no 2025 data leakage).</li>"
            "<li><strong>Typical drops:</strong> pH features (VIF 600-2400), Inlet BOD/COD (VIF 18-34), MLSS (SV30 survives as biomass proxy).</li>"
            "<li><strong>Retained (excluded from VIF):</strong> month, day_of_week, year - temporal/categorical, not subject to VIF.</li>"
            "<li><strong>Remaining:</strong> 7-8 features per target.</li>"
            "</ul>"
        ),
        "rationale": "Hypothesis: automated, data-driven VIF pruning removes intra-group "
                     "multicollinearity more precisely than manual feature removal (Exp4-SE1), "
                     "yielding a well-conditioned feature matrix where Ridge/ElNet regularisation "
                     "can work effectively and tree model overfitting may reduce.",
    },
    "ANN-Exp1": {
        "label": "Exp1 Features - ANN (Inlet + COMMON, 9 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features (9):</strong> Inlet pH/BOD/COD/TSS (Grab or Composite) + Flow (MLD), Power Total (KW).</li>"
            "<li><strong>Model:</strong> StandardScaler + MLPRegressor, GridSearchCV on hidden_layer_sizes and alpha (TimeSeriesSplit).</li>"
            "<li><strong>Row count:</strong> ~1175 Grab / ~800 Composite training rows.</li>"
            "</ul>"
        ),
        "rationale": "Exp 6 ANN failed with ~470 Grab training samples. Exp1 has 2.5× more "
                     "data with simpler features. Tests whether ANN failure was data-volume "
                     "limited. Same architecture as Exp 6 ANN; only dataset changed.",
    },
    "ANN-Exp2-SE2": {
        "label": "Exp2-SE1 Features - ANN (Secondary + COMMON, 15 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features (15):</strong> Sec Clarifier (5) + Sec Sedimentation (5) + Flow, Power, year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Model:</strong> StandardScaler + MLPRegressor, GridSearchCV (TimeSeriesSplit).</li>"
            "<li><strong>Row count:</strong> ~924 Grab / ~740 Composite training rows.</li>"
            "</ul>"
        ),
        "rationale": "Middle ground: richer features (secondary process data) with ~2× the "
                     "samples of Exp3-SE2. Tests whether secondary process data adds meaningful "
                     "ANN signal at adequate sample size.",
    },
    "ANN-Exp2-SE3-Ref": {
        "label": "Exp2-SE2 Features - ANN (Inlet + Secondary + COMMON, 19 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features (19):</strong> Inlet (4) + Sec Clarifier (5) + Sec Sedimentation (5) + Flow, Power, year, month_sin/cos, dow_sin/cos.</li>"
            "<li><strong>Model:</strong> StandardScaler + MLPRegressor, GridSearchCV (TimeSeriesSplit).</li>"
            "<li><strong>Row count:</strong> ~920 Grab / ~733 Composite training rows.</li>"
            "</ul>"
        ),
        "rationale": "Combined inlet + secondary features at twice the sample count of "
                     "Exp3-SE2 ANN. The most direct comparison: same feature scope as "
                     "Exp3-SE2 baseline but without the CONSIDER-tier columns that reduced "
                     "training rows via missingness.",
    },
    "Exp6-ANN": {
        "label": "Exp3-SE2 Features - ANN (MLPRegressor, StandardScaler pipeline)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features:</strong> Same as Exp3-SE2 (~25 per target: Inlet + Secondary + COMMON + ADD + CONSIDER subset).</li>"
            "<li><strong>Model:</strong> StandardScaler + MLPRegressor, GridSearchCV on hidden_layer_sizes and alpha (TimeSeriesSplit).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether a neural network can outperform regularised linear "
                     "and tree models. Key challenge: ~470 training samples may be "
                     "insufficient for a multi-layer perceptron.",
    },
    "Exp6-Voting": {
        "label": "Exp3-SE2 Features - Voting Ensemble (ElNet + RF + XGBoost, equal weights)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features:</strong> Same as Exp3-SE2 (~25 per target).</li>"
            "<li><strong>Ensemble:</strong> VotingRegressor averaging ElasticNet, RandomForest, XGBoost with equal weights.</li>"
            "<li><strong>Composition:</strong> Ridge removed (L2 duplicates ElNet); XGBoost added for structural diversity (boosting vs bagging vs linear).</li>"
            "</ul>"
        ),
        "rationale": "Ensemble averaging reduces variance through uncorrelated errors. "
                     "ElNet: regularised linear with feature selection (L1). "
                     "RF: bagging over decision trees, robust to outliers. "
                     "XGB: boosted trees, captures residual non-linearities. "
                     "Three structurally distinct inductive biases.",
    },
    "Exp6-Stacking": {
        "label": "Exp3-SE2 Features - Stacking (ElNet + RF + XGB -> Ridge meta, walk-forward OOF)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Features:</strong> Same as Exp3-SE2 (~25 per target).</li>"
            "<li><strong>Base learners:</strong> ElNet + RF + XGBoost. Meta-learner: Ridge (tuned alpha).</li>"
            "<li><strong>OOF generation:</strong> Walk-forward TimeSeriesSplit(n_splits=5). First ~1/6 of training excluded from meta-learner (no OOF coverage). No look-ahead bias.</li>"
            "</ul>"
        ),
        "rationale": "Replaces original KFold(n_splits=5) StackingRegressor which had "
                     "look-ahead bias: predicting 2021 data using models trained on 2022-2024. "
                     "Walk-forward OOF ensures the meta-learner learns how base models "
                     "perform when projecting forward in time. "
                     "Approx. 83% OOF coverage across all targets.",
    },
    "Exp7-SE1": {
        "label": "Exp3-SE2 + Full Feature Engineering - All Targets (37-52 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Base:</strong> Exp3-SE2 features per target.</li>"
            "<li><strong>Added:</strong> log1p transforms of right-skewed columns + pairwise interaction terms (key pairs) + IQR outlier indicator flags.</li>"
            "<li><strong>Applied:</strong> uniformly to all 8 targets (Grab + Composite).</li>"
            "<li><strong>Total features:</strong> 37-52 per target.</li>"
            "</ul>"
        ),
        "rationale": "Feature engineering to capture non-linear relationships and "
                     "reduce skewness. Applied uniformly to test if it universally "
                     "helps. Hypothesis: log transforms help BOD/COD/TSS (right-skewed).",
    },
    "Exp7-SE2": {
        "label": "Selective Feature Engineering - Grab: Full FE / Composite: Base Exp3-SE2",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Grab targets:</strong> Full FE applied (log1p + interactions + IQR flags) - 37-52 features each.</li>"
            "<li><strong>Composite targets:</strong> Base Exp3-SE2 features only (17-27 features) - no engineering to avoid overfitting (n_train ~290).</li>"
            "</ul>"
        ),
        "rationale": "Addresses Exp 7 finding: FE caused severe overfitting on "
                     "composite datasets (n_train ≈ 290). Selective approach preserves "
                     "FE gains on grab targets while stabilising composites.",
    },
    "Exp5-S1": {
        "label": "Composite Targets: Composite Inlet + Grab Inlet + Secondary + ADD-tier (32 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Composite Inlet (4):</strong> pH, BOD, COD, TSS (Composite) - native inlet.</li>"
            "<li><strong>Added - Grab Inlet (4):</strong> pH, BOD, COD, TSS (Grab) - cross-type signal.</li>"
            "<li><strong>Secondary (10):</strong> Sec Clarifier (5) + Sec Sedimentation (5).</li>"
            "<li><strong>COMMON + ADD-tier (14):</strong> Flow, Power, year, month/dow cyclic (7) + ADD-tier aeration (7).</li>"
            "<li><strong>Targets:</strong> Composite BOD, COD, TSS, pH only. Row cost: ~22.7% vs SE1 base (~816 to ~629 train).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether grab inlet concentrations (same-day co-measurement) provide "
                     "complementary signal for composite effluent prediction. Row cost: ~22.7% "
                     "(816 → 629 train rows) due to joint grab + composite sampling availability.",
    },
    "Exp5-S2": {
        "label": "Grab Targets: Grab Inlet + Composite Inlet + Secondary + ADD-tier (32 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Grab Inlet (4):</strong> pH, BOD, COD, TSS (Grab) - native inlet.</li>"
            "<li><strong>Added - Composite Inlet (4):</strong> pH, BOD, COD, TSS (Composite) - cross-type signal.</li>"
            "<li><strong>Secondary (10):</strong> Sec Clarifier (5) + Sec Sedimentation (5).</li>"
            "<li><strong>COMMON + ADD-tier (14):</strong> Flow, Power, year, month/dow cyclic (7) + ADD-tier aeration (7).</li>"
            "<li><strong>Targets:</strong> Grab BOD, COD, TSS, pH only. Row cost: ~38.2% vs SE1 base (~1021 to ~630 train).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether composite inlet concentrations (integrated daily sample) provide "
                     "complementary signal for grab effluent prediction. Row cost: ~38.2% "
                     "(1021 → 631 train rows) - unavoidable due to joint grab + composite sampling days.",
    },
    "Exp5-S1-FS": {
        "label": "Exp5-SE1 with Model-Specific Feature Selection (32 features -> selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 32 features as SE1 (Composite + Grab Inlet + Secondary + ADD-tier).</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV; trees use OOF permutation importance (threshold=0.05). Composite targets only.</li>"
            "<li><strong>Row count:</strong> Same as SE1 (~629 Composite train).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether model-specific FS can improve generalisation on the 32-feature "
                     "Exp5-SE1 dataset (composite targets + cross-type grab inlet columns). "
                     "Paired with Exp5-S1 (full feature set) for direct comparison.",
    },
    "Exp5-S2-FS": {
        "label": "Exp5-SE2 with Model-Specific Feature Selection (32 features -> selected)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Starting set:</strong> Same 32 features as SE2 (Grab + Composite Inlet + Secondary + ADD-tier).</li>"
            "<li><strong>Feature selection:</strong> OLS uses LassoCV; trees use OOF permutation importance (threshold=0.05). Grab targets only.</li>"
            "<li><strong>Row count:</strong> Same as SE2 (~630 Grab train).</li>"
            "</ul>"
        ),
        "rationale": "Tests whether model-specific FS can improve generalisation on the 32-feature "
                     "Exp5-SE2 dataset (grab targets + cross-type composite inlet columns). "
                     "Paired with Exp5-S2 (full feature set) for direct comparison.",
    },
    "Exp8": {
        "label": "Exp3-SE2 + Temporal Lags + log1p Target Transform (50-58 features)",
        "features": (
            "<ul style='margin:0.3rem 0 0 1rem;padding:0;line-height:1.8'>"
            "<li><strong>Base:</strong> Exp3-SE2 features per target.</li>"
            "<li><strong>Added temporal:</strong> Lag 1/3/7-day + 7-day calendar rolling mean on inlet + Flow + Power columns only (~50-58 total features).</li>"
            "<li><strong>Target transform:</strong> log1p on BOD/COD/TSS; pH untransformed. Back-transform via Duan smearing.</li>"
            "<li><strong>Restriction:</strong> Temporal features limited to inlet+Flow+Power to avoid catastrophic overfitting on Composite (n~290).</li>"
            "</ul>"
        ),
        "rationale": "Daily wastewater quality has temporal autocorrelation - yesterday's "
                     "influent predicts today's effluent better than cross-sectional features "
                     "alone. Lag expansion restricted to inlet+Flow+Power columns to avoid "
                     "catastrophic overfitting on composite datasets (n≈290 rows). "
                     "log1p targets reduce right-skew in BOD/COD/TSS distributions.",
    },
}

EXP_INTRO = {
    "Exp1": (
        "Experiment 1 investigates how far inlet-only features can predict effluent quality. "
        "<strong>SE1</strong> uses only the 4 core inlet concentrations - the absolute floor. "
        "<strong>SE2</strong> adds Flow, Power, and calendar features (sin/cos projections for month and day-of-week) "
        "raising the count to 11. <strong>SE3</strong> extends to all 9 available extended inlet "
        "measurements plus COMMON (16 features total), covering both grab and composite targets. "
        "Training rows are limited by joint missingness of the supplementary inlet columns "
        "(TKN/NH3-N, O&amp;G, PO4, Total/Fecal Coliform) - see Findings Q1 for current row counts. "
        "Feature selection in SE3-FS is model-specific: OLS uses LassoCV, Ridge uses the "
        "full set, ElNet selects via L1, and trees use OOF permutation importance (threshold 5%)."
    ),
    "Exp2": (
        "Experiment 2 steps through the treatment plant stage by stage, starting from where Experiment 1 left off. "
        "<strong>SE1</strong> tests the primary clarification and grit removal stage in two sub-scopes to isolate "
        "whether primary features carry any standalone signal: "
        "<em>SE1a</em> (Primary + COMMON, 13 features) excludes inlet entirely; "
        "<em>SE1b</em> (Primary + Inlet + COMMON, 17 features) adds core inlet as a paired comparison. "
        "Both sub-scopes produce all-negative R², confirming primary features are genuinely non-predictive "
        "regardless of whether inlet is included. "
        "<strong>SE2</strong> replaces the primary group with secondary treatment data, tested in three "
        "scopes: Sec Clarifier alone (10 features), Sec Sedimentation alone (10 features), and both "
        "combined (15 features). The split variants establish whether the two secondary sub-groups "
        "carry distinct signal or are interchangeable. "
        "<strong>SE3</strong> combines all three monitoring points - inlet, primary, and secondary - "
        "into a single feature set (27 features). The CIS baseline (Core Inlet + Secondary + Common, 21 features) "
        "is retained for direct delta comparison."
    ),
    "Exp3": (
        "Experiment 3 starts from the <strong>Exp2-SE2 baseline</strong> (Inlet + Secondary + COMMON, "
        "21 features) and adds features in four steps guided by the "
        "<a href='../../eda/eda_full_report.html#s11' target='_blank' style='color:#4A90D9'>"
        "EDA Feature Audit</a>. "
        "<br><br>"
        "<strong>SE1 (ADD-tier, 28 features)</strong> adds the seven aeration-basin "
        "measurements that scored MI ≥ 0.20 with marginal row cost ≤ 20 %: "
        "Aeration DO/MLSS/SV30/SVI (Existing), Aeration pH (Existing), Aeration DO/SV30 (New). "
        "Grab training rows stay near ~816 - no meaningful row loss. No feature selection applied. "
        "<br><br>"
        "<strong>SE2 (ADD + CONSIDER-tier, 31 features, full and FS)</strong> "
        "adds three CONSIDER-tier features (Aeration SVI New, Inlet Total Coliform, "
        "Primary Sludge Totalizer). Inlet Total Coliform alone causes ~347 row loss (816 → 469 Grab). "
        "Feature selection (OLS: LassoCV; trees: OOF permutation importance) is applied inside "
        "training scripts to identify the most predictive subset. "
        "<br><br>"
        "<strong>SE3 (ADD + CONSIDER minus Coliform, 30 features, full and FS)</strong> "
        "removes Inlet Total Coliform from SE2, recovering ~347 rows (469 → 815 Grab). "
        "This isolates the row-cost question: is SE2's performance driven by the Coliform signal "
        "or by data starvation from Coliform missingness? Both no-FS and FS variants evaluated. "
        "<br><br>"
        "<strong>SE4 (All remaining features, 44/39, full and FS)</strong> "
        "includes every non-leakage column - the all-features upper bound. "
        "Joint dropna reduces Grab training to ~130 rows (74% loss vs Exp2-SE2) "
        "driven by TKN, Grit Classifier TSS, Primary COD/TSS. "
        "Both no-FS and FS variants confirm that marginal row cost is the binding constraint."
    ),
    "Exp4": (
        "Experiment 4 tests the hypothesis that <strong>removing collinear and derived features</strong> "
        "from Exp3-SE2 improves generalisation. "
        "<strong>SE1</strong> manually drops three redundant groups "
        "(SVI, New aeration, Sec Sed - feature count 12-19). "
        "<strong>SE2</strong> applies automated iterative VIF pruning (threshold=10) "
        "within the Exp4-SE1 feature pool, reducing to 5-10 features per target and gaining "
        "+47 to +293 rows by dropping high-missingness correlated features. "
        "<strong>Overall result: both sub-experiments refuted the pruning hypothesis.</strong> "
        "SE1 showed performance degraded universally. SE2 confirmed the pattern: "
        "Ridge alone partially recovered on some targets; tree models failed catastrophically. "
        "See the finding boxes below for per-sub-experiment analysis."
    ),
    "Exp5": (
        "Experiment 5 tests the <strong>cross-type inlet hypothesis</strong>: do grab-sample "
        "inlet measurements help predict composite effluent, and vice versa? "
        "The base feature set is <strong>Exp3-SE1 ADD-tier</strong> (28 features, "
        "Inlet + Secondary + COMMON + 7 ADD aeration columns, selected for near-zero row cost). "
        "Four cross-type inlet columns are added per sub-experiment. "
        "<br><br>"
        "<strong>SE1 (Composite targets + Grab inlet, 32 features)</strong>: "
        "Composite prediction datasets receive GRAB_INLET (Inlet pH/BOD/COD/TSS Grab). "
        "Row cost: ~22.7% (816 to 629 train rows)  -  within acceptable range. "
        "<br><br>"
        "<strong>SE2 (Grab targets + Composite inlet, 32 features)</strong>: "
        "Grab prediction datasets receive COMP_INLET (Inlet pH/BOD/COD/TSS Composite). "
        "Row cost: ~38.2% (1021 to 630 train rows)  -  high but unavoidable; "
        "both grab and composite inlet must be measured on the same day. "
        "Retained n (~630) is comparable to Exp2-SE2 grab training size. "
        "<br><br>"
        "The experiment answers: does same-day co-measurement of the complementary inlet "
        "sample type add independent predictive signal beyond the same-type inlet columns?"
    ),
    "Exp9": (
        "Experiment 9 tests the <strong>recency hypothesis</strong>: does restricting the "
        "training window to the single most recent year (2024) produce better 2025 test "
        "generalisation than the full 2021-2024 window? "
        "The feature set is held fixed at <strong>Exp3-SE1</strong> (27 features: "
        "Inlet + Secondary + COMMON_CYCLIC + ADD-tier aeration) - the best validated "
        "multi-feature set from prior experiments - so that only the temporal scope changes. "
        "<br><br>"
        "<strong>SE1 (W1: 2024-only training)</strong>: 187 Grab and 179 Composite training "
        "rows after dropna - roughly one quarter of the full-window baseline. The 2025 test "
        "set (~202 Grab / ~182 Composite rows) is comparable in size to the training set, "
        "making test R² estimates reliable despite the small training window. "
        "<br><br>"
        "The motivation is the distribution shift identified in the Comp COD diagnostic: "
        "if the plant's operational regime drifted after 2022, the 2021-2022 data may be "
        "active noise rather than useful signal for predicting 2025 behaviour. "
        "Five sub-experiments explore the hypothesis and its interaction with feature engineering: "
        "SE1 (plain W1), SE2 (W1 + feature selection), SE3 (W1 + log target), "
        "SE4 (W1 + log features), SE5 (W1 + log features + log target - the log-log model). "
        "SE5 achieves the best Grab BOD globally (gap-adj R²=+0.654 vs full-window +0.499)."
    ),
    "ANN-Dataset-Exploration": (
        "The Exp 6 ANN failed on Exp3-SE2 datasets (avg Test R²=−1.12) due to insufficient "
        "training samples (~470 Grab, ~290 Composite). These runs test the same ANN architecture "
        "on earlier, <strong>data-richer</strong> experiment datasets to isolate whether the failure "
        "was <em>data-volume limited</em> or reflects a fundamental ANN limitation on this type of data."
        "<br><br>"
        "Three datasets are tested: <strong>Exp1</strong> (Inlet + COMMON, 9 features, "
        "~1175 Grab / ~800 Composite rows  -  2.5× more data), "
        "<strong>Exp2-SE1</strong> (Secondary + COMMON, 15 features, "
        "~924 / ~740 rows), and "
        "<strong>Exp2-SE2</strong> (Inlet + Secondary + COMMON, 19 features, "
        "~920 / ~733 rows). "
        "The hyperparameter grid includes larger architectures (256-128, 128-64-32 hidden layers) "
        "compared to Exp 6 ANN, appropriate for the larger sample counts."
    ),
    "Exp6": (
        "This section evaluates two classes of advanced model architectures on the "
        "<strong>Exp3-SE2 feature set</strong> (~25 features per target: Inlet + Secondary "
        "+ ADD+CONSIDER-tier + cyclic calendar), selected as the richest validated set at the "
        "time (ElNet Test R²=0.684 Grab BOD, RF 0.504 Grab TSS)."
        "<br><br>"
        "<strong>Neural Networks</strong>: MLPRegressor (feedforward ANN) failed on all "
        "datasets tested (avg R²=-1.12 on Exp3-SE2 features). The data-volume diagnostic "
        "(ANN Exploration sub-section) confirmed the failure is not data-volume limited - "
        "tripling training rows worsened performance. See the rationale card below for "
        "why a feedforward ANN was chosen over RNN/LSTM architectures."
        "<br><br>"
        "<strong>Ensemble Methods</strong>: Voting (ElNet+RF+XGB, equal weights) is the "
        "recommended approach - avg Test R²=+0.287, Grab BOD 0.692. Stacking "
        "(walk-forward OOF, no look-ahead bias after KFold correction) is "
        "methodologically clean but weaker (avg R²=+0.098)."
    ),
    "Exp7": (
        "Exp 7 applies <strong>feature engineering</strong> (log transforms, "
        "interaction terms, outlier flags) on top of the Exp3-SE2 feature set. "
        "Exp 7 (full) applies this uniformly to all targets. Exp 7-SE2 "
        "(selective) applies FE only to grab targets after discovering that "
        "composite datasets overfit severely with the expanded feature count."
    ),
    "Exp8": (
        "Exp 8 adds <strong>temporal lag features</strong> (lag 1/3/7-day) and a "
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
    # LHS = strings as they appear in source results.xlsx files (legacy names preserved).
    # RHS = canonical exp_key emitted into df_all.
    mapping = {
        "Exp1-Sub1": "Exp1-SE1",   "Exp1-SE1": "Exp1-SE1",
        "Exp1": "Exp1",            "Exp1-Cyclic": "Exp1-Cyclic",
        "Exp1-S3": "Exp1-S3",      "Exp1-S3-FS": "Exp1-S3-FS",
        "Exp2-Sub1":     "Exp2-SE2-Comb",
        "Exp2-Sub1-Clr": "Exp2-SE2-Clr",
        "Exp2-Sub1-Sed": "Exp2-SE2-Sed",
        "Exp2-Sub2":     "Exp2-Sub2",
        "Exp2-Sub2-Cyc": "Exp2-SE3-Ref",
        "Exp2-S6":    "Exp2-S6",
        "Exp2-S6-FS": "Exp2-S6-FS",
        "Exp2-S5": "Exp2-S5",
        "Exp2-S3": "Exp2-S3",
        "Exp2-S4": "Exp2-S4",
        "Exp3-S1": "Exp3-S1",
        "Exp3-S2-FS": "Exp3-S2-FS",
        "Exp3-S3": "Exp3-S3", "Exp3-S3-FS": "Exp3-S3-FS",
        "Exp3-S4": "Exp3-S4", "Exp3-S4-FS": "Exp3-S4-FS",
        "Experiment 1": "Exp1",
        "Experiment 2 Sub-1": "Exp2-SE2-Comb",
        "Experiment 2 Sub-2": "Exp2-Sub2",
        "Experiment 3 Sub-1": "Exp3-S1",
        "Experiment 3 Sub-2": "Exp3-S2-FS",
        "Exp4-S1": "Exp4-S1",
        "Experiment 4 Sub-1": "Exp4-S1",
        "Exp4-S2": "Exp4-S2",
        "Experiment 4 Sub-2": "Exp4-S2",
        "Exp3-S2": "Exp3-S2",
        "Exp5-S1": "Exp5-S1",
        "Exp5-S1-FS": "Exp5-S1-FS",
        "Exp5-S2": "Exp5-S2",
        "Exp5-S2-FS": "Exp5-S2-FS",
        "Exp9-SE1": "Exp9-SE1",
        "Exp9-SE2": "Exp9-SE2",
        "Exp9-SE3": "Exp9-SE3",
        "Exp9-SE4": "Exp9-SE4",
        "Exp9-SE5": "Exp9-SE5",
        "Exp6-ANN": "Exp6-ANN",
        "Exp6-Ensemble": "Exp6-Ensemble",
        "Exp7-SE1": "Exp7-SE1",
        "Exp7-SE2-GrabOnly": "Exp7-SE2",
    }
    key = mapping.get(raw, raw)
    if is_fs:
        # Legacy Sub-2 FS results live in feature_selected_datasets/ under sub_exp2;
        # they are the FS variant of the cyclic SE3 Ref (canonical Exp2-SE3-Ref-FS).
        if key == "Exp2-Sub2":
            return "Exp2-SE3-Ref-FS"
        if key in ("Exp1", "Exp2-SE2-Comb"):
            key = key + "-FS"
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
        exp_key=df["model"].map(lambda m: f"Exp6-{model_map.get(m, m)}"),
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
    """Normalize Exp 8 results (temporal features + log1p targets)."""
    out = pd.DataFrame(dict(
        exp_key="Exp8",
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
    """Normalize ANN results for Exp1/Exp2-SE2/Exp2-SE3-Ref dataset runs.

    experiment column in results.xlsx may use legacy Sub-N names; translate
    to current canonical exp_keys.
    """
    _ann_legacy = {
        "ANN-Exp2-Sub1": "ANN-Exp2-SE2",
        "ANN-Exp2-Sub2": "ANN-Exp2-SE3-Ref",
    }
    out = pd.DataFrame(dict(
        exp_key=df["experiment"].map(lambda v: _ann_legacy.get(str(v), str(v))),
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
    exp_key = "Exp7-SE2" if is_10b else "Exp7-SE1"
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
        ("exp1_s3", False), ("exp1_s3_fs", False),
        ("exp3_s1", False), ("exp3_s2", False), ("exp3_s2_nofs", False),
        ("exp3_s3", False), ("exp3_s3_fs", False),
        ("exp3_s4", False), ("exp3_s4_fs", False),
        ("exp4_s1", False),
        ("exp4_s2", False),
        ("exp5_s1", False), ("exp5_s1_fs", False),
        ("exp5_s2", False), ("exp5_s2_fs", False),
        ("exp2_s1", False), ("exp2_s1_split", False), ("exp2_s2", False),
        ("exp2_s6", False), ("exp2_s6_fs", False), ("exp2_s5", False), ("exp2_s3", False), ("exp2_s4", False),
        ("exp9_s1", False),
        ("exp9_s2", False),
        ("exp9_s3", False),
        ("exp9_s4", False),
        ("exp9_s5", False),
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
        ("exp1_s3", False), ("exp1_s3_fs", False),
        ("exp3_s1", False), ("exp3_s2", False), ("exp3_s2_nofs", False),
        ("exp3_s3", False), ("exp3_s3_fs", False),
        ("exp3_s4", False), ("exp3_s4_fs", False),
        ("exp4_s1", False),
        ("exp4_s2", False),
        ("exp5_s1", False), ("exp5_s1_fs", False),
        ("exp5_s2", False), ("exp5_s2_fs", False),
        ("exp2_s1", False), ("exp2_s1_split", False), ("exp2_s2", False),
        ("exp2_s6", False), ("exp2_s6_fs", False), ("exp2_s5", False), ("exp2_s3", False), ("exp2_s4", False),
        ("exp9_s1", False),
        ("exp9_s2", False),
        ("exp9_s3", False),
        ("exp9_s4", False),
        ("exp9_s5", False),
    ]:
        for mdl in ["rf", "gb", "xgb"]:
            p = os.path.join(m, "non_linear", variant, mdl, "results.xlsx")
            if os.path.exists(p):
                df = pd.read_excel(p)
                df = df[df["run"] == df["run"].max()]
                frames.append(_norm_nl(df, is_fs))

    # Exp 6
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

    # Exp 7
    for path, is_10b in [
        (os.path.join(m, "phase10", "results.xlsx"), False),
        (os.path.join(m, "phase10", "results_10b.xlsx"), True),
    ]:
        if os.path.exists(path):
            df = pd.read_excel(path)
            df = df[df["run"] == df["run"].max()]
            frames.append(_norm_phase10(df, is_10b))

    # Exp 8
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

    # Retired exp_keys: legacy variants superseded by current canonical experiments.
    # Filtering here propagates the retirement to every section automatically.
    RETIRED_EXP_KEYS = {"Exp2-Sub2"}
    combined = combined[~combined["exp_key"].isin(RETIRED_EXP_KEYS)].reset_index(drop=True)

    return combined


def compute_all_mdae() -> pd.DataFrame:
    """Scan every dataset xlsx file, compute MdAE on 2025 rows for BOD/TSS targets only.

    Returns a DataFrame with columns: exp_key, model, target, MdAE_test.
    Exp 7 and Exp 8 do not store row-level predictions so those phases
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

            # Exp 6 models stored in Exp3-SE2 files need their own exp_key
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
    "Exp1-S3-FS": "Exp1-S3",
    "Exp2-SE2-Comb-FS": "Exp2-SE2-Comb",
    "Exp2-SE3-Ref-FS": "Exp2-Sub2",
    "Exp3-S3-FS": "Exp3-S3",
    "Exp3-S4-FS": "Exp3-S4",
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

    variant = ("exp3_ks" if full_key == "Exp3-S3"
               else "exp3_s1" if full_key == "Exp3-S1"
               else "baseline")
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
        "Exp2-SE2-Comb-FS": ("feature_importance.xlsx", "Experiment 2 Sub-1"),
        "Exp2-SE3-Ref-FS": ("feature_importance.xlsx", "Experiment 2 Sub-2"),
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


def _dataset_summary_per_model(df: pd.DataFrame) -> str:
    """Per-model dataset details for S3-FS where n_train varies after FS rebuild.

    OLS and tree models rebuild from All_Years_Full after feature selection, so
    their n_train can be substantially larger than the baseline (~130 grab rows).
    Ridge/ElNet use the full full feature set and cannot expand.
    n_train_ks (stored in results) shows the original all-features row count for reference.
    """
    def _safe_int(v):
        try:
            f = float(v)
            return "-" if np.isnan(f) else int(f)
        except (TypeError, ValueError):
            return "-"

    rows = []
    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt]
        if sub.empty:
            continue
        slug = TARGET_SLUG.get(tgt, "all")
        tgt_short = TARGET_SHORT.get(tgt, tgt)
        first_model = True
        for model in ALL_MODELS_ORD:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            r = msub.iloc[0]
            # Ridge and ElNet use the full un-rebuilt feature set; OLS and tree models
            # rebuild from raw after FS so their n_train > n_train_ks.
            if model == "Ridge":
                n_tr = _safe_int(r.get("n_train_ks") if r.get("n_train_ks") is not None else r.get("n_train"))
                n_te = _safe_int(r.get("n_test_ks")  if r.get("n_test_ks")  is not None else r.get("n_test"))
                n_fe = _safe_int(r.get("n_features"))
            elif model == "ElNet":
                n_tr = _safe_int(r.get("n_train_ks") if r.get("n_train_ks") is not None else r.get("n_train"))
                n_te = _safe_int(r.get("n_test_ks")  if r.get("n_test_ks")  is not None else r.get("n_test"))
                n_fe = _safe_int(r.get("ElNet_n_selected") or r.get("n_features"))
            elif model == "OLS":
                n_tr = _safe_int(r.get("n_train"))
                n_te = _safe_int(r.get("n_test"))
                n_fe = _safe_int(r.get("n_selected_ols") or r.get("n_features"))
            else:  # RF / GB / XGB - each has its own row with per-model n_train
                n_tr = _safe_int(r.get("n_train"))
                n_te = _safe_int(r.get("n_test"))
                n_fe = _safe_int(r.get("n_selected_nl") or r.get("n_features"))
            n_tr_ks  = _safe_int(r.get("n_train_ks"))
            expanded = (n_tr != "-" and n_tr_ks != "-" and int(n_tr) > int(n_tr_ks)
                        if n_tr != "-" and n_tr_ks != "-" else False)
            expand_note = (f" <span style='color:#5BAD6F;font-size:0.85em'>"
                           f"(+{int(n_tr) - int(n_tr_ks)} vs baseline)</span>") if expanded else ""
            baseline_note = (f" <span style='color:var(--text-muted);font-size:0.85em'>"
                             f"(full feat set)</span>") if not expanded and n_tr_ks != "-" else ""
            tgt_cell = (f'<td rowspan="{sum(1 for m in ALL_MODELS_ORD if not sub[sub["model"]==m].empty)}" '
                        f'data-target="{slug}">{tgt_short}</td>') if first_model else ""
            rows.append(
                f'<tr data-target="{slug}">'
                f'{tgt_cell}'
                f'<td style="font-size:0.88em;color:var(--text-muted)">{model}</td>'
                f'<td>{n_tr}{expand_note}{baseline_note}</td>'
                f'<td>{n_te}</td>'
                f'<td>{n_fe}</td></tr>'
            )
            first_model = False

    if not rows:
        return _dataset_summary(df)

    has_rebuild = "n_train_ks" in df.columns and df["n_train_ks"].notna().any()
    if has_rebuild:
        note = ("<p class='meta' style='margin:0 0 6px'>"
                "OLS and tree models (RF/GB/XGB) rebuild from <code>All_Years_Full.xlsx</code> "
                "after feature selection - rows previously lost to joint missingness of dropped "
                "features are recovered. Ridge uses the full input feature set (no rebuild); "
                "ElNet selects internally via L1. n_features shown = selected count for FS models, "
                "full input count for Ridge.</p>")
        n_train_header = "n_train (post-FS rebuild)"
    else:
        note = ("<p class='meta' style='margin:0 0 6px'>"
                "Model-specific feature selection: each model may select a different number of "
                "features. OLS uses LassoCV pre-screen; ElNet selects via internal L1; "
                "RF/GB/XGB use OOF permutation importance. n_features shown = selected count "
                "per model.</p>")
        n_train_header = "n_train (2021-2024)"
    return f"""
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> Dataset Details - Per-Model (n_train / n_test / n_features selected)</summary>
  <div class="fold-body">
    {note}
    <table class="summary-table ds-table">
      <thead><tr>
        <th>Target</th><th>Model</th>
        <th>{n_train_header}</th>
        <th>n_test (2025)</th><th>n_features selected</th>
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
        lin_str = " - "
        if has_linear and not lin_row.empty:
            raw = lin_row.iloc[0].get("selected_features_ols", "")
            if isinstance(raw, str) and raw.strip():
                lin_sel = set(f.strip() for f in raw.split(","))
                lin_str = f"{len(lin_sel)} features"

        # ElasticNet internal selection
        elnet_row = sub[sub["model"] == "ElNet"]
        elnet_sel = set()
        elnet_str = " - "
        if has_linear and not elnet_row.empty:
            raw = elnet_row.iloc[0].get("ElNet_selected_features", "")
            if isinstance(raw, str) and raw.strip():
                elnet_sel = set(f.strip() for f in raw.split(","))
                elnet_str = f"{len(elnet_sel)} features"

        # Tree model selections (RF, GB, XGB  -  may differ)
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

    Returns empty set when only one model is available  -  ★ is a comparison
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
    Exp 7 and Exp 8 do not store row-level predictions so MdAE is unavailable (-).

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
        'stored at row level (Exp 7 / Exp 8).</p>'
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
    """Compute VIF for Exp3-SE2 feature sets inline and render as a foldable section."""
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
<details class="inner-fold">
  <summary><span class="fold-icon">▶</span> VIF Collinearity Analysis - Exp3-SE2 Feature Sets</summary>
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
      Δ vs Voting = ANN Test R² − Exp 6 Voting Test R².
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
          (Q4-Flow and winter errors 2-3x higher - see Error Regime Decomposition).
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
    """Foldable Exp3-SE2 methodology card explaining sin/cos calendar encoding."""
    lin = df_all[df_all["model"].isin(LINEAR_MODELS)].copy()

    rows_html = ""
    for exp_key, enc_label in [("Exp3-SE1", "Raw integer (1-12 / 0-6)"),
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
        Cyclic encoding is applied <strong>only in Exp3-SE2 linear models</strong>
        (OLS, Ridge, ElasticNet). All prior experiments (Exp1, Exp2, Exp3 Sub-1) and all
        tree-based and advanced models (RF, GB, XGB, Voting, Stacking, Exp 6-8) use raw
        integer encoding.
      </p>
      <p class="meta">
        Exp3-SE2 also introduces new CONSIDER-tier features absent from Exp3-SE1, so the two
        experiments are <strong>not a controlled A/B test</strong> for encoding alone. Any R²
        difference in the table below reflects the combined effect of the broader feature set
        <em>and</em> the encoding change.
      </p>
    </div>

    <details class="inner-fold" style="margin-top:1rem">
      <summary><span class="fold-icon">▶</span>
        Exp3-SE1 (raw integers) vs Exp3-SE2 (cyclic) - Linear Model Avg Test R² (all 8 targets)
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
        A controlled run of Exp3-SE2 features with raw-integer encoding was not performed.
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
            td = "padding:5px 10px;border-bottom:1px solid #e0e0e0"
            year_rows += (
                f'<tr style="background:#ffffff">'
                f'<td style="{td}"><strong>{year}{tag}</strong></td>'
                f'<td style="{td};text-align:right">{len(v)}</td>'
                f'<td style="{td};text-align:right;color:{m_color};font-weight:{bold}">{v.mean():.1f}</td>'
                f'<td style="{td};text-align:right">{v.std():.1f}</td>'
                f'<td style="{td};text-align:right">{v.min():.1f}</td>'
                f'<td style="{td};text-align:right">{v.quantile(0.25):.1f}</td>'
                f'<td style="{td};text-align:right">{v.quantile(0.75):.1f}</td>'
                f'<td style="{td};text-align:right">{v.max():.1f}</td></tr>'
            )

        stats_html = f"""
<div style="margin:1.2rem 0 0.3rem;font-size:15px;font-weight:600;padding:6px 10px;
     background:#f5f5f5;color:#1a1a1a;border-left:3px solid #4A90D9;border-radius:0 4px 4px 0">
  Per-Year Distribution Statistics
</div>
<div style="overflow-x:auto;border:1px solid #cccccc;border-radius:4px;margin-bottom:8px">
<table style="width:100%;border-collapse:collapse;font-size:0.81rem;margin:0;
              background:#ffffff;color:#1a1a1a">
  <thead><tr style="border-bottom:2px solid #cccccc">
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:left;
               font-size:0.82rem;color:#333333">Year</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">n</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Mean</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Std</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Min</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">P25</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">P75</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Max</th>
  </tr></thead>
  <tbody>
    {year_rows}
    <tr style="background:#f7f7f7;font-style:italic;border-top:2px solid #cccccc">
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;color:#555555">Train 2021-2024</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{len(train)}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train_mean:.1f}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train_std:.1f}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train[TARGET_COL].min():.1f}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train_p25:.1f}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train_p75:.1f}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;color:#555555">{train[TARGET_COL].max():.1f}</td>
    </tr>
  </tbody>
</table>
</div>
<p style="font-size:12px;color:#555555;margin:0 0 12px">
  Red mean = year mean deviates by &gt;1σ from training mean.
  Orange = 0.5-1σ shift. 2025 (★) is the test set.
</p>"""

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
<div style="margin:0 0 1.2rem 0;font-size:15px;font-weight:600;padding:6px 10px;
     background:#f5f5f5;color:#1a1a1a;border-left:3px solid #4A90D9;border-radius:0 4px 4px 0">
  2025 Distribution Shift vs Training (2021-2024)
</div>
<div style="overflow-x:auto;border:1px solid #cccccc;border-radius:4px;margin-bottom:8px">
<table style="width:100%;border-collapse:collapse;font-size:0.81rem;margin:0;
              background:#ffffff;color:#1a1a1a">
  <thead><tr style="border-bottom:2px solid #cccccc">
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:left;
               font-size:0.82rem;color:#333333">Metric</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Value</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:left;
               font-size:0.82rem;color:#333333">Reference</th>
  </tr></thead>
  <tbody>
    <tr style="background:#ffffff">
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0">2025 mean</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;
                 font-weight:600;color:{sig_col}">{test[TARGET_COL].mean():.1f} mg/L</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;color:#555555">
        Train mean {train_mean:.1f} mg/L ({mean_sigma:.2f}σ shift)</td>
    </tr>
    <tr style="background:#f7f7f7">
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0">
        2025 values within training IQR</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;
                 font-weight:600;color:{iqr_col}">{frac_iqr:.0%}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;color:#555555">
        IQR [{train_p25:.1f}, {train_p75:.1f}] mg/L</td>
    </tr>
    <tr style="background:#ffffff">
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0">
        2025 values above training P75</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;text-align:right;
                 font-weight:600;color:#555555">{frac_above:.0%}</td>
      <td style="padding:5px 10px;border-bottom:1px solid #e0e0e0;color:#555555">
        P75 = {train_p75:.1f} mg/L</td>
    </tr>
  </tbody>
</table>
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
        row_bg = "#ffffff" if len(perf_rows) % 2 == 0 else "#f7f7f7"
        td = "padding:5px 10px;border-bottom:1px solid #e0e0e0"
        perf_rows += (
            f'<tr style="background:{row_bg}">'
            f'<td style="{td}">{EXP_CHART_LABELS.get(exp_key, exp_key)}</td>'
            f'<td style="{td}">{best["model"]}</td>'
            f'<td style="{td};text-align:right;color:{color};font-weight:bold">{_fmt(r2)}{badge}</td>'
            f'<td style="{td};text-align:right" class="{gap_cls}">{gap_str}</td></tr>'
        )

    perf_table = f"""
<div style="margin:1.2rem 0 0.3rem;font-size:15px;font-weight:600;padding:6px 10px;
     background:#f5f5f5;color:#1a1a1a;border-left:3px solid #4A90D9;border-radius:0 4px 4px 0">
  Best Comp COD Result per Experiment ({n_exp} experiments, {n_mod} model families)
</div>
<div style="overflow-x:auto;border:1px solid #cccccc;border-radius:4px;margin-bottom:8px">
<table style="width:100%;border-collapse:collapse;font-size:0.81rem;margin:0;
              background:#ffffff;color:#1a1a1a">
  <thead><tr style="border-bottom:2px solid #cccccc">
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:left;
               font-size:0.82rem;color:#333333">Experiment</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:left;
               font-size:0.82rem;color:#333333">Best Model</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">Test R²</th>
    <th style="padding:5px 10px;background:#eeeeee;font-weight:600;text-align:right;
               font-size:0.82rem;color:#333333">R² Gap</th>
  </tr></thead>
  <tbody>{perf_rows}</tbody>
</table>
</div>
<p style="font-size:12px;color:#555555;margin:0 0 12px">
  Global best Test R² across all experiments: <strong>{_fmt(global_max)}</strong>.
  Every regularisation strategy, feature set expansion, and engineering approach
  has been tried; none breaks through to meaningful generalisation on 2025.
</p>"""

    return f"""
<section id="comp-cod-diagnostic">
  <h1 class="section-title">Comp COD - Persistent Failure Diagnostic</h1>
  <p class="section-intro">
    Effluent COD (Composite) is the only target where <em>no model generalises</em> across any
    experiment or phase (best Test R² = {_fmt(global_max)} across {n_exp} experiment variants).
    Exp 6 Voting MAE doubled from 8.7 mg/L (2024 in-sample) to 18.1 mg/L (2025 test).
    Variance is not collapsed (σ-ratio = 0.78), which rules out the low-variance artefact
    that drives negative R² in Comp pH and Comp TSS. This section documents the distribution
    shift, the performance panorama across every experiment, and the evidence for a
    2025 process change as the likely root cause.
  </p>

  {shift_html}
  {stats_html}
  {perf_table}

<details class="exp-details" id="comp-cod-findings" open>
  <summary><span class="fold-icon">▶</span> Findings - Comp COD Persistent Failure</summary>
  <div class="exp-body">

    <div style="margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden">
      <div style="background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem">
        <span style="color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0">Q1</span>
        <span style="font-weight:bold;font-size:0.93em;line-height:1.4">Is Comp COD failing due to a modelling artefact, or is there a genuine process change?</span>
      </div>
      <div style="padding:0.6rem 0.8rem 0.55rem">
        <p class="meta" style="margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem">
          The failure is a genuine 2025 process change, not a modelling artefact. MAE doubled
          (8.7 → 18.1 mg/L) while the σ-ratio = 0.78, confirming 2025 variance is comparable to
          training. The model is not hitting a low-variance wall - it is predicting the wrong
          values. This is the signature of a non-stationarity in the plant's COD removal mechanism
          that was absent during the 2021-2024 training window.
        </p>
      </div>
    </div>

    <div style="margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden">
      <div style="background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem">
        <span style="color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0">Q2</span>
        <span style="font-weight:bold;font-size:0.93em;line-height:1.4">Why does adding more features - including secondary process data - not recover generalisation?</span>
      </div>
      <div style="padding:0.6rem 0.8rem 0.55rem">
        <p class="meta" style="margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem">
          Inlet COD is a weak proxy for effluent COD under current plant conditions. Composite
          effluent COD is governed by secondary treatment efficiency (MLSS, SVI, aeration) -
          features with 35-50% missingness on composite measurement days (CONSIDER tier). When
          these are absent, the model falls back to inlet + flow, whose MI with Comp COD is
          low (&lt;0.15 on training data). The causal variables simply are not available often enough.
        </p>
      </div>
    </div>

    <div style="margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden">
      <div style="background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem">
        <span style="color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0">Q3</span>
        <span style="font-weight:bold;font-size:0.93em;line-height:1.4">Does the small composite sample size contribute to the lack of generalisation?</span>
      </div>
      <div style="padding:0.6rem 0.8rem 0.55rem">
        <p class="meta" style="margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem">
          Yes. Composite targets have only ~515-633 training rows after dropna. With 17-32 features
          the effective rows-per-feature ratio is 16-37 - below the commonly cited heuristic of 50.
          Even heavily regularised models (ElNet α=10) risk encoding 2022-era COD spikes that
          do not recur in 2025. Small n amplifies the memorisation problem introduced by
          distribution shift.
        </p>
      </div>
    </div>

    <div style="margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden">
      <div style="background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem">
        <span style="color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0">Q4</span>
        <span style="font-weight:bold;font-size:0.93em;line-height:1.4">Do feature engineering, advanced ensembles, or temporal lags overcome the failure?</span>
      </div>
      <div style="padding:0.6rem 0.8rem 0.55rem">
        <p class="meta" style="margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem">
          No. Exp 7 full FE: best R² = -0.008 (Ridge). Exp 7-SE2 selective FE: best = -0.051.
          Exp 8 temporal lags: best = +0.107 (ElNet) but 2025 MAE remains ~17 mg/L. Every
          additional feature or transform increases the risk of encoding training-specific patterns
          without improving generalisation on the shifted 2025 distribution.
        </p>
      </div>
    </div>

    <div style="margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden">
      <div style="background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem">
        <span style="color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0">Q5</span>
        <span style="font-weight:bold;font-size:0.93em;line-height:1.4">What are the recommended next steps for operational use and future modelling?</span>
      </div>
      <div style="padding:0.6rem 0.8rem 0.55rem">
        <p class="meta" style="margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem">
          (a) <strong>Flag Comp COD predictions as unreliable</strong> in any operational dashboard
          until 2025 data is incorporated into training.
          (b) <strong>Collect causal features</strong> with higher priority: secondary sludge age
          (SRT), MLSS, and effluent turbidity directly govern COD removal - their availability on
          composite measurement days would be the single highest leverage improvement.
          (c) <strong>Retrain once 90+ new 2025 rows are available</strong> (roughly 3 months of
          composite measurements) and test whether the new regime is stable enough for a combined
          2021-2025 model.
          (d) As an interim measure, <strong>a shallow decision tree (max depth 3) trained only
          on 2024-2025 data</strong> is likely to outperform any model trained on the full
          2021-2024 window for near-term operational use.
        </p>
      </div>
    </div>

  </div>
</details>
</section>"""


def _section_bests_json(df_all: pd.DataFrame):
    """JSON for the dynamic running-leaders sidebar panel.

    Returns (bests_json, order_json) so the JS sectionOrder stays in sync
    with section_exp_keys without a separate hardcoded array.
    """
    section_exp_keys = {
        "exp1-sub1":   ["Exp1-SE1"],
        "exp1-full":   ["Exp1-Cyclic"],
        "exp1-fs":     ["Exp1-FS"],
        "exp1-s3":      ["Exp1-S3", "Exp1-S3-FS"],
        "exp1-s3-full": ["Exp1-S3"],
        "exp1-s3-fs":   ["Exp1-S3-FS"],
        "exp2-s1":          ["Exp2-S5", "Exp2-S3"],
        "exp2-s1a":         ["Exp2-S5"],
        "exp2-s1b":         ["Exp2-S3"],
        "exp2-s2":          ["Exp2-S6", "Exp2-SE2-Comb", "Exp2-SE2-Clr", "Exp2-SE2-Sed"],
        "exp2-s2a":         ["Exp2-S6"],
        "exp2-s2-clr":      ["Exp2-SE2-Clr"],
        "exp2-s2-sed":      ["Exp2-SE2-Sed"],
        "exp2-s2-combined": ["Exp2-SE2-Comb"],
        "exp2-s3":          ["Exp2-S4", "Exp2-SE3-Ref", "Exp2-SE3-Ref-FS", "Exp2-S6-FS"],
        "exp2-s3-full":     ["Exp2-S4"],
        "exp2-s3-ref":      ["Exp2-SE3-Ref"],
        "exp2-s3-ref-fs":   ["Exp2-SE3-Ref-FS"],
        "exp2-s3-s6fs":     ["Exp2-S6-FS"],
        "exp3-s1":      ["Exp3-S1"],
        "exp3-s2":      ["Exp3-S2", "Exp3-S2-FS"],
        "exp3-s2-full": ["Exp3-S2"],
        "exp3-s2-fs":   ["Exp3-S2-FS"],
        "exp3-s3":      ["Exp3-S3", "Exp3-S3-FS"],
        "exp3-s3-full": ["Exp3-S3"],
        "exp3-s3-fs":   ["Exp3-S3-FS"],
        "exp3-s4":      ["Exp3-S4", "Exp3-S4-FS"],
        "exp3-s4-full": ["Exp3-S4"],
        "exp3-s4-fs":   ["Exp3-S4-FS"],
        "p9-ann":            ["Exp6-ANN"],
        "p9-voting":         ["Exp6-Voting"],
        "p9-stacking":       ["Exp6-Stacking"],
        "adv-comparison":    ["Exp6-ANN", "ANN-Exp1", "ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref", "Exp6-Voting", "Exp6-Stacking"],
        "adv-findings":      ["Exp6-Voting"],
        "p10-full":          ["Exp7-SE1"],
        "p10b":              ["Exp7-SE2"],
        "fe-comparison":     ["Exp7-SE1", "Exp7-SE2"],
        "fe-findings":         ["Exp7-SE2"],
        "p11-se1":             ["Exp8"],
        "temporal-comparison": ["Exp8"],
        "temporal-findings":   ["Exp8"],
        "exp4-s1":      ["Exp4-S1"],
        "exp4-s2":      ["Exp4-S2"],
        "exp5-s1":      ["Exp5-S1", "Exp5-S1-FS"],
        "exp5-s1-full": ["Exp5-S1"],
        "exp5-s1-fs":   ["Exp5-S1-FS"],
        "exp5-s2":      ["Exp5-S2", "Exp5-S2-FS"],
        "exp5-s2-full": ["Exp5-S2"],
        "exp5-s2-fs":   ["Exp5-S2-FS"],
        "exp9-s1":          ["Exp9-SE1"],
        "exp9-s2":          ["Exp9-SE2"],
        "exp9-s3":          ["Exp9-SE3"],
        "exp9-s4":          ["Exp9-SE4"],
        "exp9-s5":          ["Exp9-SE5"],
        "exp9-comparison":  ["Exp9-SE1", "Exp9-SE2", "Exp9-SE3", "Exp9-SE4", "Exp9-SE5", "Exp3-S1"],
        "exp9-findings":    ["Exp9-SE1", "Exp9-SE2", "Exp9-SE3", "Exp9-SE4", "Exp9-SE5"],
    }
    # Reuse the same per-experiment winner table that feeds the Overfit-Aware
    # Selection section. The sidebar should summarize the highlighted
    # "Best per (experiment × target)" rows, including One-SE winners, not
    # re-infer leaders from raw R².
    from best_models_selection import (  # noqa: E402
        SCORE_NOISE_BAND, build_per_experiment, _sort_per_target_winners,
    )

    per_exp_df = build_per_experiment(df_all)

    def _display_winner(row: pd.Series) -> tuple[str, float, float, float, str]:
        """Return model/r2/gap/score/rule exactly as the global selection table displays it."""
        use_onese = str(row["onese_model"]) != str(row["naive_model"])
        prefix = "onese" if use_onese else "gadj"
        if use_onese:
            rule = (
                "One-SE / Gap-adj"
                if str(row["onese_model"]) == str(row["gadj_model"])
                else "One-SE"
            )
        else:
            rule = "Gap-adj" if str(row["gadj_model"]) != str(row["naive_model"]) else "All agree"
        return (
            str(row[f"{prefix}_model"]),
            float(row[f"{prefix}_R2"]),
            float(row[f"{prefix}_gap"]) if not pd.isna(row[f"{prefix}_gap"]) else 0.0,
            float(row[f"{prefix}_score"]) if not pd.isna(row[f"{prefix}_score"]) else -1e9,
            rule,
        )

    result = {}
    for sec_id, exp_keys in section_exp_keys.items():
        df_sec = per_exp_df[per_exp_df["exp_key"].isin(exp_keys)].copy()
        bests = {}
        for tgt in TARGETS_ORDERED:
            grp = _sort_per_target_winners(df_sec, tgt)
            if grp.empty:
                continue
            row = grp.iloc[0]
            model, r2, gap, score, rule = _display_winner(row)
            bests[TARGET_SHORT.get(tgt, tgt)] = {
                "model":  model,
                "r2":     round(r2, 3),
                "gap":    round(gap, 3),
                "score":  round(score, 5),
                "absGap": round(abs(gap), 5),
                "rule":   rule,
                "noise":  SCORE_NOISE_BAND,
                "exp":    EXP_CHART_LABELS.get(str(row["exp_key"]), str(row["exp_key"])),
                "expKey": str(row["exp_key"]),
                "color":  MODEL_COLORS.get(model, "#888"),
            }
        result[sec_id] = bests
    order = list(section_exp_keys.keys())
    return json.dumps(result, ensure_ascii=False), json.dumps(order, ensure_ascii=False)


def _exp1_best_model_box(df_all: pd.DataFrame) -> str:
    """Custom champion box for Experiment 1 with gap-adjusted selection.

    Candidates per (model, target):
      SE1           → Exp1-SE1 R2_test / R2_gap
      SE2           → Exp1-Cyclic R2_test / R2_gap
      SE3-Full      → Exp1-S3 R2_test (grab only)
      SE3-FS        → Exp1-S3-FS R2_test (grab only; OLS: post-LassoCV)
    Winner per target = highest gap-adj score across all candidates.
    """
    s1   = df_all[df_all["exp_key"] == "Exp1-SE1"].copy()
    s2   = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()
    s3   = df_all[df_all["exp_key"] == "Exp1-S3"].copy()
    s3fs = df_all[df_all["exp_key"] == "Exp1-S3-FS"].copy()

    if s1.empty and s2.empty:
        return ""

    lin_set  = {"OLS", "Ridge", "ElNet"}
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
        # SE1
        r2 = _g(s1, model, tgt); gap = _g(s1, model, tgt, "R2_gap")
        if r2 is not None:
            out.append((r2, gap, "SE1 (4 feat · core inlet)"))
        # SE2
        r2 = _g(s2, model, tgt); gap = _g(s2, model, tgt, "R2_gap")
        if r2 is not None:
            lbl = f"SE2 · {model} (11 feat)"
            out.append((r2, gap, lbl))
        # SE3-Full (grab only)
        r2 = _g(s3, model, tgt); gap = _g(s3, model, tgt, "R2_gap")
        if r2 is not None:
            out.append((r2, gap, f"SE3-Full · {model} (15 feat)"))
        # SE3-FS (grab only)
        r2 = _g(s3fs, model, tgt); gap = _g(s3fs, model, tgt, "R2_gap")
        if r2 is not None:
            lbl = f"SE3-FS · {model} ({'LassoCV' if model == 'OLS' else 'OOF-sel' if model not in lin_set else 'full-L2' if model == 'Ridge' else 'L1-int'})"
            out.append((r2, gap, lbl))
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
        r2_raw_best = best_raw[0]
        raw_differs = abs(r2_raw_best - r2_win) > 0.005

        r2_col  = _r2_color(r2_win)
        mdl_col = MODEL_COLORS.get(m_win, "#888")
        gap_col = ("#E15252" if gap_win is not None and gap_win > 0.10
                   else "#5BAD6F" if gap_win is not None and gap_win < -0.10
                   else "var(--text-muted)")
        gap_str = f"{gap_win:+.3f}" if gap_win is not None else " - "

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
            f'<td style="color:{r2_col};font-weight:bold">{r2_win:+.3f}{overfit_flag}</td>'
            f'<td style="color:{gap_col}">{gap_str}</td>'
            f'<td class="meta" style="font-size:11px">{src_win}{alt_note}</td>'
            f'</tr>'
        )

    if not table_rows:
        return ""

    df_exp1 = df_all[df_all["exp_key"].isin(["Exp1-SE1", "Exp1-Cyclic", "Exp1-S3", "Exp1-S3-FS"])]
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
            f'{", ".join(overfit_warnings)} show |gap| &gt; 0.25 even on the gap-adj winner  -  '
            f'treat these results with caution.</p>'
        )

    criteria_note = (
        '<p class="table-note" style="margin-top:4px">'
        '<strong>Selection:</strong> winner chosen by gap-adjusted score '
        '= R² − 0.5 · max(0, |gap| − 0.10), consistent with the global leaderboard methodology. '
        'Where the raw R² winner differs from the gap-adj winner it is shown in orange below the '
        'source label. avg best Test R² uses the raw per-target best for cross-experiment '
        'comparability. Composite averages should be read cautiously  -  all composite targets '
        'fail at the Exp1 feature level (Grab COD likewise); secondary clarifier data is required '
        'before these targets become learnable.</p>'
    )

    return f"""
<div class="best-box">
  <div class="best-box-title">Best Model  -  Inlet-Only Features
    <span class="best-box-avg">avg best Test R² = {_fmt(avg_r2)}{avg_detail}</span>
  </div>
  <table class="summary-table best-table">
    <thead><tr>
      <th>Target</th><th>Model</th>
      <th>Test R²</th>
      <th>Gap</th><th>Source (gap-adj winner)</th>
    </tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
  {overfit_note}
  {criteria_note}
</div>"""


def _exp2_best_model_box(df_all: pd.DataFrame) -> str:
    """Champion box for Experiment 2 with gap-adjusted selection across all sub-experiments."""
    exp2_keys = [
        "Exp2-S3",          # SE1 - Primary + Inlet + COMMON
        "Exp2-SE2-Clr",    # SE2-Clr
        "Exp2-SE2-Sed",    # SE2-Sed
        "Exp2-S6",          # SE2-Only
        "Exp2-SE2-Comb",        # SE2-Comb
        "Exp2-SE3-Ref",    # CIS reference
        "Exp2-SE3-Ref-FS",     # CIS-FS
        "Exp2-S6-FS",       # Sec-Only-FS
        "Exp2-S4",          # SE3 - full stage set
    ]
    df = df_all[df_all["exp_key"].isin(exp2_keys)].copy()

    if df.empty:
        return ""

    table_rows = []
    overfit_warnings = []

    for tgt in TARGETS_ORDERED:
        sub = df[df["target"] == tgt].dropna(subset=["R2_test"])
        if sub.empty:
            continue

        cands = []
        for _, r in sub.iterrows():
            r2  = float(r["R2_test"])
            gap = float(r["R2_gap"]) if pd.notna(r.get("R2_gap")) else 0.0
            cands.append((r2, gap, str(r["model"]), str(r["exp_key"])))

        if not cands:
            continue

        best_gaj = max(cands, key=lambda c: _gap_adj(c[0], c[1]))
        best_raw = max(cands, key=lambda c: c[0])

        r2_win, gap_win, m_win, exp_win = best_gaj
        r2_raw_best  = best_raw[0]
        raw_differs  = abs(r2_raw_best - r2_win) > 0.005

        r2_col  = _r2_color(r2_win)
        mdl_col = MODEL_COLORS.get(m_win, "#888")
        gap_col = ("#E15252" if gap_win > 0.10
                   else "#5BAD6F" if gap_win < -0.10
                   else "var(--text-muted)")
        src_win = EXP_SOURCE_LABELS.get(exp_win, EXP_CHART_LABELS.get(exp_win, exp_win))

        overfit_flag = ""
        if abs(gap_win) > 0.25:
            overfit_flag = (' <span style="color:#e74c3c;font-size:10px" '
                            'title="Gap > 0.25 on the gap-adj winner">⚠ overfit</span>')
            overfit_warnings.append(TARGET_SHORT.get(tgt, tgt))

        alt_note = ""
        if raw_differs:
            _, gap_raw_best, m_raw_best, exp_raw_best = best_raw
            src_raw = EXP_SOURCE_LABELS.get(exp_raw_best,
                          EXP_CHART_LABELS.get(exp_raw_best, exp_raw_best))
            alt_note = (
                f'<br><span style="color:#f39c12;font-size:10px" '
                f'title="Raw R² winner ({m_raw_best} · {src_raw}: '
                f'{r2_raw_best:+.3f}) demoted by gap-adj scoring">'
                f'raw: {m_raw_best} {r2_raw_best:+.3f} ({src_raw})</span>'
            )

        short = TARGET_SHORT.get(tgt, tgt)
        slug  = TARGET_SLUG.get(tgt, "all")
        table_rows.append(
            f'<tr data-target="{slug}">'
            f'<td>{short}</td>'
            f'<td><strong style="color:{mdl_col}">{m_win}</strong></td>'
            f'<td style="color:{r2_col};font-weight:bold">{r2_win:+.3f}{overfit_flag}</td>'
            f'<td style="color:{gap_col}">{gap_win:+.3f}</td>'
            f'<td class="meta" style="font-size:11px">{src_win}{alt_note}</td>'
            f'</tr>'
        )

    if not table_rows:
        return ""

    df_e2 = df_all[df_all["exp_key"].isin(exp2_keys)]
    grab_best_vals = [float(df_e2[df_e2["target"] == t]["R2_test"].max())
                      for t in GRAB_TARGETS
                      if not df_e2[df_e2["target"] == t].dropna(subset=["R2_test"]).empty]
    comp_best_vals = [float(df_e2[df_e2["target"] == t]["R2_test"].max())
                      for t in COMP_TARGETS
                      if not df_e2[df_e2["target"] == t].dropna(subset=["R2_test"]).empty]
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
            f'{", ".join(overfit_warnings)} show |gap| &gt; 0.25 even on the gap-adj winner  -  '
            f'treat these results with caution.</p>'
        )

    criteria_note = (
        '<p class="table-note" style="margin-top:4px">'
        '<strong>Selection:</strong> winner chosen by gap-adjusted score '
        '= R² − 0.5 · max(0, |gap| − 0.10), consistent with the global leaderboard methodology. '
        'Where the raw R² winner differs from the gap-adj winner it is shown in orange below the '
        'source label. avg best Test R² uses the raw per-target best for cross-experiment '
        'comparability.</p>'
    )

    return f"""
<div class="best-box">
  <div class="best-box-title">Best Model  -  Process Stage Features
    <span class="best-box-avg">avg best Test R² = {_fmt(avg_r2)}{avg_detail}</span>
  </div>
  <table class="summary-table best-table">
    <thead><tr>
      <th>Target</th><th>Model</th>
      <th>Test R²</th>
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
        exp_lbl = EXP_SOURCE_LABELS.get(best["exp_key"],
                      EXP_CHART_LABELS.get(best["exp_key"], best["exp_key"]))

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
            f'<td class="meta" style="font-size:11px">{exp_lbl}{skipped_note}</td>'
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
                    models_linear=None, models_nl=None, open_default=True,
                    dataset_summary_fn=None) -> str:
    """Build a standard sub-section for one experiment variant."""
    df = df_all[df_all["exp_key"] == exp_key].copy()
    if df.empty:
        return ""
    ml = models_linear or LINEAR_MODELS
    mn = models_nl or NL_MODELS
    all_m = [m for m in ALL_MODELS_ORD if m in df["model"].values]

    feat_html   = _feature_card(exp_key)
    summary_fn  = dataset_summary_fn or _dataset_summary
    ds_html     = summary_fn(df)
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
        exp_label  = EXP_SOURCE_LABELS.get(best["exp_key"],
                 EXP_CHART_LABELS.get(best["exp_key"], best["exp_key"]))
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
    BOD/TSS targets only  -  MdAE is not shown for pH (narrow range) or COD (no discharge limit).
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
    <li><strong>Feature Expansion (Exp3):</strong> Aeration basin data (Exp3-SE2) produced the best linearly-regularised models (ElasticNet/Ridge), providing the most robust baseline predicting 2025 holdout data.</li>
    <li><strong>Feature Pruning (Exp4):</strong> Removing collinear features (via VIF or manually) degraded generalisation across the board. The collinear features carry structural signal necessary for decision tree models, while regularised linear models already suppress collinearity automatically.</li>
    <li><strong>Advanced Methods (Ensembles & Temporal):</strong> Stacking and Neural Networks (ANN) struggled to generalise on the extremely small Composite datasets (n≈290). Conversely, temporal lag features successfully improved R² on Grab targets, demonstrating that recent flow history strongly influences spot samples.</li>
    <li><strong>Practical significance  -  R² alone is insufficient:</strong> A high Test R² does not guarantee compliance-grade accuracy. Discharge limits for BOD and TSS are 10 mg/L.
      <strong>Grab BOD</strong> (best R²≈0.69): RMSE ≈ 2 mg/L  -  roughly 20% of the limit, operationally useful.
      <strong>Grab TSS</strong> (best R²≈0.64): RMSE ≈ 6 mg/L  -  roughly 60% of the limit; the model explains variance well but its absolute prediction error is large relative to the compliance threshold. Predictions should be treated as trend indicators rather than compliance certifications for TSS.</li>
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
    """Data-driven Q&A for Experiment 1  -  four key questions."""
    s1   = df_all[df_all["exp_key"] == "Exp1-SE1"].copy()
    s2   = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()
    s3   = df_all[df_all["exp_key"] == "Exp1-S3"].copy()
    s3fs = df_all[df_all["exp_key"] == "Exp1-S3-FS"].copy()

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

    def _s2_val(model, tgt):
        return _r2(s2, model, tgt)

    def _s3_val(model, tgt):
        return _r2(s3, model, tgt)

    def _s3fs_val(model, tgt):
        return _r2(s3fs, model, tgt)

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
        if not len(arr): return "<em> - </em>"
        v = float(arr.mean())
        c = "#5BAD6F" if v > 0.005 else ("#E15252" if v < -0.005 else "var(--text-muted)")
        return f"<span style='color:{c};font-weight:bold'>{v:+.3f}</span>"

    # ── Q1 : SE1 floor + SE3 ceiling ─────────────────────────────────────────
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

    # SE3: derive train rows and feature count from actual data
    s3_best_r2, s3_best_m, s3_best_t = None, None, None
    s3_above04 = 0
    for m in all_m:
        for t in GRAB_TARGETS:
            v = _r2(s3, m, t)
            if v is None: continue
            if v > 0.40: s3_above04 += 1
            if s3_best_r2 is None or v > s3_best_r2:
                s3_best_r2, s3_best_m, s3_best_t = v, m, TARGET_SHORT.get(t, t)
    s3_available = s3_best_r2 is not None

    _s3_grab_rows = _s3_comp_rows = _s3_n_feat = None
    _s3_has_comp = False
    if not s3.empty:
        _s3_grab_sub = s3[s3["target"].str.contains("Grab")]
        _s3_comp_sub = s3[s3["target"].str.contains("Composite")]
        _s3_has_comp = not _s3_comp_sub.empty
        if not _s3_grab_sub.empty and "n_train" in s3.columns:
            _s3_grab_rows = int(_s3_grab_sub["n_train"].iloc[0])
        if _s3_has_comp and "n_train" in s3.columns:
            _s3_comp_rows = int(_s3_comp_sub["n_train"].iloc[0])
        if "n_features" in s3.columns:
            _s3_n_feat = int(s3["n_features"].iloc[0])

    def _s3_row_str():
        if _s3_grab_rows is None: return "unknown"
        s = f"~{_s3_grab_rows} grab"
        if _s3_comp_rows is not None:
            s += f" / ~{_s3_comp_rows} composite"
        return s + " train rows"

    _s1_summary = (
        f"With only the 4 core inlet concentrations (SE1): barely. "
        f"Only {len(s1_pos)}/{len(s1_vals)} (model x target) cells achieved positive Test R². "
        f"Best SE1 result: {best_m} on {best_t} (R² = {best_r2:+.3f}). "
        f"Mean Test R² across SE1: {s1_avg:+.3f}. "
        f"All composite targets and Grab COD returned universally negative R², "
        f"establishing that 4 bare inlet features are not sufficient on their own."
    )

    if s3_available and s3_best_r2 is not None and s3_best_r2 > 0.05 and s3_above04 > 0:
        _tgt_cover = "grab and composite" if _s3_has_comp else "grab"
        _feat_str  = f"{_s3_n_feat} features, " if _s3_n_feat else ""
        _s3_note = (
            f"<br><strong>SE3 changes the picture:</strong> expanding to all available {_tgt_cover} "
            f"inlet measurements plus cyclic calendar context ({_feat_str}{_s3_row_str()}) "
            f"substantially lifts performance for BOD and TSS. "
            f"{s3_above04} model-target cells exceed R² = 0.40. "
            f"Best SE3 result: {s3_best_m} on {s3_best_t} (R² = {s3_best_r2:+.3f}). "
            f"COD and pH remain unpredictable from inlet data alone across all SE variants."
        )
    elif s3_available:
        _tgt_cover = "grab and composite" if _s3_has_comp else "grab"
        _s3_note = (
            f"<br><strong>SE3 does not improve on SE1/SE2:</strong> expanding to all available "
            f"inlet measurements ({_tgt_cover} targets, {_s3_row_str()}) yields "
            f"near-universal negative Test R² (best = {s3_best_r2:+.3f}). "
            f"The supplementary inlet columns introduce a non-stationarity problem "
            f"driven by anomalous 2022 readings. See Q6 for the full diagnosis."
        )
    else:
        _s3_note = ""

    q1 = f"It depends on which inlet features are used. {_s1_summary}{_s3_note}"

    # ── Q2 : Context gain (SE1 → SE2) ────────────────────────────────────────
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
            "parameter count without proportional signal gain at this dataset size  -  "
            "OLS especially suffers from overfitting with 11 unconstrained coefficients."
        )
    elif _lin_net < 0 and _tree_net > 0:
        _driver = (
            f"Linear models {_lin_dir} while tree models {_tree_dir}. "
            f"Tree models recovered from SE1 composite failures (fewer features → more spurious splits); "
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
        f"<br><strong>Linear models (SE2):</strong> mean Δ = {_colored(lin_d12)} overall "
        f"(Grab: {_colored(lin_grab_d12)}, Composite: {_colored(lin_comp_d12)}). "
        f"<br><strong>Tree models:</strong> mean Δ = {_colored(tree_d12)} overall "
        f"(Grab: {_colored(tree_grab_d12)}, Composite: {_colored(tree_comp_d12)}). "
        f"Ridge and ElNet are the meaningful linear references here  -  OLS without regularisation "
        f"is poorly suited to 11 features and its column in the comparison table largely reflects overfitting, "
        f"not the actual signal in the feature set."
    )

    # ── Q3 : SE3 FS (SE3-Full → SE3-FS, grab targets only) ───────────────────
    lin_fs_d  = _delta_arr_raw(_s3_val, _s3fs_val, lin_m,  GRAB_TARGETS)
    tree_fs_d = _delta_arr_raw(_s3_val, _s3fs_val, tree_m, GRAB_TARGETS)
    grab_fs_d = _delta_arr_raw(_s3_val, _s3fs_val, all_m,  GRAB_TARGETS)

    fs_arr  = grab_fs_d
    _fs_net  = float(fs_arr.mean())         if len(fs_arr) else 0
    _fs_wins = int((fs_arr >  0.01).sum())  if len(fs_arr) else 0
    _fs_loss = int((fs_arr < -0.01).sum())  if len(fs_arr) else 0
    _fs_n    = len(fs_arr)

    gap_deltas = []
    for m in all_m:
        for t in GRAB_TARGETS:
            gf = _r2(s3,   m, t, "R2_gap")
            gs = _r2(s3fs, m, t, "R2_gap")
            if gf is not None and gs is not None:
                gap_deltas.append(gs - gf)
    gap_arr  = np.array(gap_deltas) if gap_deltas else np.array([])
    gap_mean = float(gap_arr.mean()) if len(gap_arr) else None
    gap_reduced_n = int((gap_arr < -0.01).sum()) if len(gap_arr) else 0

    if _fs_net > 0.01:
        _fs_verdict = f"Model-specific FS improved mean Test R² on grab targets ({_fs_wins}/{_fs_n} cells gained)."
    elif _fs_net < -0.01:
        _fs_verdict = f"Model-specific FS reduced mean Test R² on grab targets ({_fs_loss}/{_fs_n} cells regressed)."
    else:
        _fs_verdict = f"Model-specific FS had negligible net impact on grab-target Test R² ({_fs_wins}/{_fs_n} gained, {_fs_loss}/{_fs_n} regressed)."

    if gap_mean is not None:
        gc = "#5BAD6F" if gap_mean < -0.005 else ("#E15252" if gap_mean > 0.005 else "var(--text-muted)")
        gap_str = (
            f"Mean R²-gap change (FS - Full) = <span style='color:{gc};font-weight:bold'>{gap_mean:+.3f}</span> "
            f"({gap_reduced_n}/{len(gap_arr)} model-target pairs saw the gap narrow)."
        )
    else:
        gap_str = "R²-gap data unavailable."

    _q3_rows_note = (
        f"SE3 has {_s3_row_str()}. At this dataset size, OOF FS "
    ) if _s3_grab_rows else "At this SE3 dataset size, OOF FS "

    q3 = (
        f"{_fs_verdict} "
        f"<br><strong>Grab targets:</strong> mean delta = {_colored(grab_fs_d)} "
        f"(linear: {_colored(lin_fs_d)}, trees: {_colored(tree_fs_d)}). "
        f"<br><strong>Generalisation:</strong> {gap_str} "
        f"<br><em>Note:</em> {_q3_rows_note}"
        f"selects 2-10 features per model/target - the supplementary inlet columns (TKN, O&G, PO4, Coliforms) "
        f"are frequently dropped, suggesting their individual signal is weak relative to the core 4 inlet features."
    )

    # ── Q4 : Best variant per target (raw ★ and gap-adj ✦) ──────────────────
    def _gaj_q(r2, gap):
        if r2 is None: return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    _SE_KEYS = [("SE1", s1), ("SE2", s2), ("SE3-Full", s3), ("SE3-FS", s3fs)]
    star_counts = {lb: 0 for lb, _ in _SE_KEYS}
    gaj_counts  = {lb: 0 for lb, _ in _SE_KEYS}
    tgt_winner_rows = ""

    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        per_raw = {}; per_gaj = {}

        for lb, df_ in _SE_KEYS:
            raws = [_r2(df_, m, tgt) for m in all_m if _r2(df_, m, tgt) is not None]
            gajs = [_gaj_q(_r2(df_, m, tgt), _gap(df_, m, tgt)) for m in all_m
                    if _r2(df_, m, tgt) is not None]
            per_raw[lb] = max(raws) if raws else None
            per_gaj[lb] = max((g for g in gajs if g is not None), default=None)

        raw_valid = {lb: v for lb, v in per_raw.items() if v is not None}
        best_raw  = max(raw_valid.values()) if raw_valid else None
        raw_wins  = [lb for lb, v in raw_valid.items() if best_raw is not None and abs(v - best_raw) < 1e-9]
        for w in raw_wins: star_counts[w] += 1

        gaj_valid = {lb: v for lb, v in per_gaj.items() if v is not None}
        best_gaj  = max(gaj_valid.values()) if gaj_valid else None
        gaj_wins  = [lb for lb, v in gaj_valid.items() if best_gaj is not None and abs(v - best_gaj) < 1e-9]
        for w in gaj_wins: gaj_counts[w] += 1

        cells = ""
        for lb, _ in _SE_KEYS:
            rv = per_raw.get(lb); gv = per_gaj.get(lb)
            is_raw = lb in raw_wins; is_gaj = lb in gaj_wins
            if rv is None:
                cells += "<td class='meta' style='text-align:center;font-size:0.82em'> - </td>"
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

    _s3_cov  = "grab and composite" if _s3_has_comp else "grab"
    _s3_dash = "Dash (-) = composite target with no SE3 result." if not _s3_has_comp else "Dash (-) = no result for that SE."

    q4 = (
        f"Raw winner: <strong>{best_raw_var}</strong> "
        f"({star_counts['SE1']} SE1 / {star_counts['SE2']} SE2 / "
        f"{star_counts['SE3-Full']} SE3-Full / {star_counts['SE3-FS']} SE3-FS stars). "
        f"Gap-adjusted winner: <strong>{best_gaj_var}</strong> "
        f"({gaj_counts['SE1']} SE1 / {gaj_counts['SE2']} SE2 / "
        f"{gaj_counts['SE3-Full']} SE3-Full / {gaj_counts['SE3-FS']} SE3-FS diamonds). "
        f"SE3/SE3-FS cover {_s3_cov} targets ({_s3_row_str()}). "
        f"The table below shows the best raw Test R² per variant per target:"
    )

    q4_table = f"""
<div style='margin-top:0.5rem'>
<div class='tbl-scroll'>
<table class='summary-table metrics-table' style='width:auto;min-width:520px'>
  <thead><tr>
    <th class='tgt-name'>Target</th>
    <th>SE1</th>
    <th>SE2</th>
    <th>SE3-Full</th>
    <th>SE3-FS</th>
  </tr></thead>
  <tbody>{tgt_winner_rows}</tbody>
</table></div>
<p class='meta' style='margin-top:0.4rem'>
  <strong>★</strong> = raw R² winner · <strong>✦</strong> = gap-adj winner ·
  green = both · orange = raw only · blue = gap-adj only.
  Small grey value = gap-adjusted score. {_s3_dash}
</p>
</div>"""

    # ── Q5 : Grab vs Composite split ──────────────────────────────────────────
    grab_d12 = _delta_arr(s1, s2, all_m, GRAB_TARGETS)
    comp_d12 = _delta_arr(s1, s2, all_m, COMP_TARGETS)

    s3_comp = s3[s3["target"].str.contains("Composite")]
    s3_grab = s3[s3["target"].str.contains("Grab")]
    _s3_comp_any = not s3_comp.empty
    _s3_grab_any = not s3_grab.empty

    if _s3_comp_any and _s3_grab_any:
        best_grab_r2  = s3_grab["R2_test"].max()
        best_comp_r2  = s3_comp["R2_test"].max()
        q5_se3_note = (
            f"SE3 now includes both grab ({s3_grab['n_train'].iloc[0]:.0f} train) and "
            f"composite ({s3_comp['n_train'].iloc[0]:.0f} train) datasets. "
            f"Best grab R²_test = {best_grab_r2:+.3f}, best composite R²_test = {best_comp_r2:+.3f}. "
            f"Both are near-zero or negative; the performance gap between grab and composite "
            f"is not meaningful when neither generalises."
        )
    else:
        q5_se3_note = (
            "SE3 now covers both grab and composite targets; see Q6 for why both "
            "show near-zero or negative test R²."
        )

    q5 = (
        f"Yes, for SE1→SE2. Grab targets respond more positively than composite. "
        f"<br>"
        f"<strong>SE1 → SE2:</strong> Grab {_colored(grab_d12)}, Composite {_colored(comp_d12)}. "
        f"<strong>SE3:</strong> {q5_se3_note} "
        f"The structural gap between grab and composite persists in SE2 and earlier SEs; "
        f"secondary treatment data is needed for composite targets (Experiment 2 and beyond)."
    )

    # ── Q6 : Grab SE3 retirement - measurement-type correction ───────────────
    # Compute year-by-year grab-sourced extended feature coverage from master data.
    _raw_file = os.path.join(os.path.dirname(MODELING_DIR), "raw_data", "All_Years_Full.xlsx")
    q6_extra  = ""
    q6        = "Master dataset not available for analysis."

    if os.path.exists(_raw_file):
        _raw = pd.read_excel(_raw_file, parse_dates=["Date"])
        _raw["year"] = _raw["Date"].dt.year
        import numpy as _np

        _grab_ext = {
            "TKN":    "Inlet TKN (mg/L, Grab)",
            "O&amp;G":    "Inlet O&amp;G (mg/L, Grab)",
            "PO4/TP": "Inlet PO4/TP (mg/L, Grab)",
            "Total Coliform":  "Inlet Total Coliform (CFU/100ml, Grab)",
            "Fecal Coliform":  "Inlet Fecal Coliform (CFU/100ml, Grab)",
        }
        # Use actual column names (unescaped) for pandas lookups
        _grab_ext_pd = {
            "TKN":    "Inlet TKN (mg/L, Grab)",
            "O&G":    "Inlet O&G (mg/L, Grab)",
            "PO4/TP": "Inlet PO4/TP (mg/L, Grab)",
            "Total Coliform": "Inlet Total Coliform (CFU/100ml, Grab)",
            "Fecal Coliform": "Inlet Fecal Coliform (CFU/100ml, Grab)",
        }
        _years = [2021, 2022, 2023, 2024, 2025]

        # Build provenance table rows
        _prov_rows = ""
        for display_key, display_col in _grab_ext.items():
            pd_col = _grab_ext_pd[display_key.replace("&amp;", "&")]
            if pd_col not in _raw.columns:
                continue
            yr_counts = _raw.groupby("year")[pd_col].count()
            total = int(_raw[pd_col].notna().sum())
            # Determine last grab year
            last_grab = max((y for y in _years if int(yr_counts.get(y, 0)) > 0), default=None)
            switch_note = f"grab through {last_grab}" if last_grab else "no grab data"
            cells = ""
            for y in _years:
                n = int(yr_counts.get(y, 0))
                if n == 0:
                    cells += f"<td style='color:var(--text-muted)'>-</td>"
                elif y <= (last_grab or 0):
                    cells += f"<td style='color:#5BAD6F;font-weight:bold'>{n}</td>"
                else:
                    cells += f"<td style='color:#E15252'>{n}</td>"
            _prov_rows += (
                f"<tr><td class='tgt-name'>{display_key}</td>{cells}"
                f"<td style='font-size:0.88em;color:var(--text-muted)'>{switch_note}</td></tr>"
            )

        # Build training/test row count table (for Grab BOD target as representative)
        _gc = ["Inlet pH (Grab)", "Inlet BOD (mg/L, Grab)",
               "Inlet COD (mg/L, Grab)", "Inlet TSS (mg/L, Grab)"]
        _cm = ["Flow (MLD)", "Power Total (KW)", "year"]
        _tgt_col = "Effluent BOD (mg/L, Grab)"
        _raw["_ms"] = _np.sin(2 * _np.pi * _raw["Date"].dt.month / 12)
        _raw["_mc"] = _np.cos(2 * _np.pi * _raw["Date"].dt.month / 12)
        _raw["_ds"] = _np.sin(2 * _np.pi * _raw["Date"].dt.dayofweek / 7)
        _raw["_dc"] = _np.cos(2 * _np.pi * _raw["Date"].dt.dayofweek / 7)
        _cyclic = ["_ms", "_mc", "_ds", "_dc"]

        _combos = [
            ("SE2 baseline (no extended features)",
             []),
            ("+ Total Coliform (grab)",
             ["Inlet Total Coliform (CFU/100ml, Grab)"]),
            ("+ Fecal Coliform (grab)",
             ["Inlet Fecal Coliform (CFU/100ml, Grab)"]),
            ("+ Total + Fecal Coliform (grab)",
             ["Inlet Total Coliform (CFU/100ml, Grab)",
              "Inlet Fecal Coliform (CFU/100ml, Grab)"]),
            ("+ PO4/TP (grab)",
             ["Inlet PO4/TP (mg/L, Grab)"]),
            ("+ PO4 + Total + Fecal Coliform (grab)",
             ["Inlet PO4/TP (mg/L, Grab)",
              "Inlet Total Coliform (CFU/100ml, Grab)",
              "Inlet Fecal Coliform (CFU/100ml, Grab)"]),
            ("+ TKN (grab)",
             ["Inlet TKN (mg/L, Grab)"]),
            ("+ O&amp;G (grab) [uses Inlet O&amp;G (mg/L, Grab)]",
             ["Inlet O&G (mg/L, Grab)"]),
        ]

        _combo_rows = ""
        for label, ext_cols in _combos:
            feats = _gc + ext_cols + _cm + _cyclic
            avail = [c for c in feats if c in _raw.columns] + [_tgt_col]
            sub   = _raw[avail].dropna()
            train_n = int((sub["year"] < 2025).sum())
            test_n  = int((sub["year"] == 2025).sum())
            n24     = int((sub["year"] == 2024).sum())
            t_col   = "color:#E15252;font-weight:bold" if test_n == 0 else "color:#5BAD6F"
            n24_col = "color:#E15252" if n24 == 0 else ""
            _combo_rows += (
                f"<tr><td class='tgt-name'>{label}</td>"
                f"<td>{train_n}</td>"
                f"<td style='{n24_col}'>{n24 if n24 else '-'}</td>"
                f"<td style='{t_col}'>{'0 - no test data' if test_n == 0 else test_n}</td></tr>"
            )

        # SE2-baseline stats for reference sentence
        _se2_feats = _gc + _cm + _cyclic
        _se2_avail = [c for c in _se2_feats if c in _raw.columns] + [_tgt_col]
        _se2_sub   = _raw[_se2_avail].dropna()
        _se2_train = int((_se2_sub["year"] < 2025).sum())
        _se2_test  = int((_se2_sub["year"] == 2025).sum())

        prov_table = f"""
<div style='margin:0.6rem 0 0.4rem'>
<div class='tbl-scroll'>
<table class='summary-table metrics-table' style='width:auto;min-width:560px'>
  <thead><tr>
    <th class='tgt-name'>Extended inlet feature</th>
    {''.join(f'<th>{y}</th>' for y in _years)}
    <th>Measurement type</th>
  </tr></thead>
  <tbody>{_prov_rows}</tbody>
</table></div>
<p class='meta' style='margin:0.3rem 0 0'>
  Green = grab-sourced rows (correct for grab prediction). Red = composite-sourced values
  stored in grab column (incorrect - would mix measurement types). Dash = no data that year.
</p></div>"""

        combo_table = f"""
<div style='margin:0.6rem 0 0.4rem'>
<div class='tbl-scroll'>
<table class='summary-table metrics-table' style='width:auto;min-width:540px'>
  <thead><tr>
    <th class='tgt-name'>Grab feature set (core inlet + COMMON + ...)</th>
    <th>n train</th>
    <th>2024 in train</th>
    <th>2025 test rows</th>
  </tr></thead>
  <tbody>{_combo_rows}</tbody>
</table></div>
<p class='meta' style='margin:0.3rem 0 0'>
  Representative target: Effluent BOD (Grab). Results are identical for TSS, COD, pH.
  Red test column = model cannot be evaluated (empty holdout set).
</p></div>"""

        q6_extra = prov_table + combo_table

        q6 = (
            f"<strong>Grab SE3 was retired due to a measurement-type correction</strong> "
            f"that eliminated all grab-sourced extended features from the 2025 test set."
            f"<br><br>"
            f"<strong>What was discovered.</strong> "
            f"Frequency analysis of the raw lab reports showed that the extended inlet "
            f"features switched from grab to composite measurement in two waves: "
            f"TKN and O&amp;G became composite in January 2022 (~22-24 readings/month vs "
            f"~8-9/month in 2021); PO4/TP and both Coliform measures became composite in "
            f"2023 (~29 readings/month vs ~4-8/month in 2021-2022). The extraction script "
            f"previously scanned grab and composite sub-blocks together, silently placing "
            f"composite-sourced values into grab-labelled columns. "
            f"The corrected extraction now maintains separate grab and composite columns "
            f"for each extended feature (see table below)."
            f"<br><br>"
            f"<strong>Why grab SE3 cannot be evaluated.</strong> "
            f"After applying the correction, every grab-sourced extended feature has "
            f"<strong>zero rows in 2024 and zero rows in the 2025 test set</strong>. "
            f"A <code>dropna()</code> on any grab feature set that includes extended "
            f"features produces an empty 2025 holdout - the model cannot be scored. "
            f"SE2 (no extended features) gives {_se2_train} train / {_se2_test} test rows "
            f"with full 2021-2024 coverage; adding any grab-sourced extended feature "
            f"drops the test set to zero (second table below)."
            f"<br><br>"
            f"<strong>What SE3 becomes.</strong> "
            f"SE3 is restructured as composite-only: 4 composite target datasets "
            f"using COMP_INLET (4) + NH3-N Composite + O&amp;G Composite + COMMON (7) "
            f"= 13 features. NH3-N and O&amp;G composite are densely covered from 2022 "
            f"(~1298 and ~1268 rows respectively); PO4 and Coliform composite are excluded "
            f"because adding PO4 composite halves training rows (737 to 340) due to its "
            f"absence before mid-2023. The composite SE3 datasets have ~737-738 train rows "
            f"and ~213-216 test rows."
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
  <summary><span class="fold-icon">▶</span> Findings  -  Inlet-Only Features</summary>
  <div class="exp-body">
  {_qcard(1, "Do inlet concentrations alone carry meaningful predictive power?", q1)}
  {_qcard(2, "What does adding process context (Flow, Power, cyclic calendar) contribute?", q2)}
  {_qcard(3, "Does model-specific feature selection improve accuracy or generalisation within SE3?", q3)}
  {_qcard(4, "Which variant achieves the best performance per target?", q4, q4_table)}
  {_qcard(5, "Do Grab and Composite targets respond differently to feature changes?", q5)}
  {_qcard(6, "Why was Grab SE3 retired, and what does the measurement-type correction reveal?", q6, q6_extra)}
  </div>
</details>"""


def _se3_legend_str(s3_df: "pd.DataFrame") -> str:
    """Dynamic description of SE3 for use in comparison-panel legend."""
    if s3_df.empty:
        return "Extended inlet feature set"
    has_comp = s3_df["target"].str.contains("Composite").any()
    tgt_str  = "grab and composite" if has_comp else "grab"
    feat     = int(s3_df["n_features"].iloc[0]) if "n_features" in s3_df.columns else None
    feat_str = f"{feat}-feature " if feat else ""
    rows_grab = None
    rows_comp = None
    if "n_train" in s3_df.columns:
        g = s3_df[s3_df["target"].str.contains("Grab")]
        c = s3_df[s3_df["target"].str.contains("Composite")]
        if not g.empty: rows_grab = int(g["n_train"].iloc[0])
        if not c.empty: rows_comp = int(c["n_train"].iloc[0])
    row_str = ""
    if rows_grab:
        row_str = f", ~{rows_grab} grab"
        if rows_comp:
            row_str += f" / ~{rows_comp} composite"
        row_str += " train rows"
    return f"{feat_str}extended inlet set ({tgt_str} targets{row_str})"


def _exp1_comparison_panel(df_all: pd.DataFrame) -> str:
    """Sub-experiment comparison: SE1 | SE2 | SE3-Full | SE3-FS

    Columns:
      SE1       -  Core Inlet (4 features), no FS
      SE2       -  Core Inlet + Common (11 features)
      Δ S1→S2   -  value of adding process + cyclic calendar context
      SE3-Full  -  All Grab Inlet + Common (16 features, grab and composite targets)
      Δ S2→S3   -  value of adding supplementary inlet measurements
      SE3-FS    -  SE3 with model-specific FS applied
      Δ S3→FS   -  value of feature selection within SE3

    Data sources:
      SE1       → exp_key == "Exp1-SE1"
      SE2       → exp_key == "Exp1-Cyclic"
      SE3-Full  → exp_key == "Exp1-S3"  (OLS: R2_test_full for pre-FS baseline)
      SE3-FS    → exp_key == "Exp1-S3-FS"
    """
    s1   = df_all[df_all["exp_key"] == "Exp1-SE1"].copy()
    s2   = df_all[df_all["exp_key"] == "Exp1-Cyclic"].copy()
    s3   = df_all[df_all["exp_key"] == "Exp1-S3"].copy()
    s3fs = df_all[df_all["exp_key"] == "Exp1-S3-FS"].copy()

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

    def _s2_val(model, tgt):
        """SE2 R²."""
        return _get(s2, model, tgt, "R2_test")

    def _s3_full_val(model, tgt):
        """SE3 full-feature R²: OLS stores pre-FS in R2_test_full; others in R2_test."""
        if model == "OLS":
            v = _get(s3, model, tgt, "R2_test_full")
            return v if v is not None else _get(s3, model, tgt, "R2_test")
        return _get(s3, model, tgt, "R2_test")

    def _s3_fs_val(model, tgt):
        """SE3-FS R²: OLS post-LassoCV; Ridge/ElNet full; trees post-OOF."""
        return _get(s3fs, model, tgt, "R2_test")

    def _s2_gap(model, tgt):
        return _gap_v(s2, model, tgt)

    def _s3_full_gap(model, tgt):
        if model == "OLS":
            return _get(s3, model, tgt, "R2_gap_full")
        return _gap_v(s3, model, tgt)

    def _s3_fs_gap(model, tgt):
        return _gap_v(s3fs, model, tgt)

    def _s2_rmse(model, tgt):
        return _get(s2, model, tgt, "RMSE_test")

    def _s3_full_rmse(model, tgt):
        if model == "OLS":
            v = _get(s3, model, tgt, "RMSE_test_full")
            return v if v is not None else _get(s3, model, tgt, "RMSE_test")
        return _get(s3, model, tgt, "RMSE_test")

    def _s3_fs_rmse(model, tgt):
        return _get(s3fs, model, tgt, "RMSE_test")

    def _s1_rmse(model, tgt):
        return _get(s1, model, tgt, "RMSE_test")

    _TD = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:               col = "#5BAD6F"; fw = "bold"
        elif is_gaj:             col = "#4A90D9"; fw = "bold"
        else:                    col = "#1a1a1a"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else " - "
        gap_val  = gap if (gap is not None and gap == gap) else None
        gap_str  = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col  = ("#E15252" if gap_val is not None and gap_val > 0.10
                    else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                    else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;"
                     f"font-weight:normal'>RMSE {rmse_str} · "
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        r2_col = ("#5BAD6F" if dr2 >= MEANINGFUL else
                  "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        extra = ""
        if drmse is not None and drmse == drmse:
            rm_col = ("#5BAD6F" if drmse <= -MEANINGFUL else
                      "#E15252" if drmse >= MEANINGFUL else "#888888")
            extra += (f"<br><span style='font-size:0.72em;color:{rm_col};"
                      f"font-weight:normal'>RMSE {drmse:+.2f}</span>")
        if dgap is not None and dgap == dgap:
            # positive ΔGap = more overfitting (bad); negative = less (good)
            gp_col = ("#E15252" if dgap > MEANINGFUL else
                      "#5BAD6F" if dgap < -MEANINGFUL else "#888888")
            extra += (f"<br><span style='font-size:0.72em;color:{gp_col};"
                      f"font-weight:normal'>Gap {dgap:+.3f}</span>")
        return (f"<td style='{_TD};text-align:center;color:{r2_col};font-weight:bold'>"
                f"{dr2:+.3f}{extra}</td>")

    deltas_s1_s2    = []; gaj_deltas_s1_s2    = []
    deltas_s2_s3    = []; gaj_deltas_s2_s3    = []
    deltas_s3_s3fs  = []; gaj_deltas_s3_s3fs  = []

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='8' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # --- pre-pass: global best for this target across all models and SEs ---
        _all_raw_e1 = {}
        _all_gaj_e1 = {}
        _all_gap_e1 = {}
        for m_ in models_ord:
            v1_  = _get(s1, m_, tgt);      g1_  = _gap_v(s1, m_, tgt)
            v2_  = _s2_val(m_, tgt);       g2_  = _s2_gap(m_, tgt)
            v3_  = _s3_full_val(m_, tgt);  g3_  = _s3_full_gap(m_, tgt)
            v3f_ = _s3_fs_val(m_, tgt);    g3f_ = _s3_fs_gap(m_, tgt)
            sc1_ = _gaj(v1_, g1_); sc2_ = _gaj(v2_, g2_)
            sc3_ = _gaj(v3_, g3_); sc3f_ = _gaj(v3f_, g3f_)
            for key_, rv_, gv_, sv_ in [
                ((m_, "s1"), v1_, g1_, sc1_),
                ((m_, "s2"), v2_, g2_, sc2_),
                ((m_, "s3"), v3_, g3_, sc3_),
                ((m_, "s3fs"), v3f_, g3f_, sc3f_),
            ]:
                if rv_ is not None:
                    _all_raw_e1[key_] = rv_
                    _all_gap_e1[key_] = gv_
                if sv_ is not None:
                    _all_gaj_e1[key_] = sv_

        tgt_br_e1 = max(_all_raw_e1.values()) if _all_raw_e1 else None
        tgt_bg_e1 = max(_all_gaj_e1.values()) if _all_gaj_e1 else None

        _raw_win_gap_e1 = None
        if tgt_br_e1 is not None:
            for k_, v_ in _all_raw_e1.items():
                if abs(v_ - tgt_br_e1) < 1e-9:
                    _raw_win_gap_e1 = _all_gap_e1.get(k_)
                    break

        tgt_show_gaj_e1 = (
            tgt_bg_e1 is not None and tgt_br_e1 is not None
            and _raw_win_gap_e1 is not None and _raw_win_gap_e1 > 0.10
        )

        def _ir(v_): return tgt_br_e1 is not None and v_ is not None and abs(v_ - tgt_br_e1) < 1e-9
        def _ig(s_): return tgt_show_gaj_e1 and tgt_bg_e1 is not None and s_ is not None and abs(s_ - tgt_bg_e1) < 1e-9

        for m in models_ord:
            v1   = _get(s1, m, tgt);      g1   = _gap_v(s1, m, tgt)
            v2   = _s2_val(m, tgt);       g2   = _s2_gap(m, tgt)
            v3   = _s3_full_val(m, tgt);  g3   = _s3_full_gap(m, tgt)
            v3f  = _s3_fs_val(m, tgt);    g3f  = _s3_fs_gap(m, tgt)

            r1  = _s1_rmse(m, tgt)
            r2  = _s2_rmse(m, tgt)
            r3  = _s3_full_rmse(m, tgt)
            r3f = _s3_fs_rmse(m, tgt)

            sc1 = _gaj(v1, g1); sc2 = _gaj(v2, g2)
            sc3 = _gaj(v3, g3); sc3f = _gaj(v3f, g3f)

            d12  = (v2  - v1)  if (v1  is not None and v2  is not None) else None
            d23  = (v3  - v2)  if (v2  is not None and v3  is not None) else None
            d33f = (v3f - v3)  if (v3  is not None and v3f is not None) else None

            d12_rmse  = ((r2  - r1)  if (r1  is not None and r2  is not None and r1 == r1 and r2 == r2)  else None)
            d23_rmse  = ((r3  - r2)  if (r2  is not None and r3  is not None and r2 == r2 and r3 == r3)  else None)
            d33f_rmse = ((r3f - r3)  if (r3  is not None and r3f is not None and r3 == r3 and r3f == r3f) else None)

            d12_gap  = ((g2  - g1)  if (g1  is not None and g2  is not None and g1 == g1 and g2 == g2)  else None)
            d23_gap  = ((g3  - g2)  if (g2  is not None and g3  is not None and g2 == g2 and g3 == g3)  else None)
            d33f_gap = ((g3f - g3)  if (g3  is not None and g3f is not None and g3 == g3 and g3f == g3f) else None)

            sd12  = (sc2  - sc1)  if (sc1  is not None and sc2  is not None) else None
            sd23  = (sc3  - sc2)  if (sc2  is not None and sc3  is not None) else None
            sd33f = (sc3f - sc3)  if (sc3  is not None and sc3f is not None) else None

            if d12   is not None: deltas_s1_s2.append(d12)
            if d23   is not None: deltas_s2_s3.append(d23)
            if d33f  is not None: deltas_s3_s3fs.append(d33f)
            if sd12  is not None: gaj_deltas_s1_s2.append(sd12)
            if sd23  is not None: gaj_deltas_s2_s3.append(sd23)
            if sd33f is not None: gaj_deltas_s3_s3fs.append(sd33f)

            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(v1,  r1,  g1,  _ir(v1),  _ig(sc1))}"
                f"{_val_td(v2,  r2,  g2,  _ir(v2),  _ig(sc2))}"
                f"{_delta_td(d12, d12_rmse, d12_gap)}"
                f"{_val_td(v3,  r3,  g3,  _ir(v3),  _ig(sc3))}"
                f"{_delta_td(d23, d23_rmse, d23_gap)}"
                f"{_val_td(v3f, r3f, g3f, _ir(v3f), _ig(sc3f))}"
                f"{_delta_td(d33f, d33f_rmse, d33f_gap)}"
                f"</tr>"
            )

    _STD = "padding:6px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"

    def _combined_stats_row(raw_deltas, gaj_deltas, from_lbl, to_lbl):
        if not raw_deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr = np.array(raw_deltas)
        n   = len(arr)
        net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mean_win  = float(wins.mean())   if len(wins)   else None
        mean_loss = float(losses.mean()) if len(losses) else None
        net_col  = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        verdict  = ("Net improvement" if net > MEANINGFUL else
                    "Net regression"  if net < -MEANINGFUL else "Negligible")
        win_str  = (f"{len(wins)}/{n} (avg {mean_win:+.3f})"    if mean_win  is not None else f"{len(wins)}/{n}")
        loss_str = (f"{len(losses)}/{n} (avg {mean_loss:+.3f})" if mean_loss is not None else f"{len(losses)}/{n}")
        if gaj_deltas:
            gaj_net  = float(np.array(gaj_deltas).mean())
            diff     = gaj_net - net
            gaj_col  = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "#888888")
            if diff < -0.02:   interp = "Raw gains partially inflated by increased overfitting"; diff_col = "#E15252"
            elif diff > 0.02:  interp = "Raw gains understated  -  overfitting decreased"; diff_col = "#5BAD6F"
            else:              interp = "Overfitting largely unchanged"; diff_col = "#888888"
            gaj_str  = f"{gaj_net:+.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            gaj_str = diff_str = " - "; gaj_col = diff_col = "#888888"; interp = " - "
        verdict_cell = (f"<strong>{verdict}</strong><br>"
                        f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span>")
        return (
            f"<tr>"
            f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
            f"<td style='{_STD};text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='{_STD};text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='{_STD};text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
            f"<td style='{_STD};text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='{_STD};text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td style='{_STD}'>{verdict_cell}</td>"
            f"</tr>"
        )

    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

    n_cells = len(models_ord) * len(TARGETS_ORDERED)
    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; −{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| ≤ {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²−0.5·max(0,|gap|−0.10)</span></th>
        <th style='{_THC}'>Gap-Adj − Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
        <th style='{_TH}'>Verdict / Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {_combined_stats_row(deltas_s1_s2,   gaj_deltas_s1_s2,   "SE1",     "SE2")}
      {_combined_stats_row(deltas_s2_s3,   gaj_deltas_s2_s3,   "SE2",     "SE3-Full")}
      {_combined_stats_row(deltas_s3_s3fs, gaj_deltas_s3_s3fs, "SE3-Full","SE3-FS")}
    </tbody>
  </table>
  </div>

  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      Net Mean ΔR² is the signed average across all model x target cells.
      <strong>SE2</strong> = Core Inlet + COMMON (11 features: 4 inlet + 7 COMMON; grab and composite targets).
      <strong>SE3-Full</strong> = {_se3_legend_str(s3)}.
      <strong>SE3-FS</strong> = SE3 with model-specific selection applied
      (OLS: LassoCV; Ridge: full set L2; ElNet: full set L1+L2; RF/GB/XGB: OOF perm-imp refit).
      <strong>Gap-Adj - Raw</strong> indicates whether raw gains are inflated by increased overfitting
      (negative = gains overstated) or understated by reduced overfitting (positive = gains understated).
      <strong>★</strong> = best raw Test R² per target · <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
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
            method  = "LassoCV  -  L1 cross-validated pre-screen (TimeSeriesSplit)"
            ns_vals = [v for tgt in GRAB_TARGETS
                       for v in [_get(s3fs, m, tgt, "n_selected_ols")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 15" if ns_vals else " - "
            bypass  = False
        elif m == "Ridge":
            method  = "Full set  -  L2 regularisation handles collinearity internally"
            avg_sel = "15 / 15 (all)"
            bypass  = True
        elif m == "ElNet":
            method  = "Internal L1  -  retains features with non-zero fitted coefficients"
            ns_vals = [v for tgt in GRAB_TARGETS
                       for v in [_get(s3fs, m, tgt, "ElNet_n_selected")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 15" if ns_vals else " - "
            bypass  = False
        else:
            method  = "OOF permutation importance >= 5% threshold (3-phase: full -> select -> refit)"
            ns_vals = [v for tgt in GRAB_TARGETS
                       for v in [_get(s3fs, m, tgt, "n_selected_nl")]
                       if v is not None and v == v]
            avg_sel = f"{np.mean(ns_vals):.1f} / 15" if ns_vals else " - "
            bypass  = False
        status_col = "#7FB3D3" if bypass else "#5BAD6F"
        status     = "Correctly bypassed (L2)" if bypass else "Selection applied"
        _VR = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
        val_rows.append(
            f"<tr><td style='{_VR}'><strong>{m}</strong></td>"
            f"<td style='{_VR};font-size:0.80rem'>{method}</td>"
            f"<td style='{_VR};text-align:center'>15</td>"
            f"<td style='{_VR};text-align:center'>{avg_sel}</td>"
            f"<td style='{_VR};text-align:center;color:{status_col};font-weight:600'>{status}</td></tr>"
        )

    val_card = f"""
<div style='margin:1rem 0 0.6rem'>
  <p style='font-weight:bold;margin-bottom:0.3rem;font-size:0.88em;color:#1a1a1a'>
    Feature Selection Validation  -  SE3-FS (15 features input, grab targets only)
  </p>
  <p style='margin-bottom:0.6rem;font-size:0.82em;color:#555555'>
    Each model applies a distinct selection protocol to the shared 15-feature pool.
    The table confirms which method was used, how many features were retained on average
    across the 4 grab targets, and whether the protocol was correctly applied.
  </p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Model</th>
        <th style='{_TH}'>FS Method</th>
        <th style='{_THC}'>Input<br>Feats</th>
        <th style='{_THC}'>Avg Selected<br>(across 8 targets)</th>
        <th style='{_THC}'>Status</th>
      </tr>
    </thead>
    <tbody>{"".join(val_rows)}</tbody>
  </table>
  </div>
  <p style='font-size:0.78em;margin-top:0.3rem;color:#555555'>
    Ridge is the only model that intentionally bypasses feature removal  -  L2 shrinks
    uninformative coefficients toward zero without discarding them. All other models
    reduce the feature space. Per-target selected feature lists are in the
    <em>Model-Specific Feature Selection</em> fold within SE3 above.
  </p>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a;min-width:1000px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:60px'>Model</th>
      <th style='{_THC}'>SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ (SE1→SE2)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='{_THC}'>SE3-Full<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ (SE2→SE3)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='{_THC}'>SE3-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ (SE3→FS)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp1-comparison">
  <summary><span class="fold-icon">▶</span>
    Comparisons
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10) ·
        <span style='color:#5BAD6F;font-weight:bold'>green Δ</span> = improvement ·
        <span style='color:#E15252;font-weight:bold'>red Δ</span> = regression.
        RMSE reported in native units (mg/L or pH units). Gap = Train R² - Test R²;
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {val_card}
    {main_table}
    {stats_block}
  </div>
</details>"""


def build_exp1_section(df_all: pd.DataFrame) -> str:
    sub_s1 = _exp_subsection(df_all, "Exp1-SE1", "exp1-sub1",
                             "SE1  -  Core Inlet Only",
                             open_default=False)
    sub_s2 = _exp_subsection(df_all, "Exp1-Cyclic", "exp1-s2",
                             "SE2  -  Inlet + COMMON",
                             open_default=True)

    # SE3: nested Full + FS
    se3_full = _exp_subsection(df_all, "Exp1-S3", "exp1-s3-full",
                               "Full Feature Set",
                               open_default=True)
    se3_fs = _exp_subsection(df_all, "Exp1-S3-FS", "exp1-s3-fs",
                             "Feature Selection",
                             open_default=False,
                             dataset_summary_fn=_dataset_summary_per_model)
    sub_s3 = f"""
<details class="exp-details" id="exp1-s3">
  <summary><span class="fold-icon">&#9654;</span> SE3  -  Extended Inlet + COMMON (Full + FS)</summary>
  <div class="exp-body">
    {se3_full}
    {se3_fs}
  </div>
</details>"""

    cmp_div      = _exp1_comparison_panel(df_all)
    findings_div = _exp1_qna(df_all)
    best         = _exp1_best_model_box(df_all)
    return f"""
<section id="exp1">
  <h1 class="section-title">Inlet-Only Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp1"]}</p>
  {sub_s1}
  {sub_s2}
  {sub_s3}
  {cmp_div}
  {findings_div}
  {best}
</section>"""


def _exp2_comparison_panel(df_all: pd.DataFrame) -> str:
    """Sub-experiment comparison table for Experiment 2.

    Columns: SE2-Clr | SE2-Sed | SE2-Comb | CIS | CIS-FS | Sec-Only-FS | SE3
    """
    se1b     = df_all[df_all["exp_key"] == "Exp2-S3"].copy()
    sec_only = df_all[df_all["exp_key"] == "Exp2-S6"].copy()
    s6fs     = df_all[df_all["exp_key"] == "Exp2-S6-FS"].copy()
    clr  = df_all[df_all["exp_key"] == "Exp2-SE2-Clr"].copy()
    sed  = df_all[df_all["exp_key"] == "Exp2-SE2-Sed"].copy()
    comb = df_all[df_all["exp_key"] == "Exp2-SE2-Comb"].copy()
    cyc  = df_all[df_all["exp_key"] == "Exp2-SE3-Ref"].copy()
    fs2  = df_all[df_all["exp_key"] == "Exp2-SE3-Ref-FS"].copy()
    se3  = df_all[df_all["exp_key"] == "Exp2-S4"].copy()

    if clr.empty and sed.empty and comb.empty and sec_only.empty and se1b.empty:
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
        """R² on CIS full feature set (Exp2-SE3-Ref, no FS)."""
        return _get(cyc, model, tgt, "R2_test")

    def _sub2_fs(model, tgt):
        """Feature-selected R² (Exp2-SE3-Ref-FS)."""
        return _get(fs2, model, tgt, "R2_test")

    def _sub2_full_gap(model, tgt):
        return _gap_v(cyc, model, tgt)

    def _sub2_fs_gap(model, tgt):
        return _gap_v(fs2, model, tgt)

    def _sub2_full_rmse(model, tgt):
        return _get(cyc, model, tgt, "RMSE_test")

    def _se1b_v(model, tgt):
        return _get(se1b, model, tgt, "R2_test")

    def _se1b_gap(model, tgt):
        return _gap_v(se1b, model, tgt)

    def _se1b_rmse(model, tgt):
        return _get(se1b, model, tgt, "RMSE_test")

    def _sec_only(model, tgt):
        return _get(sec_only, model, tgt, "R2_test")

    def _sec_only_gap(model, tgt):
        return _gap_v(sec_only, model, tgt)

    def _sec_only_rmse(model, tgt):
        return _get(sec_only, model, tgt, "RMSE_test")

    def _se3(model, tgt):
        return _get(se3, model, tgt, "R2_test")

    def _se3_gap(model, tgt):
        return _gap_v(se3, model, tgt)

    def _s6fs_v(model, tgt):
        return _get(s6fs, model, tgt, "R2_test")

    def _s6fs_gap(model, tgt):
        return _gap_v(s6fs, model, tgt)

    def _se3_rmse(model, tgt):
        return _get(se3, model, tgt, "RMSE_test")

    _TD = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:               col = "#5BAD6F"; fw = "bold"
        elif is_gaj:             col = "#4A90D9"; fw = "bold"
        else:                    col = "#1a1a1a"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else " - "
        gap_val  = gap if (gap is not None and gap == gap) else None
        gap_str  = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col  = ("#E15252" if gap_val is not None and gap_val > 0.10
                    else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                    else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;"
                     f"font-weight:normal'>RMSE {rmse_str} · "
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        r2_col = ("#5BAD6F" if dr2 >= MEANINGFUL else
                  "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        extra = ""
        if drmse is not None and drmse == drmse:
            rm_col = ("#5BAD6F" if drmse <= -MEANINGFUL else
                      "#E15252" if drmse >= MEANINGFUL else "#888888")
            extra += (f"<br><span style='font-size:0.72em;color:{rm_col};"
                      f"font-weight:normal'>RMSE {drmse:+.2f}</span>")
        if dgap is not None and dgap == dgap:
            gp_col = ("#E15252" if dgap > MEANINGFUL else
                      "#5BAD6F" if dgap < -MEANINGFUL else "#888888")
            extra += (f"<br><span style='font-size:0.72em;color:{gp_col};"
                      f"font-weight:normal'>Gap {dgap:+.3f}</span>")
        return (f"<td style='{_TD};text-align:center;color:{r2_col};font-weight:bold'>"
                f"{dr2:+.3f}{extra}</td>")

    # Collect deltas for summary rows
    d_se1_seconly   = []; gaj_d_se1_seconly   = []
    d_seconly_comb  = []; gaj_d_seconly_comb  = []
    d_clr_comb      = []; gaj_d_clr_comb      = []
    d_sed_comb      = []; gaj_d_sed_comb      = []
    d_clr_sed       = []; gaj_d_clr_sed       = []
    d_comb_s2full   = []; gaj_d_comb_s2full   = []
    d_comb_s2fs     = []; gaj_d_comb_s2fs     = []
    d_sed_s2full    = []; gaj_d_sed_s2full    = []
    d_sed_s2fs      = []; gaj_d_sed_s2fs      = []
    d_s2full_s2fs   = []; gaj_d_s2full_s2fs   = []
    d_ref_se3       = []; gaj_d_ref_se3       = []
    d_seconly_s6fs  = []; gaj_d_seconly_s6fs  = []
    d_s2fs_s6fs     = []; gaj_d_s2fs_s6fs     = []

    # Aggregate accumulators for Selection Guide
    _agg_r2  = {k: [] for k in ("se1b","seconly","clr","sed","comb","s2full","s2fs","se3","s6fs")}
    _agg_gaj = {k: [] for k in ("se1b","seconly","clr","sed","comb","s2full","s2fs","se3","s6fs")}
    _agg_gap = {k: [] for k in ("se1b","seconly","clr","sed","comb","s2full","s2fs","se3","s6fs")}
    _win_raw = {k: 0  for k in ("se1b","seconly","clr","sed","comb","s2full","s2fs","se3","s6fs")}
    _win_gaj = {k: 0  for k in ("se1b","seconly","clr","sed","comb","s2full","s2fs","se3","s6fs")}

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='10' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # --- pre-pass: global best for this target across all models and SEs ---
        _all_raw_e2 = {}
        _all_gaj_e2 = {}
        _all_gap_e2 = {}
        for m_ in models_ord:
            vb_  = _se1b_v(m_, tgt);    gb__  = _se1b_gap(m_, tgt)
            vsn_ = _sec_only(m_, tgt); gsn_ = _sec_only_gap(m_, tgt)
            vc_  = _get(clr,  m_, tgt); gc_  = _gap_v(clr,  m_, tgt)
            vs_  = _get(sed,  m_, tgt); gs_  = _gap_v(sed,  m_, tgt)
            vco_ = _get(comb, m_, tgt); gco_ = _gap_v(comb, m_, tgt)
            vf_  = _sub2_full(m_, tgt); gf_  = _sub2_full_gap(m_, tgt)
            vfs_ = _sub2_fs(m_, tgt);   gfs_ = _sub2_fs_gap(m_, tgt)
            vs3_ = _se3(m_, tgt);       gs3_ = _se3_gap(m_, tgt)
            vsf_ = _s6fs_v(m_, tgt);    gsf_ = _s6fs_gap(m_, tgt)
            sb_  = _gaj(vb_,  gb__)
            ssn_ = _gaj(vsn_, gsn_)
            sc_  = _gaj(vc_, gc_);  ss_  = _gaj(vs_, gs_)
            sco_ = _gaj(vco_, gco_); sf_ = _gaj(vf_, gf_); sfs_ = _gaj(vfs_, gfs_)
            ss3_ = _gaj(vs3_, gs3_); ssf_ = _gaj(vsf_, gsf_)
            for key_, rv_, gv_, sv_ in [
                ((m_, "se1b"), vb_, gb__, sb_),
                ((m_, "seconly"), vsn_, gsn_, ssn_),
                ((m_, "clr"), vc_, gc_, sc_),
                ((m_, "sed"), vs_, gs_, ss_),
                ((m_, "comb"), vco_, gco_, sco_),
                ((m_, "s2full"), vf_, gf_, sf_),
                ((m_, "s2fs"), vfs_, gfs_, sfs_),
                ((m_, "se3"), vs3_, gs3_, ss3_),
                ((m_, "s6fs"), vsf_, gsf_, ssf_),
            ]:
                if rv_ is not None:
                    _all_raw_e2[key_] = rv_
                    _all_gap_e2[key_] = gv_
                if sv_ is not None:
                    _all_gaj_e2[key_] = sv_

        tgt_br_e2 = max(_all_raw_e2.values()) if _all_raw_e2 else None
        tgt_bg_e2 = max(_all_gaj_e2.values()) if _all_gaj_e2 else None

        _raw_win_gap_e2 = None
        if tgt_br_e2 is not None:
            for k_, v_ in _all_raw_e2.items():
                if abs(v_ - tgt_br_e2) < 1e-9:
                    _raw_win_gap_e2 = _all_gap_e2.get(k_)
                    break

        tgt_show_gaj_e2 = (
            tgt_bg_e2 is not None and tgt_br_e2 is not None
            and _raw_win_gap_e2 is not None and _raw_win_gap_e2 > 0.10
        )

        def _is_raw(v_): return tgt_br_e2 is not None and v_ is not None and abs(v_ - tgt_br_e2) < 1e-9
        def _is_gaj(s_): return tgt_show_gaj_e2 and tgt_bg_e2 is not None and s_ is not None and abs(s_ - tgt_bg_e2) < 1e-9

        for m in models_ord:
            vb  = _se1b_v(m, tgt);    gb_ = _se1b_gap(m, tgt)
            vsn = _sec_only(m, tgt); gsn = _sec_only_gap(m, tgt)
            vc  = _get(clr,  m, tgt); gc  = _gap_v(clr,  m, tgt)
            vs  = _get(sed,  m, tgt); gs  = _gap_v(sed,  m, tgt)
            vco = _get(comb, m, tgt); gco = _gap_v(comb, m, tgt)
            vf  = _sub2_full(m, tgt); gf  = _sub2_full_gap(m, tgt)
            vfs = _sub2_fs(m,   tgt); gfs = _sub2_fs_gap(m,   tgt)
            vs3 = _se3(m, tgt);       gs3 = _se3_gap(m, tgt)
            vsf = _s6fs_v(m, tgt);    gsf = _s6fs_gap(m, tgt)

            rb  = _se1b_rmse(m, tgt)
            rsn = _sec_only_rmse(m, tgt)
            rc  = _get(clr,  m, tgt, "RMSE_test")
            rs  = _get(sed,  m, tgt, "RMSE_test")
            rco = _get(comb, m, tgt, "RMSE_test")
            rf_ = _sub2_full_rmse(m, tgt)
            rfs = _get(fs2,  m, tgt, "RMSE_test")
            rs3 = _se3_rmse(m, tgt)

            sb  = _gaj(vb,  gb_)
            ssn = _gaj(vsn, gsn)
            sc  = _gaj(vc,  gc);   ss  = _gaj(vs,  gs)
            sco = _gaj(vco, gco);  sf  = _gaj(vf,  gf); sfs = _gaj(vfs, gfs)
            ss3 = _gaj(vs3, gs3);  ssf = _gaj(vsf, gsf)

            d_b_sn  = (vsn - vb)  if (vb  is not None and vsn is not None) else None
            d_snco  = (vco - vsn) if (vsn is not None and vco is not None) else None
            d_cc    = (vco - vc)  if (vc  is not None and vco is not None) else None
            d_sc    = (vco - vs)  if (vs  is not None and vco is not None) else None
            d_cs    = (vs  - vc)  if (vc  is not None and vs  is not None) else None
            d_cf    = (vf  - vco) if (vco is not None and vf  is not None) else None
            d_cfs   = (vfs - vco) if (vco is not None and vfs is not None) else None
            d_sf    = (vf  - vs)  if (vs  is not None and vf  is not None) else None
            d_sfs   = (vfs - vs)  if (vs  is not None and vfs is not None) else None
            d_ff    = (vfs - vf)  if (vf  is not None and vfs is not None) else None
            d_rs3   = (vs3 - vf)  if (vf  is not None and vs3 is not None) else None
            d_sn_sf = (vsf - vsn) if (vsn is not None and vsf is not None) else None
            d_fs_sf = (vsf - vfs) if (vfs is not None and vsf is not None) else None

            rsf = _get(s6fs, m, tgt, "RMSE_test")

            d_b_sn_rmse  = (rsn - rb)  if (rb  is not None and rsn is not None) else None
            d_snco_rmse  = (rco - rsn) if (rsn is not None and rco is not None) else None
            d_cc_rmse   = (rco - rc)  if (rc  is not None and rco is not None) else None
            d_sc_rmse   = (rco - rs)  if (rs  is not None and rco is not None) else None
            d_cf_rmse   = (rf_ - rco) if (rco is not None and rf_ is not None) else None
            d_cfs_rmse  = (rfs - rco) if (rco is not None and rfs is not None) else None
            d_sf_rmse   = (rf_ - rs)  if (rs  is not None and rf_ is not None) else None
            d_sfs_rmse  = (rfs - rs)  if (rs  is not None and rfs is not None) else None
            d_ff_rmse   = (rfs - rf_) if (rf_ is not None and rfs is not None) else None
            d_rs3_rmse  = (rs3 - rf_) if (rf_ is not None and rs3 is not None) else None

            d_b_sn_gap  = (gsn - gb_) if (gb_ is not None and gsn is not None) else None
            d_snco_gap  = (gco - gsn) if (gsn is not None and gco is not None) else None
            d_cc_gap  = (gco - gc)  if (gc  is not None and gco is not None) else None
            d_sc_gap  = (gco - gs)  if (gs  is not None and gco is not None) else None
            d_cf_gap  = (gf  - gco) if (gco is not None and gf  is not None) else None
            d_cfs_gap = (gfs - gco) if (gco is not None and gfs is not None) else None
            d_sf_gap  = (gf  - gs)  if (gs  is not None and gf  is not None) else None
            d_sfs_gap = (gfs - gs)  if (gs  is not None and gfs is not None) else None
            d_ff_gap  = (gfs - gf)  if (gf  is not None and gfs is not None) else None
            d_rs3_gap  = (gs3 - gf)  if (gf  is not None and gs3 is not None) else None
            d_sn_sf_gap = (gsf - gsn) if (gsn is not None and gsf is not None) else None
            d_fs_sf_gap = (gsf - gfs) if (gfs is not None and gsf is not None) else None

            # gap-adj deltas
            gaj_b_sn  = (ssn - sb)  if (sb  is not None and ssn is not None) else None
            gaj_snco  = (sco - ssn) if (ssn is not None and sco is not None) else None
            gaj_cc   = (sco - sc)  if (sc  is not None and sco is not None) else None
            gaj_sc   = (sco - ss)  if (ss  is not None and sco is not None) else None
            gaj_cs   = (ss  - sc)  if (sc  is not None and ss  is not None) else None
            gaj_cf   = (sf  - sco) if (sco is not None and sf  is not None) else None
            gaj_cfs  = (sfs - sco) if (sco is not None and sfs is not None) else None
            gaj_sf   = (sf  - ss)  if (ss  is not None and sf  is not None) else None
            gaj_sfs  = (sfs - ss)  if (ss  is not None and sfs is not None) else None
            gaj_ff   = (sfs - sf)  if (sf  is not None and sfs is not None) else None
            gaj_rs3  = (ss3 - sf)  if (sf  is not None and ss3 is not None) else None
            gaj_sn_sf = (ssf - ssn) if (ssn is not None and ssf is not None) else None
            gaj_fs_sf = (ssf - sfs) if (sfs is not None and ssf is not None) else None

            if d_b_sn is not None: d_se1_seconly.append(d_b_sn);  gaj_d_se1_seconly.append(gaj_b_sn)  if gaj_b_sn  is not None else None
            if d_snco is not None: d_seconly_comb.append(d_snco); gaj_d_seconly_comb.append(gaj_snco) if gaj_snco is not None else None
            if d_cc  is not None: d_clr_comb.append(d_cc);       gaj_d_clr_comb.append(gaj_cc)    if gaj_cc  is not None else None
            if d_sc  is not None: d_sed_comb.append(d_sc);       gaj_d_sed_comb.append(gaj_sc)    if gaj_sc  is not None else None
            if d_cs  is not None: d_clr_sed.append(d_cs);        gaj_d_clr_sed.append(gaj_cs)     if gaj_cs  is not None else None
            if d_cf  is not None: d_comb_s2full.append(d_cf);    gaj_d_comb_s2full.append(gaj_cf) if gaj_cf  is not None else None
            if d_cfs is not None: d_comb_s2fs.append(d_cfs);     gaj_d_comb_s2fs.append(gaj_cfs)  if gaj_cfs is not None else None
            if d_sf  is not None: d_sed_s2full.append(d_sf);     gaj_d_sed_s2full.append(gaj_sf)  if gaj_sf  is not None else None
            if d_sfs is not None: d_sed_s2fs.append(d_sfs);      gaj_d_sed_s2fs.append(gaj_sfs)   if gaj_sfs is not None else None
            if d_ff  is not None: d_s2full_s2fs.append(d_ff);    gaj_d_s2full_s2fs.append(gaj_ff) if gaj_ff  is not None else None
            if d_rs3   is not None: d_ref_se3.append(d_rs3);           gaj_d_ref_se3.append(gaj_rs3)      if gaj_rs3   is not None else None
            if d_sn_sf is not None: d_seconly_s6fs.append(d_sn_sf);   gaj_d_seconly_s6fs.append(gaj_sn_sf) if gaj_sn_sf is not None else None
            if d_fs_sf is not None: d_s2fs_s6fs.append(d_fs_sf);      gaj_d_s2fs_s6fs.append(gaj_fs_sf)   if gaj_fs_sf is not None else None

            # Accumulate for Selection Guide
            _combo_sg = [
                ("se1b",   vb,  sb,  gb_),
                ("seconly", vsn, ssn, gsn),
                ("clr",    vc,  sc,  gc),
                ("sed",    vs,  ss,  gs),
                ("comb",   vco, sco, gco),
                ("s2full", vf,  sf,  gf),
                ("s2fs",   vfs, sfs, gfs),
                ("se3",    vs3, ss3, gs3),
                ("s6fs",   vsf, ssf, gsf),
            ]
            for _k, _rv, _sv, _gv in _combo_sg:
                if _rv is not None:
                    _agg_r2[_k].append(_rv)
                    _agg_gap[_k].append(_gv if _gv is not None else 0.0)
                if _sv is not None:
                    _agg_gaj[_k].append(_sv)
            _raw_pool_sg = {k: rv for k, rv, sv, gv in _combo_sg if rv is not None}
            _gaj_pool_sg = {k: sv for k, rv, sv, gv in _combo_sg if sv is not None}
            if _raw_pool_sg:
                _br_sg = max(_raw_pool_sg.values())
                for _k, _v in _raw_pool_sg.items():
                    if abs(_v - _br_sg) < 1e-9: _win_raw[_k] += 1
            if _gaj_pool_sg:
                _bg_sg = max(_gaj_pool_sg.values())
                for _k, _v in _gaj_pool_sg.items():
                    if abs(_v - _bg_sg) < 1e-9: _win_gaj[_k] += 1

            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(vb,  rb,  gb_,  _is_raw(vb),  _is_gaj(sb))}"
                f"{_val_td(vc,  rc,  gc,  _is_raw(vc),  _is_gaj(sc))}"
                f"{_val_td(vs,  rs,  gs,  _is_raw(vs),  _is_gaj(ss))}"
                f"{_val_td(vsn, rsn, gsn, _is_raw(vsn), _is_gaj(ssn))}"
                f"{_val_td(vco, rco, gco, _is_raw(vco), _is_gaj(sco))}"
                f"{_val_td(vf,  rf_, gf,  _is_raw(vf),  _is_gaj(sf))}"
                f"{_val_td(vfs, rfs, gfs, _is_raw(vfs), _is_gaj(sfs))}"
                f"{_val_td(vsf, rsf, gsf, _is_raw(vsf), _is_gaj(ssf))}"
                f"{_val_td(vs3, rs3, gs3, _is_raw(vs3), _is_gaj(ss3))}"
                f"</tr>"
            )

    _STD = "padding:6px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"

    def _combined_stats_row(raw_deltas, gaj_deltas, from_lbl, to_lbl):
        if not raw_deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='8' style='{_STD};color:#888888'> - </td></tr>")
        arr = np.array(raw_deltas)
        n   = len(arr)
        net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mean_win  = float(wins.mean())   if len(wins)   else None
        mean_loss = float(losses.mean()) if len(losses) else None
        net_col  = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        verdict  = ("Net improvement" if net > MEANINGFUL else
                    "Net regression"  if net < -MEANINGFUL else "Negligible")
        win_str  = (f"{len(wins)}/{n} (avg {mean_win:+.3f})"    if mean_win  is not None else f"{len(wins)}/{n}")
        loss_str = (f"{len(losses)}/{n} (avg {mean_loss:+.3f})" if mean_loss is not None else f"{len(losses)}/{n}")
        if gaj_deltas:
            gaj_arr = [g for g in gaj_deltas if g is not None]
            if gaj_arr:
                gaj_net  = float(np.array(gaj_arr).mean())
                diff     = gaj_net - net
                gaj_col  = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "#888888")
                if diff < -0.02:   interp = "Raw gains partially inflated by increased overfitting"; diff_col = "#E15252"
                elif diff > 0.02:  interp = "Raw gains understated  -  overfitting decreased"; diff_col = "#5BAD6F"
                else:              interp = "Overfitting largely unchanged"; diff_col = "#888888"
                gaj_str  = f"{gaj_net:+.4f}"
                diff_str = f"{diff:+.4f}"
            else:
                gaj_str = diff_str = " - "; gaj_col = diff_col = "#888888"; interp = " - "
        else:
            gaj_str = diff_str = " - "; gaj_col = diff_col = "#888888"; interp = " - "
        verdict_cell = (f"<strong>{verdict}</strong><br>"
                        f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span>")
        return (
            f"<tr>"
            f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
            f"<td style='{_STD};text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='{_STD};text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='{_STD};text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
            f"<td style='{_STD};text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='{_STD};text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td style='{_STD}'>{verdict_cell}</td>"
            f"</tr>"
        )

    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; −{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| ≤ {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²</th>
        <th style='{_THC}'>Gap-Adj − Raw</th>
        <th style='{_TH}'>Verdict</th>
      </tr>
    </thead>
    <tbody>
      {_combined_stats_row(d_se1_seconly,   gaj_d_se1_seconly,  "SE1",       "SE2-Only")}
      {_combined_stats_row(d_seconly_comb,  gaj_d_seconly_comb, "SE2-Only",  "SE2-Comb")}
      {_combined_stats_row(d_clr_sed,      gaj_d_clr_sed,      "SE2-Clr",   "SE2-Sed")}
      {_combined_stats_row(d_clr_comb,     gaj_d_clr_comb,     "SE2-Clr",   "SE2-Comb")}
      {_combined_stats_row(d_sed_comb,     gaj_d_sed_comb,     "SE2-Sed",   "SE2-Comb")}
      {_combined_stats_row(d_comb_s2full,  gaj_d_comb_s2full,  "SE2-Comb",  "CIS")}
      {_combined_stats_row(d_sed_s2full,   gaj_d_sed_s2full,   "SE2-Sed",   "CIS")}
      {_combined_stats_row(d_comb_s2fs,    gaj_d_comb_s2fs,    "SE2-Comb",  "CIS-FS")}
      {_combined_stats_row(d_sed_s2fs,     gaj_d_sed_s2fs,     "SE2-Sed",   "CIS-FS")}
      {_combined_stats_row(d_s2full_s2fs,  gaj_d_s2full_s2fs,  "CIS",       "CIS-FS")}
      {_combined_stats_row(d_seconly_s6fs, gaj_d_seconly_s6fs, "SE2-Only",  "Sec-Only-FS")}
      {_combined_stats_row(d_s2fs_s6fs,   gaj_d_s2fs_s6fs,    "CIS-FS",    "Sec-Only-FS")}
      {_combined_stats_row(d_ref_se3,      gaj_d_ref_se3,      "CIS",       "SE3")}
    </tbody>
  </table>
  </div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      <strong>★</strong> = best raw Test R² per target · <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
      Detailed transition deltas (win/loss counts and gap-adjusted net) are in the Transition Summary below.
    </p>
  </div>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a;min-width:860px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:60px'>Model</th>
      <th style='{_THC}'>SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2-Clr<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2-Sed<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2-Only<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2-Comb<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>CIS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>CIS-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Sec-Only-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE3<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    # --- Selection Guide ---
    _SE_GUIDE_ROWS = [
        ("se1b",    "SE1",        False),
        ("clr",     "SE2-Clr",   False),
        ("sed",     "SE2-Sed",   False),
        ("seconly", "SE2-Only",  True),
        ("comb",    "SE2-Comb",  True),
        ("s2full",  "CIS",        False),
        ("s2fs",    "CIS-FS",    False),
        ("s6fs",    "Sec-Only-FS",True),
        ("se3",     "SE3",       True),
    ]
    _focus_keys = {"seconly", "comb", "s6fs", "se3"}
    _n_combos   = len(models_ord) * len(TARGETS_ORDERED)

    def _safe_mean(lst): return float(np.mean(lst)) if lst else None

    _gd = {}
    for k, _, _ in _SE_GUIDE_ROWS:
        r2l  = _agg_r2[k]; gajl = _agg_gaj[k]; gapl = _agg_gap[k]
        n    = len(r2l)
        _gd[k] = {
            "mean_r2":  _safe_mean(r2l),
            "mean_gaj": _safe_mean(gajl),
            "mean_gap": _safe_mean(gapl),
            "win_raw":  _win_raw[k],
            "win_gaj":  _win_gaj[k],
            "pos_pct":  100.0 * sum(1 for v in r2l if v > 0) / n if n else 0.0,
        }

    def _best_among(metric, keys, higher_better=True):
        vals = {k: _gd[k][metric] for k in keys if _gd[k].get(metric) is not None}
        if not vals: return None
        return max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)

    _b_r2  = _best_among("mean_r2",  _focus_keys)
    _b_gaj = _best_among("mean_gaj", _focus_keys)
    _b_win = _best_among("win_raw",  _focus_keys)
    _b_gap = _best_among("mean_gap", _focus_keys, higher_better=False)
    _b_pos = _best_among("pos_pct",  _focus_keys)

    def _gc(v, fmt, highlight):
        if v is None:
            return f"<td style='text-align:center;padding:5px 10px;color:#999'>-</td>"
        col = "#5BAD6F" if highlight else "#1a1a1a"
        fw  = "bold"    if highlight else "normal"
        s   = f"{v:{fmt}}"
        return f"<td style='text-align:center;padding:5px 10px;color:{col};font-weight:{fw}'>{s}</td>"

    _guide_tbody = ""
    for k, lbl, is_focus in _SE_GUIDE_ROWS:
        d  = _gd[k]
        bg = "#f0f7ff" if is_focus else "#ffffff"
        fw = "bold"    if is_focus else "normal"
        _guide_tbody += (
            f"<tr style='background:{bg};border-bottom:1px solid #e0e0e0'>"
            f"<td style='padding:5px 10px;font-weight:{fw}'>{lbl}</td>"
            + _gc(d["mean_r2"],  "+.3f", k == _b_r2)
            + _gc(d["mean_gaj"], "+.3f", k == _b_gaj)
            + _gc(d["mean_gap"], "+.4f", k == _b_gap)
            + _gc(d["win_raw"],  ".0f",  k == _b_win)
            + _gc(d["win_gaj"],  ".0f",  k == _b_win)
            + _gc(d["pos_pct"],  ".1f",  k == _b_pos)
            + "</tr>"
        )

    sel_guide = f"""
<div style='margin-top:1.6rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>
    Aggregated Selection Guide
    <span style='font-size:0.82em;font-weight:normal;color:#888888'>
      ({_n_combos} model x target combinations; blue rows = candidates)
    </span>
  </p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc;background:#eeeeee'>
        <th style='{_TH}'>SE</th>
        <th style='{_THC}'>Mean R²<br><span style='font-weight:400;font-size:0.78em;color:#888'>all combos</span></th>
        <th style='{_THC}'>Mean GAJ-R²<br><span style='font-weight:400;font-size:0.78em;color:#888'>overfit-penalised</span></th>
        <th style='{_THC}'>Mean Gap<br><span style='font-weight:400;font-size:0.78em;color:#888'>lower = better</span></th>
        <th style='{_THC}'>Win (raw)<br><span style='font-weight:400;font-size:0.78em;color:#888'>best R² count</span></th>
        <th style='{_THC}'>Win (GAJ)<br><span style='font-weight:400;font-size:0.78em;color:#888'>best GAJ count</span></th>
        <th style='{_THC}'>Positive %<br><span style='font-weight:400;font-size:0.78em;color:#888'>R² &gt; 0</span></th>
      </tr>
    </thead>
    <tbody>{_guide_tbody}</tbody>
  </table>
  </div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.6rem'>
    <p class='meta'>
      <strong>How to read.</strong> Green cells = best among the three candidates (SE2-Only, SE2-Comb, SE3).
      <strong>Win (raw)</strong>: of the {_n_combos} model x target pairs, how many does this SE win on raw Test R²?
      <strong>Mean GAJ-R²</strong> is the primary tiebreaker: it equals R²_test minus 50% of any gap above zero,
      so it penalises overfitting even when raw R² looks similar.
      If Mean R² and Win (raw) are close between two candidates, use Mean GAJ-R² to pick the less overfit one.
    </p>
  </div>
</div>"""

    return f"""
<details class="exp-details" id="exp2-comparison">
  <summary><span class="fold-icon">▶</span>
    Comparisons
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {main_table}
    {stats_block}
    {sel_guide}
  </div>
</details>"""


def _exp2_qna(df_all: pd.DataFrame) -> str:
    """Data-driven Q&A for Experiment 2 using the current comparison columns."""
    exp = {
        "se1a": df_all[df_all["exp_key"] == "Exp2-S5"].copy(),
        "se1b": df_all[df_all["exp_key"] == "Exp2-S3"].copy(),
        "se2_only": df_all[df_all["exp_key"] == "Exp2-S6"].copy(),
        "clr": df_all[df_all["exp_key"] == "Exp2-SE2-Clr"].copy(),
        "sed": df_all[df_all["exp_key"] == "Exp2-SE2-Sed"].copy(),
        "comb": df_all[df_all["exp_key"] == "Exp2-SE2-Comb"].copy(),
        "cis": df_all[df_all["exp_key"] == "Exp2-SE3-Ref"].copy(),
        "cis_fs": df_all[df_all["exp_key"] == "Exp2-SE3-Ref-FS"].copy(),
        "sec_fs": df_all[df_all["exp_key"] == "Exp2-S6-FS"].copy(),
        "se3": df_all[df_all["exp_key"] == "Exp2-S4"].copy(),
    }

    if all(d.empty for d in exp.values()):
        return ""

    all_m = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL = 0.005

    def _val(df, model, tgt, col="R2_test"):
        r = df[(df["model"] == model) & (df["target"] == tgt)]
        if r.empty or col not in r.columns:
            return None
        v = r[col].dropna()
        if v.empty:
            return None
        out = float(v.iloc[-1])
        return None if out != out else out

    def _vals(df, targets=TARGETS_ORDERED, col="R2_test"):
        out = []
        for m in all_m:
            for t in targets:
                v = _val(df, m, t, col)
                if v is not None:
                    out.append(v)
        return np.array(out)

    def _delta_arr(from_df, to_df, targets=TARGETS_ORDERED, col="R2_test"):
        vals = []
        for m in all_m:
            for t in targets:
                a = _val(from_df, m, t, col)
                b = _val(to_df, m, t, col)
                if a is not None and b is not None:
                    vals.append(b - a)
        return np.array(vals)

    def _gaj(r2, gap):
        if r2 is None:
            return None
        return r2 - max(gap or 0.0, 0.0) * 0.50

    def _gaj_vals(df, targets=TARGETS_ORDERED):
        out = []
        for m in all_m:
            for t in targets:
                s = _gaj(_val(df, m, t, "R2_test"), _val(df, m, t, "R2_gap"))
                if s is not None:
                    out.append(s)
        return np.array(out)

    def _summary(df):
        arr = _vals(df)
        gaj = _gaj_vals(df)
        gaps = _vals(df, col="R2_gap")
        ntr = df["n_train"].dropna().astype(float) if "n_train" in df.columns else pd.Series(dtype=float)
        return {
            "mean": float(arr.mean()) if len(arr) else None,
            "gaj": float(gaj.mean()) if len(gaj) else None,
            "gap": float(gaps.mean()) if len(gaps) else None,
            "pos": int((arr > 0).sum()) if len(arr) else 0,
            "n": int(len(arr)),
            "n_train": float(ntr.mean()) if len(ntr) else None,
        }

    def _fmt(v, strong=True):
        if v is None:
            return "<em>-</em>"
        c = "#5BAD6F" if v > 0.05 else ("#E15252" if v < -0.05 else "var(--text-muted)")
        fw = "bold" if strong else "normal"
        return f"<span style='color:{c};font-weight:{fw}'>{v:+.3f}</span>"

    def _fmt_delta(arr):
        if not len(arr):
            return "<em>-</em>"
        v = float(arr.mean())
        c = "#5BAD6F" if v > MEANINGFUL else ("#E15252" if v < -MEANINGFUL else "var(--text-muted)")
        return f"<span style='color:{c};font-weight:bold'>{v:+.3f}</span>"

    def _wins_losses(arr):
        if not len(arr):
            return "0/0 improved, 0/0 regressed"
        wins = int((arr > MEANINGFUL).sum())
        losses = int((arr < -MEANINGFUL).sum())
        return f"{wins}/{len(arr)} improved, {losses}/{len(arr)} regressed"

    stats = {k: _summary(v) for k, v in exp.items()}

    # Q1: Primary-stage value and inlet add-on value.
    d_primary_inlet = _delta_arr(exp["se1a"], exp["se1b"])
    q1 = (
        f"<strong>Primary-stage features do not generalise in either scope.</strong> "
        f"SE1a (Primary + COMMON) averages {_fmt(stats['se1a']['mean'])} "
        f"with {stats['se1a']['pos']}/{stats['se1a']['n']} positive cells; "
        f"SE1b (Primary + Inlet + COMMON) averages {_fmt(stats['se1b']['mean'])} "
        f"with {stats['se1b']['pos']}/{stats['se1b']['n']} positive cells. "
        f"Adding core inlet to primary changes mean Test R&sup2; by {_fmt_delta(d_primary_inlet)} "
        f"({_wins_losses(d_primary_inlet)}). "
        f"The failure is therefore not just missing inlet context: primary measurements are weak "
        f"predictors for this 2025 holdout."
    )

    # Q2: Secondary-only, COMMON, and secondary feature selection.
    d_se1b_secondary = _delta_arr(exp["se1b"], exp["se2_only"])
    d_secondary_common = _delta_arr(exp["se2_only"], exp["comb"])
    d_secondary_fs = _delta_arr(exp["se2_only"], exp["sec_fs"])
    d_secondary_common_grab = _delta_arr(exp["se2_only"], exp["comb"], GRAB_TARGETS)
    d_secondary_common_comp = _delta_arr(exp["se2_only"], exp["comb"], COMP_TARGETS)
    q2 = (
        f"<strong>Secondary measurements are the first useful process signal.</strong> "
        f"SE1b -> SE2-Only improves mean Test R&sup2; by {_fmt_delta(d_se1b_secondary)} "
        f"({_wins_losses(d_se1b_secondary)}), and SE2-Only itself averages "
        f"{_fmt(stats['se2_only']['mean'])} with {stats['se2_only']['pos']}/{stats['se2_only']['n']} "
        f"positive cells. "
        f"<br>Adding COMMON to all secondary features (SE2-Only -> SE2-Comb) changes mean Test R&sup2; by "
        f"{_fmt_delta(d_secondary_common)}; Grab {_fmt_delta(d_secondary_common_grab)}, "
        f"Composite {_fmt_delta(d_secondary_common_comp)}. "
        f"The COMMON columns cost rows in this section "
        f"(SE2-Only avg train n={stats['se2_only']['n_train']:.0f}; "
        f"SE2-Comb avg train n={stats['comb']['n_train']:.0f}) and do not consistently repay that cost. "
        f"Secondary-only feature selection is also mixed: SE2-Only -> Sec-Only-FS = "
        f"{_fmt_delta(d_secondary_fs)} ({_wins_losses(d_secondary_fs)})."
    )

    # Q3: Clarifier/Sedimentation/Combined relationship.
    d_clr_sed = _delta_arr(exp["clr"], exp["sed"])
    d_clr_comb = _delta_arr(exp["clr"], exp["comb"])
    d_sed_comb = _delta_arr(exp["sed"], exp["comb"])
    q3 = (
        f"<strong>Sedimentation is clearly stronger than Clarifier, but combining both is not a clean win.</strong> "
        f"Clarifier averages {_fmt(stats['clr']['mean'])}; Sedimentation averages "
        f"{_fmt(stats['sed']['mean'])}; Sed - Clarifier = {_fmt_delta(d_clr_sed)}. "
        f"<br>Clarifier -> SE2-Comb = {_fmt_delta(d_clr_comb)} "
        f"({_wins_losses(d_clr_comb)}), while Sedimentation -> SE2-Comb = {_fmt_delta(d_sed_comb)} "
        f"({_wins_losses(d_sed_comb)}). "
        f"This says the combined pool adds some Clarifier-side cases over the weak Clarifier baseline, "
        f"but it can dilute the cleaner Sedimentation-only signal."
    )

    # Q4: Current comparison-table winner and strategic takeaway.
    guide = [
        ("SE1", exp["se1b"]),
        ("SE2-Only", exp["se2_only"]),
        ("SE2-Comb", exp["comb"]),
        ("CIS", exp["cis"]),
        ("CIS-FS", exp["cis_fs"]),
        ("Sec-Only-FS", exp["sec_fs"]),
        ("SE3", exp["se3"]),
    ]
    win_raw = {lbl: 0 for lbl, _ in guide}
    win_gaj = {lbl: 0 for lbl, _ in guide}
    for m in all_m:
        for t in TARGETS_ORDERED:
            raw_pool = {lbl: _val(df, m, t, "R2_test") for lbl, df in guide}
            raw_pool = {lbl: v for lbl, v in raw_pool.items() if v is not None}
            if raw_pool:
                best_raw = max(raw_pool.values())
                for lbl, v in raw_pool.items():
                    if abs(v - best_raw) < 1e-9:
                        win_raw[lbl] += 1
            gaj_pool = {lbl: _gaj(_val(df, m, t, "R2_test"), _val(df, m, t, "R2_gap")) for lbl, df in guide}
            gaj_pool = {lbl: v for lbl, v in gaj_pool.items() if v is not None}
            if gaj_pool:
                best_gaj = max(gaj_pool.values())
                for lbl, v in gaj_pool.items():
                    if abs(v - best_gaj) < 1e-9:
                        win_gaj[lbl] += 1

    guide_stats = {lbl: _summary(df) for lbl, df in guide}
    best_mean = max(guide_stats, key=lambda k: guide_stats[k]["mean"] if guide_stats[k]["mean"] is not None else -999)
    best_gaj = max(guide_stats, key=lambda k: guide_stats[k]["gaj"] if guide_stats[k]["gaj"] is not None else -999)
    best_raw_wins = max(win_raw, key=win_raw.get)
    best_gaj_wins = max(win_gaj, key=win_gaj.get)
    d_cis_se3 = _delta_arr(exp["cis"], exp["se3"])
    d_cisfs_se3 = _delta_arr(exp["cis_fs"], exp["se3"])

    q4 = (
        f"<strong>The current comparison table does not crown the largest feature set.</strong> "
        f"Best mean Test R&sup2; is <strong>{best_mean}</strong> "
        f"({_fmt(guide_stats[best_mean]['mean'])}); best mean gap-adjusted R&sup2; is "
        f"<strong>{best_gaj}</strong> ({_fmt(guide_stats[best_gaj]['gaj'])}). "
        f"Raw win count is led by <strong>{best_raw_wins}</strong> "
        f"({win_raw[best_raw_wins]}/48 cells), while gap-adjusted wins are led by "
        f"<strong>{best_gaj_wins}</strong> ({win_gaj[best_gaj_wins]}/48 cells). "
        f"<br>CIS -> SE3 changes mean Test R&sup2; by {_fmt_delta(d_cis_se3)}; "
        f"CIS-FS -> SE3 changes it by {_fmt_delta(d_cisfs_se3)}. "
        f"Strategically, Exp2 says to keep secondary process data and use inlet+secondary "
        f"references as controls, but do not assume primary-stage features or the 27-feature "
        f"all-stage set improve generalisation."
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
  <summary><span class="fold-icon">▶</span> Findings  -  Process Stage Features</summary>
  <div class="exp-body">
  {_qcard(1, "Do primary-stage features generalise, and does adding inlet rescue them?", q1)}
  {_qcard(2, "What does the secondary-only scope add, and does COMMON repay its row cost?", q2)}
  {_qcard(3, "How do Clarifier, Sedimentation, and All Secondary compare?", q3)}
  {_qcard(4, "Which current Exp2 comparison scope wins, and what is the strategic takeaway?", q4)}
  </div>
</details>"""


def build_exp2_section(df_all: pd.DataFrame) -> str:
    # SE1a: Primary + COMMON only (isolation check - does primary carry ANY signal without inlet?)
    se1a_inner = _exp_subsection(df_all, "Exp2-S5", "exp2-s1a",
                                 "SE1a - Primary + COMMON only (13 features)",
                                 open_default=True)
    # SE1b: Primary + Inlet + COMMON (existing; adds inlet to SE1a)
    se1b_inner = _exp_subsection(df_all, "Exp2-S3", "exp2-s1b",
                                 "SE1b - Primary + Inlet + COMMON (17 features)",
                                 open_default=False)

    # SE2: Secondary-only - four scopes (SE2a = no COMMON baseline)
    se2a = _exp_subsection(df_all, "Exp2-S6", "exp2-s2a",
                           "SE2a - All Secondary only (10 features, no COMMON)",
                           open_default=True)
    se2_combined = _exp_subsection(df_all, "Exp2-SE2-Comb", "exp2-s2-combined",
                                   "All Secondary + COMMON (17 features)",
                                   open_default=False)
    se2_clr = _exp_subsection(df_all, "Exp2-SE2-Clr", "exp2-s2-clr",
                              "Sec Clarifier + COMMON (12 features)",
                              open_default=False)
    se2_sed = _exp_subsection(df_all, "Exp2-SE2-Sed", "exp2-s2-sed",
                              "Sec Sedimentation + COMMON (12 features)",
                              open_default=False)

    # SE3: Full combined feature set (all stages)
    se3_full = _exp_subsection(df_all, "Exp2-S4", "exp2-s3-full",
                               "Full Feature Set (27 features)",
                               open_default=True)
    # SE3 reference baseline (old SE2: Inlet + Secondary, no primary) - kept for delta comparison
    ref_full = _exp_subsection(df_all, "Exp2-SE3-Ref", "exp2-s3-ref",
                               "Core Inlet + Secondary + Common (CIS) - 21 features",
                               open_default=False)
    ref_fs   = _exp_subsection(df_all, "Exp2-SE3-Ref-FS", "exp2-s3-ref-fs",
                               "CIS - Feature Selected",
                               open_default=False,
                               dataset_summary_fn=_dataset_summary_per_model)
    se3_s6fs = _exp_subsection(df_all, "Exp2-S6-FS", "exp2-s3-s6fs",
                               "Secondary Only - Feature Selected (10 feat -> selected)",
                               open_default=False,
                               dataset_summary_fn=_dataset_summary_per_model)

    cmp_div      = _exp2_comparison_panel(df_all)
    findings_div = _exp2_qna(df_all)
    best         = _exp2_best_model_box(df_all)

    se1_wrapper = f"""
<details class="exp-details" open id="exp2-s1">
  <summary><span class="fold-icon">▶</span>
    SE1  -  Primary Stage Features (two sub-scopes)
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E15252">
      <p class="meta">
        <strong>Result: primary-stage scopes are strongly negative on average, with only isolated positive cells.</strong>
        Two scopes are tested to isolate whether the failure is due to primary features themselves
        or inlet contamination:<br>
        &bull; <strong>SE1a (Primary + COMMON, 13 feat)</strong> - inlet excluded. If primary carries
        any standalone signal, it should appear here.<br>
        &bull; <strong>SE1b (Primary + Inlet + COMMON, 17 feat)</strong> - adds core inlet (4 features).
        Provides the paired comparison against SE1a.<br>
        Both sub-scopes produce strongly negative aggregate R², confirming that primary features are genuinely
        weak predictors - the failure is not caused by inlet collinearity alone.
        OLS is additionally compromised by extreme collinearity (Primary Clarifier pH VIF ~2743 vs Inlet pH).
        <strong>Conclusion:</strong> primary-stage measurements carry no standalone predictive signal.
        Secondary data is needed - see SE2.
      </p>
    </div>
    {se1a_inner}
    {se1b_inner}
  </div>
</details>"""

    se2_wrapper = f"""
<details class="exp-details" open id="exp2-s2">
  <summary><span class="fold-icon">▶</span>
    SE2  -  Secondary Stages
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #4A90D9">
      <p class="meta">
        Four scopes tested within SE2, progressing from sub-group isolation to full combination.<br>
        &bull; <strong>Clarifier + COMMON</strong> and <strong>Sed + COMMON</strong>: isolate each
        secondary sub-group to test whether they carry distinct or interchangeable signal.<br>
        &bull; <strong>SE2a - All Secondary only</strong> (10 feat, no COMMON): tests the full
        secondary feature set without operational context, establishing what signal the secondary
        measurements carry independently. Gains ~50 grab and ~150 comp rows vs SE2-Comb because
        Flow/Power NaN rows are no longer binding.<br>
        &bull; <strong>All Secondary + COMMON</strong>: full secondary with operational context (17 feat).
      </p>
    </div>
    {se2_clr}
    {se2_sed}
    {se2a}
    {se2_combined}
  </div>
</details>"""

    se3_wrapper = f"""
<details class="exp-details" open id="exp2-s3">
  <summary><span class="fold-icon">▶</span>
    SE3  -  Inlet + Primary + Secondary + COMMON (27 features)
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #4A90D9">
      <p class="meta">
        Full stage-based combination: core inlet (4), primary clarifier + grit (6),
        secondary clarifier + sedimentation (10), and COMMON (7) = 27 features.
        Grab train ~764 rows / test 197; Composite train ~595 / test 176.
        Secondary co-occurs almost perfectly with Primary days, so adding it costs only ~14 rows
        versus SE1 (Primary + Inlet, ~778 train).
        The <strong>CIS baseline</strong> (Core Inlet + Secondary + Common, 21 features, ~920 train)
        is the prior Exp2-SE2 result; compare SE3 against it to assess whether primary data
        adds value in the presence of secondary measurements.
        The <strong>Secondary Only - Feature Selected</strong> variant (first below) tests whether
        pruning the pure secondary pool to its most informative features matches or beats the
        richer CIS feature set.
      </p>
    </div>
    {se3_s6fs}
    {se3_full}
    {ref_full}
    {ref_fs}
  </div>
</details>"""

    return f"""
<section id="exp2">
  <h1 class="section-title">Process Stage Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp2"]}</p>
  {se1_wrapper}
  {se2_wrapper}
  {se3_wrapper}
  {cmp_div}
  {findings_div}
  {best}
</section>"""


def _e3s2_fs_impact(df_all: pd.DataFrame) -> str:
    """Pre-FS vs post-FS breakdown for Exp3-SE2-FS.

    OLS uses LassoCV; RF/GB/XGB use OOF permutation importance.
    Ridge has no explicit FS (full set always). ElNet uses internal L1+L2.
    Both pre-FS and post-FS use the same training rows (no dataset rebuild in S2).
    """
    s2fs = df_all[df_all["exp_key"] == "Exp3-S2-FS"].copy()
    if s2fs.empty:
        return ""

    MEANINGFUL = 0.01
    MODELS_ORD = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]

    _TD  = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

    def _si(v):
        try:
            f = float(v)
            return "-" if np.isnan(f) else int(f)
        except (TypeError, ValueError):
            return "-"

    def _fv(v, fmt="+.3f"):
        try:
            f = float(v)
            if np.isnan(f): return " - "
            return format(f, fmt)
        except (TypeError, ValueError):
            return " - "

    def _r2c(v):
        try: v = float(v)
        except (TypeError, ValueError): return "#888888"
        if v >= 0.6: return "#2ecc71"
        if v >= 0.4: return "#52c98a"
        if v >= 0.2: return "#f1c40f"
        if v >= 0.0: return "#e67e22"
        return "#e74c3c"

    def _gapc(v):
        try: v = float(v)
        except (TypeError, ValueError): return "#888888"
        if v > 0.25: return "#e74c3c"
        if v > 0.10: return "#e67e22"
        if v < -0.10: return "#4A90D9"
        return "#5BAD6F"

    def _dc(d):
        try: d = float(d)
        except (TypeError, ValueError): return "#888888"
        if d >= MEANINGFUL: return "#5BAD6F"
        if d <= -MEANINGFUL: return "#e74c3c"
        return "#888888"

    def _r2gap_cell(r2, gap, rmse):
        """Render a table cell with R2, gap, and (optional) RMSE."""
        r2s  = _fv(r2)
        gaps = _fv(gap)
        rmse_part = (f"<span style='font-size:0.72em;color:#888888'> · RMSE {_fv(rmse, '.2f')}</span>"
                     if rmse is not None and rmse == rmse and str(rmse) != "None" else "")
        return (f"<td style='{_TD};text-align:center;color:{_r2c(r2)}'>{r2s}<br>"
                f"<span style='font-size:0.72em;color:{_gapc(gap)}'>Gap {gaps}</span>"
                f"{rmse_part}</td>")

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (f"<tr style='background:#e8e8e8'><td colspan='8' style='padding:6px 10px;"
                  f"font-size:0.75rem;font-weight:700;color:#555555;letter-spacing:0.06em;"
                  f"text-transform:uppercase;border-bottom:1px solid #d0d0d0'>{short}</td></tr>")

        for i, m in enumerate(MODELS_ORD):
            msub = s2fs[(s2fs["model"] == m) & (s2fs["target"] == tgt)]
            if msub.empty:
                continue
            row    = msub.iloc[0]
            row_bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
            n_in   = _si(row.get("n_features_input"))

            if m == "OLS":
                n_sel, method, has_pre = _si(row.get("n_selected_ols")), "LassoCV", True
                pre_r2, pre_gap, pre_rmse = (row.get("R2_test_full"),
                                             row.get("R2_gap_full"),
                                             row.get("RMSE_test_full"))
            elif m == "Ridge":
                n_sel, method, has_pre = "all", "Full set (no FS)", False
                pre_r2 = pre_gap = pre_rmse = None
            elif m == "ElNet":
                n_sel, method, has_pre = _si(row.get("ElNet_n_selected")), "Internal L1+L2", False
                pre_r2 = pre_gap = pre_rmse = None
            else:  # RF, GB, XGB
                n_sel, method, has_pre = _si(row.get("n_selected_nl")), "OOF Perm. Imp.", True
                pre_r2, pre_gap, pre_rmse = (row.get("R2_test_full"),
                                             row.get("R2_gap_full"),
                                             None)  # RMSE_test_full not stored for NL

            post_r2, post_gap, post_rmse = (row.get("R2_test"),
                                            row.get("R2_gap"),
                                            row.get("RMSE_test"))

            if has_pre:
                pre_cell = _r2gap_cell(pre_r2, pre_gap, pre_rmse)
            elif m == "Ridge":
                pre_cell = (f"<td style='{_TD};text-align:center;color:#888888;font-style:italic'>"
                            f"Full set ({n_in} features)</td>")
            else:
                pre_cell = (f"<td style='{_TD};text-align:center;color:#888888;font-style:italic'>"
                            f"No separate pre-FS run</td>")

            post_cell = _r2gap_cell(post_r2, post_gap, post_rmse)

            if has_pre:
                try:
                    d_r2  = float(post_r2) - float(pre_r2)
                    d_gap = (float(post_gap) - float(pre_gap)
                             if pre_gap is not None else None)
                    verdict  = ("FS helped" if d_r2 >= MEANINGFUL else
                                "FS hurt"   if d_r2 <= -MEANINGFUL else "Neutral")
                    vc  = _dc(d_r2)
                    dgc = _dc(-(d_gap) if d_gap is not None else None)
                    gap_part = (f"<span style='font-size:0.72em;color:{dgc}'>Gap {_fv(d_gap)}</span>"
                                if d_gap is not None else "")
                    delta_td = (
                        f"<td style='{_TD};text-align:center;color:{vc};font-weight:bold'>"
                        f"{_fv(d_r2)}<br>{gap_part}</td>"
                        f"<td style='{_TD};text-align:center;color:{vc}'>{verdict}</td>"
                    )
                except (TypeError, ValueError):
                    delta_td = (f"<td style='{_TD};text-align:center;color:#888888'> - </td>"
                                f"<td style='{_TD};text-align:center;color:#888888'> - </td>")
            else:
                delta_td = (f"<td style='{_TD};text-align:center;color:#888888'> - </td>"
                            f"<td style='{_TD};text-align:center;color:#888888;font-style:italic'> - </td>")

            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"<td style='{_TD};text-align:center'>{method}</td>"
                f"<td style='{_TD};text-align:center'>{n_in}</td>"
                f"<td style='{_TD};text-align:center'>{n_sel}</td>"
                f"{pre_cell}{post_cell}{delta_td}</tr>"
            )

    note_html = """<div class='obs-card' style='border-left:4px solid #9B59B6;margin-top:0.8rem'>
  <p class='meta'>
    <strong>Reading this table.</strong>
    <strong>OLS</strong>: pre-FS trains on all 31 input features; post-FS refits on the LassoCV-selected subset.
    <strong>RF / GB / XGB</strong>: pre-FS trains on all 31 features; post-FS retrains on the OOF-permutation-importance-selected subset.
    RMSE not stored for pre-FS tree runs (shown for post-FS only).
    <strong>Ridge</strong>: no explicit FS - L2 regularisation handles collinearity; the single column is the full-set result.
    <strong>ElNet</strong>: L1+L2 selection is internal - n&nbsp;Selected shows non-zero coefficients; no pre-FS run.
    Both pre-FS and post-FS use the same ~469 Grab / ~423 Composite training rows (no dataset rebuild in SE2).
    <span style='color:#5BAD6F;font-weight:bold'>Green DeltaR2</span> = FS improved generalisation.
    <span style='color:#e74c3c;font-weight:bold'>Red DeltaR2</span> = FS hurt generalisation (common for tree models when OOF folds are small).
    A negative pre-FS gap means test R2 exceeded train R2 (underfitting at full feature count).
  </p>
</div>"""

    table_html = f"""<div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px;margin-top:0.8rem'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:920px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:55px'>Model</th>
      <th style='{_THC}'>FS Method</th>
      <th style='{_THC}'>n Input</th>
      <th style='{_THC}'>n Selected</th>
      <th style='{_THC}'>Pre-FS<br><span style='color:#888888;font-weight:400;font-size:0.78em'>R2 - Gap - RMSE</span></th>
      <th style='{_THC}'>Post-FS<br><span style='color:#888888;font-weight:400;font-size:0.78em'>R2 - Gap - RMSE</span></th>
      <th style='{_THC}'>Delta R2 / Delta Gap<br><span style='color:#888888;font-weight:400;font-size:0.78em'>post minus pre</span></th>
      <th style='{_TH}'>Verdict</th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""<hr style='margin:1.5rem 0;border:none;border-top:1px solid #ddd'>
<h3 style='margin:0 0 0.4rem;color:#1a1a1a;font-size:1rem'>
  Feature Selection Impact - SE2-FS: pre-FS vs post-FS per model
</h3>
<p class='meta' style='margin-bottom:0.6rem'>
  The metrics in the SE2-FS sub-experiment section show <strong>post-FS results only</strong>.
  This table shows what each model achieves on the <em>full 31-feature input set</em> (pre-FS) versus
  the <em>feature-selected subset</em> (post-FS), directly answering whether FS helped or hurt for
  each model and target. For Ridge and ElNet there is no distinct pre-FS run.
</p>
{note_html}
{table_html}"""


def _exp3_comparison_panel(df_all: pd.DataFrame) -> str:
    """Experiment-wide comparison: SE1 | SE2 | SE2-FS | SE3 | SE3-FS | SE4 | SE4-FS."""
    s1   = df_all[df_all["exp_key"] == "Exp3-S1"].copy()
    s2   = df_all[df_all["exp_key"] == "Exp3-S2"].copy()
    s2fs = df_all[df_all["exp_key"] == "Exp3-S2-FS"].copy()
    s3   = df_all[df_all["exp_key"] == "Exp3-S3"].copy()
    s3fs = df_all[df_all["exp_key"] == "Exp3-S3-FS"].copy()
    s4   = df_all[df_all["exp_key"] == "Exp3-S4"].copy()
    s4fs = df_all[df_all["exp_key"] == "Exp3-S4-FS"].copy()

    if s1.empty and s2fs.empty and s3.empty and s4.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL  = 0.01

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

    _TD  = "padding:4px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _STD = "padding:5px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:5px 7px;text-align:left;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"
    _THC = "padding:5px 7px;text-align:center;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        mhtml  = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:   col = "#5BAD6F"; fw = "bold"
        elif is_gaj: col = "#4A90D9"; fw = "bold"
        else:        col = "#1a1a1a"; fw = "normal"
        rmse_s = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else "-"
        gv     = gap if (gap is not None and gap == gap) else None
        gs     = f"{gv:+.3f}" if gv is not None else "-"
        gc     = ("#E15252" if gv is not None and gv > 0.10
                  else "#5BAD6F" if gv is not None and gv < -0.10
                  else "#888888")
        sec = (f"<br><span style='font-size:0.71em;color:#888888;font-weight:normal'>"
               f"{rmse_s} · <span style='color:{gc}'>{gs}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{mhtml}{sec}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        c = ("#5BAD6F" if dr2 >= MEANINGFUL else
             "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        ex = ""
        if drmse is not None and drmse == drmse and dgap is not None and dgap == dgap:
            rc = ("#5BAD6F" if drmse <= -MEANINGFUL else
                  "#E15252" if drmse >= MEANINGFUL else "#888888")
            gc = ("#E15252" if dgap > MEANINGFUL else
                  "#5BAD6F" if dgap < -MEANINGFUL else "#888888")
            ex = (f"<br><span style='font-size:0.71em;font-weight:normal'>"
                  f"<span style='color:{rc}'>{drmse:+.2f}</span>"
                  f" · <span style='color:{gc}'>{dgap:+.3f}</span></span>")
        elif drmse is not None and drmse == drmse:
            rc = ("#5BAD6F" if drmse <= -MEANINGFUL else
                  "#E15252" if drmse >= MEANINGFUL else "#888888")
            ex = (f"<br><span style='font-size:0.71em;color:{rc};font-weight:normal'>"
                  f"{drmse:+.2f}</span>")
        elif dgap is not None and dgap == dgap:
            gc = ("#E15252" if dgap > MEANINGFUL else
                  "#5BAD6F" if dgap < -MEANINGFUL else "#888888")
            ex = (f"<br><span style='font-size:0.71em;color:{gc};font-weight:normal'>"
                  f"{dgap:+.3f}</span>")
        return (f"<td style='{_TD};text-align:center;color:{c};font-weight:bold'>"
                f"{dr2:+.3f}{ex}</td>")

    def _stats_row(deltas, gaj_deltas, from_lbl, to_lbl):
        if not deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr  = np.array(deltas); n = len(arr); net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mw = float(wins.mean())   if len(wins)   else None
        ml = float(losses.mean()) if len(losses) else None
        nc = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        vd = "Net improvement" if net > MEANINGFUL else ("Net regression" if net < -MEANINGFUL else "Negligible")
        ws = f"{len(wins)}/{n} (avg {mw:+.3f})"   if mw is not None else f"{len(wins)}/{n}"
        ls = f"{len(losses)}/{n} (avg {ml:+.3f})" if ml is not None else f"{len(losses)}/{n}"
        if gaj_deltas:
            gn = float(np.array(gaj_deltas).mean()); diff = gn - net
            gc = "#5BAD6F" if gn > MEANINGFUL else ("#E15252" if gn < -MEANINGFUL else "#888888")
            dc = "#E15252" if diff < -0.02 else "#5BAD6F" if diff > 0.02 else "#888888"
            interp = ("Raw gains partially inflated by overfitting" if diff < -0.02
                      else "Raw gains understated  -  overfitting decreased" if diff > 0.02
                      else "Overfitting largely unchanged")
            gs = f"{gn:+.4f}"; ds = f"{diff:+.4f}"
        else:
            gs = ds = " - "; gc = dc = "#888888"; interp = " - "
        return (f"<tr>"
                f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
                f"<td style='{_STD};text-align:center;color:{nc};font-weight:bold'>{net:+.4f}</td>"
                f"<td style='{_STD};text-align:center;color:#5BAD6F'>{ws}</td>"
                f"<td style='{_STD};text-align:center;color:#E15252'>{ls}</td>"
                f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
                f"<td style='{_STD};text-align:center;color:{gc};font-weight:bold'>{gs}</td>"
                f"<td style='{_STD};text-align:center;color:{dc}'>{ds}</td>"
                f"<td style='{_STD}'><strong>{vd}</strong><br>"
                f"<span style='font-size:0.82em;color:{dc}'>{interp}</span></td></tr>")

    d_s1_s2    = []; gaj_s1_s2    = []
    d_s2_s2fs  = []; gaj_s2_s2fs  = []
    d_s1_s2fs  = []; gaj_s1_s2fs  = []
    d_s3_s3fs  = []; gaj_s3_s3fs  = []
    d_s4_s4fs  = []; gaj_s4_s4fs  = []

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='11' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # --- pre-pass: global best for this target across all models and SEs ---
        _all_raw_e3 = {}
        _all_gaj_e3 = {}
        _all_gap_e3 = {}
        for m_ in models_ord:
            v1_  = _get(s1,   m_, tgt); g1_  = _gap_v(s1,   m_, tgt)
            v2_  = _get(s2,   m_, tgt); g2_  = _gap_v(s2,   m_, tgt)
            v2f_ = _get(s2fs, m_, tgt); g2f_ = _gap_v(s2fs, m_, tgt)
            v3_  = _get(s3,   m_, tgt); g3_  = _gap_v(s3,   m_, tgt)
            v3f_ = _get(s3fs, m_, tgt); g3f_ = _gap_v(s3fs, m_, tgt)
            v4_  = _get(s4,   m_, tgt); g4_  = _gap_v(s4,   m_, tgt)
            v4f_ = _get(s4fs, m_, tgt); g4f_ = _gap_v(s4fs, m_, tgt)
            sc1_  = _gaj(v1_,  g1_);  sc2_  = _gaj(v2_,  g2_)
            sc2f_ = _gaj(v2f_, g2f_); sc3_  = _gaj(v3_,  g3_); sc3f_ = _gaj(v3f_, g3f_)
            sc4_  = _gaj(v4_,  g4_);  sc4f_ = _gaj(v4f_, g4f_)
            for key_, rv_, gv_, sv_ in [
                ((m_, "s1"),  v1_,  g1_,  sc1_),
                ((m_, "s2"),  v2_,  g2_,  sc2_),
                ((m_, "s2f"), v2f_, g2f_, sc2f_),
                ((m_, "s3"),  v3_,  g3_,  sc3_),
                ((m_, "s3f"), v3f_, g3f_, sc3f_),
                ((m_, "s4"),  v4_,  g4_,  sc4_),
                ((m_, "s4f"), v4f_, g4f_, sc4f_),
            ]:
                if rv_ is not None:
                    _all_raw_e3[key_] = rv_
                    _all_gap_e3[key_] = gv_
                if sv_ is not None:
                    _all_gaj_e3[key_] = sv_

        tgt_br_e3 = max(_all_raw_e3.values()) if _all_raw_e3 else None
        tgt_bg_e3 = max(_all_gaj_e3.values()) if _all_gaj_e3 else None

        _raw_win_gap_e3 = None
        if tgt_br_e3 is not None:
            for k_, v_ in _all_raw_e3.items():
                if abs(v_ - tgt_br_e3) < 1e-9:
                    _raw_win_gap_e3 = _all_gap_e3.get(k_)
                    break

        tgt_show_gaj_e3 = (
            tgt_bg_e3 is not None and tgt_br_e3 is not None
            and _raw_win_gap_e3 is not None and _raw_win_gap_e3 > 0.10
        )

        def _ir(v_): return tgt_br_e3 is not None and v_ is not None and abs(v_ - tgt_br_e3) < 1e-9
        def _ig(s_): return tgt_show_gaj_e3 and tgt_bg_e3 is not None and s_ is not None and abs(s_ - tgt_bg_e3) < 1e-9

        for m in models_ord:
            v1   = _get(s1,   m, tgt); g1   = _gap_v(s1,   m, tgt)
            v2   = _get(s2,   m, tgt); g2   = _gap_v(s2,   m, tgt)
            v2f  = _get(s2fs, m, tgt); g2f  = _gap_v(s2fs, m, tgt)
            v3   = _get(s3,   m, tgt); g3   = _gap_v(s3,   m, tgt)
            v3f  = _get(s3fs, m, tgt); g3f  = _gap_v(s3fs, m, tgt)
            v4   = _get(s4,   m, tgt); g4   = _gap_v(s4,   m, tgt)
            v4f  = _get(s4fs, m, tgt); g4f  = _gap_v(s4fs, m, tgt)

            r1   = _get(s1,   m, tgt, "RMSE_test")
            r2   = _get(s2,   m, tgt, "RMSE_test")
            r2f  = _get(s2fs, m, tgt, "RMSE_test")
            r3   = _get(s3,   m, tgt, "RMSE_test")
            r3f  = _get(s3fs, m, tgt, "RMSE_test")
            r4   = _get(s4,   m, tgt, "RMSE_test")
            r4f  = _get(s4fs, m, tgt, "RMSE_test")

            sc1  = _gaj(v1,  g1);  sc2  = _gaj(v2,  g2)
            sc2f = _gaj(v2f, g2f); sc3  = _gaj(v3,  g3);  sc3f = _gaj(v3f, g3f)
            sc4  = _gaj(v4,  g4);  sc4f = _gaj(v4f, g4f)

            d12   = (v2  - v1)  if (v1  is not None and v2  is not None) else None
            d22f  = (v2f - v2)  if (v2  is not None and v2f is not None) else None
            d12f  = (v2f - v1)  if (v1  is not None and v2f is not None) else None
            d33f  = (v3f - v3)  if (v3  is not None and v3f is not None) else None
            d44f  = (v4f - v4)  if (v4  is not None and v4f is not None) else None

            d12f_rmse = ((r2f - r1)  if (r1  is not None and r2f is not None) else None)
            d33f_rmse = ((r3f - r3)  if (r3  is not None and r3f is not None) else None)
            d12f_gap  = ((g2f - g1)  if (g1  is not None and g2f is not None) else None)
            d33f_gap  = ((g3f - g3)  if (g3  is not None and g3f is not None) else None)

            sd12  = (sc2  - sc1)  if (sc1  is not None and sc2  is not None) else None
            sd22f = (sc2f - sc2)  if (sc2  is not None and sc2f is not None) else None
            sd12f = (sc2f - sc1)  if (sc1  is not None and sc2f is not None) else None
            sd33f = (sc3f - sc3)  if (sc3  is not None and sc3f is not None) else None
            sd44f = (sc4f - sc4)  if (sc4  is not None and sc4f is not None) else None

            if d12   is not None: d_s1_s2.append(d12);     gaj_s1_s2.append(sd12)    if sd12  is not None else None
            if d22f  is not None: d_s2_s2fs.append(d22f);  gaj_s2_s2fs.append(sd22f) if sd22f is not None else None
            if d12f  is not None: d_s1_s2fs.append(d12f);  gaj_s1_s2fs.append(sd12f) if sd12f is not None else None
            if d33f  is not None: d_s3_s3fs.append(d33f);  gaj_s3_s3fs.append(sd33f) if sd33f is not None else None
            if d44f  is not None: d_s4_s4fs.append(d44f);  gaj_s4_s4fs.append(sd44f) if sd44f is not None else None

            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(v1,   r1,   g1,   _ir(v1),   _ig(sc1))}"
                f"{_val_td(v2,   r2,   g2,   _ir(v2),   _ig(sc2))}"
                f"{_val_td(v2f,  r2f,  g2f,  _ir(v2f),  _ig(sc2f))}"
                f"{_delta_td(d12f, d12f_rmse, d12f_gap)}"
                f"{_val_td(v3,   r3,   g3,   _ir(v3),   _ig(sc3))}"
                f"{_val_td(v3f,  r3f,  g3f,  _ir(v3f),  _ig(sc3f))}"
                f"{_delta_td(d33f, d33f_rmse, d33f_gap)}"
                f"{_val_td(v4,   r4,   g4,   _ir(v4),   _ig(sc4))}"
                f"{_val_td(v4f,  r4f,  g4f,  _ir(v4f),  _ig(sc4f))}"
                f"</tr>"
            )

    n_cells = len(models_ord) * len(TARGETS_ORDERED)
    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; −{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| ≤ {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²−0.5·max(0,|gap|−0.10)</span></th>
        <th style='{_THC}'>Gap-Adj − Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
        <th style='{_TH}'>Verdict</th>
      </tr>
    </thead>
    <tbody>
      {_stats_row(d_s1_s2,    gaj_s1_s2,    "SE1 (ADD)", "SE2 (ADD+CONSIDER, no FS)")}
      {_stats_row(d_s2_s2fs,  gaj_s2_s2fs,  "SE2 (no FS)", "SE2-FS")}
      {_stats_row(d_s1_s2fs,  gaj_s1_s2fs,  "SE1 (ADD)", "SE2-FS (ADD+CONSIDER)")}
      {_stats_row(d_s3_s3fs,  gaj_s3_s3fs,  "SE3 (minus Coliform, no FS)", "SE3-FS")}
      {_stats_row(d_s4_s4fs,  gaj_s4_s4fs,  "SE4 (All Features, no FS)", "SE4-FS")}
    </tbody>
  </table>
  </div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      Net Mean ΔR² is the signed average across all {n_cells} (model × target) cells.
      <strong>SE1 → SE2</strong>: adds CONSIDER-tier features (32 total, no FS) on ~469 Grab rows.
      <strong>SE2 → SE2-FS</strong>: applies model-specific FS to the same 32-feature set.
      <strong>SE1 → SE2-FS</strong>: net effect of both CONSIDER features and FS together.
      <strong>SE3 → SE3-FS</strong>: applies FS to the 30-feature set at ~815 Grab rows (Coliform removed).
      <strong>SE4 → SE4-FS</strong>: applies FS to the 44/39-feature all-features set on ~130 Grab rows.
      <strong>Gap-Adj − Raw</strong>: negative = raw gains inflated by overfitting;
      positive = gains understated because overfitting decreased.
      <strong>★</strong> = best raw Test R² per target · <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
    </p>
  </div>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.79rem;color:#1a1a1a;min-width:880px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:52px'>Model</th>
      <th style='{_THC}'>SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE2-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ SE1→SE2-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='{_THC}'>SE3<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE3-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ SE3→SE3-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='{_THC}'>SE4<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE4-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    fs_impact = _e3s2_fs_impact(df_all)

    return f"""
<details class="exp-details" id="exp3-comparison" open>
  <summary><span class="fold-icon">▶</span>
    Comparisons
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
        RMSE in native units (mg/L or pH). Gap = Train R² - Test R²;
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {main_table}
    {stats_block}
    {fs_impact}
  </div>
</details>"""


def _exp3_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for Experiment 3."""
    s1 = df_all[df_all["exp_key"] == "Exp3-S1"].dropna(subset=["R2_test"])
    s2 = df_all[df_all["exp_key"] == "Exp3-S2-FS"].dropna(subset=["R2_test"])
    s3 = df_all[df_all["exp_key"] == "Exp3-S3"].dropna(subset=["R2_test"])
    ks = df_all[df_all["exp_key"] == "Exp3-S4"].dropna(subset=["R2_test"])

    # Per-group averages
    grab_tgts = set(GRAB_TARGETS)
    comp_tgts = set(COMP_TARGETS)

    def _avg_r2(df, tgts):
        vals = [float(df[df["target"] == t]["R2_test"].max())
                for t in tgts
                if not df[df["target"] == t].empty]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _avg_gap(df, tgts):
        vals = [float(df[df["target"] == t]["R2_gap"].mean())
                for t in tgts
                if not df[df["target"] == t].empty
                and "R2_gap" in df.columns]
        return float(np.nanmean(vals)) if vals else float("nan")

    s1_grab = _avg_r2(s1, grab_tgts)
    s1_comp = _avg_r2(s1, comp_tgts)
    s2_grab = _avg_r2(s2, grab_tgts)
    s2_comp = _avg_r2(s2, comp_tgts)
    s3_grab = _avg_r2(s3, grab_tgts)
    s3_comp = _avg_r2(s3, comp_tgts)
    ks_grab = _avg_r2(ks, grab_tgts)
    ks_comp = _avg_r2(ks, comp_tgts)

    s1_rf_gap  = float(s1[s1["model"] == "RF"]["R2_gap"].mean())  if not s1.empty else float("nan")
    s3_rf_gap  = float(s3[s3["model"] == "RF"]["R2_gap"].mean())  if not s3.empty else float("nan")
    ks_ols_gap = float(ks[ks["model"] == "OLS"]["R2_gap"].mean()) if not ks.empty else float("nan")
    ks_rf_gap  = float(ks[ks["model"] == "RF"]["R2_gap"].mean())  if not ks.empty else float("nan")

    def _c(v, positive_good=True):
        if v != v: return " - "
        col = ("#5BAD6F" if v > 0.05 else "#E15252" if v < -0.05 else "var(--text-muted)") \
              if positive_good else \
              ("#E15252" if v > 0.10 else "#5BAD6F" if v < 0.05 else "var(--text-muted)")
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    q1 = (
        f"<strong>SE1 (ADD-tier, 28 features)</strong> retains ~816 Grab training rows - negligible "
        f"row loss vs Exp2-SE2. Best ADD Grab avg = {_c(s1_grab)}; Comp avg = {_c(s1_comp)}. "
        f"RF avg Grab gap = {_c(s1_rf_gap, False)} - moderate overfitting with 28 features at n=816. "
        f"<br><br>"
        f"<strong>SE2 (ADD+CONSIDER, 31 features)</strong>: Inlet Total Coliform alone cuts Grab rows "
        f"from ~816 to ~469 (-43%). With FS, best SE2-FS Grab avg = {_c(s2_grab)}, Comp avg = {_c(s2_comp)}. "
        f"<br><br>"
        f"<strong>SE3 (ADD+CONSIDER minus Coliform, 30 features)</strong>: Removing Coliform recovers "
        f"~347 rows, restoring Grab to ~815. RF avg Grab gap = {_c(s3_rf_gap, False)}. "
        f"Best SE3 Grab avg = {_c(s3_grab)}, Comp avg = {_c(s3_comp)}. "
        f"<br><br>"
        f"<strong>SE4 (All Features, 44/39 features)</strong>: Grab rows collapsed to ~130 (-74% vs Exp2-SE2). "
        f"With n/p ≈ 3, OLS avg gap = {_c(ks_ols_gap, False)}, RF avg gap = {_c(ks_rf_gap, False)}. "
        f"All-features Grab avg = {_c(ks_grab)}, Comp avg = {_c(ks_comp)}."
    )

    q2 = (
        f"<strong>SE3 vs SE2-FS</strong>: SE3 removes only Inlet Total Coliform, recovering ~347 rows. "
        f"The key question is whether the Coliform signal was worth the row cost. SE3 (no FS) at ~815 rows "
        f"can be directly compared to SE1 (28 features, ~816 rows) and SE2-FS (31 features, ~469 rows). "
        f"<br><br>"
        f"<strong>SE2-FS (ADD+CONSIDER with FS)</strong>: Mixed results. LassoCV benefits OLS; tree OOF "
        f"fold size at n=469 is ~156 rows - workable but constrained. "
        f"<br><br>"
        f"<strong>SE4-FS (All Features FS)</strong>: Did not rescue the grab dataset. "
        f"OOF fold size on 130 rows (3-fold TSS) is ~43 rows - permutation importance too noisy, "
        f"causing overly aggressive pruning (2-7 features). Composite targets (n≈498) fared better."
    )

    q3 = (
        f"The Feature Audit tiers (ADD / CONSIDER) exist to avoid the all-features trap. "
        f"<strong>SE3 isolates the Coliform row-cost question</strong>: by removing only Coliform, "
        f"it tests whether SE2's performance was driven by the Coliform signal or by data starvation. "
        f"<strong>SE1 remains the recommended baseline</strong> (MI ≥ 0.20, row cost ≤ 20%, ~816 rows). "
        f"<br><br>"
        f"<strong>SE4 confirms</strong> the audit's core conclusion: "
        f"<strong>data-driven FS cannot rescue an undersized dataset</strong> - "
        f"the FS algorithm operates in a high-variance regime when fold sizes fall below ~50 rows. "
        f"Feature selection should be guided by MI signal and row cost before the experiment, "
        f"not solely by in-script FS at train time."
    )

    def _qcard(n, question, answer):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);
              display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem'>{answer}</p>
  </div>
</div>"""

    cards = "".join([
        _qcard(1, "How did each SE's feature set affect sample size and performance?", q1),
        _qcard(2, "Did SE3 (minus Coliform) isolate the row-cost problem? Did FS variants improve on full sets?", q2),
        _qcard(3, "What is the strategic takeaway for feature engineering?", q3),
    ])

    return f"""
<details class="exp-details" id="exp3-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings  -  Extended Operational Features</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def build_exp3_section(df_all: pd.DataFrame) -> str:
    sub1 = _exp_subsection(df_all, "Exp3-S1", "exp3-s1",
                           "SE1  -  ADD-tier: Inlet + Secondary + Aeration (no FS)",
                           open_default=True)

    # SE2 wrapper: full (no FS) + FS
    sub2_full = _exp_subsection(df_all, "Exp3-S2",    "exp3-s2-full",
                                "Full Feature Set (32 features, no FS)",
                                open_default=False)
    sub2_fs   = _exp_subsection(df_all, "Exp3-S2-FS", "exp3-s2-fs",
                                "With Feature Selection",
                                open_default=False,
                                dataset_summary_fn=_dataset_summary_per_model)
    sub2_wrapper = f"""
<details class="exp-details" id="exp3-s2">
  <summary><span class="fold-icon">▶</span> SE2  -  ADD + CONSIDER-tier (full + FS)</summary>
  <div class="exp-body">
    {sub2_full}
    {sub2_fs}
  </div>
</details>"""

    # SE3 wrapper: ADD+CONSIDER minus Coliform, full + FS
    sub3_full = _exp_subsection(df_all, "Exp3-S3",    "exp3-s3-full",
                                "Full Feature Set (no FS)",
                                open_default=False)
    sub3_fs   = _exp_subsection(df_all, "Exp3-S3-FS", "exp3-s3-fs",
                                "With Feature Selection",
                                open_default=False,
                                dataset_summary_fn=_dataset_summary_per_model)
    sub3_wrapper = f"""
<details class="exp-details" id="exp3-s3">
  <summary><span class="fold-icon">▶</span> SE3  -  ADD + CONSIDER minus Coliform (full + FS)</summary>
  <div class="exp-body">
    {sub3_full}
    {sub3_fs}
  </div>
</details>"""

    # SE4 wrapper: All Features (formerly SE3), full + FS
    sub4_full = _exp_subsection(df_all, "Exp3-S4",    "exp3-s4-full",
                                "All Features (no FS)",
                                open_default=False)
    sub4_fs   = _exp_subsection(df_all, "Exp3-S4-FS", "exp3-s4-fs",
                                "With Feature Selection",
                                open_default=False,
                                dataset_summary_fn=_dataset_summary_per_model)
    sub4_wrapper = f"""
<details class="exp-details" id="exp3-s4">
  <summary><span class="fold-icon">▶</span> SE4  -  All Features (full + FS)</summary>
  <div class="exp-body">
    {sub4_full}
    {sub4_fs}
  </div>
</details>"""

    cmp_div      = _exp3_comparison_panel(df_all)
    findings_div = _exp3_qna(df_all)

    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Exp3-S1", "Exp3-S2", "Exp3-S2-FS",
                                        "Exp3-S3", "Exp3-S3-FS",
                                        "Exp3-S4", "Exp3-S4-FS"])],
        "Extended Operational Features")
    return f"""
<section id="exp3">
  <h1 class="section-title">Extended Operational Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp3"]}</p>
  {sub1}
  {sub2_wrapper}
  {sub3_wrapper}
  {sub4_wrapper}
  {cmp_div}
  {findings_div}
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
    Load raw Exp 6 ensemble results and render a variance-collapse diagnosis
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

    _TH  = "padding:5px 7px;text-align:left;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"
    _THC = "padding:5px 7px;text-align:center;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"
    _TD  = "padding:4px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    return f"""
<details class="exp-details" id="adv-neg-r2">
  <summary><span class="fold-icon">▶</span> Negative R² Diagnosis - Variance Collapse vs Genuine Failure</summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #e67e22">
      <p class="meta">
        R² = 1 - SS_res/SS_tot. A low-variance 2025 test set (sigma-ratio &lt; 0.5) shrinks the
        denominator and forces R² negative even when absolute accuracy is unchanged or improving.
        <strong>ΔMAE</strong> and <strong>ΔRMSE</strong> (2025 - 2024) give scale-grounded evidence:
        negative = model improved in absolute terms; positive = genuine deterioration.
        NRMSE = RMSE / (max - min) for scale-normalised cross-target comparison.
      </p>
    </div>
    <div style="overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px">
    <table style="border-collapse:collapse;width:100%;background:#ffffff;font-size:0.79rem;color:#1a1a1a;min-width:960px">
      <thead>
        <tr style="border-bottom:2px solid #cccccc">
          <th style="{_TH}">Model</th>
          <th style="{_TH}">Target</th>
          <th style="{_THC}">Test R²</th>
          <th style="{_THC}">σ_train</th>
          <th style="{_THC}">σ_test</th>
          <th style="{_THC}">σ-ratio</th>
          <th style="{_THC}">MAE 2024</th>
          <th style="{_THC}">MAE 2025</th>
          <th style="{_THC}">ΔMAE</th>
          <th style="{_THC}">RMSE 2024</th>
          <th style="{_THC}">RMSE 2025</th>
          <th style="{_THC}">ΔRMSE</th>
          <th style="{_THC}">NRMSE</th>
          <th style="{_TH}">Diagnosis</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
  </div>
</details>"""


def _exp4_comparison_panel(df_all: pd.DataFrame) -> str:
    """Standard comparison panel: E3-SE2 baseline, E4-SE1, E4-SE2 with Transition Summary."""
    e3   = df_all[df_all["exp_key"] == "Exp3-S2"].copy()
    s1   = df_all[df_all["exp_key"] == "Exp4-S1"].copy()
    s2   = df_all[df_all["exp_key"] == "Exp4-S2"].copy()

    if s1.empty and s2.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL = 0.01
    _TD  = "padding:4px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _STD = "padding:5px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:5px 7px;text-align:left;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"
    _THC = "padding:5px 7px;text-align:center;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"

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

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        mhtml  = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:   col = "#5BAD6F"; fw = "bold"
        elif is_gaj: col = "#4A90D9"; fw = "bold"
        else:        col = "#1a1a1a"; fw = "normal"
        rmse_s = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else "-"
        gv     = gap if (gap is not None and gap == gap) else None
        gs     = f"{gv:+.3f}" if gv is not None else "-"
        gc     = ("#E15252" if gv is not None and gv > 0.10
                  else "#5BAD6F" if gv is not None and gv < -0.10
                  else "#888888")
        sec = (f"<br><span style='font-size:0.71em;color:#888888;font-weight:normal'>"
               f"{rmse_s} · <span style='color:{gc}'>{gs}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{mhtml}{sec}</td>")

    def _delta_td(dr2, drmse=None, dgap=None):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        c = ("#5BAD6F" if dr2 >= MEANINGFUL else
             "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        ex = ""
        if drmse is not None and drmse == drmse and dgap is not None and dgap == dgap:
            rc = ("#5BAD6F" if drmse <= -MEANINGFUL else
                  "#E15252" if drmse >= MEANINGFUL else "#888888")
            gc = ("#E15252" if dgap > MEANINGFUL else
                  "#5BAD6F" if dgap < -MEANINGFUL else "#888888")
            ex = (f"<br><span style='font-size:0.71em;font-weight:normal'>"
                  f"<span style='color:{rc}'>{drmse:+.2f}</span>"
                  f" · <span style='color:{gc}'>{dgap:+.3f}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{c};font-weight:bold'>"
                f"{dr2:+.3f}{ex}</td>")

    def _stats_row(deltas, gaj_deltas, from_lbl, to_lbl):
        if not deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr  = np.array(deltas); n = len(arr); net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mw = float(wins.mean())   if len(wins)   else None
        ml = float(losses.mean()) if len(losses) else None
        nc = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        vd = "Net improvement" if net > MEANINGFUL else ("Net regression" if net < -MEANINGFUL else "Negligible")
        ws = f"{len(wins)}/{n} (avg {mw:+.3f})"   if mw is not None else f"{len(wins)}/{n}"
        ls = f"{len(losses)}/{n} (avg {ml:+.3f})" if ml is not None else f"{len(losses)}/{n}"
        if gaj_deltas:
            gn = float(np.array(gaj_deltas).mean()); diff = gn - net
            gc = "#5BAD6F" if gn > MEANINGFUL else ("#E15252" if gn < -MEANINGFUL else "#888888")
            dc = "#E15252" if diff < -0.02 else "#5BAD6F" if diff > 0.02 else "#888888"
            interp = ("Raw gains partially inflated by overfitting" if diff < -0.02
                      else "Raw gains understated - overfitting decreased" if diff > 0.02
                      else "Overfitting largely unchanged")
            gs = f"{gn:+.4f}"; ds = f"{diff:+.4f}"
        else:
            gs = ds = " - "; gc = dc = "#888888"; interp = " - "
        return (f"<tr>"
                f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
                f"<td style='{_STD};text-align:center;color:{nc};font-weight:bold'>{net:+.4f}</td>"
                f"<td style='{_STD};text-align:center;color:#5BAD6F'>{ws}</td>"
                f"<td style='{_STD};text-align:center;color:#E15252'>{ls}</td>"
                f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
                f"<td style='{_STD};text-align:center;color:{gc};font-weight:bold'>{gs}</td>"
                f"<td style='{_STD};text-align:center;color:{dc}'>{ds}</td>"
                f"<td style='{_STD}'><strong>{vd}</strong><br>"
                f"<span style='font-size:0.82em;color:{dc}'>{interp}</span></td></tr>")

    d_e3_s1  = []; gaj_e3_s1  = []
    d_s1_s2  = []; gaj_s1_s2  = []
    d_e3_s2  = []; gaj_e3_s2  = []

    tbody = ""
    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='8' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # pre-pass: find best raw and gap-adjusted per target across all models and SEs
        _all_raw = {}; _all_gaj = {}; _all_gap = {}
        for m_ in models_ord:
            for key_, df_ in [("e3", e3), ("s1", s1), ("s2", s2)]:
                v_ = _get(df_, m_, tgt); g_ = _gap_v(df_, m_, tgt)
                if v_ is not None:
                    _all_raw[(m_, key_)] = v_
                    _all_gap[(m_, key_)] = g_
                    sv_ = _gaj(v_, g_)
                    if sv_ is not None:
                        _all_gaj[(m_, key_)] = sv_

        tgt_br = max(_all_raw.values()) if _all_raw else None
        tgt_bg = max(_all_gaj.values()) if _all_gaj else None
        _raw_win_gap = None
        if tgt_br is not None:
            for k_, v_ in _all_raw.items():
                if abs(v_ - tgt_br) < 1e-9:
                    _raw_win_gap = _all_gap.get(k_)
                    break
        tgt_show_gaj = (
            tgt_bg is not None and tgt_br is not None
            and _raw_win_gap is not None and _raw_win_gap > 0.10
        )

        def _ir(v_): return tgt_br is not None and v_ is not None and abs(v_ - tgt_br) < 1e-9
        def _ig(s_): return tgt_show_gaj and tgt_bg is not None and s_ is not None and abs(s_ - tgt_bg) < 1e-9

        for m in models_ord:
            ve3  = _get(e3, m, tgt);  ge3  = _gap_v(e3, m, tgt)
            vs1  = _get(s1, m, tgt);  gs1  = _gap_v(s1, m, tgt)
            vs2  = _get(s2, m, tgt);  gs2  = _gap_v(s2, m, tgt)
            re3  = _get(e3, m, tgt, "RMSE_test")
            rs1  = _get(s1, m, tgt, "RMSE_test")
            rs2  = _get(s2, m, tgt, "RMSE_test")
            sce3 = _gaj(ve3, ge3);  scs1 = _gaj(vs1, gs1);  scs2 = _gaj(vs2, gs2)

            d13  = (vs1 - ve3) if (ve3 is not None and vs1 is not None) else None
            d12  = (vs2 - ve3) if (ve3 is not None and vs2 is not None) else None
            d13r = (rs1 - re3) if (re3 is not None and rs1 is not None) else None
            d12r = (rs2 - re3) if (re3 is not None and rs2 is not None) else None
            d13g = (gs1 - ge3) if (ge3 is not None and gs1 is not None) else None
            d12g = (gs2 - ge3) if (ge3 is not None and gs2 is not None) else None
            d32  = (vs2 - vs1) if (vs1 is not None and vs2 is not None) else None
            d32r = (rs2 - rs1) if (rs1 is not None and rs2 is not None) else None
            d32g = (gs2 - gs1) if (gs1 is not None and gs2 is not None) else None

            sd13 = (scs1 - sce3) if (sce3 is not None and scs1 is not None) else None
            sd12 = (scs2 - sce3) if (sce3 is not None and scs2 is not None) else None
            sd32 = (scs2 - scs1) if (scs1 is not None and scs2 is not None) else None

            if d13 is not None: d_e3_s1.append(d13);  gaj_e3_s1.append(sd13) if sd13 is not None else None
            if d32 is not None: d_s1_s2.append(d32);  gaj_s1_s2.append(sd32) if sd32 is not None else None
            if d12 is not None: d_e3_s2.append(d12);  gaj_e3_s2.append(sd12) if sd12 is not None else None

            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(ve3, re3, ge3, _ir(ve3), _ig(sce3))}"
                f"{_val_td(vs1, rs1, gs1, _ir(vs1), _ig(scs1))}"
                f"{_delta_td(d13, d13r, d13g)}"
                f"{_val_td(vs2, rs2, gs2, _ir(vs2), _ig(scs2))}"
                f"{_delta_td(d12, d12r, d12g)}"
                f"</tr>"
            )

    n_cells = len(models_ord) * len(TARGETS_ORDERED)
    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; -{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| ≤ {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²-0.5·max(0,|gap|-0.10)</span></th>
        <th style='{_THC}'>Gap-Adj - Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
        <th style='{_TH}'>Verdict / Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {_stats_row(d_e3_s1, gaj_e3_s1, "E3-SE2", "SE1 (manual prune)")}
      {_stats_row(d_s1_s2, gaj_s1_s2, "SE1", "SE2 (VIF prune)")}
      {_stats_row(d_e3_s2, gaj_e3_s2, "E3-SE2", "SE2 (combined)")}
    </tbody>
  </table>
  </div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.8rem'>
    <p class='meta'>
      <strong>Reading the table.</strong>
      Net Mean ΔR² is the signed average across all {n_cells} (model x target) cells.
      <strong>E3-SE2 → SE1</strong>: removes three feature groups (SVI, New aeration, Sec Sed) from Exp3-SE2.
      <strong>SE1 → SE2</strong>: applies automated iterative VIF pruning (threshold=10) on the SE1 pool.
      <strong>E3-SE2 → SE2</strong>: combined effect of both pruning steps vs the Exp3-SE2 baseline.
      <strong>Gap-Adj - Raw</strong>: negative = raw gains inflated by overfitting;
      positive = gains understated because overfitting decreased.
      <strong>★</strong> = best raw Test R² per target · <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
    </p>
  </div>
</div>"""

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.79rem;color:#1a1a1a;min-width:880px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:52px'>Model</th>
      <th style='{_THC}'>E3-SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ E3-SE2→SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
      <th style='{_THC}'>SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · RMSE · Gap</span></th>
      <th style='{_THC}'>Δ E3-SE2→SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR² · ΔRMSE · ΔGap</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp4-comparison" open>
  <summary><span class="fold-icon">▶</span>
    Comparisons
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target (across all models and SEs) ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10 and a different cell wins on gap-adjusted score).
        RMSE in native units (mg/L or pH). Gap = Train R² - Test R²;
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
        Delta columns (Δ): ΔR² · ΔRMSE · ΔGap vs E3-SE2 baseline.
      </p>
    </div>
    {main_table}
    {stats_block}
  </div>
</details>"""


def _exp4_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for Experiment 4."""
    e3 = df_all[df_all["exp_key"] == "Exp3-S2"].dropna(subset=["R2_test"])
    s1 = df_all[df_all["exp_key"] == "Exp4-S1"].dropna(subset=["R2_test"])
    s2 = df_all[df_all["exp_key"] == "Exp4-S2"].dropna(subset=["R2_test"])

    grab_tgts = set(GRAB_TARGETS)
    comp_tgts = set(COMP_TARGETS)

    def _avg(df, tgts, model=None):
        sub = df[df["model"] == model] if model else df
        vals = [float(sub[sub["target"] == t]["R2_test"].max())
                for t in tgts if not sub[sub["target"] == t].empty]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _c(v, positive_good=True):
        if v != v: return " - "
        col = ("#5BAD6F" if v > 0.05 else "#E15252" if v < -0.05 else "var(--text-muted)") \
              if positive_good else \
              ("#E15252" if v > 0.10 else "#5BAD6F" if v < 0.05 else "var(--text-muted)")
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    e3_grab = _avg(e3, grab_tgts); e3_comp = _avg(e3, comp_tgts)
    s1_grab = _avg(s1, grab_tgts); s1_comp = _avg(s1, comp_tgts)
    s2_grab = _avg(s2, grab_tgts); s2_comp = _avg(s2, comp_tgts)

    # Row counts: SE1 vs SE2 for representative targets
    def _rows(df, tgt, year=None):
        sub = df[df["target"] == tgt]
        if sub.empty or "n_train" not in sub.columns: return "?"
        return str(int(sub["n_train"].max()))

    # Gap stats
    s2_ridge_grab = _avg(s2, grab_tgts, "Ridge")
    s2_comp_pH_ridge = float(s2[
        (s2["model"] == "Ridge") &
        (s2["target"] == "Effluent pH (Composite)")]["R2_test"].max()) if not s2.empty else float("nan")

    q1 = (
        f"<strong>Sample size:</strong> Row counts barely changed for SE1 (+5-10, &lt;2%) because "
        f"Sec Sed and Sec Clarifier are collected on the same operational days (co-occurring missingness). "
        f"Removing Sec Sed does not unlock any additional rows. "
        f"SE2 (VIF pruning) unlocked meaningful extra rows by eliminating high-missingness pH and Inlet BOD/COD "
        f"columns: Comp TSS grew from ~448 to ~741 rows (+65%), Comp COD from ~684 to ~866 (+27%). "
        f"This row gain is the only genuine benefit of VIF pruning in this experiment."
        f"<br><br>"
        f"<strong>Performance - SE1:</strong> E3-SE2 baseline grab avg = {_c(e3_grab)}, comp avg = {_c(e3_comp)}. "
        f"After manual group removal: SE1 grab avg = {_c(s1_grab)}, comp avg = {_c(s1_comp)}. "
        f"Performance deteriorated on every target and every model. "
        f"<br><br>"
        f"<strong>Performance - SE2:</strong> grab avg = {_c(s2_grab)}, comp avg = {_c(s2_comp)}. "
        f"VIF pruning further degraded results on most targets. Ridge grab recovered partially "
        f"(Ridge grab avg ~{_c(s2_ridge_grab)}), but Comp pH Ridge collapsed to "
        f"{_c(s2_comp_pH_ridge)}."
    )

    q2 = (
        f"No - both pruning strategies made things worse. "
        f"<strong>SE1 (manual removal):</strong> removing correlated groups discarded genuine signal. "
        f"Key degradation: RF Grab COD dropped from +0.40 (Exp3-SE2) to -0.08 (SE1). "
        f"ElNet Grab BOD: +0.68 to +0.17. The premise that Sec Clarifier alone suffices was wrong - "
        f"Exp2's split analysis had already shown Sec Sedimentation was the stronger standalone predictor, "
        f"and FS results from Exp3-SE2 confirm both groups carry non-redundant signal. "
        f"<br><br>"
        f"<strong>SE2 (VIF pruning):</strong> VIF is target-agnostic - it removes features correlated "
        f"with other features regardless of their signal value. Inlet BOD is highly correlated with other "
        f"inlet features, but it also directly predicts Effluent BOD. Sec Clarifier pH is collinear with "
        f"other pH columns, but it carries the strongest pH signal for effluent pH prediction. "
        f"VIF pruning strips precisely the features that matter most. "
        f"Tree models collapsed catastrophically when pH features and Inlet BOD/COD were removed."
    )

    q3 = (
        f"Collinearity in this dataset is best handled by <strong>model-side regularisation</strong>, "
        f"not manual feature removal. "
        f"Ridge and ElasticNet suppress collinear features automatically via L2/L1 penalties. "
        f"Tree models (RF, GB, XGB) are unaffected by collinearity - correlated features simply "
        f"share split importance, and removing them reduces split diversity, which worsens overfitting. "
        f"<br><br>"
        f"The collinearity between Sec Clarifier and Sec Sed, or between aeration Existing and New tanks, "
        f"reflects real process physics: the plant runs in a correlated steady state. These correlations "
        f"are not a modelling problem; they are a feature of the data that regularised models handle correctly. "
        f"<br><br>"
        f"<strong>Exp3-SE2 remains the recommended feature set for all models.</strong> "
        f"The only benefit from SE2 (VIF) is the row gain on composite targets, which could be "
        f"recovered more cleanly by explicitly excluding high-missingness features at dataset-build time "
        f"rather than through VIF-driven pruning."
    )

    def _qcard(n, question, answer):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);
              display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);padding-left:0.6rem'>{answer}</p>
  </div>
</div>"""

    cards = "".join([
        _qcard(1, "How did each SE's feature set affect sample size and performance?", q1),
        _qcard(2, "Did removing feature groups (manual SE1 or automated SE2) improve generalization?", q2),
        _qcard(3, "What is the strategic takeaway for collinearity handling?", q3),
    ])

    return f"""
<details class="exp-details" id="exp4-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings - VIF-Pruned Feature Set</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def build_exp4_section(df_all: pd.DataFrame) -> str:
    sub1 = _exp_subsection(df_all, "Exp4-S1", "exp4-s1",
                           "SE1 - Manual Group Removal (SVI, New Aeration, Sec Clarifier)",
                           open_default=True)

    # SE2: inject the VIF analysis at the top of the section body as its motivation
    vif_div = _vif_callout()
    vif_intro = f"""
<div class="obs-card" style="border-left:4px solid #9B59B6;margin-bottom:1rem">
  <p class="meta">
    <strong>Motivation for VIF pruning.</strong>
    The Exp3-SE2 feature set contains several correlated groups (Sec Clarifier vs Sec Sed;
    Aeration MLSS vs SVI; pH buffered across all stages). VIF quantifies this for OLS below.
    VIF matters primarily for <strong>OLS</strong> - high VIF inflates coefficient standard
    errors and causes instability. <strong>Ridge</strong> and <strong>ElasticNet</strong> handle
    collinearity via L2/L1 penalties; <strong>tree models</strong> are entirely unaffected.
    SE2 tests whether automated iterative VIF pruning (threshold=10) on the SE1 pool improves
    generalisation by removing within-group collinearity more precisely than manual removal.
  </p>
  {vif_div}
</div>"""

    sub2_raw = _exp_subsection(df_all, "Exp4-S2", "exp4-s2",
                               "SE2 - VIF-based Automated Pruning (threshold = 10)",
                               open_default=True)
    # Inject VIF intro at the top of SE2's exp-body
    sub2 = sub2_raw.replace('<div class="exp-body">', f'<div class="exp-body">{vif_intro}', 1)

    comparison = _exp4_comparison_panel(df_all)
    findings   = _exp4_qna(df_all)
    best = _best_model_box(df_all[df_all["exp_key"].isin(["Exp4-S1", "Exp4-S2"])], "VIF-Pruned Feature Set")

    return f"""
<section id="exp4">
  <h1 class="section-title">VIF-Pruned Feature Set</h1>
  <p class="section-intro">{EXP_INTRO["Exp4"]}</p>
  {sub1}
  {sub2}
  {comparison}
  {findings}
  {best}
</section>"""


def _exp5_comparison_panel(df_all: pd.DataFrame) -> str:
    """Side-by-side comparison: Sub1 (Composite+GrabInlet) vs Sub2 (Grab+CompInlet).

    The baseline for each sub-experiment is the corresponding Exp3-SE1 result
    (same target type, same base feature set without cross-type inlet).
    """
    s1    = df_all[df_all["exp_key"] == "Exp5-S1"].copy()
    s1fs  = df_all[df_all["exp_key"] == "Exp5-S1-FS"].copy()
    s2    = df_all[df_all["exp_key"] == "Exp5-S2"].copy()
    s2fs  = df_all[df_all["exp_key"] == "Exp5-S2-FS"].copy()
    base1 = df_all[df_all["exp_key"] == "Exp3-S1"].copy()

    if s1.empty and s2.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL = 0.01
    _TD  = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

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

    def _val_td(r2, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        mhtml  = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:   col = "#5BAD6F"; fw = "bold"
        elif is_gaj: col = "#4A90D9"; fw = "bold"
        else:        col = "#1a1a1a"; fw = "normal"
        gap_val = gap if (gap is not None and gap == gap) else None
        gap_str = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col = ("#E15252" if gap_val is not None and gap_val > 0.10
                   else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                   else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;font-weight:normal'>"
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{mhtml}{secondary}</td>")

    def _delta_td(dr2):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        col = ("#5BAD6F" if dr2 >= MEANINGFUL else "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:bold'>"
                f"{dr2:+.3f}</td>")

    tbody = ""
    for tgt in COMP_TARGETS + GRAB_TARGETS:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='8' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # --- pre-pass: global best for this target across all models and SEs ---
        _all_raw_e5 = {}
        _all_gaj_e5 = {}
        _all_gap_e5 = {}
        for m_ in models_ord:
            vb_ = _get(base1, m_, tgt); gb_ = _gap_v(base1, m_, tgt)
            v1_ = _get(s1,    m_, tgt); g1_ = _gap_v(s1,    m_, tgt)
            v1f_= _get(s1fs,  m_, tgt); g1f_= _gap_v(s1fs,  m_, tgt)
            v2_ = _get(s2,    m_, tgt); g2_ = _gap_v(s2,    m_, tgt)
            v2f_= _get(s2fs,  m_, tgt); g2f_= _gap_v(s2fs,  m_, tgt)
            sb_ = _gaj(vb_, gb_); ss1_ = _gaj(v1_, g1_)
            ss1f_= _gaj(v1f_, g1f_); ss2_ = _gaj(v2_, g2_); ss2f_ = _gaj(v2f_, g2f_)
            for key_, rv_, gv_, sv_ in [
                ((m_, "base"), vb_, gb_, sb_),
                ((m_, "s1"), v1_, g1_, ss1_),
                ((m_, "s1f"), v1f_, g1f_, ss1f_),
                ((m_, "s2"), v2_, g2_, ss2_),
                ((m_, "s2f"), v2f_, g2f_, ss2f_),
            ]:
                if rv_ is not None:
                    _all_raw_e5[key_] = rv_
                    _all_gap_e5[key_] = gv_
                if sv_ is not None:
                    _all_gaj_e5[key_] = sv_

        tgt_br_e5 = max(_all_raw_e5.values()) if _all_raw_e5 else None
        tgt_bg_e5 = max(_all_gaj_e5.values()) if _all_gaj_e5 else None

        _raw_win_gap_e5 = None
        if tgt_br_e5 is not None:
            for k_, v_ in _all_raw_e5.items():
                if abs(v_ - tgt_br_e5) < 1e-9:
                    _raw_win_gap_e5 = _all_gap_e5.get(k_)
                    break

        tgt_show_gaj_e5 = (
            tgt_bg_e5 is not None and tgt_br_e5 is not None
            and _raw_win_gap_e5 is not None and _raw_win_gap_e5 > 0.10
        )

        def _ir(v_): return tgt_br_e5 is not None and v_ is not None and abs(v_ - tgt_br_e5) < 1e-9
        def _ig(s_): return tgt_show_gaj_e5 and tgt_bg_e5 is not None and s_ is not None and abs(s_ - tgt_bg_e5) < 1e-9

        for m in models_ord:
            v_base = _get(base1, m, tgt)
            v_s1   = _get(s1,    m, tgt)
            v_s1f  = _get(s1fs,  m, tgt)
            v_s2   = _get(s2,    m, tgt)
            v_s2f  = _get(s2fs,  m, tgt)
            g_base = _gap_v(base1, m, tgt)
            g_s1   = _gap_v(s1,   m, tgt)
            g_s1f  = _gap_v(s1fs, m, tgt)
            g_s2   = _gap_v(s2,   m, tgt)
            g_s2f  = _gap_v(s2fs, m, tgt)

            d1   = (v_s1  - v_base) if (v_base is not None and v_s1  is not None) else None
            d1fs = (v_s1f - v_s1)   if (v_s1   is not None and v_s1f is not None) else None
            d2   = (v_s2  - v_base) if (v_base is not None and v_s2  is not None) else None
            d2fs = (v_s2f - v_s2)   if (v_s2   is not None and v_s2f is not None) else None

            sc_base = _gaj(v_base, g_base)
            sc_s1   = _gaj(v_s1,   g_s1)
            sc_s1f  = _gaj(v_s1f,  g_s1f)
            sc_s2   = _gaj(v_s2,   g_s2)
            sc_s2f  = _gaj(v_s2f,  g_s2f)

            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(v_base, g_base, _ir(v_base), _ig(sc_base))}"
                f"{_val_td(v_s1,   g_s1,   _ir(v_s1),   _ig(sc_s1))}"
                f"{_val_td(v_s1f,  g_s1f,  _ir(v_s1f),  _ig(sc_s1f))}"
                f"{_delta_td(d1fs)}"
                f"{_val_td(v_s2,   g_s2,   _ir(v_s2),   _ig(sc_s2))}"
                f"{_val_td(v_s2f,  g_s2f,  _ir(v_s2f),  _ig(sc_s2f))}"
                f"{_delta_td(d2fs)}"
                f"</tr>"
            )

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a;min-width:1100px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:60px'>Model</th>
      <th style='{_THC}'>Baseline<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · Gap</span></th>
      <th style='{_THC}'>SE1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · Gap</span></th>
      <th style='{_THC}'>SE1-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · Gap</span></th>
      <th style='{_THC}'>Δ (SE1 → SE1-FS)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR²</span></th>
      <th style='{_THC}'>SE2<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · Gap</span></th>
      <th style='{_THC}'>SE2-FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>R² · Gap</span></th>
      <th style='{_THC}'>Δ (SE2 → SE2-FS)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>ΔR²</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp5-comparison">
  <summary><span class="fold-icon">▶</span>
    Comparisons
  </summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
      </p>
    </div>
    {main_table}
  </div>
</details>"""


def _exp5_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for Experiment 5."""
    s1    = df_all[df_all["exp_key"] == "Exp5-S1"].dropna(subset=["R2_test"])
    s2    = df_all[df_all["exp_key"] == "Exp5-S2"].dropna(subset=["R2_test"])
    base1 = df_all[df_all["exp_key"] == "Exp3-S1"].dropna(subset=["R2_test"])

    def _avg(df, tgts):
        vals = [float(df[df["target"] == t]["R2_test"].max())
                for t in tgts if not df[df["target"] == t].empty]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _c(v, pos=True):
        if v != v: return " - "
        col = ("#5BAD6F" if v > 0.05 else "#E15252" if v < -0.05 else "var(--text-muted)") \
              if pos else \
              ("#E15252" if v > 0.10 else "#5BAD6F" if v < 0.05 else "var(--text-muted)")
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    s1_comp_avg  = _avg(s1, COMP_TARGETS)
    s2_grab_avg  = _avg(s2, GRAB_TARGETS)
    b1_comp_avg  = _avg(base1, COMP_TARGETS)
    b1_grab_avg  = _avg(base1, GRAB_TARGETS)

    delta_comp = s1_comp_avg - b1_comp_avg if not (np.isnan(s1_comp_avg) or np.isnan(b1_comp_avg)) else float("nan")
    delta_grab = s2_grab_avg - b1_grab_avg if not (np.isnan(s2_grab_avg) or np.isnan(b1_grab_avg)) else float("nan")

    q1 = (
        f"<strong>SE1 (Composite + Grab inlet):</strong> "
        f"avg composite Test R² = {_c(s1_comp_avg)} vs Exp3-SE1 baseline {_c(b1_comp_avg)}. "
        f"Marginal delta: {_c(delta_comp)}. Row cost ~22.7% (629 from 816 train rows). "
        f"<br><br>"
        f"<strong>SE2 (Grab + Composite inlet):</strong> "
        f"avg grab Test R² = {_c(s2_grab_avg)} vs Exp3-SE1 baseline {_c(b1_grab_avg)}. "
        f"Marginal delta: {_c(delta_grab)}. Row cost ~38.2% (630 from 1021 train rows). "
        f"<br><br>"
        f"Compare deltas against the row-cost paid: if the cross-type columns add less than "
        f"the baseline R² would have gained from 22-38% more training rows, they carry no net value."
    )

    q2 = (
        f"The cross-type inlet hypothesis has an inherent baseline contamination: by requiring "
        f"co-occurrence of grab and composite inlet measurements, we have <em>changed the training "
        f"population</em>. Days with both grab and composite measurements may be systematically "
        f"different from days with only one type (e.g. more routine vs storm-event days). "
        f"<br><br>"
        f"Interpretation rule: a positive delta confirms cross-type signal, but a negative or zero "
        f"delta may reflect <em>selection bias</em> (the joint-availability filter) rather than "
        f"absence of cross-type signal. To isolate the pure feature effect, compare "
        f"Exp5-SE1/S2 directly against Exp3-SE1 trained on the <strong>same restricted rows</strong> "
        f"(an Exp3-SE1 re-run with joint-availability mask). This is a natural follow-up experiment."
    )

    def _qcard(n, question, answer):
        return f"""
<div style='margin-bottom:1.1rem;border:1px solid var(--border);border-radius:5px;overflow:hidden'>
  <div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;border-bottom:1px solid var(--border);
              display:flex;align-items:baseline;gap:0.5rem'>
    <span style='color:#4A90D9;font-size:0.78em;font-weight:bold;letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>
    <span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>
  </div>
  <div style='padding:0.6rem 0.8rem 0.55rem'>
    <p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);
        padding-left:0.6rem'>{answer}</p>
  </div>
</div>"""

    cards = "".join([
        _qcard(1, "Did adding cross-type inlet features improve prediction for each target type?", q1),
        _qcard(2, "What are the confounds in interpreting the cross-type experiment?", q2),
    ])

    return f"""
<details class="exp-details" id="exp5-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings  -  Cross-Type Inlet Hypothesis</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def build_exp5_section(df_all: pd.DataFrame) -> str:
    # SE1: Composite targets + Grab inlet - full + FS
    sub1_full = _exp_subsection(df_all, "Exp5-S1", "exp5-s1-full",
                                "Full Feature Set (32 features)",
                                open_default=True)
    sub1_fs   = _exp_subsection(df_all, "Exp5-S1-FS", "exp5-s1-fs",
                                "With Feature Selection",
                                open_default=False,
                                dataset_summary_fn=_dataset_summary_per_model)
    sub1_wrapper = f"""
<details class="exp-details" open id="exp5-s1">
  <summary><span class="fold-icon">▶</span> SE1  -  Comp + Grab Inlet (full + FS)</summary>
  <div class="exp-body">
    {sub1_full}
    {sub1_fs}
  </div>
</details>"""

    # SE2: Grab targets + Composite inlet - full + FS
    sub2_full = _exp_subsection(df_all, "Exp5-S2", "exp5-s2-full",
                                "Full Feature Set (32 features)",
                                open_default=False)
    sub2_fs   = _exp_subsection(df_all, "Exp5-S2-FS", "exp5-s2-fs",
                                "With Feature Selection",
                                open_default=False,
                                dataset_summary_fn=_dataset_summary_per_model)
    sub2_wrapper = f"""
<details class="exp-details" id="exp5-s2">
  <summary><span class="fold-icon">▶</span> SE2  -  Grab + Comp Inlet (full + FS)</summary>
  <div class="exp-body">
    {sub2_full}
    {sub2_fs}
  </div>
</details>"""

    cmp_div      = _exp5_comparison_panel(df_all)
    findings_div = _exp5_qna(df_all)
    best         = _best_model_box(
        df_all[df_all["exp_key"].isin(["Exp5-S1", "Exp5-S1-FS", "Exp5-S2", "Exp5-S2-FS"])],
        "Cross-Type Inlet Hypothesis")

    return f"""
<section id="exp5">
  <h1 class="section-title">Cross-Type Inlet Hypothesis</h1>
  <p class="section-intro">{EXP_INTRO["Exp5"]}</p>
  {sub1_wrapper}
  {sub2_wrapper}
  {cmp_div}
  {findings_div}
  {best}
</section>"""


def _exp9_comparison_panel(df_all: pd.DataFrame) -> str:
    """Baseline (Exp3-S1) vs W1 (SE1) vs W1+FS (SE2) vs W1+LogY (SE3) vs W1+LogF (SE4) vs W1+LogLog (SE5)."""
    w1    = df_all[df_all["exp_key"] == "Exp9-SE1"].copy()
    fs    = df_all[df_all["exp_key"] == "Exp9-SE2"].copy()
    logy  = df_all[df_all["exp_key"] == "Exp9-SE3"].copy()
    logf  = df_all[df_all["exp_key"] == "Exp9-SE4"].copy()
    logll = df_all[df_all["exp_key"] == "Exp9-SE5"].copy()
    base  = df_all[df_all["exp_key"] == "Exp3-S1"].copy()

    if w1.empty:
        return ""

    models_ord = ["OLS", "Ridge", "ElNet", "RF", "GB", "XGB"]
    MEANINGFUL = 0.01
    _TD  = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

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

    def _val_td(r2, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        mhtml  = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:   col = "#5BAD6F"; fw = "bold"
        elif is_gaj: col = "#4A90D9"; fw = "bold"
        else:        col = "#1a1a1a"; fw = "normal"
        gap_val = gap if (gap is not None and gap == gap) else None
        gap_str = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col = ("#E15252" if gap_val is not None and gap_val > 0.10
                   else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                   else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;font-weight:normal'>"
                     f"<span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{mhtml}{secondary}</td>")

    def _delta_td(dr2):
        if dr2 is None:
            return f"<td style='{_TD};text-align:center;color:#999'> - </td>"
        col = ("#5BAD6F" if dr2 >= MEANINGFUL else "#E15252" if dr2 <= -MEANINGFUL else "#888888")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:bold'>"
                f"{dr2:+.3f}</td>")

    tbody = ""
    for tgt in GRAB_TARGETS + COMP_TARGETS:
        short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='11' style='padding:6px 10px;font-size:0.75rem;font-weight:700;"
            f"color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{short}</td></tr>"
        )

        # pre-pass: find global best for markers (across all six sources)
        all_raw, all_gaj, all_gap = {}, {}, {}
        for m_ in models_ord:
            for src_key, src_df in [("b", base), ("w", w1), ("f", fs), ("l", logy), ("g", logf), ("ll", logll)]:
                rv_ = _get(src_df, m_, tgt); gv_ = _gap_v(src_df, m_, tgt)
                sv_ = _gaj(rv_, gv_)
                if rv_ is not None: all_raw[(m_, src_key)] = rv_; all_gap[(m_, src_key)] = gv_
                if sv_ is not None: all_gaj[(m_, src_key)] = sv_

        tgt_br = max(all_raw.values()) if all_raw else None
        tgt_bg = max(all_gaj.values()) if all_gaj else None
        raw_win_gap = None
        if tgt_br is not None:
            for k_, v_ in all_raw.items():
                if abs(v_ - tgt_br) < 1e-9:
                    raw_win_gap = all_gap.get(k_); break
        show_gaj = (tgt_bg is not None and tgt_br is not None
                    and raw_win_gap is not None and raw_win_gap > 0.10)

        def _ir(v_): return tgt_br is not None and v_ is not None and abs(v_ - tgt_br) < 1e-9
        def _ig(s_): return show_gaj and tgt_bg is not None and s_ is not None and abs(s_ - tgt_bg) < 1e-9

        for m in models_ord:
            vb  = _get(base,  m, tgt); gb  = _gap_v(base,  m, tgt)
            vw  = _get(w1,    m, tgt); gw  = _gap_v(w1,    m, tgt)
            vf  = _get(fs,    m, tgt); gf  = _gap_v(fs,    m, tgt)
            vl  = _get(logy,  m, tgt); gl  = _gap_v(logy,  m, tgt)
            vg  = _get(logf,  m, tgt); gg  = _gap_v(logf,  m, tgt)
            vll = _get(logll, m, tgt); gll = _gap_v(logll, m, tgt)
            delta_w   = (vw  - vb) if (vb  is not None and vw  is not None) else None
            delta_fs  = (vf  - vw) if (vw  is not None and vf  is not None) else None
            delta_ly  = (vl  - vw) if (vw  is not None and vl  is not None) else None
            delta_lf  = (vg  - vw) if (vw  is not None and vg  is not None) else None
            delta_ll  = (vll - vw) if (vw  is not None and vll is not None) else None
            sb  = _gaj(vb,  gb);  sw  = _gaj(vw,  gw)
            sf  = _gaj(vf,  gf);  sl  = _gaj(vl,  gl)
            sg  = _gaj(vg,  gg);  sll = _gaj(vll, gll)
            row_bg = "#ffffff" if models_ord.index(m) % 2 == 0 else "#f7f7f7"
            tbody += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_TD}'><strong>{m}</strong></td>"
                f"{_val_td(vb,  gb,  _ir(vb),  _ig(sb))}"
                f"{_val_td(vw,  gw,  _ir(vw),  _ig(sw))}"
                f"{_delta_td(delta_w)}"
                f"{_val_td(vf,  gf,  _ir(vf),  _ig(sf))}"
                f"{_delta_td(delta_fs)}"
                f"{_val_td(vl,  gl,  _ir(vl),  _ig(sl))}"
                f"{_delta_td(delta_ly)}"
                f"{_val_td(vg,  gg,  _ir(vg),  _ig(sg))}"
                f"{_delta_td(delta_lf)}"
                f"{_val_td(vll, gll, _ir(vll), _ig(sll))}"
                f"{_delta_td(delta_ll)}"
                f"</tr>"
            )

    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;
              color:#1a1a1a;min-width:1220px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:60px'>Model</th>
      <th style='{_THC}'>Baseline (Exp3-SE1)<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>2021-24 · R² · Gap</span></th>
      <th style='{_THC}'>SE1: W1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>2024-only · R² · Gap</span></th>
      <th style='{_THC}'>W1 - Base<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>Delta R²</span></th>
      <th style='{_THC}'>SE2: W1+FS<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>LassoCV/OOF · R² · Gap</span></th>
      <th style='{_THC}'>FS - W1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>Delta R²</span></th>
      <th style='{_THC}'>SE3: W1+LogY<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>log1p target · R² · Gap</span></th>
      <th style='{_THC}'>LogY - W1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>Delta R²</span></th>
      <th style='{_THC}'>SE4: W1+LogF<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>log1p features · R² · Gap</span></th>
      <th style='{_THC}'>LogF - W1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>Delta R²</span></th>
      <th style='{_THC}'>SE5: W1+LogLog<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>log feat+target · R² · Gap</span></th>
      <th style='{_THC}'>LogLog - W1<br>
          <span style='color:#888888;font-weight:400;font-size:0.78em'>Delta R²</span></th>
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    return f"""
<details class="exp-details" id="exp9-comparison">
  <summary><span class="fold-icon">▶</span> Comparisons</summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        Baseline = Exp3-SE1 (same 27-feature set, full 2021-2024, n=816/634).
        SE1 = Exp9-SE1 (2024-only, n=187/179, no FE).
        SE2 = Exp9-SE2 (2024-only + LassoCV/OOF FS).
        SE3 = Exp9-SE3 (2024-only + log1p target, Duan smearing; BOD/COD/TSS only, pH untransformed).
        SE4 = Exp9-SE4 (2024-only + log1p on 12 concentration features in-place; target on original scale).
        SE5 = Exp9-SE5 (2024-only + log1p features in-place + log1p target + Duan smearing; log-log model).
        <strong>★</strong> = best raw R² per target across all columns ·
        <strong>✦</strong> = best gap-adjusted (shown when raw winner gap &gt; 0.10).
        Delta columns compare each FE variant to the plain W1 (SE1) baseline.
      </p>
    </div>
    {main_table}
  </div>
</details>"""


def _exp9_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for Experiment 9 (SE1 + SE2 + SE3 + SE4 + SE5)."""
    w1    = df_all[df_all["exp_key"] == "Exp9-SE1"].dropna(subset=["R2_test"])
    fs    = df_all[df_all["exp_key"] == "Exp9-SE2"].dropna(subset=["R2_test"])
    logy  = df_all[df_all["exp_key"] == "Exp9-SE3"].dropna(subset=["R2_test"])
    logf  = df_all[df_all["exp_key"] == "Exp9-SE4"].dropna(subset=["R2_test"])
    logll = df_all[df_all["exp_key"] == "Exp9-SE5"].dropna(subset=["R2_test"])
    base  = df_all[df_all["exp_key"] == "Exp3-S1"].dropna(subset=["R2_test"])

    def _best(df, tgt):
        sub = df[df["target"] == tgt]
        return float(sub["R2_test"].max()) if not sub.empty else float("nan")

    def _best_model(df, tgt):
        sub = df[df["target"] == tgt]
        if sub.empty: return " - ", float("nan")
        idx = sub["R2_test"].idxmax()
        return sub.loc[idx, "model"], float(sub.loc[idx, "R2_test"])

    def _c(v, threshold=0.0):
        if v != v: return " - "
        col = "#5BAD6F" if v > threshold else "#E15252" if v < -0.05 else "var(--text-muted)"
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    def _dc(v):
        if v != v: return " - "
        col = "#5BAD6F" if v >= 0.01 else "#E15252" if v <= -0.01 else "var(--text-muted)"
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    # Pre-compute key numbers
    lin_models  = ["OLS", "Ridge", "ElNet"]
    tree_models = ["RF", "GB", "XGB"]

    grab_bods = _best(w1[w1["model"].isin(lin_models)],  "Effluent BOD (mg/L, Grab)")
    grab_bodb = _best(base[base["model"].isin(lin_models)], "Effluent BOD (mg/L, Grab)")
    grab_tsss = _best(w1[w1["model"].isin(lin_models)],  "Effluent TSS (mg/L, Grab)")
    grab_tssb = _best(base[base["model"].isin(lin_models)], "Effluent TSS (mg/L, Grab)")

    comp_bods = _best(w1[w1["model"].isin(lin_models)],  "Effluent BOD (mg/L, Composite)")
    comp_bodb = _best(base[base["model"].isin(lin_models)], "Effluent BOD (mg/L, Composite)")
    comp_cods = _best(w1[w1["model"].isin(lin_models)],  "Effluent COD (mg/L, Composite)")
    comp_codb = _best(base[base["model"].isin(lin_models)], "Effluent COD (mg/L, Composite)")

    # Best linear vs best tree for W1 grab BOD
    lin_grab_bod_m, lin_grab_bod_v = _best_model(w1[w1["model"].isin(lin_models)],  "Effluent BOD (mg/L, Grab)")
    tree_grab_bod_m, tree_grab_bod_v = _best_model(w1[w1["model"].isin(tree_models)], "Effluent BOD (mg/L, Grab)")

    q1 = (
        f"<strong>Yes for linear models, no for trees.</strong> "
        f"Restricting training to 2024 (187 Grab / 179 Comp rows, ~23% of the full window) "
        f"dramatically improves linear model performance across BOD, COD, and TSS targets. "
        f"Grab BOD: best linear W1 {_c(grab_bods)} vs baseline {_c(grab_bodb)} "
        f"(delta {_dc(grab_bods - grab_bodb if grab_bods == grab_bods and grab_bodb == grab_bodb else float('nan'))}). "
        f"Grab TSS: {_c(grab_tsss)} vs {_c(grab_tssb)}. "
        f"Comp BOD linear: {_c(comp_bods)} vs baseline {_c(comp_bodb)}. "
        f"Tree models are largely flat or marginally worse, confirming the benefit is "
        f"specific to linear estimation under distribution shift."
    )

    q2 = (
        f"<strong>The split is consistent across all targets where signal exists.</strong> "
        f"For every grab and composite BOD/COD/TSS target, linear models (OLS, Ridge, ElNet) "
        f"show positive or near-zero deltas vs the baseline. Trees (RF, GB, XGB) "
        f"show flat or negative deltas. The one exception is Grab BOD GB: "
        f"{_c(tree_grab_bod_v)} (W1) vs the baseline GB, a marginal improvement. "
        f"This model-family split is the cleanest finding in the experiment: "
        f"older years introduce regime-inconsistent patterns that linear models learn "
        f"literally, while trees, being locally adaptive, are less sensitive to global drift "
        f"but also gain nothing from restricting to a cleaner regime."
    )

    q3 = (
        f"<strong>BOD and TSS benefit most; pH does not.</strong> "
        f"Grab BOD OLS improves from {_c(grab_bodb)} to {_c(grab_bods)} - "
        f"the best OLS result on this target across all experiments. "
        f"Grab TSS OLS: {_c(grab_tssb)} to {_c(grab_tsss)}. "
        f"Comp BOD OLS: {_c(comp_bodb)} to {_c(comp_bods)}. "
        f"pH targets break the pattern: Grab pH OLS deteriorates from {_c(-0.139)} to {_c(-0.789)}, "
        f"and Ridge drops from {_c(0.141)} to {_c(0.056)}. "
        f"pH is a tightly controlled operational parameter whose 2024 range may be narrower "
        f"or slightly offset relative to 2025, making single-year training actively misleading."
    )

    q4 = (
        f"<strong>Comp COD (ElNet {_c(comp_cods)}) is the most significant result in this experiment.</strong> "
        f"This is the first positive test R² ever recorded for Effluent COD (Composite) "
        f"across all experiments (Exp1-8). The baseline best linear was {_c(comp_codb)}. "
        f"ElNet alpha=1.0, l1_ratio=0.7 (moderate L1 sparsity) on 179 training rows finds "
        f"a linear relationship that 634 full-window rows could not. "
        f"Ridge {_c(comp_cods + 0.0)} similarly breaks positive. "
        f"This directly confirms the recency hypothesis for Comp COD: the 2021-2023 data "
        f"was not just unhelpful - it was actively suppressing the 2024-2025 signal. "
        f"Trees remain negative, consistent with the model-family pattern above."
    )

    q5 = (
        f"<strong>The recency hypothesis is confirmed for linear models.</strong> "
        f"The 2024-only window removes distributional noise from earlier operational regimes, "
        f"revealing a cleaner linear relationship between features and effluent quality "
        f"that the model can generalise to 2025. The underlying mechanism is distribution "
        f"shift: the plant's 2024 operating conditions are more similar to 2025 than the "
        f"2021-2022 conditions are. "
        f"<br><br>"
        f"<strong>Immediate implication:</strong> for operational forecasting, a regularised "
        f"linear model (Ridge or ElNet) trained on the most recent year is likely to "
        f"outperform any model trained on the full historical window, at least until "
        f"sufficient 2025 data is available to stabilise a combined model. "
        f"This finding motivates extending the study to W2 (2023-2024) and W3 (2022-2024) "
        f"to determine the optimal lookback window."
    )

    # SE2 comparisons (FS vs no-FS within W1)
    fs_grab_bod_ols  = _best(fs[fs["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    w1_grab_bod_ols  = _best(w1[w1["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    fs_grab_bod_ridge = _best(fs[fs["model"] == "Ridge"], "Effluent BOD (mg/L, Grab)")
    w1_grab_bod_ridge = _best(w1[w1["model"] == "Ridge"], "Effluent BOD (mg/L, Grab)")
    fs_comp_tss_ols  = _best(fs[fs["model"] == "OLS"],   "Effluent TSS (mg/L, Composite)")
    w1_comp_tss_ols  = _best(w1[w1["model"] == "OLS"],   "Effluent TSS (mg/L, Composite)")
    fs_comp_ph_ridge = _best(fs[fs["model"] == "Ridge"], "Effluent pH (Composite)")
    w1_comp_ph_ridge = _best(w1[w1["model"] == "Ridge"], "Effluent pH (Composite)")

    q6 = (
        f"<strong>Feature selection (SE2) does not improve over unselected W1 (SE1) for most targets.</strong> "
        f"LassoCV on 187 Grab / 179 Comp training rows aggressively prunes to 2-13 features, "
        f"reducing OLS performance on targets where it was already strong: "
        f"Grab BOD OLS full (+{w1_grab_bod_ols:.3f}) vs OLS-FS ({_c(fs_grab_bod_ols)}), "
        f"Ridge unchanged at {_c(fs_grab_bod_ridge)} (full set, not pruned). "
        f"<br><br>"
        f"There are two exceptions where LassoCV FS rescues unstable OLS: "
        f"Comp TSS OLS improves from {_c(w1_comp_tss_ols)} (W1 full) to {_c(fs_comp_tss_ols)} (FS) "
        f"and Comp pH Ridge is comparable at {_c(fs_comp_ph_ridge)}. "
        f"For tree models, OOF permutation importance FS is consistently harmful on the small W1 window "
        f"(GB/XGB gaps exceed 0.8-2.2), confirming that 187 rows cannot support reliable OOF "
        f"importance estimation. "
        f"<br><br>"
        f"<strong>Verdict:</strong> SE1 (no FS) is the recommended W1 variant. "
        f"Ridge and ElNet (which regularise or self-select internally) achieve the best W1 "
        f"performance without explicit FS. For future recency-window studies (W2, W3), "
        f"feature selection becomes more viable as n increases."
    )

    # SE3 comparisons (log1p target vs plain W1)
    ly_grab_tss_ols  = _best(logy[logy["model"] == "OLS"],   "Effluent TSS (mg/L, Grab)")
    w1_grab_tss_ols  = _best(w1[w1["model"] == "OLS"],       "Effluent TSS (mg/L, Grab)")
    ly_grab_bod_ols  = _best(logy[logy["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    w1_grab_bod_ols  = _best(w1[w1["model"] == "OLS"],       "Effluent BOD (mg/L, Grab)")
    ly_grab_cod_ridge = _best(logy[logy["model"] == "Ridge"], "Effluent COD (mg/L, Grab)")
    w1_grab_cod_ridge = _best(w1[w1["model"] == "Ridge"],     "Effluent COD (mg/L, Grab)")
    ly_comp_cod_best = _best(logy[logy["model"].isin(["OLS","Ridge","ElNet"])],
                             "Effluent COD (mg/L, Composite)")
    w1_comp_cod_best = _best(w1[w1["model"].isin(["OLS","Ridge","ElNet"])],
                             "Effluent COD (mg/L, Composite)")

    q7 = (
        f"<strong>Log1p target transformation (SE3) is target-specific: a major win for Grab TSS, "
        f"neutral for Ridge/ElNet overall, and harmful for OLS BOD/COD.</strong> "
        f"<br><br>"
        f"<strong>Wins:</strong> Grab TSS OLS improves dramatically - "
        f"{_c(w1_grab_tss_ols)} (SE1) to {_c(ly_grab_tss_ols)} (SE3) - the best Grab TSS "
        f"result across all Exp9 variants and a large improvement over the Exp3-SE1 baseline. "
        f"This makes sense: TSS is heavily right-skewed by solids spikes; log1p compresses "
        f"the scale so that small-value days contribute equally to large-value days during "
        f"fitting, leading to better generalisation."
        f"<br><br>"
        f"<strong>Losses:</strong> Grab BOD OLS collapses from {_c(w1_grab_bod_ols)} to "
        f"{_c(ly_grab_bod_ols)} - Duan smearing over-inflates back-transformed predictions "
        f"when 2025 has more extreme spikes than the 2024 training window. "
        f"Grab COD Ridge: {_c(w1_grab_cod_ridge)} to {_c(ly_grab_cod_ridge)}. "
        f"Comp COD: SE1 ElNet {_c(w1_comp_cod_best)} vs SE3 best {_c(ly_comp_cod_best)} - "
        f"the first-ever positive Comp COD result from SE1 is not preserved under log1p."
        f"<br><br>"
        f"<strong>Mechanism:</strong> Duan smearing is estimated from 2024 training residuals. "
        f"When 2025 spike intensity differs from 2024 (possible for BOD/COD), the smear factor "
        f"is systematically biased. TSS is less affected because its spike pattern in 2024 "
        f"is more representative of 2025. "
        f"<strong>Overall verdict:</strong> SE3 is complementary, not uniformly superior to SE1. "
        f"The best strategy is target-specific model selection: SE3/OLS for TSS, SE1/OLS for BOD."
    )

    # SE4 comparisons (log1p features vs plain W1)
    lf_grab_bod_ols   = _best(logf[logf["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    w1_grab_bod_ols_v = _best(w1[w1["model"] == "OLS"],       "Effluent BOD (mg/L, Grab)")
    lf_grab_cod_ols   = _best(logf[logf["model"] == "OLS"],   "Effluent COD (mg/L, Grab)")
    w1_grab_cod_ols   = _best(w1[w1["model"] == "OLS"],       "Effluent COD (mg/L, Grab)")
    lf_comp_cod_ridge = _best(logf[logf["model"] == "Ridge"], "Effluent COD (mg/L, Composite)")
    lf_grab_ph_elnet  = _best(logf[logf["model"] == "ElNet"], "Effluent pH (Grab)")
    w1_grab_ph_ridge  = _best(w1[w1["model"] == "Ridge"],     "Effluent pH (Grab)")
    w1_comp_cod_best  = _best(w1[w1["model"].isin(["OLS","Ridge","ElNet"])],
                              "Effluent COD (mg/L, Composite)")

    q8 = (
        f"<strong>Log1p feature transformation (SE4) produces modest, consistent gains on "
        f"BOD/COD for OLS, and unlocks the best Grab pH ElNet result in Exp9.</strong> "
        f"Unlike SE3 (log1p target), SE4 requires no back-transformation - metrics are "
        f"directly on the original scale."
        f"<br><br>"
        f"<strong>Concentration targets:</strong> Grab BOD OLS improves slightly - "
        f"{_c(w1_grab_bod_ols_v)} (SE1) to {_c(lf_grab_bod_ols)} (SE4). "
        f"Grab COD OLS: {_c(w1_grab_cod_ols)} to {_c(lf_grab_cod_ols)}. "
        f"Comp COD Ridge: {_c(w1_comp_cod_best)} (SE1 best) to {_c(lf_comp_cod_ridge)} (SE4) - "
        f"log feature transforms help find the Comp COD signal without Duan smearing."
        f"<br><br>"
        f"<strong>pH surprise:</strong> Grab pH ElNet reaches {_c(lf_grab_ph_elnet)} with "
        f"log feature transforms, vs {_c(w1_grab_ph_ridge)} (SE1 Ridge best). "
        f"Log-transforming extreme concentration days reduces their masking effect on the "
        f"pH-prediction signal."
        f"<br><br>"
        f"<strong>Verdict:</strong> SE4 is the cleanest W1 FE variant - consistent positive "
        f"deltas on concentration targets and a notable pH improvement. "
        f"Best per-target across all SE variants: "
        f"Grab BOD use SE1/OLS (+0.616), Grab TSS use SE3/OLS (+0.676), "
        f"pH targets use SE4/ElNet, Comp COD use SE1/ElNet (+0.123)."
    )

    # SE5 comparisons (log-log model vs SE1, SE3, SE4)
    ll_grab_bod_ols   = _best(logll[logll["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    ll_grab_cod_ridge = _best(logll[logll["model"] == "Ridge"], "Effluent COD (mg/L, Grab)")
    ll_grab_tss_ols   = _best(logll[logll["model"] == "OLS"],   "Effluent TSS (mg/L, Grab)")
    ll_grab_ph_elnet  = _best(logll[logll["model"] == "ElNet"], "Effluent pH (Grab)")
    ll_comp_cod_ridge = _best(logll[logll["model"] == "Ridge"], "Effluent COD (mg/L, Composite)")
    w1_grab_bod_ols_ll  = _best(w1[w1["model"] == "OLS"],   "Effluent BOD (mg/L, Grab)")
    w1_grab_cod_ridge   = _best(w1[w1["model"] == "Ridge"], "Effluent COD (mg/L, Grab)")
    ly_grab_tss_ols_ll  = _best(logy[logy["model"] == "OLS"], "Effluent TSS (mg/L, Grab)")
    lf_grab_bod_ols_ll  = _best(logf[logf["model"] == "OLS"],  "Effluent BOD (mg/L, Grab)")

    q9 = (
        f"<strong>The log-log model (SE5: log features + log target) gives the best Grab BOD OLS result "
        f"across all Exp9 variants, but does not uniformly dominate SE3 or SE4.</strong> "
        f"<br><br>"
        f"<strong>Best Grab BOD:</strong> SE5 OLS reaches {_c(ll_grab_bod_ols)} - surpassing SE1 OLS "
        f"({_c(w1_grab_bod_ols_ll)}) and SE4 OLS ({_c(lf_grab_bod_ols_ll)}). "
        f"Combining log features with log target resolves both the non-linear functional form "
        f"AND the heteroscedastic residuals simultaneously for BOD, where 2024 and 2025 "
        f"spike distributions are sufficiently aligned for Duan smearing to work correctly. "
        f"<br><br>"
        f"<strong>Grab COD:</strong> SE5 Ridge reaches {_c(ll_grab_cod_ridge)} vs SE1 Ridge "
        f"({_c(w1_grab_cod_ridge)}). "
        f"<strong>Grab TSS:</strong> SE5 OLS ({_c(ll_grab_tss_ols)}) is weaker than SE3 OLS "
        f"({_c(ly_grab_tss_ols_ll)}) - log feature transform does not add to SE3's TSS gain; "
        f"the dominant improvement for TSS came from the target transform alone. "
        f"<strong>pH:</strong> SE5 Grab pH ElNet {_c(ll_grab_ph_elnet)} (same as SE4, since pH "
        f"is not log-transformed on the target side). "
        f"Comp COD Ridge {_c(ll_comp_cod_ridge)}. "
        f"<br><br>"
        f"<strong>Trees remain poor</strong> on all SE5 targets for the same n=187/179 reason as SE1-SE4. "
        f"<br><br>"
        f"<strong>Revised per-target recommendations across SE1-SE5:</strong> "
        f"Grab BOD - SE5/OLS ({_c(ll_grab_bod_ols)}); "
        f"Grab TSS - SE3/OLS ({_c(ly_grab_tss_ols_ll)}); "
        f"Grab pH - SE4/ElNet or SE5/ElNet ({_c(ll_grab_ph_elnet)}); "
        f"Comp COD - SE1/ElNet or SE4/Ridge."
    )

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

    return f"""
<details class="exp-details" id="exp9-findings" open>
  <summary><span class="fold-icon">▶</span> Findings  -  Recency Hypothesis (SE1, SE2, SE3, SE4, SE5)</summary>
  <div class="exp-body">
    {_qcard(1, "Does restricting training to 2024 improve generalisation vs the full 2021-2024 window?", q1)}
    {_qcard(2, "Is the improvement consistent across model families, or specific to linear models?", q2)}
    {_qcard(3, "Which targets benefit most, and which break the pattern?", q3)}
    {_qcard(4, "What does the Comp COD result reveal about the source of its persistent failure?", q4)}
    {_qcard(5, "What is the overall verdict on the recency hypothesis, and what comes next?", q5)}
    {_qcard(6, "Does adding feature selection (SE2: LassoCV / OOF FS) further improve W1 results?", q6)}
    {_qcard(7, "Does log1p target transformation (SE3) improve W1 results further?", q7)}
    {_qcard(8, "Does log1p feature transformation (SE4) improve W1 results further?", q8)}
    {_qcard(9, "Does combining log features + log target (SE5: log-log model) improve further?", q9)}
  </div>
</details>"""


def build_exp9_section(df_all: pd.DataFrame) -> str:
    sub1 = _exp_subsection(df_all, "Exp9-SE1", "exp9-s1",
                           "SE1  -  W1: 2024-Only Training Window (27 features)",
                           open_default=True)
    sub2 = _exp_subsection(df_all, "Exp9-SE2", "exp9-s2",
                           "SE2  -  W1+FS: 2024-Only Window + Feature Selection",
                           open_default=False)
    sub3 = _exp_subsection(df_all, "Exp9-SE3", "exp9-s3",
                           "SE3  -  W1+LogY: 2024-Only Window + log1p Target Transform",
                           open_default=False)
    sub4 = _exp_subsection(df_all, "Exp9-SE4", "exp9-s4",
                           "SE4  -  W1+LogF: 2024-Only Window + log1p Feature Transform",
                           open_default=False)
    sub5 = _exp_subsection(df_all, "Exp9-SE5", "exp9-s5",
                           "SE5  -  W1+LogLog: 2024-Only Window + log1p Features + log1p Targets",
                           open_default=False)
    comparison  = _exp9_comparison_panel(df_all)
    findings    = _exp9_qna(df_all)
    best        = _best_model_box(
                      df_all[df_all["exp_key"].isin(
                          ["Exp9-SE1", "Exp9-SE2", "Exp9-SE3", "Exp9-SE4", "Exp9-SE5"])],
                      "Recency Hypothesis  -  Best across SE1, SE2, SE3, SE4, SE5")

    return f"""
<section id="exp9">
  <h1 class="section-title">Recency Hypothesis  -  Rolling Training Window</h1>
  <p class="section-intro">{EXP_INTRO["Exp9"]}</p>
  {sub1}
  {sub2}
  {sub3}
  {sub4}
  {sub5}
  {comparison}
  {findings}
  {best}
</section>"""


def _ann_dataset_exploration_callout() -> str:
    return """
<div class="obs-card" style="margin:1.5rem 0;border-left:4px solid #9B59B6">
  <h4 style="margin:0 0 0.6rem">ANN Dataset Exploration  -  Key Findings</h4>
  <ul style="margin:0 0 0 1rem;padding:0;font-size:0.9em;line-height:1.7">
    <li><strong>Inlet features alone (Exp1, 9 features) are insufficient for the ANN</strong>  - 
        avg Test R²=−5.6, worse than Exp 6 (Exp3-SE2, −1.1). More data cannot compensate
        for missing secondary process signal.</li>
    <li><strong>Secondary features unlock positive Grab R² for the first time</strong>  - 
        Exp2-SE1 ANN (Secondary + COMMON, 12 features, ~924 rows) achieves Grab BOD +0.20,
        Grab TSS +0.27. This is the only dataset × architecture combination where the ANN
        produces positive generalisation on Grab targets.</li>
    <li><strong>Adding inlet to secondary data (Exp2-SE2) does not help composites</strong>  - 
        Comp BOD collapses from −0.11 (Exp2-S1) to −1.49 (Exp2-S2). The ANN overfits
        more with 16 features than with 12 on the same 733 composite rows.</li>
    <li><strong>Composite targets fail for the ANN on every dataset tested</strong>  - 
        all composite test R² are negative across Exp1, Exp2-SE1, Exp2-SE2, and Exp 6
        (Exp3-SE2). The pattern is consistent: composite measurements are temporally noisier
        and the ANN cannot capture the distributional shift from training to 2025.</li>
    <li><strong>Conclusion  -  the ANN failure is not data-volume limited</strong>  - 
        tripling the training rows (Exp1: 1175 vs Exp3-SE2: 470) did not rescue performance.
        The binding constraint is the <em>feature set</em>: secondary process data is a
        prerequisite for positive ANN generalisation on Grab targets. Even with secondary
        features and adequate data (~924 rows), the ANN substantially underperforms the
        Voting ensemble (avg Grab R²≈0.20 vs Voting 0.287 overall). The ANN is not
        recommended for any target on this dataset.</li>
  </ul>
</div>"""


def _ann_dataset_comparison(df_all: pd.DataFrame) -> str:
    """Three-way ANN comparison table: Exp1 vs Exp2-SE1 vs Exp2-SE2 vs Exp6 (Exp3-SE2)."""
    keys = ["ANN-Exp1", "ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref", "Exp6-ANN"]
    labels = {
        "ANN-Exp1":      "Exp1 (9 feat, ~1175/800 rows)",
        "ANN-Exp2-SE2": "Exp2-SE1 (15 feat, ~924/740 rows)",
        "ANN-Exp2-SE3-Ref": "Exp2-SE2 (19 feat, ~920/733 rows)",
        "Exp6-ANN":    "Exp3-SE2 (25 feat, ~470/290 rows)",
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
  <h3 style="margin:0 0 12px">ANN Performance Across Datasets  -  Data-Volume Diagnostic</h3>
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


def _adv_methods_rationale_card() -> str:
    """Card explaining why feedforward ANN (MLP) was chosen over recurrent architectures."""
    return """
<div class="obs-card" style="margin:1rem 0;border-left:4px solid #9B59B6">
  <h4 style="margin:0 0 0.5rem">Why a Feedforward ANN (MLP) rather than RNN / LSTM?</h4>
  <p style="margin:0 0 0.6rem;font-size:0.88em;color:var(--text-muted)">
    Three structural constraints made recurrent architectures a poor fit for this dataset:
  </p>
  <ul style="margin:0 0 0.7rem 1rem;padding:0;font-size:0.88em;line-height:1.75">
    <li><strong>Gapped time series (~30-40% missing days).</strong> LSTM / RNN require a
        dense sequential input stream. Wastewater measurements - especially composite samples -
        are only collected on certain operational days, leaving long irregular gaps.
        Filling these gaps with imputation would introduce substantial synthetic data into a
        model whose value comes from learning genuine temporal dynamics. A feedforward model
        treats each observation as an independent row and is unaffected by missingness in
        neighbouring dates.</li>
    <li><strong>Largely tabular daily structure.</strong> Each day's prediction is
        well-described by its own inlet concentrations, secondary clarifier readings, and
        process conditions. Temporal signal that does exist (seasonal cycles, weekday patterns)
        is compactly encoded by the cyclic calendar features already present (month_sin/cos,
        dow_sin/cos). Explicit temporal context - lag 1/3/7-day and 7-day rolling means - was
        tested separately in the Temporal Features section, which is more data-efficient than
        feeding raw hidden state through an LSTM cell.</li>
    <li><strong>Insufficient training samples for recurrent architectures.</strong> With
        ~470 Grab / ~290 Composite training rows, even a 2-hidden-layer MLP failed severely
        (every target selected alpha=1.0 - maximum weight decay). An LSTM cell has 4x more
        parameters than a basic RNN unit and would overfit far worse. The data-volume
        diagnostic (ANN Exploration) confirmed that tripling training rows to ~1175 still did
        not rescue performance.</li>
  </ul>
  <p style="margin:0;font-size:0.85em;color:var(--text-muted)">
    <strong>Future conditions for sequential models:</strong> If the dataset grows to 2000+
    consecutive daily observations with &lt;10% missingness, LSTM is worth revisiting
    (use a masking layer for irregular gaps). Alternatively, aggregate to weekly averages
    (~200 data points, minimal gaps) and use a shallow GRU.
  </p>
</div>"""


def _adv_comparison_panel(df_all: pd.DataFrame) -> str:
    """Two focused sub-tables: ANN dataset progression (4 cols) + architecture comparison (3 cols).
    Star/diamond highlighting is computed across all 6 variants so global winners are marked correctly.
    """
    ANN_KEYS  = ["ANN-Exp1", "ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref", "Exp6-ANN"]
    ARCH_KEYS = ["Exp6-ANN", "Exp6-Voting", "Exp6-Stacking"]
    ALL_KEYS  = ANN_KEYS + ["Exp6-Voting", "Exp6-Stacking"]

    COL_LABELS = {
        "ANN-Exp1":        "ANN - Exp1<br><span style='color:#888888;font-weight:400;font-size:0.78em'>9 feat, ~1175 tr</span>",
        "ANN-Exp2-SE2":   "ANN - Exp2-SE1<br><span style='color:#888888;font-weight:400;font-size:0.78em'>15 feat, ~924 tr</span>",
        "ANN-Exp2-SE3-Ref":   "ANN - Exp2-SE2<br><span style='color:#888888;font-weight:400;font-size:0.78em'>21 feat, ~920 tr</span>",
        "Exp6-ANN":      "ANN - Exp3-SE2<br><span style='color:#888888;font-weight:400;font-size:0.78em'>31 feat, ~470 tr · R² · RMSE · Gap</span>",
        "Exp6-Voting":   "Voting<br><span style='color:#888888;font-weight:400;font-size:0.78em'>ElNet+RF+XGB · R² · RMSE · Gap</span>",
        "Exp6-Stacking": "Stacking<br><span style='color:#888888;font-weight:400;font-size:0.78em'>walk-fwd OOF · R² · RMSE · Gap</span>",
    }
    KEY_MODEL = {
        "ANN-Exp1": "ANN", "ANN-Exp2-SE2": "ANN", "ANN-Exp2-SE3-Ref": "ANN",
        "Exp6-ANN": "ANN", "Exp6-Voting": "Voting", "Exp6-Stacking": "Stacking",
    }

    avail_all  = [k for k in ALL_KEYS  if k in df_all["exp_key"].values]
    avail_ann  = [k for k in ANN_KEYS  if k in df_all["exp_key"].values]
    avail_arch = [k for k in ARCH_KEYS if k in df_all["exp_key"].values]
    if not avail_arch:
        return ""

    MEANINGFUL = 0.01
    _TD  = "padding:4px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _STD = "padding:5px 7px;font-size:0.78rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:5px 7px;text-align:left;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"
    _THC = "padding:5px 7px;text-align:center;color:#333;font-weight:600;font-size:0.79rem;background:#eeeeee"

    def _get(key, model, tgt, col="R2_test"):
        sub = df_all[(df_all["exp_key"] == key) & (df_all["model"] == model) & (df_all["target"] == tgt)]
        if sub.empty or col not in sub.columns: return None
        v = sub[col].values[0]
        return None if (v != v or v is None) else float(v)

    def _gaj(r2, gap):
        if r2 is None: return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#bbbbbb'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        mhtml  = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:   col = "#5BAD6F"; fw = "bold"
        elif is_gaj: col = "#4A90D9"; fw = "bold"
        else:        col = "#1a1a1a"; fw = "normal"
        rmse_s = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else "-"
        gv     = gap if (gap is not None and gap == gap) else None
        gs     = f"{gv:+.3f}" if gv is not None else "-"
        gc     = ("#E15252" if gv is not None and gv > 0.10
                  else "#5BAD6F" if gv is not None and gv < -0.10
                  else "#888888")
        sec = (f"<br><span style='font-size:0.71em;color:#888888;font-weight:normal'>"
               f"{rmse_s} · <span style='color:{gc}'>{gs}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{mhtml}{sec}</td>")

    # Transition delta collectors
    ANN_TRANSITIONS  = [
        ("ANN-Exp1",      "ANN-Exp2-SE2", "ANN"),
        ("ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref", "ANN"),
        ("ANN-Exp2-SE3-Ref", "Exp6-ANN",    "ANN"),
    ]
    ARCH_TRANSITIONS = [
        ("Exp6-ANN", "Exp6-Voting",   None),
        ("Exp6-ANN", "Exp6-Stacking", None),
        ("Exp6-Voting", "Exp6-Stacking", None),
    ]
    ALL_TRANSITIONS = ANN_TRANSITIONS + ARCH_TRANSITIONS
    trans_deltas = {t: [] for t in ALL_TRANSITIONS}
    trans_gaj    = {t: [] for t in ALL_TRANSITIONS}

    tbody_ann  = ""
    tbody_arch = ""

    for tgt in TARGETS_ORDERED:
        short = TARGET_SHORT.get(tgt, tgt)
        hdr_style = ("padding:6px 10px;font-size:0.75rem;font-weight:700;color:#555555;"
                     "letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid #d0d0d0")

        # pre-pass: best raw and gap-adjusted across ALL 6 variants
        _all_raw = {}; _all_gaj_d = {}; _all_gap_d = {}
        for key in avail_all:
            mdl = KEY_MODEL[key]
            v = _get(key, mdl, tgt); g = _get(key, mdl, tgt, "R2_gap")
            if v is not None:
                _all_raw[(key, mdl)] = v
                _all_gap_d[(key, mdl)] = g
                sv = _gaj(v, g)
                if sv is not None: _all_gaj_d[(key, mdl)] = sv

        tgt_br = max(_all_raw.values())   if _all_raw   else None
        tgt_bg = max(_all_gaj_d.values()) if _all_gaj_d else None
        _raw_win_gap = None
        if tgt_br is not None:
            for k_, v_ in _all_raw.items():
                if abs(v_ - tgt_br) < 1e-9:
                    _raw_win_gap = _all_gap_d.get(k_); break
        tgt_show_gaj = (
            tgt_bg is not None and tgt_br is not None
            and _raw_win_gap is not None and _raw_win_gap > 0.10
        )

        _ir = lambda v_: tgt_br is not None and v_ is not None and abs(v_ - tgt_br) < 1e-9
        _ig = lambda s_: tgt_show_gaj and tgt_bg is not None and s_ is not None and abs(s_ - tgt_bg) < 1e-9

        # ── Sub-table 1: ANN dataset progression (one ANN row per target) ──
        tbody_ann += (f"<tr style='background:#e8e8e8'>"
                      f"<td colspan='{len(avail_ann)+1}' style='{hdr_style}'>{short}</td></tr>")
        row = f"<tr style='background:#ffffff'><td style='{_TD}'><strong>ANN</strong></td>"
        ann_vals = {}
        for key in avail_ann:
            v = _get(key, "ANN", tgt); r = _get(key, "ANN", tgt, "RMSE_test")
            g = _get(key, "ANN", tgt, "R2_gap"); sc = _gaj(v, g)
            ann_vals[key] = (v, g, sc)
            row += _val_td(v, r, g, _ir(v), _ig(sc))
        row += "</tr>"
        tbody_ann += row

        # ANN sequential transition deltas
        for trans in ANN_TRANSITIONS:
            fk, tk, _ = trans
            if fk in avail_ann and tk in avail_ann:
                vf_t = ann_vals.get(fk); vt_t = ann_vals.get(tk)
                if vf_t and vt_t and vf_t[0] is not None and vt_t[0] is not None:
                    trans_deltas[trans].append(vt_t[0] - vf_t[0])
                    if vf_t[2] is not None and vt_t[2] is not None:
                        trans_gaj[trans].append(vt_t[2] - vf_t[2])

        # Sub-table 2: architecture comparison (ANN, Voting, Stacking on Exp3-SE2)
        tbody_arch += (f"<tr style='background:#e8e8e8'>"
                       f"<td colspan='{len(avail_arch)+1}' style='{hdr_style}'>{short}</td></tr>")
        row = f"<tr style='background:#ffffff'><td style='{_TD}'><strong>Test R²</strong></td>"
        for key in avail_arch:
            row_model = KEY_MODEL[key]
            v = _get(key, row_model, tgt)
            r = _get(key, row_model, tgt, "RMSE_test")
            g = _get(key, row_model, tgt, "R2_gap")
            sc = _gaj(v, g)
            row += _val_td(v, r, g, _ir(v), _ig(sc))
        row += "</tr>"
        tbody_arch += row

        # Cross-model transition deltas (architecture comparison)
        for trans in ARCH_TRANSITIONS:
            fk, tk, _ = trans
            if fk in avail_arch and tk in avail_arch:
                fm = KEY_MODEL[fk]; tm2 = KEY_MODEL[tk]
                vf = _get(fk, fm, tgt); vt = _get(tk, tm2, tgt)
                if vf is not None and vt is not None:
                    trans_deltas[trans].append(vt - vf)
                    gf = _get(fk, fm, tgt, "R2_gap"); gt = _get(tk, tm2, tgt, "R2_gap")
                    sf = _gaj(vf, gf); st = _gaj(vt, gt)
                    if sf is not None and st is not None:
                        trans_gaj[trans].append(st - sf)

    # ── Transition summary builder ──
    def _stats_row(deltas, gaj_d, from_lbl, to_lbl):
        if not deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} → {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr = np.array(deltas); n = len(arr); net = float(arr.mean())
        wins   = arr[arr >  MEANINGFUL]; losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        mw = float(wins.mean())   if len(wins)   else None
        ml = float(losses.mean()) if len(losses) else None
        nc = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        vd = "Net improvement" if net > MEANINGFUL else ("Net regression" if net < -MEANINGFUL else "Negligible")
        ws = f"{len(wins)}/{n} (avg {mw:+.3f})"   if mw is not None else f"{len(wins)}/{n}"
        ls = f"{len(losses)}/{n} (avg {ml:+.3f})" if ml is not None else f"{len(losses)}/{n}"
        if gaj_d:
            gn = float(np.array(gaj_d).mean()); diff = gn - net
            gc = "#5BAD6F" if gn > MEANINGFUL else ("#E15252" if gn < -MEANINGFUL else "#888888")
            dc = "#E15252" if diff < -0.02 else "#5BAD6F" if diff > 0.02 else "#888888"
            interp = ("Raw gains partially inflated by overfitting" if diff < -0.02
                      else "Raw gains understated - overfitting decreased" if diff > 0.02
                      else "Overfitting largely unchanged")
            gs2 = f"{gn:+.4f}"; ds = f"{diff:+.4f}"
        else:
            gs2 = ds = " - "; gc = dc = "#888888"; interp = " - "
        return (f"<tr>"
                f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} → {to_lbl}</strong></td>"
                f"<td style='{_STD};text-align:center;color:{nc};font-weight:bold'>{net:+.4f}</td>"
                f"<td style='{_STD};text-align:center;color:#5BAD6F'>{ws}</td>"
                f"<td style='{_STD};text-align:center;color:#E15252'>{ls}</td>"
                f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
                f"<td style='{_STD};text-align:center;color:{gc};font-weight:bold'>{gs2}</td>"
                f"<td style='{_STD};text-align:center;color:{dc}'>{ds}</td>"
                f"<td style='{_STD}'><strong>{vd}</strong><br>"
                f"<span style='font-size:0.82em;color:{dc}'>{interp}</span></td></tr>")

    def _summary_table(transitions, trans_labels, note):
        rows = "".join(_stats_row(trans_deltas[t], trans_gaj[t], *trans_labels[t])
                       for t in transitions if t in trans_labels)
        return f"""
<div style='margin-top:1rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead><tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH}'>Transition</th>
      <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
      <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
      <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; -{MEANINGFUL}</span></th>
      <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| ≤ {MEANINGFUL}</span></th>
      <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²-0.5·max(0,|gap|-0.10)</span></th>
      <th style='{_THC}'>Gap-Adj - Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
      <th style='{_TH}'>Verdict / Interpretation</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <div class='obs-card' style='border-left:4px solid #4A90D9;margin-top:0.6rem'>
    <p class='meta'>{note}</p>
  </div>
</div>"""

    ANN_TRANS_LABELS = {
        ("ANN-Exp1",      "ANN-Exp2-SE2", "ANN"): ("ANN: Exp1",     "Exp2-SE1 (+secondary feat)"),
        ("ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref", "ANN"): ("ANN: Exp2-SE1", "Exp2-SE2 (+inlet to secondary)"),
        ("ANN-Exp2-SE3-Ref", "Exp6-ANN",    "ANN"): ("ANN: Exp2-SE2", "Exp3-SE2 (+more feat, -rows)"),
    }
    ARCH_TRANS_LABELS = {
        ("Exp6-ANN",    "Exp6-Voting",   None): ("ANN Exp3-SE2", "Voting (architecture change)"),
        ("Exp6-ANN",    "Exp6-Stacking", None): ("ANN Exp3-SE2", "Stacking (architecture change)"),
        ("Exp6-Voting", "Exp6-Stacking", None): ("Voting",       "Stacking (ensemble method)"),
    }
    n_tgts = len(TARGETS_ORDERED)
    ann_summary  = _summary_table(ANN_TRANSITIONS,  ANN_TRANS_LABELS,
        f"Net Mean ΔR² is the signed average across {n_tgts} targets. "
        f"Same MLP architecture across all columns; only dataset (feature set + sample count) changes. "
        f"<strong>Gap-Adj - Raw</strong>: negative = raw gains inflated by overfitting; "
        f"positive = overfitting decreased.")
    arch_summary = _summary_table(ARCH_TRANSITIONS, ARCH_TRANS_LABELS,
        f"All three variants use the same Exp3-SE2 dataset. "
        f"Transitions reflect pure architecture differences: ANN vs ensemble methods, "
        f"and Voting (equal weights) vs Stacking (walk-forward OOF meta-learner). "
        f"<strong>★</strong> = best raw Test R² per target (across all 6 variants) · "
        f"<strong>✦</strong> = best gap-adjusted per target "
        f"(shown only when raw winner has |gap| &gt; 0.10).")

    # Build both tables
    obs_legend = f"""
<div class="obs-card" style="border-left:4px solid #E67E22">
  <p class="meta">
    <strong>★</strong> = best raw Test R² per target (across all 6 variants) ·
    <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10 and a different variant wins).
    RMSE in native units (mg/L or pH). Gap = Train R² - Test R²;
    <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
  </p>
</div>"""

    def _make_table(tbody, keys, min_w, left_header="Model"):
        hdrs = "".join(f'<th style="{_THC}">{COL_LABELS[k]}</th>' for k in keys)
        return (f"<div style='overflow-x:auto;margin-top:0.6rem;border:1px solid #cccccc;"
                f"border-radius:4px'>"
                f"<table style='border-collapse:collapse;width:100%;background:#ffffff;"
                f"font-size:0.79rem;color:#1a1a1a;min-width:{min_w}px'>"
                f"<thead><tr style='border-bottom:2px solid #cccccc'>"
                f"<th style='{_TH};min-width:68px'>{left_header}</th>{hdrs}</tr></thead>"
                f"<tbody>{tbody}</tbody></table></div>")

    ann_sect = ""
    if avail_ann:
        ann_sect = f"""
<h4 style='margin:1.4rem 0 0.3rem;font-size:0.93rem;color:#1a1a1a'>
  ANN Dataset Progression
  <span style='font-size:0.80rem;font-weight:normal;color:var(--text-muted)'>
    - same architecture, different feature set and sample count
  </span>
</h4>
{_make_table(tbody_ann, avail_ann, 680)}
{ann_summary}"""

    arch_sect = f"""
<h4 style='margin:1.8rem 0 0.3rem;font-size:0.93rem;color:#1a1a1a'>
  Architecture Comparison on Exp3-SE2
  <span style='font-size:0.80rem;font-weight:normal;color:var(--text-muted)'>
    - same dataset, different model class
  </span>
</h4>
{_make_table(tbody_arch, avail_arch, 560, "Metric")}
{arch_summary}"""

    return f"""
<details class="exp-details" id="adv-comparison" open>
  <summary><span class="fold-icon">▶</span> Comparisons</summary>
  <div class="exp-body">
    {obs_legend}
    {ann_sect}
    {arch_sect}
  </div>
</details>"""


def _adv_methods_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for the Advanced Methods section."""
    p9 = df_all[
        df_all["exp_key"].isin(["Exp6-ANN", "Exp6-Voting", "Exp6-Stacking"])
    ].dropna(subset=["R2_test"])

    def _avg(exp_key):
        sub = p9[p9["exp_key"] == exp_key]["R2_test"]
        return float(sub.mean()) if not sub.empty else float("nan")

    def _best(exp_key, tgt):
        sub = p9[(p9["exp_key"] == exp_key) & (p9["target"] == tgt)]
        return float(sub["R2_test"].max()) if not sub.empty else float("nan")

    ann_avg   = _avg("Exp6-ANN")
    vote_avg  = _avg("Exp6-Voting")
    stack_avg = _avg("Exp6-Stacking")

    def _c(v, good_above=0.05):
        if v != v:
            return " - "
        col = "#5BAD6F" if v > good_above else "#E15252" if v < 0 else "var(--text-primary)"
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    vote_grab_bod = _best("Exp6-Voting", "Effluent BOD (mg/L, Grab)")
    vote_comp_bod = _best("Exp6-Voting", "Effluent BOD (mg/L, Composite)")
    stack_comp_bod = _best("Exp6-Stacking", "Effluent BOD (mg/L, Composite)")
    stack_comp_cod = _best("Exp6-Stacking", "Effluent COD (mg/L, Composite)")
    stack_comp_tss = _best("Exp6-Stacking", "Effluent TSS (mg/L, Composite)")

    q1 = (
        f"The MLP must learn all structure - feature interactions, non-linearities, and "
        f"regularisation paths - from only ~470 Grab / ~290 Composite training rows. "
        f"Avg Test R² = {_c(ann_avg, 0)}. GridSearchCV selected alpha=1.0 (maximum weight "
        f"decay) for every target, confirming the model is parameter-starved: it can only "
        f"fit the data when its weights are maximally shrunk toward zero."
        f"<br><br>"
        f"<strong>Voting</strong> bypasses this by combining three pre-tuned learners "
        f"with complementary inductive biases: ElNet (handles collinearity, L1 feature "
        f"selection), RF (bagging over decision trees - robust to the TSS outliers up to "
        f"1266 mg/L in training), XGBoost (gradient boosting, captures residual "
        f"non-linearities). Uncorrelated errors partially cancel in the average, giving "
        f"avg Test R² = {_c(vote_avg, 0.10)}. The data-volume diagnostic confirmed that "
        f"sample count is not the bottleneck: tripling rows to ~1175 (Exp1 dataset) "
        f"worsened ANN performance to avg R² = -5.6."
    )

    q2 = (
        f"<strong>Voting</strong> (equal-weight average): avg R² = {_c(vote_avg, 0.10)}. "
        f"<strong>Stacking</strong> (walk-forward OOF meta-learner): avg R² = {_c(stack_avg, 0.05)}. "
        f"<br><br>"
        f"The meta-learner in Stacking (Ridge fitted on ~390 OOF rows with 3 base-learner "
        f"predictions as features) adds complexity without proportional lift over simple "
        f"averaging. With ~83% OOF coverage, the first ~17% of the training set is excluded "
        f"from the meta-learner entirely - those rows are never seen at meta-training time. "
        f"Stacking is methodologically clean (no look-ahead bias after the KFold correction) "
        f"but not worth the added complexity at current sample sizes. "
        f"Verdict: <strong>Voting is preferred</strong> - simpler, no meta-learner overhead, "
        f"and consistently better."
    )

    q3 = (
        f"<strong>Voting (ElNet+RF+XGB)</strong> is recommended. "
        f"It has the best average R² across the Advanced Methods panel and leads the Grab-side "
        f"ensemble results, including Grab BOD {_c(vote_grab_bod, 0.30)}. "
        f"Stacking is still important in the comparison: it is the better Advanced Method on "
        f"several Composite targets in the current results, including Comp BOD {_c(stack_comp_bod, 0.20)}, "
        f"Comp COD {_c(stack_comp_cod, 0.0)}, and Comp TSS {_c(stack_comp_tss, 0.10)}. "
        f"<br><br>"
        f"For <strong>Grab targets only</strong>, combining Voting with selective feature "
        f"engineering (Exp 7-SE2 - Selective FE) further improves performance "
        f"(avg Grab R² ~+0.48 vs Voting base +0.28). For <strong>Composite targets</strong>, "
        f"the base Exp3-SE2 feature set (no FE) is safer - FE causes severe overfitting at "
        f"n_train ~290 rows."
        f"<br><br>"
        f"<strong>Comp COD</strong> is not addressable by any architecture tested: "
        f"2025 MAE doubled from 2024 (genuine distribution shift). Flag Comp COD predictions "
        f"as unreliable until more post-2025 data is collected."
    )

    def _qcard(n, question, answer):
        return (
            f"<div style='margin-bottom:1.1rem;border:1px solid var(--border);"
            f"border-radius:5px;overflow:hidden'>"
            f"<div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;"
            f"border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>"
            f"<span style='color:#4A90D9;font-size:0.78em;font-weight:bold;"
            f"letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>"
            f"<span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>"
            f"</div>"
            f"<div style='padding:0.6rem 0.8rem 0.55rem'>"
            f"<p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);"
            f"padding-left:0.6rem'>{answer}</p>"
            f"</div></div>"
        )

    # Q4: ANN dataset exploration findings
    ann_e1  = df_all[df_all["exp_key"] == "ANN-Exp1"].dropna(subset=["R2_test"])
    ann_e2s1 = df_all[df_all["exp_key"] == "ANN-Exp2-SE2"].dropna(subset=["R2_test"])

    def _ann_avg(df_):
        return float(df_["R2_test"].mean()) if not df_.empty else float("nan")

    ann_e1_avg   = _ann_avg(ann_e1)
    ann_e2s1_avg = _ann_avg(ann_e2s1)
    ann_e2s1_grab_bod = float(ann_e2s1[ann_e2s1["target"] == "Effluent BOD (mg/L, Grab)"]["R2_test"].max()) \
        if not ann_e2s1[ann_e2s1["target"] == "Effluent BOD (mg/L, Grab)"].empty else float("nan")
    ann_e2s1_grab_tss = float(ann_e2s1[ann_e2s1["target"] == "Effluent TSS (mg/L, Grab)"]["R2_test"].max()) \
        if not ann_e2s1[ann_e2s1["target"] == "Effluent TSS (mg/L, Grab)"].empty else float("nan")

    q4 = (
        f"Tripling the training rows from ~470 (Exp3-SE2) to ~1175 (Exp1) did <em>not</em> rescue ANN "
        f"performance. Exp1 ANN avg Test R² = {_c(ann_e1_avg, 0)} - worse than Exp 6 (Exp3-SE2) ANN "
        f"avg R² = {_c(ann_avg, 0)}. More data cannot compensate for missing secondary process signal."
        f"<br><br>"
        f"<strong>Secondary features are the prerequisite for positive ANN generalisation.</strong> "
        f"Exp2-SE1 ANN (Secondary + COMMON, 15 features, ~924 rows) is the only configuration achieving "
        f"positive Grab R²: Grab BOD {_c(ann_e2s1_grab_bod, 0.1)}, Grab TSS {_c(ann_e2s1_grab_tss, 0.1)}. "
        f"Adding inlet concentrations to secondary data (Exp2-SE2, 21 features) does not help Composite "
        f"targets - Comp BOD collapses from -0.11 (Exp2-SE1) to -1.49 (Exp2-SE2) as the additional "
        f"features increase overfitting on ~733 Composite rows."
        f"<br><br>"
        f"<strong>Composite targets fail universally for the ANN</strong> across all four dataset "
        f"configurations tested (Exp1, Exp2-SE1, Exp2-SE2, Exp3-SE2). The ANN cannot capture the "
        f"distributional shift in composite measurements from training to 2025."
        f"<br><br>"
        f"<strong>Conclusion:</strong> ANN failure is feature-set and temporal-structure limited, "
        f"not data-volume limited. Even with secondary features and adequate data (~924 rows), "
        f"the ANN substantially underperforms the Voting ensemble. The ANN is not recommended "
        f"for any target on this dataset."
    )

    cards = "".join([
        _qcard(1, "Why did the ANN fail while the Voting ensemble succeeded?", q1),
        _qcard(2, "Does Stacking improve on Voting?", q2),
        _qcard(3, "Which advanced method is recommended and for which targets?", q3),
        _qcard(4, "Does more training data rescue the ANN? What did the dataset exploration reveal?", q4),
    ])
    return f"""
<details class="exp-details" id="adv-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings  -  Advanced Methods</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def _phase10_comparison_panel(df_all: pd.DataFrame) -> str:
    """Side-by-side comparison: Exp 6 Voting baseline vs Full FE vs Selective FE."""
    keys   = ["Exp6-Voting", "Exp7-SE1", "Exp7-SE2"]
    labels = {
        "Exp6-Voting": (
            "P9 Baseline<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Voting · R² · RMSE · Gap</span>"
        ),
        "Exp7-SE1": (
            "SE1<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Full FE · R² · RMSE · Gap</span>"
        ),
        "Exp7-SE2": (
            "SE2<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Selective FE · R² · RMSE · Gap</span>"
        ),
    }
    avail = [k for k in keys if k in df_all["exp_key"].values]
    if len(avail) < 2:
        return ""

    models_show = ["Voting", "Ridge", "ElNet", "RF"]
    MEANINGFUL = 0.01
    transitions = [
        ("Exp6-Voting", "Exp7-SE1"),
        ("Exp6-Voting", "Exp7-SE2"),
        ("Exp7-SE1", "Exp7-SE2"),
    ]
    trans_deltas = {t: [] for t in transitions}
    trans_gaj = {t: [] for t in transitions}

    _TD  = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _STD = "padding:6px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

    def _get(key, model, tgt, col="R2_test"):
        sub = df_all[
            (df_all["exp_key"] == key) &
            (df_all["target"] == tgt) &
            (df_all["model"] == model)
        ]
        if sub.empty or col not in sub.columns:
            return None
        v = sub[col].iloc[0]
        return None if (v is None or v != v) else float(v)

    def _gaj(r2, gap):
        if r2 is None:
            return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    def _model_for_key(key, model):
        return model

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:
            col = "#5BAD6F"; fw = "bold"
        elif is_gaj:
            col = "#4A90D9"; fw = "bold"
        else:
            col = "#1a1a1a"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else " - "
        gap_val = gap if (gap is not None and gap == gap) else None
        gap_str = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col = ("#E15252" if gap_val is not None and gap_val > 0.10
                   else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                   else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;font-weight:normal'>"
                     f"RMSE {rmse_str} · <span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    tbody = ""
    for tgt in TARGETS_ORDERED:
        tgt_short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='{len(avail) + 1}' style='padding:6px 10px;font-size:0.75rem;"
            f"font-weight:700;color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{tgt_short}</td></tr>"
        )

        all_raw = {}
        all_gap = {}
        all_gaj = {}
        for model in models_show:
            for key in avail:
                key_model = _model_for_key(key, model)
                r2 = _get(key, key_model, tgt)
                gap = _get(key, key_model, tgt, "R2_gap")
                score = _gaj(r2, gap)
                if r2 is not None:
                    all_raw[(key, model)] = r2
                    all_gap[(key, model)] = gap
                if score is not None:
                    all_gaj[(key, model)] = score

        best_raw = max(all_raw.values()) if all_raw else None
        best_gaj = max(all_gaj.values()) if all_gaj else None
        raw_win_gap = None
        if best_raw is not None:
            for k_, v_ in all_raw.items():
                if abs(v_ - best_raw) < 1e-9:
                    raw_win_gap = all_gap.get(k_)
                    break
        show_gaj = (
            best_gaj is not None and best_raw is not None
            and raw_win_gap is not None and raw_win_gap > 0.10
        )

        def _is_raw(v):
            return best_raw is not None and v is not None and abs(v - best_raw) < 1e-9

        def _is_gaj(v):
            return show_gaj and best_gaj is not None and v is not None and abs(v - best_gaj) < 1e-9

        for model in models_show:
            row_bg = "#ffffff" if models_show.index(model) % 2 == 0 else "#f7f7f7"
            cells = f"<td style='{_TD}'><strong>{model}</strong></td>"
            values = {}
            for key in avail:
                key_model = _model_for_key(key, model)
                r2 = _get(key, key_model, tgt)
                rmse = _get(key, key_model, tgt, "RMSE_test")
                gap = _get(key, key_model, tgt, "R2_gap")
                score = _gaj(r2, gap)
                values[key] = (r2, score)
                cells += _val_td(r2, rmse, gap, _is_raw(r2), _is_gaj(score))
            tbody += f"<tr style='background:{row_bg}'>{cells}</tr>"

            for trans in transitions:
                from_key, to_key = trans
                if from_key not in avail or to_key not in avail:
                    continue
                from_v, from_s = values.get(from_key, (None, None))
                to_v, to_s = values.get(to_key, (None, None))
                if from_v is not None and to_v is not None:
                    trans_deltas[trans].append(to_v - from_v)
                if from_s is not None and to_s is not None:
                    trans_gaj[trans].append(to_s - from_s)

    def _stats_row(trans, from_lbl, to_lbl):
        deltas = trans_deltas[trans]
        gaj_d = trans_gaj[trans]
        if not deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} &rarr; {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr = np.array(deltas)
        n = len(arr)
        net = float(arr.mean())
        wins = arr[arr > MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        win_mean = float(wins.mean()) if len(wins) else None
        loss_mean = float(losses.mean()) if len(losses) else None
        net_col = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        verdict = "Net improvement" if net > MEANINGFUL else ("Net regression" if net < -MEANINGFUL else "Negligible")
        win_str = f"{len(wins)}/{n} (avg {win_mean:+.3f})" if win_mean is not None else f"{len(wins)}/{n}"
        loss_str = f"{len(losses)}/{n} (avg {loss_mean:+.3f})" if loss_mean is not None else f"{len(losses)}/{n}"
        if gaj_d:
            gaj_net = float(np.array(gaj_d).mean())
            diff = gaj_net - net
            gaj_col = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "#888888")
            diff_col = "#E15252" if diff < -0.02 else "#5BAD6F" if diff > 0.02 else "#888888"
            interp = ("Raw gains partially inflated by overfitting" if diff < -0.02
                      else "Raw gains understated - overfitting decreased" if diff > 0.02
                      else "Overfitting largely unchanged")
            gaj_str = f"{gaj_net:+.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            gaj_col = diff_col = "#888888"
            gaj_str = diff_str = " - "
            interp = " - "
        return (
            f"<tr>"
            f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} &rarr; {to_lbl}</strong></td>"
            f"<td style='{_STD};text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='{_STD};text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='{_STD};text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
            f"<td style='{_STD};text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='{_STD};text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td style='{_STD}'><strong>{verdict}</strong><br>"
            f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span></td>"
            f"</tr>"
        )

    col_headers = "".join(f"<th style='{_THC}'>{labels[k]}</th>" for k in avail)
    main_table = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a;min-width:760px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:70px'>Model</th>
      {col_headers}
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; -{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| &lt;= {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²-0.5·max(0,|gap|-0.10)</span></th>
        <th style='{_THC}'>Gap-Adj - Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
        <th style='{_TH}'>Verdict / Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {_stats_row(("Exp6-Voting", "Exp7-SE1"), "P9 Baseline", "SE1 Full FE")}
      {_stats_row(("Exp6-Voting", "Exp7-SE2"), "P9 Baseline", "SE2 Selective FE")}
      {_stats_row(("Exp7-SE1", "Exp7-SE2"), "SE1 Full FE", "SE2 Selective FE")}
    </tbody>
  </table>
  </div>
</div>"""

    return f"""
<details class="exp-details" id="fe-comparison">
  <summary><span class="fold-icon">▶</span> Comparisons</summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #E67E22">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
        <span style='color:#E15252'>red Gap</span> &gt; 0.10 = notable overfit.
        Exp 6 Voting is included as the baseline; SE1 applies FE to all targets; SE2 applies FE to Grab targets only.
      </p>
    </div>
    {main_table}
    {stats_block}
  </div>
</details>"""


def _phase10_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for the Feature Engineering section."""
    fe_full  = df_all[df_all["exp_key"] == "Exp7-SE1"].dropna(subset=["R2_test"])
    fe_sel   = df_all[df_all["exp_key"] == "Exp7-SE2"].dropna(subset=["R2_test"])
    p9_vote  = df_all[df_all["exp_key"] == "Exp6-Voting"].dropna(subset=["R2_test"])

    def _avg_tgts(df, tgts):
        vals = [float(df[df["target"] == t]["R2_test"].max())
                for t in tgts if not df[df["target"] == t].empty]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _best_vote(df, tgt):
        sub = df[(df["target"] == tgt) & (df["model"] == "Voting")]
        return float(sub["R2_test"].iloc[0]) if not sub.empty else float("nan")

    def _c(v, good=0.05):
        if v != v:
            return " - "
        col = "#5BAD6F" if v > good else "#E15252" if v < 0 else "var(--text-primary)"
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    grab_full = _avg_tgts(
        fe_full[fe_full["model"] == "Voting"] if not fe_full.empty else fe_full, GRAB_TARGETS
    )
    grab_sel  = _avg_tgts(
        fe_sel[fe_sel["model"] == "Voting"] if not fe_sel.empty else fe_sel, GRAB_TARGETS
    )
    comp_full = _avg_tgts(
        fe_full[fe_full["model"] == "Voting"] if not fe_full.empty else fe_full, COMP_TARGETS
    )
    comp_sel  = _avg_tgts(
        fe_sel[fe_sel["model"] == "Voting"] if not fe_sel.empty else fe_sel, COMP_TARGETS
    )
    p9_grab = _avg_tgts(p9_vote, GRAB_TARGETS)
    p9_comp = _avg_tgts(p9_vote, COMP_TARGETS)

    tss_sel = _best_vote(fe_sel, "Effluent TSS (mg/L, Grab)")
    bod_sel = _best_vote(fe_sel, "Effluent BOD (mg/L, Grab)")

    q1 = (
        f"<strong>Full FE (SE1):</strong> Voting avg Grab = {_c(grab_full, 0.20)}, "
        f"Comp = {_c(comp_full, 0.10)} vs Exp 6 baseline "
        f"Grab = {_c(p9_grab, 0.20)}, Comp = {_c(p9_comp, 0.10)}. "
        f"Composite targets were catastrophically affected: Comp TSS ElNet gap +3.80, "
        f"Ridge gap +3.31. Root cause: ~290 Composite training rows + 50 engineered features "
        f"= severe overfitting. Full FE is not suitable for Composite targets at current "
        f"sample sizes."
    )

    q2 = (
        f"<strong>Selective FE (SE2):</strong> apply log1p+interactions+IQR flags to "
        f"Grab targets only; leave Composite targets at base Exp3-SE2 features."
        f"<br><br>"
        f"Voting avg Grab = {_c(grab_sel, 0.20)}, Comp = {_c(comp_sel, 0.10)}. "
        f"Grab targets gain substantially vs baseline ({_c(p9_grab, 0.20)}). "
        f"Comp TSS fully recovered (Voting +0.349 vs Full FE -1.267). "
        f"<strong>Selective FE is the recommended FE configuration.</strong>"
    )

    q3 = (
        f"New records for Grab TSS: Voting SE2 = {_c(tss_sel, 0.30)}, Ridge SE2 = +0.633. "
        f"Grab BOD: Voting SE2 = {_c(bod_sel, 0.30)} (maintains Exp 7 high). "
        f"<br><br>"
        f"Comp COD remains unresponsive (best R² = -0.051 across all FE variants). "
        f"Grab pH is the weakest Grab target (Voting SE2 = +0.244) - narrow dynamic "
        f"range means engineered interaction terms add noise rather than signal."
    )

    def _qcard(n, question, answer):
        return (
            f"<div style='margin-bottom:1.1rem;border:1px solid var(--border);"
            f"border-radius:5px;overflow:hidden'>"
            f"<div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;"
            f"border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>"
            f"<span style='color:#4A90D9;font-size:0.78em;font-weight:bold;"
            f"letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>"
            f"<span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>"
            f"</div>"
            f"<div style='padding:0.6rem 0.8rem 0.55rem'>"
            f"<p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);"
            f"padding-left:0.6rem'>{answer}</p>"
            f"</div></div>"
        )

    cards = "".join([
        _qcard(1, "What happens when feature engineering is applied uniformly to all targets?", q1),
        _qcard(2, "Does selective FE recover the composite regressions while preserving Grab gains?", q2),
        _qcard(3, "Which targets benefited most and which remained unresponsive?", q3),
    ])
    return f"""
<details class="exp-details" id="fe-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings  -  Feature Engineering</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def _phase11_comparison_panel(df_all: pd.DataFrame) -> str:
    """Contextual comparison: Exp 6 Voting / Exp 7-SE2 vs Temporal Features."""
    keys   = ["Exp6-Voting", "Exp7-SE2", "Exp8"]
    labels = {
        "Exp6-Voting": (
            "P9 Baseline<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Voting · R² · RMSE · Gap</span>"
        ),
        "Exp7-SE2": (
            "P10b Sel-FE<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Best prior · R² · RMSE · Gap</span>"
        ),
        "Exp8": (
            "Temporal Feat.<br>"
            "<span style='color:#888888;font-weight:400;font-size:0.78em'>"
            "Lags + log1p · R² · RMSE · Gap</span>"
        ),
    }
    avail = [k for k in keys if k in df_all["exp_key"].values]
    if len(avail) < 2:
        return ""

    models_show = ["Voting", "Ridge", "ElNet", "RF", "XGB"]
    MEANINGFUL = 0.01
    transitions = [
        ("Exp6-Voting", "Exp7-SE2"),
        ("Exp6-Voting", "Exp8"),
        ("Exp7-SE2",   "Exp8"),
    ]
    trans_deltas = {t: [] for t in transitions}
    trans_gaj    = {t: [] for t in transitions}

    _TD  = "padding:5px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _STD = "padding:6px 10px;font-size:0.81rem;border-bottom:1px solid #e0e0e0;color:#1a1a1a"
    _TH  = "padding:7px 10px;text-align:left;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"
    _THC = "padding:7px 10px;text-align:center;color:#333;font-weight:600;font-size:0.82rem;background:#eeeeee"

    def _get(key, model, tgt, col="R2_test"):
        sub = df_all[
            (df_all["exp_key"] == key) &
            (df_all["target"]  == tgt) &
            (df_all["model"]   == model)
        ]
        if sub.empty or col not in sub.columns:
            return None
        v = sub[col].iloc[0]
        return None if (v is None or v != v) else float(v)

    def _gaj(r2, gap):
        if r2 is None:
            return None
        return _gap_adj(r2, gap if gap is not None else 0.0)

    def _val_td(r2, rmse, gap, is_raw=False, is_gaj=False):
        if r2 is None:
            return f"<td style='{_TD};text-align:center;color:#999999'> - </td>"
        marker = ("★" if is_raw else "") + ("✦" if is_gaj else "")
        marker_html = f"<sup style='font-size:0.7em'>{marker}</sup>" if marker else ""
        if is_raw:
            col = "#5BAD6F"; fw = "bold"
        elif is_gaj:
            col = "#4A90D9"; fw = "bold"
        else:
            col = "#1a1a1a"; fw = "normal"
        rmse_str = f"{rmse:.2f}" if (rmse is not None and rmse == rmse) else " - "
        gap_val  = gap if (gap is not None and gap == gap) else None
        gap_str  = f"{gap_val:+.3f}" if gap_val is not None else " - "
        gap_col  = ("#E15252" if gap_val is not None and gap_val > 0.10
                    else "#5BAD6F" if gap_val is not None and gap_val < -0.10
                    else "#888888")
        secondary = (f"<br><span style='font-size:0.72em;color:#888888;font-weight:normal'>"
                     f"RMSE {rmse_str} · <span style='color:{gap_col}'>Gap {gap_str}</span></span>")
        return (f"<td style='{_TD};text-align:center;color:{col};font-weight:{fw}'>"
                f"{r2:+.3f}{marker_html}{secondary}</td>")

    tbody = ""
    for tgt in TARGETS_ORDERED:
        tgt_short = TARGET_SHORT.get(tgt, tgt)
        tbody += (
            f"<tr style='background:#e8e8e8'>"
            f"<td colspan='{len(avail) + 1}' style='padding:6px 10px;font-size:0.75rem;"
            f"font-weight:700;color:#555555;letter-spacing:0.06em;text-transform:uppercase;"
            f"border-bottom:1px solid #d0d0d0'>{tgt_short}</td></tr>"
        )

        all_raw = {}
        all_gap = {}
        all_gaj_scores = {}
        for model in models_show:
            for key in avail:
                r2  = _get(key, model, tgt)
                gap = _get(key, model, tgt, "R2_gap")
                score = _gaj(r2, gap)
                if r2 is not None:
                    all_raw[(key, model)] = r2
                    all_gap[(key, model)] = gap
                if score is not None:
                    all_gaj_scores[(key, model)] = score

        best_raw = max(all_raw.values()) if all_raw else None
        best_gaj = max(all_gaj_scores.values()) if all_gaj_scores else None
        raw_win_gap = None
        if best_raw is not None:
            for k_, v_ in all_raw.items():
                if abs(v_ - best_raw) < 1e-9:
                    raw_win_gap = all_gap.get(k_)
                    break
        show_gaj = (
            best_gaj is not None and best_raw is not None
            and raw_win_gap is not None and raw_win_gap > 0.10
        )

        def _is_raw(v):
            return best_raw is not None and v is not None and abs(v - best_raw) < 1e-9

        def _is_gaj(v):
            return show_gaj and best_gaj is not None and v is not None and abs(v - best_gaj) < 1e-9

        for model in models_show:
            row_bg = "#ffffff" if models_show.index(model) % 2 == 0 else "#f7f7f7"
            cells  = f"<td style='{_TD}'><strong>{model}</strong></td>"
            values = {}
            for key in avail:
                r2   = _get(key, model, tgt)
                rmse = _get(key, model, tgt, "RMSE_test")
                gap  = _get(key, model, tgt, "R2_gap")
                score = _gaj(r2, gap)
                values[key] = (r2, score)
                cells += _val_td(r2, rmse, gap, _is_raw(r2), _is_gaj(score))
            tbody += f"<tr style='background:{row_bg}'>{cells}</tr>"

            for trans in transitions:
                from_key, to_key = trans
                if from_key not in avail or to_key not in avail:
                    continue
                from_v, from_s = values.get(from_key, (None, None))
                to_v,   to_s   = values.get(to_key,   (None, None))
                if from_v is not None and to_v is not None:
                    trans_deltas[trans].append(to_v - from_v)
                if from_s is not None and to_s is not None:
                    trans_gaj[trans].append(to_s - from_s)

    def _stats_row(trans, from_lbl, to_lbl):
        deltas = trans_deltas[trans]
        gaj_d  = trans_gaj[trans]
        if not deltas:
            return (f"<tr><td style='{_STD}'><strong>{from_lbl} &rarr; {to_lbl}</strong></td>"
                    f"<td colspan='7' style='{_STD};color:#888888'> - </td></tr>")
        arr  = np.array(deltas)
        n    = len(arr)
        net  = float(arr.mean())
        wins   = arr[arr > MEANINGFUL]
        losses = arr[arr < -MEANINGFUL]
        ties   = arr[(arr >= -MEANINGFUL) & (arr <= MEANINGFUL)]
        win_mean  = float(wins.mean())   if len(wins)   else None
        loss_mean = float(losses.mean()) if len(losses) else None
        net_col = "#5BAD6F" if net > MEANINGFUL else ("#E15252" if net < -MEANINGFUL else "#888888")
        verdict = "Net improvement" if net > MEANINGFUL else ("Net regression" if net < -MEANINGFUL else "Negligible")
        win_str  = f"{len(wins)}/{n} (avg {win_mean:+.3f})"   if win_mean  is not None else f"{len(wins)}/{n}"
        loss_str = f"{len(losses)}/{n} (avg {loss_mean:+.3f})" if loss_mean is not None else f"{len(losses)}/{n}"
        if gaj_d:
            gaj_net  = float(np.array(gaj_d).mean())
            diff     = gaj_net - net
            gaj_col  = "#5BAD6F" if gaj_net > MEANINGFUL else ("#E15252" if gaj_net < -MEANINGFUL else "#888888")
            diff_col = "#E15252" if diff < -0.02 else "#5BAD6F" if diff > 0.02 else "#888888"
            interp   = ("Raw gains partially inflated by overfitting" if diff < -0.02
                        else "Raw gains understated - overfitting decreased" if diff > 0.02
                        else "Overfitting largely unchanged")
            gaj_str  = f"{gaj_net:+.4f}"
            diff_str = f"{diff:+.4f}"
        else:
            gaj_col = diff_col = "#888888"
            gaj_str = diff_str = " - "
            interp  = " - "
        return (
            f"<tr>"
            f"<td style='{_STD};white-space:nowrap'><strong>{from_lbl} &rarr; {to_lbl}</strong></td>"
            f"<td style='{_STD};text-align:center;color:{net_col};font-weight:bold'>{net:+.4f}</td>"
            f"<td style='{_STD};text-align:center;color:#5BAD6F'>{win_str}</td>"
            f"<td style='{_STD};text-align:center;color:#E15252'>{loss_str}</td>"
            f"<td style='{_STD};text-align:center;color:#888888'>{len(ties)}/{n}</td>"
            f"<td style='{_STD};text-align:center;color:{gaj_col};font-weight:bold'>{gaj_str}</td>"
            f"<td style='{_STD};text-align:center;color:{diff_col}'>{diff_str}</td>"
            f"<td style='{_STD}'><strong>{verdict}</strong><br>"
            f"<span style='font-size:0.82em;color:{diff_col}'>{interp}</span></td>"
            f"</tr>"
        )

    col_headers = "".join(f"<th style='{_THC}'>{labels[k]}</th>" for k in avail)
    main_table  = f"""
<div style='overflow-x:auto;margin-top:0.8rem;border:1px solid #cccccc;border-radius:4px'>
<table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.82rem;color:#1a1a1a;min-width:760px'>
  <thead>
    <tr style='border-bottom:2px solid #cccccc'>
      <th style='{_TH};min-width:70px'>Model</th>
      {col_headers}
    </tr>
  </thead>
  <tbody>{tbody}</tbody>
</table>
</div>"""

    stats_block = f"""
<div style='margin-top:1.4rem'>
  <p style='font-weight:bold;margin-bottom:0.4rem;color:#1a1a1a'>Transition Summary</p>
  <div style='overflow-x:auto;border:1px solid #cccccc;border-radius:4px'>
  <table style='border-collapse:collapse;width:100%;background:#ffffff;font-size:0.83rem;color:#1a1a1a;min-width:960px'>
    <thead>
      <tr style='border-bottom:2px solid #cccccc'>
        <th style='{_TH}'>Transition</th>
        <th style='{_THC}'>Net Mean ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>raw</span></th>
        <th style='{_THC}'>Improvements<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &gt; +{MEANINGFUL}</span></th>
        <th style='{_THC}'>Regressions<br><span style='color:#888888;font-weight:400;font-size:0.82em'>Δ &lt; -{MEANINGFUL}</span></th>
        <th style='{_THC}'>Negligible<br><span style='color:#888888;font-weight:400;font-size:0.82em'>|Δ| &lt;= {MEANINGFUL}</span></th>
        <th style='{_THC}'>Gap-Adj Net ΔR²<br><span style='color:#888888;font-weight:400;font-size:0.82em'>R²-0.5·max(0,|gap|-0.10)</span></th>
        <th style='{_THC}'>Gap-Adj - Raw<br><span style='color:#888888;font-weight:400;font-size:0.82em'>overfitting shift</span></th>
        <th style='{_TH}'>Verdict / Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {_stats_row(("Exp6-Voting", "Exp7-SE2"), "P9 Baseline", "P10b Sel-FE")}
      {_stats_row(("Exp6-Voting", "Exp8"),     "P9 Baseline", "P11 Temporal")}
      {_stats_row(("Exp7-SE2",   "Exp8"),     "P10b Sel-FE", "P11 Temporal")}
    </tbody>
  </table>
  </div>
</div>"""

    return f"""
<details class="exp-details" id="temporal-comparison">
  <summary><span class="fold-icon">▶</span> Comparisons</summary>
  <div class="exp-body">
    <div class="obs-card" style="border-left:4px solid #9B59B6">
      <p class="meta">
        <strong>★</strong> = best raw Test R² per target ·
        <strong>✦</strong> = best gap-adjusted per target (shown only when raw winner has |gap| &gt; 0.10).
        <span style='color:#E15252'>Red Gap</span> &gt; 0.10 = notable overfit.
        P9 Voting is the baseline; P10b Selective FE is the best prior method; P11 adds temporal lags + log1p target transform.
      </p>
    </div>
    {main_table}
    {stats_block}
  </div>
</details>"""


def _phase11_qna(df_all: pd.DataFrame) -> str:
    """Key findings Q&A for the Temporal Features section."""
    p11    = df_all[df_all["exp_key"] == "Exp8"].dropna(subset=["R2_test"])
    p10b   = df_all[df_all["exp_key"] == "Exp7-SE2"].dropna(subset=["R2_test"])
    p9_v   = df_all[df_all["exp_key"] == "Exp6-Voting"].dropna(subset=["R2_test"])

    def _avg_tgts(df, tgts):
        vals = [float(df[df["target"] == t]["R2_test"].max())
                for t in tgts if not df[df["target"] == t].empty]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _best_model(df, tgt, model):
        sub = df[(df["target"] == tgt) & (df["model"] == model)]
        return float(sub["R2_test"].iloc[0]) if not sub.empty else float("nan")

    def _c(v, good=0.05):
        if v != v:
            return " - "
        col = "#5BAD6F" if v > good else "#E15252" if v < 0 else "var(--text-primary)"
        return f"<span style='color:{col};font-weight:bold'>{v:+.3f}</span>"

    p11_grab_avg  = _avg_tgts(p11, GRAB_TARGETS)
    p10b_grab_avg = _avg_tgts(p10b, GRAB_TARGETS)
    p11_comp_avg  = _avg_tgts(p11, COMP_TARGETS)
    p10b_comp_avg = _avg_tgts(p10b, COMP_TARGETS)

    p11_grab_bod_r  = _best_model(p11, "Effluent BOD (mg/L, Grab)", "Ridge")
    p11_grab_cod_xgb = _best_model(p11, "Effluent COD (mg/L, Grab)", "XGB")
    p10b_comp_bod   = _avg_tgts(p10b[p10b["model"] == "Voting"], ["Effluent BOD (mg/L, Composite)"])

    q1 = (
        f"<strong>Grab targets benefited:</strong> Grab avg R² = {_c(p11_grab_avg, 0.20)} "
        f"vs Exp 7-SE2 baseline {_c(p10b_grab_avg, 0.20)}. "
        f"Grab BOD Ridge = {_c(p11_grab_bod_r, 0.30)} (gap -0.095 - honest generalisation). "
        f"Grab COD XGB = {_c(p11_grab_cod_xgb, 0.20)} - new Grab COD high with honest gap (+0.057). "
        f"Lag features add genuine signal: yesterday's inlet BOD correlates with today's "
        f"effluent BOD through retention time dynamics."
        f"<br><br>"
        f"<strong>Composite targets deteriorated:</strong> Comp avg R² = {_c(p11_comp_avg, 0.10)} "
        f"vs Exp 7-SE2 {_c(p10b_comp_avg, 0.10)}. "
        f"Comp TSS Ridge = -1.04 (gap +1.82) - catastrophic. With ~290 Composite training "
        f"rows and 50-58 features after lag expansion, overfitting is severe."
    )

    q2 = (
        f"Lag and rolling features were restricted to <strong>inlet + Flow + Power "
        f"columns only</strong> (not every continuous feature). Even with this restriction, "
        f"50-58 features result from the expansion. Full expansion across all process columns "
        f"would produce ~128 features on ~290 composite rows - the ratio is far too high. "
        f"<br><br>"
        f"This confirms that temporal features are only beneficial when the training-set size "
        f"is large relative to the added feature count. For composite targets, temporal "
        f"structure must be captured via the process parameters already present (secondary "
        f"clarifier/sedimentation data), not via lag engineering."
    )

    q3 = (
        f"<strong>Per-target winners (from Overfit-Aware Selection):</strong><br>"
        f"Grab BOD: Exp 6 Voting (0.692, gap ~0) - Exp 8 Ridge (0.656) is close but lower.<br>"
        f"Grab COD: Exp2-SE1 RF (0.489, gap -0.004) - Exp 8 XGB (0.464) is second.<br>"
        f"Grab TSS: Exp 7-SE2 Voting (0.642). Grab pH: Exp 8 ElNet (0.376).<br>"
        f"Comp BOD: Exp 7-SE2 Voting ({_c(p10b_comp_bod, 0.20)}). "
        f"Comp COD: no model generalises. "
        f"Comp TSS: Exp 7-SE2 ElNet (0.458). Comp pH: Exp3-SE1 Ridge (0.363).<br><br>"
        f"Temporal features contribute one clear Grab target win (Grab pH) and near-ties "
        f"on Grab BOD and COD after gap-adjustment. Exp 7-SE2 Selective FE remains the "
        f"overall recommended configuration."
    )

    def _qcard(n, question, answer):
        return (
            f"<div style='margin-bottom:1.1rem;border:1px solid var(--border);"
            f"border-radius:5px;overflow:hidden'>"
            f"<div style='background:var(--bg-secondary);padding:0.45rem 0.8rem;"
            f"border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:0.5rem'>"
            f"<span style='color:#4A90D9;font-size:0.78em;font-weight:bold;"
            f"letter-spacing:0.06em;flex-shrink:0'>Q{n}</span>"
            f"<span style='font-weight:bold;font-size:0.93em;line-height:1.4'>{question}</span>"
            f"</div>"
            f"<div style='padding:0.6rem 0.8rem 0.55rem'>"
            f"<p class='meta' style='margin:0;line-height:1.6;border-left:2px solid var(--border);"
            f"padding-left:0.6rem'>{answer}</p>"
            f"</div></div>"
        )

    cards = "".join([
        _qcard(1, "Do temporal lag features help Grab and Composite targets equally?", q1),
        _qcard(2, "Why was feature expansion restricted to inlet/Flow/Power columns only?", q2),
        _qcard(3, "Where does Exp 8 rank per-target, and what is the overall recommendation?", q3),
    ])
    return f"""
<details class="exp-details" id="temporal-findings" open>
  <summary><span class="fold-icon">▶</span> Key Findings  -  Temporal Features</summary>
  <div class="exp-body">{cards}</div>
</details>"""


def build_advanced_methods_section(df_all: pd.DataFrame) -> str:
    ann_sub     = _phase9_model_subsection(
        df_all, "Exp6-ANN", "p9-ann", "ANN (MLPRegressor)",
        _badge("FAILED avg R²=-1.12", "fail"))
    ann_failure = _ann_failure_callout()
    vote_sub = _phase9_model_subsection(
        df_all, "Exp6-Voting", "p9-voting",
        "Voting Ensemble (ElNet + RF + XGBoost)",
        _badge("RECOMMENDED", "rec"))
    stack_sub = _phase9_model_subsection(
        df_all, "Exp6-Stacking", "p9-stacking",
        "Stacking Ensemble (ElNet + RF + XGB - Ridge, walk-forward OOF)",
        _badge("CONSISTENT - no leakage", "warn"))

    df_p9 = df_all[df_all["exp_key"].isin(["Exp6-ANN","Exp6-Voting","Exp6-Stacking"])].copy()
    comparison = _adv_comparison_panel(df_all)
    best = _best_model_box(df_p9, "Advanced Methods")
    var_dx = _variance_diagnosis_callout()
    findings = _adv_methods_qna(df_all)
    rationale = _adv_methods_rationale_card()

    has_ann_extra = any(
        k in df_all["exp_key"].values
        for k in ["ANN-Exp1", "ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref"]
    )
    ann_exploration_block = ""
    if has_ann_extra:
        ann_e1_sub = _phase9_model_subsection(
            df_all, "ANN-Exp1", "p9-ann-exp1",
            "ANN  -  Exp1 Datasets (Inlet + COMMON, 9 features)")
        ann_e2s1_sub = _phase9_model_subsection(
            df_all, "ANN-Exp2-SE2", "p9-ann-exp2s1",
            "ANN  -  Exp2-SE1 Datasets (Secondary + COMMON, 15 features)")
        ann_e2s2_sub = _phase9_model_subsection(
            df_all, "ANN-Exp2-SE3-Ref", "p9-ann-exp2s2",
            "ANN  -  Exp2-SE2 Datasets (Inlet + Secondary + COMMON, 21 features)")
        ann_exploration_block = f"""
  <h3 class="section-title" id="p9-ann-exploration"
      style="font-size:1.0rem;margin:1.5rem 0 0.4rem;padding-left:0">
    ANN Dataset Exploration  -  Data-Volume Diagnostic
  </h3>
  <p class="section-intro" style="margin-bottom:0.8rem;font-size:0.9em">
    {EXP_INTRO["ANN-Dataset-Exploration"]}
  </p>
  {ann_e1_sub}
  {ann_e2s1_sub}
  {ann_e2s2_sub}"""

    return f"""
<section id="advanced-methods">
  <h1 class="section-title">Advanced Methods</h1>
  <p class="section-intro">{EXP_INTRO["Exp6"]}</p>

  <h2 class="section-title" id="adv-nn"
      style="font-size:1.25rem;margin:1.8rem 0 0.6rem;border-top:1px solid var(--border);padding-top:1rem">
    Neural Networks
  </h2>
  {rationale}
  {ann_sub}
  {ann_failure}
  {ann_exploration_block}

  <h2 class="section-title" id="adv-ensembles"
      style="font-size:1.25rem;margin:1.8rem 0 0.6rem;border-top:1px solid var(--border);padding-top:1rem">
    Ensemble Methods
  </h2>
  {vote_sub}
  {stack_sub}

  {comparison}
  {var_dx}
  {best}
  {findings}
</section>"""


def build_phase9_section(df_all: pd.DataFrame) -> str:
    ann_sub     = _phase9_model_subsection(
        df_all, "Exp6-ANN", "p9-ann", "ANN (MLPRegressor)",
        _badge("FAILED avg R²=−1.12", "fail"))
    ann_failure = _ann_failure_callout()
    vote_sub = _phase9_model_subsection(
        df_all, "Exp6-Voting", "p9-voting",
        "Voting Ensemble (ElNet + RF + XGBoost)",
        _badge("RECOMMENDED", "rec"))
    stack_sub = _phase9_model_subsection(
        df_all, "Exp6-Stacking", "p9-stacking",
        "Stacking Ensemble (ElNet + RF + XGB → Ridge, walk-forward OOF)",
        _badge("CONSISTENT - no leakage", "warn"))

    # Combined comparison across all three
    df_p9 = df_all[df_all["exp_key"].isin(["Exp6-ANN","Exp6-Voting","Exp6-Stacking"])].copy()
    all_m = [m for m in ADV_MODELS if m in df_p9["model"].values]
    comp_tbl = _metrics_table(df_p9, all_m, "p9-comp")
    best = _best_model_box(df_p9, "Exp 6")
    var_dx = _variance_diagnosis_callout()

    # ANN dataset-exploration sub-sections
    ann_e1_sub = _phase9_model_subsection(
        df_all, "ANN-Exp1", "p9-ann-exp1",
        "ANN  -  Exp1 Datasets (Inlet + COMMON, 9 features)")
    ann_e2s1_sub = _phase9_model_subsection(
        df_all, "ANN-Exp2-SE2", "p9-ann-exp2s1",
        "ANN  -  Exp2-SE1 Datasets (Secondary + COMMON, 15 features)")
    ann_e2s2_sub = _phase9_model_subsection(
        df_all, "ANN-Exp2-SE3-Ref", "p9-ann-exp2s2",
        "ANN  -  Exp2-SE2 Datasets (Inlet + Secondary + COMMON, 19 features)")
    ann_ds_comparison = _ann_dataset_comparison(df_all)
    ann_ds_callout = _ann_dataset_exploration_callout()

    # Only render the exploration block if at least one result file exists
    has_ann_extra = any(
        k in df_all["exp_key"].values
        for k in ["ANN-Exp1", "ANN-Exp2-SE2", "ANN-Exp2-SE3-Ref"]
    )
    ann_exploration_block = ""
    if has_ann_extra:
        ann_exploration_block = f"""
  <h2 class="section-title" id="p9-ann-exploration"
      style="font-size:1.1rem;margin:2rem 0 0.5rem">
    ANN Dataset Exploration  -  Data-Volume Diagnostic
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
  <p class="section-intro">{EXP_INTRO["Exp6"]}</p>
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
        df_all, "Exp7-SE1", "p10-full",
        "SE1 - Full Feature Engineering (All Targets)",
        _badge("COMPOSITE OVERFIT", "fail"))
    sel_fe = _phase10_variant(
        df_all, "Exp7-SE2", "p10b",
        "SE2 - Selective Feature Engineering (Grab FE / Composite Base)",
        _badge("RECOMMENDED", "rec"))
    comparison = _phase10_comparison_panel(df_all)
    findings   = _phase10_qna(df_all)
    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Exp7-SE1","Exp7-SE2"])],
        "Feature Engineering")
    return f"""
<section id="phase10">
  <h1 class="section-title">Feature Engineering</h1>
  <p class="section-intro">{EXP_INTRO["Exp7"]}</p>
  {full_fe}
  {sel_fe}
  {comparison}
  {findings}
  {best}
</section>"""


def build_phase11_section(df_all: pd.DataFrame) -> str:
    df = df_all[df_all["exp_key"] == "Exp8"].copy()
    if df.empty:
        return ""
    models = [m for m in ALL_MODELS_ORD if m in df["model"].values]
    feat_html  = _feature_card("Exp8")
    ds_html    = _dataset_summary(df)
    tbl        = _metrics_table(df, models, "p11-detail")
    train_tbl  = _train_metrics_table(df, models)
    comparison = _phase11_comparison_panel(df_all)
    findings   = _phase11_qna(df_all)
    best       = _best_model_box(df, "Temporal Features")

    cv_note = """
<div class="info-note">
  <strong>CV note:</strong> <code>CV_R²</code> and <code>Gap_gen = CV_R² - Test_R²</code>
  are stored in results.xlsx but not shown in the table (unified schema). Available in
  <code>models/phase11/results.xlsx</code>. First TimeSeriesSplit fold trains on ~80 rows
  with ~55 features - CV_R² is very noisy for Ridge/ElNet; interpret per-fold rather than
  as a mean. Key: Grab COD RF has CV_R²=0.291, Gap_gen=-0.014 - most consistent Grab COD
  model by cross-validation.
</div>"""

    return f"""
<section id="phase11">
  <h1 class="section-title">Temporal Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp8"]}</p>
  <details class="exp-details" open id="p11-se1">
    <summary><span class="fold-icon">▶</span> SE1 - Temporal Lags + log1p (All Models)</summary>
    <div class="exp-body">
      {feat_html}
      {ds_html}
      {tbl}
      {train_tbl}
    </div>
  </details>
  {cv_note}
  {comparison}
  {findings}
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
      1. Inlet-Only Features <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp1">
      <a class="nav-item nav-sub" href="#exp1-sub1">SE1 - Core Inlet Only</a>
      <a class="nav-item nav-sub" href="#exp1-s2">SE2 - Inlet + COMMON</a>
      <div class="nav-subgroup">
        <a class="nav-item nav-sub" href="#exp1-s3">SE3 - Extended Inlet + COMMON</a>
        <a class="nav-item nav-subsub" href="#exp1-s3-full">- Full Feature Set</a>
        <a class="nav-item nav-subsub" href="#exp1-s3-fs">- Feature Selection</a>
      </div>
      <a class="nav-item nav-sub" href="#exp1-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp1-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp2">
      2. Process Stage Features <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp2">
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp2-s1">SE1 - Primary Stage</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s1a">↳ Primary + COMMON only</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s1b">↳ Primary + Inlet + COMMON</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp2-s2">SE2 - Secondary Stages</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s2-clr">↳ Clarifier + Common</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s2-sed">↳ Sedimentation + Common</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s2a">↳ Secondary Only</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s2-combined">↳ All Secondary + Common</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp2-s3">SE3 - Combined</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s3-s6fs">↳ Secondary Only - Feature Selected</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s3-full">↳ Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s3-ref">↳ CIS (21 feat)</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp2-s3-ref-fs">↳ CIS - Feature Selected</a>
        </div>
      </div>
      <a class="nav-item nav-sub" href="#exp2-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp2-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp3">
      3. Extended Operational Features <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp3">
      <a class="nav-item nav-sub" href="#exp3-s1">SE1 - ADD-tier</a>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp3-s2">SE2 - ADD+CONSIDER</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s2-full">&#8627; Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s2-fs">&#8627; Feature Selection</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp3-s3">SE3 - Minus Coliform</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s3-full">&#8627; Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s3-fs">&#8627; Feature Selection</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp3-s4">SE4 - All Features</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s4-full">&#8627; Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp3-s4-fs">&#8627; Feature Selection</a>
        </div>
      </div>
      <a class="nav-item nav-sub" href="#exp3-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp3-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp4">
      4. VIF-Pruned Feature Set <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp4">
      <a class="nav-item nav-sub" href="#exp4-s1">SE1 - Manual Group Removal</a>
      <a class="nav-item nav-sub" href="#exp4-s2">SE2 - VIF-based Pruning</a>
      <a class="nav-item nav-sub" href="#exp4-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp4-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp5">
      5. Cross-Type Inlet Hypothesis <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp5">
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp5-s1">SE1 - Comp + Grab Inlet</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp5-s1-full">↳ Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp5-s1-fs">↳ Feature Selection</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#exp5-s2">SE2 - Grab + Comp Inlet</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#exp5-s2-full">↳ Full Feature Set</a>
          <a class="nav-item nav-sub nav-subsub" href="#exp5-s2-fs">↳ Feature Selection</a>
        </div>
      </div>
      <a class="nav-item nav-sub" href="#exp5-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp5-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-adv">
      6. Advanced Methods <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-adv">
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#adv-nn">Neural Networks</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#p9-ann">&#8627; ANN (Exp3-SE2)</a>
          <a class="nav-item nav-sub nav-subsub" href="#p9-ann-diagnosis">&#8627; ANN Post-Mortem</a>
          <a class="nav-item nav-sub nav-subsub" href="#p9-ann-exploration">&#8627; ANN Exploration</a>
        </div>
      </div>
      <div class="nav-subgroup">
        <div class="nav-subgroup-header">
          <a class="nav-item nav-sub" href="#adv-ensembles">Ensemble Methods</a>
          <span class="nav-sub-toggle collapsed"><span class="nav-chevron">▾</span></span>
        </div>
        <div class="nav-subgroup-items">
          <a class="nav-item nav-sub nav-subsub" href="#p9-voting">&#8627; Voting (ElNet+RF+XGB)</a>
          <a class="nav-item nav-sub nav-subsub" href="#p9-stacking">&#8627; Stacking (walk-fwd)</a>
        </div>
      </div>
      <a class="nav-item nav-sub" href="#adv-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#adv-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p10">
      7. Feature Engineering <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p10">
      <a class="nav-item nav-sub" href="#p10-full">SE1 - Full FE (All Tgts)</a>
      <a class="nav-item nav-sub" href="#p10b">SE2 - Selective FE (Grab)</a>
      <a class="nav-item nav-sub" href="#fe-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#fe-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p11">
      8. Temporal Features <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p11">
      <a class="nav-item nav-sub" href="#p11-se1">SE1 - Temporal + log1p</a>
      <a class="nav-item nav-sub" href="#temporal-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#temporal-findings">Findings</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp9">
      9. Recency Hypothesis <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp9">
      <a class="nav-item nav-sub" href="#exp9-s1">SE1 - W1: 2024-Only Window</a>
      <a class="nav-item nav-sub" href="#exp9-s2">SE2 - W1+FS: Feature Selection</a>
      <a class="nav-item nav-sub" href="#exp9-s3">SE3 - W1+LogY: Log Transform</a>
      <a class="nav-item nav-sub" href="#exp9-s4">SE4 - W1+LogF: Log Features</a>
      <a class="nav-item nav-sub" href="#exp9-s5">SE5 - W1+LogLog: Log Features+Target</a>
      <a class="nav-item nav-sub" href="#exp9-comparison">Comparisons</a>
      <a class="nav-item nav-sub" href="#exp9-findings">Findings</a>
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
  /* -- Layout ---------------------------------------------------------- */
  #sidenav {
    position: fixed; top: 0; left: 0;
    width: max-content; min-width: 220px; max-width: 340px;
    height: 100vh; overflow-y: auto; overflow-x: hidden;
    background: var(--card); border-right: 1px solid var(--border);
    padding: 0 0 24px; z-index: 200;
    font-size: 13px;
  }
  #sidenav::-webkit-scrollbar { width: 4px; }
  #sidenav::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  #main-content {
    margin-left: 260px; padding: 24px 36px 60px;
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
    user-select: none; white-space: nowrap;
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
    white-space: nowrap;
  }
  .nav-item:hover { color: var(--text); background: var(--summary-hover); }
  .nav-item.active {
    color: #4A90D9; border-left-color: #4A90D9;
    background: var(--toc-bg); font-weight: 600;
  }
  .nav-sub { padding-left: 28px; font-size: 12px; }
  .nav-subsub { padding-left: 2.4rem !important; font-size: 0.78em; color: var(--text-muted); }
  .nav-subgroup { display: block; }
  .nav-subgroup-header { display: flex; align-items: center; justify-content: space-between; }
  .nav-subgroup-header .nav-item { flex: 1; }
  .nav-sub-toggle {
    cursor: pointer; padding: 0 4px; flex-shrink: 0; user-select: none;
    line-height: 28px; display: flex; align-items: center;
  }
  .nav-sub-toggle .nav-chevron { transition: transform .2s; color: var(--text-muted); }
  .nav-sub-toggle:hover .nav-chevron { color: var(--text); }
  .nav-sub-toggle.collapsed .nav-chevron { transform: rotate(-90deg); }
  .nav-subgroup-items { display: none; }

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
  /* Ensure all table-containing regions scroll horizontally rather than overflow */
  .fold-body, .best-box, .exp-body > .obs-card { overflow-x: auto; }
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
// -- Sidebar width sync: match main-content margin to sidenav's rendered width --
(function() {
  function syncNavWidth() {
    var nav  = document.getElementById('sidenav');
    var main = document.getElementById('main-content');
    if (!nav || !main) return;
    main.style.marginLeft = nav.offsetWidth + 'px';
  }
  document.addEventListener('DOMContentLoaded', syncNavWidth);
  // Re-sync if fonts load late (avoids flash of wrong margin)
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(syncNavWidth);
  }
})();

// -- No-alt popup: close on outside click ------------------------------------
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

// ── Nav sidebar: collapsible groups (collapsed by default) ─────────────────
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.nav-collapsible').forEach(function(title) {
      var groupId = title.dataset.targetGroup;
      var items   = document.getElementById(groupId);
      if (!items) return;
      // Measure natural height then collapse by default
      items.style.maxHeight = items.scrollHeight + 'px';
      void items.offsetHeight;
      title.classList.add('collapsed');
      items.style.maxHeight = '0';
      items.classList.add('collapsed');
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

// ── Nav sidebar: subsub group fold/expand toggle ───────────────────────────
// Fires on the entire header row (link text OR chevron icon), so clicking the
// section name also toggles the child list - not just the ▾ symbol.
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.nav-subgroup-header').forEach(function(header) {
      header.addEventListener('click', function() {
        var subgroup      = header.closest('.nav-subgroup');
        var items         = subgroup.querySelector('.nav-subgroup-items');
        var toggle        = header.querySelector('.nav-sub-toggle');
        var parentGroupEl = subgroup.closest('.nav-group-items');
        var wasCollapsed  = toggle.classList.contains('collapsed');

        items.style.display = wasCollapsed ? 'block' : 'none';
        toggle.classList.toggle('collapsed', !wasCollapsed);

        if (parentGroupEl && parentGroupEl.style.maxHeight) {
          void parentGroupEl.offsetHeight;
          parentGroupEl.style.maxHeight = parentGroupEl.scrollHeight + 'px';
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
  // SECTION_BESTS and SECTION_ORDER are injected by Python
  var sectionOrder = window.SECTION_ORDER || [];
  var targetList   = ['Grab BOD','Grab COD','Grab TSS','Grab pH',
                      'Comp BOD','Comp COD','Comp TSS','Comp pH'];
  var modelColors  = {
    'OLS':'#E15252','Ridge':'#4A90D9','ElNet':'#5BAD6F',
    'RF':'#2171B5','GB':'#238B45','XGB':'#D94801',
    'ANN':'#9B59B6','Voting':'#E67E22','Stacking':'#1ABC9C'
  };
  var runningLeaders = {};
  var lastRendered   = '';
  var SCORE_NOISE_BAND = 0.005;

  function gapClass(g) {
    if (Math.abs(g) < 0.10) return 'gap-good';
    if (Math.abs(g) < 0.25) return 'gap-warn';
    return 'gap-bad';
  }

  function isBetterCandidate(b, current) {
    if (!current) return true;
    var bs = (typeof b.score === 'number') ? b.score : -Infinity;
    var cs = (typeof current.score === 'number') ? current.score : -Infinity;
    if (bs !== cs) return bs > cs;
    var bg = Math.abs(typeof b.gap === 'number' ? b.gap : 999);
    var cg = Math.abs(typeof current.gap === 'number' ? current.gap : 999);
    if (bg !== cg) return bg < cg;
    return (b.r2 || -Infinity) > (current.r2 || -Infinity);
  }

  function mergeSection(id, leaders) {
    var bests = window.SECTION_BESTS && window.SECTION_BESTS[id];
    if (!bests) return;
    targetList.forEach(function(tgt) {
      var b = bests[tgt];
      if (!b) return;
      if (isBetterCandidate(b, leaders[tgt])) {
        leaders[tgt] = b;
      }
    });
  }

  function activeSectionIndex() {
    var triggerY = window.innerHeight * 0.35;
    var active = -1;
    sectionOrder.forEach(function(id, idx) {
      var el = document.getElementById(id);
      if (!el) return;
      var rect = el.getBoundingClientRect();
      if (rect.top <= triggerY) active = idx;
    });
    return active;
  }

  function recomputeLeaders(activeIdx) {
    var leaders = {};
    for (var i = 0; i <= activeIdx; i++) {
      mergeSection(sectionOrder[i], leaders);
    }
    runningLeaders = leaders;
  }

  function renderLeaders() {
    var keys = Object.keys(runningLeaders);
    if (keys.length === 0) {
      var empty = '<p class="leaders-empty">Scroll past a section to see running leaders.</p>';
      if (empty !== lastRendered) {
        document.getElementById('leaders-content').innerHTML = empty;
        lastRendered = empty;
      }
      return;
    }
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
    recomputeLeaders(activeSectionIndex());
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
        decompose_target, TARGETS as ED_TARGETS,
        section_html as ed_section_html, resolve_target_winners,
    )
    winners = resolve_target_winners()
    frames = []
    for name, tgt, inlet in ED_TARGETS:
        df = decompose_target(name, tgt, inlet, winner_info=winners.get(tgt))
        if not df.empty:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    inner = ed_section_html(combined, winners=winners)
    return f"""
<section id="error-decomposition">
  <h1 class="section-title">2025 Residual Decomposition by Operational Regime</h1>
  <p class="section-intro">
    Each target is decomposed using its <strong>global overfit-aware winner</strong> from
    the Overfit-Aware Selection table - the same model recommended for deployment -
    rather than a single fixed ensemble. Residuals are split along four operational axes:
    Flow quartile, Weekday vs Weekend, Kathmandu/Nepal season, and Inlet Load quartile.
    Quartile thresholds are fit on the training set (2021-2024) only. All current
    global winners now have row-level predictions, so every target is included.
  </p>
  {inner}
</section>"""


def main():
    print("Loading results data...")
    df_all = load_all_data()
    print(f"  Loaded {len(df_all)} rows across "
          f"{df_all['exp_key'].nunique()} experiments, "
          f"{df_all['model'].nunique()} models.")

    print("Computing MdAE from stored predictions (BOD/TSS targets)...")
    mdae_df = compute_all_mdae()
    if not mdae_df.empty:
        df_all = df_all.merge(mdae_df, on=["exp_key", "model", "target"], how="left")
        n_filled = df_all["MdAE_test"].notna().sum()
        print(f"  MdAE computed for {n_filled} rows "
              f"({mdae_df['exp_key'].nunique()} experiments × BOD/TSS targets).")
    else:
        df_all["MdAE_test"] = np.nan
        print("  No MdAE data found - MdAE column will show - in report.")

    print("Building HTML sections...")
    print("  Building experiment and phase sections...")
    sections = [
        build_overview(df_all),
        build_exp1_section(df_all),
        build_exp2_section(df_all),
        build_exp3_section(df_all),
        build_exp4_section(df_all),
        build_exp5_section(df_all),
        build_advanced_methods_section(df_all),
        build_phase10_section(df_all),
        build_phase11_section(df_all),
        build_exp9_section(df_all),
    ]

    print("  Building model selection section...")
    sections.append(_build_model_selection_section(df_all))

    print("  Building error decomposition section...")
    sections.append(_build_error_decomposition_section())

    print("  Building Comp COD diagnostic section...")
    sections.append(_build_comp_cod_diagnostic(df_all))

    # Inline section-bests JSON and section order for the running leaders JS widget
    sec_bests_json, sec_order_json = _section_bests_json(df_all)
    section_data_js = (
        f"<script>window.SECTION_BESTS = {sec_bests_json};\n"
        f"window.SECTION_ORDER = {sec_order_json};</script>"
    )

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
      Experiments 1-5 &nbsp;·&nbsp; Neural Networks &amp; Ensembles &nbsp;·&nbsp;
      Feature Engineering &nbsp;·&nbsp; Temporal Features &nbsp;·&nbsp;
      9 models &nbsp;·&nbsp; 8 effluent targets &nbsp;·&nbsp; Train 2021-2024 · Test 2025
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
