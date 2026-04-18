"""
generate_unified_report.py — Unified HTML report covering all experiments and phases.

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

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR  = os.path.join(MODELING_DIR, "reports")

sys.path.insert(0, MODELING_DIR)
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

# Experiment key → short chart label
EXP_CHART_LABELS = {
    "Exp1": "Exp1",        "Exp1-FS": "Exp1-FS",
    "Exp2-Sub1": "E2-S1",  "Exp2-Sub1-FS": "E2-S1-FS",
    "Exp2-Sub2": "E2-S2",  "Exp2-Sub2-FS": "E2-S2-FS",
    "Exp3-S1": "E3-S1",    "Exp3-S1-FS": "E3-S1-FS",
    "Exp3-S2": "E3-S2",
    "Phase9-ANN": "P9-ANN", "Phase9-Voting": "P9-Vote",
    "Phase9-Stacking": "P9-Stack",
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
    ("experiment3/sub_exp2",                           "Exp3-S2"),
    ("experiment3/sub_exp1/feature_selected_datasets", "Exp3-S1-FS"),
    ("experiment3/sub_exp1",                           "Exp3-S1"),
    ("experiment2/sub_exp2/feature_selected_datasets", "Exp2-Sub2-FS"),
    ("experiment2/sub_exp2",                           "Exp2-Sub2"),
    ("experiment2/sub_exp1/feature_selected_datasets", "Exp2-Sub1-FS"),
    ("experiment2/sub_exp1",                           "Exp2-Sub1"),
    ("experiment1/feature_selected_datasets",          "Exp1-FS"),
    ("experiment1",                                    "Exp1"),
]

# predicted_<TAG>_run_N → canonical model name
_PRED_MODEL_MAP = {
    "OLS": "OLS", "Ridge": "Ridge", "ElNet": "ElNet",
    "RF_NL": "RF", "GB_NL": "GB", "XGB_NL": "XGB",
    "ANN": "ANN", "Voting": "Voting", "Stacking": "Stacking",
}

# Phase 9 models live inside Exp3-S2 files but belong to a different exp_key
_PHASE9_EXP = {"ANN": "Phase9-ANN", "Voting": "Phase9-Ensemble", "Stacking": "Phase9-Ensemble"}

FEATURE_DESCRIPTIONS = {
    "Exp1": {
        "label": "Inlet + COMMON (9 features)",
        "features": "Inlet pH, Inlet BOD, Inlet COD, Inlet TSS (Grab or Composite) + "
                    "Flow (MLD), Power Total (KW), month, day_of_week, year",
        "rationale": "Baseline: can inlet water quality alone predict effluent quality? "
                     "Tests the simplest, most operationally available feature set — "
                     "no secondary treatment data required.",
    },
    "Exp1-FS": {
        "label": "Inlet + COMMON — Feature Selected (5–8 features)",
        "features": "Subset of Exp1 features retained by RF permutation importance "
                    "(importance ≥ 0.03 threshold) and mutual information screening",
        "rationale": "Checks whether pruning the 9-feature set to its most informative "
                     "subset improves generalisation on the 2025 holdout.",
    },
    "Exp2-Sub1": {
        "label": "Secondary Clarifier + COMMON (15 features)",
        "features": "Sec Clarifier pH/TSS/BOD/COD/RAS, Sec Sed pH/TSS/BOD/COD/RAS "
                    "+ Flow, Power, month, day_of_week, year",
        "rationale": "Tests secondary treatment process data only (no inlet). "
                     "Do downstream process indicators predict effluent better than inlet?",
    },
    "Exp2-Sub1-FS": {
        "label": "Secondary + COMMON — Feature Selected",
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
        "label": "Inlet + Secondary + COMMON — Feature Selected",
        "features": "Subset of Exp2-S2 combined features after permutation importance "
                    "and mutual information screening",
        "rationale": "Remove redundant or low-signal features from the full Exp2-S2 set.",
    },
    "Exp3-S1": {
        "label": "Exp2-S2 + ADD-tier Aeration Features (16–22 features)",
        "features": "Exp2-S2 baseline + ADD-tier features (MI ≥ 0.20, marginal row "
                    "cost ≤ 20%): MLSS, SV30, SVI, DO, Aeration pH — adds ~3–7 columns",
        "rationale": "Expand with aeration basin data that shows high mutual information "
                     "and low missingness cost on top of the Exp2-S2 baseline.",
    },
    "Exp3-S1-FS": {
        "label": "Exp3-S1 — Core + Useful (Feature Selected, 5–8 features)",
        "features": "RF permutation importance ≥ 0.03 threshold applied to the "
                    "full Exp3-S1 feature set → 5–8 features per target",
        "rationale": "Aggressive pruning of the expanded Exp3-S1 set to the most "
                     "predictive features per target. Tests if less is more.",
    },
    "Exp3-S2": {
        "label": "Exp2-S2 + ADD + CONSIDER-tier Features (20–32 features)",
        "features": "Exp2-S2 + ADD features + CONSIDER-tier (MI ≥ 0.15 & cost ≤ 35%, "
                    "or MI ≥ 0.25 & cost ≤ 50%) — 20–32 total features per target",
        "rationale": "Full feature expansion per the audit-tier logic. Tests whether "
                     "CONSIDER-tier features help despite higher missingness cost. "
                     "Key question: do regularised linear models benefit from "
                     "additional features even if non-linear models overfit?",
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
                    "Replaces original Ridge+ElNet+RF — Ridge removed because it duplicates "
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
                    "(no OOF fold coverage) — strictly no look-ahead bias. "
                    "Base models retrained on full training set before inference.",
        "rationale": "Replaces original KFold(n_splits=5) StackingRegressor which had "
                     "look-ahead bias: predicting 2021 data using models trained on 2022–2024. "
                     "Walk-forward OOF ensures the meta-learner learns how base models "
                     "perform when projecting forward in time. "
                     "Approx. 83% OOF coverage across all targets.",
    },
    "Phase10-FE": {
        "label": "Exp3-S2 + Full Feature Engineering — All Targets (37–52 features)",
        "features": "Exp3-S2 base + log1p transforms of skewed cols + pairwise "
                    "interaction terms (key pairs) + IQR outlier indicator flags; "
                    "applied uniformly to all 8 targets",
        "rationale": "Feature engineering to capture non-linear relationships and "
                     "reduce skewness. Applied uniformly to test if it universally "
                     "helps. Hypothesis: log transforms help BOD/COD/TSS (right-skewed).",
    },
    "Phase10b-FE": {
        "label": "Selective Feature Engineering — Grab: Full FE / Composite: Base Exp3-S2",
        "features": "Grab targets: full FE (37–52 features each). "
                    "Composite targets: base Exp3-S2 features only (17–27 features) "
                    "— no engineering applied to avoid overfitting on small composite n.",
        "rationale": "Addresses Phase 10 finding: FE caused severe overfitting on "
                     "composite datasets (n_train ≈ 290). Selective approach preserves "
                     "FE gains on grab targets while stabilising composites.",
    },
    "Phase11": {
        "label": "Exp3-S2 + Temporal Lags + log1p Target Transform (50–58 features)",
        "features": "Base Exp3-S2 features + lag 1/3/7-day + 7-day calendar rolling mean "
                    "applied to inlet + Flow + Power columns only (~50–58 total features per target). "
                    "BOD/COD/TSS targets log1p-transformed; pH untransformed. "
                    "Back-transform via Duan smearing: expm1(ŷ) × mean(exp(training residuals)).",
        "rationale": "Daily wastewater quality has temporal autocorrelation — yesterday's "
                     "influent predicts today's effluent better than cross-sectional features "
                     "alone. Lag expansion restricted to inlet+Flow+Power columns to avoid "
                     "catastrophic overfitting on composite datasets (n≈290 rows). "
                     "log1p targets reduce right-skew in BOD/COD/TSS distributions.",
    },
}

EXP_INTRO = {
    "Exp1": (
        "Experiment 1 establishes the baseline: how well can <strong>inlet water quality "
        "alone</strong> predict effluent concentrations? This is the simplest and most "
        "operationally available feature set — no secondary treatment measurements needed. "
        "A feature-selected sub-variant is also included to test whether pruning to "
        "the most informative subset improves generalisation."
    ),
    "Exp2": (
        "Experiment 2 expands the feature scope. <strong>Sub-experiment 1</strong> tests "
        "secondary clarifier data without any inlet measurements, while "
        "<strong>Sub-experiment 2</strong> combines both inlet and secondary data for the "
        "richest baseline. Feature-selected variants are included for each sub-experiment."
    ),
    "Exp3": (
        "Experiment 3 extends the Exp2-S2 combined feature set with aeration basin data "
        "identified in the Phase 6 feature audit. <strong>Sub-experiment 1</strong> adds "
        "ADD-tier features (high MI, low missingness cost). "
        "<strong>Sub-experiment 2</strong> further adds CONSIDER-tier features to test "
        "whether more features — at higher missingness cost — still benefit regularised "
        "linear models."
    ),
    "Phase9": (
        "Phase 9 evaluates advanced model architectures on the <strong>Exp3-S2 feature set</strong>, "
        "which was selected as the richest validated set (ElNet Test R²=0.684 Grab BOD, "
        "RF 0.504 Grab TSS — new records at the time). "
        "<br><br>"
        "Three architectures are tested: "
        "<strong>ANN</strong> (MLPRegressor) — failed; ~470 training samples insufficient for "
        "MLP (avg Test R²=−1.12). All targets selected alpha=1.0 (max regularisation). Not recommended. "
        "<br>"
        "<strong>Voting ensemble</strong> (ElNet + RF + XGBoost, equal weights) — "
        "original composition (Ridge+ElNet+RF) was corrected: Ridge duplicates ElNet's L2 "
        "penalty, making two of three voters collinear. XGBoost replaces Ridge for structural "
        "diversity (gradient boosting vs. bagging vs. regularised linear). "
        "All three voters pre-tuned via TimeSeriesSplit GridSearchCV/RandomizedSearchCV. "
        "<br>"
        "<strong>Stacking ensemble</strong> (ElNet + RF + XGB → Ridge meta, walk-forward OOF) — "
        "original KFold(n_splits=5) stacking had look-ahead bias: 2021 samples were predicted "
        "using base models trained on 2022–2024 data. Replaced with manual walk-forward OOF "
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
        "<strong>Grab COD</strong> (XGB 0.464, gap +0.057 — new Grab COD high with an honest gap). "
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
        "Exp1": "Exp1", "Exp2-Sub1": "Exp2-Sub1", "Exp2-Sub2": "Exp2-Sub2",
        "Exp3-S1": "Exp3-S1", "Exp3-S1-FS": "Exp3-S1-FS", "Exp3-S2": "Exp3-S2",
        "Experiment 1": "Exp1",
        "Experiment 2 Sub-1": "Exp2-Sub1",
        "Experiment 2 Sub-2": "Exp2-Sub2",
        "Experiment 3 Sub-1": "Exp3-S1",
        "Experiment 3 Sub-1 FS": "Exp3-S1-FS",
        "Experiment 3 Sub-2": "Exp3-S2",
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
        ("exp3_s1", False), ("exp3_s1_fs", False), ("exp3_s2", False),
    ]:
        p = os.path.join(m, "linear", variant, "results.xlsx")
        if os.path.exists(p):
            df = pd.read_excel(p)
            df = df[df["run"] == df["run"].max()]
            frames.append(_melt_linear(df, is_fs))

    # Non-linear
    for variant, is_fs in [
        ("baseline", False), ("feature_selected", True),
        ("exp3_s1", False), ("exp3_s1_fs", False), ("exp3_s2", False),
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
    are absent — MdAE_test will be NaN for them after the merge.
    """
    import glob, re
    ds_base = os.path.join(MODELING_DIR, "datasets")
    records = []

    for fpath in sorted(glob.glob(os.path.join(ds_base, "**", "*.xlsx"), recursive=True)):
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
        return "—"
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
        n_tr = int(row0["n_train"]) if not np.isnan(row0["n_train"]) else "—"
        n_te = int(row0["n_test"])  if not np.isnan(row0["n_test"])  else "—"
        n_fe = int(row0["n_features"]) if not np.isnan(row0["n_features"]) else "—"
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
        <th>Target</th><th>n_train (2021–2024)</th>
        <th>n_test (2025)</th><th>n_features</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</details>"""


def _pick_best(avail, sub):
    """
    Threshold best-model criterion per target row:
      - Primary:  highest Test R², when at least one model has R² ≥ 0.
      - Fallback: lowest RMSE_test, when ALL models have R² < 0
                  (variance collapse makes R² unreliable).
    Returns a set of model names that qualify as 'best' for this row.
    """
    r2_vals = {}
    rmse_vals = {}
    for m in avail:
        msub = sub[sub["model"] == m]
        if msub.empty:
            continue
        r2   = msub["R2_test"].values[0]
        rmse = msub["RMSE_test"].values[0]
        if not np.isnan(r2):
            r2_vals[m] = r2
        if not np.isnan(rmse):
            rmse_vals[m] = rmse

    if not r2_vals:
        return set()

    max_r2 = max(r2_vals.values())
    if max_r2 >= 0:
        # At least one model beats naive baseline — use R²
        threshold = max_r2 - 1e-9
        return {m for m, v in r2_vals.items() if v >= threshold}
    else:
        # All models fail; fall back to lowest RMSE
        if not rmse_vals:
            return set()
        min_rmse = min(rmse_vals.values())
        return {m for m, v in rmse_vals.items() if v <= min_rmse + 1e-9}


def _metrics_table(df: pd.DataFrame, models: list, section_id: str) -> str:
    """Metric table: rows = targets, cols = (model × Test R² / Gap / RMSE / MAE / MdAE).

    MdAE (Median Absolute Error) is shown for BOD and TSS targets only — these have
    severe outlier distributions (up to 9.8% severe outliers in training) where RMSE
    is dominated by spikes. MdAE is the primary reliability metric for those targets.
    Phase 10 and Phase 11 do not store row-level predictions so MdAE is unavailable (—).

    Best-model highlighting uses a threshold rule:
      - Any model has R² ≥ 0 → highlight highest R² (★)
      - All models have R² < 0 → fall back to lowest RMSE (★ in RMSE cell)
    """
    avail = [m for m in models if m in df["model"].values]
    if not avail:
        return "<p class='meta'>No data for these models.</p>"

    has_mdae = "MdAE_test" in df.columns

    hdr1 = "".join(
        '<th colspan="5" style="color:{};">{}</th>'.format(MODEL_COLORS.get(m, "#888"), m)
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
        r2_vals = {m: sub[sub["model"] == m]["R2_test"].values[0]
                   for m in avail if m in sub["model"].values}
        use_rmse_fallback = bool(r2_vals) and max(
            (v for v in r2_vals.values() if not np.isnan(v)), default=float("-inf")
        ) < 0

        for m in avail:
            msub = sub[sub["model"] == m]
            if msub.empty:
                cells += "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>"
                continue
            r2   = msub["R2_test"].values[0]
            gap  = msub["R2_gap"].values[0]
            rmse = msub["RMSE_test"].values[0]
            mae  = msub["MAE_test"].values[0]
            is_best = m in best_models

            cell_bg = "background:rgba(74,144,217,0.20);font-weight:bold;" if is_best else ""

            r2_star   = "★ " if (is_best and not use_rmse_fallback) else ""
            rmse_star = "★ " if (is_best and use_rmse_fallback)     else ""

            # MdAE cell — only for BOD/TSS targets; dash otherwise
            if show_mdae and has_mdae:
                mdae_val = msub["MdAE_test"].values[0]
                mdae_str = _fmt(mdae_val) if not (isinstance(mdae_val, float) and np.isnan(mdae_val)) else "—"
                mdae_title = 'title="Primary reliability metric for outlier-prone targets"'
                mdae_cell = (
                    f'<td style="color:#f1c40f;font-style:italic;" {mdae_title}>'
                    f'{mdae_str}</td>'
                )
            else:
                mdae_cell = '<td style="color:var(--text-muted)">—</td>'

            cells += (
                f'<td style="color:{_r2_color(r2)};{cell_bg}">'
                f'{r2_star}{_fmt(r2)}</td>'
                f'<td class="{_gap_cls(gap)}">{_fmt(gap)}</td>'
                f'<td style="{cell_bg if use_rmse_fallback else ""}">'
                f'{rmse_star}{_fmt(rmse)}</td>'
                f'<td>{_fmt(mae)}</td>'
                f'{mdae_cell}'
            )
        rows.append(f'<tr data-target="{slug}">{cells}</tr>')

    legend = (
        '<p class="table-note">'
        'Test R²: '
        '<span style="color:#2ecc71">≥0.6 strong</span> · '
        '<span style="color:#52c98a">0.4–0.6 good</span> · '
        '<span style="color:#f1c40f">0.2–0.4 moderate</span> · '
        '<span style="color:#e67e22">0–0.2 weak</span> · '
        '<span style="color:#e74c3c">&lt;0 fails baseline</span>. '
        'Best per row: <span style="background:rgba(74,144,217,0.20);'
        'padding:1px 4px;border-radius:3px;font-weight:bold">★ highest R²</span> '
        'when any model ≥ 0; '
        '<span style="background:rgba(74,144,217,0.20);'
        'padding:1px 4px;border-radius:3px;font-weight:bold">★ lowest RMSE</span> '
        'when all R² &lt; 0 (variance-collapse fallback).</p>'
        '<p class="table-note">'
        'R² Gap (Train − Test): '
        '<span class="gap-good">■ &lt;0.10 OK</span> · '
        '<span class="gap-warn">■ 0.10–0.25 mild overfit</span> · '
        '<span class="gap-bad">■ &gt;0.25 severe overfit — treat result with caution</span>.</p>'
        '<p class="table-note">'
        '<span style="color:#f1c40f;font-style:italic;">MdAE</span> '
        '(Median Absolute Error) shown for BOD and TSS targets only — RMSE is unreliable '
        'for these targets due to severe outlier distributions (up to 9.8% severe outliers '
        'in training data, TSS train max = 1266 mg/L). '
        'MdAE is the primary reliability metric for BOD/TSS; — indicates predictions not '
        'stored at row level (Phase 10 / Phase 11).</p>'
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
            f"signal for those targets — consider a higher importance threshold.")
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
        verdict = (f"The row cost is <strong>justified on {justified}/{total} targets</strong> — "
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


def _vif_callout() -> str:
    """Compute VIF for Exp3-S2 feature sets inline and render as a foldable section."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
    except ImportError:
        return '<div class="info-note">VIF analysis requires statsmodels (pip install statsmodels).</div>'

    ds_dir = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
    DATASETS = [
        ("s2_stage3_grab_BOD", "Effluent BOD (mg/L, Grab)"),
        ("s2_stage3_grab_COD", "Effluent COD (mg/L, Grab)"),
        ("s2_stage3_grab_TSS", "Effluent TSS (mg/L, Grab)"),
        ("s2_stage3_grab_pH",  "Effluent pH (Grab)"),
        ("s2_stage3_comp_BOD", "Effluent BOD (mg/L, Composite)"),
        ("s2_stage3_comp_COD", "Effluent COD (mg/L, Composite)"),
        ("s2_stage3_comp_TSS", "Effluent TSS (mg/L, Composite)"),
        ("s2_stage3_comp_pH",  "Effluent pH (Composite)"),
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

        short_name = name.replace("s2_stage3_", "").replace("_", " ").title()
        tbl_rows = ""
        for feat, vif_val in vif_rows:
            flag, fc = _vflag(vif_val)
            vif_str = f"{vif_val:.2f}" if not (isinstance(vif_val, float) and np.isnan(vif_val)) else "—"
            bold = "bold" if vif_val > 10 else "normal"
            tbl_rows += (
                f'<tr><td>{feat}</td>'
                f'<td style="color:{fc};font-weight:{bold}">{vif_str}</td>'
                f'<td><span style="color:{fc};font-size:11px">{flag}</span></td></tr>'
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
    <p class="meta">n_train (complete rows): {len(X_clean)} · n_features: {len(features)}</p>
    <table class="summary-table" style="font-size:12px;max-width:540px">
      <thead><tr><th>Feature</th><th>VIF</th><th>Flag</th></tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table>
  </div>
</details>""")

    summary_html = (
        f'<p class="meta">Global across all 8 datasets: '
        f'<strong style="color:#e74c3c">{total_high} HIGH (VIF&gt;10)</strong> · '
        f'<strong style="color:#f39c12">{total_mod} MODERATE (VIF 5–10)</strong>. '
        f'Concentrated in Sec Sed vs Sec Clarifier pairs (adjacent measurements of the same parameter) '
        f'and Aeration SVI (mathematically derived from SV30/MLSS). '
        f'ElasticNet handles collinearity via L1 penalty — this explains why ElNet outperforms '
        f'OLS/Ridge on composite targets. OLS/Ridge coefficients are directly inflated by collinear pairs.</p>'
    )

    return f"""
<details class="exp-details" id="exp3-vif">
  <summary><span class="fold-icon">▶</span> VIF Collinearity Analysis — Exp3-S2 Feature Sets</summary>
  <div class="exp-body">
    <p class="meta">
      Variance Inflation Factor computed on training rows (year &lt; 2025) after listwise deletion.
      VIF &gt; 10 = problematic collinearity for linear models.
      VIF 5–10 = moderate.
      The threshold rule of thumb: VIF &gt; 10 indicates that the variance of a coefficient estimate
      is inflated by a factor of 10 relative to an orthogonal design — OLS/Ridge predictions are
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
                  if "hidden_layer_sizes" in params else "—")
        alpha  = (params.split("alpha': ")[1].split("}")[0].strip()
                  if "alpha" in params else "—")
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
          <th title="Training rows ÷ features — MLPs need n/p ≥ 100">n/p</th>
          <th>Best arch</th><th>Best α</th>
          <th>Train R²</th><th>Test R²</th><th>R² Gap</th>
          <th>Status</th><th>Δ vs Voting</th>
        </tr>
      </thead>
      <tbody>{tbl_rows}</tbody>
    </table>
    <p class="table-note">
      n/p = training rows ÷ features. <span style="color:#e74c3c">Red</span> = n/p &lt; 20 (underdetermined — high overfitting risk).
      MLPs typically need n/p ≥ 100–200 for stable generalisation on tabular data.
      Δ vs Voting = ANN Test R² − Phase 9 Voting Test R².
    </p>

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Root-Cause Analysis</summary>
      <div class="fold-body">
        <p class="meta"><strong style="color:var(--accent-blue)">1. Sample-to-feature ratio.</strong>
          Worst case: Comp TSS — 290 training rows, 27 features → n/p = 10.7.
          MLPs require far more data than tree ensembles to generalise from small tabular datasets.
          Typical guidance: n/p ≥ 100–200 for stable MLP generalisation.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">2. Temporal distribution shift.</strong>
          2025 test data shows different distributional properties than 2021–2024 training data
          (Q4-Flow and DJF errors 2–3× higher — see Error Regime Decomposition).
          MLPs form highly non-linear decision boundaries that diverge further from shifted data
          than regularised linear models or bagged trees.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">3. Early stopping did not prevent 2025 failure.</strong>
          Early stopping (validation_fraction=0.10, n_iter_no_change=20) monitors in-sample
          generalisation on the 2021–2024 fold, not 2025 generalisation.
          Gaps range from +0.41 (Grab COD) to +6.77 (Comp pH) despite early stopping —
          confirming that early stopping alone cannot address distributional shift.</p>
        <p class="meta"><strong style="color:var(--accent-blue)">4. Network capacity is not the bottleneck.</strong>
          Best architectures are small — (64,) or (128,) single hidden layer with α=1.0 for 6 of 8 targets.
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


def _section_bests_json(df_all: pd.DataFrame) -> str:
    """JSON for the dynamic running-leaders sidebar panel."""
    section_exp_keys = {
        "exp1-full":   ["Exp1"],
        "exp1-fs":     ["Exp1-FS"],
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
            overfit_flag = ' <span style="color:#e74c3c;font-size:10px" title="R² gap > 0.25 — possible overfit">⚠ overfit</span>'
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
            f'High-gap winners may reflect the model memorising 2021–2024 patterns absent in 2025 — '
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
    ds_html     = _dataset_summary(df)
    lin_tbl     = _metrics_table(df[df["model"].isin(ml)], ml, f"{section_id}-lin")
    nl_tbl      = _metrics_table(df[df["model"].isin(mn)], mn, f"{section_id}-nl")
    comp_tbl    = _metrics_table(df, all_m, f"{section_id}-comp")
    train_tbl   = _train_metrics_table(df, all_m)

    open_attr = " open" if open_default else ""
    return f"""
<details class="exp-details"{open_attr} id="{section_id}">
  <summary><span class="fold-icon">▶</span> {title}</summary>
  <div class="exp-body">
    {feat_html}
    {ds_html}

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Linear Models (OLS · Ridge · ElNet)</summary>
      <div class="fold-body">{lin_tbl}</div>
    </details>

    <details class="inner-fold">
      <summary><span class="fold-icon">▶</span> Non-Linear Models (RF · GB · XGB)</summary>
      <div class="fold-body">{nl_tbl}</div>
    </details>

    <details class="inner-fold" open>
      <summary><span class="fold-icon">▶</span> All Models — Combined Comparison</summary>
      <div class="fold-body">{comp_tbl}{train_tbl}</div>
    </details>
  </div>
</details>"""


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW SECTION
# ═══════════════════════════════════════════════════════════════════════════════

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
            f"The smallest |gap| among them is {min_gap_in_window:+.3f} — "
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
    # Late import — best_models_selection imports from this module, so must be deferred
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
            gadj_lbl  = str(gpick.get("gadj_model", "—"))
            gadj_r2   = float(gpick.get("gadj_R2",  float("nan")))
            gadj_gap  = float(gpick.get("gadj_gap", float("nan")))
            gadj_rmse = float(gpick.get("gadj_RMSE", float("nan")))
        else:
            gadj_lbl = naive_compound; gadj_r2 = naive_r2
            gadj_gap = naive_gap; gadj_rmse = naive_rmse

        # Classify this row into three display states:
        #   • gap_ok       — naive gap is acceptable (< 0.10); no concern, blank right side
        #   • has_alt      — gap is concerning AND gap-adj picks a DIFFERENT model
        #   • no_alt       — gap is concerning but gap-adj picks the SAME model (no better option)
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

        rows.append(
            f'<tr data-target="{slug}">'
            # Naive champion
            f'<td><strong>{TARGET_SHORT.get(tgt, tgt)}</strong></td>'
            f'<td><strong style="color:{mdl_col}">{naive_mdl}</strong></td>'
            f'<td style="color:{_r2_color(naive_r2)};font-weight:bold">{_fmt(naive_r2)}</td>'
            f'<td class="{_gap_cls(naive_gap)}">{_fmt(naive_gap)}</td>'
            f'<td class="meta">{_fmt(naive_rmse)}</td>'
            f'<td class="meta" style="font-size:11px">{exp_label}</td>'
            + right_cells + warn_cell +
            f'</tr>'
        )

    # Compute split averages (naive best per target)
    grab_vals = [df_all[df_all["target"] == t]["R2_test"].max()
                 for t in GRAB_TARGETS
                 if not df_all[df_all["target"] == t].dropna(subset=["R2_test"]).empty]
    comp_vals = [df_all[df_all["target"] == t]["R2_test"].max()
                 for t in COMP_TARGETS
                 if not df_all[df_all["target"] == t].dropna(subset=["R2_test"]).empty]
    grab_avg = float(np.nanmean(grab_vals)) if grab_vals else float("nan")
    comp_avg = float(np.nanmean(comp_vals)) if comp_vals else float("nan")
    avg_all  = float(np.nanmean(grab_vals + comp_vals)) if (grab_vals or comp_vals) else float("nan")

    def _avg_row(label, val):
        return (
            f"<tr style='background:var(--toc-bg);font-style:italic'>"
            f"<td colspan='10' style='text-align:right;color:var(--text-muted);font-size:11px'>"
            f"{label}</td>"
            f"<td style='font-weight:700;color:{_r2_color(val)};white-space:nowrap'>"
            f"avg R² {_fmt(val)}</td></tr>"
        )

    avg_rows = (
        _avg_row("Avg naive best R² — Grab targets", grab_avg) +
        _avg_row("Avg naive best R² — Composite targets", comp_avg) +
        _avg_row("Avg naive best R² — All targets", avg_all)
    )

    return f"""
<div class="card section-card" id="overview-leaderboard">
  <h2>Global Leaderboard — Best Result Per Target</h2>
  <p class="meta">
    Left half: the <strong>highest raw Test R²</strong> achieved for each target across all
    experiments — the naive ceiling.
    Right half: the <strong>gap-adjusted recommendation</strong> — the best model after
    penalising large train/test gaps (see
    <a href="#model-selection">Model Selection</a> for the full rule explanation).
    A <span style="color:#E67E22">⚠</span> in the last column flags targets where the
    recommended model differs from the naive champion — those are the cases where
    overfitting matters for deployment.
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
        <th rowspan="2">⚠</th>
      </tr>
      <tr>
        <th>Model</th><th>Test R²</th><th>R² Gap</th><th>RMSE</th><th>Experiment</th>
        <th>Model · Experiment</th><th>R²</th><th>Gap</th><th>RMSE</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}{avg_rows}</tbody>
  </table>
  </div>
</div>"""


def _progression_chart(df_all: pd.DataFrame) -> str:
    """Chart.js line chart with R² / RMSE / MAE toggle per model family."""
    chart_models = ["Ridge", "ElNet", "RF", "Voting", "Stacking", "ANN"]
    metrics = [
        ("R2_test",   "Avg Test R²",   False),   # (col, y-label, lower_is_better)
        ("RMSE_test", "Avg Test RMSE", True),
        ("MAE_test",  "Avg Test MAE",  True),
    ]

    # Build one grp per metric
    grp_all = {}
    for col, _, _ in metrics:
        if col in df_all.columns:
            grp_all[col] = df_all.groupby(["exp_key", "model"])[col].mean().reset_index()
        else:
            grp_all[col] = pd.DataFrame(columns=["exp_key", "model", col])

    # Determine x-axis from R2 presence
    grp_r2 = grp_all["R2_test"]
    exp_order_present = [e for e in EXP_CHART_ORDER if e in grp_r2["exp_key"].values]
    labels = [EXP_CHART_LABELS.get(e, e) for e in exp_order_present]

    def _build_datasets(col):
        grp = grp_all[col]
        out = []
        for mdl in chart_models:
            data = []
            for ek in exp_order_present:
                row = grp[(grp["exp_key"] == ek) & (grp["model"] == mdl)]
                val = float(row[col].values[0]) if not row.empty else None
                data.append(val if val is not None and not np.isnan(val) else None)
            if any(v is not None for v in data):
                out.append({
                    "label": mdl,
                    "data": data,
                    "borderColor": MODEL_COLORS.get(mdl, "#888"),
                    "backgroundColor": MODEL_COLORS.get(mdl, "#888"),
                    "spanGaps": True,
                    "tension": 0.3,
                    "pointRadius": 5,
                    "borderWidth": 2,
                })
        return out

    all_data = {
        "r2":   {"labels": labels, "datasets": _build_datasets("R2_test")},
        "rmse": {"labels": labels, "datasets": _build_datasets("RMSE_test")},
        "mae":  {"labels": labels, "datasets": _build_datasets("MAE_test")},
    }
    chart_json = json.dumps(all_data)

    return f"""
<div class="card section-card" id="overview-progression">
  <h2>Metric Progression Across Experiments</h2>
  <p class="meta">Average metric across all 8 targets per model per experiment.
  Gaps in a line indicate the model was not run in that experiment.
  <strong>R²:</strong> higher is better. <strong>RMSE / MAE:</strong> lower is better (original units, mg/L or pH).</p>
  <div style="margin-bottom:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="prog-btn-r2"   onclick="switchProgMetric('r2')"
      style="padding:4px 14px;border-radius:4px;cursor:pointer;border:1px solid #4A90D9;background:#4A90D9;color:#fff;font-weight:600">R²</button>
    <button id="prog-btn-rmse" onclick="switchProgMetric('rmse')"
      style="padding:4px 14px;border-radius:4px;cursor:pointer;border:1px solid var(--border-color);background:var(--card-bg);color:var(--text-color)">RMSE</button>
    <button id="prog-btn-mae"  onclick="switchProgMetric('mae')"
      style="padding:4px 14px;border-radius:4px;cursor:pointer;border:1px solid var(--border-color);background:var(--card-bg);color:var(--text-color)">MAE</button>
  </div>
  <div style="position:relative;height:380px;max-width:1000px;margin:0 auto">
    <canvas id="progressionChart"></canvas>
  </div>
  <script>
  (function() {{
    var allData   = {chart_json};
    var isDark    = document.documentElement.getAttribute('data-theme') === 'dark';
    var gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    var textColor = isDark ? '#A0A6B2' : '#555';

    var yLabels = {{ r2: 'Avg Test R²', rmse: 'Avg Test RMSE', mae: 'Avg Test MAE' }};
    var lowerBetter = {{ r2: false, rmse: true, mae: true }};

    var ctx = document.getElementById('progressionChart').getContext('2d');
    var progChart = new Chart(ctx, {{
      type: 'line',
      data: allData['r2'],
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
          y: {{ title: {{ display: true, text: 'Avg Test R²', color: textColor }},
                ticks: {{ color: textColor }},
                grid:  {{ color: gridColor }},
                suggestedMin: -0.2, suggestedMax: 0.7 }}
        }}
      }}
    }});

    window.switchProgMetric = function(metric) {{
      progChart.data = allData[metric];
      // Update y-axis label and scale hints
      var isLower = lowerBetter[metric];
      progChart.options.scales.y.title.text = yLabels[metric];
      if (metric === 'r2') {{
        progChart.options.scales.y.suggestedMin = -0.2;
        progChart.options.scales.y.suggestedMax = 0.7;
      }} else {{
        delete progChart.options.scales.y.suggestedMin;
        delete progChart.options.scales.y.suggestedMax;
      }}
      progChart.update();

      // Update button styles
      ['r2','rmse','mae'].forEach(function(m) {{
        var btn = document.getElementById('prog-btn-' + m);
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


def build_overview(df_all: pd.DataFrame) -> str:
    return f"""
<section id="overview">
  <h1 class="section-title">Overview</h1>
  {_global_leaderboard(df_all)}
  {_progression_chart(df_all)}
</section>"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_exp1_section(df_all: pd.DataFrame) -> str:
    sub1   = _exp_subsection(df_all, "Exp1", "exp1-full",
                             "Full Feature Set (Inlet + COMMON)", open_default=True)
    sub2   = _exp_subsection(df_all, "Exp1-FS", "exp1-fs",
                             "Feature Selected Variant", open_default=False)
    fs_div = _fs_analysis_div(df_all, "Exp1", "Exp1-FS")
    best   = _best_model_box(df_all[df_all["exp_key"].isin(["Exp1","Exp1-FS"])],
                             "Experiment 1")
    return f"""
<section id="exp1">
  <h1 class="section-title">Experiment 1 — Inlet + COMMON</h1>
  <p class="section-intro">{EXP_INTRO["Exp1"]}</p>
  {sub1}
  {sub2}
  {fs_div}
  {best}
</section>"""


def build_exp2_section(df_all: pd.DataFrame) -> str:
    sub1   = _exp_subsection(df_all, "Exp2-Sub1", "exp2-s1",
                             "Sub-experiment 1 — Secondary Clarifier + COMMON", open_default=True)
    fs1div = _fs_analysis_div(df_all, "Exp2-Sub1", "Exp2-Sub1-FS")
    sub1fs = _exp_subsection(df_all, "Exp2-Sub1-FS", "exp2-s1-fs",
                             "Sub-experiment 1 — Feature Selected", open_default=False)
    sub2   = _exp_subsection(df_all, "Exp2-Sub2", "exp2-s2",
                             "Sub-experiment 2 — Inlet + Secondary + COMMON", open_default=True)
    fs2div = _fs_analysis_div(df_all, "Exp2-Sub2", "Exp2-Sub2-FS")
    sub2fs = _exp_subsection(df_all, "Exp2-Sub2-FS", "exp2-s2-fs",
                             "Sub-experiment 2 — Feature Selected", open_default=False)
    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Exp2-Sub1","Exp2-Sub1-FS","Exp2-Sub2","Exp2-Sub2-FS"])],
        "Experiment 2")
    return f"""
<section id="exp2">
  <h1 class="section-title">Experiment 2 — Secondary & Combined Features</h1>
  <p class="section-intro">{EXP_INTRO["Exp2"]}</p>
  {sub1}
  {sub1fs}
  {fs1div}
  {sub2}
  {sub2fs}
  {fs2div}
  {best}
</section>"""


def build_exp3_section(df_all: pd.DataFrame) -> str:
    sub1   = _exp_subsection(df_all, "Exp3-S1", "exp3-s1",
                             "Sub-experiment 1 — ADD-tier Aeration Features", open_default=True)
    fs1div = _fs_analysis_div(df_all, "Exp3-S1", "Exp3-S1-FS")
    sub1fs = _exp_subsection(df_all, "Exp3-S1-FS", "exp3-s1-fs",
                             "Sub-experiment 1 — Feature Selected", open_default=False)
    sub2   = _exp_subsection(df_all, "Exp3-S2", "exp3-s2",
                             "Sub-experiment 2 — ADD + CONSIDER-tier Features", open_default=True)
    cost_div = _data_cost_div(df_all, "Exp3-S2", "Exp3-S1")
    vif_div  = _vif_callout()

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
  <h1 class="section-title">Experiment 3 — Expanded Feature Sets</h1>
  <p class="section-intro">{EXP_INTRO["Exp3"]}</p>
  {sub1}
  {sub1fs}
  {fs1div}
  {sub2}
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
    feat_html = _feature_card(exp_key)
    ds_html   = _dataset_summary(df)
    tbl       = _metrics_table(df, models, section_id)
    train_tbl = _train_metrics_table(df, models)
    return f"""
<details class="exp-details" id="{section_id}">
  <summary><span class="fold-icon">▶</span> {title} {badge_html}</summary>
  <div class="exp-body">
    {feat_html}
    {ds_html}
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
        # Variance collapse: σ-ratio < 0.5 AND (MAE ratio < 1.3 — not much absolute change)
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
            diagnosis  = f"Variance collapse (σ-ratio={_fmt(ratio,2)}) — R² unreliable; check MAE"
        elif is_failure and not is_collapse:
            dx_class   = "fail"
            badge_col  = "#e74c3c"
            diagnosis  = f"Genuine deterioration (MAE ×{_fmt(mae_ratio,1)} from 2024→2025)"
        else:
            dx_class   = "warn"
            badge_col  = "#e67e22"
            r_str = _fmt(ratio,2) if _safe(ratio) else "—"
            m_str = _fmt(mae_ratio,1) if _safe(mae_ratio) else "—"
            diagnosis  = f"Mixed — σ-ratio={r_str}, MAE ×{m_str}"

        tgt_short = TARGET_SHORT.get(r["target"], r["target"])

        def _delta_cell(val, is_good_if_negative=True):
            """Format a delta value with colour: green if improving, red if degrading."""
            if not _safe(val):
                return "—"
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
          <td>{"—" if not _safe(ratio) else _fmt(ratio, 2)}</td>
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
  <h4 style="margin:0 0 0.6rem">Negative R² Diagnosis — Variance Collapse vs Genuine Failure</h4>
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
        _badge("CONSISTENT — no leakage", "warn"))

    # Combined comparison across all three
    df_p9 = df_all[df_all["exp_key"].isin(["Phase9-ANN","Phase9-Voting","Phase9-Stacking"])].copy()
    all_m = [m for m in ADV_MODELS if m in df_p9["model"].values]
    comp_tbl = _metrics_table(df_p9, all_m, "p9-comp")
    best = _best_model_box(df_p9, "Phase 9")
    var_dx = _variance_diagnosis_callout()

    return f"""
<section id="phase9">
  <h1 class="section-title">Phase 9 — Advanced Models (Corrected)</h1>
  <p class="section-intro">{EXP_INTRO["Phase9"]}</p>
  {ann_sub}
  {ann_failure}
  {vote_sub}
  {stack_sub}
  <details class="exp-details" id="p9-comparison">
    <summary><span class="fold-icon">▶</span> Phase 9 — All Models Combined</summary>
    <div class="exp-body">{comp_tbl}</div>
  </details>
  {var_dx}
  {best}
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
        "Phase 10 — Full Feature Engineering (All Targets)",
        _badge("COMPOSITE OVERFIT", "fail"))
    sel_fe = _phase10_variant(
        df_all, "Phase10b-FE", "p10b",
        "Phase 10b — Selective Feature Engineering (Grab FE / Composite Base)",
        _badge("CURRENT BEST", "rec"))
    best = _best_model_box(
        df_all[df_all["exp_key"].isin(["Phase10-FE","Phase10b-FE"])],
        "Phase 10")
    return f"""
<section id="phase10">
  <h1 class="section-title">Phase 10 — Feature Engineering</h1>
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
  ~80 rows with ~55 features, making CV_R² very noisy for Ridge/ElNet — interpret per-fold
  rather than as a mean. Key finding: Grab COD RF has CV_R²=0.291, Gap_gen=−0.014 — the most
  consistent Grab COD model by cross-validation.
</div>"""

    return f"""
<section id="phase11">
  <h1 class="section-title">Phase 11 — Temporal Features + log1p Targets</h1>
  <p class="section-intro">{EXP_INTRO["Phase11"]}</p>
  <details class="exp-details" open id="p11-detail">
    <summary><span class="fold-icon">▶</span> Phase 11 — All Models</summary>
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
      <a class="nav-item nav-sub" href="#exp1-full">Full Feature Set</a>
      <a class="nav-item nav-sub" href="#exp1-fs">Feature Selected</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-exp2">
      Experiment 2 <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-exp2">
      <a class="nav-item nav-sub" href="#exp2-s1">Sub-exp 1 (Secondary)</a>
      <a class="nav-item nav-sub" href="#exp2-s1-fs">Sub-exp 1 FS</a>
      <a class="nav-item nav-sub" href="#exp2-s2">Sub-exp 2 (Combined)</a>
      <a class="nav-item nav-sub" href="#exp2-s2-fs">Sub-exp 2 FS</a>
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
      <a class="nav-item nav-sub" href="#exp3-vif">VIF Collinearity</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p9">
      Phase 9 — Advanced <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p9">
      <a class="nav-item nav-sub" href="#p9-ann">ANN</a>
      <a class="nav-item nav-sub" href="#p9-ann-diagnosis">ANN Failure Post-Mortem</a>
      <a class="nav-item nav-sub" href="#p9-voting">Voting (ElNet+RF+XGB)</a>
      <a class="nav-item nav-sub" href="#p9-stacking">Stacking (walk-fwd OOF)</a>
      <a class="nav-item nav-sub" href="#p9-comparison">Combined</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p10">
      Phase 10 — Feature Eng. <span class="nav-chevron">▾</span>
    </div>
    <div class="nav-group-items" id="nav-p10">
      <a class="nav-item nav-sub" href="#p10-full">Full FE (P10)</a>
      <a class="nav-item nav-sub" href="#p10b">Selective FE (P10b) ★</a>
    </div>
  </div>

  <div class="nav-group">
    <div class="nav-group-title nav-collapsible" data-target-group="nav-p11">
      Phase 11 — Temporal <span class="nav-chevron">▾</span>
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
    </div>
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
  .exp-body { padding: 16px 20px; }

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
  var sectionOrder = ['exp1-full','exp1-fs','exp2-s1','exp2-s1-fs','exp2-s2','exp2-s2-fs',
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
    operational interpretability — they rank identically to R² within a target but
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
    training set (2021–2024) only. Headline finding: error concentrates in
    <strong>Q4 high-flow</strong> and <strong>DJF winter</strong> regimes (2–3× overall
    MAE) for every BOD/TSS target — predictions during hydraulic or thermal stress
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
        print("  No MdAE data found — MdAE column will show — in report.")

    print("Building HTML sections…")
    print("  Building experiment and phase sections…")
    sections = [
        build_overview(df_all),
        build_exp1_section(df_all),
        build_exp2_section(df_all),
        build_exp3_section(df_all),
        build_phase9_section(df_all),
        build_phase10_section(df_all),
        build_phase11_section(df_all),
    ]

    print("  Building model selection section…")
    sections.append(_build_model_selection_section(df_all))

    print("  Building error decomposition section…")
    sections.append(_build_error_decomposition_section())

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
  <title>Unified Modeling Report — Wastewater Treatment</title>
  <style>{css}</style>
  {DARK_MODE_JS}
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  {section_data_js}
</head>
<body>
{_sidebar()}
<div id="main-content">
  <div class="page-header">
    <h1>Wastewater Treatment — Unified Modeling Report</h1>
    <p class="meta">Generated {ts} &nbsp;·&nbsp;
      Experiments 1–3 + Phases 9–11 &nbsp;·&nbsp;
      9 models &nbsp;·&nbsp; 8 effluent targets &nbsp;·&nbsp;
      Train 2021–2024 · Test 2025
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
