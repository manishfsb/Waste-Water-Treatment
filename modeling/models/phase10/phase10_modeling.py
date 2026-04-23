"""
phase10_modeling.py - Feature Engineering + Model Re-run for Phase 10.

Loads Experiment 3 Sub-2 datasets and engineers three types of new features
in-memory (no new xlsx files created):
  1. Log transforms  (log1p) - concentration columns & coliform
  2. Interaction terms (A×B) - domain-relevant pairs (inlet×secondary, BOD×flow, etc.)
  3. Outlier indicator flags (binary IQR) - IQR computed from train split only

Models trained: ElNet, Ridge, RF, Voting (RF + Ridge + ElNet ensemble).
Stacking excluded - showed instability on composite targets in Phase 9.

Outputs:
  - phase10/results.xlsx
  - phase10/models/<name>_<model>_run_N.pkl
  - phase10/plots/<name>_<model>_run_N_{scatter,timeseries,lc}.png

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase10/phase10_modeling.py
"""

import os
import pickle
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DS_DIR       = os.path.join(MODELING_DIR, "datasets", "experiment3", "sub_exp2")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.xlsx")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = [
    {"name": "grab_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_BOD.xlsx"),
     "target": "Effluent BOD (mg/L, Grab)",       "experiment": "Phase10-FE"},
    {"name": "grab_COD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_COD.xlsx"),
     "target": "Effluent COD (mg/L, Grab)",       "experiment": "Phase10-FE"},
    {"name": "grab_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_grab_TSS.xlsx"),
     "target": "Effluent TSS (mg/L, Grab)",       "experiment": "Phase10-FE"},
    {"name": "grab_pH",   "file": os.path.join(DS_DIR, "s2_stage3_grab_pH.xlsx"),
     "target": "Effluent pH (Grab)",               "experiment": "Phase10-FE"},
    {"name": "comp_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_BOD.xlsx"),
     "target": "Effluent BOD (mg/L, Composite)",  "experiment": "Phase10-FE"},
    {"name": "comp_COD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_COD.xlsx"),
     "target": "Effluent COD (mg/L, Composite)",  "experiment": "Phase10-FE"},
    {"name": "comp_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_comp_TSS.xlsx"),
     "target": "Effluent TSS (mg/L, Composite)",  "experiment": "Phase10-FE"},
    {"name": "comp_pH",   "file": os.path.join(DS_DIR, "s2_stage3_comp_pH.xlsx"),
     "target": "Effluent pH (Composite)",          "experiment": "Phase10-FE"},
]

# ── Hyperparameter configs ─────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ELNET_GRID   = {
    "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "max_iter": [5000],
}
RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt",
    max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1
)

# ── Feature engineering constants ─────────────────────────────────────────────

# 1. Log transforms - applied to skewed concentration columns (log1p handles zeros)
LOG_COL_NAMES = {
    "Inlet BOD (mg/L, Grab)":                "log_Inlet_BOD_Grab",
    "Inlet COD (mg/L, Grab)":                "log_Inlet_COD_Grab",
    "Inlet TSS (mg/L, Grab)":                "log_Inlet_TSS_Grab",
    "Inlet BOD (mg/L, Composite)":           "log_Inlet_BOD_Comp",
    "Inlet COD (mg/L, Composite)":           "log_Inlet_COD_Comp",
    "Inlet TSS (mg/L, Composite)":           "log_Inlet_TSS_Comp",
    "Sec Clarifier BOD (mg/L)":              "log_Sec_Clar_BOD",
    "Sec Clarifier COD (mg/L)":              "log_Sec_Clar_COD",
    "Sec Clarifier TSS (mg/L)":              "log_Sec_Clar_TSS",
    "Sec Sed BOD (mg/L)":                    "log_Sec_Sed_BOD",
    "Sec Sed COD (mg/L)":                    "log_Sec_Sed_COD",
    "Sec Sed TSS (mg/L)":                    "log_Sec_Sed_TSS",
    "Inlet Total Coliform (CFU/100ml, Grab)":"log_Inlet_Coliform",
    "Primary Sludge Totalizer (m3)":         "log_Primary_Sludge",
}
# NOT log-transformed: pH columns, Flow (MLD), Power, DO, MLSS, SVI, SV30

# 2. Interaction terms - domain-meaningful products (A × B)
#    Each tuple: (col_A, col_B, output_col_name)
#    Grab and Composite share the same output name for same-physics interactions.
INTERACTION_PAIRS = [
    # Removal efficiency proxies (inlet load × secondary load)
    ("Inlet BOD (mg/L, Grab)",      "Sec Clarifier BOD (mg/L)",       "inter_InletBOD_x_SecBOD"),
    ("Inlet COD (mg/L, Grab)",      "Sec Clarifier COD (mg/L)",       "inter_InletCOD_x_SecCOD"),
    ("Inlet TSS (mg/L, Grab)",      "Sec Clarifier TSS (mg/L)",       "inter_InletTSS_x_SecTSS"),
    ("Inlet BOD (mg/L, Composite)", "Sec Clarifier BOD (mg/L)",       "inter_InletBOD_x_SecBOD"),
    ("Inlet COD (mg/L, Composite)", "Sec Clarifier COD (mg/L)",       "inter_InletCOD_x_SecCOD"),
    ("Inlet TSS (mg/L, Composite)", "Sec Clarifier TSS (mg/L)",       "inter_InletTSS_x_SecTSS"),
    # BOD loading = concentration × flow
    ("Inlet BOD (mg/L, Grab)",      "Flow (MLD)",                     "inter_InletBOD_x_Flow"),
    ("Inlet BOD (mg/L, Composite)", "Flow (MLD)",                     "inter_InletBOD_x_Flow"),
    # Biomass loading = secondary BOD × MLSS
    ("Sec Clarifier BOD (mg/L)",    "Aeration MLSS (mg/L, Existing)", "inter_SecBOD_x_MLSS"),
    # Oxygen availability relative to biomass
    ("Aeration DO (mg/L, Existing)","Aeration MLSS (mg/L, Existing)", "inter_DO_x_MLSS"),
]

# 3. Outlier indicator flags - 1 if > Q3 + 1.5×IQR (IQR from train only)
FLAG_COL_NAMES = {
    "Inlet BOD (mg/L, Grab)":      "flag_Inlet_BOD_Grab",
    "Inlet COD (mg/L, Grab)":      "flag_Inlet_COD_Grab",
    "Inlet TSS (mg/L, Grab)":      "flag_Inlet_TSS_Grab",
    "Inlet BOD (mg/L, Composite)": "flag_Inlet_BOD_Comp",
    "Inlet COD (mg/L, Composite)": "flag_Inlet_COD_Comp",
    "Inlet TSS (mg/L, Composite)": "flag_Inlet_TSS_Comp",
    "Sec Clarifier BOD (mg/L)":    "flag_Sec_Clar_BOD",
    "Sec Clarifier COD (mg/L)":    "flag_Sec_Clar_COD",
    "Sec Clarifier TSS (mg/L)":    "flag_Sec_Clar_TSS",
}

# ── Core helpers ───────────────────────────────────────────────────────────────

def infer_features(df: pd.DataFrame, target: str) -> list:
    """Return all usable numeric feature columns (excludes date, temporal derivations,
    target, and any prediction columns)."""
    exclude = {"Date", "year", "month", "day_of_week", target}
    return [c for c in df.columns
            if c not in exclude and not c.startswith("predicted_")]


def engineer_features(df: pd.DataFrame, train_mask: pd.Series):
    """
    Add log_, inter_, and flag_ columns to a copy of df.

    Parameters
    ----------
    df         : DataFrame AFTER initial dropna (base cols only, no NaNs).
    train_mask : Boolean Series (True = training rows). Used for IQR - no leakage.

    Returns
    -------
    df_eng  : Augmented DataFrame
    n_log   : Number of log columns added
    n_inter : Number of interaction columns added
    n_flag  : Number of flag columns added
    iqr_info: Dict {flag_col: {"q1", "q3", "upper"}} for audit/logging
    """
    df_eng = df.copy()
    n_log, n_inter, n_flag = 0, 0, 0
    iqr_info = {}

    # 1. Log transforms
    for src, dst in LOG_COL_NAMES.items():
        if src in df_eng.columns:
            df_eng[dst] = np.log1p(df_eng[src])
            n_log += 1

    # 2. Interaction terms
    for col_a, col_b, dst in INTERACTION_PAIRS:
        if col_a in df_eng.columns and col_b in df_eng.columns:
            df_eng[dst] = df_eng[col_a] * df_eng[col_b]
            n_inter += 1

    # 3. Outlier flags - IQR from train only
    for src, dst in FLAG_COL_NAMES.items():
        if src in df_eng.columns:
            train_vals = df_eng.loc[train_mask, src]
            q1  = train_vals.quantile(0.25)
            q3  = train_vals.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            df_eng[dst] = (df_eng[src] > upper).astype(int)
            iqr_info[dst] = {"q1": round(q1, 3), "q3": round(q3, 3),
                             "iqr": round(iqr, 3), "upper": round(upper, 3)}
            n_flag += 1

    return df_eng, n_log, n_inter, n_flag, iqr_info


def _mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    return r2, rmse, mae, mape


def _next_run(results_file: str) -> int:
    if not os.path.exists(results_file):
        return 1
    df = pd.read_excel(results_file)
    return int(df["run"].max()) + 1 if len(df) > 0 else 1


# ── Plotting ───────────────────────────────────────────────────────────────────

def _scatter_plot(y_train, y_train_pred, y_test, y_test_pred,
                  r2_train, r2_test, name, model_tag, run):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, yt, yp, r2, label, color in [
        (ax1, y_train, y_train_pred, r2_train, "Train", "#4A90D9"),
        (ax2, y_test,  y_test_pred,  r2_test,  "Test",  "#E15252"),
    ]:
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "w--", lw=0.8, alpha=0.5)
        ax.scatter(yt, yp, alpha=0.55, s=18, color=color)
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{label} - R²={r2:+.3f}", fontsize=9)
    fig.suptitle(f"{model_tag} - {name}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_scatter.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _timeseries_plot(df_full, target, y_pred_full, name, model_tag, run):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_full["Date"], df_full[target], color="white", lw=1.2,
            alpha=0.8, label="Actual")
    ax.plot(df_full["Date"], y_pred_full, color="#B07FD4", lw=1.0,
            alpha=0.85, label=f"{model_tag} Predicted")
    split_date = df_full.loc[df_full["Date"].dt.year == 2025, "Date"].min()
    if pd.notna(split_date):
        ax.axvline(split_date, color="#F0B849", lw=1.2, linestyle="--",
                   alpha=0.7, label="Train | Test")
    ax.set_title(f"{model_tag} - {target}", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_timeseries.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _learning_curve_plot(model, X_train, y_train, name, model_tag, run):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes = np.linspace(0.2, 1.0, 7)
    try:
        ts, train_sc, val_sc = learning_curve(
            model, X_train, y_train,
            cv=tscv, train_sizes=train_sizes,
            scoring="r2", n_jobs=1
        )
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, train_sc.mean(axis=1), "o-", color="#4A90D9", lw=1.5, label="Train R²")
    ax.fill_between(ts,
                    train_sc.mean(axis=1) - train_sc.std(axis=1),
                    train_sc.mean(axis=1) + train_sc.std(axis=1),
                    alpha=0.15, color="#4A90D9")
    ax.plot(ts, val_sc.mean(axis=1), "s-", color="#E15252", lw=1.5, label="CV Val R²")
    ax.fill_between(ts,
                    val_sc.mean(axis=1) - val_sc.std(axis=1),
                    val_sc.mean(axis=1) + val_sc.std(axis=1),
                    alpha=0.15, color="#E15252")
    ax.axhline(0, color="white", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Training examples", fontsize=9)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title(f"Learning Curve - {model_tag} - {name}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_{model_tag}_run_{run}_lc.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Model builders ─────────────────────────────────────────────────────────────

def _fit_ridge(X_tr, y_tr):
    """Ridge with best alpha from TimeSeriesSplit CV."""
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    best_alpha, best_score = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = []
        for ti, vi in tscv.split(Xs):
            m = Ridge(alpha=a)
            m.fit(Xs[ti], y_tr[ti])
            scores.append(r2_score(y_tr[vi], m.predict(Xs[vi])))
        if np.mean(scores) > best_score:
            best_score, best_alpha = np.mean(scores), a
    model = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    model.fit(X_tr, y_tr)
    return model, {"ridge_alpha": best_alpha}


def _fit_elnet(X_tr, y_tr):
    """ElasticNet with GridSearchCV."""
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    gs = GridSearchCV(ElasticNet(max_iter=5000), ELNET_GRID,
                      cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(Xs, y_tr)
    bp = gs.best_params_
    model = Pipeline([("sc", StandardScaler()),
                      ("en", ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"],
                                        max_iter=5000))])
    model.fit(X_tr, y_tr)
    return model, {"elnet_alpha": bp["alpha"], "elnet_l1_ratio": bp["l1_ratio"]}


def _fit_rf(X_tr, y_tr):
    """RF with fixed params (same as Phase 9)."""
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    return rf, RF_PARAMS


def _build_voting(X_tr, y_tr):
    """VotingRegressor: Ridge (tuned) + ElNet (tuned) + RF (fixed)."""
    tscv = TimeSeriesSplit(n_splits=3)
    scaler_tmp = StandardScaler()
    Xs = scaler_tmp.fit_transform(X_tr)

    # Tune Ridge alpha
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [r2_score(y_tr[vi],
                           Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
                  for ti, vi in tscv.split(Xs)]
        if np.mean(scores) > best_sc:
            best_sc, best_alpha = np.mean(scores), a

    # Tune ElNet
    gs_en = GridSearchCV(ElasticNet(max_iter=5000), ELNET_GRID,
                         cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_en.fit(Xs, y_tr)
    bp = gs_en.best_params_

    estimators = [
        ("ridge", Ridge(alpha=best_alpha)),
        ("elnet", ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"], max_iter=5000)),
        ("rf",    RandomForestRegressor(**RF_PARAMS)),
    ]
    voting = Pipeline([("sc", StandardScaler()),
                       ("voting", VotingRegressor(estimators=estimators))])
    voting.fit(X_tr, y_tr)
    return voting, {"ridge_alpha": best_alpha, "elnet_alpha": bp["alpha"],
                    "elnet_l1_ratio": bp["l1_ratio"]}


# ── Per-dataset training ───────────────────────────────────────────────────────

MODEL_SUITE = [
    ("ElNet",  _fit_elnet),
    ("Ridge",  _fit_ridge),
    ("RF",     _fit_rf),
    ("Voting", _build_voting),
]


def train_dataset(ds: dict, run: int) -> list:
    name   = ds["name"]
    target = ds["target"]

    df_raw = pd.read_excel(ds["file"])
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # Step 1: initial dropna on base features only
    base_feats = infer_features(df_raw, target)
    df_clean   = df_raw[["Date"] + base_feats + [target]].dropna()

    # Step 2: train mask - computed BEFORE engineering (IQR leakage prevention)
    train_mask = df_clean["Date"].dt.year < 2025

    # Step 3: engineer features in-memory
    df_eng, n_log, n_inter, n_flag, iqr_info = engineer_features(df_clean, train_mask)

    # Step 4: infer final feature list (auto-picks up log_/inter_/flag_ cols)
    feat_cols = infer_features(df_eng, target)
    n_base    = len(base_feats)
    n_eng     = n_log + n_inter + n_flag

    train = df_eng[train_mask]
    test  = df_eng[~train_mask]

    X_train = train[feat_cols].values
    y_train = train[target].values
    X_test  = test[feat_cols].values
    y_test  = test[target].values

    print(f"  base={n_base} feats, +{n_log} log +{n_inter} inter +{n_flag} flag "
          f"→ total={len(feat_cols)} | train={len(train)} test={len(test)}")

    records = []
    for model_tag, builder_fn in MODEL_SUITE:
        print(f"    [{model_tag}] fitting...", end=" ", flush=True)
        try:
            result = builder_fn(X_train, y_train)
            model, params = result[0], result[1]

            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)

            r2_tr, rmse_tr, mae_tr, _    = _metrics(y_train, y_tr_pred)
            r2_te, rmse_te, mae_te, mape = _metrics(y_test,  y_te_pred)
            gap = r2_tr - r2_te

            print(f"Train R²={r2_tr:+.3f}  Test R²={r2_te:+.3f}  Gap={gap:+.3f}")

            # Save model
            pkl_path = os.path.join(MODELS_DIR,
                                    f"{name}_{model_tag.lower()}_run_{run}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

            # Plots
            _scatter_plot(y_train, y_tr_pred, y_test, y_te_pred,
                          r2_tr, r2_te, name, model_tag, run)
            _timeseries_plot(df_eng, target,
                             model.predict(df_eng[feat_cols].values),
                             name, model_tag, run)
            _learning_curve_plot(model, X_train, y_train, name, model_tag, run)

            # Determine target type slug for model_name
            parts = name.split("_")  # e.g. ['grab', 'BOD']
            ttype = parts[-2] if len(parts) >= 2 else "unk"
            tslug = parts[-1] if len(parts) >= 1 else "unk"

            records.append({
                "experiment":     ds["experiment"],
                "model_name":     f"p10_{model_tag.lower()}_{ttype}_{tslug}",
                "run":            run,
                "model":          model_tag,
                "target":         target,
                "n_train":        len(train),
                "n_test":         len(test),
                "n_features":     len(feat_cols),
                "n_base_features":n_base,
                "n_engineered":   n_eng,
                "n_log":          n_log,
                "n_inter":        n_inter,
                "n_flag":         n_flag,
                "R2_train":       r2_tr,
                "RMSE_train":     rmse_tr,
                "MAE_train":      mae_tr,
                "R2_test":        r2_te,
                "RMSE_test":      rmse_te,
                "MAE_test":       mae_te,
                "MAPE_test":      mape,
                "R2_gap":         gap,
                "best_params":    str(params),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run = _next_run(RESULTS_FILE)
    print(f"=== Phase 10 - Feature Engineering (log + interaction + flag) - Run {run} ===")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    all_records = []
    for ds in DATASETS:
        print(f"\n[{ds['name']}]  target: {ds['target']}")
        try:
            recs = train_dataset(ds, run)
            all_records.extend(recs)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if all_records:
        results = pd.DataFrame(all_records)
        if os.path.exists(RESULTS_FILE):
            old = pd.read_excel(RESULTS_FILE)
            results = pd.concat([old, results], ignore_index=True)
        results.to_excel(RESULTS_FILE, index=False)
        print(f"\nResults → {RESULTS_FILE}  ({len(all_records)} rows)")

        print("\n=== Summary ===")
        for model_tag in ["ElNet", "Ridge", "RF", "Voting"]:
            sub = [r for r in all_records if r["model"] == model_tag]
            if sub:
                avg = np.mean([r["R2_test"] for r in sub])
                best = max(sub, key=lambda r: r["R2_test"])
                print(f"\n  {model_tag} (avg Test R²={avg:+.3f}, best={best['target'].split('(')[0].strip()} {best['R2_test']:+.3f}):")
                for r in sub:
                    print(f"    {r['target']:48s}  Test R²={r['R2_test']:+.3f}  Gap={r['R2_gap']:+.3f}")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
