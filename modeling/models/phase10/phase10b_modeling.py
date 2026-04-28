"""
phase10b_modeling.py - Feature Engineering (Grab only) for Phase 10b.

Addresses the Phase 10 finding that feature engineering (log + interaction + flags)
caused catastrophic overfitting on composite targets (e.g. Comp TSS gap +3.80) due to
the high feature-to-sample ratio (~50 features vs 290 training rows).

Strategy:
  - Grab targets    → apply full feature engineering (log + interaction + flags)
  - Composite targets → base Exp3-S2 features only (no engineering)

Models: ElNet, Ridge, RF, Voting (same as Phase 10).

Outputs:
  - phase10/results_10b.xlsx
  - phase10/models_10b/<name>_<model>_run_N.pkl
  - phase10/plots_10b/<name>_<model>_run_N_{scatter,timeseries,lc}.png

Usage (from project root):
    .venv/bin/python3 21-25/modeling/models/phase10/phase10b_modeling.py
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
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models_10b")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots_10b")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results_10b.xlsx")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = [
    {"name": "grab_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_BOD.xlsx"),
     "target": "Effluent BOD (mg/L, Grab)",       "experiment": "Phase10b-FE-GrabOnly", "is_grab": True},
    {"name": "grab_COD",  "file": os.path.join(DS_DIR, "s2_stage3_grab_COD.xlsx"),
     "target": "Effluent COD (mg/L, Grab)",       "experiment": "Phase10b-FE-GrabOnly", "is_grab": True},
    {"name": "grab_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_grab_TSS.xlsx"),
     "target": "Effluent TSS (mg/L, Grab)",       "experiment": "Phase10b-FE-GrabOnly", "is_grab": True},
    {"name": "grab_pH",   "file": os.path.join(DS_DIR, "s2_stage3_grab_pH.xlsx"),
     "target": "Effluent pH (Grab)",               "experiment": "Phase10b-FE-GrabOnly", "is_grab": True},
    {"name": "comp_BOD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_BOD.xlsx"),
     "target": "Effluent BOD (mg/L, Composite)",  "experiment": "Phase10b-FE-GrabOnly", "is_grab": False},
    {"name": "comp_COD",  "file": os.path.join(DS_DIR, "s2_stage3_comp_COD.xlsx"),
     "target": "Effluent COD (mg/L, Composite)",  "experiment": "Phase10b-FE-GrabOnly", "is_grab": False},
    {"name": "comp_TSS",  "file": os.path.join(DS_DIR, "s2_stage3_comp_TSS.xlsx"),
     "target": "Effluent TSS (mg/L, Composite)",  "experiment": "Phase10b-FE-GrabOnly", "is_grab": False},
    {"name": "comp_pH",   "file": os.path.join(DS_DIR, "s2_stage3_comp_pH.xlsx"),
     "target": "Effluent pH (Composite)",          "experiment": "Phase10b-FE-GrabOnly", "is_grab": False},
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

# ── Feature engineering constants (Grab only) ──────────────────────────────────
LOG_COL_NAMES = {
    "Inlet BOD (mg/L, Grab)":                 "log_Inlet_BOD_Grab",
    "Inlet COD (mg/L, Grab)":                 "log_Inlet_COD_Grab",
    "Inlet TSS (mg/L, Grab)":                 "log_Inlet_TSS_Grab",
    "Sec Clarifier BOD (mg/L)":               "log_Sec_Clar_BOD",
    "Sec Clarifier COD (mg/L)":               "log_Sec_Clar_COD",
    "Sec Clarifier TSS (mg/L)":               "log_Sec_Clar_TSS",
    "Sec Sed BOD (mg/L)":                     "log_Sec_Sed_BOD",
    "Sec Sed COD (mg/L)":                     "log_Sec_Sed_COD",
    "Sec Sed TSS (mg/L)":                     "log_Sec_Sed_TSS",
    "Inlet Total Coliform (CFU/100ml, Grab)": "log_Inlet_Coliform",
    "Primary Sludge Totalizer (m3)":          "log_Primary_Sludge",
}

INTERACTION_PAIRS = [
    ("Inlet BOD (mg/L, Grab)", "Sec Clarifier BOD (mg/L)",       "inter_InletBOD_x_SecBOD"),
    ("Inlet COD (mg/L, Grab)", "Sec Clarifier COD (mg/L)",       "inter_InletCOD_x_SecCOD"),
    ("Inlet TSS (mg/L, Grab)", "Sec Clarifier TSS (mg/L)",       "inter_InletTSS_x_SecTSS"),
    ("Inlet BOD (mg/L, Grab)", "Flow (MLD)",                     "inter_InletBOD_x_Flow"),
    ("Sec Clarifier BOD (mg/L)", "Aeration MLSS (mg/L, Existing)", "inter_SecBOD_x_MLSS"),
    ("Aeration DO (mg/L, Existing)", "Aeration MLSS (mg/L, Existing)", "inter_DO_x_MLSS"),
]

FLAG_COL_NAMES = {
    "Inlet BOD (mg/L, Grab)":   "flag_Inlet_BOD_Grab",
    "Inlet COD (mg/L, Grab)":   "flag_Inlet_COD_Grab",
    "Inlet TSS (mg/L, Grab)":   "flag_Inlet_TSS_Grab",
    "Sec Clarifier BOD (mg/L)": "flag_Sec_Clar_BOD",
    "Sec Clarifier COD (mg/L)": "flag_Sec_Clar_COD",
    "Sec Clarifier TSS (mg/L)": "flag_Sec_Clar_TSS",
}

# ── Core helpers ───────────────────────────────────────────────────────────────

def infer_features(df: pd.DataFrame, target: str) -> list:
    exclude = {"Date", "year", target}
    return [c for c in df.columns
            if c not in exclude and not c.startswith("predicted_")]


def engineer_features(df: pd.DataFrame, train_mask: pd.Series):
    """Apply log + interaction + flag engineering. Grab-only constants used."""
    df_eng = df.copy()
    n_log, n_inter, n_flag = 0, 0, 0
    iqr_info = {}

    for src, dst in LOG_COL_NAMES.items():
        if src in df_eng.columns:
            df_eng[dst] = np.log1p(df_eng[src])
            n_log += 1

    for col_a, col_b, dst in INTERACTION_PAIRS:
        if col_a in df_eng.columns and col_b in df_eng.columns:
            df_eng[dst] = df_eng[col_a] * df_eng[col_b]
            n_inter += 1

    for src, dst in FLAG_COL_NAMES.items():
        if src in df_eng.columns:
            tv    = df_eng.loc[train_mask, src]
            q1, q3 = tv.quantile(0.25), tv.quantile(0.75)
            upper  = q3 + 1.5 * (q3 - q1)
            df_eng[dst] = (df_eng[src] > upper).astype(int)
            iqr_info[dst] = {"q1": round(q1, 3), "q3": round(q3, 3), "upper": round(upper, 3)}
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
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
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
    ax.plot(df_full["Date"], df_full[target], color="white", lw=1.2, alpha=0.8, label="Actual")
    ax.plot(df_full["Date"], y_pred_full, color="#B07FD4", lw=1.0, alpha=0.85,
            label=f"{model_tag} Predicted")
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
    try:
        ts, tr_sc, val_sc = learning_curve(
            model, X_train, y_train,
            cv=tscv, train_sizes=np.linspace(0.2, 1.0, 7),
            scoring="r2", n_jobs=1
        )
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, tr_sc.mean(axis=1),  "o-", color="#4A90D9", lw=1.5, label="Train R²")
    ax.fill_between(ts, tr_sc.mean(axis=1) - tr_sc.std(axis=1),
                        tr_sc.mean(axis=1) + tr_sc.std(axis=1), alpha=0.15, color="#4A90D9")
    ax.plot(ts, val_sc.mean(axis=1), "s-", color="#E15252", lw=1.5, label="CV Val R²")
    ax.fill_between(ts, val_sc.mean(axis=1) - val_sc.std(axis=1),
                        val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.15, color="#E15252")
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
    tscv = TimeSeriesSplit(n_splits=3)
    Xs   = StandardScaler().fit_transform(X_tr)
    best_alpha, best_score = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [r2_score(y_tr[vi], Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
                  for ti, vi in tscv.split(Xs)]
        if np.mean(scores) > best_score:
            best_score, best_alpha = np.mean(scores), a
    model = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
    model.fit(X_tr, y_tr)
    return model, {"ridge_alpha": best_alpha}


def _fit_elnet(X_tr, y_tr):
    tscv = TimeSeriesSplit(n_splits=3)
    Xs   = StandardScaler().fit_transform(X_tr)
    gs   = GridSearchCV(ElasticNet(max_iter=5000), ELNET_GRID,
                        cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(Xs, y_tr)
    bp   = gs.best_params_
    model = Pipeline([("sc", StandardScaler()),
                      ("en", ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"],
                                        max_iter=5000))])
    model.fit(X_tr, y_tr)
    return model, {"elnet_alpha": bp["alpha"], "elnet_l1_ratio": bp["l1_ratio"]}


def _fit_rf(X_tr, y_tr):
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    return rf, RF_PARAMS


def _build_voting(X_tr, y_tr):
    tscv = TimeSeriesSplit(n_splits=3)
    Xs   = StandardScaler().fit_transform(X_tr)
    best_alpha, best_sc = RIDGE_ALPHAS[0], -np.inf
    for a in RIDGE_ALPHAS:
        scores = [r2_score(y_tr[vi], Ridge(alpha=a).fit(Xs[ti], y_tr[ti]).predict(Xs[vi]))
                  for ti, vi in tscv.split(Xs)]
        if np.mean(scores) > best_sc:
            best_sc, best_alpha = np.mean(scores), a
    gs_en = GridSearchCV(ElasticNet(max_iter=5000), ELNET_GRID,
                         cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs_en.fit(Xs, y_tr)
    bp = gs_en.best_params_
    voting = Pipeline([
        ("sc", StandardScaler()),
        ("voting", VotingRegressor([
            ("ridge", Ridge(alpha=best_alpha)),
            ("elnet", ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"], max_iter=5000)),
            ("rf",    RandomForestRegressor(**RF_PARAMS)),
        ])),
    ])
    voting.fit(X_tr, y_tr)
    return voting, {"ridge_alpha": best_alpha, "elnet_alpha": bp["alpha"],
                    "elnet_l1_ratio": bp["l1_ratio"]}


MODEL_SUITE = [
    ("ElNet",  _fit_elnet),
    ("Ridge",  _fit_ridge),
    ("RF",     _fit_rf),
    ("Voting", _build_voting),
]

# ── Per-dataset training ───────────────────────────────────────────────────────

def train_dataset(ds: dict, run: int) -> list:
    name, target, is_grab = ds["name"], ds["target"], ds["is_grab"]

    df_raw = pd.read_excel(ds["file"])
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # Step 1: base dropna
    base_feats = infer_features(df_raw, target)
    df_clean   = df_raw[["Date"] + base_feats + [target]].dropna()
    train_mask = df_clean["Date"].dt.year < 2025

    # Step 2: conditionally engineer
    if is_grab:
        df_eng, n_log, n_inter, n_flag, _ = engineer_features(df_clean, train_mask)
        feat_cols = infer_features(df_eng, target)
        eng_note  = f"+{n_log} log +{n_inter} inter +{n_flag} flag → {len(feat_cols)} total"
    else:
        df_eng    = df_clean
        n_log, n_inter, n_flag = 0, 0, 0
        feat_cols = base_feats
        eng_note  = f"no engineering → {len(feat_cols)} base features"

    n_base = len(base_feats)
    n_eng  = n_log + n_inter + n_flag

    train = df_eng[train_mask]
    test  = df_eng[~train_mask]
    X_train, y_train = train[feat_cols].values, train[target].values
    X_test,  y_test  = test[feat_cols].values,  test[target].values

    print(f"  {'[GRAB-FE]' if is_grab else '[COMP-BASE]'} base={n_base}  {eng_note}"
          f"  | train={len(train)} test={len(test)}")

    records = []
    for model_tag, builder_fn in MODEL_SUITE:
        print(f"    [{model_tag}] fitting...", end=" ", flush=True)
        try:
            model, params = builder_fn(X_train, y_train)
            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)
            r2_tr, rmse_tr, mae_tr, _    = _metrics(y_train, y_tr_pred)
            r2_te, rmse_te, mae_te, mape = _metrics(y_test,  y_te_pred)
            gap = r2_tr - r2_te
            print(f"Train R²={r2_tr:+.3f}  Test R²={r2_te:+.3f}  Gap={gap:+.3f}")

            pkl_path = os.path.join(MODELS_DIR,
                                    f"{name}_{model_tag.lower()}_run_{run}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

            _scatter_plot(y_train, y_tr_pred, y_test, y_te_pred,
                          r2_tr, r2_te, name, model_tag, run)
            _timeseries_plot(df_eng, target,
                             model.predict(df_eng[feat_cols].values),
                             name, model_tag, run)
            _learning_curve_plot(model, X_train, y_train, name, model_tag, run)

            parts = name.split("_")
            records.append({
                "experiment":      ds["experiment"],
                "model_name":      f"p10b_{model_tag.lower()}_{parts[-2]}_{parts[-1]}",
                "run":             run,
                "model":           model_tag,
                "target":          target,
                "is_grab":         is_grab,
                "n_train":         len(train),
                "n_test":          len(test),
                "n_features":      len(feat_cols),
                "n_base_features": n_base,
                "n_engineered":    n_eng,
                "n_log":           n_log,
                "n_inter":         n_inter,
                "n_flag":          n_flag,
                "R2_train":        r2_tr,
                "RMSE_train":      rmse_tr,
                "MAE_train":       mae_tr,
                "R2_test":         r2_te,
                "RMSE_test":       rmse_te,
                "MAE_test":        mae_te,
                "MAPE_test":       mape,
                "R2_gap":          gap,
                "best_params":     str(params),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    run = _next_run(RESULTS_FILE)
    print(f"=== Phase 10b - Feature Engineering (Grab only) - Run {run} ===")
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
            if not sub:
                continue
            avg      = np.mean([r["R2_test"] for r in sub])
            avg_grab = np.mean([r["R2_test"] for r in sub if r["is_grab"]])
            avg_comp = np.mean([r["R2_test"] for r in sub if not r["is_grab"]])
            print(f"\n  {model_tag}  avg={avg:+.3f}  (grab={avg_grab:+.3f}  comp={avg_comp:+.3f})")
            for r in sub:
                tag = "[G]" if r["is_grab"] else "[C]"
                print(f"    {tag} {r['target']:48s}  Test R²={r['R2_test']:+.3f}  Gap={r['R2_gap']:+.3f}")

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
