"""
run_thesis_experiments.py
=========================

Complete experimental pipeline for

    "A Data Driven Approach to Binary Distillation Column Modelling
     Using Machine Learning", C. Naidoo, MEng thesis.

Running this file reproduces every table and figure in Chapters 4 and 5 and
every verification test in Appendix B, from the three deposited datasets.

    python run_thesis_experiments.py --data-dir . --out revision_outputs

Add --fast for a reduced hyperparameter search when checking that the pipeline
runs; the results reported in the thesis use the full search.

Outputs
    <out>/tables/    every table, as CSV
    <out>/figures/   every figure, 300 dpi PNG
    <out>/appendix_C_environment.txt
    <out>/MANIFEST.csv

Requires  numpy, pandas, scipy, scikit-learn, matplotlib, and optionally
seaborn (palette) and xgboost (one of the ten models).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

# ============================================================================
# Configuration
# ============================================================================
RANDOM_STATE = 42
SEEDS = [42, 7, 13, 2024, 99]
TARGET = "Distillate_Heavy_Fraction"

FEATURES_FULL = [
    "Feed_Flow_kg_h", "Feed_Heavy_Fraction", "Column_Pressure_kPa", "Reflux_Ratio",
    "Num_Stages", "Murphree_Efficiency", "Relative_Volatility", "Effective_Stages",
    "Separation_Power", "Distillate_to_Feed", "Reflux_to_Feed",
]
FEATURES_SAMPLED = [
    "Feed_Flow_kg_h", "Feed_Heavy_Fraction", "Column_Pressure_kPa",
    "Reflux_Ratio", "Num_Stages", "Murphree_Efficiency",
]
FEATURES_NO_SP = [f for f in FEATURES_FULL if f != "Separation_Power"]

SENS_VARS = ["Reflux_Ratio", "Num_Stages", "Murphree_Efficiency", "Feed_Heavy_Fraction",
             "Feed_Flow_kg_h", "Relative_Volatility", "Column_Pressure_kPa"]
STEPS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

SHOW_TITLES = False          # figure numbers live in the Word caption, not the image

LABEL = {
    "Feed_Flow_kg_h": "Feed flow (kg/h)", "Feed_Heavy_Fraction": "Feed heavy fraction",
    "Column_Pressure_kPa": "Pressure (kPa)", "Reflux_Ratio": "Reflux ratio",
    "Num_Stages": "Number of trays", "Murphree_Efficiency": "Murphree efficiency",
    "Relative_Volatility": "Relative volatility", "Effective_Stages": "Effective stages",
    "Separation_Power": "Separation power", "Distillate_to_Feed": "Distillate-to-feed ratio",
    "Reflux_to_Feed": "Reflux-to-feed ratio", TARGET: "Distillate heavy fraction",
}

try:
    import seaborn as sns
    CREST = [matplotlib.colors.to_hex(sns.color_palette("crest", as_cmap=True)(x))
             for x in (0.05, 0.25, 0.45, 0.65, 0.85, 1.0)]
    FLARE = [matplotlib.colors.to_hex(sns.color_palette("flare", as_cmap=True)(x))
             for x in (0.15, 0.45, 0.75)]
    VLAG = sns.color_palette("vlag", as_cmap=True)
except Exception:
    CREST = ["#a5cd90", "#66ae90", "#3d908c", "#22738b", "#254c80", "#2c3172"]
    FLARE = ["#e98b6a", "#c14168", "#6e2c6d"]
    VLAG = "RdBu_r"
DSET = [CREST[4], CREST[2], CREST[0]]
ACCENT = FLARE[0]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=CREST),
})

EPS = 1e-9
MANIFEST: list[tuple[str, str, str]] = []


# ============================================================================
# Small helpers
# ============================================================================
def c01(x):
    return np.clip(x, EPS, 1.0 - EPS)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    return {"R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "Bias": float(np.mean(err)),
            "SD_resid": float(np.std(err)),
            "Bias_frac_of_MSE": float(np.mean(err) ** 2 / max(mse, 1e-30))}


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def save_table(df, name, description, tabdir):
    p = tabdir / f"{name}.csv"
    df.to_csv(p, index=False)
    MANIFEST.append((f"tables/{name}.csv", description, "table"))
    print(f"  -> {p.name}")
    return df


def savefig(name, description, figdir):
    p = figdir / f"{name}.png"
    plt.savefig(p)
    plt.close()
    MANIFEST.append((f"figures/{name}.png", description, "figure"))
    print(f"  -> {p.name}")


def title(ax, text):
    if SHOW_TITLES:
        ax.set_title(text)


def suptitle(text, **kw):
    if SHOW_TITLES:
        plt.suptitle(text, **kw)


def label_bars(ax, bars, fmt="%.3f", fontsize=7.5, rotation=90, padding=2):
    """ax.bar_label needs matplotlib 3.4; fall back for older versions."""
    try:
        ax.bar_label(bars, fmt=fmt, fontsize=fontsize, rotation=rotation, padding=padding)
    except AttributeError:
        for b in bars:
            h = b.get_height()
            ax.annotate(fmt % h, xy=(b.get_x() + b.get_width() / 2.0, h),
                        xytext=(0, padding), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize, rotation=rotation)


def no_xgrid(ax):
    ax.grid(False, axis="x")


# ============================================================================
# Part 1 - first-principles model library
# ============================================================================
def theta_asis(alpha, z_LK, q=1.0):
    """Underwood root as originally implemented. Solves z_L/(a-t)+z_H/(1-t)=1,
    which is not the Underwood equation. Retained to report the effect."""
    a, zL = float(alpha), float(z_LK)
    disc = max(a * a - 4.0 * zL * (a - 1.0), 1e-14)
    return min(max(0.5 * (a + math.sqrt(disc)), 1.0 + 1e-9), a - 1e-9)


def theta_underwood(alpha, z_LK, q=1.0):
    """Underwood first equation, binary, heavy key as reference:
        alpha z_L/(alpha-t) + z_H/(1-t) = 1-q
    which clears to  (1-q)t^2 + [(alpha z_L + z_H)-(1-q)(alpha+1)]t - alpha q = 0."""
    a, zL = float(alpha), float(z_LK)
    zH = 1.0 - zL
    if abs(1.0 - q) < 1e-12:
        t = a / (a * zL + zH)
    else:
        r = np.roots([(1.0 - q), (a * zL + zH) - (1.0 - q) * (a + 1.0), -a * q])
        r = r[np.abs(r.imag) < 1e-9].real
        inside = r[(r > 1.0) & (r < a)]
        t = float(inside[0]) if len(inside) else float(r[np.argmin(np.abs(r - 0.5 * (1 + a)))])
    return min(max(t, 1.0 + 1e-12), a - 1e-12)


def underwood_residual(theta, alpha, z_LK, q=1.0):
    return alpha * z_LK / (alpha - theta) + (1 - z_LK) / (1 - theta) - (1 - q)


def fenske_Nmin(alpha, xD_H, xB_H):
    xD_H, xB_H = c01(xD_H), c01(xB_H)
    return np.log(((1 - xD_H) / xD_H) * (xB_H / (1 - xB_H))) / np.log(np.maximum(alpha, 1 + 1e-12))


def Rmin_underwood(theta, alpha, xD_H):
    xD_H = c01(xD_H)
    return (1 - xD_H) * alpha / (alpha - theta) + xD_H / (1.0 - theta) - 1.0


def gilliland_davis(X):
    t = np.clip(X, 0.0, 1.0) ** 0.0031
    d = 1.0 - 0.99357 * t
    return (1.0 - t) / np.where(np.abs(d) < 1e-12, 1e-12, d)


def gilliland_eduljee(X):
    return 0.75 * (1.0 - np.clip(X, 0.0, 1.0) ** 0.5668)


def gilliland_molokanov(X):
    X = np.clip(X, 1e-12, 1.0)
    return 1.0 - np.exp(((1 + 54.4 * X) / (11 + 117.2 * X)) * ((X - 1) / np.sqrt(X)))


def solve_bisect(resid_fn, n, lo0, hi0, iters=80, n_scan=61):
    """Bisection that first checks a root exists. Rows with no root are clamped
    to the boundary minimising |residual|, never to the midpoint of the bracket."""
    xs = np.linspace(lo0, hi0, n_scan)
    Fm = np.stack([resid_fn(np.full(n, x)) for x in xs])
    ends = (Fm[0] * Fm[-1]) <= 0
    change = (np.sign(Fm[:-1]) * np.sign(Fm[1:])) <= 0
    interior = change.any(axis=0)
    first = np.argmax(change, axis=0)
    solved = ends | interior
    lo = np.where(ends, xs[0], np.where(interior, xs[first], xs[0]))
    hi = np.where(ends, xs[-1], np.where(interior, xs[first + 1], xs[-1]))
    f_lo = resid_fn(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = resid_fn(mid)
        left = (f_lo * f_mid) <= 0
        hi, lo = np.where(left, mid, hi), np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    boundary = np.where(np.abs(Fm[0]) <= np.abs(Fm[-1]), xs[0], xs[-1])
    return np.where(solved, 0.5 * (lo + hi), boundary), solved


def fug_predict(df, mode="corrected", q=1.0, alpha_scale=1.0, a_map=1.0, b_add=1.0,
                return_status=False):
    zF_H = df["Feed_Heavy_Fraction"].values.astype(float)
    alpha = np.maximum(df["Relative_Volatility"].values.astype(float) * alpha_scale, 1.0001)
    R = df["Reflux_Ratio"].values.astype(float)
    N = np.maximum(a_map * (df["Murphree_Efficiency"].values.astype(float)
                            * df["Num_Stages"].values.astype(float)) + b_add, 1.01)
    F = df["Feed_Flow_kg_h"].values.astype(float)
    D = np.maximum(df["Distillate_Rate_kg_h"].values.astype(float), 1e-12)
    Bf = np.maximum(df["Bottoms_Rate_kg_h"].values.astype(float), 1e-12)
    tfn = theta_asis if mode == "asis" else theta_underwood
    theta = np.array([tfn(a, 1.0 - z, q) for a, z in zip(alpha, zF_H)])

    def resid(xD):
        xD = c01(xD)
        xB = c01((F * zF_H - D * xD) / Bf)
        Y = np.clip((N - fenske_Nmin(alpha, xD, xB)) / (N + 1.0), 0.0, 1.0)
        X = np.clip((R - Rmin_underwood(theta, alpha, xD)) / (R + 1.0), 0.0, 1.0)
        return Y - gilliland_davis(X)

    root, solved = solve_bisect(resid, len(df), 1e-8, 0.95)
    return (root, solved) if return_status else root


def y_eq(alpha, x):
    x = c01(x)
    return alpha * x / (1.0 + (alpha - 1.0) * x)


def x_eq(alpha, y):
    y = c01(y)
    return c01(y / (alpha - (alpha - 1.0) * y))


def invert_pseudo(Y, alpha, E, m, c):
    """Invert (1-E)(m x + c) + E alpha x/(1+(alpha-1)x) = Y in closed form."""
    a1 = (1.0 - E) * m * (alpha - 1.0)
    b1 = (1.0 - E) * (m + c * (alpha - 1.0)) + E * alpha - Y * (alpha - 1.0)
    c1 = (1.0 - E) * c - Y
    lin = np.abs(a1) < 1e-14
    x_lin = np.where(np.abs(b1) < 1e-14, 0.5, -c1 / np.where(np.abs(b1) < 1e-14, 1.0, b1))
    disc = np.maximum(b1 * b1 - 4.0 * a1 * c1, 0.0)
    sq = np.sqrt(disc)
    den = np.where(lin, 1.0, 2.0 * a1)
    r1, r2 = (-b1 + sq) / den, (-b1 - sq) / den
    x_quad = np.where((r1 > 0) & (r1 < 1), r1, r2)
    return c01(np.where(lin, x_lin, x_quad))


def mesh_march(xD_H, df, alpha_scale=1.0, eff_scale=1.0, stage_add=1.0):
    """McCabe-Thiele march from the condenser down, stepping from the operating
    line onto the Murphree pseudo-equilibrium curve."""
    zF_H = df["Feed_Heavy_Fraction"].values.astype(float)
    zF_L = 1.0 - zF_H
    alpha = np.maximum(df["Relative_Volatility"].values.astype(float) * alpha_scale, 1.0001)
    R = np.maximum(df["Reflux_Ratio"].values.astype(float), 1e-6)
    Nt = df["Num_Stages"].values.astype(float)
    E = np.clip(df["Murphree_Efficiency"].values.astype(float) * eff_scale, 1e-4, 1.0)
    F = df["Feed_Flow_kg_h"].values.astype(float)
    D = np.maximum(df["Distillate_Rate_kg_h"].values.astype(float), 1e-9)
    Bf = np.maximum(df["Bottoms_Rate_kg_h"].values.astype(float), 1e-9)

    xD_L = c01(1.0 - xD_H)
    xB_H_t = c01((F * zF_H - D * xD_H) / Bf)
    xB_L_t = 1.0 - xB_H_t

    L_R = R * D
    V_R = L_R + D
    L_S = L_R + F                      # q = 1
    m_r, c_r = R / (R + 1.0), xD_L / (R + 1.0)
    m_s = L_S / np.maximum(V_R, 1e-12)
    # y = (L'/V')x - (B/V')x_B with B/V' = m_s - 1, so the line passes through
    # (x_B, x_B) on the diagonal.
    c_s = -(m_s - 1.0) * xB_L_t

    n_steps = np.maximum(np.round(Nt).astype(int) + int(stage_add), 1)
    x = xD_L.copy()
    for k in range(int(n_steps.max())):
        active = k < n_steps
        rect = x > zF_L
        m = np.where(rect, m_r, m_s)
        c = np.where(rect, c_r, c_s)
        y_up = c01(m * x + c)
        x = np.where(active, invert_pseudo(y_up, alpha, E, m, c), x)
    return c01(x), xB_L_t


def mesh_predict(df, alpha_scale=1.0, eff_scale=1.0, stage_add=1.0, return_status=False):
    def resid(xD):
        p, t = mesh_march(xD, df, alpha_scale, eff_scale, stage_add)
        return p - t
    root, solved = solve_bisect(resid, len(df), 1e-6, 0.99, iters=60, n_scan=41)
    return (root, solved) if return_status else root


# ============================================================================
# Machine-learning models
# ============================================================================
def build_models(seed, fast=False):
    M = [
        ("Linear", Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
         {"m__fit_intercept": [True]}),
        ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(random_state=seed))]),
         {"m__alpha": [0.01, 0.1, 1.0, 10.0]}),
        ("Lasso", Pipeline([("s", StandardScaler()), ("m", Lasso(random_state=seed, max_iter=5000))]),
         {"m__alpha": [0.0001, 0.001, 0.01, 0.1]}),
        ("ElasticNet", Pipeline([("s", StandardScaler()),
                                 ("m", ElasticNet(random_state=seed, max_iter=5000))]),
         {"m__alpha": [0.0001, 0.001, 0.01, 0.1], "m__l1_ratio": [0.2, 0.5, 0.8]}),
        ("RandomForest", Pipeline([("m", RandomForestRegressor(random_state=seed, n_jobs=-1))]),
         {"m__n_estimators": [400] if fast else [200, 400],
          "m__max_depth": [None] if fast else [None, 8, 16],
          "m__min_samples_leaf": [1] if fast else [1, 2]}),
        ("ExtraTrees", Pipeline([("m", ExtraTreesRegressor(random_state=seed, n_jobs=-1))]),
         {"m__n_estimators": [600] if fast else [300, 600],
          "m__max_depth": [None] if fast else [None, 8, 16],
          "m__min_samples_leaf": [1] if fast else [1, 2]}),
        ("SVR", Pipeline([("s", StandardScaler()), ("m", SVR())]),
         {"m__kernel": ["rbf"], "m__C": [3.0] if fast else [1.0, 3.0, 10.0, 30.0],
          "m__gamma": ["scale"] if fast else ["scale", 0.1, 0.01],
          "m__epsilon": [0.01] if fast else [0.001, 0.01, 0.1]}),
        ("MLP", Pipeline([("s", StandardScaler()),
                          ("m", MLPRegressor(random_state=seed, max_iter=3000,
                                             early_stopping=True))]),
         {"m__hidden_layer_sizes": [(64, 32)] if fast else [(64,), (128,), (64, 32)],
          "m__activation": ["relu"] if fast else ["relu", "tanh"],
          "m__alpha": [1e-4] if fast else [1e-5, 1e-4, 1e-3]}),
    ]
    if XGB_OK:
        M.append(("XGBoost", Pipeline([("m", XGBRegressor(
            random_state=seed, objective="reg:squarederror", tree_method="hist", n_jobs=-1))]),
            {"m__n_estimators": [800] if fast else [400, 800, 1200],
             "m__max_depth": [5] if fast else [3, 5, 7],
             "m__learning_rate": [0.1] if fast else [0.03, 0.1],
             "m__subsample": [1.0] if fast else [0.7, 1.0],
             "m__colsample_bytree": [1.0] if fast else [0.7, 1.0],
             "m__reg_lambda": [1.0] if fast else [1.0, 2.0]}))
    M.append(("Mean", Pipeline([("m", DummyRegressor(strategy="mean"))]), {}))
    return M


def run_ml(features, tag, train_df, val_df, eval_sets, seed=RANDOM_STATE,
           fast=False, verbose=True):
    Xtr, ytr = train_df[features].values, train_df[TARGET].values
    Xrf = np.vstack([Xtr, val_df[features].values])
    yrf = np.hstack([ytr, val_df[TARGET].values])
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    out, params, fitted = [], {}, {}
    for name, pipe, grid in build_models(seed, fast=fast):
        if verbose:
            print(f"    {name}", flush=True)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error",
                              cv=cv, n_jobs=-1, refit=True)
            gs.fit(Xtr, ytr)
            params[name] = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
            est = gs.best_estimator_
        else:
            est = pipe
        est.fit(Xrf, yrf)
        fitted[name] = est
        for label, df in eval_sets:
            m = metrics(df[TARGET].values, est.predict(df[features].values))
            m.update({"Model": name, "Split": label, "FeatureSet": tag, "Seed": seed})
            out.append(m)
    return pd.DataFrame(out), params, fitted


# ============================================================================
# Main
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="directory holding the three dataset CSVs")
    ap.add_argument("--out", default="revision_outputs", help="output directory")
    ap.add_argument("--fast", action="store_true", help="reduced grids for a trial run")
    a = ap.parse_args(argv)

    data_dir = Path(a.data_dir)
    out = Path(a.out)
    tabdir, figdir = out / "tables", out / "figures"
    for d in (out, tabdir, figdir):
        d.mkdir(parents=True, exist_ok=True)
    fast = a.fast

    # ---------------------------------------------------------------- 0. setup
    banner("0.  ENVIRONMENT  (Appendix C)")
    env = [f"Generated        : {datetime.now().isoformat(timespec='seconds')}",
           f"Platform         : {platform.platform()}",
           f"Python           : {sys.version.split()[0]}",
           f"numpy            : {np.__version__}",
           f"pandas           : {pd.__version__}",
           f"scipy            : {scipy.__version__}",
           f"scikit-learn     : {sklearn.__version__}",
           f"matplotlib       : {matplotlib.__version__}",
           f"xgboost          : {__import__('xgboost').__version__ if XGB_OK else 'not installed'}",
           "",
           f"RANDOM_STATE     : {RANDOM_STATE}",
           f"Repeatability seeds : {SEEDS}",
           f"Fast mode        : {fast}",
           "",
           "Partitioning protocol (Section 4.7.1):",
           "  Dataset A is split 60/20/20 as",
           f"    idx_train, idx_tmp  = train_test_split(idx,     test_size=0.40, random_state={RANDOM_STATE})",
           f"    idx_val,   idx_test = train_test_split(idx_tmp, test_size=0.50, random_state={RANDOM_STATE})",
           "  ML models: 5-fold CV on A_train for selection, refit on A_train + A_val.",
           "  FP models: FUG+ and MESH+ calibrated on A_train + A_val.",
           "  Evaluation: A_test, the whole of Dataset B, the whole of Dataset C."]
    (out / "appendix_C_environment.txt").write_text("\n".join(env))
    MANIFEST.append(("appendix_C_environment.txt", "Computational environment, Appendix C", "text"))
    print("\n".join(env))

    # ---------------------------------------------------------------- 1. data
    banner("1.  DATA AND INTEGRITY")

    def load(name):
        p = data_dir / f"dataset_{name}_distillation.csv"
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")
        return pd.read_csv(p).dropna().reset_index(drop=True)

    A, B, C = load("A"), load("B"), load("C")
    idx = np.arange(len(A))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.40, random_state=RANDOM_STATE)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_STATE)
    A_tr = A.iloc[idx_tr].reset_index(drop=True)
    A_va = A.iloc[idx_va].reset_index(drop=True)
    A_te = A.iloc[idx_te].reset_index(drop=True)
    A_tv = pd.concat([A_tr, A_va], ignore_index=True)
    EVAL = [("A_test", A_te), ("B_all", B), ("C_all", C)]
    print(f"  A {len(A)} rows -> train {len(A_tr)} / val {len(A_va)} / test {len(A_te)}")
    print(f"  B {len(B)} rows | C {len(C)} rows")

    rows = []
    for nm, df in [("A", A), ("B", B), ("C", C)]:
        F, zF = df.Feed_Flow_kg_h.values, df.Feed_Heavy_Fraction.values
        D, Bf = df.Distillate_Rate_kg_h.values, df.Bottoms_Rate_kg_h.values
        yD, xB = df[TARGET].values, df.Bottoms_Heavy_Fraction.values
        ov, cp = F - (D + Bf), F * zF - (D * yD + Bf * xB)
        rows.append({"Dataset": nm,
                     "overall_max_abs": float(np.abs(ov).max()),
                     "component_max_abs": float(np.abs(cp).max()),
                     "component_rel_ppm": float(np.mean(np.abs(cp) / np.maximum(F * zF, 1e-12)) * 1e6)})
    bal = pd.DataFrame(rows)
    print(bal.round(6).to_string(index=False))
    save_table(bal, "balance_audit", "Mass and component balance closure", tabdir)

    ranges = pd.DataFrame([{"Variable": c, "Min": float(A[c].min()), "Max": float(A[c].max()),
                            "Mean": float(A[c].mean()), "SD": float(A[c].std())}
                           for c in A.columns if pd.api.types.is_numeric_dtype(A[c])])
    save_table(ranges, "table_4_3_actual_ranges", "Table 4.3, operating ranges", tabdir)

    # ------------------------------------------------- 2. Appendix B verification
    banner("2.  APPENDIX B - VERIFICATION OF THE FIRST-PRINCIPLES IMPLEMENTATIONS")

    rows = []
    for alpha in [1.3, 1.7, 2.1, 2.5, 2.9, 3.5]:
        for zL in [0.2, 0.35, 0.5, 0.65, 0.8]:
            ta, tc = theta_asis(alpha, zL), theta_underwood(alpha, zL)
            rows.append({"alpha": alpha, "z_LK": zL,
                         "theta_as_implemented": ta,
                         "residual_as_implemented": underwood_residual(ta, alpha, zL),
                         "theta_corrected": tc,
                         "residual_corrected": underwood_residual(tc, alpha, zL)})
    t1 = pd.DataFrame(rows)
    print(f"  B.1 Underwood root: max |residual| as implemented "
          f"{t1.residual_as_implemented.abs().max():.4f}, corrected "
          f"{t1.residual_corrected.abs().max():.2e}")
    save_table(t1, "appendix_B1_underwood_root", "Appendix B.1", tabdir)

    def theta_scan(alpha, zL, n=400000):
        ts = np.linspace(1.0 + 1e-7, alpha - 1e-7, n)
        return float(ts[np.argmin(np.abs(alpha * zL / (alpha - ts) + (1 - zL) / (1 - ts)))])

    t2 = pd.DataFrame([{"alpha": al, "z_LK": z, "theta_numerical_scan": theta_scan(al, z),
                        "theta_closed_form": theta_underwood(al, z),
                        "abs_difference": abs(theta_scan(al, z) - theta_underwood(al, z))}
                       for al, z in [(2.4, .5), (2.0, .4), (3.0, .65), (1.5, .3), (3.5, .75)]])
    print(f"  B.2 closed form vs numerical scan: max difference {t2.abs_difference.max():.2e}")
    save_table(t2, "appendix_B2_crosscheck", "Appendix B.2", tabdir)

    Xs = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])
    t3 = pd.DataFrame({"X": Xs, "Y_Davis": gilliland_davis(Xs), "Y_Eduljee": gilliland_eduljee(Xs),
                       "Y_Molokanov": gilliland_molokanov(Xs)})
    t3["abs_diff_Davis_Eduljee"] = np.abs(t3.Y_Davis - t3.Y_Eduljee)
    print(f"  B.3 Gilliland forms agree to within {t3.abs_diff_Davis_Eduljee.max():.4f} in Y")
    save_table(t3, "appendix_B3_gilliland", "Appendix B.3", tabdir)

    rows = []
    for mode in ["asis", "corrected"]:
        for label, df in EVAL:
            pred, solved = fug_predict(df, mode=mode, return_status=True)
            rows.append({"Model": f"FUG ({mode})", "Split": label,
                         "solve_rate": float(solved.mean()),
                         "pred_mean": float(pred.mean()), "pred_sd": float(pred.std()),
                         "truth_mean": float(df[TARGET].mean()), "truth_sd": float(df[TARGET].std())})
    t4 = pd.DataFrame(rows)
    print(f"  B.4 solver convergence: minimum solve rate {t4.solve_rate.min():.4f}")
    save_table(t4, "appendix_B4_solve_rate", "Appendix B.4", tabdir)

    x = 0.90
    traj = [x]
    for _ in range(8):
        y_op = (2.0 / 3.0) * x + 0.90 / 3.0
        y_act = y_op + 1.0 * (y_eq(2.5, x) - y_op)
        x = float(x_eq(2.5, y_act))
        traj.append(x)
    print(f"  B.5 original march at E_M = 1: change over 8 stages = {traj[0] - traj[-1]:.2e}")
    probe = A_te.head(400)
    mono = []
    for mult in [1.0, 2.0, 4.0]:
        p = probe.copy()
        p["Reflux_Ratio"] = p["Reflux_Ratio"] * mult
        xb, _ = mesh_march(np.full(len(p), 0.10), p)
        mono.append({"reflux_multiplier": mult, "mean_bottoms_light_key": float(xb.mean())})
    t5 = pd.DataFrame(mono)
    print("      corrected march, bottoms light key vs reflux: "
          + ", ".join(f"{r.mean_bottoms_light_key:.5f}" for r in t5.itertuples()))
    save_table(t5, "appendix_B5_mesh_monotonicity", "Appendix B.5", tabdir)

    pts = A_te.sample(3, random_state=RANDOM_STATE).reset_index(drop=True)
    rows = []
    for i, r in pts.iterrows():
        F, D, Bf = float(r.Feed_Flow_kg_h), float(r.Distillate_Rate_kg_h), float(r.Bottoms_Rate_kg_h)
        R, xB_L = float(r.Reflux_Ratio), 1.0 - float(r.Bottoms_Heavy_Fraction)
        m = (R * D + F) / (R * D + D)
        c_o, c_c = -(1.0 - m) * xB_L, -(m - 1.0) * xB_L
        rows.append({"Point": i + 1, "F": round(F, 1), "R": round(R, 4), "x_B,L": round(xB_L, 6),
                     "slope_m": round(m, 5), "intercept_original": round(c_o, 6),
                     "intercept_corrected": round(c_c, 6),
                     "y_at_xB_original": round(m * xB_L + c_o, 6),
                     "y_at_xB_corrected": round(m * xB_L + c_c, 6),
                     "deviation_original": round(abs(m * xB_L + c_o - xB_L), 6),
                     "deviation_corrected": float(f"{abs(m * xB_L + c_c - xB_L):.1e}")})
    t6 = pd.DataFrame(rows)
    print(f"  B.6 stripping line: original deviates from the diagonal by up to "
          f"{t6.deviation_original.max():.4f}, corrected by {t6.deviation_corrected.max():.1e}")
    save_table(t6, "appendix_B6_stripping_line", "Appendix B.6", tabdir)

    rows = []
    for i, r in pts.iterrows():
        zF_H, alpha, R = float(r.Feed_Heavy_Fraction), float(r.Relative_Volatility), float(r.Reflux_Ratio)
        th = theta_underwood(alpha, 1 - zF_H)
        one = pts.iloc[[i]]
        xD = float(fug_predict(one, mode="corrected")[0])
        xB = float(c01((float(r.Feed_Flow_kg_h) * zF_H - float(r.Distillate_Rate_kg_h) * xD)
                       / float(r.Bottoms_Rate_kg_h)))
        Rm = float(Rmin_underwood(th, alpha, xD))
        Xg = float(np.clip((R - Rm) / (R + 1), 0, 1))
        rows.append({"Point": i + 1, "alpha": round(alpha, 5), "theta": round(th, 6),
                     "underwood_residual": float(f"{underwood_residual(th, alpha, 1 - zF_H):.1e}"),
                     "R_min": round(Rm, 5),
                     "N_min": round(float(fenske_Nmin(alpha, xD, xB)), 5),
                     "X": round(Xg, 5), "Y": round(float(gilliland_davis(np.array([Xg]))[0]), 5),
                     "xD_predicted": round(xD, 6), "xD_reference": round(float(r[TARGET]), 6),
                     "abs_error": round(abs(xD - float(r[TARGET])), 6)})
    t7 = pd.DataFrame(rows)
    print("  B.7 worked points written")
    save_table(t7, "appendix_B7_worked_points", "Appendix B.7", tabdir)

    # B.8 - published binary test case (Seader, Henley & Roper, 3rd ed., section 9.1.6)
    al_r, zL_r, xDL_r, xBL_r = 5.0, 0.05, 0.40, 0.001
    Nmin_r = math.log((xDL_r / (1 - xDL_r)) * ((1 - xBL_r) / xBL_r)) / math.log(al_r)
    th_r = theta_underwood(al_r, zL_r)
    Rmin_r = al_r * xDL_r / (al_r - th_r) + (1 - xDL_r) / (1 - th_r) - 1
    R_r = 1.20 * Rmin_r
    X_r = (R_r - Rmin_r) / (R_r + 1)
    rows = []
    for nm, Y in [("Molokanov", float(gilliland_molokanov(np.array([X_r]))[0])),
                  ("Davis explicit fit", float(gilliland_davis(np.array([X_r]))[0])),
                  ("Eduljee", float(gilliland_eduljee(np.array([X_r]))[0]))]:
        rows.append({"Gilliland form": nm, "Y": round(Y, 6),
                     "N": round((Nmin_r + Y) / (1 - Y), 4)})
    t8 = pd.DataFrame(rows)
    t8_head = pd.DataFrame([{"Quantity": "N_min (Fenske)", "Published": 4.04, "Computed": round(Nmin_r, 4)},
                            {"Quantity": "theta (Underwood)", "Published": np.nan, "Computed": round(th_r, 6)},
                            {"Quantity": "R_min (Underwood)", "Published": 1.21, "Computed": round(Rmin_r, 5)},
                            {"Quantity": "X", "Published": np.nan, "Computed": round(X_r, 6)},
                            {"Quantity": "N (Molokanov)", "Published": 10.3,
                             "Computed": round((Nmin_r + float(gilliland_molokanov(np.array([X_r]))[0]))
                                               / (1 - float(gilliland_molokanov(np.array([X_r]))[0])), 4)}])
    print("  B.8 published binary test case: N_min "
          f"{Nmin_r:.4f} (published 4.04), R_min {Rmin_r:.5f} (published 1.21)")
    save_table(t8_head, "appendix_B8_published_case", "Appendix B.8, published comparison", tabdir)
    save_table(t8, "appendix_B8_gilliland_forms", "Appendix B.8, Gilliland forms", tabdir)

    # ------------------------------------------------- 3. FP results, Table 5.1
    banner("3.  FIRST-PRINCIPLES RESULTS  (Table 5.1)")
    CAL = A_tv if not fast else A_tv.sample(min(1500, len(A_tv)), random_state=RANDOM_STATE)
    y_cal = CAL[TARGET].values

    print("  calibrating FUG+ ...")
    best, best_fug = None, None
    for q0 in np.linspace(0.8, 1.2, 5):
        for a0 in np.linspace(0.9, 1.3, 5):
            for b0 in [0.0, 0.5, 1.0, 1.5]:
                for s0 in np.linspace(0.9, 1.1, 5):
                    L = float(np.mean((y_cal - fug_predict(CAL, "corrected", q0, s0, a0, b0)) ** 2))
                    if best is None or L < best:
                        best, best_fug = L, dict(q=q0, alpha_scale=s0, a_map=a0, b_add=b0)
    print(f"    {best_fug}  train MSE {best:.6f}")

    print("  calibrating MESH+ ...")
    best, best_mesh = None, None
    for a_s in np.linspace(0.90, 1.15, 6):
        for e_s in np.linspace(0.85, 1.15, 4):
            for s_a in [0.0, 1.0, 2.0]:
                L = float(np.mean((y_cal - mesh_predict(CAL, a_s, e_s, s_a)) ** 2))
                if best is None or L < best:
                    best, best_mesh = L, dict(alpha_scale=a_s, eff_scale=e_s, stage_add=s_a)
    print(f"    {best_mesh}  train MSE {best:.6f}")
    json.dump({"FUG_plus": best_fug, "MESH_plus": best_mesh},
              open(tabdir / "table_4_1_calibration_parameters.json", "w"), indent=2)
    MANIFEST.append(("tables/table_4_1_calibration_parameters.json", "Table 4.1", "table"))

    FP_PRED, rows = {}, []
    for label, df in EVAL:
        variants = {
            "FUG (original, uncorrected)": lambda d: fug_predict(d, "asis", return_status=True),
            "FUG (corrected)": lambda d: fug_predict(d, "corrected", return_status=True),
            "FUG+ (calibrated)": lambda d: fug_predict(d, "corrected", return_status=True, **best_fug),
            "MESH (corrected)": lambda d: mesh_predict(d, return_status=True),
            "MESH+ (calibrated)": lambda d: mesh_predict(d, return_status=True, **best_mesh),
        }
        for name, fn in variants.items():
            yp, solved = fn(df)
            FP_PRED[(name, label)] = yp
            m = metrics(df[TARGET].values, yp)
            m.update({"Model": name, "Split": label, "SolveRate": float(solved.mean())})
            rows.append(m)
    t51 = pd.DataFrame(rows)[["Model", "Split", "R2", "RMSE", "MAE", "Bias", "SD_resid",
                              "Bias_frac_of_MSE", "SolveRate"]].sort_values(["Model", "Split"])
    print(t51.round(5).to_string(index=False))
    save_table(t51, "table_5_1_first_principles", "Table 5.1", tabdir)

    # ------------------------------------------------- 4. ML results
    banner("4.  MACHINE-LEARNING RESULTS  (Tables 4.5, 5.2, 5.3)")
    print("  full feature set (11)")
    res_full, params_full, fitted_full = run_ml(FEATURES_FULL, "Full (11)", A_tr, A_va, EVAL,
                                                fast=fast)
    print("  without separation power (10)")
    res_nosp, _, _ = run_ml(FEATURES_NO_SP, "No Separation_Power (10)", A_tr, A_va, EVAL,
                            fast=fast, verbose=False)
    print("  sampled operating variables only (6)")
    res_samp, _, _ = run_ml(FEATURES_SAMPLED, "Sampled only (6)", A_tr, A_va, EVAL,
                            fast=fast, verbose=False)

    t52 = res_full[["Model", "Split", "R2", "RMSE", "MAE"]].sort_values(["Model", "Split"])
    print(t52.round(5).to_string(index=False))
    save_table(t52, "table_5_2_machine_learning", "Table 5.2", tabdir)

    abl = pd.concat([res_full, res_nosp, res_samp], ignore_index=True)
    for metric in ["R2", "RMSE"]:
        piv = abl.pivot_table(index=["Model", "Split"], columns="FeatureSet",
                              values=metric).reset_index()
        save_table(piv, f"table_5_3_{metric}_by_feature_set", f"Table 5.3 ({metric})", tabdir)
    print("\n  Table 5.3 (R2 by feature set)")
    print(abl.pivot_table(index=["Model", "Split"], columns="FeatureSet",
                          values="R2").round(4).to_string())

    grids = {n: g for n, _, g in build_models(RANDOM_STATE, fast=fast)}
    rows = []
    for model, grid in grids.items():
        if not grid:
            rows.append({"Model": model, "Hyperparameter": "-", "Search range": "-",
                         "Selected value": "-", "Selection method": "-"})
            continue
        for p, vals in grid.items():
            rows.append({"Model": model, "Hyperparameter": p.replace("m__", ""),
                         "Search range": str(vals),
                         "Selected value": str(params_full.get(model, {}).get(p.replace("m__", ""), "")),
                         "Selection method": "5-fold CV on A_train, scored by RMSE"})
    save_table(pd.DataFrame(rows), "table_4_5_hyperparameters", "Table 4.5", tabdir)

    # ------------------------------------------------- 5. extrapolation
    banner("5.  EXTRAPOLATIVE GENERALISATION  (Table 5.4)")
    AB = pd.concat([A, B], ignore_index=True)
    rows = []
    for var in ["Reflux_Ratio", "Relative_Volatility", "Num_Stages"]:
        lo, hi = AB[var].quantile(0.15), AB[var].quantile(0.85)
        inner = AB[(AB[var] >= lo) & (AB[var] <= hi)].reset_index(drop=True)
        outer = AB[(AB[var] < lo) | (AB[var] > hi)].reset_index(drop=True)
        itr, iva = train_test_split(inner, test_size=0.25, random_state=RANDOM_STATE)
        print(f"  {var}: interior [{lo:.3f}, {hi:.3f}] n={len(inner)}, extrapolation n={len(outer)}")
        r, _, _ = run_ml(FEATURES_SAMPLED, f"holdout_{var}", itr, iva,
                         [("interior_val", iva), ("EXTRAPOLATION", outer)],
                         fast=True, verbose=False)
        r["HeldOutVariable"] = var
        rows.append(r)
    extrap = pd.concat(rows, ignore_index=True)
    t54 = extrap.pivot_table(index=["HeldOutVariable", "Model"], columns="Split",
                             values="R2").reset_index()
    print(t54.round(4).to_string(index=False))
    save_table(t54, "table_5_4_extrapolation", "Table 5.4", tabdir)

    # ------------------------------------------------- 6. repeatability
    banner("6.  SEED REPEATABILITY")
    seed_rows = []
    for s in SEEDS:
        print(f"  seed {s}", flush=True)
        i = np.arange(len(A))
        itr, itmp = train_test_split(i, test_size=0.40, random_state=s)
        iva, ite = train_test_split(itmp, test_size=0.50, random_state=s)
        r, _, _ = run_ml(FEATURES_FULL, "Full", A.iloc[itr].reset_index(drop=True),
                         A.iloc[iva].reset_index(drop=True),
                         [("A_test", A.iloc[ite].reset_index(drop=True)),
                          ("B_all", B), ("C_all", C)],
                         seed=s, fast=True, verbose=False)
        seed_rows.append(r)
    seeds_df = pd.concat(seed_rows, ignore_index=True)
    summ = seeds_df.groupby(["Model", "Split"])[["R2", "RMSE", "MAE"]].agg(["mean", "std"]).round(5)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    print(summ.to_string(index=False))
    save_table(summ, "table_5_2_multiseed", "Table 5.2 with seed variability", tabdir)

    # ------------------------------------------------- 7. figures
    banner("7.  FIGURES")
    VARS = [v for v in FEATURES_FULL + [TARGET] if v in A.columns]

    # 5.1 histograms, 4 x 3
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 11.6))
    for ax, v in zip(axes.ravel(), VARS):
        ax.hist(A[v].values, bins=45, color=CREST[3], alpha=0.9,
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel(LABEL.get(v, v))
        ax.set_ylabel("Count")
    for ax in axes.ravel()[len(VARS):]:
        ax.axis("off")
    suptitle("Distribution of process variables in Dataset A", y=1.002, fontsize=13)
    plt.tight_layout()
    savefig("figure_5_1_histograms_dataset_A", "Figure 5.1", figdir)

    # 5.2 core relationships, 3 x 2
    CORE = ["Reflux_Ratio", "Relative_Volatility", "Effective_Stages",
            "Feed_Heavy_Fraction", "Column_Pressure_kPa", "Separation_Power"]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.4))
    for ax, v in zip(axes.ravel(), CORE):
        ax.scatter(A[v].values, A[TARGET].values, s=5, alpha=0.28, color=CREST[3],
                   edgecolors="none")
        q = pd.qcut(A[v], 24, duplicates="drop")
        med = A.groupby(q, observed=True)[TARGET].median()
        ax.plot([iv.mid for iv in med.index], med.values, color=ACCENT, lw=2.6,
                label="Median trend")
        ax.set_xlabel(LABEL.get(v, v))
        ax.set_ylabel("Distillate heavy fraction")
        ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    savefig("figure_5_2_core_relationships", "Figure 5.2", figdir)

    # 5.3 noise injection
    NV = [v for v in ["Feed_Flow_kg_h", "Reflux_Ratio", "Column_Pressure_kPa", TARGET]
          if v in A.columns and v in C.columns]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for ax, v in zip(axes.ravel(), NV):
        bins = np.linspace(min(A[v].min(), C[v].min()), max(A[v].max(), C[v].max()), 50)
        ax.hist(A[v], bins=bins, alpha=0.75, color=CREST[4], label="Dataset A (baseline)")
        ax.hist(C[v], bins=bins, alpha=0.60, color=ACCENT, label="Dataset C (perturbed)")
        ax.set_xlabel(LABEL.get(v, v))
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("figure_5_3_dataset_distributions", "Figure 5.3", figdir)

    # 5.4 correlation heatmaps
    short = {v: LABEL.get(v, v).replace(" (kg/h)", "").replace(" (kPa)", "") for v in VARS}
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6))
    for ax, (nm, df) in zip(axes, [("Dataset A", A), ("Dataset C", C)]):
        M = df[VARS].corr().values
        im = ax.imshow(M, cmap=VLAG, vmin=-1, vmax=1)
        ax.set_xticks(range(len(VARS)))
        ax.set_yticks(range(len(VARS)))
        ax.set_xticklabels([short[v] for v in VARS], rotation=55, ha="right", fontsize=8)
        ax.set_yticklabels([short[v] for v in VARS], fontsize=8)
        ax.set_title(nm)
        ax.grid(False)
        for i in range(len(VARS)):
            for j in range(len(VARS)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else "#222222")
    fig.colorbar(im, ax=axes, shrink=0.75, label="Pearson correlation coefficient")
    savefig("figure_5_4_correlation_heatmaps", "Figure 5.4", figdir)

    # 5.5 RMSE across all models
    allm = pd.concat([t52[["Model", "Split", "RMSE"]],
                      t51[["Model", "Split", "RMSE"]]], ignore_index=True)
    order = [m for m in ["SVR", "XGBoost", "MLP", "ExtraTrees", "RandomForest", "ElasticNet",
                         "Ridge", "Linear", "Lasso", "FUG+ (calibrated)", "MESH+ (calibrated)",
                         "MESH (corrected)", "FUG (corrected)", "Mean"] if m in set(allm.Model)]
    splits = ["A_test", "B_all", "C_all"]
    SL = {"A_test": "Dataset A (test)", "B_all": "Dataset B", "C_all": "Dataset C"}
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    for k, sp in enumerate(splits):
        vals = [allm[(allm.Model == m) & (allm.Split == sp)].RMSE.mean() for m in order]
        bars = ax.bar(x + (k - 1) * 0.26, vals, 0.26, label=SL[sp], color=DSET[k],
                      edgecolor="white", linewidth=0.5)
        label_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, max(allm[allm.Model.isin(order)].RMSE) * 1.22)
    ax.legend(title="Dataset", loc="upper left")
    no_xgrid(ax)
    plt.tight_layout()
    savefig("figure_5_5_rmse_all_models", "Figure 5.5", figdir)

    # 5.6 FP vs ML, stacked panels
    KEY = [m for m in ["SVR", "XGBoost", "MLP", "Linear", "FUG+ (calibrated)",
                       "MESH+ (calibrated)", "FUG (corrected)", "MESH (corrected)", "Mean"]
           if m in set(allm.Model)]
    r2all = pd.concat([t52[["Model", "Split", "R2"]], t51[["Model", "Split", "R2"]]],
                      ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 11.0), sharex=True)
    x = np.arange(len(KEY))
    for k, sp in enumerate(splits):
        vals = [r2all[(r2all.Model == m) & (r2all.Split == sp)].R2.mean() for m in KEY]
        bars = axes[0].bar(x + (k - 1) * 0.26, vals, 0.26, label=SL[sp], color=DSET[k],
                           edgecolor="white", linewidth=0.5)
        label_bars(axes[0], bars)
    axes[0].axhline(0, color="0.35", lw=0.9)
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(-0.15, 1.20)
    no_xgrid(axes[0])
    for k, sp in enumerate(splits):
        vals = [allm[(allm.Model == m) & (allm.Split == sp)].RMSE.mean() for m in KEY]
        bars = axes[1].bar(x + (k - 1) * 0.26, vals, 0.26, color=DSET[k],
                           edgecolor="white", linewidth=0.5)
        label_bars(axes[1], bars)
    axes[1].set_ylabel("RMSE")
    axes[1].set_ylim(0, max(allm[allm.Model.isin(KEY)].RMSE) * 1.30)
    no_xgrid(axes[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(KEY, rotation=30, ha="right")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3,
               frameon=False, title="Dataset")
    plt.tight_layout()
    savefig("figure_5_6_fp_vs_ml_comparison", "Figure 5.6", figdir)

    # 5.7 - 5.9 parity
    top_ml = res_full[res_full.Split == "A_test"].sort_values("RMSE").head(2)["Model"].tolist()
    for i, (label, df) in enumerate(EVAL):
        yt = df[TARGET].values
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
        for j, nm in enumerate(top_ml):
            ax.scatter(yt, fitted_full[nm].predict(df[FEATURES_FULL].values), s=5,
                       alpha=0.3, color=CREST[j * 2 + 1], label=nm)
        ax.scatter(yt, FP_PRED[("FUG+ (calibrated)", label)], s=5, alpha=0.3,
                   color=ACCENT, label="FUG+")
        ax.scatter(yt, FP_PRED[("MESH+ (calibrated)", label)], s=5, alpha=0.3,
                   color=FLARE[2], label="MESH+")
        allp = np.concatenate([fitted_full[nm].predict(df[FEATURES_FULL].values) for nm in top_ml]
                              + [FP_PRED[("FUG+ (calibrated)", label)],
                                 FP_PRED[("MESH+ (calibrated)", label)], yt])
        pad = 0.03 * (allp.max() - allp.min())
        lim = [max(allp.min() - pad, 0.0), allp.max() + pad]
        ax.plot(lim, lim, "k-", lw=1.2, label="Parity")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("True distillate heavy fraction")
        ax.set_ylabel("Predicted distillate heavy fraction")
        leg = ax.legend(markerscale=3)
        for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
            lh.set_alpha(1.0)
        plt.tight_layout()
        savefig(f"figure_5_{7 + i}_parity_{label}", f"Figure 5.{7 + i}", figdir)

    # 5.10 - 5.16 sensitivity
    best_ml_name = res_full[res_full.Split == "A_test"].sort_values("RMSE").iloc[0]["Model"]
    ml_model = fitted_full[best_ml_name]
    base = A_te.reset_index(drop=True).copy()
    y0_ml = ml_model.predict(base[FEATURES_FULL].values)
    y0_fug = fug_predict(base, "corrected", **best_fug)
    y0_mesh = mesh_predict(base, **best_mesh)
    sens_rows = []
    for var in SENS_VARS:
        for st in STEPS:
            p = base.copy()
            p[var] = p[var] * (1.0 + st)
            if var in ("Num_Stages", "Murphree_Efficiency"):
                p["Effective_Stages"] = p["Num_Stages"] * p["Murphree_Efficiency"]
            if var in ("Reflux_Ratio", "Feed_Flow_kg_h"):
                p["Reflux_to_Feed"] = p["Reflux_Ratio"] * p["Distillate_Rate_kg_h"] / p["Feed_Flow_kg_h"]
            if var == "Feed_Flow_kg_h":
                p["Distillate_to_Feed"] = p["Distillate_Rate_kg_h"] / p["Feed_Flow_kg_h"]
            d_ml = ml_model.predict(p[FEATURES_FULL].values) - y0_ml
            sens_rows.append({"Variable": var, "Step_pct": st * 100,
                              "dxD_ML_mean": float(np.mean(d_ml)), "dxD_ML_sd": float(np.std(d_ml)),
                              "dxD_FUG_mean": float(np.mean(fug_predict(p, "corrected", **best_fug) - y0_fug)),
                              "dxD_MESH_mean": float(np.mean(mesh_predict(p, **best_mesh) - y0_mesh))})
    sens = pd.DataFrame(sens_rows)
    save_table(sens, "sensitivity_all_seven", "Sensitivity analysis, seven variables", tabdir)

    for i, var in enumerate(SENS_VARS):
        s = sens[sens.Variable == var].sort_values("Step_pct")
        fig, ax = plt.subplots(figsize=(6.2, 4.1))
        ax.plot(s.Step_pct, s.dxD_ML_mean, "o-", color=CREST[4], lw=2, label=best_ml_name)
        ax.fill_between(s.Step_pct, s.dxD_ML_mean - s.dxD_ML_sd, s.dxD_ML_mean + s.dxD_ML_sd,
                        color=CREST[4], alpha=0.15)
        ax.plot(s.Step_pct, s.dxD_FUG_mean, "s--", color=ACCENT, lw=2, label="FUG+")
        ax.plot(s.Step_pct, s.dxD_MESH_mean, "^:", color=CREST[1], lw=2, label="MESH+")
        ax.axhline(0, color="0.35", lw=0.9)
        ax.set_xlabel(f"Step change in {LABEL.get(var, var)} (%)")
        ax.set_ylabel("Change in distillate heavy fraction")
        ax.legend()
        plt.tight_layout()
        savefig(f"figure_5_{10 + i}_sensitivity_{var}", f"Figure 5.{10 + i}", figdir)

    # 5.17 - 5.18 residuals
    yt = A_te[TARGET].values
    resid = {nm: yt - fitted_full[nm].predict(A_te[FEATURES_FULL].values) for nm in top_ml}
    resid["FUG+"] = yt - FP_PRED[("FUG+ (calibrated)", "A_test")]
    resid["MESH+"] = yt - FP_PRED[("MESH+ (calibrated)", "A_test")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for j, (nm, r) in enumerate(resid.items()):
        ax.hist(r, bins=60, alpha=0.5, label=nm, color=(CREST + FLARE)[j % 9])
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    savefig("figure_5_17_residual_distributions", "Figure 5.17", figdir)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.boxplot(list(resid.values()), labels=list(resid.keys()), showfliers=False)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_ylabel("Residual")
    plt.xticks(rotation=15)
    plt.tight_layout()
    savefig("figure_5_18_residual_spread", "Figure 5.18", figdir)

    save_table(pd.DataFrame([{"Model": nm, "Bias": float(np.mean(r)), "SD": float(np.std(r))}
                             for nm, r in resid.items()]),
               "residual_summary", "Residual bias and dispersion", tabdir)

    # ------------------------------------------------- 8. manifest
    banner("8.  MANIFEST")
    man = pd.DataFrame(MANIFEST, columns=["File", "Description", "Kind"])
    man.to_csv(out / "MANIFEST.csv", index=False)
    print(man.to_string(index=False))
    print(f"\nAll outputs written to {out.resolve()}")
    print(f"  {len(man[man.Kind == 'table'])} tables, {len(man[man.Kind == 'figure'])} figures")
    if fast:
        print("\nNOTE: --fast was used. Re-run without it for the values reported in the thesis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
