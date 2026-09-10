"""
generate_datasets.py
====================

Synthetic data-generation procedure used in

    "A Data Driven Approach to Binary Distillation Column Modelling
     Using Machine Learning", C. Naidoo, MEng thesis."
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------

#: alpha = A0 + A1 ln P, with P in kPa.  Max error 2.3e-03 over Dataset A.
VOLATILITY = (3.4827, -0.29549)

#: ln(1 - HRB) = H_S * S + H_F * (F/1e4 - 1).  Max error ~4e-03 on ln(1-HRB).
HEAVY_RECOVERY = (-0.35, 0.30)

#: Sampling ranges for Dataset A, measured from the delivered data.
RANGES_A = {
    "Feed_Flow_kg_h":      (7000.0, 13000.0),
    "Feed_Heavy_Fraction": (0.30, 0.70),
    "Column_Pressure_kPa": (120.0, 230.0),
    "Reflux_Ratio":        (1.00, 4.00),
    "Num_Stages":          (16, 27),
    "Murphree_Efficiency": (0.65, 0.90),
}

#: Feed heavy fraction, reflux ratio and Murphree efficiency are uniform in
#: Dataset A (excess kurtosis -1.20, -1.23, -1.17).  Feed flow and pressure are
#: bounded but more peaked (-0.53, -0.11), consistent with truncated normal.
UNIFORM_IN_A = {"Feed_Heavy_Fraction", "Reflux_Ratio", "Murphree_Efficiency"}

COLUMNS = [
    "Feed_Flow_kg_h", "Feed_Heavy_Fraction", "Column_Pressure_kPa", "Reflux_Ratio",
    "Num_Stages", "Murphree_Efficiency", "Effective_Stages", "Relative_Volatility",
    "Separation_Power", "Distillate_Rate_kg_h", "Bottoms_Rate_kg_h", "Reflux_Flow_kg_h",
    "Distillate_Heavy_Fraction", "Bottoms_Heavy_Fraction", "Top_Tray_Temperature_C",
    "Bottom_Temperature_C", "Distillate_to_Feed", "Reflux_to_Feed",
    "Heavy_Recovery_to_Bottoms", "Dataset", "Mass_Balance_Error_kg_h",
    "Heavy_Component_Balance_Error_kg_h",
]


# ----------------------------------------------------------------------------
# Deterministic relationships
# ----------------------------------------------------------------------------
def relative_volatility(P):
    """alpha as a function of column pressure (kPa)."""
    a0, a1 = VOLATILITY
    return a0 + a1 * np.log(P)


def effective_stages(N, E_M):
    """Efficiency-weighted stage count."""
    return N * E_M


def separation_power(R, N_eff, alpha):
    """S = (L/V) N_eff ln(alpha).

    R/(R+1) is the internal reflux ratio L/V, the slope of the rectifying
    operating line; N_eff ln(alpha) is the Fenske separation exponent evaluated
    at the effective stage count.
    """
    return (R / (R + 1.0)) * N_eff * np.log(alpha)


def heavy_recovery(S, F):
    """Fraction of the heavy component leaving in the bottoms."""
    h_s, h_f = HEAVY_RECOVERY
    return 1.0 - np.exp(h_s * S + h_f * (F / 1e4 - 1.0))


def tray_temperatures(P, xD_H, xB_H):
    """Linearised bubble-point approximations (diagnostic outputs only)."""
    T_top = 74.697 + 0.038583 * P + 13.211135 * xD_H
    T_bot = 89.1573 + 0.049474 * P + 6.611382 * xB_H
    return T_top, T_bot


def close_balances(F, zF, DF, HRB):
    """Derive the product streams from the split and the heavy recovery.

    HRB = B x_B / (F z_F) and the component balance F z_F = D x_D + B x_B
    together give HRB = 1 - (D/F)(x_D / z_F), hence x_D directly.
    """
    D = F * DF
    B = F - D
    xD_H = zF * (1.0 - HRB) / DF
    xB_H = (F * zF - D * xD_H) / B
    return D, B, xD_H, xB_H


def assemble(F, zF, P, R, N, E_M, DF, label):
    """Build the full 22-column frame from the sampled variables and the split."""
    alpha = relative_volatility(P)
    N_eff = effective_stages(N, E_M)
    S = separation_power(R, N_eff, alpha)
    HRB = heavy_recovery(S, F)
    D, B, xD_H, xB_H = close_balances(F, zF, DF, HRB)
    L = R * D
    T_top, T_bot = tray_temperatures(P, xD_H, xB_H)

    df = pd.DataFrame({
        "Feed_Flow_kg_h": F,
        "Feed_Heavy_Fraction": zF,
        "Column_Pressure_kPa": P,
        "Reflux_Ratio": R,
        "Num_Stages": N,
        "Murphree_Efficiency": E_M,
        "Effective_Stages": N_eff,
        "Relative_Volatility": alpha,
        "Separation_Power": S,
        "Distillate_Rate_kg_h": D,
        "Bottoms_Rate_kg_h": B,
        "Reflux_Flow_kg_h": L,
        "Distillate_Heavy_Fraction": xD_H,
        "Bottoms_Heavy_Fraction": xB_H,
        "Top_Tray_Temperature_C": T_top,
        "Bottom_Temperature_C": T_bot,
        "Distillate_to_Feed": DF,
        "Reflux_to_Feed": L / F,
        "Heavy_Recovery_to_Bottoms": B * xB_H / (F * zF),
        "Dataset": label,
    })
    df["Mass_Balance_Error_kg_h"] = df.Feed_Flow_kg_h - (df.Distillate_Rate_kg_h + df.Bottoms_Rate_kg_h)
    df["Heavy_Component_Balance_Error_kg_h"] = (
        df.Feed_Flow_kg_h * df.Feed_Heavy_Fraction
        - (df.Distillate_Rate_kg_h * df.Distillate_Heavy_Fraction
           + df.Bottoms_Rate_kg_h * df.Bottoms_Heavy_Fraction)
    )
    return df[COLUMNS]


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify(path):
    A = pd.read_csv(path).dropna().reset_index(drop=True)
    F = A.Feed_Flow_kg_h.values
    zF = A.Feed_Heavy_Fraction.values
    P = A.Column_Pressure_kPa.values
    R = A.Reflux_Ratio.values
    N = A.Num_Stages.values
    E_M = A.Murphree_Efficiency.values
    N_eff = A.Effective_Stages.values
    alpha = A.Relative_Volatility.values
    D = A.Distillate_Rate_kg_h.values
    B = A.Bottoms_Rate_kg_h.values
    xD = A.Distillate_Heavy_Fraction.values
    xB = A.Bottoms_Heavy_Fraction.values
    DF = A.Distillate_to_Feed.values
    HRB = A.Heavy_Recovery_to_Bottoms.values

    checks = [
        ("alpha = 3.4827 - 0.29549 ln P",          alpha, relative_volatility(P)),
        ("N_eff = N E_M",                          N_eff, effective_stages(N, E_M)),
        ("S = [R/(R+1)] N_eff ln alpha",           A.Separation_Power.values,
                                                   separation_power(R, N_eff, alpha)),
        ("D = F (D/F)",                            D, F * DF),
        ("B = F - D",                              B, F - D),
        ("L = R D",                                A.Reflux_Flow_kg_h.values, R * D),
        ("L/F = R (D/F)",                          A.Reflux_to_Feed.values, R * DF),
        ("x_B = (F z_F - D x_D) / B",              xB, (F * zF - D * xD) / B),
        ("HRB = B x_B / (F z_F)",                  HRB, B * xB / (F * zF)),
        ("HRB = 1 - (D/F)(x_D / z_F)",             HRB, 1.0 - DF * xD / zF),
        ("x_D from HRB and the balance",           xD, zF * (1.0 - HRB) / DF),
        ("T_top (linearised bubble point)",        A.Top_Tray_Temperature_C.values,
                                                   tray_temperatures(P, xD, xB)[0]),
        ("T_bot (linearised bubble point)",        A.Bottom_Temperature_C.values,
                                                   tray_temperatures(P, xD, xB)[1]),
        ("HRB from S and F (approximate)",         HRB, heavy_recovery(A.Separation_Power.values, F)),
    ]

    print(f"Verifying {len(A)} rows from {path}\n")
    print(f"  {'relationship':44s}{'max abs err':>14s}{'mean abs err':>14s}")
    print("  " + "-" * 72)
    worst = 0.0
    for name, got, want in checks:
        err = np.abs(np.asarray(got, float) - np.asarray(want, float))
        print(f"  {name:44s}{err.max():14.2e}{err.mean():14.2e}")
        if "approximate" not in name and "bubble point" not in name:
            worst = max(worst, err.max())
    print("  " + "-" * 72)
    print(f"\n  Worst exact-relationship error: {worst:.2e}")
    return worst


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def _sample(rng, lo, hi, n, uniform):
    if uniform:
        return rng.uniform(lo, hi, n)
    # truncated normal centred on the range, sigma chosen so the bounds sit at 2.5 sd
    mu, sd = 0.5 * (lo + hi), (hi - lo) / 5.0
    out = rng.normal(mu, sd, n)
    bad = (out < lo) | (out > hi)
    while bad.any():
        out[bad] = rng.normal(mu, sd, int(bad.sum()))
        bad = (out < lo) | (out > hi)
    return out


def generate(n=5000, seed=42, label="A", uniform_all=False, split_source=None):
    """Generate a dataset of n observations"""
    rng = np.random.default_rng(seed)
    v = {}
    for name, (lo, hi) in RANGES_A.items():
        if name == "Num_Stages":
            v[name] = rng.integers(lo, hi + 1, n).astype(float)
        else:
            v[name] = _sample(rng, lo, hi, n, uniform_all or name in UNIFORM_IN_A)

    if split_source is not None:
        pool = pd.read_csv(split_source).Distillate_to_Feed.dropna().values
        DF = rng.choice(pool, n, replace=True)
    else:
        DF = rng.uniform(0.1573, 0.6095, n)

    return assemble(v["Feed_Flow_kg_h"], v["Feed_Heavy_Fraction"], v["Column_Pressure_kPa"],
                    v["Reflux_Ratio"], v["Num_Stages"], v["Murphree_Efficiency"], DF, label)


def add_noise(df, seed=42, offsets=(1.05, 1.10, 0.95), noise_frac=0.01):
    """Produce a Dataset C style perturbation of a baseline frame."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    fo, ro, po = offsets
    out["Feed_Flow_kg_h"] *= fo
    out["Reflux_Ratio"] *= ro
    out["Column_Pressure_kPa"] *= po
    for c in ["Feed_Flow_kg_h", "Feed_Heavy_Fraction", "Column_Pressure_kPa",
              "Reflux_Ratio", "Murphree_Efficiency"]:
        out[c] = out[c] * (1.0 + rng.normal(0.0, noise_frac, len(out)))
    return assemble(out.Feed_Flow_kg_h.values, out.Feed_Heavy_Fraction.values,
                    out.Column_Pressure_kPa.values, out.Reflux_Ratio.values,
                    out.Num_Stages.values, out.Murphree_Efficiency.values,
                    out.Distillate_to_Feed.values, "C")


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="check the relationships against a dataset")
    v.add_argument("path")

    g = sub.add_parser("generate", help="generate a new dataset")
    g.add_argument("--n", type=int, default=5000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--label", default="A")
    g.add_argument("--uniform-all", action="store_true", help="Dataset B sampling scheme")
    g.add_argument("--split-source", default=None, help="dataset to resample D/F from")
    g.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "verify":
        verify(a.path)
    else:
        df = generate(a.n, a.seed, a.label, a.uniform_all, a.split_source)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
