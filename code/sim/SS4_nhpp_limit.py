"""SS4 — NHPP rare-failure limit verification (Theorem T5).

Scaling sequence of forward chains approaching the rare-failure regime
(eps_n -> 0, d_n -> inf, d_n * eps_n -> lambda).  Compares R(d) to the
Goel-Okumoto fit 1 - exp(-eps * d) and reports the Kolmogorov-Smirnov
p-value.  Produces fig_nhpp_limit.pdf.

Usage:
    python SS4_nhpp_limit.py
    python SS4_nhpp_limit.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

SEED = 20260420_04
EPS_SEQUENCE = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
LAMBDA_TARGET = 2.0   # d_n * eps_n -> lambda

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "synthetic"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def poisson_limit_survival(eps: float, d_vals: np.ndarray) -> np.ndarray:
    """Poisson-limit approximation of survival prob: exp(-eps * d).

    Under rare-failure scaling, (1-eps)^d → exp(-eps*d) as eps → 0, d*eps fixed.
    R(d) = P[no failure in d steps] = (1-eps)^d ≈ exp(-eps*d).
    This is the NHPP survival function (zero-failure probability under Goel-Okumoto).
    """
    return np.exp(-eps * d_vals)


def ks_statistic(empirical: np.ndarray, theoretical: np.ndarray) -> float:
    """One-sample KS statistic: max |F_n(x) - F(x)|."""
    return float(np.max(np.abs(empirical - theoretical)))


def run(eps_seq=EPS_SEQUENCE) -> dict:
    summary = []

    for eps in eps_seq:
        d_n = int(LAMBDA_TARGET / eps)
        d_vals = np.arange(0, d_n + 1)
        # R(d) = (1 - eps)^d for a Bernoulli failure process
        rdc = (1 - eps) ** d_vals
        go_rdc = poisson_limit_survival(eps, d_vals)
        ks = ks_statistic(rdc, go_rdc)
        # KS p-value approximation (asymptotic)
        n = len(d_vals)
        ks_scaled = ks * np.sqrt(n)
        p_val = 2 * np.exp(-2 * ks_scaled ** 2) if ks_scaled > 0 else 1.0

        print(f"  eps={eps:.4f}  d_n={d_n:5d}  KS={ks:.4f}  "
              f"KS_scaled={ks_scaled:.2f}  p≈{p_val:.4f}  "
              f"{'PASS' if p_val >= 0.05 else 'FAIL'}")
        summary.append({
            "eps": float(eps),
            "d_n": d_n,
            "ks": ks,
            "ks_scaled": float(ks_scaled),
            "p_approx": float(p_val),
            "pass": bool(p_val >= 0.05),
        })

    return {"eps_sequence": eps_seq, "lambda_target": LAMBDA_TARGET, "summary": summary}


def make_figure(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    eps_seq = data["eps_sequence"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for ax, eps in zip(axes, eps_seq):
        d_n = int(LAMBDA_TARGET / eps)
        d_vals = np.arange(0, d_n + 1)
        rdc = (1 - eps) ** d_vals
        go_rdc = poisson_limit_survival(eps, d_vals)

        ax.plot(d_vals / d_n, rdc, label="Exact $R(d)$", color="steelblue")
        ax.plot(d_vals / d_n, go_rdc, "--", label="Poisson limit $e^{-\\varepsilon d}$", color="tomato")
        ax.set_title(f"$\\varepsilon={eps}$, $d_n={d_n}$")
        ax.set_xlabel("$d / d_n$")
        ax.set_ylabel("$R(d)$")
        ax.legend(fontsize=7)

    plt.suptitle("SS4: NHPP Rare-Failure Limit (T5 validation)", y=1.01)
    plt.tight_layout()
    out = FIGS_DIR / "fig_nhpp_limit.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    eps_seq = EPS_SEQUENCE[-3:] if args.quick else EPS_SEQUENCE

    print("SS4 — NHPP rare-failure limit verification")
    print(f"  lambda_target={LAMBDA_TARGET}")
    data = run(eps_seq=eps_seq)
    (DATA_DIR / "SS4_summary.json").write_text(json.dumps(data, indent=2))
    make_figure(data)

    passes = sum(1 for row in data["summary"] if row["pass"])
    total = len(data["summary"])
    print(f"\nKS test passed for {passes}/{total} eps values at alpha=0.05")
    print("[DONE] SS4 complete.")


if __name__ == "__main__":
    main()
