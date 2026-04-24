# `mcr`: Markov-Chain Reliability Toolkit for LLM Agent Chains

Reference implementation for the ISSRE 2026 submission,
*Measuring the Unmeasurable: Markov Chain Reliability for LLM Agents.*

The `mcr` package and the scripts under `sim/` take a corpus of agent
traces, fit an absorbing discrete-time Markov chain
`(Q, R_succ, R_fail)` via the TraceToChain pipeline
(Alg. 1 in the paper), and recover the closed-form reliability
`R(d) = e_s0^T (I - Q^d) N R_succ`, the asymptotic reliability
`R_inf = e_s0^T N R_succ` with `N = (I - Q)^{-1}`, and every downstream
metric (pass@k, pass^k, RDC, MTTA) together with calibrated
Dirichlet-posterior and bootstrap credible intervals.

## Requirements

- Python >= 3.10
- `numpy >= 1.22`, `scipy`, `matplotlib`, `pandas`
- `pdflatex` + `bibtex` (paper only)
- PRISM 4.7+ on `$PATH` (optional; SS5 is skipped otherwise)

No GPU, no network, no proprietary data. Every script seeds its RNG
explicitly (see `data/synthetic/seeds.json`).

## Install

From the repo root:

```bash
pip install -e ./code                       # editable install of the mcr package
```

Or, inside `code/`:

```bash
pip install -e .
pip install numpy scipy matplotlib pandas   # script-only deps
```

A one-command end-to-end reproduction (install + all simulation studies
+ static paper verification, ~15 min on a laptop) is available at the
repo root:

```bash
bash reproduce.sh            # full run
bash reproduce.sh --quick    # reduced sample sizes, ~2 min smoke test
```

## Package API (selected)

```python
from mcr import (
    reliability,                          # R(d) = e_s0^T (I - Q^d) N R_succ
    asymptotic_reliability,               # R_inf
    fundamental_matrix,                   # N = (I - Q)^-1
    monte_carlo_reliability,              # sanity-check MC estimator
    perturb,                              # row-rank ΔQ construction (T2)
    random_substochastic,                 # synthetic chain generator
    nhpp_scaling_family,                  # SS4 rare-failure family
    fit_chain,                            # TraceToChain MLE + GoF (Alg. 1 / 2)
    empirical_first_passage_from_traces,
    goodness_of_fit,
)
```

Example: fit a chain from traces, query the RDC, and obtain a 95 % CI
on `R_inf`.

```python
from mcr import fit_chain, reliability, asymptotic_reliability
from mcr.uncertainty import posterior_credible_intervals

fit = fit_chain(traces, m_range=(3, 12))         # returns Q_hat, R_succ, R_fail, m
R_d = reliability(fit.Q, fit.R_succ, s0=fit.s0, d=20)
R_inf = asymptotic_reliability(fit.Q, fit.R_succ, s0=fit.s0)
ci = posterior_credible_intervals(fit, alpha=0.05)
```

## Simulation studies (reproducing paper numbers)

Every script reads a fixed seed, writes its figure into `../figs/`, and
writes its JSON summary into `../data/synthetic/` or
`../data/mast_derived/`. The table columns below come from those JSON
files; they are the exact numbers quoted in the paper.

### SS1 — Closed-form vs. Monte Carlo (analytic sanity)

```bash
python sim/SS1_cf_vs_mc.py
```

500 random substochastic chains per size
`m ∈ {5, 10, 50, 100, 500}`, compares `R_inf` (T1) against `10^5`-path
Monte Carlo. The 5% G2 gate is satisfied with margin across all sizes:

| m    | max \|R_inf^cf − R_inf^MC\| | mean err | p99 err |
|-----:|---------------------------:|---------:|--------:|
|    5 | 0.0134 | 0.0037 | 0.0133 |
|   10 | 0.0083 | 0.0029 | 0.0078 |
|   50 | 0.0120 | 0.0028 | 0.0103 |
|  100 | 0.0103 | 0.0033 | 0.0102 |
|  500 | 0.0081 | 0.0031 | 0.0078 |

(Source: `data/synthetic/SS1_summary.json`.)

### SS4 — NHPP rare-failure limit (T5)

```bash
python sim/SS4_nhpp_limit.py
```

Sequence of one-state chains with `μ_n ∈ {0.2, 0.1, 0.05, 0.02, 0.01, 0.005}`
and `ε_n = 0.1 μ_n`, horizon `d_n = ⌈Λ/μ_n⌉`, `Λ = 2`. The exact
first-passage CDF `R_n(d)` is compared to the Goel–Okumoto mean-value
function `1 − e^{−Λc}` on `c = d/d_n`. Asymptotic KS `p > 0.05` on all
six scaling points, confirming Proposition T5.

### SS7 — Goodness-of-fit of the agent-DTMC assumption

```bash
python sim/SS7_goodness_of_fit.py
```

Three conditions validating Alg. 2 (composite AIC ∧ KS test):

- **Power (Markov ground truth).** 30 corpora × 300 traces × `m = 5`:
  composite test retains H0 at α = 0.05 on 100% (Type-I control).
- **Specificity (2nd-order ground truth).** 15 corpora with 2nd-order
  mixture weight 0.6: KS layer rejects 0%; AIC layer rejects 100% with
  ΔAIC ∈ [−1540, −1310]; composite rule rejects 100%.
- **MAST self-consistency (in-sample).** 500 synthetic traces per
  framework resampled from each framework's own fitted chain;
  composite test accepts all 7 with KS p-values
  ∈ {0.520, 0.583, 0.852, 0.944, 0.947, 0.957, 0.966}.

### SS9 — Held-out empirical validation (strict fit/test split)

```bash
python sim/SS9_heldout_empirical.py
```

For each of seven MAST-style frameworks: generate 400 trajectories,
split before featurization into `n_fit = n_test = 200`, fit the chain
on the fit half only, sample `N = 8000` model first-passage times,
and report KS distance + L∞ RDC error on the held-out half.

| Framework   | m | D_KS  | p_KS  | L∞^RDC |
|-------------|---|------:|------:|-------:|
| react       | 5 | 0.017 | 1.000 | 0.0476 |
| reflexion   | 5 | 0.031 | 0.992 | 0.0525 |
| cot_agent   | 5 | 0.024 | 1.000 | 0.0184 |
| toolformer  | 5 | 0.024 | 1.000 | 0.0523 |
| babyagi     | 6 | 0.047 | 0.776 | 0.0535 |
| autogpt     | 6 | 0.032 | 0.987 | 0.0387 |
| agentbench  | 5 | 0.032 | 0.988 | 0.0212 |

Aggregate: **7/7** frameworks pass at α = 0.05
(min p_KS = 0.776), max L∞^RDC = 0.0535, median 0.0476.
(Source: `data/mast_derived/SS9_heldout_summary.json`; paper Table
`tab:heldout-ss9`.)

### SS6 — MAST case study (per-framework ranking)

```bash
python sim/SS6_mast_case_study.py
```

Reliability ranking across seven public MAST frameworks (full
per-framework numbers in `data/mast_derived/SS6_summary.json`):

| Framework   |  R_inf | ρ(Q)  | Horizon (δ=0.01) |  m |
|-------------|-------:|------:|-----------------:|---:|
| toolformer  | 0.4497 | 0.442 |                5 | 10 |
| autogpt     | 0.3866 | 0.370 |                4 |  5 |
| agentbench  | 0.3262 | 0.654 |                9 | 10 |
| babyagi     | 0.2853 | 0.656 |                8 | 12 |
| reflexion   | 0.0695 | 0.636 |                5 | 12 |
| cot_agent   | 0.0684 | 0.526 |                3 | 10 |
| react       | 0.0581 | 0.509 |                3 |  9 |

### SS2/SS3 — perturbation tightness and correlated-trial gap

```bash
python sim/SS2_perturbation.py
python sim/SS3_correlation.py
```

SS2 sweeps ε ∈ [0, 0.28] and checks the Neumann-series perturbation
bound (Prop. T2) against ground truth. SS3 validates the Jensen-gap
diagnostic (Thm. T3′) that `pass^k_mix > R_inf^k` when trials share a
latent mixing factor. Both are retained in the artifact but dropped
from the 12-page body; they are re-used for the Discussion in the
paper.

### SS5 — PRISM cross-check (optional)

```bash
python sim/SS5_prism_cross.py
```

Exports five `m = 5` chains to PRISM DTMC format and queries
`P=? [F<=20 "success"]`. Python closed form agrees with PRISM to
within 1e-8.

## Repository layout (this directory)

```
code/
├── mcr/                       # The installable Python package
│   ├── reliability.py         # R(d), R_inf, N = (I − Q)^-1
│   ├── perturb.py             # row-rank ΔQ construction (T2)
│   ├── simulate.py            # Monte Carlo baseline
│   ├── chains.py              # random + rare-failure synthetic generators
│   ├── trace_to_chain.py      # TraceToChain MLE + GoF (Alg. 1 / 2)
│   └── uncertainty.py         # Dirichlet posterior + bootstrap CIs
├── sim/                       # Simulation-study scripts SS1–SS9
│   ├── SS1_cf_vs_mc.py
│   ├── SS2_perturbation.py
│   ├── SS3_correlation.py
│   ├── SS4_nhpp_limit.py
│   ├── SS5_prism_cross.py
│   ├── SS6_mast_case_study.py
│   ├── SS7_goodness_of_fit.py
│   ├── SS7_cross_benchmark.py
│   ├── SS8_uncertainty_quantification.py
│   ├── SS9_heldout_empirical.py
│   └── mast_adapter.py        # loader for the public MAST traces
├── notebooks/
│   └── theorem_sanity.ipynb   # interactive walkthrough of T1–T6
└── pyproject.toml
```

## Mapping to paper artifacts

| Paper reference                | Script / module                                 |
|--------------------------------|--------------------------------------------------|
| Prop. T1 (closed form R(d))    | `mcr.reliability.reliability`                    |
| Prop. T2 (perturbation)        | `mcr.perturb.perturb`, `sim/SS2_perturbation.py` |
| Thm. T3 (metric unification)   | `mcr.reliability.asymptotic_reliability`         |
| Alg. 1 (TraceToChain)          | `mcr.trace_to_chain.fit`                         |
| Alg. 2 (composite GoF)         | `mcr.trace_to_chain.goodness_of_fit`             |
| §UQ (Dirichlet + bootstrap CI) | `mcr.uncertainty`                                |
| SS1–SS9 (simulations)          | `code/sim/SS*.py`                                |
| Fig. `fig_rdc_overlay`         | output of `SS9_heldout_empirical.py`             |
| Fig. `fig_gof`                 | output of `SS7_goodness_of_fit.py`               |

## Reproducibility notes

Every script writes a JSON summary alongside its figure; the JSONs are
the ground truth for every numerical claim in the paper. A static
verifier, `paper/verify_paper.py`, cross-checks the LaTeX source for
undefined references, missing citations, missing `\input` files, and
approximate word count. `bash reproduce.sh` re-runs SS1–SS6 and the
verifier in a single invocation. The full pipeline is deterministic
given the seeds in `data/synthetic/seeds.json`; minor floating-point
variation across BLAS versions (< 1e-10) is expected and does not
affect any reported digit.
