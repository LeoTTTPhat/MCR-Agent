# mcr-agent

Companion artifact for the paper
*A Markov-Chain Semantics for LLM Agent Reliability: Closed-Form Bounds and a Unification of pass@$k$, pass$^k$, and the Reliability Decay Curve.*

## Repository layout

```
mcr-agent/
├── paper/            IEEEtran LaTeX source for the research paper
├── proofs/           One .tex per theorem, plus notation.tex
├── code/
│   ├── mcr/          The `mcr` Python package (reliability calculator)
│   ├── sim/          Simulation study scripts SS1–SS5
│   └── notebooks/    (optional) Jupyter notebooks
├── data/
│   ├── synthetic/    Seeded synthetic chain families
│   └── mast_derived/ Transition matrices derived from MAST public traces
├── figs/             Publication-ready figures
└── docs/             Supplementary documentation
```

## Install and run

```bash
cd code
pip install -e ./mcr

# Run all simulation studies
python sim/SS1_cf_vs_mc.py
python sim/SS2_perturbation.py
python sim/SS3_correlation.py
python sim/SS4_nhpp_limit.py
python sim/SS5_prism_cross.py    # requires PRISM (optional)
```

Each simulation script writes a `.pdf` figure into `../figs/` and a JSON summary into `../data/synthetic/`.

## Reproducibility

Every simulation uses explicit random seeds recorded at the top of each script.
Closed-form computations in `mcr.reliability` are deterministic up to floating-point
tolerance. Expected runtime on a modern laptop: ~15 minutes for SS1–SS4 combined.

## Venue

Targeted at ISSRE 2026 (Research track) with TSE as a journal upgrade path.
See `../MCRAgentPlan.md` for the full execution plan and go/no-go gates.
# MCR-Agent
