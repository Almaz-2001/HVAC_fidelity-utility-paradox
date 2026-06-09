# The Fidelity–Utility Paradox

**Surrogate-based reinforcement learning for HVAC control.**

This repository accompanies the manuscript submitted to *Results in Engineering*
(Elsevier). It contains the code, configurations, trained controllers, calibration
data, and the full LaTeX sources of the manuscript and its supplementary material.

> **Headline result — the fidelity–utility paradox.** On the BOPTEST
> `bestest_air` testcase, a physically-calibrated grey-box surrogate (v3.5,
> RC–Neural-ODE) is the *more accurate* predictor (24 h rollout RMSE
> 0.644 °C vs 1.557 °C for a black-box surrogate v3), yet used directly as a
> reinforcement-learning training environment it produces an *unusable* controller
> (maintenance score `m_s = 1.046`, >77 % comfort violation). Training on the
> predictively weaker v3 yields a usable controller. A **hybrid** that uses v3 for
> smooth rollout dynamics and a *frozen* v3.5 as a per-step reward-shaping censor
> recovers the best controller (`m_s = 0.041` typical, <5 % violation) at ~85×
> the live-simulator throughput. The optimal censor weight is **controller-family
> specific**, and cross-testcase transfer resolves into a component-level boundary:
> the inverse-calibration pipeline transfers, the frozen policy does not.

---

## Repository structure

```
configs/        YAML configs for surrogates, controllers, and the Block 3 protocol
surrogate/      surrogate model definitions (v3 black-box, v3.5 RC-Neural-ODE)
layers/         shared neural-network building blocks
envs/           BOPTEST gym environments, observation wrappers, reward shaping
training/       training pipelines (PPO, HDRL, MORL surrogate-pretrain pipeline)
evaluation/     benchmarking, BOPTEST runners, metric computation (m_s, CV(RMSE), NMBE)
models/         24 trained Stable-Baselines3 policy checkpoints (.zip)
data/           calibration / training corpora (CSV) and manifests
reports/        result CSV/JSON tables that back the manuscript figures and tables
docs/
  paper_combined/   self-contained LaTeX bundle (main_paper.tex + supplementary.tex)
  build_integrated_paper.py    assembles the manuscript + supplementary from sources
  results{1,2,3}_*_overleaf/   per-block data-driven generators (build_*.py, main.tex)
roadmap.md      provenance maps: every figure/table/number -> its source artifact
```

## What is and is **not** included

**Included:** all source code, configurations, the 24 trained controller
checkpoints, the calibration/training corpora, the result tables (CSV/JSON), and
the complete manuscript + supplementary LaTeX sources with figures.

**Not included (by design):**
- `outputs/` — the ~7 GB of raw per-seed run artifacts are **not** shipped.
  However, the small audit trail referenced by the provenance maps — calibration
  summaries, scenario/pipeline manifests, and the Stage-B `C_zon` histories — **is**
  included under `outputs/` (preserving the original paths) so that the maps in
  [`roadmap.md`](roadmap.md) (§3.2, §11.1, §15.7) resolve directly in this
  repository. The full raw per-seed artifacts are available from the authors on
  reasonable request.
- **BOPTEST** — the building emulator is a separate open-source project and is
  **not** vendored here. Install it from
  <https://github.com/ibpsa/project1-boptest> and point the environments at your
  local runtime (see `configs/`).
- **Reference PDFs** — the cited papers under the working tree are third-party
  copyrighted material and are deliberately excluded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A running BOPTEST instance (Docker or local) is required for any live-simulator
evaluation; surrogate-only training and the paper build do not need it.

## Reproducing the results

See **[REPRODUCE.md](REPRODUCE.md)** for the full step-by-step guide. In short: one
command per block (`python docs/results{1,2,3}_*_overleaf/build_*.py`) regenerates
every figure, table, and number of Blocks 1–3 from the shipped artifacts **without
BOPTEST** (Level B); re-running the experiments end-to-end (Level C) needs a BOPTEST
instance and substantial compute.

**Primary controller pipeline (MORL surrogate-pretrain → BOPTEST validation):**

```bash
python training/run_morl_surrogate_pipeline.py --mode full --seed 42
```

Stages: `pretrain` (MORL/PPO on the surrogate) → `finetune` (transfer to BOPTEST)
→ `eval` (yearly BOPTEST validation). The `full_eram` mode runs the vector-reward
ERAM branch. Configs live in `configs/morl_surrogate_ppo/`.

**Block 2 controller comparison and report tables:**

```bash
python evaluation/run_block2.py build-reports
```

**Rebuild the manuscript and supplementary PDFs** (requires a LaTeX
distribution with `pdflatex` + `bibtex`):

```bash
python docs/build_integrated_paper.py          # regenerates the .tex bundle
cd docs/paper_combined
pdflatex main_paper.tex && bibtex main_paper && pdflatex main_paper.tex && pdflatex main_paper.tex
pdflatex supplementary.tex && pdflatex supplementary.tex
```

The three Results sections are themselves regenerated, data-driven, by
`docs/results{1,2,3}_*_overleaf/build_*.py`.

## Provenance and data availability

Every figure, table, and inline number in the manuscript is read directly from the
versioned artifacts. The content-to-artifact maps are in [`roadmap.md`](roadmap.md)
(§3.2 for Results I, §11.1 for Results II, §15.7 for Results III) and are mirrored
as Supplementary Tables S1–S3 in the manuscript. The result tables themselves are
provided here under `reports/`.

## Citation

Please cite the manuscript (details to be completed on acceptance):

```bibtex
@article{HVAC_FidelityUtility_2026,
  title   = {The Fidelity--Utility Paradox in Surrogate-Based Reinforcement
             Learning for HVAC Control},
  author  = {<AUTHORS>},
  journal = {Results in Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).
