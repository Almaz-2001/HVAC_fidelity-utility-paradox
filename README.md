# The Fidelity–Utility Paradox

**Surrogate-based reinforcement learning for HVAC control.**

This repository accompanies the manuscript under review at *IEEE Access*. It is a
**results-reproducibility package**: it contains the code, configurations, trained
controllers, calibration data, the result tables, and the data-driven generators
and figure sources behind every result. The typeset manuscript and supplementary
PDFs are the journal's artifact and are not vendored here.

> **Headline result — the fidelity–utility paradox.** On the BOPTEST
> `bestest_air` testcase, a physically-calibrated grey-box surrogate **GB** (v3.5,
> RC–Neural-ODE) is the *more accurate* predictor (24 h rollout RMSE
> 0.644 °C vs 1.557 °C for a black-box surrogate **BB**), yet used directly as a
> reinforcement-learning training environment it produces an *unusable* controller
> (maintenance score `m_s = 1.046`, >77 % comfort violation). Training on the
> predictively weaker **BB** yields a usable controller. The effect is consistent with a
> temporal-coarseness (coarse per-step increment) effect, not the model class: retraining the *same* black-box
> surrogate at the finer 15-min control resolution makes it strictly *more* accurate
> (0.876 °C) yet also unusable as a training environment (`m_s = 1.14`/`1.21`,
> >85 % violation), on a par with the calibrated twin. A **hybrid** that uses **BB** for
> smooth rollout dynamics and a *frozen* **GB** as a per-step reward-shaping censor
> recovers the best **cross-window robustness** (`m_s = 0.041` typical, <5 % violation
> on both windows) at ~85× the live-simulator throughput. The optimal censor weight is
> **controller-family specific**, and cross-testcase transfer resolves into a
> component-level boundary: the inverse-calibration pipeline transfers, the frozen
> policy does not.

> **Predictive error misses a second defect entirely.** The finer-resolution
> retraining also **inverts the model's response to the control command** in about
> nine sampled states out of ten — a hotter supply-temperature command predicting a
> colder zone — reproducibly across four independent training draws, while the 24 h
> rollout error improves in every one. Constraining training to preserve the response
> sign restores directional validity to 100 % at a *lower* rollout error still
> (0.745 °C), and a controller trained on that corrected surrogate fails as well
> (`m_s` 1.426 peak / 1.597 typical). Accuracy, model class and response sign can all
> be held right while the surrogate stays unusable as a training environment.
> Both defects are measurable on the surrogate alone, before any controller is
> trained — see the checkpoint-only probes in [`REPRODUCE.md`](REPRODUCE.md).

> **Not specific to reinforcement learning.** A pre-registered receding-horizon MPC
> planning on the same surrogates reproduces the same inverted ordering, so the
> effect belongs to optimising *through* the surrogate rather than to
> policy-gradient search. That was our own prediction, and it is falsified.

> **Naming.** The manuscript uses **BB** (black-box surrogate) and **GB** (grey-box
> RC–Neural-ODE) throughout; the code and these docs retain the legacy artifact names
> **v3** (= BB) and **v3.5** (= GB). The internal "Block 1 / 2 / 3" organisation
> corresponds to **Results I / II / III** of the paper.

---

## Repository structure

```
configs/        YAML configs for surrogates, controllers, and the Block 3 protocol
surrogate/      surrogate model definitions (v3 = BB black-box, v3.5 = GB RC-Neural-ODE)
layers/         shared neural-network building blocks
envs/           BOPTEST gym environments, observation wrappers, reward shaping
training/       training pipelines (PPO, HDRL, MORL surrogate-pretrain pipeline)
evaluation/     benchmarking, BOPTEST runners, metric computation (m_s, CV(RMSE), NMBE)
models/         24 trained Stable-Baselines3 policy checkpoints (.zip)
data/           calibration / training corpora (CSV) and manifests
reports/        result CSV/JSON tables that back the manuscript figures and tables
docs/
  paper_combined/figures/      the manuscript + supplementary figures (PDF/PNG)
  results{1,2,3}_*_overleaf/    per-block data-driven generators (build_*.py): each
                                reads the reports/ and outputs/ artifacts and (re)builds
                                its block's figures and the data-filled section text
roadmap.md      provenance maps: every figure/table/number -> its source artifact
```

## What is and is **not** included

**Included:** all source code, configurations, the 24 trained controller
checkpoints, the calibration/training corpora, the result tables (CSV/JSON), the
data-driven section generators, and the manuscript + supplementary figures.

**Not included (by design):**
- The **typeset LaTeX manuscript and supplementary PDFs** (the journal's artifact).
  Every figure, table, and inline number remains reproducible here from the data and
  the generators; the content-to-artifact provenance maps are in
  [`roadmap.md`](roadmap.md).
- `outputs/` — the ~7 GB of raw per-seed run artifacts are **not** shipped.
  However, the small audit trail referenced by the provenance maps — calibration
  summaries, scenario/pipeline manifests, and the Stage-B `C_zon` histories — **is**
  included under `outputs/` (preserving the original paths) so that the maps in
  [`roadmap.md`](roadmap.md) (§3.2, §11.1, §15.7) resolve directly in this
  repository. A map decoding the directory names is in
  [`outputs/README.md`](outputs/README.md). The full raw per-seed artifacts are
  available from the authors on reasonable request.
- **BOPTEST** — the building emulator is a separate open-source project and is
  **not** vendored here (it is ~GB with its own repository and license). We used the
  containerized BOPTEST **service** deployment (`web`/`provision` Docker-Compose),
  version **`1.0.0-dev`**, from <https://github.com/ibpsa/project1-boptest>. The
  training code runs in its own container (built from the shipped
  [`Dockerfile`](Dockerfile)) on the same Docker network, reaching BOPTEST at
  `http://web:8000` (`boptest_url` in `configs/`). The exact step-by-step bring-up is
  in [REPRODUCE.md](REPRODUCE.md) (Level C, step 1).
- **Reference PDFs** — the cited papers under the working tree are third-party
  copyrighted material and are deliberately excluded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A running BOPTEST instance (Docker or local) is required for any live-simulator
evaluation; surrogate-only training and the figure/section generators do not need it.

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

**Multi-seed robustness (N=3 seeds {42,43,44}).** The pure-v3, matched-resolution-v3,
and hybrid controllers are also reported as a seed band: each `usable`/`collapse`
verdict is seed-stable (usable controllers stay sub-5 % on every seed; the
matched-resolution v3 collapses on every seed). Train/benchmark the extra seeds with
`run_block2.py thermostatic-train/-benchmark --variant <v> --seed <s>`, then aggregate
with `run_block2.py seed-band` (see [REPRODUCE.md](REPRODUCE.md)).

**Measured paradox mechanism.** The reason the *more accurate* surrogates fail as
training environments is measured directly on the surrogates, not just asserted:
`run_block2.py surface-diagnostic` probes each surrogate's own one-step action map
(no controller, no BOPTEST) and reports a scale-free *relative roughness*
(curvature ÷ slope, independent of the control step length). The only usable training
environment, the hourly v3, exposes the smoothest action→next-temperature landscape
(`0.169`), whereas the matched-resolution v3 and the calibrated v3.5 are `9.4×` and
`7.9×` rougher — the surrogate-side cause of the near-bang-bang action gap of the
collapsed controllers (`reports/block2_mechanism_surface_sharpness.csv`).

**Regenerate the figures and the data-filled section text** (each block, data-driven
from the `reports/` and `outputs/` artifacts):

```bash
python docs/results1_digital_twin_overleaf/build_results1_overleaf.py
python docs/results2_control_overleaf/build_results2_overleaf.py
python docs/results3_transferability_overleaf/build_results3_overleaf.py
```

Each generator rebuilds its block's figures and prints the section text with every
number substituted from the source CSV/JSON — i.e., the manuscript content is
*computed from the data*, not hand-entered. (The typeset LaTeX manuscript itself is
the journal's artifact and is not included here.)

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
  title   = {Surrogate Selection for Reinforcement-Learning HVAC Control:
             Step Size, Not Predictive Accuracy, Predicts Control Utility},
  author  = {Mukhanbet, Aksultan and Sapargali, Almaz and Aibagarov, Serik and
             Daribayev, Beimbet and Shinassylov, Shona and Trigo, Paulo},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## Funding

This research was funded by the Committee of Science of the Ministry of Science and
Higher Education of the Republic of Kazakhstan (Grant No. AP23488794).

## License

Released under the [MIT License](LICENSE).
