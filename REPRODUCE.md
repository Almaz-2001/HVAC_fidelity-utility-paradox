# Reproducing the results

This repository supports two levels of reproduction. **Level B** regenerates every
figure, table, and reported number of Blocks 1–3 directly from the shipped
artifacts — no BOPTEST and no GPU, in minutes. **Level C** re-runs the experiments
end-to-end and requires a BOPTEST instance and substantial compute.

> **Naming.** Code artifacts keep the legacy names **v3** (= BB, black-box surrogate) and
> **v3.5** (= GB, grey-box RC–Neural-ODE); the manuscript uses BB/GB. Blocks 1/2/3
> correspond to Results I/II/III of the paper.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Level B — regenerate all figures, tables, and numbers (no BOPTEST)

**One command — regenerate every main + supplementary figure and the graphical abstract:**

```bash
python make_figures.py          # ~1 min, no BOPTEST, no GPU  (or: ./make_all_figures.sh)
python make_figures.py --list   # show the ordered generator list it will run
```

`make_figures.py` reads [`paper_artifacts/figure_manifest.yaml`](paper_artifacts/figure_manifest.yaml)
and runs every generator it lists in dependency order (the scripts that rewrite the
`reports/*.csv` consumed by `_figstyle.paper_numbers()` run first; Fig 1/4/8 and the
graphical abstract run last), printing a per-script PASS/FAIL summary.

**Or run the three per-block builders individually.** Each reads the committed
`reports/`, `outputs/` (audit artifacts), and `data/` files and writes the section's
`main.tex` (with every table) plus its figures:

```bash
# Block 1 — digital-twin fidelity (surrogate accuracy, C_zon, Fisher CI, physics audit)
python docs/results1_digital_twin_overleaf/build_results1_overleaf.py

# Block 2 — downstream control utility (paradox, matched-resolution ablation, hybrid, HDRL, MORL)
python docs/results2_control_overleaf/build_results2_overleaf.py

# Block 3 — pre-registered transferability (transfer matrix, C_zon ratio consistency)
python docs/results3_transferability_overleaf/build_results3_overleaf.py
```

Each command (re)creates the block's figures and prints the block's section text
with every number substituted from the source artifacts. Every figure and every
data-driven number (e.g. the 0.644 / 1.557 / 0.876 °C rollout RMSEs, the controller
scores `m_s = 1.046` for direct v3.5, `1.14`/`1.21` for the matched-resolution v3
ablation, and `0.041` typical for the hybrid, the 60.2–87.8 % transfer-RMSE gains)
is recomputed from the committed `reports/`/`outputs/` files and matches the
published manuscript. The figure → generator-script → source-artefact mapping for
every main figure and the graphical abstract is catalogued in
[`paper_artifacts/figure_manifest.yaml`](paper_artifacts/figure_manifest.yaml).

> **Scope note.** This is a results-reproducibility package: the typeset LaTeX
> manuscript and supplementary PDFs (the journal's artifact) are not included. What
> *is* here is the machinery that produces the results they report — so every
> number in the paper can be regenerated from the data by the generators above, and
> traced to its source artifact through the provenance maps in
> [`roadmap.md`](roadmap.md) (§3.2, §11.1, §15.7). The manuscript figures are also
> committed under `docs/paper_combined/figures/`.

### The manuscript figures themselves

Every figure in the paper is native TikZ/pgfplots, not a raster. The sources are
in `docs/paper_combined/`: one standalone `fig_*_tikz.tex` per figure, a shared
`_pgfstyle.tex` holding the palette and unit macros, and the numbers they plot in
`tikz_data/*.dat`. Any one of them compiles on its own:

```bash
cd docs/paper_combined
pdflatex fig_seed_band_tikz.tex        # -> fig_seed_band_tikz.pdf
python propagate_tikz.py               # compile all, copy into figures/
```

The `.dat` files are exported from the committed CSVs rather than typed by hand
(`export_tikz_data.py`, plus `export_surface_data.py` and `export_rie09_data.py`,
which load the `.pt` surrogates directly). So a figure can be traced back to the
artifact it came from without leaving the repository.

---

## Level C — re-run the experiments end-to-end (needs BOPTEST + compute)

This re-derives the underlying data rather than re-plotting it. It requires the
BOPTEST emulator and is computationally heavy (millions of policy-gradient steps
across `N = 5` seeds and several controller families; hours to days).

1. **Stand up BOPTEST and the training container.** BOPTEST is a separate
   open-source project and is deliberately **not** vendored here (it is ~GB and has its
   own repository and license). We ran the **containerized BOPTEST service** (the
   `web`/`provision` Docker-Compose deployment), version **`1.0.0-dev`**. The training
   code runs in its own container on the same Docker network, so the environments reach
   BOPTEST at the in-network URL `http://web:8000` (set in `configs/env.yaml` and
   `configs/boptest_15min/env.yaml` as `boptest_url`).

   The exact workflow we used:

   ```bash
   # (a) Get BOPTEST (service deployment) at the pinned version and start it.
   #     Clone into a directory named e.g. boptest_rte; its compose project name then
   #     defines the docker network name <dir>_default referenced below.
   git clone https://github.com/ibpsa/project1-boptest.git boptest_rte
   cd boptest_rte && git checkout 1.0.0-dev          # match the version we used
   docker compose up -d
   docker compose up -d web
   docker compose run --rm provision                  # registers the testcases
   cd ..

   # (b) Build the training image from the shipped Dockerfile (base: sinergym).
   docker build -t hvac-drl:latest .

   # (c) Enter the training container ON THE BOPTEST NETWORK so http://web:8000 resolves.
   #     Drop --gpus all if you have no GPU; replace boptest_rte_default if your BOPTEST
   #     clone directory has a different name.
   docker run -it --gpus all --network boptest_rte_default \
       -v "${PWD}:/app" -w /app --name hvac-drl hvac-drl:latest bash
   ```

   All Level-C commands below are run **inside** this container. Testcases needed:
   `bestest_air` (Blocks 1–2) and the hydronic family — `bestest_hydronic`,
   `bestest_hydronic_heat_pump`, `singlezone_commercial_hydronic` (Block 3). If you run
   BOPTEST differently (e.g. a single-container testcase on `localhost`), set
   `boptest_url` in the configs to your runtime URL instead.

2. **Block 1 — surrogates and calibration:**
   ```bash
   python surrogate/train_surrogate_v2.py                 # black-box v3
   python surrogate/calibrate_surrogate_v35.py            # physics-informed v3.5 (Stage A/B/C)
   python evaluation/run_block1.py                        # rollout fidelity + reports
   ```

3. **Block 2 — controllers (surrogate-pretrain → BOPTEST):**
   ```bash
   python training/run_morl_surrogate_pipeline.py --mode full --seed 42   # primary MORL line
   python evaluation/run_block2.py build-reports                          # KPI tables/figures
   ```
   Thermostatic-PPO and HDRL lines have their own `training/train_*.py` runners.

   **Multi-seed robustness band (pure v3, matched-resolution v3, hybrid; N=3 seeds).**
   The seed-42 controllers are reported canonically; the seed band that backs the
   supplementary seed-robustness table is regenerated by training and benchmarking the
   two extra seeds and aggregating (every individual seed stays sub-5% for the usable
   controllers, while the matched-resolution v3 collapses on every seed):
   ```bash
   for s in 43 44; do
     for v in pure pure_v3_15min hybrid_l010; do
       python evaluation/run_block2.py thermostatic-train     --variant $v --seed $s
       python evaluation/run_block2.py thermostatic-benchmark --variant $v --seed $s
     done
   done
   python evaluation/run_block2.py seed-band --seeds 42,43,44   # -> reports/block2_thermostatic_seed_band.csv
   ```

   The direct-GB (`v35_direct`) and Delta-t-rescaled (`pure_dt_scaled`) collapse controls
   are seed-swept the same way but evaluated via `thermostatic-transfer` (they run zero-shot
   on live BOPTEST rather than through the benchmark loop); every seed stays above the
   `m_s = 1` collapse line, confirming both collapses are seed-stable:
   ```bash
   for s in 42 43 44; do
     for v in v35_direct pure_dt_scaled; do
       python evaluation/run_block2.py thermostatic-train    --variant $v --seed $s
       python evaluation/run_block2.py thermostatic-transfer --variant $v --seed $s
     done
   done
   ```

   **Measured paradox mechanism (response-surface smoothness).** This probe needs only
   the three committed surrogate checkpoints (no controller, no BOPTEST): it sweeps each
   surrogate's one-step action map and reports the scale-free relative roughness
   (curvature ÷ slope) that separates the usable hourly v3 from the collapsing matched-v3
   and v3.5 backends:
   ```bash
   python evaluation/run_block2.py surface-diagnostic   # -> reports/block2_mechanism_surface_sharpness.csv
   ```

   **Directional validity of the control response.** Also checkpoint-only, also no
   BOPTEST. Predictive error cannot see a surrogate whose response to the control
   command has the wrong sign; this samples 400 states and scores each actuated
   input separately, because a controller finds whichever input is wrong:
   ```bash
   python evaluation/check_surrogate_response_sign.py
   python evaluation/audit_matched_bb_sign.py
   ```
   The matched-resolution black box comes out inverted in about nine states out of
   ten while its rollout error improves. Committed evidence:
   [`reports/block1_matched_bb_seed_audit.json`](reports/block1_matched_bb_seed_audit.json)
   (four training draws) and
   [`reports/block1_matched_bb_monotonic_audit.json`](reports/block1_matched_bb_monotonic_audit.json)
   (the constrained retraining). The checkpoints for every draw are under
   `outputs/surrogate_v3_15min_{matched,mono}_seed*/`, so both audits re-run from
   this repository alone.

4. **Block 3 — pre-registered transferability:**
   ```bash
   python evaluation/run_block3_surrogate_recalibration.py   # Stage A/B/C on the hydronic family
   ```

5. **Monotonicity-constrained retraining.** `train_surrogate_backbone.py` takes
   `--lambda-mono`, `--mono-margin`, `--mono-jitter` and `--lambda-mono-fan`. The
   fan penalty is signed — it follows `sign(T_sup − T_zone)`, since with supply
   colder than the zone more fan must *lower* the temperature:
   ```bash
   python evaluation/run_matched_bb_seed_audit.py --seeds 42 43 44
   python evaluation/run_matched_bb_monotonic.py  --seeds 42 43 44
   python evaluation/run_matched_bb_monotonic.py  --fan-sweep 500 1000 2000
   ```
   Constraining one channel relocates the defect to the other. Constraining both
   yields a surrogate that is accurate, directionally valid, and still trains a
   failing controller.

6. **Receding-horizon MPC baseline (H5b).** Pre-registered in
   [`configs/mpc_baseline_preregistration.yaml`](configs/mpc_baseline_preregistration.yaml),
   whose commit is the audit anchor. It carries an admission gate — a surrogate
   enters the comparison only at ≥95 % directional validity on *every* actuated
   input — and keeps the superseded H5 verbatim rather than rewriting it:
   ```bash
   python evaluation/run_mpc_baseline.py --backends mpc_bb_mono mpc_bb_hourly mpc_gb
   ```
   H5b is **falsified**: a gradient-based planner reproduces the same inverted
   ordering, so the effect belongs to optimising through the surrogate rather than
   to policy-gradient search specifically
   ([`reports/block2_mpc_baseline_report.md`](reports/block2_mpc_baseline_report.md)).

   The first execution of H5 was invalidated by a defect in our own planner, and
   that run is kept rather than deleted
   ([`reports/block2_mpc_h5_invalidated.md`](reports/block2_mpc_h5_invalidated.md)).
   Re-running quietly would have left the pre-registration looking untouched when
   a completed run had already been seen.

7. **Seed-replicated HDRL censor-weight sweep.** `train_hdrl.py` takes `--seed`;
   before that only the environments were seeded and PPO drew its own entropy, so
   a "seed sweep" was not reproducible. 12 cells, resumable:
   ```bash
   python evaluation/run_hdrl_seed_sweep.py --stage train     --skip-existing
   python evaluation/run_hdrl_seed_sweep.py --stage benchmark --skip-existing
   python evaluation/build_hdrl_seed_band.py
   ```
   Protocol and cost breakdown: [`reports/hdrl_seed_sweep_runbook.md`](reports/hdrl_seed_sweep_runbook.md).

**Determinism note.** Exact bit-for-bit reproduction is not expected: results
depend on the BOPTEST version, RNG seeds, hardware, and library versions (the
throughput figures in particular are hardware-dependent). The scientific
conclusions — the fidelity–utility paradox, controller-family specificity, and the
component-level transfer boundary — are robust to these factors and are what
Level C verifies.

---

## What each level establishes

| Level | Needs BOPTEST? | Time | Verifies |
|-------|----------------|------|----------|
| **A** (inspect) | no | minutes | every number traces to a committed artifact via [`roadmap.md`](roadmap.md) |
| **B** (regenerate) | no | minutes | the figures/tables faithfully represent the committed data |
| **C** (re-run) | yes | hours–days | the committed data itself can be re-derived from the experiments |

Two of the paper's central claims sit between B and C: they re-derive a result
rather than re-plot it, but need only the committed `.pt` checkpoints — no
BOPTEST, no GPU, seconds to run. Worth doing before Level C, since they are the
cheapest independent check on the main finding:

```bash
python evaluation/run_block2.py surface-diagnostic     # per-step increment / roughness
python evaluation/check_surrogate_response_sign.py     # directional validity per input
```

Provenance for every figure, table, and number is mapped in
[`roadmap.md`](roadmap.md) (§3.2 Block 1, §11.1 Block 2, §15.7 Block 3) and
mirrored as Supplementary Tables S1–S3 in the manuscript.
