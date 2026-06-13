# HVAC DRL/MORL Reproduction Roadmap

Date: 2026-05-18

This roadmap is the command-level path to reproduce the current article state:

- Block 1: surrogate fidelity and Hou-and-Evins numerical artifacts.
- Block 2: thermostatic, HDRL, and MORL control results on `bestest_air`.
- Block 3: hydronic-family transferability with pre-registered adapters.
- Paper artifacts: canonical CSV evidence, tables, and figures in
  `paper_artifacts/`; manuscript DOCX drafts in `docs/`.

Paper artifact dataflow:

```text
outputs/        raw run artifacts, large, ignored by Git
reports/*.csv   compact computed evidence from outputs, tracked by Git
paper_artifacts final paper-facing package: figures, tables, CSV evidence, manifests
```

Do not commit bulk `outputs/` or generated figure variants under
`reports/figures/`. The canonical GitHub-facing paper package is
`paper_artifacts/`.

Use Linux/bash syntax inside the project container (`/app`). Do not paste Windows
PowerShell backticks into bash. Use `\` for line continuation.

## 0. Runtime Checks

If BOPTEST RTE is not running yet, start it from the BOPTEST runtime folder on
the host machine:

```powershell
cd C:\Users\user\Desktop\HVAC_DRL_MORL\boptest_rte
docker compose up -d
docker compose ps
```

Then run the project commands from inside the project container or another
environment where `http://web:8000` resolves.

```bash
cd /app
python3 - <<'PY'
import torch, gymnasium, stable_baselines3, pandas, numpy, requests
print("python_env_ok")
print(requests.get("http://web:8000/version", timeout=20).json())
PY
```

If `testcases/bestest_air/select` hangs, restart the BOPTEST RTE lifecycle before
running yearly validation:

```bash
python3 - <<'PY'
import requests, time
t0 = time.time()
r = requests.post("http://web:8000/testcases/bestest_air/select", json={}, timeout=120)
print("status:", r.status_code)
print("elapsed:", time.time() - t0)
print(r.text[:500])
PY
```

## 0.5. Dependency Table

The structural dependencies between roadmap sections are maintained as a
separate file, `roadmap_dependencies.md`, in the repository root. That file
is the single source of truth for which prerequisite sections must already
be complete before any given section can run. It covers all three blocks
(Block 1, Block 2, Block 3) and explicitly marks independent sections
(runtime checks, audit anchors, cleanup workflow, pre-registration
manifests).

To inspect dependencies before planning a re-run:

```bash
cat roadmap_dependencies.md
```

Within each command-bearing section below, dependencies are kept terse
because the table already lists them; only non-obvious or cross-section
dependencies are repeated inline.

## 1. Block 1: v3 Direct-TSup Surrogate

Block 1 commands are routed through `evaluation/run_block1.py`. The wrapper
does not change underlying scripts; it only fixes canonical paths, presets,
sweep values, and the two-step v3.5 calibration so the roadmap stays short.
To inspect any command without running compute, add `--dry-run` before the
sub-command, for example:

```bash
python3 -B evaluation/run_block1.py --dry-run v3-train
```

To run the entire Block 1 pipeline end-to-end (data → v3 → v3.5 → rollouts →
reports → speed benchmark) in one call:

```bash
python3 -B evaluation/run_block1.py all
```

The four sub-commands below are the per-stage entry points used during
incremental re-runs.

Collect the direct supply-temperature dataset:

```bash
python3 -B evaluation/run_block1.py collect-data
```

Train the v3 backbone (also renames the checkpoint to the canonical name
`rc_node_v3_tsupply.pt`; pass `--skip-rename` to keep only the compat name):

```bash
python3 -B evaluation/run_block1.py v3-train
```

Defaults are the canonical training hyperparameters (`--epochs 500
--batch-size 256 --lr 1e-3 --hidden-dim 64 --patience 30`). Override any of
them on the same line for ablation runs.

Expected canonical model:

```bash
ls -lh outputs/surrogate_v2/rc_node_v3_tsupply.pt
```

## 2. Block 1: v3.5 Inverse Calibration

Sequencing note:

- Technically, v3.5 calibration does not require the trained v3 checkpoint.
- Semantically, v3.5 is the physical extension of the v3 surrogate family and
  uses the same direct-TSup modeling convention.
- Stage A/B/C means: Stage A cleans and aligns 15-min BOPTEST telemetry, Stage B
  solves the inverse grey-box task for `C_zon`, and Stage C calibrates residual
  heads while preserving the identified physical backbone.

**The canonical v3.5 artifact is built by TWO sequential preset runs.**
Step 1 (`block1_15min_episodeaware`) runs Stage A + B + C with the
rollout-heads selection metric — this is where `C_zon` is identified
(120 Stage-B epochs) and the temperature head is calibrated.  Step 2
(`block1_15min_power_head_only`) reads `init_summary_json` from Step 1's
output directory, freezes `C_zon`, and re-runs Stage C with `power_head_only`
mode (80 epochs) for a tighter power calibration.  Skipping Step 1 on a
fresh repository is a silent correctness bug — Step 2's preset hard-codes
the init-JSON path from Step 1's output, so calling Step 2 alone will fail
with a missing-file error.  The wrapper's `--preset canonical` runs both
steps in the correct order; `--preset episodeaware` and `--preset
power_head_only` are available for selective re-runs.

Prepare the 15-min corpus used by v3.5:

```bash
python3 -B evaluation/run_block1.py prepare-15min
```

Run the calibrated v3.5 physical twin (both presets in order):

```bash
python3 -B evaluation/run_block1.py v35-calibrate --preset canonical
```

Expected summary:

```bash
ls -lh outputs/surrogate_v35_inverse_boptest_15min_episodeaware/calibration_summary_boptest_v35.json
ls -lh outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json
```

Validate prepared rollouts for v3 and v3.5 (use `--variant v3` or
`--variant v35` to run only one side):

```bash
python3 -B evaluation/run_block1.py validate-rollouts --variant all
```

## 2.5. Block 1: Corpus-Matched v3 Retraining (Reviewer Mitigation)

Why this section exists:

- The canonical v3 surrogate is trained on a 51,200-row **hourly** corpus
  (`data/surrogate_v2/boptest_v2_tsupply.csv`); the canonical v3.5 surrogate
  is trained on a 10,744-row **15-minute** corpus
  (`data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv`).
- A reviewer can correctly argue that comparing v3 (24h RMSE ~1.56 °C) to
  calibrated v3.5 (24h RMSE 0.644 °C) is confounded by these dataset
  differences: corpus size, time step, season coverage, and policy mix all
  differ simultaneously.
- This section retrains v3 on the *same* 10,744-row 15-min corpus used by
  v3.5.  The v3 model has no hard-coded timestep in its forward pass
  (`t_next = t_zone + dT(x)`, see `surrogate/rc_node_v2.py`), so it can be
  trained on 15-min data without modifying the architecture.
- The resulting *apples-to-apples* comparison isolates the contribution of
  Stage A/B/C inverse calibration from any corpus effect, which is what
  reviewers will ask for.

Dependencies:

- This section depends on Section 1 ONLY for the original canonical `v3`
  checkpoint (used as the reference row in the comparison report).
- It depends on Section 2 for both the prepared 15-min corpus and the
  canonical calibrated `v3.5` summary.
- It does NOT change any Block 2 / Block 3 artifact; the canonical v3 used
  for downstream PPO training remains the original hourly checkpoint.

Retrain v3 on the 15-min v3.5 corpus (~30-60 min wall-clock on one CPU):

```bash
python3 -B evaluation/run_block1.py v3-train-15min
```

Defaults match the canonical v3 training hyperparameters
(`--epochs 500 --batch-size 256 --lr 1e-3 --hidden-dim 64 --patience 30`)
so any RMSE difference is attributable to the corpus, not to the optimizer.
Output: `outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt`.

Validate the matched-corpus checkpoint on the same prepared rollouts that
v3 and v3.5 were already evaluated against:

```bash
python3 -B evaluation/run_block1.py validate-rollouts --variant v3_15min
```

Or run all three rollout validations (v3, v3_15min, v3.5) in one call:

```bash
python3 -B evaluation/run_block1.py validate-rollouts --variant matched
```

Build the corpus-matched comparison report:

```bash
python3 -B evaluation/run_block1.py build-corpus-matched-report
```

This produces `reports/block1_corpus_matched_comparison.csv` and
`reports/block1_corpus_matched_comparison.json` with the four-variant table
(v3 hourly, v3 15-min matched, raw v3.5, calibrated v3.5) and a structured
decomposition of the RMSE drop into a `delta_corpus` term (v3 hourly →
v3 15-min) and a `delta_calibration` term (raw v3.5 → calibrated v3.5).
The `json` payload includes a single-sentence interpretation suitable for
quoting verbatim in the §5.3 reviewer-mitigation paragraph of the paper.

Expected artifacts:

```bash
ls -lh outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt
ls outputs/surrogate_v3_15min_matched_rollout_prepared/v3
cat reports/block1_corpus_matched_comparison.csv
```

The reviewer-defensible claim is: if `delta_calibration` is several times
larger than `delta_corpus`, the headline finding "Stage A/B/C inverse
calibration is responsible for the v3.5 fidelity advantage" survives the
matched-corpus check.  If they are comparable, the paper's §5.3 framing
needs to be softened from "calibration drives the gap" to "calibration and
corpus jointly drive the gap"; the wrapper's report computes the exact
percentage split so the paper text can quote the correct attribution.

## 3. Block 1: Article-Facing Fidelity Tables and Figures

Transfer-gap diagnostics are intentionally not placed here because they need
trained controllers from Sections 4 and 5. They are run later in Section 5.5.

Build the Hou-and-Evins numerical artifacts and the real-data article
figures in one pass:

```bash
python3 -B evaluation/run_block1.py build-reports
```

Build the speed benchmark table. This compares the same BOPTEST RTE HTTP loop
used by the paper against local surrogate stepping. The defaults
(`--episodes 100 --steps-per-episode 96 --step-sec 900` against
`http://web:8000`) match the canonical paper protocol:

```bash
python3 -B evaluation/run_block1.py speed-benchmark
```

Expected article-facing artifacts:

```bash
ls reports/hou_evins_*.csv
cat reports/speed_benchmark_table.csv
```

Current headline speed result:

- Hybrid backend: about `1,786.8` env-steps/s on one CPU thread.
- Conservative speed-up versus the standard BOPTEST RTE HTTP-Docker deployment
  used in production benchmarking: `85.0x`.

### 3.1. Boundary: why direct-v3.5 failure and hybridization are in Block 2

Block 1 stops at **digital-twin fidelity**:

- v3 direct-TSup surrogate is trained and validated as a fast rollout model.
- v3.5 is calibrated and validated as a physically informed predictive twin.
- Corpus-matched v3 retraining separates data-resolution effects from
  Stage A/B/C calibration effects.
- Hou-and-Evins tables and speed benchmark establish predictive validity and
  training feasibility.

The negative result "direct calibrated v3.5 is not a usable PPO rollout
environment" is **not** a Block 1 result. It requires a trained controller and
live BOPTEST transfer, so it appears only after the first Block 2 controller
baseline has been established.

The intended Block 2 sequence is:

1. Section 4: train and benchmark the **pure v3 thermostatic PPO** baseline.
2. Section 4.5: run the **direct v3.5 warm-start negative control**.
3. Section 5: run the **hybrid v3/v3.5 sweep** motivated by that negative
   result.
4. Section 5.5: run transfer-gap diagnostics that compare pure v3, direct
   v3.5, and hybrid side by side.

The hybrid backend is also a Block 2 controller-training backend, not a
separate Block 1 surrogate checkpoint.  It is assembled at PPO training time
from the existing Block 1 artifacts:

- v3 rollout dynamics: `outputs/surrogate_v2/rc_node_v3_tsupply.pt`
- calibrated v3.5 reference: `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/`
- hybrid loss terms: `lambda_temp_disagree` and `lambda_power_disagree`

Therefore **do not run hybrid commands immediately after Block 1**. The next
executable section after Block 1 is Section 4, the pure v3 controller baseline.
Hybridization starts only in Section 5, after the direct-v3.5 negative control
in Section 4.5 has documented why direct use of the calibrated twin is not a
sufficient RL-training backend.

### 3.2. Results I provenance map (reviewer-facing data and figures)

The Results I / Block 1 manuscript section lives in
`docs/results1_digital_twin_overleaf/` and is regenerated entirely from the
versioned artifacts below by one command:

```bash
python3 -B docs/results1_digital_twin_overleaf/build_results1_overleaf.py
```

Reviewers do not need the manuscript build to inspect the evidence: every
figure, table, and inline number in Results I is read directly from `reports/`
and `outputs/`. The provenance map is:

```text
Results I content                                  -> source artifact
--------------------------------------------------    -----------------------------------------------------------
Data corpora table                                 -> reports/hou_evins_sample_generation_table.csv
v3 architecture / training hyperparameters         -> reports/hou_evins_training_hyperparams_table.csv
Feature scaling + physical-constraint table        -> reports/hou_evins_scaling_table.csv
v3 learning curve + best-epoch / val R^2           -> outputs/surrogate_v2/train_history_v2.csv
Multi-horizon rollout RMSE and 24 h R^2 (S11)      -> reports/hou_evins_predictive_validity_table.csv
Stage A alignment, Stage B excitation, C_zon       -> outputs/surrogate_v35_inverse_boptest_15min_episodeaware/calibration_summary_boptest_v35.json
Stage B C_zon trajectory + stability band          -> outputs/surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv
Canonical Stage C power head (power MAE 482 W)     -> outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json
Temperature rollout: 24 h RMSE, P95, residuals,    -> outputs/surrogate_v35_rollout_prepared_15min_power_head_only/calibrated_v35/
  persistence baseline                                  (all_full_rollouts.csv, horizon_metrics.csv, window_errors.csv, episode_summary.csv)
Power-channel ASHRAE-G14 metrics (canonical)       -> outputs/surrogate_v35_rollout_prepared_15min_power_head_only/calibrated_v35/all_full_rollouts.csv
v3 rollout reference                               -> outputs/surrogate_v3_rollout_prepared_15min/v3/
Matched-corpus four-variant + decomposition        -> reports/block1_corpus_matched_comparison.csv (+ .json)
Runtime throughput (steps/s, median/P95 ms)        -> reports/speed_benchmark_table.csv
Figures rie_fig01..08 (PDF + PNG)                  -> docs/results1_digital_twin_overleaf/figures/  (generated)
```

To rebuild the artifacts themselves (not just locate them), run the Block 1
stages through `evaluation/run_block1.py`; each command writes the files
referenced in the map above:

```bash
# v3 corpus + training            -> outputs/surrogate_v2/
python3 -B evaluation/run_block1.py collect-data
python3 -B evaluation/run_block1.py v3-train

# v3.5 Stage A/B/C, both passes    -> outputs/surrogate_v35_inverse_boptest_15min_{episodeaware,power_head_only}/
python3 -B evaluation/run_block1.py prepare-15min
python3 -B evaluation/run_block1.py v35-calibrate --preset canonical

# corpus-matched v3 + attribution  -> reports/block1_corpus_matched_comparison.{csv,json}
python3 -B evaluation/run_block1.py v3-train-15min
python3 -B evaluation/run_block1.py build-corpus-matched-report

# prepared rollouts (v3, raw/cal v3.5, matched v3) -> outputs/surrogate_*_rollout_prepared_*/
python3 -B evaluation/run_block1.py validate-rollouts --variant matched

# Hou-and-Evins tables + speed benchmark -> reports/hou_evins_*.csv, reports/speed_benchmark_table.csv
python3 -B evaluation/run_block1.py build-reports
python3 -B evaluation/run_block1.py speed-benchmark

# ...or the whole Block 1 pipeline end-to-end:
python3 -B evaluation/run_block1.py all
```

After the artifacts exist, regenerate the manuscript section, figures, and
tables with:

```bash
python3 -B docs/results1_digital_twin_overleaf/build_results1_overleaf.py
```

Power-channel caveat for reviewers: the canonical calibrated v3.5 power head is
the `power_head_only` (second-pass) checkpoint, used for all reported power
metrics (MAE 482 W, CV(RMSE) ~69%, NMBE ~-12%). The
`surrogate_v35_rollout_prepared_15min_episodeaware/` directory holds the
intermediate first-pass head and must not be used for power-channel numbers; the
temperature head is identical between the two passes.

The same data-driven section pattern applies to the other blocks: Results II
(`docs/results2_control_overleaf/`, Block 2) and Results III
(`docs/results3_transferability_overleaf/`, Block 3) are generated from their own
`reports/` and `outputs/` artifacts by analogous builders. The Block 2 / Results
II provenance map is in Section 11.1; Results I is the reference implementation
(`docs/results1_digital_twin_overleaf/build_results1_overleaf.py`).

## 4. Block 2: First Control Baseline -- Pure v3 Thermostatic PPO

Block 2 commands are routed through `evaluation/run_block2.py`. The wrapper
does not change the underlying scripts; it only fixes the canonical paths,
feature modes, artifact roots, and sweep values so the roadmap stays short.
To inspect any command without running compute, add `--dry-run` before the
subcommand, for example:

```bash
python3 -B evaluation/run_block2.py --dry-run thermostatic-train --variant pure
```

Why this section comes first:

- Block 1 showed that v3 is the fast control-oriented rollout surrogate.
- Before testing whether v3.5 helps or hurts control, we need the baseline PPO
  controller trained only on v3.
- This section does **not** use the hybrid backend and does **not** use v3.5.
  It establishes the reference controller for the later negative-control and
  hybrid comparisons.

Train pure v3 thermostatic PPO:

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant pure
```

Benchmark the pure v3 controller in BOPTEST:

```bash
python3 -B evaluation/run_block2.py thermostatic-benchmark --variant pure
```

Expected summary:

```bash
cat outputs/bestest_air_article7_style_15min/summary.csv
```

## 4.5. Block 2 Negative Control: Direct v3.5 Warm-Start

This is an explicit negative experiment. It tests whether a policy pretrained
directly on calibrated v3.5 becomes better after BOPTEST fine-tuning than a
scratch BOPTEST fine-tune. The frozen result says no: direct v3.5 warm-start is
worse than scratch.

This section must be completed **before** the hybrid sweep in Section 5. The
hybrid backend is not an arbitrary extra model; it is the response to this
negative result. The logic is:

```text
v3 works as a control-oriented rollout baseline
calibrated v3.5 improves predictive fidelity
direct v3.5 warm-start hurts PPO utility
therefore use v3 for rollout and v3.5 only as a frozen disagreement regularizer
```

Dependency:

- Reads the calibrated v3.5 checkpoint from Section 2
  (`outputs/surrogate_v35_inverse_boptest_15min_power_head_only/`).
- The launcher performs two PPO training runs internally -- scratch (no
  warm-start) and warm-start-from-v3.5 -- and benchmarks both on the same
  BOPTEST scenario window so the result is self-contained within this section.
- The v3.5-pretrained model created inside this section is used only as an
  initialization checkpoint for the subsequent BOPTEST fine-tune. It is **not**
  the standalone direct-v3.5 zero-shot controller used in Section 5.5 transfer
  diagnostics.

```bash
python3 -B evaluation/run_block2.py warmstart
```

Expected outputs:

```bash
ls outputs/block2_thermostatic_warmstart_utility
cat outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv
```

Article interpretation:

- This is the failure mode that motivates hybridization.
- The calibrated physical twin is useful as a regularizer, not as a standalone
  replacement for the smoother control-oriented v3 environment.
- Do not confuse this section's warm-start checkpoint with Section 5.5's
  `v35_direct` diagnostic checkpoint:

```text
Section 4.5 warm-start model:
  calibrated v3.5 pretraining -> BOPTEST fine-tune -> compare with scratch

Section 5.5 v35_direct model:
  calibrated v3.5 training only -> zero-shot/live-transfer diagnostics
```

## 4.6. Block 2: Corpus-Matched v3 Closed-Loop Control Utility (Reviewer Mitigation)

Why this section exists:

- Section 4 trains the pure-v3 controller on a v3 surrogate that was **fit** at a
  1-hour step but **used** at the 15-minute control step.
- Section 2.5 already retrained v3 on the matched 15-minute corpus and validated it
  as a *predictor* (24 h rollout RMSE ~0.876 C), but did **not** run it in closed
  loop. A reviewer can therefore argue that v3's *training utility* may come partly
  from the train/control timestep mismatch rather than from its black-box nature
  (Threats to validity, manuscript 8.5(iii)).
- This section closes that loop: it trains a PPO controller on the matched 15-min v3
  with the **identical** recipe as the canonical pure-v3 baseline (only the surrogate
  checkpoint changes) and benchmarks it on the same two targeted 14-day windows.

Dependencies:

- Reads the matched-resolution checkpoint from Section 2.5
  (`outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt`).
- Reads the canonical pure-v3 benchmark from Section 4
  (`outputs/bestest_air_article7_style_15min/summary.json`) as the hourly reference row.
- Requires the BOPTEST RTE for the benchmark step (as in Section 4).
- Does **not** modify any canonical Block 2/3 artifact; the downstream PPO families
  still use the original hourly v3.

Smoke-check the matched checkpoint loads and the env steps (≈2-3 min):

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant pure_v3_15min --smoke
```

Train the matched-v3 controller (full 10M steps; pure-surrogate, no BOPTEST needed):

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant pure_v3_15min
```

Benchmark it in BOPTEST on the two targeted windows:

```bash
python3 -B evaluation/run_block2.py thermostatic-benchmark --variant pure_v3_15min
```

Build the hourly-vs-matched comparison (verdict + ready-to-quote sentence for 8.5):

```bash
python3 -B evaluation/run_block2.py v3-15min-report
```

Expected artifacts:

```bash
cat outputs/bestest_air_pure_v3_15min/summary.csv
cat reports/block2_v3_15min_closed_loop_comparison.json
```

Reviewer-defensible reading (computed automatically by the report's `verdict`):

- `confound_rejected` — matched-v3 trains a usable controller with a comparable
  maintenance score, so v3's utility is its black-box smoothness, not the timestep
  mismatch; manuscript 8.5(iii) is promoted from open limitation to resolved control.
- `partial_confound` / `timestep_driven` — temporal coarse-graining contributes to (or
  drives) the utility; the §8.1/§8.5 framing is refined accordingly.

The step-by-step commands are listed above; the outcome table and verdict are
written to `reports/block2_v3_15min_closed_loop_comparison.{csv,json}` by the
`v3-15min-report` step.

**Executed result (verdict: `timestep_driven`).** The matched-15min v3 is a
strictly more accurate predictor (val rollout RMSE ~0.31 °C short-horizon;
0.876 °C at 24 h vs 1.557 °C for the hourly v3) yet, trained as an RL environment
with the identical pure-v3 recipe (10M steps, 32 envs, seed 42), the controller
**collapses** on live BOPTEST: `m_s = 1.142 / 1.211` with `85.6% / 91.4%` comfort
violation on the peak/typical windows (`reports/block2_v3_15min_closed_loop_comparison.{csv,json}`,
`outputs/bestest_air_pure_v3_15min/`), on a par with the direct-v3.5 failure and far
worse than the hourly-trained controller (`0.073 / 0.095`). So v3's training utility
comes from its **coarser temporal resolution**, not its black-box architecture. This
is folded into the manuscript as the Results II *Temporal-coarse-graining ablation*
(Table `tab:coarse_graining`) and the resolved §8.5(iii) / §8.1 framing; it is a
reviewer-mitigation control and does **not** alter any canonical downstream artifact
(the PPO families still use the hourly v3).

## 5. Block 2: Thermostatic Hybrid Sweep

Run this section only after Sections 4 and 4.5. It tests the engineering fix
for the direct-v3.5 negative result: keep v3 as the rollout dynamics and use
calibrated v3.5 only as a soft physical censor in the reward.

The canonical thermostatic hybrid backend is:

- `surrogate-kind=hybrid_v3_v35`
- `lambda_temp_disagree=0.10`
- `lambda_power_disagree=5e-5`
- 15-min step, comfort band 21-24 C

Train the sweep points:

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant hybrid_sweep
```

Benchmark each sweep point:

```bash
python3 -B evaluation/run_block2.py thermostatic-benchmark --variant hybrid_sweep
```

Current canonical result:

- `hybrid_l010` is the thermostatic canonical point.
- It keeps the energy advantage while avoiding the stronger comfort degradation
  seen at lower/higher settings.

## 5.5. Block 1.3 / Block 2 Transfer Diagnostics

This section produces the empirical evidence for the fidelity-to-RL gap
documented in paper §5.4. The final aggregation table
`reports/hybrid_transfer_comparison.csv` compares THREE controllers side by
side; one is trained inside this section, two come from earlier sections:

1. pure v3 thermostatic       (trained in Section 4)
2. hybrid_l010 thermostatic   (trained in Section 5)
3. direct v3.5 thermostatic   (trained as Step A below; this is the failure
                               control whose role is to demonstrate why
                               direct calibrated v3.5 is not a usable RL
                               training environment)

### Step A. Train the direct-v3.5 thermostatic policy (failure control)

This is the policy that will fail to transfer; its existence is the central
evidence for the fidelity-to-RL gap. The observation interface intentionally
uses `comfort_centered` t_zone encoding (not the `raw` encoding of canonical
hybrid_l010) because that is the configuration in which the failure was
originally documented.

This command intentionally trains a separate artifact from the Section 4.5
warm-start checkpoint. Section 4.5 asks whether v3.5 pretraining helps after
BOPTEST fine-tuning; this Step A asks whether a standalone v3.5-trained policy
has usable live-transfer behavior before any BOPTEST fine-tune. Both consume
the same calibrated v3.5 surrogate, but they answer different experimental
questions.

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant v35_direct
```

### Step B. Validate live BOPTEST closed-loop transfer for all three controllers

```bash
python3 -B evaluation/run_block2.py thermostatic-transfer --variant all
```

### Step C. Compute first_divergence_step and action_gap_norm for all three

```bash
python3 -B evaluation/run_block2.py thermostatic-diagnose --variant all
```

### Step D. Aggregate into the article-facing evidence table

```bash
python3 -B evaluation/run_block2.py build-hybrid-evidence
```

Expected outputs:

```bash
cat reports/hybrid_transfer_comparison.csv
ls reports/figures/hybrid_transfer_gap_comparison.png
```

Current article interpretation:

- Direct v3.5 zero-shot transfer diverges immediately.
- Hybrid_l010 reduces the transfer gap relative to direct v3.5 and is
  especially better in the typical heat window.
- This is transfer evidence within the bestest_air testcase, NOT a claim of
  universal building transfer (that is Block 3).

## 6. Block 2: HDRL Sweep

HDRL uses the same 15-min step and 21-24 C comfort band. The sweep showed that
temperature disagreement regularization hurts the hierarchical controller, so the
best HDRL setting is `lambda_temp_disagree=0.00`.

Train HDRL sweep points:

```bash
python3 -B evaluation/run_block2.py hdrl-train --variant sweep
```

Benchmark HDRL models by pointing `--controllers hdrl` to the winter/summer model
pair produced by the run. Preserve the output folders:

```bash
python3 -B evaluation/run_block2.py hdrl-benchmark --variant sweep
```

Expected output folders:

```bash
ls outputs/block2_hdrl_hybrid_v3_v35_l000
ls outputs/block2_hdrl_hybrid_v3_v35_l003
ls outputs/block2_hdrl_hybrid_v3_v35_l005
ls outputs/block2_hdrl_hybrid_v3_v35_l010
```

Current conclusion:

- `lambda_temp=0.00` is best for HDRL.
- This is a negative result about controller-family specificity, not a failed
  surrogate result.

## 6.5. MORL 5D Observation Failure  it is observation-interface negative control

This is an observation-interface negative control. The canonical Block 2 MORL
path uses the 17D TSup-style observation interface
(`configs/morl_surrogate_ppo/env.yaml` has `obs_mode: extended`).



```text
configs/morl_surrogate_ppo_5d/env.yaml
obs_mode: basic
```

This constructed preset uses the same current backend and reward machinery as
the 17D path, but forces the 5D basic TSup observation vector. 

Run the constructed 5D legacy preset for the practical comparison point
(`w=(0.75,0.25,0.00)`, seed 42):

```bash
python3 -B evaluation/run_block2.py morl-5d --point comfort_075_energy_025 --seed 42
```

Expected constructed output root:

```bash
ls outputs/morl_5d_legacy_rerun
```

Build the current-code constructed 5D versus 17D comparison table:

```bash
python3 -B evaluation/run_block2.py build-morl-5d-comparison
cat reports/block2_morl_5d_reconstructed_comparison.csv
```

Current constructed result (`w=(0.75,0.25,0.00)`, seed 42):

- constructed 5D: `RMSE_T=2.721 C`, `violation=37.77%`, `m_s=0.680`,
  `energy=195.93 kWh`.
- 17D reference: `RMSE_T=0.72 C`, `violation=4.9%`, `m_s=0.099`.

Scientific interpretation:

```text
The constructed 5D run confirms that the basic 5D observation interface
remains substantially weaker than the 17D interface under the current codebase,
The constructed 5D run is therefore reported as the
current-code reproducibility evidence for the MORL observation-interface
ablation.
```

Paper placement:

- Main paper: use constructed 5D as reproducible evidence for the
  observation-interface ablation..
- Do not use constructed 5D results to rewrite the pre-registered 17D
  canonical seed analysis; the N=5 17D audit chain remains unchanged.

Optional full constructed 5D preference sweep (supplement only, not required
for the main-paper claim):

```bash
python3 -B evaluation/run_block2.py morl-5d --point all --seed 42
```

## 7. Block 2: MORL 17D Power-Only Backend

MORL17D also uses:

- hybrid backend
- 17D observation interface
- `lambda_temp_disagree=0.00`
- `lambda_power_disagree=5e-5`
- 15-min step
- comfort band 21-24 C

Important scope note:

- Both the reconstructed 5D run in Section 6.5 and the canonical 17D runs in
  Sections 8-9 use the **hybrid surrogate backend** (`hybrid_v3_v35`).
- In MORL, "power-only" means the calibrated v3.5 reference is used only through
  the power-disagreement term:

```text
lambda_temp_disagree  = 0.00
lambda_power_disagree = 5e-5
```

- There is no separate MORL `lambda_temp_disagree` sweep in the audit trail.
  The temperature-disagreement sweep was run for thermostatic PPO (Section 5)
  and HDRL (Section 6). MORL was then evaluated under the fixed power-only
  hybrid setting to avoid reopening another controller-family-specific
  hyperparameter search.
- Therefore, the MORL claim is **not** "lambda_temp=0.00 and
  lambda_power=5e-5 are globally optimal for MORL"; the claim is narrower:
  under the fixed power-only hybrid backend, the 17D observation interface
  substantially outperforms the reconstructed 5D interface and supports the
  Pareto/canonical analysis.

## 8. MORL Pareto Sweep

Purpose:

- This section maps the single-seed MORL comfort-energy Pareto landscape under
  the fixed 17D power-only hybrid backend defined in Section 7.
- It runs five pre-defined weight points with seed 42 only. This is the
  structured search stage, not the statistical robustness stage.
- The result of this section is used to select two canonical operating points:
  the neutral canonical `w=(0.50,0.50,0.00)` and the practical-deployment
  canonical `w=(0.75,0.25,0.00)`.

Run the five single-seed Pareto points:

```bash
python3 -B evaluation/run_block2.py morl-17d --point all --seed 42
```

## 9. MORL Canonical Seed Analysis

Purpose:

- This section takes the two canonical points selected from the Section 8
  Pareto sweep and reruns each of them on N=5 seeds.
- It is the robustness/audit stage: it estimates seed variance, supports
  Pareto error bars, and checks that the canonical points are not single-seed
  artifacts.

The pre-registered neutral canonical is `w=(0.50,0.50,0.00)`.
The practical deployment canonical is `w=(0.75,0.25,0.00)`.

Why a separate `_seedfix` artifact root:

- Section 8 produces single-seed Pareto evidence (five points x seed 42 only)
  in `outputs/morl_pareto_hybrid_power_only/<tag>`.
- Section 9 extends two of those Pareto points to N=5 seeds with strengthened
  seed propagation (torch, numpy, python random, env, action_space, and
  observation_space all explicitly seeded; the replay test in this section's
  notes confirmed bit-identical BOPTEST determinism for a fixed checkpoint).
- The `_seedfix` suffix in the artifact root marks runs that used the
  strengthened seed-propagation code path. Both directories are preserved
  because the seed-42 result under the strengthened propagation may differ
  numerically from the seed-42 result in the original Pareto sweep due to the
  seed-fix change itself.

Run the N=5 neutral canonical:

```bash
python3 -B evaluation/run_block2.py morl-canonical --canonical neutral --seeds 42,43,44,45,46
```

Run the N=5 practical canonical:

```bash
python3 -B evaluation/run_block2.py morl-canonical --canonical practical --seeds 42,43,44,45,46
```

Important BOPTEST note:

- Surrogate pretraining can be parallelized.
- BOPTEST fine-tune and yearly validation should be sequential because the RTE
  testcase lifecycle is fragile under parallel HTTP sessions.

Current N=5 findings:

- Neutral canonical: `m_s=0.187 +/- 0.078`, `sigma/mean=0.418`.
- Practical canonical: `m_s=0.139 +/- 0.085`, `sigma/mean=0.613`.
- BOPTEST replay test is bit-identical for a fixed checkpoint.
- The action-saturation/seasonal-inversion hypothesis was falsified at N=5.
- MORL remains promising but not deployment-stable without future stabilization.

## 10. PI Baseline

This section is independent of all other Block 2 controller runs but is
positioned here for three reasons:

- The article-facing comparison in Section 11 normalises RL controller
  performance against this PI yearly baseline; running it now leaves Section
  11 with no missing dependencies.
- PI yearly evaluation takes roughly four hours on the RTE container; placing
  it after Section 9 lets it queue behind the heavier MORL seed compute
  without blocking earlier sections.
- PI is deterministic; no seed analysis is needed.

Run the reproducible BOPTEST built-in PI yearly baseline:

```bash
python3 -B evaluation/run_block2.py pi-yearly
```

Current PI framing:

- This is the default BOPTEST reference, not a custom-tuned strong PI.
- Current yearly result: `m_s=0.910`, `violation=63.59%`, `energy=104.07 kWh`,
  `RMSE_T=3.395 C`.

## 11. Rebuild Block 2 Tables and Figures

This section depends on Sections 4, 4.5, 5, 5.5, 6, 6.5, 8, 9, and 10 (see the
graph in Section 0.5).

```bash
python3 -B evaluation/run_block2.py build-reports
```

Expected outputs:

```bash
ls reports/morl_*canonical*.csv
ls reports/morl_pareto_front_table.csv
ls reports/hou_evins_*.csv
```

### 11.1. Results II provenance map (reviewer-facing data and figures)

Results II (Block 2 control results) lives in `docs/results2_control_overleaf/`.
Following the Results I pattern, it is regenerated from versioned artifacts by a
data-driven builder `build_results2_overleaf.py` (to be added, mirroring
`docs/results1_digital_twin_overleaf/build_results1_overleaf.py`). Reviewers can
inspect every Block 2 number directly from `reports/` and `outputs/`:

```text
Results II content                                   -> source artifact
---------------------------------------------------     ----------------------------------------------------------
Pure v3 thermostatic baseline KPIs                   -> outputs/bestest_air_article7_style_15min/summary.csv
Temporal-coarse-graining ablation (tab:coarse_graining) -> reports/block2_v3_15min_closed_loop_comparison.csv ; outputs/bestest_air_pure_v3_15min/summary.csv
Direct-v3.5 warm-start negative control              -> outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv
Thermostatic hybrid sweep (canonical hybrid_l010)    -> outputs/block2_thermostatic_*hybrid*/  (benchmark summaries)
Architecture justification on live BOPTEST (S9)      -> reports/hou_evins_architecture_justification_table.csv
Fidelity-to-RL transfer-gap diagnostics              -> reports/hybrid_transfer_comparison.csv (+ reports/figures/hybrid_transfer_gap_comparison.png)
HDRL sweep (best lambda_temp = 0.00)                 -> outputs/block2_hdrl_hybrid_v3_v35_l0{00,03,05,10}/
MORL 5D vs 17D observation-interface ablation        -> reports/block2_morl_5d_reconstructed_comparison.csv
MORL single-seed Pareto sweep                        -> outputs/morl_pareto_hybrid_power_only/<tag>/ ; reports/morl_pareto_front_table.csv
MORL N=5 canonical seed variance                     -> reports/morl_*canonical*.csv
PI yearly baseline                                   -> evaluation/run_block2.py pi-yearly  (m_s, violation, energy, RMSE_T)
Block 2 tables/figures rebuild                       -> evaluation/run_block2.py build-reports  (Section 11)
m_s metric definition (r_time + r_sev)               -> evaluation/benchmark_bestest_air_article7_style.py (compute_safety_metrics)
Reward / action / comfort-band parameters            -> configs/env.yaml  (morl + comfort_shaping + action_wrappers)
17D observation feature groups                       -> envs/tsup_features.py  (obs_mode = extended)
Targeted + yearly scenario definitions               -> outputs/block2_*/scenario_manifest.json
Hybrid v3-vs-v3.5 disagreement statistics            -> reports/hybrid_disagreement_summary.csv  (overall row)
```

To rebuild the Block 2 artifacts, run the Section 4-11 commands through
`evaluation/run_block2.py` (thermostatic-train/benchmark, warmstart,
hdrl-train/benchmark, morl-5d, morl-17d, morl-canonical, pi-yearly,
build-reports), then regenerate the section with `build_results2_overleaf.py`.

Claim discipline for Results II must match Sections 9 and 13: the MORL result is
the narrowed claim (17D power-only hybrid is substantially stronger than the
reconstructed 5D interface and yields a useful Pareto structure, but N=5
canonical variance is too high for a deployment-stability claim), and the
fidelity-to-RL gap / hybrid role assignment is the Block 1->Block 2 dependency
established in Section 3.1.

## 12. Rebuild the Word Article Skeleton

```bash
python3 -B docs/build_hvac_paper_docx.py
```

Expected output:

```bash
ls -lh docs/hvac_paper_skeleton_q1_restructured.docx
```

This document now contains:

- Block 1 surrogate fidelity tables and figures.
- Block 2 thermostatic/HDRL/MORL results.
- Hou-and-Evins supplementary tables S1-S11.
- Post-N=5 MORL falsification result and seasonal variance diagnostic figure.

## 13. Audit Anchors

Purpose:

- Section 13 is not a compute section. It records the Block 2 audit backbone
  used to protect the MORL canonical analysis from cherry-picking and post-hoc
  reinterpretation.
- The audit chain separates what was promised before the final seed expansion
  from what was observed after N=5.
- The paper should cite this chain when discussing MORL seed variance,
  falsified seasonal/action-saturation hypotheses, and the narrowed MORL claim.

Do not rewrite these commits when preparing the paper:

- Pre-registration: `93df9b364657ac77bbe3642e4bc277d1eb8a8b60`
- Post-N=5 falsification: `62dc859d02f5f4a75fa4b55d8477c1d4e6206449`

> **Note for reviewers (public snapshot).** This public repository is a curated
> single-snapshot release, so the two anchor commits above are not reachable from
> `main`. They are published as annotated Git tags so the version-locked timeline
> remains independently verifiable:
>
> ```bash
> git fetch --tags
> git show audit-pre-registration   # -> 93df9b3, 2026-05-16 (predictions logged first)
> git show audit-post-n5            # -> 62dc859, 2026-05-17 (results appended after)
> ```
>
> The tag dates prove the pre-registration predates the N=5 results; the frozen
> plan content itself is also preserved verbatim in the machine-readable protocol
> file below. The full per-commit development history is retained in the authors'
> development repository and is available on request.

The corresponding machine-readable protocol file is:

```bash
cat configs/morl_canonical_selection_log.yaml
```

These commits preserve the timeline:

1. Predictions were logged before seeds 45/46.
2. Results were appended later.
3. The falsification was not retrofitted after the fact.

Concrete logic:

- Section 8 first maps the MORL Pareto landscape on seed 42.
- Before expanding the canonical points to additional seeds, commit
  `93df9b3` freezes the canonical plan: selected points, expectations,
  seed-extension logic, and PASS/FAIL interpretation rules.
- Section 9 then runs the N=5 canonical seed analysis for `w=(0.50,0.50,0.00)`
  and `w=(0.75,0.25,0.00)`.
- After the N=5 results are known, commit `62dc859` appends the observed seed
  variance and falsification outcome without rewriting the pre-registration.

Reviewer-facing protection:

- Weight selection: the canonical MORL points were not chosen after observing
  all seeds.
- Seed transparency: weak or high-variance seeds were not hidden.
- Negative-result integrity: the seasonal/action-saturation hypothesis and
  deployment-stability expectation were allowed to fail.
- Claim discipline: the paper claim is narrowed to "17D MORL under the fixed
  power-only hybrid backend is substantially stronger than reconstructed 5D and
  gives a useful Pareto structure, but N=5 canonical variance remains too high
  for deployment-stability claims."

## 13.5. Pre-Block-3 Cleanup Workflow

Before opening Block 3, keep cleanup commits separate from article/result
commits. Do not use `git add -A`.

Recommended sequence:

```powershell
git status --short
git diff -- docs\build_hvac_paper_docx.py roadmap.md

# Article/roadmap commit only
git add docs\build_hvac_paper_docx.py docs\hvac_paper_skeleton_q1_restructured.docx roadmap.md
git diff --cached --stat
git commit -m "article roadmap: freeze Block 1, Block 2, and Block 3 reproduction path"

# Cleanup/archive commit only, if staged cleanup moves are still pending
git status --short
git commit -m "cleanup: archive legacy docs, logs, results, literature, and planning notes"
```

Rules:

- Keep `.claude/worktrees/` out of git.
- Keep Word lock files (`~$*.docx`) out of git.
- Do not mix cleanup moves with scientific result updates.
- Do not rewrite commits `93df9b3` or `62dc859`.

## 14. Block 3: Transferability Pre-Registration

Block 3 opened only after the Block 1/2 article state was committed. The first
Block 3 artifact is the pre-registration manifest:

```bash
configs/block3_testcase_manifest.yaml
```

The manifest pre-registers, BEFORE any Block 3 BOPTEST run:

- three testcase candidates (`bestest_hydronic_heat_pump`, `bestest_hydronic`,
  `singlezone_commercial_hydronic`)
- three recalibration regimes (`none`, `partial` Stage C only, `full` Stage A/B/C)
- four pre-registered hypotheses (H1_strong, H2_medium, H3_weak surrogate side,
  H3_weak controller side, plus hierarchy_consistency)
- a single passfail threshold: `m_s_RL <= 1.25 * m_s_PI` evaluated yearly
- a bounded extension policy (N=3 max per cell; no N=5 cascade within Block 3)
- an early-termination clause if H3_weak is falsified on the easiest testcase
- a self-referential audit anchor (commit SHA of the first manifest commit)

Audit anchor for Block 3 pre-registration: the commit SHA that introduced
`configs/block3_testcase_manifest.yaml` for the first time. This SHA is logged
inside the manifest's `audit` block.

## 15. Block 3 Execution: Transferability on Hydronic Family

Block 3 executed three testcases under the pre-registered three-regime protocol.
Each cell is an append-only `cell_results` entry in the manifest.

### 15.1. Adapter for hydronic actuator interface

`bestest_air` exposes a direct supply-temperature command. The three hydronic
testcases expose `oveTSet_u`, `oveHeaPumY_u`, `ovePum_u`, `oveFan_u` (heat-pump
case) or analogous boiler/pump variants. A documented adapter converts the
frozen direct-TSup policy output to the hydronic setpoint + heat pump / pump /
fan override commands. Block 3 transfer is therefore explicitly
adapter-mediated, not literal direct-TSup transfer.

The adapter code is centralized in `evaluation/block3_testcase_adapters.py`.
The testcase-specific mappings are recorded in:

- `configs/block3_actuator_mapping_bestest_hydronic_heat_pump.yaml`
- `configs/block3_actuator_mapping_bestest_hydronic.yaml`
- `configs/block3_actuator_mapping_singlezone_commercial_hydronic.yaml`

A smoke test that verifies low-vs-high override actually moves heat output is
the first command per testcase below.

### 15.2. Per-testcase command pattern

Unlike Sections 4-11, which call `evaluation/run_block2.py` as a single
short-command wrapper, Block 3 commands are parameterized by `--testcase`
and call individual testcase-aware scripts directly. A wrapper would add
no abstraction at this level because each script is already its own short
command, parameterized identically across the three testcases.

For each testcase in `{bestest_hydronic_heat_pump, bestest_hydronic, singlezone_commercial_hydronic}`:

```bash
# Optional: choose a testcase once per shell session
export TESTCASE=bestest_hydronic

# 1. Adapter smoke test: confirms low/high override maps to physical response
python3 -B evaluation/smoke_block3_hydronic_adapter.py --testcase ${TESTCASE}
cat reports/block3_${TESTCASE}_adapter_smoke_summary.csv

# 2. PI baseline: testcase-specific reference for the normaliser
python3 -B evaluation/yearly_validation_universal_adapter.py \
  --testcase ${TESTCASE} \
  --controller pi \
  --skip-existing
cat outputs/universal_validation/${TESTCASE}/pi_block3_hydronic/pi_universal_yearly_summary.csv

# 3. mode=none: frozen thermostatic controller through documented adapter
python3 -B evaluation/yearly_validation_universal_adapter.py \
  --testcase ${TESTCASE} \
  --controller thermostatic \
  --skip-existing
cat outputs/universal_validation/${TESTCASE}/thermostatic_block3_hydronic/thermostatic_universal_yearly_summary.csv

# 4. Target telemetry for Stage C/full recalibration
python3 -B evaluation/collect_block3_hydronic_adapter_telemetry.py \
  --testcase ${TESTCASE}
ls -lh data/block3_${TESTCASE}/hydronic_adapter_stage_c_15min.csv
cat data/block3_${TESTCASE}/hydronic_adapter_stage_c_15min.manifest.json

# 5. mode=partial: Stage C heads-only recalibration, frozen bestest_air C_zon
python3 -B evaluation/run_block3_surrogate_recalibration.py \
  --testcase ${TESTCASE} \
  --regime partial
cat outputs/block3_${TESTCASE}/surrogate_v35_partial_stage_c_allrows_heads/calibration_summary_boptest_v35.json

# 6. mode=full: complete Stage A/B/C on target telemetry
python3 -B evaluation/run_block3_surrogate_recalibration.py \
  --testcase ${TESTCASE} \
  --regime full
cat outputs/block3_${TESTCASE}/surrogate_v35_full_stage_abc_allrows_heads/calibration_summary_boptest_v35.json
```

The short commands above use testcase-aware defaults:

- BOPTEST URL: `http://web:8000`
- control step: `900 s`
- scenario length: `14 days`
- hydronic adapter config: selected from `configs/block3_actuator_mapping_*.yaml`
- yearly output root: `outputs/universal_validation/<testcase>/<controller>_<preset>/`
- telemetry output: `data/block3_<testcase>/hydronic_adapter_stage_c_15min.csv`
- recalibration output: `outputs/block3_<testcase>/surrogate_v35_<regime>...`

Manual overrides remain available through `--boptest-url`, `--adapter-config`,
`--output-dir`, `--output-csv`, `--data`, and `--model`, but the paper-facing
Block 3 reproduction path should use the short commands unless debugging a
specific cell.

### 15.3. Per-testcase results

`bestest_hydronic_heat_pump`:

- mode=none controller: `m_s_RL=0.665`, `m_s_PI=0.464`, threshold `0.579` →
  FAIL (energy `-7.3%` vs PI, comfort violation dominates)
- mode=partial surrogate: RMSE_T `0.977 -> 0.781 C` (-20.1%), Power MAE
  `2362 -> 1768 W` (-25.1%), C_zon frozen at `4.413e+05 J/K`, Stage B epochs
  ran = 0
- mode=full surrogate: RMSE_T `1.421 -> 0.565 C` (-60.2%), Power MAE
  `2921 -> 1767 W` (-39.5%), C_zon re-identified at `8.347e+05 J/K`
  (`1.89x` bestest_air canonical), Stage B epochs ran = 120
- mode=partial and mode=full controller verdicts: FAIL by structural
  definition (controller frozen + adapter unchanged → live KPI identical to
  mode=none)

`bestest_hydronic`:

- mode=none controller: `m_s_RL=0.976`, `m_s_PI=0.7502`, threshold `0.9377` →
  FAIL (energy `-5.84%` vs PI, comfort violation dominates)
- mode=full surrogate: RMSE_T `2.666 -> 0.335 C` (-87.4%), Power MAE
  `784 -> 85 W` (-89.2%), C_zon re-identified at `8.622e+05 J/K`
  (`1.95x` bestest_air canonical)

`singlezone_commercial_hydronic`:

- mode=none controller: `m_s_RL=0.431`, `m_s_PI=0.628`, threshold `0.785` →
  THRESHOLD PASS (energy `+35.3%` vs PI, comfort within tolerance but energy
  penalty substantial)
- mode=full surrogate: RMSE_T `1.952 -> 0.238 C` (-87.8%), C_zon re-identified
  at `8.425e+05 J/K` (`1.91x` bestest_air canonical)

### 15.4. Aggregate findings (N=3 hydronic family)

```bash
cat reports/block3_transfer_matrix.csv
```

Surrogate component:

- Transferable on N=3 under full Stage A/B/C
- RMSE_T improvement: 60.2% / 87.4% / 87.8%
- C_zon ratio vs bestest_air: 1.89x / 1.95x / 1.91x (mean 1.91, spread 3.2%)

Controller component:

- Not deployment-ready on N=3 under frozen-controller scope
- Failure mode is regime-dependent:
  - fast dynamics (residential heat pump, residential boiler) -> comfort
    violation failure (m_s_RL > threshold)
  - slow dynamics (commercial hydronic) -> threshold-PASS but energy-inflation
    failure (`+35.3%` vs PI)

Pre-registered hypothesis status at Block 3 closure:

- H1_strong: FALSIFIED (deployment-PASS not achieved on `>=2` of 3 testcases;
  commercial threshold-PASS does not qualify as deployment-ready)
- H2_medium: FALSIFIED by structural definition (partial regime keeps
  controller frozen)
- H3_weak surrogate side: SUPPORTED on N=3
- H3_weak controller side: FALSIFIED with regime-dependent failure modes
- hierarchy_consistency: SUPPORTED (verdict monotone in recalibration depth)

### 15.5. Block 3 closure artefacts

```bash
ls reports/block3_*_transfer_summary.csv
cat reports/block3_transfer_matrix.csv
cat configs/block3_testcase_manifest.yaml | yq '.aggregated_results'
```

Audit anchor for Block 3 closure: the commit SHA that wrote
`aggregated_results.status: closed` into the manifest. Logged inside
`audit.block3_close_commit_sha`.

### 15.6. Block 3 frozen copy bundle

The Block 3 code/data/results snapshot is also packaged as a copy-only bundle:

```bash
ls block3_transferability_bundle
cat block3_transferability_bundle/README.md
cat block3_transferability_bundle/manifests/bundle_manifest.json
```

The bundle mirrors the Block 1/2 organization:

- `block3_transferability_bundle/code/` — adapter, yearly validation, telemetry, and recalibration scripts
- `block3_transferability_bundle/configs/` — pre-registration manifest and adapter YAMLs
- `block3_transferability_bundle/data/` — target hydronic telemetry CSVs
- `block3_transferability_bundle/outputs/` — yearly validation and surrogate recalibration outputs
- `block3_transferability_bundle/reports/` — transfer summaries and transfer matrix
- `block3_transferability_bundle/models/` — frozen thermostatic checkpoints

This is a non-destructive copy bundle. Source paths remain active in
`evaluation/`, `configs/`, `data/`, `outputs/`, and `reports/`.

### 15.7. Results III provenance map (reviewer-facing data and figures)

Results III (Block 3 transferability) lives in
`docs/results3_transferability_overleaf/` and is regenerated from the artifacts
below by the data-driven builder `build_results3_overleaf.py` (mirroring the
Results I/II builders). Numeric values are read from `reports/`; pre-registered
hypotheses, predictions, and audit anchors are verified literals from the
manifest and the git audit chain.

```text
Results III content                                   -> source artifact
----------------------------------------------------     ----------------------------------------------------------
Headline transfer matrix (m_s, threshold, RMSE, C_zon) -> reports/block3_transfer_matrix.csv
Primary per-regime detail (none/partial/full)          -> reports/block3_bestest_hydronic_heat_pump_transfer_summary.csv
Secondary / stretch per-regime detail                  -> reports/block3_bestest_hydronic_transfer_summary.csv ; reports/block3_singlezone_commercial_hydronic_transfer_summary.csv
N=2 hydronic-family roll-up                            -> reports/block3_hydronic_family_n2_summary.csv
C_zon ratios + hydronic-family mean/std                -> reports/block3_transfer_matrix.csv (c_zon_ratio_vs_bestest_air); baseline 4.413e5 J/K (Block 1)
Testcases / adapters / regimes / hypotheses / predictions -> configs/block3_testcase_manifest.yaml ; configs/block3_actuator_mapping_*.yaml
Audit anchors                                          -> git log: 1861e48, 2f9d596, eb7091e, 46fbaa9, 645626e, b915bfc (close), 7ada793 (record close SHA), cb7025f
Figures (protocol, heatmap, RMSE gain, C_zon, closure) -> docs/results3_transferability_overleaf/figures/  (Block 3 evaluation scripts)
```

To rebuild the artifacts themselves, run the per-testcase Section 15.2 commands
(`smoke_block3_hydronic_adapter.py`, `yearly_validation_universal_adapter.py`,
`collect_block3_hydronic_adapter_telemetry.py`,
`run_block3_surrogate_recalibration.py`) and then
`evaluation/build_block3_transfer_matrix.py`; finally regenerate the section with
`build_results3_overleaf.py`.

## 16. Audit Anchor Chain (Updated)

After Block 3 closure the full audit chain is:

- MORL pre-registration:           `93df9b364657ac77bbe3642e4bc277d1eb8a8b60`
- MORL post-N=5 falsification:     `62dc859d02f5f4a75fa4b55d8477c1d4e6206449`
- Block 3 pre-registration:        `<SHA from manifest audit.pre_registration_commit_sha>`
- Block 3 closure:                 `<SHA from manifest audit.block3_close_commit_sha>`

All four commits must remain untouched when preparing the paper. The
machine-readable cross-references are in `configs/morl_canonical_selection_log.yaml`
and `configs/block3_testcase_manifest.yaml`.

> **Public-snapshot verification.** In this curated public release the two MORL
> anchors are reachable as the Git tags `audit-pre-registration` (→ `93df9b3`) and
> `audit-post-n5` (→ `62dc859`); fetch them with `git fetch --tags` and inspect with
> `git show <tag>`. The Block 3 pre-registration and closure anchors are recorded
> inside `configs/block3_testcase_manifest.yaml` (`audit.*_commit_sha` fields). The
> complete development history is held in the authors' development repository and is
> available on request.

## 17. Paper Manuscript Build Path

Once Block 3 is closed:

```bash
# Compact evidence tables and intermediate report CSVs
python3 -B evaluation/build_hou_evins_q1_gap_tables.py
python3 -B evaluation/build_hybrid_evidence_closure.py
python3 -B evaluation/build_morl_pareto_table.py
python3 -B evaluation/build_morl_canonical_variance_diagnostics.py
python3 -B evaluation/build_morl_seasonal_variance_inversion.py
python3 -B evaluation/build_block3_transfer_matrix.py

# Rebuild the Word skeleton with Block 3 results, if regenerating from source
python3 -B docs/build_hvac_paper_docx.py

# Consolidate the canonical GitHub-facing paper package
python3 -B evaluation/organize_paper_artifacts.py
```

The canonical paper package after this step is:

```bash
ls paper_artifacts/figures/main
ls paper_artifacts/figures/supplementary
ls paper_artifacts/tables/main
ls paper_artifacts/csv/reports
cat paper_artifacts/manifests/main_figures_manifest.csv
cat paper_artifacts/manifests/paper_artifacts_inventory.csv
```

Expected canonical counts:

- `paper_artifacts/figures/main/`: 12 main figures, each as PNG and PDF
  (`24` files).
- `paper_artifacts/tables/main/`: 7 main-paper tables exported from
  `docs/hvac_paper_final_q1.docx`.
- `paper_artifacts/csv/reports/`: compact report-level CSV evidence used by
  the paper.
- `paper_artifacts/manifests/`: main-figure manifest and full artifact
  inventory.

`reports/figures/` is legacy/generated output and is ignored by Git. Old figure
variants are archived under `draft/legacy_archive/figure_variants_archive/`;
they are not the paper-facing artifact location.

Final paper checklist:

- All four audit anchor SHAs cited in Methods §4.5 reproducibility statement
- Section 7 of paper draws cell-by-cell from `cell_results` block of the
  Block 3 manifest
- Threats-to-Validity (§8) explicitly mentions the threshold-framework
  limitation revealed by the commercial-testcase result
- Cover letter highlights pre-registration discipline (three audit anchors)
  as the methodological differentiator
- which metrics define transferability success

Git staging for the paper package must remain explicit:

```bash
git add .gitignore evaluation/organize_paper_artifacts.py paper_artifacts
git add -u reports/figures/article_real
git status --short
```

Do not use `git add -A` for the paper-artifact cleanup because the repository
may also contain unrelated model-output deletions, Word lock files, or legacy
Sinergym archive changes.
