# Structured Plan Status Update

Date: 2026-04-14

## Central Q1 Axis

The paper should now be framed around one main claim:

**Calibrated digital twin fidelity is the foundation for reliable transfer and evaluation of downstream HVAC controllers, with MORL/PPO as the target controller family.**

This means:

- The surrogate and inverse problem are not separate side projects. They are the enabling layer.
- Thermostatic PPO and HDRL are supporting baselines and ablation anchors.
- MORL/PPO is the central destination, but it is not yet the strongest validated result.

## Current Verified Status

### 1. Surrogate Fidelity Block

Status: `implemented`

Evidence:

- `results/research_benchmark/tables/surrogate_rollout_benchmark.csv`
- `results/research_benchmark/tables/inverse_calibration_benchmark.csv`

Current measured numbers:

- `raw_v35`: one-step RMSE `0.6135 C`, 4h rollout RMSE `0.7627 C`, 8h rollout RMSE `0.7659 C`, 24h rollout RMSE `0.7613 C`, mean bias `+0.0087 C`, power RMSE `72.23 W`, `C_zon error = 3.07%`
- `calibrated_v35`: one-step RMSE `0.6161 C`, 4h rollout RMSE `0.7661 C`, 8h rollout RMSE `0.7695 C`, 24h rollout RMSE `0.7656 C`, mean bias `-0.0659 C`, power RMSE `72.23 W`, `C_zon error = 3.07%`
- `v35_heads_only` inverse calibration: baseline RMSE `0.5338 C` -> calibrated RMSE `0.5256 C`, improvement `1.52%`, `C_zon error = 3.07%`

Interpretation:

- The inverse problem is already successful in the **physical identification sense**.
- The current calibrated twin does **not** yet beat the raw twin on free-run rollout realism.
- Therefore the honest claim is: **Stage A/B/C solved structural parameter recovery better than rollout drift correction.**
- The later `rollout24 fixed-czon` experimental branch was reverted because it made live 24h rollout worse, so the canonical Block 1 baseline remains `v3.5 heads_only`.

### 2. Main Controller Benchmark

Status: `partial`

Evidence:

- `outputs/bestest_air_article7_style_15min/summary.csv`
- `results/research_benchmark/tables/controller_benchmark.csv`

Current measured 15 min heating-window results on `bestest_air`:

- `thermostatic`, `peak_heat_window`: `m_s = 0.0725`, `violation_pct = 1.49`, `rmse_22_c = 0.8691`, `mean_power_w = 958.9`
- `thermostatic`, `typical_heat_window`: `m_s = 0.0947`, `violation_pct = 4.39`, `rmse_22_c = 0.6222`, `mean_power_w = 1096.1`
- `hdrl`, `peak_heat_window`: `m_s = 0.1466`, `violation_pct = 7.22`, `rmse_22_c = 1.2407`, `mean_power_w = 864.8`
- `hdrl`, `typical_heat_window`: `m_s = 0.2301`, `violation_pct = 7.14`, `rmse_22_c = 1.0003`, `mean_power_w = 1035.9`

Interpretation:

- Thermostatic PPO is the current strongest validated comfort controller.
- HDRL is structured and energy-aware, but currently worse than thermostatic PPO on heating `m_s`.
- The benchmark is still incomplete because there is no single canonical 15 min table yet with `PI + thermostatic + HDRL + MORL` under the exact same protocol.

### 3. Predictive-Information Ablation

Status: `partial`

Evidence:

- `results/minimum_paper_suite/tables/minimum_experiment_status.csv`

Current state:

- M.2 proxy exists.
- M.3 GRU path exists and is aligned with Article 22 style logic.
- M.1 infrastructure now exists, but a benchmarked M.1 result table is still missing.

Interpretation:

- The ablation axis is structurally ready.
- It is not yet publication-ready because the final unified `M.1 vs M.2 vs M.3` table does not exist.

### 4. Transfer Block

Status: `partial`

Evidence:

- `training/run_morl_surrogate_pipeline.py`
- `surrogate/direct_tsup_adapter.py`
- `results/research_benchmark/tables/experiment_matrix.csv`

Current state:

- The `v35_calibrated` direct-TSup adapter is already wired into MORL, thermostatic, HDRL, and evaluation paths.
- This means retraining on the calibrated twin is no longer blocked.
- The current running command
  `python training/train_thermostatic.py --surrogate-kind v35_calibrated --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json`
  belongs exactly to this downstream-control-alignment stage.

Interpretation:

- Architecturally this block is now unblocked.
- Empirically it is still incomplete until we produce before/after transfer tables.

### 5. Time-Step Ablation

Status: `partial`

Evidence:

- `results/minimum_paper_suite/tables/minimum_experiment_status.csv`

Current state:

- We already have 1h benchmark families.
- We already have 15 min peak/typical heating benchmarks.
- What is still missing is one aligned comparison table for the same controller family under `1h vs 15 min`.

## Match Against the Original Structured Plan

The original `structured_plan.docx` is directionally correct, but it needed three important corrections:

1. MORL/PPO should be promoted from an optional controller branch to the **central target of the paper**.
2. The plan must explicitly distinguish `implemented`, `partial`, and `experimental` blocks.
3. The calibrated twin must be described honestly: it already improves **physical identifiability**, but not yet **free-run rollout realism**.

## Immediate Next Steps

1. Finish the current thermostatic run on `v35_calibrated`.
2. Retrain HDRL on `v35_calibrated`.
3. Run MORL surrogate pretraining on `v35_calibrated`, then fine-tune/evaluate on BOPTEST.
4. Build one canonical 15 min controller benchmark table with `PI`, `thermostatic`, `HDRL`, and `MORL` on:
   - `peak_heat_window`
   - `typical_heat_window`
5. Finalize the predictive-information ablation `M.1 / M.2 / M.3`.
6. Add one transfer table: `trained on surrogate -> fine-tuned on BOPTEST -> evaluated on BOPTEST`.
7. Keep the inverse problem frozen at `v3.5 heads_only` unless a new rollout-aware calibration clearly improves 24h free-run metrics.

## Writing Constraints for the Q1 Paper

- Do not claim that calibrated v3.5 already improves rollout realism. It does not.
- Do not present MORL/PPO as the strongest result yet. Present it as the target controller family and current active workstream.
- Use thermostatic PPO as the current strongest validated heating controller.
- Use the calibrated twin as the methodological bridge that makes the MORL transfer story credible.

## Updated Core Hypothesis

**A calibrated digital twin with explicitly identified building physics enables more reliable controller transfer and makes multi-objective HVAC control evaluation reproducible, comparable, and scientifically defensible in BOPTEST.**
