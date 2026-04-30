# Structured Plan Status Update

Date: 2026-04-22

## Central Q1 Axis

The paper remains centered on one explicit claim:

**A calibrated digital twin with explicit building physics is valuable for downstream HVAC control, with MORL/PPO as the target controller family.**

The practical interpretation is now sharper:

- Block 1 must prove physical identifiability and rollout realism.
- Block 2 must prove downstream utility.
- The most defensible downstream utility claim is now `warm-start utility`, not `zero-shot transfer`.

## Current Verified Status

### 1. Block 1 Surrogate Fidelity

Status: `frozen for this paper iteration`

Canonical report:

- [block1_surrogate_final_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block1_surrogate_final_report.md)

Frozen Block 1 facts:

- best 15-minute temperature-alignment run:
  - baseline RMSE `0.3738 C`
  - calibrated RMSE `0.2319 C`
  - improvement `38.0%`
  - `C_zon final = 4.413e+05 J/K`
- canonical downstream backend:
  - `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json`
  - baseline power MAE `807.83 W`
  - calibrated power MAE `482.03 W`
  - `C_zon final = 4.413e+05 J/K`
- prepared rollout validation on the canonical backend:
  - raw 24h rollout RMSE `1.4665 C`
  - calibrated 24h rollout RMSE `0.6441 C`
  - calibrated mean episode power RMSE `687.57 W`

Interpretation:

- Block 1 is successful on physical identification and open-loop realism.
- The calibrated 15-minute surrogate is good enough to serve as a pretraining environment.
- We are no longer trying to squeeze Block 1 into a zero-shot controller simulator before moving on.

### 2. Zero-Shot Transfer Diagnosis

Status: `measured and frozen as a known limitation`

Frozen limitation:

- thermostatic zero-shot transfer to live BOPTEST still fails badly
- in the latest diagnostic setting:
  - `peak_heat_window m_s = 1.0465`
  - `typical_heat_window m_s = 1.1020`
  - `first_divergence_step = 1` in both scenarios
  - dominant first-divergence channel is `t_zone_norm`

Interpretation:

- The current surrogate is not yet valid for direct controller deployment without fine-tuning on BOPTEST.
- This limitation does not block the warm-start research question.

### 3. Block 2 Warm-Start Utility Benchmark

Status: `measured and frozen as negative`

Measured result:

- `v3.5 warm-start + BOPTEST fine-tune` is materially worse than `scratch on BOPTEST`
- peak scenario:
  - scratch `m_s = 0.4653`
  - warm-start `m_s = 1.2701`
- typical scenario:
  - scratch `m_s = 0.5776`
  - warm-start `m_s = 1.2888`

Interpretation:

- the current calibrated `v3.5` surrogate is not yet useful as a direct policy-pretraining environment
- the failure mode is consistent with the measured zero-shot transfer gap
- this branch remains useful as a negative control for the paper

Implementation retained for reproducibility:

- [launch_thermostatic_warmstart_benchmark.py](C:/Users/user/Desktop/HVAC_DRL_MORL/training/launch_thermostatic_warmstart_benchmark.py)
- [finetune_tsup_policies_boptest.py](C:/Users/user/Desktop/HVAC_DRL_MORL/training/finetune_tsup_policies_boptest.py)

### 4. Block 2 Hybrid Regularized Surrogate

Status: `active`

New Block 2 question:

- can `v3` remain the control-oriented training dynamics while `v3.5` acts as a physics regularizer?

Current hybrid setup:

- training dynamics: `v3`
- regularizer: `v3.5 disagreement penalty`
- controller family: `thermostatic PPO`
- comfort band: `21-24 C`
- step size: `900 s`
- default `lambda_temp_disagree = 0.10`
- default `lambda_power_disagree = 5e-5`

Current verified hybrid result:

- peak scenario:
  - `m_s = 0.0866`
  - violation `4.69%`
  - energy `305.3 kWh`
- typical scenario:
  - `m_s = 0.0411`
  - violation `2.38%`
  - energy `352.8 kWh`

Interpretation:

- hybrid regularization rescues the `v3.5` branch from catastrophic control performance
- `lambda = 0.10` is the best current compromise after the sweep
- hybrid is now close to pure `v3` on the peak scenario and better than pure `v3` on the typical scenario
- hybrid remains better than pure `v3` on energy in both scenarios
- the immediate next step is to transfer this exact default to the next controller family, not to continue thermostatic tuning

### 5. MORL Track

Status: `architecturally ready, empirically pending`

Verified infrastructure:

- surrogate pretrain launcher exists
- BOPTEST fine-tune launcher exists
- yearly BOPTEST evaluation launcher exists

Interpretation:

- MORL remains the final controller family for the paper.
- The next defensible empirical path is:
  - prove warm-start utility first on thermostatic
  - then promote the same methodology to MORL

## Immediate Next Steps

1. Treat Block 1 as frozen except for bug fixes and report cleanup.
2. Treat direct `v3.5 warm-start` as a frozen negative baseline.
3. Run the hybrid `lambda_temp_disagree` sweep:
   - completed
   - winner: `0.10`
4. Promote the same hybrid default to `HDRL`.
5. If `HDRL` remains stable, promote the same hybrid default to `MORL`.
6. Only if the next controller family regresses badly, revisit the surrogate split-role architecture.

## Writing Position for the Paper

We can now state the surrogate result honestly:

- explicit-physics calibration at 15-minute resolution works
- rollout realism improves materially for both temperature and power
- zero-shot closed-loop transfer is still not solved
- direct `v3.5` warm-start is not sufficient for downstream RL
- the current positive Block 2 direction is hybrid regularization: `v3` for controllability, `v3.5` for physical censorship
- the current canonical hybrid setting is `lambda_temp_disagree = 0.10`

## Updated Core Hypothesis

**A calibrated digital twin with explicit structural building physics is more useful as a physical regularizer for control-oriented surrogate learning than as a direct standalone policy-training environment, with MORL/PPO as the final target.**
