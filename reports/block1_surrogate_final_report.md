# Block 1 Final Surrogate Report

Date: 2026-04-22

## Scope

This report freezes the final Block 1 surrogate status before moving the main project focus to Block 2 in the form of a warm-start utility benchmark.

The Block 1 conclusion is:

- the calibrated 15-minute `v3.5` surrogate is strong enough for inverse calibration, rollout realism, and downstream warm-start experiments
- the same surrogate is still not valid for zero-shot closed-loop controller transfer to live BOPTEST

## Canonical Artifacts

Primary Block 1 artifacts:

- Temperature-alignment run:
  - [calibration_summary_boptest_v35.json](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop/calibration_summary_boptest_v35.json)
- Canonical downstream backend:
  - [calibration_summary_boptest_v35.json](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json)
- Prepared rollout benchmark:
  - [v35_prepared_compare_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_rollout_prepared_15min_power_head_only/v35_prepared_compare_summary.csv)
- Zero-shot thermostatic transfer benchmark:
  - [summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/summary.csv)
- First-divergence diagnostic:
  - [first_divergence_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/block13_obs_gap_no_delta_t_powerlog_tzone/first_divergence_summary.csv)
  - [feature_drift_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/block13_obs_gap_no_delta_t_powerlog_tzone/feature_drift_summary.csv)

## Table 1. Inverse Calibration and Structural Parameter

| item | value |
| --- | ---: |
| prepared dataset rows | 10,744 |
| runtime step | 900 s |
| legacy checkpoint step | 3600 s |
| temperature baseline RMSE | 0.3738 C |
| temperature calibrated RMSE | 0.2319 C |
| temperature RMSE improvement | 38.0% |
| temperature baseline MAE | 0.2783 C |
| temperature calibrated MAE | 0.1320 C |
| calibrated temperature bias | 0.0661 C |
| `C_zon` prior | 4.200e+05 J/K |
| `C_zon` after Stage B | 4.413e+05 J/K |
| `C_zon` final | 4.413e+05 J/K |
| Stage C mode | `rollout_temp_head_only` |
| Stage C selection metric | `val_rollout_rmse_free` |

Interpretation:

- Explicit `C_zon` remains stable through Stage C.
- The 15-minute staged calibration materially improves temperature fidelity without breaking the structural parameter.

## Table 2. Canonical Downstream Backend for Block 2

| item | value |
| --- | ---: |
| backend artifact | `surrogate_v35_inverse_boptest_15min_power_head_only` |
| calibrated temperature RMSE | 0.2333 C |
| calibrated temperature MAE | 0.1324 C |
| temperature bias | 0.0753 C |
| baseline power MAE | 807.83 W |
| calibrated power MAE | 482.03 W |
| power MAE reduction | 40.3% |
| `C_zon` final | 4.413e+05 J/K |
| Stage C mode | `power_head_only` |

Interpretation:

- This checkpoint is the best current compromise for downstream control work.
- It preserves the identified building physics and materially improves the energy channel.

## Table 3. Prepared Rollout Validation

| variant | 1-step RMSE C | 24h rollout RMSE C | mean episode RMSE C | mean episode bias C | mean episode power RMSE W |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_v35` | 1.4908 | 1.4665 | 1.4842 | 0.4297 | 1223.81 |
| `calibrated_v35` | 0.6546 | 0.6441 | 0.6495 | -0.1173 | 687.57 |

Interpretation:

- Temperature rollout realism improves strongly relative to the raw surrogate.
- Power rollout error also drops materially after the dedicated power-head calibration.
- This is sufficient evidence to use the surrogate as a warm-start environment.

## Table 4. Zero-Shot Thermostatic Closed-Loop Transfer

Observation stack used in the final diagnostic:

- `obs_ablation = no_delta_t`
- `power_feature_mode = clipped_log`
- `t_zone_feature_mode = comfort_centered`

| scenario | live `m_s` | live violation % | temp RMSE C | power RMSE W | action RMSE | surrogate `m_s` | `m_s` gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `peak_heat_window` | 1.0465 | 77.08 | 4.3199 | 1453.09 | 0.9962 | 0.1602 | -0.8862 |
| `typical_heat_window` | 1.1020 | 82.37 | 4.4014 | 1592.63 | 1.0315 | 0.0876 | -1.0144 |

Interpretation:

- The surrogate still massively underestimates discomfort and violation in live closed loop.
- Therefore the current surrogate is not valid for zero-shot controller deployment.

## Table 5. First-Divergence Diagnostic

| scenario | first divergence step | first divergence day | action gap norm | top feature | top feature abs drift |
| --- | ---: | ---: | ---: | --- | ---: |
| `peak_heat_window` | 1 | 0.03125 | 2.0000 | `t_zone_norm` | 2.0000 |
| `typical_heat_window` | 1 | 0.03125 | 2.0142 | `t_zone_norm` | 2.0000 |

Interpretation:

- The first-step transfer gap is still immediate.
- The dominant representation mismatch is now `t_zone_norm`, not `p_total_norm`.
- This is the main reason Block 1 is frozen as "warm-start ready" rather than "zero-shot ready".

## Canonical Block 1 Graphs

Prepared rollout comfort trace:

![Comfort trace 21-24 vs BOPTEST](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_trace_21_24_vs_boptest.png)

Prepared rollout HVAC power trace:

![HVAC power trace vs BOPTEST](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_rollout_prepared_15min_power_head_only/hvac_power_trace_vs_boptest.png)

Prepared rollout cumulative energy trace:

![Cumulative energy trace vs BOPTEST](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_rollout_prepared_15min_power_head_only/cumulative_energy_trace_vs_boptest.png)

Prepared rollout comfort violation comparison:

![Comfort violation comparison](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_violation_comparison.png)

## Final Block 1 Verdict

Frozen Block 1 claim:

- `v3.5` with explicit `C_zon` is a physically defensible calibrated surrogate at 15-minute resolution
- the surrogate materially improves temperature and power rollout realism relative to the raw model
- the surrogate is strong enough for pretraining and warm-start studies
- the surrogate is still not strong enough for zero-shot closed-loop transfer on live BOPTEST

Therefore the main project focus now moves to Block 2:

- `from scratch on BOPTEST`
- `pretrained on surrogate -> fine-tuned on BOPTEST`
- compare jumpstart, convergence speed, and final performance
