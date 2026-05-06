# Block 2 MORL Canonical Result

Date: 2026-05-06

## Scope

This report freezes the canonical MORL result for Block 2 after the observation-path redesign.

Canonical MORL setup:

- backend: `hybrid_v3_v35`
- step: `900 s`
- comfort band: `21-24 C`
- `lambda_temp_disagree = 0.00`
- `lambda_power_disagree = 5e-5`
- observation path:
  - `obs_mode = extended`
  - `obs_ablation = none`
  - `delta_feature_mode = causal_smooth`
  - `power_feature_mode = clipped_log`
  - `t_zone_feature_mode = raw`

## Main Result

Canonical yearly validation output:

- [morl_yearly_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/morl_hybrid_v3_v35_power_only_17d/seed42/yearly_eval/morl_yearly_summary.csv)

Mean yearly metrics:

| variant | rmse_c | mae_c | within_1c_pct | within_05c_pct | violation_pct | energy_kwh | m_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MORL 17D power-only | 0.72 | 0.56 | 83 | 57 | 4.9 | 248.6 | 0.099 |

## Context Against the Previous 5D MORL Path

Reference comparison table:

- [block2_morl_comparison_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_morl_comparison_summary.csv)

The previous `5D` MORL path failed materially on yearly BOPTEST validation:

| variant | rmse_c | mae_c | within_1c_pct | within_05c_pct | violation_pct | energy_kwh | m_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MORL 5D basic | 4.96 | 4.17 | 19 | 9 | 74.5 | 121.0 | 1.046 |
| MORL 17D power-only | 0.72 | 0.56 | 83 | 57 | 4.9 | 248.6 | 0.099 |

Key deltas:

- `RMSE`: `4.96 -> 0.72`
- `MAE`: `4.17 -> 0.56`
- `within_1C`: `19% -> 83%`
- `within_0.5C`: `9% -> 57%`
- `violation`: `74.5% -> 4.9%`
- `m_s`: `1.046 -> 0.099`

## Interpretation

- The earlier MORL failure was not primarily a backend failure.
- The dominant limitation was the `5D` observation interface.
- After moving MORL to the `17D TSup-style` observation path, the same power-only hybrid backend became viable.
- Therefore the current Block 2 MORL result supports the split-role interpretation already established by the thermostatic and HDRL branches:
  - `v3` supplies the smooth control-oriented temperature dynamics
  - `v3.5` supplies a soft energy regularizer

## Position Relative to Other Controller Families

- Thermostatic remains the positive result for temperature disagreement regularization:
  - `lambda_temp_disagree = 0.10`
- HDRL remains the negative boundary result for temperature disagreement regularization:
  - best at `lambda_temp_disagree = 0.00`
- MORL now joins the HDRL promotion branch:
  - temperature regularization off
  - power regularization on
  - richer TSup-style observation path required

## Consequence for the Paper

Block 2 is now closed across the intended controller stack:

- thermostatic positive hybrid result
- HDRL controller-family limit result
- MORL canonical power-only result

The next active research step is no longer Block 2 tuning. The next step is Block 3 cross-case transferability.
