# Block 2 HDRL Lambda Sweep

Date: 2026-05-05

## Scope

This sweep tested whether the thermostatic hybrid regularization default transfers to the next controller family:

- backend: `hybrid_v3_v35`
- step: `900 s`
- comfort band: `21-24 C`
- observation path: `no_delta_t`, `power_feature_mode=clipped_log`, `t_zone_feature_mode=raw`
- fixed `lambda_power_disagree = 5e-5`
- swept `lambda_temp_disagree = {0.00, 0.03, 0.05, 0.10}`

## Results

Reference table:

- [block2_hdrl_lambda_sweep_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hdrl_lambda_sweep_summary.csv)

### Peak heat window

| variant | m_s | violation_pct | rmse_center_c | energy_kwh |
| --- | ---: | ---: | ---: | ---: |
| l000 | 0.1803 | 6.10 | 0.751 | 329.6 |
| l003 | 0.3073 | 11.90 | 0.993 | 311.5 |
| l005 | 0.4184 | 20.98 | 1.303 | 298.1 |
| l010 | 0.4395 | 22.99 | 1.343 | 300.4 |

### Typical heat window

| variant | m_s | violation_pct | rmse_center_c | energy_kwh |
| --- | ---: | ---: | ---: | ---: |
| l000 | 0.2337 | 3.12 | 0.691 | 385.1 |
| l003 | 0.2964 | 9.38 | 0.959 | 369.6 |
| l005 | 0.5118 | 27.38 | 1.491 | 354.5 |
| l010 | 0.5114 | 30.65 | 1.455 | 357.1 |

## Interpretation

- `l000` is the best HDRL setting on both scenarios.
- Increasing `lambda_temp_disagree` consistently degrades HDRL control quality.
- Therefore the thermostatic-best hybrid default `lambda_temp_disagree = 0.10` does **not** transfer to the hierarchical controller family.

The strongest current claim is:

**Hybrid regularization is controller-family specific. It is beneficial for thermostatic PPO, but the HDRL branch rejects temperature disagreement regularization and performs best with `lambda_temp_disagree = 0.00`.**

## Consequence for MORL

The next controller-family promotion should start from:

- `surrogate_kind = hybrid_v3_v35`
- `lambda_temp_disagree = 0.00`
- `lambda_power_disagree = 5e-5`

This keeps the control-oriented `v3` temperature dynamics while preserving a soft `v3.5` power regularizer.
