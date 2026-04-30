# Hou-and-Evins Final Open-Items Closure

Date: 2026-04-30

## Scope

This note closes the last previously open methodology items:

1. explicit sample-size justification
2. explicit split representativeness table
3. explicit paper positioning: formal HPO vs targeted sensitivity analysis

## Generated Artifacts

- [hou_evins_sample_size_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_size_justification_table.csv)
- [hou_evins_split_representativeness_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_split_representativeness_table.csv)
- [hou_evins_targeted_sensitivity_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_targeted_sensitivity_table.csv)

## 1. Sample-Size Justification

The project now makes the dataset-size decision explicit rather than implicit.

Key justification:

- the prepared 15-minute bootstrap corpus uses only 10744 rows and no new live BOPTEST collection time
- despite that smaller size, it supports the canonical calibrated `v3.5` backend with:
  - inverse RMSE `0.2319 C`
  - 24-hour rollout RMSE `0.6441 C`
- the much larger collected 15-minute exploration corpus uses 48384 rows and 17.18 minutes of new BOPTEST collection
- yet it still underperforms on 1-hour safety fidelity:
  - `h4_rmse = 1.1146 C`

This closes the sample-size argument as a cost-vs-accuracy decision:

- broad hourly data for `v3`
- compact prepared 15-minute bootstrap for canonical `v3.5`
- larger collected 15-minute corpora kept as robustness experiments, not canonical corpora

## 2. Split Representativeness

The project now makes split strategy and representativeness explicit.

Important nuance:

- the legacy `v3` hourly control corpus uses a contiguous row split
- this leaves validation with all policies but only the autumn tail of the corpus
- that limitation is now frozen explicitly in the split table

Therefore the paper should state clearly:

- one-step supervised validation is not the only judge
- final surrogate usefulness is validated through external prepared-rollout checks and live BOPTEST transfer benchmarks

## 3. HPO Positioning

The paper should **not** claim formal HPO.

The correct and defensible framing is:

- **targeted sensitivity analysis**

This is now explicit in the sensitivity table:

- observation ablations
- feature-encoding sweeps
- hybrid `lambda_temp_disagree` sweep

The canonical hybrid winner remains:

- `l010`

## Result

The Hou-and-Evins packaging is now complete enough for the thermostatic branch.

The remaining work is no longer methodology closure.

The remaining work is controller-family promotion:

1. `HDRL` on canonical `hybrid_l010`
2. then `MORL` on the same hybrid default

