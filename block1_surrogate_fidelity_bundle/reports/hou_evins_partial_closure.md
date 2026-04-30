# Hou-and-Evins Partial Closure

Date: 2026-04-30

## Scope

This document closes the previously partial items:

1. sample generation as a paper-facing table
2. Stage A preprocessing as an article-facing block
3. feature significance and encoding justification as a numerical table
4. architecture justification as a comparative table

## Generated Tables

- [hou_evins_sample_generation_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_generation_table.csv)
- [hou_evins_stage_a_processing_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_stage_a_processing_table.csv)
- [hou_evins_feature_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_feature_justification_table.csv)
- [hou_evins_architecture_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_architecture_justification_table.csv)

## 1. Sample Generation

The project now has an explicit sample-generation table covering:

- hourly `v3` direct-TSup corpus
- prepared 15-minute `v3.5` bootstrap corpus
- collected 15-minute exploration corpus

This closes the packaging gap around:

- dataset size
- step size
- excitation/controller mix
- seasonal/scenario coverage
- state and power ranges

## 2. Stage A Preprocessing

Stage A is now described as a concrete sequence:

- latency compensation search
- temperature bias removal
- power affine normalization
- rolling denoise
- causal delta recomputation

Each operation is now tied to:

- an explicit implementation rule
- a numerical selection criterion
- a purpose in the inverse-calibration pipeline

## 3. Feature Significance and Encoding Justification

The feature/encoding table now makes the numerical logic explicit.

Key result:

- the strongest direct `v3.5` diagnostic branch came from removing `delta_t_zone_norm`, then clipping/log-scaling power, then testing comfort-centered temperature encoding
- however the final best downstream result is not direct `v3.5`, but `hybrid_l010`

The best direct-`v3.5` diagnostic row in the table is:

- variant: `no_delta_t`
- peak `m_s = 1.0361`
- typical `m_s = 1.1208`

This closes the packaging gap around observation/encoding justification.

## 4. Architecture Justification

The architecture comparison is now explicit:

- `v3`: best pure control surrogate
- `v3.5 calibrated`: best physical twin, poor standalone control surrogate
- `hybrid_l010`: best verified compromise

For the canonical hybrid row:

- peak `m_s = 0.0866`
- typical `m_s = 0.0411`
- peak transfer RMSE = `0.633 C`
- typical `first_divergence_step = 16`

## Result

These four items should no longer be treated as partially documented.

They are now explicit, numerical, and article-facing.

What still remains open is the separate set of truly open Hou-and-Evins items:

- formal HPO or an explicit decision to frame the work as targeted sensitivity analysis
- sample-size justification as a cost-vs-accuracy argument
- data split representativeness as a separate paper table
