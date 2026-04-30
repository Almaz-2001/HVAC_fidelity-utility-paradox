# Hybrid Evidence Status

Date: 2026-04-29

This note answers a strict question:

Can we already make the strong hybrid claim, or do we still only have an intermediate claim?

## Target Strong Claim

To make the strongest version of the paper claim, we need all three layers:

1. physics side
2. predictive side
3. control side

## Current Status

| layer | required evidence | status | current support |
| --- | --- | --- | --- |
| physics | `C_zon` remains correct | yes | Block 1 canonical `v3.5` calibration is stable at `4.413e+05 J/K` |
| physics | disagreement stays bounded and not chaotic | partial | hybrid uses disagreement penalty, but there is no standalone summary yet for mean / p95 disagreement on trajectories |
| predictive | hybrid better than direct `v3.5` on rollout realism | partial | direct `v3.5` is worse in downstream control, but we do not yet have a dedicated hybrid open-loop rollout benchmark table against pure `v3` and direct `v3.5` |
| predictive | hybrid is not worse than pure `v3` on drift / transfer-gap | no | no canonical hybrid `first_divergence_step` and `action_gap_norm` report yet |
| control | hybrid closer to pure `v3` on `m_s` | yes | `hybrid_l010` is near pure `v3` on peak and better on typical |
| control | hybrid preserves some energy advantage | yes | `hybrid_l010` uses less energy than pure `v3` on both scenarios |

## Current Honest Conclusion

The current conclusion is still:

**hybrid regularization is promising and already useful, but not yet dominant by the full three-layer standard.**

This is because:

- control-side evidence is already strong
- physics-side evidence is acceptable but not yet summarized as a dedicated disagreement benchmark
- predictive-side evidence for the hybrid is still incomplete

## What Is Still Missing

Two additional canonical checks are still needed before the strongest claim:

1. `hybrid` disagreement summary
   - mean disagreement
   - p95 disagreement
   - ideally per scenario
2. `hybrid` predictive/transfer validation
   - rollout realism table against pure `v3` and direct `v3.5`
   - transfer-gap table:
     - `first_divergence_step`
     - `action_gap_norm`

## Practical Paper Position Right Now

Right now we can already defend:

- `v3.5` alone is physically valuable but not controller-friendly
- `v3` alone is controller-friendly but less physically grounded
- `hybrid_l010` is the best currently verified compromise

But the strongest phrasing should still be:

**hybrid regularization is currently the leading Block 2 direction, not the fully closed final proof.**
