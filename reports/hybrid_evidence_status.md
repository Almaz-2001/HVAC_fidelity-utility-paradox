# Hybrid Evidence Status

Date: 2026-04-30

This note answers the strict question:

**Do we still have an incomplete hybrid claim, or is the thermostatic hybrid branch now closed strongly enough to promote to the next controller family?**

## Required Three-Layer Standard

To justify the hybrid branch, we need all three layers:

1. physics side
2. predictive/transfer side
3. control side

## Current Status

| layer | required evidence | status | current support |
| --- | --- | --- | --- |
| physics | `C_zon` remains correct | yes | canonical `v3.5` Block 1 backend remains fixed at `4.413e+05 J/K` |
| physics | disagreement stays bounded and not chaotic | yes | [hybrid_disagreement_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_disagreement_summary.csv) shows overall mean temp disagreement `0.969 C`, p95 `2.516 C`, mean power disagreement `708.4 W`, p95 `1235.5 W` |
| predictive/transfer | hybrid materially improves over direct `v3.5` | yes | [hybrid_transfer_comparison.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_transfer_comparison.csv) shows direct `v3.5` action gap about `2.0` and catastrophic `ms_gap`, while `hybrid_l010` reduces this to `0.473` peak and `0.253` typical |
| predictive/transfer | hybrid is not worse than pure `v3` in the practically important sense | partial-yes | hybrid is slightly worse than pure `v3` on `peak` action gap (`0.473` vs `0.377`) but clearly better on `typical`, where `first_divergence_step` moves from `1` to `16` |
| control | hybrid stays close to pure `v3` on `m_s` | yes | peak `0.0866` vs `0.0725`; typical `0.0411` vs `0.0947` |
| control | hybrid preserves some energy advantage | yes | hybrid uses less energy than pure `v3` on both scenarios |

## Canonical Supporting Files

- [hybrid_evidence_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_evidence_closure.md)
- [hybrid_disagreement_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_disagreement_summary.csv)
- [hybrid_transfer_comparison.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_transfer_comparison.csv)
- [hybrid_disagreement_summary.png](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/figures/hybrid_disagreement_summary.png)
- [hybrid_transfer_gap_comparison.png](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/figures/hybrid_transfer_gap_comparison.png)

## Current Honest Conclusion

The incomplete-status version is no longer correct.

The updated conclusion is:

**thermostatic `hybrid_l010` is now the strongest verified compromise across physical consistency and downstream control utility, but it is still not a dominant standalone predictive twin.**

That wording matters.

What is now closed:

- the physics regularizer does not drift chaotically
- the hybrid branch is decisively better than direct `v3.5`
- the hybrid branch is close enough to pure `v3` on control utility to justify promotion

What is still not being claimed:

- that the hybrid backend is the best standalone predictive model
- that it fully dominates pure `v3` on every transfer statistic in every scenario

## Operational Decision

The thermostatic hybrid branch is closed strongly enough to move on.

The next active empirical step is:

1. promote canonical `hybrid_l010` to `HDRL`
2. if stable, promote the same default to `MORL`
3. do not reopen thermostatic feature tuning unless the next controller family regresses badly
