# Hou and Evins Compliance Matrix

Date: 2026-04-30

## Purpose

This document maps the current project state to the methodology requirements you summarized from Hou and Evins.

It answers one practical question:

**What is already strong enough for a Q1 paper, and what still must be added as explicit numerical justification?**

Status labels:

- `closed`: already evidenced well enough
- `partial`: engineering work exists, but article-facing evidence is still incomplete
- `open`: still missing as a paper requirement

## 1. Sample Generation

| requirement | status | current project state | what is still missing |
| --- | --- | --- | --- |
| Explicit description of sample-generation pipelines | closed | The sample-generation pipelines are now frozen in [hou_evins_sample_generation_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_generation_table.csv), covering the hourly `v3` direct-TSup corpus, the prepared 15-minute bootstrap corpus, and the collected 15-minute exploration corpus. Key runbook: [reproduce_current_state_runbook.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/reproduce_current_state_runbook.md). | Nothing essential beyond article prose. |
| Range and distribution reporting | closed | The same sample-generation table now exposes the numeric coverage of each corpus: rows, episodes, step size, controller/policy mix, scenario/season mix, and state/power ranges. | A separate figure would be optional polish, not a blocker. |
| Significance and independence of inputs | closed | The retained and rejected observation/encoding variants are now summarized numerically in [hou_evins_feature_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_feature_justification_table.csv), including `m_s`, violation, divergence step, action-gap, and dominant drift feature. | Nothing essential for the thermostatic branch. |
| Sample-size justification | closed | The cost-vs-accuracy dataset choice is now explicit in [hou_evins_sample_size_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_size_justification_table.csv), including row counts, collection cost, resulting fidelity metrics, and retain/reject decisions. | Nothing essential beyond article prose. |
| Excitation-window logic | partial | The inverse-calibration branch already uses excitation logic implicitly through Stage B/C and prepared 15-minute trajectories. | We still need a concise paper description of how excitation windows increase identifiability and why they were used. |

## 2. Data Processing

| requirement | status | current project state | what is still missing |
| --- | --- | --- | --- |
| Stage A preprocessing documented | closed | Stage A is now documented explicitly in [hou_evins_stage_a_processing_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_stage_a_processing_table.csv), including latency search, temperature bias removal, power affine normalization, rolling denoise, and causal delta recomputation. | Nothing essential beyond method-section prose. |
| Feature encoding justified numerically | closed | Numerical justification is now explicit in [hou_evins_feature_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_feature_justification_table.csv). The table ties each encoding/ablation choice to the resulting transfer and control metrics. | Nothing essential for the thermostatic branch. |
| Train/val/test split strategy | closed | Split logic is now frozen explicitly in [hou_evins_split_representativeness_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_split_representativeness_table.csv), including contiguous row split for legacy `v3`, episode-aware split for canonical `v3.5`, and external BOPTEST testing for downstream control claims. | Nothing essential beyond article prose. |
| Representativeness of splits | closed | The same split table now states where coverage is broad, where it is limited, and how external rollout/transfer benchmarks compensate when supervised validation is not fully representative. | Nothing essential for the thermostatic branch. |

## 3. NN-Based Surrogate Training

| requirement | status | current project state | what is still missing |
| --- | --- | --- | --- |
| Numerical justification for architecture | closed | The architecture comparison is now explicit in [hou_evins_architecture_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_architecture_justification_table.csv), with `v3`, `v3.5`, and `hybrid_l010` compared across Block 1 fidelity, transfer behavior, and downstream control KPI. Supporting reports remain [block2_hybrid_surrogate_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hybrid_surrogate_report.md) and [hybrid_evidence_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_evidence_closure.md). | Nothing essential beyond article prose. |
| Formal HPO | closed by explicit non-claim | We now state explicitly in [hou_evins_final_open_items_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_final_open_items_closure.md) and [hou_evins_targeted_sensitivity_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_targeted_sensitivity_table.csv) that the paper does **not** claim formal HPO. The correct framing is targeted sensitivity analysis over observation, encoding, and hybrid-regularization axes. | Nothing essential beyond keeping the claim boundary explicit in the paper. |
| Stage B/C training rationale | closed | This is one of the strongest parts of the project: explicit `C_zon` identification followed by calibration heads. | We should document it cleanly in the method section, but the evidence itself is already there. |
| Prevention of network “cheating” | closed | This is already the core rationale for moving from flexible `v3`-style regression to `v3.5` explicit physics and staged calibration. | Only article-facing prose is needed, not more experiments. |

## 4. Surrogate Validation

| requirement | status | current project state | what is still missing |
| --- | --- | --- | --- |
| Replicative validity: one-step accuracy | closed | Block 1 already fixes one-step temperature and power improvements for the canonical `v3.5` backend. | Nothing major beyond table formatting. |
| Predictive validity: rollout realism | partial | We already have canonical prepared rollout validation for `v3.5`, including 1h/4h/8h/24h horizons and comfort/power/energy plots. | We still need a cleaner article-facing comparison table that places `v3`, `v3.5`, and `hybrid` in one frame. |
| Transfer validity: `first_divergence_step`, `action_gap_norm` | closed for thermostatic | We now have direct `v3.5`, pure `v3`, and `hybrid_l010` transfer evidence in [hybrid_transfer_comparison.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_transfer_comparison.csv). | Nothing essential for thermostatic. This same validation still needs to be repeated for `HDRL` and later `MORL`. |
| Physical validity: `C_zon` correctness | closed | Canonical Block 1 `C_zon` result is stable at `4.413e+05 J/K`. | Only final article presentation remains. |
| Disagreement summary for hybrid | closed | We now have [hybrid_disagreement_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_disagreement_summary.csv). | Nothing essential for thermostatic. |

## Strongest Current Claim

The strongest defensible claim **right now** is:

**The thermostatic branch now satisfies the core Hou-and-Evins methodology packaging requirements for physical validity, transfer validity, downstream control utility, sample generation, preprocessing, feature selection, split documentation, and architecture comparison, with targeted sensitivity analysis used explicitly instead of formal HPO.**

That means:

- the engineering substance is already strong
- the thermostatic methodology packaging is now complete
- the remaining work is controller-family promotion, not reopening surrogate justification

## Minimum Remaining Work to Be Hou-and-Evins-Strong

For the **thermostatic branch**, the minimum methodology package is now complete.

The remaining work is now cross-controller promotion, not thermostatic methodology closure:

1. apply the same reporting standard to `HDRL`
2. then apply it to `MORL`
3. keep the explicit claim boundary:
   - no formal HPO claim
   - targeted sensitivity analysis only

## Recommended Position

The pragmatic position is:

- do **not** reopen Block 1 technically
- do **not** launch a large new HPO campaign unless reviewers force it
- instead, convert the current experiment history into explicit numerical-justification tables
- then move the hybrid default to `HDRL`, and after that to `MORL`

## Immediate Next Step

The next useful deliverable is no longer another thermostatic methodology table.

It is:

**promoting the same canonical hybrid methodology to `HDRL`, then to `MORL`.**
