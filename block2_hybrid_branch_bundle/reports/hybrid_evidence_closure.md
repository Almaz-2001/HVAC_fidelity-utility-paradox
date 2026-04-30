# Hybrid Evidence Closure

Date: 2026-04-30

## Scope

This report closes the two remaining evidence gaps:

1. standalone hybrid disagreement summary
2. hybrid transfer validation against pure `v3` and direct `v3.5`

## Physics Side

The hybrid backend uses `v3` as primary dynamics and `v3.5` as a physics regularizer.

On the canonical live BOPTEST hybrid traces:

- overall mean temperature disagreement: `0.969 C`
- overall p95 temperature disagreement: `2.516 C`
- overall mean power disagreement: `708.4 W`
- overall p95 power disagreement: `1235.5 W`

This is bounded disagreement, not chaotic divergence.

Also on the same hybrid trajectories:

- primary `v3` temp RMSE: `1.158 C`
- comparison `v3.5` temp RMSE: `0.206 C`
- primary `v3` power RMSE: `1184.1 W`
- comparison `v3.5` power RMSE: `621.5 W`

## Transfer Side

### Peak heat window

| variant | ms_gap | action_gap_norm | first_divergence_step | top_feature |
| --- | ---: | ---: | ---: | --- |
| pure_v3 | -0.1016 | 0.3766 | 1 | p_total_norm |
| hybrid_l010 | -0.0234 | 0.4726 | 1 | p_total_norm |
| direct_v35 | -0.8862 | 2.0000 | 1 | t_zone_norm |

### Typical heat window

| variant | ms_gap | action_gap_norm | first_divergence_step | top_feature |
| --- | ---: | ---: | ---: | --- |
| pure_v3 | -0.0682 | 0.3334 | 1 | p_total_norm |
| hybrid_l010 | -0.0214 | 0.2531 | 16 | p_total_norm |
| direct_v35 | -1.0144 | 2.0142 | 1 | t_zone_norm |

## Conclusion

- `C_zon` correctness was already closed by Block 1.
- The hybrid disagreement is now explicitly summarized and remains bounded.
- The hybrid transfer gap is no longer compared only to direct `v3.5`; it is also compared to pure `v3`.

The honest claim after this closure step is:

**hybrid regularization is no longer just promising; it is now the strongest verified compromise across physics consistency and downstream control utility, although it is still not a dominant standalone physics surrogate.**

## Figures

![Hybrid disagreement](figures/hybrid_disagreement_summary.png)

![Hybrid transfer gap comparison](figures/hybrid_transfer_gap_comparison.png)
