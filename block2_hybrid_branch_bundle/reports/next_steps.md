# Minimum Paper Suite Next Steps

## Active Backends

The active physics backend is:

- `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json`

This is the current 15-minute calibrated `v3.5` surrogate that preserves the identified `C_zon` and improves the power channel.

The active control backend is:

- `outputs/surrogate_v2/rc_node_v3_tsupply.pt`

The current Block 2 branch combines both through `hybrid_v3_v35`.

The controller-family defaults are now:

- `lambda_temp_disagree = 0.10`
- `lambda_power_disagree = 5e-5`

for the closed thermostatic branch, and:

- `lambda_temp_disagree = 0.00`
- `lambda_power_disagree = 5e-5`

for the current HDRL/MORL promotion branch.

## Block 1 Status

Block 1 is now frozen as a reporting baseline.

Canonical report:

- [block1_surrogate_final_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block1_surrogate_final_report.md)

What is fixed:

- explicit-physics inverse calibration works
- prepared rollout realism is materially better than the raw surrogate
- zero-shot thermostatic transfer still fails

Therefore the next question is no longer "can the surrogate replace BOPTEST directly?"

The next question is:

- "does surrogate pretraining improve BOPTEST learning relative to scratch?"

## Block 2 Status

Direct `v3.5` warm-start is now frozen as a negative baseline.

Measured result:

- `scratch on BOPTEST` is much better than `v3.5 warm-start + BOPTEST fine-tune`
- therefore the next question is no longer pure warm-start utility
- the next question is whether hybrid regularization can preserve `v3` controllability while adding `v3.5` physical bias

The thermostatic hybrid evidence gap is now closed by:

- [hybrid_evidence_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_evidence_closure.md)
- [hybrid_disagreement_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_disagreement_summary.csv)
- [hybrid_transfer_comparison.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hybrid_transfer_comparison.csv)

The previously partial Hou-and-Evins packaging items are now also closed by:

- [hou_evins_partial_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_partial_closure.md)
- [hou_evins_sample_generation_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_generation_table.csv)
- [hou_evins_stage_a_processing_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_stage_a_processing_table.csv)
- [hou_evins_feature_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_feature_justification_table.csv)
- [hou_evins_architecture_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_architecture_justification_table.csv)

## Immediate Focus: Promote Canonical Hybrid Beyond Thermostatic

The active benchmark track is now:

1. thermostatic `hybrid_v3_v35` is already closed
2. HDRL sweep is closed with winner:
   - `lambda_temp_disagree = 0.00`
   - `lambda_power_disagree = 5e-5`
3. MORL is now closed as:
   - `hybrid_v3_v35`
   - `lambda_temp_disagree = 0.00`
   - `lambda_power_disagree = 5e-5`
   - `17D TSup-style observation path`

Thermostatic success criterion is now satisfied:

- hybrid stays close to pure `v3` on `m_s`
- hybrid preserves energy advantage
- hybrid transfer evidence is materially better than direct `v3.5`

Therefore the next active focus is no longer controller-family promotion inside Block 2.
The next active focus is Block 3 cross-case transferability.

## Remaining Open Hou-and-Evins Items

For the thermostatic branch, there are no longer any open Hou-and-Evins packaging blockers.

The final closure package is now:

- [hou_evins_final_open_items_closure.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_final_open_items_closure.md)
- [hou_evins_sample_size_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_size_justification_table.csv)
- [hou_evins_split_representativeness_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_split_representativeness_table.csv)
- [hou_evins_targeted_sensitivity_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_targeted_sensitivity_table.csv)

The explicit paper position is now frozen:

- no formal HPO claim
- targeted sensitivity analysis only

## Canonical Block 2 Command

The canonical entry point for the current hybrid thermostatic branch is:

```powershell
python training/train_thermostatic.py --surrogate-kind hybrid_v3_v35 --surrogate-path outputs/surrogate_v2/rc_node_v3_tsupply.pt --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json --step-sec 900 --comfort-low 21 --comfort-high 24 --obs-ablation no_delta_t --power-feature-mode clipped_log --t-zone-feature-mode raw --lambda-temp-disagree 0.10 --lambda-power-disagree 5e-5 --save-name ppo_thermostatic_hybrid_v3_v35_l010
```

Then benchmark it with:

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers thermostatic --thermostatic-model models/ppo_thermostatic_hybrid_v3_v35_l010.zip --output-dir outputs/block2_thermostatic_hybrid_v3_v35_l010
```

## Current Block 2 Principle

For the paper, the key comparison is now:

- pure `v3` control surrogate
- direct `v3.5` warm-start failure
- hybrid `v3 + v3.5`
- PI baseline

The next promotion path is now explicit:

1. `HDRL lambda sweep`
   - complete
   - winner: `l000`
2. `MORL hybrid_v3_v35`
   - closed with `17D power-only` canonical result
3. `Block 3 cross-case transferability`

## Unified 15-Minute Benchmark Reference

The HDRL sweep closure is now documented in:

- [block2_hdrl_lambda_sweep_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hdrl_lambda_sweep_report.md)
- [block2_hdrl_lambda_sweep_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hdrl_lambda_sweep_summary.csv)

After model artifacts are ready, the unified benchmark entry point remains:

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers pi,thermostatic,hdrl,morl --output-dir outputs/block2_bestest_air_15min_unified
```

If explicit model paths are needed:

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers pi,thermostatic,hdrl,morl --thermostatic-model models/ppo_thermostatic.zip --hdrl-winter-model models/ppo_winter_final.zip --hdrl-summer-model models/ppo_summer_final.zip --morl-model outputs/morl_hybrid_v3_v35_power_only_17d/seed42/finetune_boptest/models/ppo_model.zip --output-dir outputs/block2_bestest_air_15min_unified
```

## Rollout Validation Reference

The canonical prepared rollout validation command for the frozen Block 1 backend is:

```powershell
python evaluation/validate_surrogate_v35_rollout_prepared.py --summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json --out-dir outputs/surrogate_v35_rollout_prepared_15min_power_head_only
```
