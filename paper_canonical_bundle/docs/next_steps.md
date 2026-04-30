# Minimum Paper Suite Next Steps

## Active Backends

The active physics backend is:

- `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json`

This is the current 15-minute calibrated `v3.5` surrogate that preserves the identified `C_zon` and improves the power channel.

The active control backend is:

- `outputs/surrogate_v2/rc_node_v3_tsupply.pt`

The current Block 2 branch combines both through `hybrid_v3_v35`.

The canonical hybrid default is now:

- `lambda_temp_disagree = 0.10`
- `lambda_power_disagree = 5e-5`

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

## Immediate Focus: Block 2 Hybrid Utility Benchmark

The active benchmark is now:

1. train thermostatic on `hybrid_v3_v35`
2. evaluate on BOPTEST
3. compare against:
   - pure `v3` thermostatic
   - PI baseline
   - failed direct `v3.5` warm-start
4. `lambda_temp_disagree` sweep completed:
   - `0.05`
   - `0.10`
   - `0.15`
5. selected canonical hybrid:
   - `0.10`

Success criterion:

- reduce hybrid `m_s` toward the pure `v3` baseline
- keep part of the current hybrid energy advantage

This criterion is now satisfied well enough to move to the next controller family.

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

1. `thermostatic hybrid_l010`
2. `HDRL hybrid_l010`
3. `MORL hybrid_l010`

## Unified 15-Minute Benchmark Reference

After model artifacts are ready, the unified benchmark entry point remains:

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers pi,thermostatic,hdrl,morl --output-dir outputs/block2_bestest_air_15min_unified
```

If explicit model paths are needed:

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers pi,thermostatic,hdrl,morl --thermostatic-model models/ppo_thermostatic.zip --hdrl-winter-model models/ppo_winter_final.zip --hdrl-summer-model models/ppo_summer_final.zip --morl-model outputs/morl_surrogate_ppo_v35_15min/seed42/finetune_boptest/models/ppo_model.zip --output-dir outputs/block2_bestest_air_15min_unified
```

## Rollout Validation Reference

The canonical prepared rollout validation command for the frozen Block 1 backend is:

```powershell
python evaluation/validate_surrogate_v35_rollout_prepared.py --summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json --out-dir outputs/surrogate_v35_rollout_prepared_15min_power_head_only
```
