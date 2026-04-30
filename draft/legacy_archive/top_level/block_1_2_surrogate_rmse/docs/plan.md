# Block 1.2 Plan

## Why this block exists

The current `inverse calibration` route solved structural parameter recovery better than long-horizon rollout drift.

Because of that, Block 1.2 shifts focus from calibration heads to a stronger `15 min` surrogate training path.

## Working principle

Keep the old canonical baseline frozen:

- `v3.5 heads_only` stays the reference for Block 1

Use this folder for the next-stage surrogate branch:

1. collect or rebuild higher-quality direct-TSup data
2. train a stronger structural surrogate backbone
3. validate by autonomous free-run rollout, not only one-step fit
4. keep the active comfort logic on `21-24 C`, not on a fixed `22 C`

## First implementation targets

1. new 15-minute surrogate dataset path
2. new structural retrain script for the direct-TSup surrogate
3. new 24h live rollout validation script for the new branch

## Canonical entrypoints

- `block_1_2_surrogate_rmse/run_surrogate_workflow.py`
- `block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py`
- `block_1_2_surrogate_rmse/data/prepare_surrogate_15min_dataset.py`
- `block_1_2_surrogate_rmse/data/collect_surrogate_15min_boptest_data.py`
- `surrogate/calibrate_surrogate_v35.py`

Recommended run order:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage prepare
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage collect
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_collected
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage calibrate_15min
```

If you need direct control instead of the short launcher:

```bash
python block_1_2_surrogate_rmse/data/prepare_surrogate_15min_dataset.py
python block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py --preset prepared_15min --validate-safety
python block_1_2_surrogate_rmse/data/collect_surrogate_15min_boptest_data.py --profile heating_focus --write-train-subset
python block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py --preset collected_15min_focus --validate-safety
python surrogate/calibrate_surrogate_v35.py --preset full --data outputs/block_1_2_surrogate_rmse/collected_15min_dataset/episodes/winter__heat__seed42.csv --step-sec 900 --legacy-step-sec 3600
```

What each stage writes:

- `data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv`
- `data/block_1_2_surrogate_rmse/boptest_block12_15min_collected.csv`
- `outputs/block_1_2_surrogate_rmse/collected_15min_dataset/episode_summary.csv`
- `outputs/block_1_2_surrogate_rmse/collected_15min_dataset/dataset_summary.json`
- `outputs/block_1_2_surrogate_rmse/prepared_15min_baseline/launcher_summary.json`
- `outputs/block_1_2_surrogate_rmse/collected_15min_focus/launcher_summary.json`
- `outputs/surrogate_v35_inverse_boptest_15min_winter_heat/calibration_summary_boptest_v35.json`

## Success criteria

- preserve `C_zon error ~= 3.07%`
- reduce 15-minute one-step and multi-step RMSE before trying a new Stage C branch
- no regression in stability
- keep the active comfort logic centered on band `21-24 C`
