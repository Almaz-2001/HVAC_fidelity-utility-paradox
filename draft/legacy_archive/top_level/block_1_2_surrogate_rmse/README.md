# Block 1.2 - 15-Minute Surrogate RMSE Improvement

This folder is the isolated workspace for the next surrogate research stage.

## Goal

Improve surrogate rollout realism beyond the current `v3.5 heads_only` baseline at `15 min` resolution.

Primary target:

- preserve the current `C_zon error ~= 3.07%`
- reduce long-horizon rollout RMSE toward the next target zone
- switch active collection and validation logic from fixed `22 C` to comfort band `21-24 C`

Current canonical baseline remains outside this folder:

- `outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json`
- `outputs/surrogate_v35_rollout_live/v35_compare_summary.csv`

## Scope of Block 1.2

This workstream is for new experiments only:

- new surrogate data collection paths
- new structural retraining paths
- new rollout-aware validation scripts
- comparison notes and experiment logs

This folder does not replace the current canonical Block 1 outputs.

## Active entrypoints

Short launcher:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage prepare
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared_rollout
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared_rollout_long
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage collect
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage build_hybrid
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_collected
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_hybrid
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage calibrate_15min
```

Direct canonical scripts:

```bash
python block_1_2_surrogate_rmse/data/prepare_surrogate_15min_dataset.py
python block_1_2_surrogate_rmse/data/collect_surrogate_15min_boptest_data.py --profile heating_focus --write-train-subset
python block_1_2_surrogate_rmse/data/build_surrogate_15min_hybrid_dataset.py
python block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py --preset prepared_15min_rollout_long --validate-safety
python surrogate/calibrate_surrogate_v35.py --preset full --data <episode_csv> --step-sec 900 --legacy-step-sec 3600
```

Legacy filenames still exist as compatibility wrappers:

- `block_1_2_surrogate_rmse/data/prepare_block12_15min_dataset.py`
- `block_1_2_surrogate_rmse/data/collect_block12_15min_dataset.py`
- `block_1_2_surrogate_rmse/training/train_block12_backbone.py`
- `surrogate/inverse_problem_boptest_v35.py`
- `surrogate/train_surrogate_v2.py`

## Internal structure

- `docs/`
  Planning notes and experiment decisions.
- `data/`
  New dataset-related helper files for this block.
- `training/`
  New training entrypoints specific to Block 1.2.
- `evaluation/`
  New evaluation and validation entrypoints specific to Block 1.2.
