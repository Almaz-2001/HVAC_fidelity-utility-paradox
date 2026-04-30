# Project Workspace Map

Date: 2026-04-29

## Active Code Roots

These are the code roots that remain in the active working set.

- `surrogate/`
  Main surrogate logic.
  Active files:
  - `direct_tsup_adapter.py`
  - `inverse_problem_boptest_v35.py`
  - `rc_node_v35.py`
  - `train_surrogate_backbone.py`
- `training/`
  Main controller training and BOPTEST fine-tune logic.
  Active files:
  - `train_thermostatic.py`
  - `train_hdrl.py`
  - `train_morl_surrogate.py`
  - `finetune_tsup_policies_boptest.py`
  - `finetune_morl_boptest.py`
  - `launch_hdrl_retrain_on_calibrated_twin.py`
  - `launch_morl_pretrain_on_calibrated_twin.py`
  - `launch_morl_finetune_on_boptest.py`
  - `launch_morl_yearly_boptest_eval.py`
- `evaluation/`
  Main benchmark and reporting logic.
  Active files:
  - `benchmark_bestest_air_article7_style.py`
  - `validate_surrogate_v35_rollout_prepared.py`
  - `validate_closed_loop_transfer_thermostatic_live.py`
  - `diagnose_thermostatic_obs_transfer_gap.py`
  - `build_hybrid_surrogate_snapshot.py`
  - `build_paper_canonical_bundle.py`
- `envs/`
  Runtime environment and feature stack.
- `configs/`
  MORL and runtime configs.
- `data/`
  Active datasets:
  - `data/surrogate_v2/`
  - `data/block_1_2_surrogate_rmse/`
- `reports/`
  Canonical narrative and frozen study summaries.
- `results/`
  Canonical paper bundle and presentation-facing exports.

## Frozen But Retained

These are still useful for reproducibility, but they are not part of the day-to-day active editing set.

- `block_1_2_surrogate_rmse/`
  Historical Block 1.2 experimentation package.
- `training/launch_thermostatic_warmstart_benchmark.py`
  Reproducibility for the negative direct `v3.5` warm-start result.
- `outputs/block2_thermostatic_warmstart_utility/`
  Frozen negative baseline for Block 2.

## Draft / Archive Candidates

These should be treated as archive candidates. They should not be deleted blindly, but they no longer belong to the active working set.

- `outputs/current/`
  Older canonical-output snapshot. Superseded by `results/paper_canonical_bundle/`.
- `block_1_3_closed_loop_transfer/`
  Top-level legacy experiment stub.
- `outputs/block2_thermostatic_hybrid_v3_v35/`
  Pre-sweep hybrid result. Superseded by `block2_thermostatic_hybrid_v3_v35_l010/`.
- `outputs/block2_thermostatic_hybrid_v3_v35_l005/`
- `outputs/block2_thermostatic_hybrid_v3_v35_l015/`
  Useful for sweep history, but not canonical after selecting `l010`.
- `outputs/block2_thermostatic_no_*`
- `outputs/block2_thermostatic_with_power/`
- `outputs/block2_thermostatic_causal_smooth/`
  Feature-ablation branches retained only for historical diagnostics.
- `outputs/block13_obs_gap_*`
- `outputs/block13_transfer_*`
  Keep only the canonical diagnostic pair in active references:
  - `outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/`
  - `outputs/block13_obs_gap_no_delta_t_powerlog_tzone/`

## Canonical Storage Rule

From now on:

- article-facing docs, csv tables, and figures live in:
  - `paper_canonical_bundle/`
- article-facing zip models live in:
  - `paper_canonical_bundle/models/`
- original experiment folders remain untouched in `outputs/` and `models/`
  for reproducibility
