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

- `training/launch_thermostatic_warmstart_benchmark.py`
  Reproducibility for the negative direct `v3.5` warm-start result.
- `outputs/block2_thermostatic_warmstart_utility/`
  Frozen negative baseline for Block 2.

## Archived Under `draft/legacy_archive`

These paths were moved out of the active working set and now live under `draft/legacy_archive/`.

- `draft/legacy_archive/top_level/block_1_2_surrogate_rmse/`
  Historical Block 1.2 experimentation package.
- `draft/legacy_archive/top_level/block_1_3_closed_loop_transfer/`
  Top-level legacy experiment stub.
- `draft/legacy_archive/training/`
  Frozen utility scripts:
  - `run_bestest_air_15min.py`
  - `sweep_weights.py`
  - `train_ppo.py`
  - `train_ppo_parallel.py`
- `draft/legacy_archive/evaluation/`
  Frozen evaluation utilities:
  - `benchmark_speed.py`
  - `evaluate_policy.py`
  - `plot_rigorous_results.py`
- `draft/legacy_archive/outputs/current/`
  Older canonical-output snapshot. Superseded by `paper_canonical_bundle/`.
- `draft/legacy_archive/outputs/block2_thermostatic_hybrid_v3_v35/`
  Pre-sweep hybrid result. Superseded by `outputs/block2_thermostatic_hybrid_v3_v35_l010/`.
- `draft/legacy_archive/outputs/block2_thermostatic_hybrid_v3_v35_l005/`
- `draft/legacy_archive/outputs/block2_thermostatic_hybrid_v3_v35_l015/`
  Hybrid sweep history retained outside the active path after selecting `l010`.
- `draft/legacy_archive/outputs/block2_thermostatic_no_*`
- `draft/legacy_archive/outputs/block2_thermostatic_with_power/`
- `draft/legacy_archive/outputs/block2_thermostatic_causal_smooth/`
  Feature-ablation branches retained only for historical diagnostics.
- `draft/legacy_archive/outputs/block13_obs_gap_*`
- `draft/legacy_archive/outputs/block13_transfer_*`
- `draft/legacy_archive/outputs/block13_closed_loop_transfer_causal_smooth/`
  Historical transfer-gap diagnostics archived after selecting the canonical pair.
- `draft/legacy_archive/logs/logs/`
  Runtime logs removed from the active root.

Keep only the canonical diagnostic pair in active references:

- `outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/`
- `outputs/block13_obs_gap_no_delta_t_powerlog_tzone/`

## Canonical Storage Rule

From now on:

- article-facing docs, csv tables, and figures live in:
  - `paper_canonical_bundle/`
- article-facing zip models live in:
  - `paper_canonical_bundle/models/`
- active reproducibility paths remain in `outputs/` and `models/`
- archived historical branches live in `draft/legacy_archive/`
