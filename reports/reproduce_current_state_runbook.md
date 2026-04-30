# Full Reproduction Runbook: Current Frozen State

Date: 2026-04-30

## Read This First

Before running any command, use:

- [reproduction_contours.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/reproduction_contours.md)

This runbook assumes your friend has the **full reproduction contour**, not only the active code contour.

That means all of the following must exist:

- [surrogate](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate)
- [training](C:/Users/user/Desktop/HVAC_DRL_MORL/training)
- [evaluation](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation)
- [envs](C:/Users/user/Desktop/HVAC_DRL_MORL/envs)
- [configs](C:/Users/user/Desktop/HVAC_DRL_MORL/configs)
- [data](C:/Users/user/Desktop/HVAC_DRL_MORL/data)
- [reports](C:/Users/user/Desktop/HVAC_DRL_MORL/reports)
- [models](C:/Users/user/Desktop/HVAC_DRL_MORL/models)
- [outputs](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs)
- [paper_canonical_bundle](C:/Users/user/Desktop/HVAC_DRL_MORL/paper_canonical_bundle)
- [draft/legacy_archive](C:/Users/user/Desktop/HVAC_DRL_MORL/draft/legacy_archive)

It also assumes:

- PowerShell
- repository root as current directory:
  - `C:\Users\user\Desktop\HVAC_DRL_MORL`
- working Python environment
- working BOPTEST endpoint
- testcase `bestest_air`

## Goal

This runbook reproduces the project up to the **current frozen state**:

1. `v3` direct-TSup surrogate
2. pure `v3` thermostatic benchmark context
3. `v3.5` inverse-calibrated surrogate
4. Block 1 rollout realism and zero-shot transfer failure
5. failed direct `v3.5` warm-start benchmark
6. hybrid `v3 + v3.5` benchmark
7. `lambda` sweep and canonical selection of `hybrid_l010`
8. final paper-facing bundle

## Success Criteria

The run is complete only when:

1. [block1_surrogate_final_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block1_surrogate_final_report.md) can be supported by regenerated artifacts
2. [block2_hybrid_surrogate_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hybrid_surrogate_report.md) can be rebuilt from regenerated artifacts
3. [bundle_manifest.json](C:/Users/user/Desktop/HVAC_DRL_MORL/paper_canonical_bundle/bundle_manifest.json) shows no missing files after rebuild

## Step 0. Validate the Runtime

Run this before everything else:

```powershell
python -c "import torch, gymnasium, stable_baselines3, pandas, numpy, requests; print('python_env_ok')"
```

Then check that BOPTEST is reachable:

```powershell
python -c "import requests; print(requests.get('http://web:8000/version', timeout=10).json())"
```

Do not continue until both commands work.

## Step 1. Collect the Hourly Direct-TSup Dataset for `v3`

```powershell
python .\data\collect_tsupply_data.py
```

Expected output file:

- `data/surrogate_v2/boptest_v2_tsupply.csv`

Expected meaning:

- `51,200` rows
- hourly step `3600 s`
- 4 seasons x 4 excitation policies

Quick check:

```powershell
Get-Item .\data\surrogate_v2\boptest_v2_tsupply.csv
```

## Step 2. Train the `v3` Backbone

```powershell
python .\surrogate\train_surrogate_backbone.py `
  --data data/surrogate_v2/boptest_v2_tsupply.csv `
  --output_dir outputs/surrogate_v2 `
  --epochs 500 `
  --batch_size 256 `
  --lr 1e-3 `
  --hidden_dim 64 `
  --patience 30 `
  --multi_horizons 2 4
```

Expected output files:

- `outputs/surrogate_v2/rc_node_backbone_best.pt`
- `outputs/surrogate_v2/rc_node_v2_best.pt`
- `outputs/surrogate_v2/train_history_v2.csv`

Canonicalize the model name expected by downstream scripts:

```powershell
Copy-Item .\outputs\surrogate_v2\rc_node_v2_best.pt .\outputs\surrogate_v2\rc_node_v3_tsupply.pt -Force
```

Quick check:

```powershell
Get-ChildItem .\outputs\surrogate_v2
```

You must see:

- `rc_node_v3_tsupply.pt`

## Step 3. Train and Benchmark the Pure `v3` Thermostatic Baseline

### 3.1 Train the pure `v3` thermostatic PPO

```powershell
python .\training\train_thermostatic.py `
  --surrogate-kind legacy_v3 `
  --surrogate-path outputs/surrogate_v2/rc_node_v3_tsupply.pt `
  --step-sec 900 `
  --comfort-low 21 `
  --comfort-high 24 `
  --save-name ppo_thermostatic
```

Expected output file:

- `models/ppo_thermostatic.zip`

### 3.2 Fine-tune that policy on live BOPTEST

```powershell
python .\training\finetune_tsup_policies_boptest.py `
  --agents thermostatic `
  --step-sec 900 `
  --episode-days 14 `
  --steps-thermostatic 120000 `
  --out-dir outputs/boptest_15min_policy_finetune `
  --thermostatic-model models/ppo_thermostatic.zip
```

Expected output file:

- `outputs/boptest_15min_policy_finetune/thermostatic_step900_finetuned.zip`

### 3.3 Benchmark the pure `v3` thermostatic result

```powershell
python .\evaluation\benchmark_bestest_air_article7_style.py `
  --step-sec 900 `
  --controllers thermostatic `
  --thermostatic-model outputs/boptest_15min_policy_finetune/thermostatic_step900_finetuned.zip `
  --output-dir outputs/bestest_air_article7_style_15min
```

Expected output file:

- `outputs/bestest_air_article7_style_15min/summary.csv`

This file is the canonical pure `v3` comparison point for the hybrid branch.

## Step 4. Build the Canonical 15-Minute Prepared Dataset Used by `v3.5`

This step uses the archived Block 1.2 helper script. It is still part of exact reproduction.

```powershell
python .\draft\legacy_archive\top_level\block_1_2_surrogate_rmse\data\prepare_surrogate_15min_dataset.py
```

Expected output files:

- `data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv`
- `outputs/block_1_2_surrogate_rmse/prepared_15min_dataset/dataset_summary.json`

This dataset is the canonical prepared 15-minute corpus for current Block 1.

## Step 5. Run `v3.5` Inverse Calibration

### 5.1 Temperature-alignment `v3.5` run

```powershell
python .\surrogate\calibrate_surrogate_v35.py --preset block1_3_15min_closed_loop
```

Expected output file:

- `outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop/calibration_summary_boptest_v35.json`

This should reproduce the strong temperature result:

- baseline RMSE around `0.3738 C`
- calibrated RMSE around `0.2319 C`

### 5.2 Canonical downstream `v3.5` backend

```powershell
python .\surrogate\calibrate_surrogate_v35.py --preset block1_15min_power_head_only
```

Expected output file:

- `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json`

This is the canonical `v3.5` backend used in downstream control comparisons.

## Step 6. Validate Block 1 Rollout Realism

```powershell
python .\evaluation\validate_surrogate_v35_rollout_prepared.py `
  --summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --out-dir outputs/surrogate_v35_rollout_prepared_15min_power_head_only
```

Expected output files:

- `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/v35_prepared_compare_summary.csv`
- `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_trace_21_24_vs_boptest.png`
- `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/hvac_power_trace_vs_boptest.png`
- `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/cumulative_energy_trace_vs_boptest.png`
- `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_violation_comparison.png`

At this point you have the canonical Block 1 rollout-realism artifacts.

## Step 7. Reproduce the Zero-Shot `v3.5` Transfer Failure

### 7.1 Train the diagnostic direct-`v3.5` thermostatic policy

Use the exact diagnostic observation configuration:

- `obs_ablation = no_delta_t`
- `power_feature_mode = clipped_log`
- `t_zone_feature_mode = comfort_centered`

```powershell
python .\training\train_thermostatic.py `
  --surrogate-kind v35_calibrated `
  --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --comfort-low 21 `
  --comfort-high 24 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode comfort_centered `
  --save-name ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone
```

Expected output file:

- `models/ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone.zip`

### 7.2 Validate closed-loop transfer on live BOPTEST

```powershell
python .\evaluation\validate_closed_loop_transfer_thermostatic_live.py `
  --thermostatic-model models/ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone.zip `
  --summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --duration-days 14 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode comfort_centered `
  --output-dir outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone
```

Expected output file:

- `outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/summary.csv`

### 7.3 Run the first-divergence diagnostic

```powershell
python .\evaluation\diagnose_thermostatic_obs_transfer_gap.py `
  --thermostatic-model models/ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone.zip `
  --summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --duration-days 14 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode comfort_centered `
  --output-dir outputs/block13_obs_gap_no_delta_t_powerlog_tzone
```

Expected output files:

- `outputs/block13_obs_gap_no_delta_t_powerlog_tzone/first_divergence_summary.csv`
- `outputs/block13_obs_gap_no_delta_t_powerlog_tzone/feature_drift_summary.csv`

Current frozen interpretation:

- `first_divergence_step = 1`
- dominant mismatch is `t_zone_norm`

At this point Block 1 is fully reproduced.

## Step 8. Reproduce the Failed Direct `v3.5` Warm-Start Benchmark

This is the canonical negative Block 2 baseline.

```powershell
python .\training\launch_thermostatic_warmstart_benchmark.py `
  --artifact-root outputs/block2_thermostatic_warmstart_utility
```

Expected output files:

- `outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv`
- `outputs/block2_thermostatic_warmstart_utility/scratch_eval/summary.csv`
- `outputs/block2_thermostatic_warmstart_utility/warmstart_eval/summary.csv`

This reproduces the result that:

- `scratch on BOPTEST` beats `direct v3.5 warm-start + BOPTEST fine-tune`

## Step 9. Train the Hybrid `v3 + v3.5` Policies

### 9.1 Train `lambda = 0.05`

```powershell
python .\training\train_thermostatic.py `
  --surrogate-kind hybrid_v3_v35 `
  --surrogate-path outputs/surrogate_v2/rc_node_v3_tsupply.pt `
  --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --comfort-low 21 `
  --comfort-high 24 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode raw `
  --lambda-temp-disagree 0.05 `
  --lambda-power-disagree 5e-5 `
  --save-name ppo_thermostatic_hybrid_v3_v35_l005
```

### 9.2 Train `lambda = 0.10`

```powershell
python .\training\train_thermostatic.py `
  --surrogate-kind hybrid_v3_v35 `
  --surrogate-path outputs/surrogate_v2/rc_node_v3_tsupply.pt `
  --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --comfort-low 21 `
  --comfort-high 24 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode raw `
  --lambda-temp-disagree 0.10 `
  --lambda-power-disagree 5e-5 `
  --save-name ppo_thermostatic_hybrid_v3_v35_l010
```

### 9.3 Train `lambda = 0.15`

```powershell
python .\training\train_thermostatic.py `
  --surrogate-kind hybrid_v3_v35 `
  --surrogate-path outputs/surrogate_v2/rc_node_v3_tsupply.pt `
  --surrogate-summary-json outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json `
  --step-sec 900 `
  --comfort-low 21 `
  --comfort-high 24 `
  --obs-ablation no_delta_t `
  --power-feature-mode clipped_log `
  --t-zone-feature-mode raw `
  --lambda-temp-disagree 0.15 `
  --lambda-power-disagree 5e-5 `
  --save-name ppo_thermostatic_hybrid_v3_v35_l015
```

Expected output files:

- `models/ppo_thermostatic_hybrid_v3_v35_l005.zip`
- `models/ppo_thermostatic_hybrid_v3_v35_l010.zip`
- `models/ppo_thermostatic_hybrid_v3_v35_l015.zip`

## Step 10. Benchmark the Hybrid Sweep

### 10.1 Benchmark `l005`

```powershell
python .\evaluation\benchmark_bestest_air_article7_style.py `
  --step-sec 900 `
  --controllers thermostatic `
  --thermostatic-model models/ppo_thermostatic_hybrid_v3_v35_l005.zip `
  --output-dir outputs/block2_thermostatic_hybrid_v3_v35_l005
```

### 10.2 Benchmark `l010`

```powershell
python .\evaluation\benchmark_bestest_air_article7_style.py `
  --step-sec 900 `
  --controllers thermostatic `
  --thermostatic-model models/ppo_thermostatic_hybrid_v3_v35_l010.zip `
  --output-dir outputs/block2_thermostatic_hybrid_v3_v35_l010
```

### 10.3 Benchmark `l015`

```powershell
python .\evaluation\benchmark_bestest_air_article7_style.py `
  --step-sec 900 `
  --controllers thermostatic `
  --thermostatic-model models/ppo_thermostatic_hybrid_v3_v35_l015.zip `
  --output-dir outputs/block2_thermostatic_hybrid_v3_v35_l015
```

### 10.4 Compare the sweep

```powershell
$v3 = Import-Csv .\outputs\bestest_air_article7_style_15min\summary.csv | Where-Object { $_.controller -eq 'thermostatic' } | Select-Object @{n='label';e={'v3_baseline'}},scenario,m_s,violation_pct,@{n='rmse';e={$_.rmse_22_c}},energy_kwh
$l005 = Import-Csv .\outputs\block2_thermostatic_hybrid_v3_v35_l005\summary.csv | Select-Object @{n='label';e={'hybrid_l005'}},scenario,m_s,violation_pct,@{n='rmse';e={$_.rmse_center_c}},energy_kwh
$l010 = Import-Csv .\outputs\block2_thermostatic_hybrid_v3_v35_l010\summary.csv | Select-Object @{n='label';e={'hybrid_l010'}},scenario,m_s,violation_pct,@{n='rmse';e={$_.rmse_center_c}},energy_kwh
$l015 = Import-Csv .\outputs\block2_thermostatic_hybrid_v3_v35_l015\summary.csv | Select-Object @{n='label';e={'hybrid_l015'}},scenario,m_s,violation_pct,@{n='rmse';e={$_.rmse_center_c}},energy_kwh
@($v3 + $l005 + $l010 + $l015) | Sort-Object scenario,label | Format-Table -AutoSize
```

Expected frozen conclusion:

- `hybrid_l010` is the canonical winner

## Step 11. Rebuild the Hybrid Snapshot Report

```powershell
python .\evaluation\build_hybrid_surrogate_snapshot.py `
  --hybrid-summary outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv `
  --hybrid-trace-dir outputs/block2_thermostatic_hybrid_v3_v35_l010/traces `
  --pi-summary outputs/block2_bestest_air_15min_thermostatic_v35/summary.csv `
  --v3-summary outputs/bestest_air_article7_style_15min/summary.csv `
  --warmstart-summary outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv
```

Expected regenerated files:

- `reports/block2_hybrid_surrogate_report.md`
- `reports/block2_hybrid_surrogate_metrics.csv`
- `reports/figures/*.png`

## Step 12. Rebuild the Canonical Article Bundle

```powershell
python .\evaluation\build_paper_canonical_bundle.py
```

Expected file:

- `paper_canonical_bundle/bundle_manifest.json`

Expected final check:

- `copied = 31`
- `missing = []`

## Fast Mode: Rebuild From Existing Frozen Outputs Only

If your friend already has all frozen models and outputs and only wants the article-facing package, they may skip training and only run:

```powershell
python .\evaluation\build_hybrid_surrogate_snapshot.py
python .\evaluation\build_paper_canonical_bundle.py
```

This fast mode is valid only if all canonical source outputs already exist.

## Final Stop Condition

Your friend should stop only when all of the following are true:

1. `outputs/surrogate_v35_inverse_boptest_15min_power_head_only/` exists
2. `outputs/surrogate_v35_rollout_prepared_15min_power_head_only/` exists
3. `outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/` exists
4. `outputs/block13_obs_gap_no_delta_t_powerlog_tzone/` exists
5. `outputs/block2_thermostatic_warmstart_utility/` exists
6. `outputs/block2_thermostatic_hybrid_v3_v35_l010/` exists
7. `reports/block1_surrogate_final_report.md` is supported by regenerated artifacts
8. `reports/block2_hybrid_surrogate_report.md` is supported by regenerated artifacts
9. `paper_canonical_bundle/bundle_manifest.json` shows no missing files

That is the exact current frozen state.
