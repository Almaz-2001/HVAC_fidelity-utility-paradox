# Minimum Paper Suite Next Steps

## Calibrated-Twin Launchers

Short Python entry points for the still-open calibrated-twin stage and immediate MORL transfer steps:

```powershell
python training/launch_hdrl_retrain_on_calibrated_twin.py
python training/launch_morl_pretrain_on_calibrated_twin.py
python training/launch_morl_finetune_on_boptest.py
python training/launch_morl_yearly_boptest_eval.py
```

## Step 1: Canonical 4h/8h/24h surrogate rollout table

Why: Adds the missing 8h horizon to the canonical surrogate fidelity source output.

```powershell
python evaluation/validate_surrogate_v35_rollout_live.py --horizons 1 4 8 24 --out-dir outputs/surrogate_v35_rollout_live
```

## Step 2: Refresh canonical surrogate benchmark snapshot

Why: Rebuilds the paper-ready surrogate table and figures after the new rollout validation.

```powershell
python evaluation/build_research_benchmark.py
```

## Step 3: Train honest M.1 thermostatic baseline

Why: Predictive-information ablation cannot start without M.1.

```powershell
python training/train_thermostatic.py --article22-variant m1 --save-name ppo_thermostatic_article22_m1
```

## Step 4: Benchmark M.1 at 15 min

Why: Creates the first comparable M.1 result on the same heating windows.

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers thermostatic --thermostatic-model models/ppo_thermostatic_article22_m1.zip --output-dir outputs/article22_m1_benchmark_15min
```

## Step 5: Fine-tune M.1 on BOPTEST 15 min

Why: Needed for transfer ablation on the same controller family.

```powershell
python training/finetune_tsup_policies_boptest.py --agents thermostatic --step-sec 900 --thermostatic-model models/ppo_thermostatic_article22_m1.zip --out-dir outputs/boptest_15min_policy_finetune_m1
```

## Step 6: Benchmark fine-tuned M.1 at 15 min

Why: Closes the surrogate->BOPTEST transfer comparison for M.1.

```powershell
python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers thermostatic --thermostatic-model outputs/boptest_15min_policy_finetune_m1/thermostatic_step900_finetuned.zip --output-dir outputs/article22_m1_benchmark_15min_finetuned
```

## Step 7: Refresh canonical benchmark outputs

Why: Rebuilds the unified paper-ready tables and figures after new controller artifacts are produced.

```powershell
python evaluation/build_research_benchmark.py
```

## Step 8: Refresh minimum paper suite status

Why: Updates the project status sheet after the canonical benchmark is rebuilt.

```powershell
python evaluation/build_minimum_paper_suite.py
```
