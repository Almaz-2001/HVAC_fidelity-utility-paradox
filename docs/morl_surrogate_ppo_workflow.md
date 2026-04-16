# MORL Surrogate PPO Workflow

## Research Center

The project is organized around one main hypothesis:

`A MORL/PPO controller pretrained on the calibrated surrogate should transfer better to BOPTEST than a controller trained only in the expensive online loop.`

This file defines the operational workflow for that line of work.

## Stage 1: Surrogate Pretrain

Entry point:

- [train_morl_surrogate.py](../training/train_morl_surrogate.py)

Purpose:

- learn a stable MORL/PPO policy on the fast surrogate backend
- debug reward shaping cheaply
- avoid burning BOPTEST time on cold-start exploration

Typical launch:

```powershell
python training/run_morl_surrogate_pipeline.py --mode pretrain
```

## Stage 1b: ERAM Pretrain

Entry point:

- [train_morl_eram.py](../training/train_morl_eram.py)

Purpose:

- keep vector-reward information explicit
- adapt objective weights before BOPTEST transfer
- save final ERAM weights for downstream fine-tuning

Typical launch:

```powershell
python training/run_morl_surrogate_pipeline.py --mode eram_pretrain
```

## Stage 2: BOPTEST Fine-Tune

Entry point:

- [finetune_morl_boptest.py](../training/finetune_morl_boptest.py)

Purpose:

- transfer the pretrained MORL policy to the real BOPTEST loop
- preserve observation/action compatibility with the surrogate-pretrained model
- optionally inject ERAM objective weights into the BOPTEST environment

Typical launch:

```powershell
python training/run_morl_surrogate_pipeline.py --mode finetune
```

## Stage 3: Yearly Validation

Entry point:

- [yearly_validation_morl.py](../evaluation/yearly_validation_morl.py)

Purpose:

- test the transferred policy on yearly BOPTEST anchor windows
- compute comfort, energy, and safety-oriented metrics
- store paper-ready CSV outputs in one dedicated folder

Typical launch:

```powershell
python training/run_morl_surrogate_pipeline.py --mode eval
```

## Unified Runner

Primary runner:

- [run_morl_surrogate_pipeline.py](../training/run_morl_surrogate_pipeline.py)

Why it exists:

- one Python entry point instead of several manual commands
- one artifact tree for every seed
- one seed manifest describing the resolved paths
- one explicit center of gravity for the whole repository

Supported modes:

- `pretrain`
- `eram_pretrain`
- `finetune`
- `eval`
- `full`
- `full_eram`

## Configs

Dedicated MORL configs:

- [env.yaml](../configs/morl_surrogate_ppo/env.yaml)
- [agent.yaml](../configs/morl_surrogate_ppo/agent.yaml)
- [train.yaml](../configs/morl_surrogate_ppo/train.yaml)
- [pipeline.yaml](../configs/morl_surrogate_ppo/pipeline.yaml)

These configs should be treated as the default MORL workspace. The older root-level `configs/*.yaml` files remain for compatibility, but they are no longer the recommended starting point for new MORL experiments.

## Output Layout

Each seed has a dedicated artifact root:

```text
outputs/morl_surrogate_ppo/seed42/
  pipeline_manifest.json
  pretrain/
    config_snapshot.json
    models/ppo_model.zip
  eram_pretrain/
    config_snapshot.json
    final_eram_weights.json
    models/ppo_model.zip
  finetune_boptest/
    config_snapshot.json
    models/ppo_model.zip
  yearly_eval/
    morl_yearly_summary.csv
    morl_scenario_*.csv
```

## What Is Secondary

These paths remain useful, but they are no longer the project center:

- thermostatic PPO
- HDRL
- legacy Sinergym experiments
- Article 7 reproduction scripts
- Article 22-style ablations outside the MORL line

They should support the MORL paper story, not define it.
