# HVAC_DRL_MORL

This repository is now organized around one primary research path:

`surrogate-calibrated MORL/PPO -> optional BOPTEST fine-tune -> BOPTEST yearly validation`

The central idea is not a generic PPO sandbox. The core workflow is a MORL controller that is first pretrained on the calibrated surrogate digital twin and then validated on BOPTEST.

## Primary Entry Point

Use:

```powershell
python training/run_morl_surrogate_pipeline.py --mode full
```

This runner is the main project entry point for MORL experiments. It executes:

1. surrogate pretraining
2. BOPTEST fine-tuning
3. yearly BOPTEST validation

The ERAM branch is also supported:

```powershell
python training/run_morl_surrogate_pipeline.py --mode full_eram
```

## MORL Config Set

The dedicated MORL config set lives in:

- [env.yaml](configs/morl_surrogate_ppo/env.yaml)
- [agent.yaml](configs/morl_surrogate_ppo/agent.yaml)
- [train.yaml](configs/morl_surrogate_ppo/train.yaml)
- [pipeline.yaml](configs/morl_surrogate_ppo/pipeline.yaml)

These configs define the default surrogate-first MORL setup and the artifact layout under `outputs/morl_surrogate_ppo/`.

## Pipeline Stages

- `pretrain`: MORL/PPO on the surrogate backend
- `eram_pretrain`: vector-reward MORL with adversarial weight updates on the surrogate
- `finetune`: transfer the pretrained model to BOPTEST
- `eval`: yearly BOPTEST validation
- `full`: `pretrain -> finetune -> eval`
- `full_eram`: `eram_pretrain -> finetune -> eval`

Examples:

```powershell
python training/run_morl_surrogate_pipeline.py --mode pretrain
python training/run_morl_surrogate_pipeline.py --mode finetune
python training/run_morl_surrogate_pipeline.py --mode eval
```

Override the config set if needed:

```powershell
python training/run_morl_surrogate_pipeline.py --config-dir configs/morl_surrogate_ppo --mode full --seed 42
```

## Artifact Layout

Each seed gets its own folder:

```text
outputs/morl_surrogate_ppo/
  seed42/
    pipeline_manifest.json
    pretrain/
    eram_pretrain/
    finetune_boptest/
    yearly_eval/
```

This layout is intentional. MORL artifacts should no longer be scattered across unrelated output folders.

## Secondary Paths

Thermostatic PPO, HDRL, surrogate inverse calibration, and Article-style comparison scripts remain in the repository, but they are secondary lines of work. The primary line for the project is the MORL surrogate-pretrain pipeline above.

## Detailed Workflow

See [docs/morl_surrogate_ppo_workflow.md](docs/morl_surrogate_ppo_workflow.md).
