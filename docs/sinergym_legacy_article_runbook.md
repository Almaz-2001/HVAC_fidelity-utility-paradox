# Legacy Sinergym Runbook for ArticleRus

This runbook restores the old `SinergymBackend` workflow separately from the
current direct-`Tsup` BOPTEST pipeline.

## 1. Prerequisites

The legacy branch depends on a working Sinergym + EnergyPlus installation.
The current repository does not vendor those binaries.

Required at runtime:

- `sinergym`
- `stable-baselines3`
- `gymnasium`
- a compatible EnergyPlus installation visible to Sinergym

Quick sanity check inside the target environment:

```bash
python -c "import sinergym, gymnasium as gym; env=gym.make('Eplus-5Zone-hot-continuous-v1', disable_env_checker=True); print(env.action_space, env.observation_space); env.close()"
```

## 2. Legacy configs

The restored legacy config set lives in:

- `configs/legacy_sinergym/env.yaml`
- `configs/legacy_sinergym/agent.yaml`
- `configs/legacy_sinergym/train.yaml`

It writes outputs to:

- `/app/outputs/legacy_sinergym`

## 3. Main reproduction commands

### Single-seed training

```bash
python legacy_sinergym_main.py --mode train --seed 42
```

### Single-seed evaluation

```bash
python legacy_sinergym_main.py --mode eval --seed 42 --eval-steps 2000
```

### Multi-seed legacy run

```bash
python legacy_sinergym_main.py --mode multiseed --seeds 42,43,44 --steps 500000 --eval-steps 2000
```

### Generate baselines only

This creates `random.csv`, `rule_based.csv`, and `zero_hold.csv` without
retraining PPO:

```bash
python legacy_sinergym_main.py --mode baselines --seeds 42,43,44 --baseline-steps 2000
```

### Pareto sweep from the legacy article branch

```bash
python legacy_sinergym_main.py --mode pareto --seeds 42,43,44 --steps 500000 --eval-steps 2000
```

This writes Pareto artifacts under:

- `/app/outputs/legacy_sinergym/pareto`

## 4. Article-style figures

The article figure generator is still:

```bash
FIGURE_OUTPUT_DIR=/app/outputs/legacy_sinergym/figures python evaluation/visualize_results.py
```

Important:

- `evaluation/visualize_results.py` contains fixed report-level numbers for the
  article figures.
- It reproduces the article plots, but it does not recompute every figure live
  from current training runs.

## 5. Honest live figures from current CSV outputs

To build figures from the actual legacy Sinergym outputs instead of hard-coded
article values, use:

```bash
python evaluation/visualize_results_live_sinergym.py --base-dir /app/outputs/legacy_sinergym --out-dir /app/outputs/legacy_sinergym/live_figures
```

This writes:

- `/app/outputs/legacy_sinergym/live_figures/live_fig1_baseline_comparison.png`
- `/app/outputs/legacy_sinergym/live_figures/live_fig2_representative_trajectories.png`
- `/app/outputs/legacy_sinergym/live_figures/live_fig3_pareto_tradeoff.png`
- `/app/outputs/legacy_sinergym/live_figures/live_baseline_summary.csv`
- `/app/outputs/legacy_sinergym/live_figures/live_pareto_summary.csv`
- `/app/outputs/legacy_sinergym/live_figures/live_sinergym_summary.txt`

## 6. Expected outputs

Core legacy run outputs:

- `/app/outputs/legacy_sinergym/seed42/morl_log.csv`
- `/app/outputs/legacy_sinergym/seed42/eval/ppo_eval.csv`
- `/app/outputs/legacy_sinergym/seed42/models/ppo_model.zip`
- `/app/outputs/legacy_sinergym/pareto/pareto_results.csv`
- `/app/outputs/legacy_sinergym/pareto/pareto_front.png`
- `/app/outputs/legacy_sinergym/figures/fig1_ms_comparison.png`
- `/app/outputs/legacy_sinergym/figures/fig6_pareto_front.png`

## 7. Scope note

This branch restores the old Sinergym/MORL-PPO path used for the article-era
results. It is intentionally separate from the current BOPTEST/direct-`Tsup`
pipeline and should be treated as a legacy reproduction workflow.
