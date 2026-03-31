# Paper Protocol

This protocol organizes the experiments in the order:

1. `standard BOPTEST controller`
2. `thermostatic baseline`
3. `HDRL`
4. `MORL`
5. `safe MORL`

All paths below are relative to the project root.

## 0. Standard BOPTEST Controller

Validate the built-in testcase PI controller without overrides:

```bash
PYTHONPATH=/app python3 evaluation/standard_controller_baseline.py
```

Primary files:

- `outputs/standard_controller_yearly_summary.csv`
- `outputs/standard_controller_scenario_*.csv`

Use for tables:

- Table `Reference Controller Baseline`
  - columns: `Scenario, rmse_22c, viol_21_25_pct, schedule_viol_pct, schedule_m_s, energy_kwh`
  - source: `outputs/standard_controller_yearly_summary.csv`

Use for figures:

- Figure `Reference Controller Comfort / Energy`
  - x-axis: `name`
  - y-axis: `energy_kwh` and `schedule_m_s`
  - source: `outputs/standard_controller_yearly_summary.csv`

## 1. Thermostatic Baseline

Train:

```bash
python training/train_thermostatic.py
```

Validate on BOPTEST:

```bash
PYTHONPATH=/app python3 evaluation/eval_thermostatic.py
```

Primary files:

- `outputs/thermostatic_yearly_summary.csv`
- `outputs/thermostatic_scenario_*.csv`

Use for tables:

- Table `Baseline Annual Accuracy`
  - columns: `Scenario, rmse, mae, within_1, within_05, energy`
  - source: `outputs/thermostatic_yearly_summary.csv`

Use for figures:

- Figure `Scenario-wise RMSE`
  - x-axis: `name`
  - y-axis: `rmse`
  - source: `outputs/thermostatic_yearly_summary.csv`
- Figure `Representative Summer Trace`
  - x-axis: `step`
  - y-axis: `t_zone`, `t_amb`, `t_supply`
  - source: `outputs/thermostatic_scenario_Jul_Summer.csv`
- Figure `Representative Winter Trace`
  - x-axis: `step`
  - y-axis: `t_zone`, `t_amb`, `t_supply`
  - source: `outputs/thermostatic_scenario_Jan_Winter.csv`

## 2. HDRL

Train:

```bash
python training/train_hdrl.py
```

Validate on BOPTEST:

```bash
PYTHONPATH=/app python3 evaluation/yearly_validation_hdrl.py
```

Primary files:

- `outputs/hdrl_yearly_summary.csv`
- `outputs/hdrl_scenario_*.csv`

Use for tables:

- Table `HDRL Seasonal Control`
  - columns: `Scenario, viol, energy, ms, winter_pct, emergency_pct`
  - source: `outputs/hdrl_yearly_summary.csv`

Use for figures:

- Figure `HDRL Energy vs Safety`
  - x-axis: `energy`
  - y-axis: `ms`
  - source: `outputs/hdrl_yearly_summary.csv`
- Figure `Meta-controller Usage`
  - stacked bars: `winter_pct`, `emergency_pct`, `100 - winter_pct - emergency_pct`
  - source: `outputs/hdrl_yearly_summary.csv`

## 3. MORL

Train one seed:

```bash
SEED=42 python main.py
```

Evaluate:

```bash
MODE=eval SEED=42 python main.py
```

Primary files:

- `outputs/seed42/morl_log.csv`
- `outputs/seed42/eval/ppo_eval.csv`

Use for tables:

- Table `MORL Reward Decomposition`
  - columns: mean `reward_scalar`, `comfort`, `energy`, `zone_temp`, `hvac_power`
  - source: `outputs/seed42/eval/ppo_eval.csv`

Use for figures:

- Figure `Training Trade-off Curve`
  - x-axis: `step`
  - y-axis: rolling `comfort` and `energy`
  - source: `outputs/seed42/morl_log.csv`
- Figure `Evaluation Temperature / Power Trace`
  - x-axis: `step`
  - y-axis: `zone_temp`, `hvac_power`
  - source: `outputs/seed42/eval/ppo_eval.csv`

## 4. Safe MORL

Evaluate without safety:

```bash
PYTHONPATH=/app python3 evaluation/eval_safe_morl.py --model /app/outputs/seed42/models/ppo_model.zip --no_safety
```

Evaluate with safety:

```bash
PYTHONPATH=/app python3 evaluation/eval_safe_morl.py --model /app/outputs/seed42/models/ppo_model.zip
```

Multi-seed comparison:

```bash
PYTHONPATH=/app python3 evaluation/eval_multi_seed.py --model /app/outputs/seed42/models/ppo_model.zip --seeds 42 43 44
```

Primary files:

- `outputs/eval_safe_morl/eval_safe_morl.csv`
- `outputs/eval_safe_morl/summary.csv`
- `outputs/eval_multi_seed/all_results_raw.csv`
- `outputs/eval_multi_seed/summary.csv`

Use for tables:

- Table `Safe MORL Single-Run`
  - columns: `m_s, r_time, r_sev, violation_pct, energy_kwh, acceptance_rate`
  - source: `outputs/eval_safe_morl/summary.csv`
- Table `Safe MORL Multi-Seed`
  - columns: aggregated means/std from `summary.csv`
  - source: `outputs/eval_multi_seed/summary.csv`

Use for figures:

- Figure `Safety Filter Intervention Timeline`
  - x-axis: `step`
  - y-axis: `t_zone`
  - color/style by `source`
  - source: `outputs/eval_safe_morl/eval_safe_morl.csv`
- Figure `PPO vs PPO+SF Comparison`
  - bars or points for `m_s`, `violation_pct`, `energy_kwh`
  - source: `outputs/eval_multi_seed/all_results_raw.csv`

## Minimal Table Set For The Paper

Use these five tables:

1. `Reference controller baseline`
2. `Thermostatic baseline annual accuracy`
3. `HDRL seasonal safety/energy summary`
4. `MORL reward decomposition and operating point`
5. `Safe MORL single-seed + multi-seed comparison`

## Minimal Figure Set For The Paper

Use these six figures:

1. `Thermostatic scenario-wise RMSE`
2. `Thermostatic representative July trace`
3. `HDRL energy vs m_s`
4. `MORL training comfort/energy curve`
5. `MORL evaluation temperature/power trace`
6. `Safe MORL intervention timeline`

## Automation Helper

After the CSV files are generated, build summary plots/tables with:

```bash
python evaluation/build_paper_artifacts.py --outputs_dir outputs --seed 42
```
