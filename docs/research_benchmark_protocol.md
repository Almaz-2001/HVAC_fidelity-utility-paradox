# Research Benchmark Protocol

This protocol defines the main evidence chain for the project:

1. calibrated surrogate fidelity
2. inverse identification of physical parameters and dynamics alignment
3. unified BOPTEST-oriented controller benchmark
4. downstream control impact of surrogate quality

## Current Canonical Builder

Use:

```powershell
python evaluation/build_research_benchmark.py
```

This script does not run heavy simulations. It consolidates already validated outputs into paper-ready tables and figures under:

```text
results/research_benchmark/
```

## Canonical Tables

- `surrogate_rollout_benchmark.csv`
  Raw vs calibrated surrogate rollout realism with canonical `1-step`, `4h`, `8h`, and `24h` temperature RMSE columns plus bias, power RMSE, and `C_zon` error.

- `inverse_calibration_benchmark.csv`
  Stage A/B/C inverse problem snapshot and best multistart prior.

- `controller_benchmark.csv`
  Unified BOPTEST comparison of implemented controllers.

- `experiment_matrix.csv`
  Project-level status map of implemented vs pending experiments.

## Canonical Figures

- `surrogate_fidelity_dashboard.png`
  Rollout realism, power realism, and inverse-calibration fit in one figure.

- `controller_benchmark_dashboard.png`
  Bar-chart comparison of PI, Thermostatic PPO, and HDRL.

- `controller_tradeoff_scatter.png`
  Energy vs comfort trade-off with bubble size proportional to `m_s`.

## What Counts As Implemented

- Surrogate fidelity on the current v3.5 line
- Inverse calibration with explicit `C_zon`
- Unified controller comparison for `PI`, `Thermostatic PPO`, and `HDRL`
- MORL pipeline infrastructure

## What Is Still Pending

- Retraining controllers on the calibrated surrogate and quantifying transfer gains
- Revalidating MORL/PPO as a strong downstream controller in the same benchmark
- Replacing snapshot-level MORL progress with a canonical yearly MORL benchmark output

There is also one concrete engineering blocker:

- the current calibrated `v3.5` checkpoint is not a drop-in replacement for the older direct-TSup surrogate backend, so downstream control retraining on the calibrated twin requires a backend/model adapter first

## Scientific Reading

This benchmark is intentionally not framed as a pure reproduction of Article 7 or Article 22.
Those papers remain external references.

The project’s own benchmark asks a different question:

`How does the fidelity of a calibrated surrogate digital twin affect the quality and transferability of downstream HVAC control policies?`
