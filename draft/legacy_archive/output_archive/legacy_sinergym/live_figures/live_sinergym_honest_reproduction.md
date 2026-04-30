# Legacy Sinergym Honest Reproduction

## Position

- Current live legacy reproduction differs from the article-era figures.
- Article-era figures were not fully generated from the current CSV pipeline.
- We rebuilt an honest live evaluation stack that reads real CSV outputs instead of hard-coded plotting dictionaries.

## What This Report Uses

- Source directory: `/app/outputs/legacy_sinergym`
- Live PPO evaluation CSVs from `seedXX/eval/ppo_eval.csv`
- Live baseline CSVs from `seedXX/baselines/*.csv`
- Live Pareto sweep CSVs from `pareto/*/seed*/eval/ppo_eval.csv` and `pareto/pareto_results.csv`

## Why The Legacy Article Figures Are Not A Faithful Live Reproduction

- `evaluation/visualize_results.py` uses hard-coded report dictionaries rather than recomputing the figures from current run outputs.
- The current legacy Pareto sweep does not produce a meaningful front: several weight settings collapse to the same live point.
- A rule-based baseline is now available in the live CSV pipeline and is included in the baseline comparison.

## Current Live Baseline Metrics

| Policy | Mean HVAC power (W) | Mean comfort penalty | 95th percentile comfort penalty | Mean zone temp (C) |
| --- | ---: | ---: | ---: | ---: |
| Learned PPO | 499.80 | 11.003 | 89.141 | 24.448 |
| Rule-based | 518.24 | 10.943 | 89.492 | 24.588 |
| Random | 507.11 | 12.014 | 86.707 | 23.548 |
| Zero-hold | 508.88 | 12.158 | 86.574 | 23.447 |

## Current Live Pareto Observation

No live Pareto summary CSV was found.

## Honest Interpretation

- The current legacy Sinergym branch can be reproduced honestly with the live CSV pipeline.
- That honest reproduction should be reported as distinct from the older article-era figure package.
- The correct wording is:
  - current live legacy reproduction differs from article figures
  - article-era figures were not fully generated from current CSV pipeline
  - we rebuilt an honest live evaluation stack

## Regeneration Commands

```bash
python legacy_sinergym_main.py --mode baselines --seeds 42,43,44 --baseline-steps 2000
python evaluation/visualize_results_live_sinergym.py --base-dir /app/outputs/legacy_sinergym --out-dir /app/outputs/legacy_sinergym/live_figures
```