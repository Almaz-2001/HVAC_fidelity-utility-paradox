# Block 1.2 Collector Notes

## Purpose

`collect_surrogate_15min_boptest_data.py` is the canonical live BOPTEST collector dedicated to Block 1.2.

It exists because `prepare_surrogate_15min_dataset.py` only repacks already generated benchmark traces. That was enough for the first 15-minute bootstrap run, but not enough for a broader surrogate corpus.

## What this collector changes

- collects directly from live BOPTEST instead of existing trace CSV files
- uses `900 s` step by default
- writes a canonical combined dataset plus per-episode CSV files
- stores richer metadata needed for later audit and retraining

## Default exploration layout

- seasons: `winter, spring, summer, autumn`
- policies: `random, heat, cool, mixed, thermostatic_noise, pulse`
- steps per episode: `2016` (`21 days` at `15 min`)

This is intentionally broader than the current 15-minute prepared dataset, which is heating-window focused and controller-biased.

## Recommended first run

```bash
python block_1_2_surrogate_rmse/data/collect_surrogate_15min_boptest_data.py --write-train-subset
```

By default the collector now uses `--profile heating_focus`, not the old broad exploration profile.

That matters because the broad profile produced temperatures such as `6 C` and `43 C`, which are useful for stress-testing but harmful for the target metric of low rollout RMSE around the operating comfort manifold.

## Main practical knobs

- `--profile heating_focus`
  Focused excitation around the heating control manifold.
- `--safe-t-zone-min-c 17.5 --safe-t-zone-max-c 28.0`
  Online safety envelope during collection.
- `--write-train-subset`
  Writes a filtered train-ready subset next to the raw collected CSV.
- `--train-t-zone-min-c 17 --train-t-zone-max-c 29`
  Keeps temperature transitions closer to the relevant control region.
- `--train-abs-delta-t-max-c 2.5`
  Removes the sharpest temperature jumps from the training subset.

## Recommended next training run

```bash
python block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py --data data/block_1_2_surrogate_rmse/boptest_block12_15min_collected_train_subset.csv --run-name collected_15min_focus --epochs 700 --hidden-dim 96 --lr 7.5e-4 --patience 40 --multi-horizons 4,8,16 --validate-safety
```
