# Block 1.2 Run Order

## Objective

- keep `C_zon error` near the current `3.07%`
- move the active surrogate branch to `15 min`
- reduce surrogate rollout RMSE toward `0.2 C`
- use comfort-band logic `21-24 C` instead of fixed-target `22 C`

## Short command path

1. Prepare the bootstrap 15-minute dataset:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage prepare
```

2. Train the first 15-minute backbone on the prepared dataset:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared
```

3. Train the prepared dataset again, but now select the best checkpoint by rollout RMSE instead of batch val loss:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared_rollout
```

4. Run the long-horizon rollout-select variant with lower LR and weighted emphasis on 8-step and 16-step rollout:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_prepared_rollout_long
```

5. Collect a focused live BOPTEST dataset centered on the comfort band:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage collect
```

6. Build a hybrid anchor dataset that keeps the strong prepared manifold and adds filtered collected transitions:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage build_hybrid
```

7. Retrain on the collected focused subset only when you specifically want to test broad exploration coverage:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_collected
```

8. Retrain on the hybrid anchor dataset for the main Block 1.2 branch:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage train_hybrid
```

9. Re-run `v3.5` calibration on a real `15 min` episode only after a better 15-minute backbone is found:

```bash
python block_1_2_surrogate_rmse/run_surrogate_workflow.py --stage calibrate_15min
```

## Decision rule after each stage

1. If `prepared_15min_rollout_long` beats `prepared_15min_rollout_select`, make the long-horizon rollout preset the default Block 1.2 trainer.
2. If `collected_15min_focus` does not beat `prepared_15min_rollout_long`, do not calibrate `v3.5` on it.
3. If `hybrid_15min_anchor` beats `prepared_15min_rollout_long`, make hybrid the active Block 1.2 training base.
4. If `15 min v3.5` hurts `C_zon`, reject the run even if one-step RMSE improves.
5. Only try a new Stage C branch after the `15 min` backbone itself is cleaner.
