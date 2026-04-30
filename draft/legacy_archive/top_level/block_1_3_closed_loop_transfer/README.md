# Block 1.3: Closed-Loop Transfer Gap

Goal:

- keep the explicit `C_zon` identification from the current 15-minute `v3.5` backend
- reduce the remaining closed-loop transfer gap between the calibrated twin and live BOPTEST
- validate this gap specifically on the `thermostatic` controller before reopening Block 2

Active code paths:

- calibration backend:
  - [inverse_problem_boptest_v35.py](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate/inverse_problem_boptest_v35.py)
- offline prepared rollout validation:
  - [validate_surrogate_v35_rollout_prepared.py](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation/validate_surrogate_v35_rollout_prepared.py)
- live closed-loop thermostatic transfer validation:
  - [validate_closed_loop_transfer_thermostatic_live.py](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation/validate_closed_loop_transfer_thermostatic_live.py)

Canonical commands:

```powershell
python surrogate/inverse_problem_boptest_v35.py --preset block1_3_15min_closed_loop
```

```powershell
python evaluation/validate_surrogate_v35_rollout_prepared.py --summary-json outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop/calibration_summary_boptest_v35.json --out-dir outputs/surrogate_v35_rollout_prepared_15min_block13_closed_loop
```

```powershell
python evaluation/validate_closed_loop_transfer_thermostatic_live.py --thermostatic-model models/ppo_thermostatic_v35_15min_power_head_only.zip --summary-json outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop/calibration_summary_boptest_v35.json --output-dir outputs/block_1_3_closed_loop_transfer_thermostatic_live_block13
```

Interpretation rule:

- if offline rollout improves but live thermostatic transfer does not, the problem is still controller-relevant state distortion rather than plain one-step fit
- Block 2 should remain paused until the thermostatic closed-loop gap is materially reduced
