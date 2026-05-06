# Q1 Article Full Progress Report

Date: 2026-05-06

## Purpose

This report consolidates the current paper-ready state of the project before opening Block 3.

It is written as a structured scientific narrative rather than a runbook. The goal is to document:

1. the surrogate-design logic behind `v3`, `v3.5`, and the hybrid backend
2. the complete Block 1 surrogate-fidelity result, including failed zero-shot transfer
3. the complete Block 2 downstream-control result, including failed and successful controller promotions
4. the exact boundary of the current claim

The central paper claim remains:

**A calibrated digital twin with explicit structural building physics is most useful as a physical regularizer for control-oriented surrogate learning, rather than as a direct standalone RL-training environment.**

---

## I. Research Structure

The project is organized into two completed blocks and one next block.

### Block 1. Surrogate Fidelity

Objective:

- build a surrogate that is physically defensible
- calibrate it against BOPTEST
- quantify rollout realism
- quantify zero-shot transfer limits

### Block 2. Downstream Control Utility

Objective:

- test whether the surrogate helps real downstream control
- separate successful and unsuccessful controller-family promotion paths
- determine whether physical regularization is controller-family invariant or controller-family specific

### Block 3. Cross-Case Transferability

Status:

- not yet open
- Block 1 and Block 2 are now sufficiently closed to justify opening it next

---

## II. Block 1: Surrogate Fidelity

Block 1 contains three main surrogate objects:

1. `v3`: comfort-oriented direct-TSup surrogate
2. `v3.5`: explicit-physics calibrated surrogate
3. `hybrid_v3_v35`: split-role runtime backend that combines both

### II.1. Comfort-Oriented Surrogate `v3`

Canonical code:

- [rc_node_v2.py](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate/rc_node_v2.py)
- [collect_tsupply_data.py](C:/Users/user/Desktop/HVAC_DRL_MORL/data/collect_tsupply_data.py)

#### II.1.1. Why `v3` was created

The original design goal of `v3` was not maximum physical interpretability. The goal was to build a surrogate that is easy for PPO-style controllers to optimize in closed loop.

This makes `v3` a **control-oriented surrogate**.

#### II.1.2. Direct supply control method

`v3` uses direct supply-temperature control rather than indirect thermostat setpoint control.

Its action space is:

- `a0 in [-1, 1]`: mapped to `T_supply`
- `a1 in [-1, 1]`: mapped to `fan_u`

The physical mapping used during data collection and control is:

- `T_supply in [18, 35] C`
- `fan_u in [0, 1]`

This direct-TSup formulation is important because it exposes the HVAC actuation channel in a way that is smooth and learnable for RL.

#### II.1.3. Observation / feature space used by the surrogate backbone

The surrogate backbone itself uses an 8-dimensional feature vector:

1. `T_zone`
2. `T_amb`
3. `hour_sin`
4. `hour_cos`
5. `day_sin`
6. `day_cos`
7. `a0`
8. `a1`

This is not the full RL observation space. It is the compact surrogate-dynamics input used to predict the next state and power.

#### II.1.4. Full architecture of `v3`

`v3` is a two-head neural surrogate:

1. **HeatFlowNetV2**
   - predicts temperature increment `dT`
   - uses residual MLP structure with LayerNorm and `tanh`
2. **PowerNetV2**
   - predicts `P_total`
   - uses a smaller MLP with final `Softplus` to keep power non-negative

The overall model is `RCNeuralODEv2`.

Forward logic:

1. build the 8-feature vector
2. predict `dT`
3. compute:
   - `T_next = clamp(T_zone + dT, 15, 35)`
4. predict:
   - `P_total`

In short:

- one head models thermal evolution directly
- one head models energy use directly

#### II.1.5. How data were collected for `v3`

Canonical collection script:

- [collect_tsupply_data.py](C:/Users/user/Desktop/HVAC_DRL_MORL/data/collect_tsupply_data.py)

The `v3` dataset was collected from live BOPTEST using:

- testcase: `bestest_air`
- step: `3600 s`
- seasons:
  - winter
  - spring
  - summer
  - autumn
- excitation policies:
  - `random`
  - `heat`
  - `cool`
  - `mixed`
- steps per episode:
  - `3200`

Total corpus:

- `4 seasons x 4 policies x 3200 = 51,200 samples`

Saved dataset:

- [boptest_v2_tsupply.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/data/surrogate_v2/boptest_v2_tsupply.csv)

Recorded fields include:

- `t_zone`
- `t_amb`
- `hour`
- `day`
- `a0_raw`
- `a1_raw`
- `t_zone_next`
- `delta_t`
- `p_total`
- `policy`
- `season`

#### II.1.6. Why `v3` is useful for RL

`v3` is useful for RL because:

- the action-to-next-state geometry is smooth
- the direct-TSup action semantics are simple
- the model predicts both comfort dynamics and energy
- the learned representation is broad because the dataset spans seasons and excitation styles

This makes `v3` a good **training environment** for RL controllers.

#### II.1.7. Why `v3` is weaker as a digital twin

`v3` is weaker as a digital twin because it is still a black-box surrogate.

Its limitations are:

- it predicts `dT` directly rather than through an explicit thermal-capacitance parameter
- it has no explicit structural building parameter such as `C_zon`
- it can be controllable without being physically interpretable

Therefore:

- `v3` is strong on control utility
- `v3` is weaker on physical identifiability

This is the reason it is described in the paper as **comfort-oriented** and **control-oriented**, but not as the most defensible grey-box twin.

---

### II.2. Explicit-Physics Surrogate `v3.5`

Canonical code:

- [rc_node_v35.py](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate/rc_node_v35.py)
- [inverse_problem_boptest_v35.py](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate/inverse_problem_boptest_v35.py)

#### II.2.1. Why `v3.5` was created

`v3.5` was created to solve the main limitation of `v3`:

- `v3` is easy to optimize
- but `v3` is not explicit enough physically

So `v3.5` introduces an explicit structural thermal parameter:

- `C_zon`

This moves the surrogate from a black-box control model toward a **grey-box digital twin**.

#### II.2.2. Core architecture of `v3.5`

`v3.5` still uses the same 8-feature input:

1. `T_zone`
2. `T_amb`
3. `hour_sin`
4. `hour_cos`
5. `day_sin`
6. `day_cos`
7. `a0`
8. `a1`

But its thermal backbone is different.

Instead of directly predicting `dT`, `v3.5` predicts:

- `Q_net`

and then computes temperature using explicit building physics:

- `dT = dt * Q_net / C_zon`
- `T_next = T_zone + dT`

The main modules are:

1. **QNetV35**
   - predicts latent net thermal flow `Q_net`
2. **PowerNetV35**
   - predicts `P_total`
3. **RCNeuralODEv35**
   - combines the heads with explicit `C_zon`

`C_zon` is stored as a positive learnable parameter using `softplus`.

#### II.2.3. Raw `v3.5` versus calibrated `v3.5`

There are two distinct forms of `v3.5`:

1. **raw `v3.5`**
   - initialized from the `v3` checkpoint
   - contains the explicit `C_zon` backbone
   - not yet adapted to BOPTEST telemetry through calibration heads

2. **calibrated `v3.5`**
   - same explicit-physics backbone
   - plus calibration heads
   - tuned on BOPTEST traces through staged inverse calibration

This distinction matters because:

- raw `v3.5` proves the architectural idea
- calibrated `v3.5` is the actual scientific digital twin used downstream

#### II.2.4. Why inverse calibration is necessary

Without inverse calibration, `v3.5` would still carry mismatch inherited from:

- the original `v3` representation
- noisy telemetry
- delay, bias, and power-channel mismatch between learned surrogate dynamics and live BOPTEST traces

Inverse calibration is needed to:

1. identify `C_zon`
2. align the twin with measured data
3. prevent the surrogate from mistaking sensor artifacts for physical structure

In paper language:

- the inverse problem transforms `v3.5` from a structural hypothesis into a calibrated grey-box twin

#### II.2.5. Data used for `v3.5`

`v3.5` does not rely on one single new dataset. It is built from three layers of evidence.

##### Layer A. Warm-start from `v3`

The `v3.5` backbone is initialized from the already-trained `v3` checkpoint.

This transfers control-useful representation into the explicit-physics model.

##### Layer B. Prepared 15-minute bootstrap dataset

Prepared dataset summary:

- [dataset_summary.json](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/block_1_2_surrogate_rmse/prepared_15min_dataset/dataset_summary.json)

Key facts:

- rows: `10,744`
- episodes: `8`
- step: `900 s`
- controllers:
  - `thermostatic`
  - `hdrl`
- scenarios:
  - `peak_heat_window`
  - `typical_heat_window`

This dataset is:

- prepared from existing 15-minute closed-loop benchmark traces
- heating-window focused
- controller-biased

So it is useful as a **bootstrap calibration corpus**, but not yet as a broad final corpus.

##### Layer C. Collected 15-minute direct-TSup dataset

Collected dataset summary:

- [dataset_summary.json](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs/block_1_2_surrogate_rmse/collected_15min_dataset/dataset_summary.json)

Key facts:

- rows: `48,384`
- episodes: `24`
- step: `900 s`
- seasons:
  - winter
  - spring
  - summer
  - autumn
- policies:
  - `cool`
  - `heat`
  - `mixed`
  - `pulse`
  - `random`
  - `thermostatic_noise`

This corpus is broader than the prepared dataset and gives a much stronger 15-minute direct-TSup envelope.

#### II.2.6. Stage A / B / C inverse-calibration logic

The inverse problem for `v3.5` is solved in three stages.

##### Stage A. Preprocessing

Goal:

- clean the telemetry before structural identification

What Stage A handles:

- temperature bias
- smoothing
- power scaling/bias
- latency alignment
- artifact injection and artifact cleaning logic for controlled studies

Why this matters:

- otherwise `C_zon` can absorb data noise that should not be interpreted as physics

##### Stage B. Structural identification

Goal:

- identify `C_zon` while keeping the explicit-physics backbone constrained

Main idea:

- most of the model is frozen
- the optimization concentrates on the structural thermal parameter
- excitation windows are emphasized

This is the key grey-box step. It is where the digital twin becomes structurally meaningful.

##### Stage C. Calibration heads

Goal:

- improve data fit without destroying the identified physical backbone

Main idea:

- attach calibration heads to temperature and/or power channel
- preserve the structural parameter
- use restricted fine-tuning modes such as:
  - `power_head_only`
  - `rollout_temp_head_only`
  - `heads_only`
  - `joint`

This is how the project avoids the common failure mode:

- a neural surrogate “cheats” by fitting trace data while losing physical meaning

#### II.2.7. Canonical Block 1 results for `v3.5`

Canonical Block 1 report:

- [block1_surrogate_final_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block1_surrogate_final_report.md)

Key inverse-calibration results:

- temperature baseline RMSE: `0.3738 C`
- temperature calibrated RMSE: `0.2319 C`
- temperature RMSE improvement: `38.0%`
- canonical downstream power baseline MAE: `807.83 W`
- canonical downstream power calibrated MAE: `482.03 W`
- power MAE reduction: `40.3%`
- `C_zon final = 4.413e+05 J/K`

Prepared rollout validation:

- raw 24h rollout RMSE: `1.4665 C`
- calibrated 24h rollout RMSE: `0.6441 C`
- calibrated mean episode power RMSE: `687.57 W`

These results justify the claim that `v3.5` is:

- physically defensible
- materially more realistic in open-loop rollout
- strong enough for pretraining and hybrid regularization studies

#### II.2.8. Block 1 failure: zero-shot transfer

Despite strong inverse calibration and rollout realism, `v3.5` failed as a direct zero-shot control simulator.

Canonical zero-shot transfer result:

- peak `m_s = 1.0465`
- typical `m_s = 1.1020`
- `first_divergence_step = 1`
- dominant first divergence feature: `t_zone_norm`

Interpretation:

- `v3.5` is a good grey-box twin
- but not yet a valid direct controller-deployment simulator

This negative result is not noise. It is one of the key scientific conclusions of Block 1.

---

### II.3. Hybrid Surrogate `hybrid_v3_v35`

Canonical code:

- [direct_tsup_adapter.py](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate/direct_tsup_adapter.py)

#### II.3.1. Why the hybrid backend was created

By the end of Block 1, the project had a clear split:

- `v3` is good for control
- `v3.5` is good for physics
- direct `v3.5` warm-start is bad for downstream RL

So the next logical question was:

- can `v3` be kept as the control-oriented dynamics while `v3.5` acts only as a physical regularizer?

That question created the hybrid backend.

#### II.3.2. How the hybrid backend works

The hybrid backend is **not** a separately trained third neural network.

It is a runtime composition:

1. primary dynamics come from `v3`
2. comparison predictions come from calibrated `v3.5`
3. the controller is penalized when both disagree

At each step, the backend returns:

- `t_next`
- `p_total`
- `comparison_t_next`
- `comparison_p_total`
- `temp_disagreement`
- `power_disagreement`

#### II.3.3. Hybrid data source

No new standalone hybrid dataset is collected.

The hybrid backend reuses:

1. the trained `v3` checkpoint
2. the calibrated `v3.5` summary and checkpoint

So the hybrid branch inherits:

- `v3` data collection for control geometry
- `v3.5` calibration data for physics

This is scientifically important: the hybrid branch is a **runtime synthesis**, not a new data-collection regime.

---

## III. Block 2: Downstream Control Utility

Block 2 asks a different question than Block 1.

Block 1 asked:

- is the surrogate physically meaningful and rollout-realistic?

Block 2 asks:

- does that surrogate actually help downstream control?

This block contains four controller-side stories:

1. direct `v3.5` warm-start failure
2. thermostatic hybrid success
3. HDRL transfer limit
4. MORL redesign and success

---

### III.1. Direct `v3.5` Warm-Start Failure

Canonical status:

- frozen negative baseline

Result:

- direct `v3.5 warm-start + BOPTEST fine-tune` is worse than `scratch on BOPTEST`

Frozen comparison:

- peak:
  - scratch `m_s = 0.4653`
  - warm-start `m_s = 1.2701`
- typical:
  - scratch `m_s = 0.5776`
  - warm-start `m_s = 1.2888`

Interpretation:

- a physically strong calibrated twin is not automatically a good direct policy-training environment

This is a critical negative control for the paper.

---

### III.2. Thermostatic Hybrid Sweep

Canonical report:

- [block2_hybrid_surrogate_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hybrid_surrogate_report.md)

#### III.2.1. Experimental setup

Fixed settings:

- backend: `hybrid_v3_v35`
- step: `900 s`
- comfort band: `21-24 C`
- observation path:
  - `obs_ablation = no_delta_t`
  - `power_feature_mode = clipped_log`
  - `t_zone_feature_mode = raw`
- `lambda_power_disagree = 5e-5`

Swept parameter:

- `lambda_temp_disagree`

Tested values:

- `0.05`
- `0.10`
- `0.15`

#### III.2.2. Why `lambda = 0.10` is the canonical thermostatic result

`lambda = 0.10` gave the best comfort-energy compromise.

Canonical thermostatic hybrid result:

- peak:
  - `m_s = 0.0866`
  - violation `4.69%`
  - RMSE `0.795 C`
  - energy `305.3 kWh`
- typical:
  - `m_s = 0.0411`
  - violation `2.38%`
  - RMSE `0.633 C`
  - energy `352.8 kWh`

Context against pure `v3`:

- peak:
  - `v3 m_s = 0.0725`
  - `hybrid m_s = 0.0866`
  - `v3 energy = 322.2 kWh`
  - `hybrid energy = 305.3 kWh`
- typical:
  - `v3 m_s = 0.0947`
  - `hybrid m_s = 0.0411`
  - `v3 energy = 368.3 kWh`
  - `hybrid energy = 352.8 kWh`

Interpretation:

- on peak, hybrid stays near pure `v3` comfort while using less energy
- on typical, hybrid beats pure `v3` on `m_s`, violation, and energy
- therefore `lambda_temp_disagree = 0.10` is the best thermostatic compromise

#### III.2.3. Scientific meaning of thermostatic success

This is the first positive Block 2 result.

It shows that:

- `v3.5` is useful as a physics regularizer
- but not as a standalone direct policy-training environment

So the hybrid strategy is justified scientifically.

---

### III.3. HDRL + Hybrid: Why It Failed

Canonical report:

- [block2_hdrl_lambda_sweep_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hdrl_lambda_sweep_report.md)

#### III.3.1. Experimental setup

Fixed settings:

- backend: `hybrid_v3_v35`
- step: `900 s`
- comfort band: `21-24 C`
- observation path:
  - `obs_ablation = no_delta_t`
  - `power_feature_mode = clipped_log`
  - `t_zone_feature_mode = raw`
- `lambda_power_disagree = 5e-5`

Sweep:

- `lambda_temp_disagree = {0.00, 0.03, 0.05, 0.10}`

#### III.3.2. HDRL result

Best result was not `0.10`.

Best result was:

- `lambda_temp_disagree = 0.00`

Peak:

- `l000 m_s = 0.1803`
- `l003 m_s = 0.3073`
- `l005 m_s = 0.4184`
- `l010 m_s = 0.4395`

Typical:

- `l000 m_s = 0.2337`
- `l003 m_s = 0.2964`
- `l005 m_s = 0.5118`
- `l010 m_s = 0.5114`

#### III.3.3. Why `lambda = 0.00` is best for HDRL

The conclusion is not that the building physics changed.

The conclusion is:

- the regularization weight is **controller-family dependent**

For thermostatic PPO:

- temperature disagreement regularization helps

For HDRL:

- temperature disagreement regularization degrades control quality

So the best HDRL regime is:

- no temperature disagreement penalty
- only soft power regularization

This is a negative but scientifically valuable result:

- hybrid regularization is not controller-family invariant

---

### III.4. MORL Experiments

Canonical report:

- [block2_morl_canonical_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_morl_canonical_report.md)

The MORL branch had two distinct phases:

1. failure under a poor observation interface
2. success after architectural redesign of the observation path

#### III.4.1. MORL phase 1: failed 5D path

Initial MORL configuration used a basic `5D` observation path.

That branch failed badly on yearly BOPTEST validation:

- `RMSE = 4.96 C`
- `MAE = 4.17 C`
- `within_1C = 19%`
- `within_0.5C = 9%`
- `violation = 74.5%`
- `energy = 121.0 kWh`
- `m_s = 1.046`

Interpretation:

- the model was not really controlling comfort
- it was effectively under-actuating and failing the temperature objective

This failure was initially ambiguous:

- was the backend bad?
- or was the MORL interface bad?

#### III.4.2. MORL redesign

The project then moved from a purely computational tweak strategy to an architectural strategy.

The MORL observation path was promoted to a TSup-style `17D` state:

- `obs_mode = extended`
- `obs_ablation = none`
- `delta_feature_mode = causal_smooth`
- `power_feature_mode = clipped_log`
- `t_zone_feature_mode = raw`

Hybrid regularization settings for MORL:

- `lambda_temp_disagree = 0.00`
- `lambda_power_disagree = 5e-5`

This made MORL consistent with the HDRL boundary finding:

- no temperature penalty
- soft power-only physical regularization

#### III.4.3. MORL phase 2: 17D success

Canonical yearly mean result:

- `RMSE = 0.72 C`
- `MAE = 0.56 C`
- `within_1C = 83%`
- `within_0.5C = 57%`
- `violation = 4.9%`
- `energy = 248.6 kWh`
- `m_s = 0.099`

#### III.4.4. Why MORL succeeded

The comparison `5D -> 17D` shows that the dominant issue was not the hybrid backend itself.

It was the observation interface.

Key deltas:

- `RMSE: 4.96 -> 0.72`
- `MAE: 4.17 -> 0.56`
- `within_1C: 19% -> 83%`
- `within_0.5C: 9% -> 57%`
- `violation: 74.5% -> 4.9%`
- `m_s: 1.046 -> 0.099`

Interpretation:

- MORL became viable only after giving the controller a richer TSup-style representation
- the power-only hybrid backend is sufficient for MORL, provided the interface is informative enough

This closes Block 2 across the intended controller stack:

1. thermostatic positive hybrid result
2. HDRL boundary / limit result
3. MORL final power-only success result

---

## IV. Final Scientific Interpretation Before Block 3

The completed results now support the following structured claim.

### IV.1. What Block 1 proved

Block 1 proved that:

- explicit-physics calibration with `C_zon` works
- `v3.5` materially improves rollout realism
- `v3.5` is a defensible grey-box digital twin
- but `v3.5` is still not a valid zero-shot controller-deployment environment

### IV.2. What Block 2 proved

Block 2 proved that:

- direct `v3.5` warm-start is not enough
- thermostatic controllers benefit from hybrid physical regularization
- HDRL rejects temperature disagreement regularization
- MORL succeeds under power-only hybrid regularization, but only with a richer observation interface

### IV.3. The current paper-level claim

The strongest defensible paper claim is:

**The calibrated physics twin is most useful as a physical regularizer for control-oriented surrogate learning. Its utility is controller-family specific, and the final MORL result depends not only on backend design but also on a sufficiently rich TSup-style observation interface.**

### IV.4. What this means for Block 3

Block 3 should not ask:

- is one frozen universal surrogate model transferable to every building?

Block 3 should ask:

- is the **hybrid surrogate construction procedure** transferable across similar building cases?

That is now the correct next scientific question.

---

## V. Frozen Deliverables for the Article

Canonical Block 1:

- [block1_surrogate_final_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block1_surrogate_final_report.md)

Canonical Block 2 thermostatic:

- [block2_hybrid_surrogate_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hybrid_surrogate_report.md)

Canonical Block 2 HDRL:

- [block2_hdrl_lambda_sweep_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_hdrl_lambda_sweep_report.md)

Canonical Block 2 MORL:

- [block2_morl_canonical_report.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_morl_canonical_report.md)

Block 2 MORL comparison:

- [block2_morl_comparison_summary.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/block2_morl_comparison_summary.csv)

Canonical frozen bundles:

- [block1_surrogate_fidelity_bundle](C:/Users/user/Desktop/HVAC_DRL_MORL/block1_surrogate_fidelity_bundle)
- [block2_hybrid_branch_bundle](C:/Users/user/Desktop/HVAC_DRL_MORL/block2_hybrid_branch_bundle)

This is the current article-ready state before Block 3.
