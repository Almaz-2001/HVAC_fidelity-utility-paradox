# MPC baseline and H5b: the fidelity–utility inversion is not specific to RL

Pre-registration: `configs/mpc_baseline_preregistration.yaml` (H5b).
Invalidated first execution: `reports/block2_mpc_h5_invalidated.md`.
Artifacts: `outputs/block2_mpc_baseline/`.

## Result

Receding-horizon MPC, 6 h horizon, planning through each surrogate, evaluated
zero-shot on live BOPTEST over the same two 14-day windows as every RL
controller in the study.

| Planning surrogate | 24 h RMSE | Sign a0 / fan | MPC peak | MPC typ. | RL peak | RL typ. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BB hourly | 1.557 (worst) | 100 % / 98.5 % | **0.547** | **0.869** | 0.073 | 0.095 |
| BB matched | 0.876 | 9.8 % / 2.2 % | 1.425 | 1.462 | 1.142 | 1.211 |
| GB calibrated | 0.644 (best) | 100 % / 98.8 % | 1.051 | 1.369 | 1.046 | 1.102 |

Reference: BOPTEST built-in PI 0.910; role-separated hybrid RL 0.087 / 0.041.

## Verdicts against the pre-registered rules

**H5b: FALSIFIED.** The rule was `m_s < 1.0 on both windows` for MPC on an
admitted surrogate. GB calibrated passes the admission gate (100 % / 98.8 %) and
the planner reaches 1.051 and 1.369. Both above 1.

This is not the degenerate failure that invalidated the first execution: energy
is 230 and 251 kWh, the planner is actuating throughout.

**Negative control: PASSED.** MPC on the sign-inverted matched-resolution model
fails (1.425 / 1.462), as it must. Had it succeeded, the harness would have been
suspect rather than the hypothesis.

**Q1 — is the new baseline stronger than the built-in PI?** Yes. MPC on the
hourly BB reaches 0.547 and 0.869 against PI's 0.910. The margin on the typical
window is small.

**Q2 — does MPC beat the role-separated hybrid RL?** No, and not closely: 0.547 /
0.869 against 0.087 / 0.041, a factor of six to twenty.

**Secondary ordering.** MPC on GB is *worse* than MPC on the hourly BB, the same
direction as the learner. The planner reproduces the inversion.

## What this changes

The inversion is **not** specific to policy-gradient learning. Both controller
classes rank the surrogates the same way, and the least accurate surrogate gives
the best controller in both. On GB the two classes land almost on top of each
other (MPC 1.051 against RL 1.046 on the peak window).

The paper's mechanism section should therefore say *gradient-based optimisation
through the surrogate*, not *policy-gradient search*. The scope is wider than
was claimed.

## Limitation that bounds this conclusion

The planner implemented here optimises an action sequence with Adam **through the
differentiable surrogate**. It is a gradient method, so it shares whatever
pathology gradients have on a rough response surface. This experiment therefore
separates *gradient-based optimisation* from *everything else*; it does **not**
separate planning from learning, which is what H5b was worded to test. That
conflation is a defect in how the hypothesis was posed, not in the run.

The clean follow-up is a derivative-free planner — CEM or random shooting —
through the same surrogates. If it also fails on GB, the mechanism is in the
response-surface geometry. If it succeeds, the mechanism is specific to
gradients. No code exists for this yet.

## Protocol notes

Planner settings are the pre-registered ones and were not tuned against the
metric being reported. MPC on the hourly BB still violates comfort 34.5 % and
55.6 % of the time, so this is a *stronger* baseline than the built-in PI, not a
well-tuned MPC. Any comparison in the manuscript should say so.
