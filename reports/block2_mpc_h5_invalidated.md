# H5, first execution: invalidated

**Two causes, one real.** See the correction immediately below.

**Status: executed 19 Aug 2026, result discarded. Not evidence for or against H5.**

This file exists because the run happened. Deleting it and quietly re-running
would leave the pre-registration looking untouched while a completed run had
already been seen, which is the thing pre-registration is meant to prevent.

## What was run

`evaluation/run_mpc_baseline.py --backends mpc_bb_mono`, against the protocol in
`configs/mpc_baseline_preregistration.yaml` as it stood at commit time.

| Window | m_s | Violation | RMSE_T | Energy |
| --- | ---: | ---: | ---: | ---: |
| peak | 1.362 | 77.2 % | 5.777 °C | 20.1 kWh |
| typical | 1.563 | 84.4 % | 7.590 °C | **0.0 kWh** |

Read at face value this falsifies H5. It does not, and the zero is why.

## Correction, 20 Aug 2026: the cause named below was wrong

This file originally attributed the zero-energy result to the inverted fan
channel of the planning surrogate. That was a misdiagnosis. The real cause is a
defect in `evaluation/mpc_baseline.py`, and it affected **every** backend,
including ones that are directionally sound.

The planner optimises an unconstrained variable and applies `tanh` to it. Once
the optimiser pushed that variable far enough negative, `tanh` saturated,
`dtanh/dseq` went to zero, the gradient died, and the receding-horizon warm start
carried the dead sequence into every later control step. Measured on the hourly
BB, which is 100 % / 98.5 % directionally valid and trains the study's usable
controller: `a0` pinned at -1.000 from step 0 of the typical window and never
recovered.

The giveaway was that three different surrogates produced **bit-identical**
typical-window results (m_s 1.5633, violation 84.4 %, energy 0.0 kWh). Different
models cannot agree to four decimal places; what they had in common was the
planner, which was emitting the minimum command regardless of what any of them
predicted.

Fixed by clamping the unconstrained variable to +/- 2.5, which keeps the full
command range reachable while leaving the gradient alive. All three MPC runs
below are discarded.

The fan-channel inversion is real and independently verified (20.8 % validity on
the constrained matched-resolution model, 2.2 % unconstrained), and the admission
gate added for H5b stays. It was simply not what produced these numbers.

## Why it is invalid (original text, superseded above)

The planner drove **both** commands to their hard minimum and held them there.
On the typical window `a0 = a1 = -1.000` for all 1344 steps: fan off, no air
delivered, zero power, zone down to 5.9 °C.

That is not an energy/comfort weighting artifact. With `lambda_comfort = 60` the
comfort term reaches tens of thousands while the normalised energy term is bounded
by 1, so the objective overwhelmingly favours heating. The planner was acting
correctly on a model that was telling it the fan does not help.

**The fan channel of the planning surrogate is inverted.** Directional validity
with hot supply air (35 °C), 400 sampled states, "more fan must not cool the zone":

| Surrogate | Supply channel `a0` | Fan channel `a1` |
| --- | ---: | ---: |
| BB hourly | 100 % | 98.5 % |
| GB calibrated | 100 % | 98.8 % |
| BB matched (unconstrained) | 9.8 % | 2.2 % |
| BB matched + supply constraint (`mono`) | 100 % | **20.8 %** |

The monotonicity constraint introduced earlier acted on `a0` only. It repaired
that channel and **relocated** the defect: the fan channel went from 2.2 % to
20.8 %, still inverted in four states out of five.

## Consequences beyond this run

The same defect invalidated a manuscript claim. The RL controller trained on the
`mono` surrogate (m_s 1.426 / 1.597) had been written up as "accurate,
directionally valid, same model class, and still fails", isolating the step-size
mechanism at matched resolution. It exploited the fan channel exactly as the
planner did: fan mean 0.201, p10 = 0.000, mean power 182 W against 959 W for the
usable controller. That claim was removed from `docs/paper_asej/manuscript.tex`
and `docs/ieee_access/ieee_access_hvac.tex` on 19 Aug 2026.

What survives: the `a0` inversion of the unconstrained matched-resolution model
across four independent training draws, with 24 h rollout RMSE improving through
it (`reports/block1_matched_bb_seed_audit.json`). That finding is unaffected and
is now stated for both channels rather than one.

## Method error, for the record

The directional audit was built on one of two actuated inputs and the conclusion
"the confound is removed" was drawn from it. A surrogate is only directionally
sound if every input a controller can move has been checked; a controller will
find whichever one is wrong. The audit scripts now cover both channels
(`evaluation/check_surrogate_response_sign.py`), and the training constraint has
a fan term (`--lambda-mono-fan`).

## What replaces it

H5 is re-registered as **H5b** in `configs/mpc_baseline_preregistration.yaml`,
with an admission gate: no surrogate enters the MPC comparison until it passes
directional validity on **both** channels at the stated threshold. The gate is
checked and recorded before the planner runs, not after.

Raw artifacts of the invalidated run are kept under
`outputs/block2_mpc_baseline/mpc_bb_mono/` and are not deleted.
