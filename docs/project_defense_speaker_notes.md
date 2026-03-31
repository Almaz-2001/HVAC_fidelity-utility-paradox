# Project Defense Speaker Notes

This file follows the slide order in [project_defense_presentation.tex](/C:/Users/user/Desktop/HVAC_DRL_MORL/docs/project_defense_presentation.tex).

## Slide 1. Title

Introduce the project as a long-term research program rather than a single controller. Emphasize that the work evolved from an initial PPO baseline into a surrogate-driven HVAC control pipeline with explicit comfort-energy trade-offs.

## Slide 2. Agenda

State clearly that the defense has three layers:
- what was completed in Phases 0, 1, and 2,
- what Phase 3 achieved in the final report,
- what has changed in the current continuation and what is still ongoing.

## Slide 3. Problem Statement

Explain that HVAC control is not a single-objective optimization task. The controller must maintain comfort, reduce energy, remain stable across seasons, and eventually become safe enough for constrained deployment. This is why MORL is relevant here.

## Slide 4. Research Roadmap

Use this slide to frame the whole thesis. The main message is that every phase solved a different bottleneck:
- Phase 0 solved reward and observation issues,
- Phase 1 solved speed,
- Phase 2 solved physical mismatch,
- Phase 3 solved control-architecture mismatch.

## Slide 5. Phase 0

Explain that Phase 0 was not just about obtaining a PPO controller. It was mainly diagnostic. The two important lessons were:
- the scalarized reward can collapse if one term dominates by scale,
- RL becomes unreliable if the declared observation space does not match the actual environment state.

The point is that the project became robust because these low-level issues were fixed early.

## Slide 6. Why Neural ODE

Make the argument that the surrogate was not added for convenience; it was necessary for research velocity. Direct BOPTEST training is too slow to support large ablations, seasonal specialization, or safety rollouts.

## Slide 7. Neural ODE Formulation

Explain the model in plain language:
- the network sees current indoor temperature, outdoor temperature, time-of-day, day-of-year, and control action,
- one head predicts thermal change,
- one head predicts HVAC power,
- the temperature update remains physically interpretable.

This is a strong point in the defense because it shows that the surrogate is not a black box replacing physics entirely.

## Slide 8. Phase 1 Results

Stress the speedup. This is one of the strongest quantitative contributions of the project. The surrogate is not only accurate enough, it is also orders of magnitude faster than BOPTEST and therefore enables the rest of the work.

## Slide 9. Phase 2 Calibration

Explain that an inaccurate surrogate is dangerous for control and especially for safety filtering. Phase 2 reduced this risk by improving the physical calibration and reducing thermal prediction error.

## Slide 10. Phase 3 in the Final Report

Say that Phase 3 systematically tested multiple ideas rather than making a single leap. This matters because it shows scientific discipline:
- hierarchical specialization,
- real weather,
- emergency logic,
- reward shaping,
- BOPTEST fine-tuning.

## Slide 11. Best Legacy Phase 3 Configuration

Present the old final report result as the state of the project before the current continuation. Do not overstate it. Mention that it was a strong milestone, but not the end state.

## Slide 12. Why Phase 3 Had to Continue

This is a key transition slide. Explain that the report itself concluded that the remaining gap was structural. Therefore, the next step was not just more training, but changing the action semantics.

## Slide 13. Direct Supply-Air Control

This is one of the central engineering contributions of the continuation.

Explain:
- before: indirect setpoint-like control,
- now: direct control of supply-air temperature and fan,
- same meaning of action in surrogate and BOPTEST.

Say explicitly that this greatly reduced train-eval mismatch.

## Slide 14. Current End-to-End Pipeline

Use this slide to show that the project is now a coherent system:
- data collection,
- surrogate training,
- comfort PPO,
- HDRL,
- PI reference,
- MORL and safety layers.

This makes the codebase look mature and modular.

## Slide 15. Standard PI Controller

Explain why this slide exists. The standard controller is important not because it is the best overall controller, but because it provides an honest low-energy reference.

Clarify that it is evaluated in two ways:
- against constant 22 C,
- against its own schedule.

## Slide 16. Why the PI Baseline Matters

This is where you preempt reviewer criticism. Explain that comparing RL only against RL is not enough. The standard controller shows what low-energy operation looks like. However, it solves a softer scheduled-comfort problem, so it is not the main comfort baseline.

## Slide 17. Thermostatic PPO Baseline

Position this controller as the strongest current comfort benchmark.

Key points to say:
- trained on the new direct-TSup surrogate,
- richer 17-dimensional state,
- exact 22 C comfort target,
- weak energy penalty,
- intentionally comfort-first.

## Slide 18. Thermostatic Results

This is one of the main evidence slides. State that the current direct-TSup thermostatic PPO is the strongest validated controller in terms of comfort.

Important interpretation:
- winter collapse is solved,
- summer is now the harder regime,
- this controller becomes the main comfort reference for all later trade-off methods.

## Slide 19. Thermostatic Interpretation

Use this slide to slow down and explain why the result matters. It is not just a better RMSE; it validates the new pipeline.

## Slide 20. HDRL Design

Present HDRL as a structured controller, not yet a winning controller.

Explain:
- two experts,
- compact state,
- seasonal specialization,
- emergency heating,
- hard threshold meta-controller.

## Slide 21. HDRL Results

Be honest here. The strongest message is:
- HDRL works,
- summer energy is slightly better,
- but annual comfort-energy trade-off is still not better than the thermostatic baseline.

That honesty will strengthen the defense.

## Slide 22. Controller Comparison

This is the summary slide for results.

Frame it as:
- PI = best energy reference,
- thermostatic PPO = best comfort reference,
- HDRL = first structured trade-off controller, but not yet dominant.

## Slide 23. Why HDRL Underperforms

This is the slide where you show that you understand your own failure modes.

Main reasons:
- different objective,
- less information,
- hard switching rule,
- band-based reward instead of exact target tracking.

This makes the gap scientifically interpretable, not accidental.

## Slide 24. MORL Design

Say that MORL is implemented and already structurally integrated into the repo, but the direct-TSup transition happened before the full MORL re-evaluation was completed.

This shows progress without claiming unsupported final metrics.

## Slide 25. Safety Filter

Explain the safety logic in one sentence:
- proposed PPO action is accepted only if short-horizon surrogate rollout remains in a safe temperature band.

This is important because it demonstrates that the project is moving toward safe deployment, not just better benchmark numbers.

## Slide 26. Honest MORL Status

This is a credibility slide. Say clearly:
- MORL and Safe MORL are implemented,
- the architecture is ready,
- the fully updated direct-TSup evaluation is the next step.

This is the right scientific stance for a defense.

## Slide 27. Main Contributions

Summarize the project as six contributions:
- MORL baseline,
- Neural ODE surrogate,
- calibration,
- systematic Phase 3 experiments,
- direct-TSup redesign,
- safety-aware architecture.

## Slide 28. Current Limitations

This slide should sound mature, not defensive.

Main message:
- the project solved major infrastructure problems,
- but the final MORL trade-off controller is still under active refinement.

## Slide 29. Next Steps

This is where you show continuation value.

Say that the next most important milestone is:
- train MORL on the direct-TSup surrogate,
- fine-tune on BOPTEST,
- evaluate Safe MORL,
- replace hard seasonal switching with a learned gate.

## Slide 30. Final Takeaways

Close with the strongest narrative:
- Phase 1 delivered speed,
- Phase 2 delivered physical fidelity,
- Phase 3 identified the real structural bottleneck,
- the current continuation solved the control interface mismatch,
- the project is now in a strong position to finish the MORL objective.

## Appendix Slides

Use these only if asked:
- file map,
- commands,
- reproducibility workflow.
