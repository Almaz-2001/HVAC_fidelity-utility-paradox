# Legacy Sinergym Thesis Branch: Presentation Draft

## Slide 1. Title
**Rebuilding the Thesis-Era Sinergym Branch and Pushing Beyond RBC**

Subtitle:
- Honest live reproduction of the legacy Sinergym pipeline
- Current status versus the thesis-era claim set

Presenter note:
- Do not say "we already beat the thesis" on this slide.
- The honest claim is: we rebuilt the branch, fixed the broken measurement/action path, and now have a materially stronger PPO than before.

## Slide 2. Why The Original Claim Needed Revalidation
Key points:
- The original article-style figures were not fully generated from live CSV pipelines.
- The legacy branch had technical issues that made older comparisons unreliable.
- We rebuilt a live evaluation stack to compare PPO, rule-based, random, and zero-hold on the same corrected environment.

Evidence files:
- `outputs/legacy_sinergym/live_figures/live_sinergym_honest_reproduction.md`
- `outputs/legacy_sinergym/live_figures/live_sinergym_summary.txt`

Presenter note:
- This slide protects credibility.
- The audience should understand that the current results are more trustworthy than the earlier static article figures.

## Slide 3. What We Fixed In The Legacy Sinergym Branch
Key fixes:
- Correct observation semantics:
  - `obs[12] = Zone Air Temperature(SPACE1-1)`
  - `obs[16] = Facility Total HVAC Electricity Demand Rate(Whole Building)`
- Name-based variable lookup instead of blind index trust
- Valid action parameterization:
  - `midpoint_gap` instead of invalid direct setpoint pairing
- Gaussian comfort reward restored as the best current reward family
- RBC-to-BC warm-start added before PPO fine-tuning

Relevant files:
- `envs/backends/sinergym_backend.py`
- `envs/rewards/morl_reward.py`
- `envs/wrappers/action_normalize.py`
- `training/bc_warmstart.py`
- `configs/legacy_sinergym/env.yaml`

Presenter note:
- This is the engineering slide that explains why the current numbers are meaningful.

## Slide 4. Current Best Honest Legacy Result
Source:
- `outputs/legacy_sinergym/live_figures/live_baseline_summary.csv`

Current live numbers:

| Policy | Mean HVAC power (W) | Mean comfort penalty | In-band (%) |
|---|---:|---:|---:|
| PPO | 499.80 | 11.00 | 41.0 |
| Rule-based | 518.24 | 10.94 | 59.0 |
| Random | 507.11 | 12.01 | 16.0 |
| Zero-hold | 508.88 | 12.16 | 16.2 |

What we can honestly say:
- PPO is now clearly stronger than random and zero-hold.
- PPO is better than RBC on energy.
- PPO is very close to RBC on comfort, but not yet better.

What we cannot honestly say:
- We cannot claim that PPO has already beaten RBC on the full comfort-energy trade-off.
- We cannot claim that thesis-era numbers have been fully surpassed.

## Slide 5. Best Improvement Achieved So Far
Before BC warm-start:
- PPO often collapsed into bad cold/high-power regimes.
- In-band occupancy was near failure-level.

After BC warm-start:
- PPO became stable.
- In-band rose to `41.0%`.
- Mean power dropped below RBC.

Interpretation:
- The cold-start failure mode is solved.
- The remaining problem is no longer system wiring.
- The remaining problem is policy optimization near the RBC comfort frontier.

Presenter note:
- This is the strongest "progress" slide.
- The right message is: we crossed from broken policy learning into meaningful policy improvement.

## Slide 6. Visual Comparison Slide
Use:
- `outputs/legacy_sinergym/live_figures/live_fig1_baseline_comparison.png`
- `outputs/legacy_sinergym/live_figures/live_fig2_representative_trajectories.png`

Talking points:
- The PPO bar is below RBC on mean HVAC power.
- PPO is now in the same qualitative operating region as RBC.
- Random and zero-hold remain clearly inferior.

Presenter note:
- This is where you visually show the audience that PPO is no longer behaving like a failed policy.

## Slide 7. Surrogate Validation Slide
Use:
- `outputs/surrogate_comfort_traces/surrogate_v3_vs_boptest_yearly.png`
- `outputs/surrogate_comfort_traces/surrogate_v3_vs_boptest_parity.png`

Suggested message:
- Surrogate v3 tracks BOPTEST trends on comfort-oriented traces.
- The model is useful for pretraining and rollout analysis.
- It still shows a warm bias and slightly overestimates comfort-band occupancy.

Do not say:
- "The surrogate is a perfect replacement for BOPTEST."

Presenter note:
- This slide supports methodology, not final control superiority.

## Slide 8. Honest Conclusion
Best current conclusion:
- We rebuilt the thesis-era legacy Sinergym branch into an honest live evaluation pipeline.
- We fixed the temperature channel, action parameterization, and reward/warm-start logic.
- PPO now beats RBC on energy and approaches RBC on comfort.
- We have not yet produced a fully defensible claim that the thesis-era benchmark has been surpassed.

Best final sentence:
- "The thesis branch has moved from non-reproducible legacy behavior to a corrected live PPO pipeline that is now competitive with RBC and positioned for final optimization."

## Slide 9. Next Experiment
Immediate next experiment:
- Keep the current gaussian reward baseline
- Run a small sweep over:
  - BC intensity
  - PPO learning rate
- Objective:
  - maintain `power < 518.24 W`
  - achieve `comfort penalty < 10.94`

Success condition:
- First run where PPO is better than RBC on both energy and comfort

## Slide 10. If You Need A Stronger Title For A Defense
Safe options:
- "Toward Surpassing Thesis-Era HVAC RL Results"
- "Correcting and Strengthening the Legacy Sinergym Thesis Pipeline"
- "From Legacy Reproduction to Competitive PPO Control in Sinergym"

Unsafe option for now:
- "We surpassed the thesis results"

