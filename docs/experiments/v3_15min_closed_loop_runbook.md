# Runbook — closed-loop RL on the matched-resolution v3 (reviewer pre-emption)

**Goal.** Settle the one open confound in §8.5 of the manuscript: does the black-box
v3 surrogate's *training utility* come from its black-box smoothness, or partly from
the fact that it is fit at a 1-hour step but used at the 15-minute control step?

The matched-resolution v3 (`rc_node_v3_15min_matched.pt`) was already validated as a
**predictor** (24 h rollout RMSE ≈ 0.876 °C). This runbook closes the loop: it trains
a PPO controller **on** that matched v3 and evaluates it on the **same two targeted
14-day live-BOPTEST windows** used for the canonical pure-v3 controller, then folds
the result into a Block 2 comparison row.

Only the surrogate's *training resolution* changes; PPO hyperparameters, control step
(900 s), comfort band, windows and seed are held fixed — a clean matched-config ablation.

---

## What each step needs

| Step | Needs BOPTEST Docker? | Dominant cost |
|------|----------------------|---------------|
| A — (re)build matched v3 predictor | no (uses the already-collected 15-min corpus) | ~30–60 min CPU |
| B — train PPO on matched v3 | **no** (pure surrogate rollouts, 32 envs) | ~2–5 h / seed |
| C — live closed-loop benchmark | **yes** (BOPTEST RTE web service) | ~0.5–1.5 h (two windows) |
| D — integrate result into Block 2 | no | seconds |

> Steps A and B are pure-surrogate and can run on the host. Only **Step C** requires the
> running BOPTEST emulator (`http://web:8000`, the same service used for every other
> live result in the paper). Run all commands from the repository root (`/app` inside
> the Docker image).

A single fixed seed (42) is enough for an apples-to-apples comparison against the
canonical pure-v3 row, which is itself a single deterministic-seed evaluation. Add
`--seed 43 --seed 44` runs only if you want a 3-seed band.

---

## Step A — matched-resolution v3 predictor (likely already present)

```bash
# Retrain v3 on the same 15-min corpus as v3.5 (canonical hyperparameters).
python3 -B evaluation/run_block1.py v3-train-15min
# -> outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt

# (optional) confirm the predictor number that appears in the paper (~0.876 C):
python3 -B evaluation/run_block1.py validate-rollouts --variant v3_15min
```

If `outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt` already exists from
the corpus-matched report, you can skip the retrain and reuse it.

---

## Step B — train the PPO controller on the matched v3

All Block 2 steps are routed through `evaluation/run_block2.py` (add `--dry-run` before
any subcommand to print the resolved command without running it).

**B0. Smoke test first (≈2–3 min)** — confirms the checkpoint loads through the
`legacy_v3` adapter and the env steps before you commit to the full 10 M-step run:

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant pure_v3_15min --smoke
```

If that produces `models/ppo_thermostatic_v3_15min_smoke.zip` without a loader/shape
error, delete it (`rm -f models/ppo_thermostatic_v3_15min_smoke.zip`) and launch the
real run:

**B1. Full run (matches the canonical pure-v3 recipe exactly except the surrogate):**

```bash
python3 -B evaluation/run_block2.py thermostatic-train --variant pure_v3_15min
# -> models/ppo_thermostatic_v3_15min.zip
```

(The canonical pure-v3 baseline is the identical recipe under
`thermostatic-train --variant pure`; only the surrogate checkpoint differs.)

---

## Step C — live closed-loop benchmark on the two targeted windows

Requires the BOPTEST web service to be up. This reuses the exact benchmark that
produced the canonical pure-v3 row:

```bash
python3 -B evaluation/run_block2.py thermostatic-benchmark --variant pure_v3_15min
# -> outputs/bestest_air_pure_v3_15min/summary.json  (peak + typical windows, m_s, violation%, energy)
```

---

## Step D — integrate into the Block 2 comparison

```bash
python3 -B evaluation/run_block2.py v3-15min-report
# -> reports/block2_v3_15min_closed_loop_comparison.csv
# -> reports/block2_v3_15min_closed_loop_comparison.json   (verdict + ready-to-quote sentence)
```

The script prints a `VERDICT` and a single-sentence `interpretation`. The JSON's
`interpretation` field is written to be quoted verbatim in §8.5; the CSV is a tidy
two-variant × two-window table ready to append to the Block 2 controller table.

---

## How to read the outcome

| Verdict (auto) | Meaning | Manuscript action |
|----------------|---------|-------------------|
| `confound_rejected` | matched v3 trains a usable controller, m_s ≈ hourly | **Strengthens the paper.** Promote §8.5(iii) from open limitation to a resolved control: the paradox holds at matched resolution; utility is the black-box architecture, not the timestep. |
| `partial_confound` | usable but measurably worse m_s | Add the row; state that temporal coarse-graining contributes part of the utility alongside the black-box nature. Minor reframing of §8. |
| `timestep_driven` | fails sub-5% on ≥1 window | Still publishable: refine the mechanism to "useful smoothness comes from temporal coarse-graining." Reframe the black-box-vs-grey-box narrative in §8.1/§8.5. |
| `inconclusive` | neither clearly matched nor worse | Report both rows; let per-window numbers stand. |

Every outcome is reportable — there is no result here that sinks the paper; the worst
case only refines the claim. Thresholds (`violation_usable_pct=5.0`,
`m_s_rel_tolerance=0.25`) are defined at the top of the integration script and can be
adjusted if the reviewer-facing criterion changes.

---

## One-shot driver (optional)

For an unattended single-seed run once BOPTEST is up:

```bash
set -e
python3 -B evaluation/run_block1.py v3-train-15min
python3 -B evaluation/run_block2.py thermostatic-train     --variant pure_v3_15min
python3 -B evaluation/run_block2.py thermostatic-benchmark --variant pure_v3_15min
python3 -B evaluation/run_block2.py v3-15min-report
```
