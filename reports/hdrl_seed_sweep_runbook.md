# HDRL multi-seed lambda sweep — runbook

Purpose: replace the single-seed HDRL `lambda_temp_disagree` sweep with a
mean ± sd band over three seeds. The single-seed sweep is flagged as the weakest
evidence in the manuscript, and IEEE Access allows exactly one revision round, so
the band has to be in the submission rather than promised in a response letter.

Run everything from the repository root **inside the container** (`/app`), with the
BOPTEST service reachable at `http://web:8000`.

## Safety properties of this pipeline

- Writes to a **new** namespace `outputs/block2_hdrl_seed_sweep/` and model prefix
  `hdrl_seedsweep_*`. The frozen artifacts (`outputs/block2_hdrl_hybrid_v3_v35_*`,
  `models/hdrl_hybrid_*`, `reports/block2_hdrl_lambda_sweep_summary.csv`) are read-only
  here and are never moved or overwritten.
- Every stage is resumable with `--skip-existing`.
- `--smoke` uses its own prefix and artifact root, so smoke checkpoints can never be
  mistaken for finished cells.
- Each cell writes a log under `logs/hdrl_seed_sweep/` and a status entry in
  `outputs/block2_hdrl_seed_sweep/sweep_status.json`.

---

## Step 0 — connectivity and chain check (~15 min)

```bash
curl -s http://web:8000/version
```

```bash
python evaluation/run_hdrl_seed_sweep.py --stage all --smoke --lambdas l000 --seeds 42
```

A clean finish means the surrogate loads, training runs, checkpoints save, and the
live benchmark completes. The numbers are meaningless at 20 000 steps.

Clean up when it passes:

```bash
rm -rf outputs/block2_hdrl_seed_sweep_smoke models/hdrl_seedsweep_smoke_*
```

---

## Step 1 — train/eval observation encoding: RESOLVED, no action needed

**Status: checked on 13 Aug 2026, the published results are clean.**

`evaluation/run_block2.py::hdrl_train_command` trains with
`--obs-ablation no_delta_t --power-feature-mode clipped_log`, but `hdrl_benchmark_command`
forwarded neither flag. The ablations zero slots rather than removing them, so a mismatch
changes no dimension and raises nothing — the policy is simply fed an encoding it never
saw. That made it worth checking whether the frozen sweep had been produced that way.

It had not. `evaluation/check_hdrl_obs_consistency.py --duration-days 3` on
`models/hdrl_hybrid_l000_*` reproduced the frozen trace
(`outputs/block2_hdrl_hybrid_v3_v35_l000/traces/peak_heat_window_hdrl.csv`) **exactly**
under the matched encoding, and not at all under the legacy one:

| arm | max abs ΔT_zone vs frozen | max abs Δaction | m_s peak | violation peak |
| --- | ---: | ---: | ---: | ---: |
| matched (`no_delta_t`, `clipped_log`) | **0.0000 °C** | **0.0000** | 0.2095 | 9.03 % |
| legacy (`none`, `raw`) | 7.9100 °C | 1.6888 | 0.6744 | 38.54 % |

The frozen 14-day run's own first three days violate 9.03 % of the time, matching the
matched arm exactly. So the published sweep used the correct flags; only the `run_block2.py`
helper omitted them.

Two consequences:

- Nothing in the manuscript needs revisiting on this account.
- The helper was still a live trap — a rerun through it would have produced silently wrong
  numbers. `hdrl_benchmark_command` and `thermostatic_benchmark_command` now forward the
  flags (the pure-v3 variants keep the defaults, which is what they were trained with).

Raw evidence: `outputs/block2_hdrl_obs_consistency_3d/`.

The thermostatic hybrid path had the same omission and feeds the headline numbers. The
same argument applies (the published run predates the helper), but it is cheap to confirm:

```bash
python evaluation/check_hdrl_obs_consistency.py --controller thermostatic --duration-days 3
```

---

## Step 2 — train the sweep

> **The seed-42 cell is a fresh draw, not a reproduction of the frozen checkpoint.**
> `train_hdrl.py` previously seeded only the environments; PPO itself drew its own
> entropy, so the frozen models are not reproducible from any seed. Now that PPO is
> seeded too, `l000_seed42` is an independent fourth sample rather than a rerun of the
> published one. That is the right thing for a seed band — three clean draws under one
> protocol — but it means the frozen point should be labelled as separate provenance in
> the figure, which `build_hdrl_seed_band_figure.py` does.


Full grid is 4 λ × 3 seeds = 12 cells. Each cell trains 5 M winter + 7 M summer steps,
≈ 1.9 h at the measured hybrid throughput, so the full grid is roughly **23 h**.

Endpoints first is the better use of the first night: λ = 0.00 and λ = 0.10 are what
carry the claim (the thermostatic-best setting does not transfer to HDRL). The two
intermediate points only add shape.

```bash
python evaluation/run_hdrl_seed_sweep.py --stage train --lambdas l000,l010 --seeds 42,43,44 --skip-existing
```

Then, if time allows:

```bash
python evaluation/run_hdrl_seed_sweep.py --stage train --lambdas l003,l005 --seeds 42,43,44 --skip-existing
```

Interrupted? Re-run the same command; `--skip-existing` resumes.

---

## Step 3 — benchmark on live BOPTEST (~5 min per cell)

```bash
python evaluation/run_hdrl_seed_sweep.py --stage benchmark --lambdas l000,l010 --seeds 42,43,44 --skip-existing
```

---

## Step 4 — aggregate and plot

```bash
python evaluation/build_hdrl_seed_band.py
```

Writes `reports/block2_hdrl_lambda_sweep_seed_band.{csv,json}` and prints the two claims
a reviewer will test: whether `m_s` rises monotonically with λ on seed means, and whether
the λ = 0.00 → 0.10 gap exceeds the seed noise. With three seeds the separation figure is
descriptive, not a significance test, and is labelled as such.

```bash
python evaluation/build_hdrl_seed_band_figure.py
```

Writes `reports/figures/hdrl_seed_band/block2_hdrl_lambda_seed_band.{pdf,png}`, with the
frozen single-seed points overlaid as hollow markers.

---

## Frozen protocol (do not change between cells)

| Setting | Value |
| --- | --- |
| surrogate | `hybrid_v3_v35`, v3 rollout + 15-min calibrated v3.5 censor |
| control step | 900 s |
| episode | 14 days |
| comfort band | 21–24 °C |
| observation | `no_delta_t`, power `clipped_log`, t_zone `raw` |
| `lambda_power_disagree` | 5e-5 |
| training | 5 M winter + 7 M summer steps, 16 envs |
| evaluation | live BOPTEST `bestest_air`, 900 s, 14 days, peak + typical heat windows |

Only the seed and `lambda_temp_disagree` vary. Changing anything else makes the band
non-comparable with the single-seed result it is meant to qualify.
