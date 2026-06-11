# Reproducing the results

This repository supports two levels of reproduction. **Level B** regenerates every
figure, table, and reported number of Blocks 1–3 directly from the shipped
artifacts — no BOPTEST and no GPU, in minutes. **Level C** re-runs the experiments
end-to-end and requires a BOPTEST instance and substantial compute.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Level B — regenerate all figures, tables, and numbers (no BOPTEST)

Each block has one data-driven generator. Run from the repository root; each reads
the committed `reports/`, `outputs/` (audit artifacts), and `data/` files and writes
the section's `main.tex` plus its figures.

```bash
# Block 1 — digital-twin fidelity (surrogate accuracy, C_zon, Fisher CI, physics audit)
python docs/results1_digital_twin_overleaf/build_results1_overleaf.py

# Block 2 — downstream control utility (fidelity-utility paradox, hybrid, HDRL, MORL)
python docs/results2_control_overleaf/build_results2_overleaf.py

# Block 3 — pre-registered transferability (transfer matrix, C_zon invariance)
python docs/results3_transferability_overleaf/build_results3_overleaf.py
```

Each command (re)creates the block's figures and prints the block's section text
with every number substituted from the source artifacts. Every figure and every
data-driven number (e.g. the 0.644 °C rollout RMSE, the `m_s = 1.046 / 0.041`
controller scores, the 60.2–87.8 % transfer-RMSE gains) is recomputed from the
committed `reports/`/`outputs/` files and matches the published manuscript.

> **Scope note.** This is a results-reproducibility package: the typeset LaTeX
> manuscript and supplementary PDFs (the journal's artifact) are not included. What
> *is* here is the machinery that produces the results they report — so every
> number in the paper can be regenerated from the data by the generators above, and
> traced to its source artifact through the provenance maps in
> [`roadmap.md`](roadmap.md) (§3.2, §11.1, §15.7). The manuscript figures are also
> committed under `docs/paper_combined/figures/`.

---

## Level C — re-run the experiments end-to-end (needs BOPTEST + compute)

This re-derives the underlying data rather than re-plotting it. It requires the
BOPTEST emulator and is computationally heavy (millions of policy-gradient steps
across `N = 5` seeds and several controller families; hours to days).

1. **Install BOPTEST** (not vendored here) from
   <https://github.com/ibpsa/project1-boptest> (Docker recommended) and start the
   `bestest_air` testcase (plus the `bestest_hydronic` family for Block 3). Point
   the environments in `envs/` and `configs/` at your local runtime URL.

2. **Block 1 — surrogates and calibration:**
   ```bash
   python surrogate/train_surrogate_v2.py                 # black-box v3
   python surrogate/calibrate_surrogate_v35.py            # physics-informed v3.5 (Stage A/B/C)
   python evaluation/run_block1.py                        # rollout fidelity + reports
   ```

3. **Block 2 — controllers (surrogate-pretrain → BOPTEST):**
   ```bash
   python training/run_morl_surrogate_pipeline.py --mode full --seed 42   # primary MORL line
   python evaluation/run_block2.py build-reports                          # KPI tables/figures
   ```
   Thermostatic-PPO and HDRL lines have their own `training/train_*.py` runners.

4. **Block 3 — pre-registered transferability:**
   ```bash
   python evaluation/run_block3_surrogate_recalibration.py   # Stage A/B/C on the hydronic family
   ```

**Determinism note.** Exact bit-for-bit reproduction is not expected: results
depend on the BOPTEST version, RNG seeds, hardware, and library versions (the
throughput figures in particular are hardware-dependent). The scientific
conclusions — the fidelity–utility paradox, controller-family specificity, and the
component-level transfer boundary — are robust to these factors and are what
Level C verifies.

---

## What each level establishes

| Level | Needs BOPTEST? | Time | Verifies |
|-------|----------------|------|----------|
| **A** (inspect) | no | minutes | every number traces to a committed artifact via [`roadmap.md`](roadmap.md) |
| **B** (regenerate) | no | minutes | the figures/tables faithfully represent the committed data |
| **C** (re-run) | yes | hours–days | the committed data itself can be re-derived from the experiments |

Provenance for every figure, table, and number is mapped in
[`roadmap.md`](roadmap.md) (§3.2 Block 1, §11.1 Block 2, §15.7 Block 3) and
mirrored as Supplementary Tables S1–S3 in the manuscript.
