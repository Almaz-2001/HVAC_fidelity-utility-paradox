# Roadmap Dependency Table

This file is the single source of truth for which prerequisite sections must
already be complete before a given section in `roadmap.md` can run. It is
referenced from `roadmap.md` Section 0.5 and is intentionally kept as a
separate file so that the main roadmap stays focused on commands while this
file stays focused on structural dependencies.

Read order: a row's "Reads from" column lists the prerequisite sections that
produce the inputs that row's section consumes. A blank "Reads from" cell
means the section is independent of compute closure.

## Compute-producing sections (Block 1, Block 2, Block 3)

| Section | Title                                                    | Reads from                                  | Produces outputs consumed by |
|---------|----------------------------------------------------------|---------------------------------------------|------------------------------|
| 1       | Block 1: v3 Direct-TSup Surrogate                        | --                                          | 2, 3, 4, 5, 6, 8, 9, 15      |
| 2       | Block 1: v3.5 Inverse Calibration                        | 1 (semantic; not strictly required)         | 2.5, 3, 4.5, 5, 5.5, 6, 8, 9, 15 |
| 2.5     | Block 1: Corpus-Matched v3 Retraining (reviewer mitigation) | 1, 2                                     | 3 (reviewer-mitigation row in §5.3), 4.6 |
| 3       | Block 1: Article-Facing Fidelity Tables and Figures      | 1, 2, 2.5 (optional reviewer-mitigation row) | 11                           |
| 4       | Block 2: First Control Baseline -- Pure v3 Thermostatic PPO | 1                                       | 4.5, 5.5, 11                 |
| 4.5     | Block 2 Negative Control: Direct v3.5 Warm-Start         | 2, 4                                        | 5, 5.5, 11                   |
| 4.6     | Block 2: Corpus-Matched v3 Closed-Loop Control Utility (reviewer mitigation) | 2.5, 4                  | 11 (reviewer-mitigation row in §8.5) |
| 5       | Block 2: Thermostatic Hybrid Sweep                       | 1, 2, 4.5                                   | 5.5, 11, 15                  |
| 5.5     | Block 1.3 / Block 2 Transfer Diagnostics                 | 2, 4, 4.5, 5                                | 11                           |
| 6       | Block 2: HDRL Sweep                                      | 1, 2                                        | 11                           |
| 6.5     | MORL 5D Observation Failure (frozen artifact + reconstructed rerun) | 1, 2 (only for reconstructed rerun) | 11                           |
| 7       | Block 2: MORL 17D Power-Only Backend (descriptive)       | --                                          | (preamble to Section 8)      |
| 8       | MORL Pareto Sweep                                        | 1, 2                                        | 11                           |
| 9       | MORL Canonical Seed Analysis                             | 1, 2, 8                                     | 11, 15                       |
| 10      | PI Baseline                                              | --                                          | 11                           |
| 15      | Block 3 Execution: Transferability on Hydronic Family    | 1, 2, 5, 9, 14                              | 16, 17                       |

## Article-build sections

| Section | Title                                                    | Reads from                                  | Produces outputs consumed by |
|---------|----------------------------------------------------------|---------------------------------------------|------------------------------|
| 11      | Rebuild Block 2 Tables and Figures                       | 3, 4, 4.5, 4.6, 5, 5.5, 6, 6.5, 8, 9, 10    | 12                           |
| 12      | Rebuild the Word Article Skeleton                        | 11                                          | (final manuscript)           |
| 17      | Paper Manuscript Build Path + Artifact Consolidation     | 11, 12, 15                                  | `paper_artifacts/` canonical paper package |

## Manuscript section packages (Overleaf, `docs/`)

Each results block has a standalone, data-driven Overleaf package that
regenerates its manuscript section, figures, and tables directly from the
`reports/` and `outputs/` artifacts produced by the sections above. These are
section-build steps (not compute), analogous to Sections 11-12 but per-block and
LaTeX-facing.

| Package (builder)                                            | Reads from (roadmap sections)        | Produces / status |
|--------------------------------------------------------------|--------------------------------------|-------------------|
| Results I  (`build_results1_overleaf.py`)                    | 1, 2, 2.5, 3                         | `docs/results1_digital_twin_overleaf/` — **done** (provenance map: roadmap 3.2) |
| Results II (`build_results2_overleaf.py`, planned)           | 4, 4.5, 5, 5.5, 6, 6.5, 8, 9, 10, 11 | `docs/results2_control_overleaf/` — planned (provenance map: roadmap 11.1) |
| Results III (`build_results3_overleaf.py`)                   | 15                                   | `docs/results3_transferability_overleaf/` — **done** (provenance map: roadmap 15.7) |

## Independent sections (no compute closure)

| Section | Title                                                    | Role                                        |
|---------|----------------------------------------------------------|---------------------------------------------|
| 0       | Runtime Checks                                           | Container health / BOPTEST RTE lifecycle    |
| 3.1     | Boundary: why direct-v3.5 failure and hybridization are in Block 2 | Explanatory bridge; no compute; maps Block 1 artifacts to Block 2 controller-backend tests |
| 13      | Audit Anchors                                            | Reference list of pre-registration commits  |
| 13.5    | Pre-Block-3 Cleanup Workflow                             | Repository hygiene before Block 3 opens     |
| 14      | Block 3: Transferability Pre-Registration                | Manifest; must be committed before Section 15 runs |
| 16      | Audit Anchor Chain (Updated)                             | Reference list including Block 3 closure SHA |

## Notes

- **Section 2 reads from Section 1**: technically v3.5 calibration does not
  require the v3 backbone checkpoint to exist, but the two are semantic
  extensions of the same surrogate family and share data conventions; for
  reproduction order we keep Section 2 after Section 1.
- **Section 14 is a manifest, not compute**: it must be committed (and the
  commit SHA recorded as the third audit anchor) before any Section 15 run.
  Otherwise Block 3 cannot claim pre-registered status.
- **Section 6.5 has two evidence layers**: the original 5D MORL failure is
  preserved as the frozen reference CSV in `reports/block2_morl_comparison_summary.csv`.
  For reviewer-side reproducibility, the current codebase also provides a
  reconstructed legacy 5D preset under `configs/morl_surrogate_ppo_5d/`
  (`obs_mode: basic`). The reconstructed rerun reads the same Block 1
  surrogate artifacts as the 17D path but does not modify the pre-registered
  17D canonical seed-analysis audit trail. The main-paper ablation should use
  the reconstructed current-code result (`m_s=0.680`, `violation=37.77%`,
  `RMSE_T=2.721 C`) against the 17D reference (`m_s=0.099`,
  `violation=4.9%`, `RMSE_T=0.72 C`). The old frozen 5D artifact remains only
  as an audit-preserved non-current artifact; it is not a main-paper result and
  must not be silently overwritten or deleted.
- **MORL has no separate lambda-temperature sweep prerequisite**: Sections 6.5,
  8, and 9 all use the fixed MORL power-only hybrid backend
  (`lambda_temp_disagree=0.00`, `lambda_power_disagree=5e-5`). The lambda
  sweeps are controller-family-specific for thermostatic PPO and HDRL. The
  MORL claim is therefore about the observation interface and Pareto/canonical
  behavior under the fixed power-only backend, not about proving a globally
  optimal MORL regularization weight.
- **Section 3.1 is not an execution step**: direct-v3.5 failure and the
  hybrid backend are intentionally placed in Block 2 because they require
  trained controllers and live BOPTEST transfer. They consume Block 1
  artifacts (`v3` rollout checkpoint and calibrated `v3.5` reference), but
  they are not additional Block 1 surrogate-calibration commands.
- **Block 2 execution order is intentional**: Section 4 establishes the pure
  v3 controller baseline first; Section 4.5 then tests the negative direct
  v3.5 warm-start hypothesis; Section 5 runs the hybrid sweep only after that
  negative result has motivated the architecture. Section 5.5 is diagnostic
  aggregation, not the first place where the v3.5 failure is introduced.
- **Section 4.5 and Section 5.5 train different v3.5-based artifacts**:
  Section 4.5's v3.5 model is only a pretraining checkpoint that is later
  fine-tuned in BOPTEST and compared with a scratch BOPTEST fine-tune.
  Section 5.5's `v35_direct` checkpoint is a standalone zero-shot diagnostic
  controller used to measure live-transfer failure and action/trajectory gap.
  They share the calibrated v3.5 surrogate input but answer different
  experimental questions.
- **Section 15 has two manifest-style prerequisites**: Section 14 (Block 3
  pre-registration), and the Block 2 frozen models referenced in the
  manifest (Sections 5 and 9). All three must be in place before any
  Section 15 cell runs.
- **Section 4.6 is a closed-loop reviewer mitigation, distinct from Section 2.5**:
  Section 2.5 retrains v3 on the matched 15-min corpus and validates it only as a
  *predictor* (rollout RMSE). Section 4.6 takes that same matched checkpoint and runs
  it through the full PPO *control* pipeline with the identical recipe as the canonical
  pure-v3 baseline (Section 4), then benchmarks it live on the two targeted windows.
  It answers the downstream-utility question that 2.5's predictor metric cannot: it
  isolates whether v3's training utility is its black-box smoothness or the train/control
  timestep mismatch. It produces only `reports/block2_v3_15min_closed_loop_comparison.*`
  and `outputs/bestest_air_pure_v3_15min/`; it does not touch any canonical downstream
  artifact (the PPO families still use the hourly v3).
- **Section 17 is the only paper-facing artifact consolidation step**:
  raw model/run artifacts remain in `outputs/` (ignored by Git), compact
  evidence is written to `reports/*.csv`, and the GitHub-facing package is
  assembled under `paper_artifacts/` by
  `evaluation/organize_paper_artifacts.py`.
- **Independent sections have no compute-closure dependency** but still have
  a temporal ordering inside the roadmap (e.g., audit anchors in Section 13
  presuppose the MORL canonical work of Section 9 has been committed).
- **Manuscript section packages are data-driven and read-only**: the
  per-block Overleaf builders under `docs/` consume `reports/`/`outputs/`
  artifacts and never modify model or pipeline code. They depend only on the
  artifact-producing sections listed in their table row, so Results II cannot be
  finalized until its Block 2 inputs (Sections 4-11) exist, and Results III
  until Section 15 closes. Results I is the reference implementation; Results II
  and III mirror its structure (provenance map, nomenclature/SI, limitations,
  data-driven tables/figures). The power-channel checkpoint caveat from
  Results I (canonical `power_head_only`, not the intermediate `episodeaware`)
  applies wherever calibrated v3.5 power is reported downstream.
