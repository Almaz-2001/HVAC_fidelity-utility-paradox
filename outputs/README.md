# `outputs/` — curated audit trail

This folder is **not** the raw run output. The full per-seed run artifacts
(~7 GB) are not shipped. What is here is the small, version-locked subset that the
manuscript's provenance maps point to, so that every figure, table, and number can be
traced to a source artifact (Level A) and regenerated without BOPTEST (Level B):

* the **surrogate checkpoints** that are the *inputs* to the controller experiments
  and to the reproducibility probes (`run_block2.py surface-diagnostic`, the rollout
  validators), and
* the **calibration summaries, manifests, and metric tables** that the provenance maps
  in [`../roadmap.md`](../roadmap.md) (§3.2 Block 1, §11.1 Block 2, §15.7 Block 3)
  reference by path.

> The exact figure/table → file mapping lives in [`../roadmap.md`](../roadmap.md); this
> README only decodes the directory names so the layout is legible at a glance.

## Naming legend

**File types**

| Pattern | What it is |
|---|---|
| `rc_node_*.pt` | trained surrogate weights (the RC-node / RC–Neural-ODE model family) |
| `calibration_summary_*.json` | v3.5 grey-box calibration summary (fitted physical params incl. `C_zon`) |
| `summary.csv`, `*_summary.csv` | controller benchmark KPIs (`m_s`, comfort violation %, energy) |
| `horizon_metrics.csv` | multi-horizon (1/4/8/24 h) rollout RMSE on the prepared corpus |
| `*manifest*.json` | scenario / pipeline / audit manifests (the pre-registration anchors) |
| `train_history*.csv` | training curves |

**Directory name fragments**

| Fragment | Meaning |
|---|---|
| `surrogate_v2` | black-box surrogate **v3** (hourly), file `rc_node_v3_tsupply.pt` |
| `surrogate_v3_15min_matched` | the matched-resolution v3 (Tactic-B reviewer control) |
| `*_inverse_boptest_*` | **v3.5** grey-box inverse-calibration runs on BOPTEST data |
| `*_rollout_prepared_15min*` | offline rollout-fidelity metrics on the prepared 15-min corpus |
| `bestest_air_article7_style*` | **pure-v3** thermostatic PPO controller (`article7_style` = legacy recipe tag) |
| `block2_thermostatic_hybrid_*_l010` | the canonical **hybrid** controller (censor weight λ_T = 0.10) |
| `block2_hdrl_*_l000..l010` | HDRL censor-weight sweep |
| `morl_pareto_*`, `morl_hybrid_*`, `morl_5d_legacy_rerun` | MORL Pareto front / agents |
| `pi_baseline_15min_yearly` | PI baseline (yearly) |
| `block3_*` | transferability testcases (hydronic, heat-pump, commercial-hydronic) |
| `block13_*` | Block 1→3 observation-gap and closed-loop transfer diagnostics |
| `article22_*`, `*_article22_gru` | architecture-ablation backends |
| `*_power_head_only`, `*_episodeaware`, `*_multistart`, `*_prior420_heads_only` | v3.5 calibration variants (head scope / episode handling / multi-start / prior) |

## Layout by block

**Block 1 — surrogate fidelity (inputs + rollout metrics)**
`surrogate_v2/`, `surrogate/`, `surrogate_v3_15min_matched/`,
`surrogate_v35_inverse_boptest_*` (v3.5 calibration variants),
`surrogate_v3_rollout_prepared_15min/`, `surrogate_v35_rollout_prepared_15min*/`,
`block_1_2_surrogate_rmse/`.

**Block 2 — downstream control utility (controller KPIs)**
`bestest_air_article7_style*/` (pure v3), `block2_thermostatic_hybrid_v3_v35_l010/`
(canonical hybrid), `block2_thermostatic_warmstart_utility/` (warm-start negative
control), `block2_bestest_air_15min_thermostatic_v35/` (direct v3.5),
`block2_hdrl_*/` (HDRL sweep), `morl_*/` (MORL), `pi_baseline_15min_yearly/`,
`article22_mlp_heat/`.

**Block 3 — pre-registered transferability**
`block3_bestest_hydronic/`, `block3_bestest_hydronic_heat_pump/`,
`block3_singlezone_commercial_hydronic/` (per-testcase calibration + manifests),
`block13_*/` (observation-gap and closed-loop transfer diagnostics).
