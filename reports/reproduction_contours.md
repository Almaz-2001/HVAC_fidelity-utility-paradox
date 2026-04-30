# Reproduction Contours

Date: 2026-04-30

## Purpose

This note answers one strict question:

**What exactly must another engineer have in order to reproduce our results up to the current frozen state?**

This file is not the execution runbook itself.

The step-by-step command sequence is in:

- [reproduce_current_state_runbook.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/reproduce_current_state_runbook.md)

This file only defines the required project contour.

## Short Answer

If your friend wants to reproduce **all current results**, the required contour is:

- **not** `minimal_active_contour`
- **yes** `full_reproduction_contour`

In addition, they also need:

- a working Python environment
- a running BOPTEST service
- the testcase `bestest_air`

Without those external runtime pieces, copying the repository alone is not enough.

## 1. `minimal_active_contour`

This is the minimum code contour for ongoing active development.

It is enough to:

- understand the current architecture
- inspect the active training/evaluation code
- continue new experiments from the current state

It is **not enough** for exact reproduction of all frozen results.

Contents:

- [surrogate](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate)
- [training](C:/Users/user/Desktop/HVAC_DRL_MORL/training)
- [evaluation](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation)
- [envs](C:/Users/user/Desktop/HVAC_DRL_MORL/envs)
- [configs](C:/Users/user/Desktop/HVAC_DRL_MORL/configs)
- [data](C:/Users/user/Desktop/HVAC_DRL_MORL/data)
- [reports](C:/Users/user/Desktop/HVAC_DRL_MORL/reports)

## 2. `full_reproduction_contour`

This is the contour required to reproduce the project up to the current frozen state.

### 2.1 Mandatory repository directories

- [surrogate](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate)
- [training](C:/Users/user/Desktop/HVAC_DRL_MORL/training)
- [evaluation](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation)
- [envs](C:/Users/user/Desktop/HVAC_DRL_MORL/envs)
- [configs](C:/Users/user/Desktop/HVAC_DRL_MORL/configs)
- [data](C:/Users/user/Desktop/HVAC_DRL_MORL/data)
- [reports](C:/Users/user/Desktop/HVAC_DRL_MORL/reports)
- [models](C:/Users/user/Desktop/HVAC_DRL_MORL/models)
- [outputs](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs)
- [paper_canonical_bundle](C:/Users/user/Desktop/HVAC_DRL_MORL/paper_canonical_bundle)
- [draft/legacy_archive](C:/Users/user/Desktop/HVAC_DRL_MORL/draft/legacy_archive)

### 2.2 Mandatory top-level files

- [README.md](C:/Users/user/Desktop/HVAC_DRL_MORL/README.md)
- [requirements.txt](C:/Users/user/Desktop/HVAC_DRL_MORL/requirements.txt)
- [Rules.md](C:/Users/user/Desktop/HVAC_DRL_MORL/Rules.md)

### 2.3 Mandatory external runtime

These are not just repository files. They are runtime requirements.

- Python with required packages:
  - `torch`
  - `gymnasium`
  - `stable_baselines3`
  - `pandas`
  - `numpy`
  - `requests`
- BOPTEST service reachable from the commands
- testcase `bestest_air`

## 3. Why `full_reproduction_contour` is necessary

The current frozen project state depends on more than the active code roots.

### 3.1 Why `models/` is mandatory

Because several current results are benchmarked from trained controller checkpoints:

- `ppo_thermostatic.zip`
- `ppo_winter_final.zip`
- `ppo_summer_final.zip`
- `ppo_thermostatic_hybrid_v3_v35_l010.zip`
- other diagnostic thermostatic checkpoints

Without `models/`, your friend must retrain everything from scratch.

That is possible, but it is no longer a guaranteed reproduction of the frozen current state.

### 3.2 Why `outputs/` is mandatory

Because the current state includes frozen artifacts, not just code:

- calibration summaries
- rollout summaries
- benchmark CSV tables
- diagnostic manifests
- trace folders

Several final reports and bundle builders read directly from `outputs/`.

### 3.3 Why `paper_canonical_bundle/` is mandatory

Because it is the current article-facing frozen package.

It is needed if your friend wants to verify:

- that the final figures and tables were rebuilt correctly
- that the current article bundle matches the frozen state

### 3.4 Why `draft/legacy_archive/` is mandatory

Because part of the reproduction path now uses archived scripts and frozen historical branches.

In particular:

- Block 1.2 helper scripts were moved there
- some 15-minute dataset preparation paths now live there

Without `draft/legacy_archive/`, the runbook is incomplete.

## 4. What can be omitted

These paths are **not mandatory** for reproducing the current frozen result set.

### 4.1 Safe to omit in most cases

- [docs](C:/Users/user/Desktop/HVAC_DRL_MORL/docs)
- [results](C:/Users/user/Desktop/HVAC_DRL_MORL/results)
- [.vscode](C:/Users/user/Desktop/HVAC_DRL_MORL/.vscode)
- [.devcontainer](C:/Users/user/Desktop/HVAC_DRL_MORL/.devcontainer)
- [__pycache__](C:/Users/user/Desktop/HVAC_DRL_MORL/__pycache__)

Reason:

- they help navigation or presentation
- but they are not required by the current runbook

### 4.2 Usually safe to omit, with one caveat

- [layers](C:/Users/user/Desktop/HVAC_DRL_MORL/layers)

Reason:

- current Block 1 and current hybrid Block 2 path do not depend on it

Caveat:

- if your friend wants to reproduce older safety-layer or MORL-safety branches, `layers/` may become relevant again

### 4.3 Can be omitted if BOPTEST is provided externally

- [boptest_rte](C:/Users/user/Desktop/HVAC_DRL_MORL/boptest_rte)
- [docker](C:/Users/user/Desktop/HVAC_DRL_MORL/docker)
- [Dockerfile](C:/Users/user/Desktop/HVAC_DRL_MORL/Dockerfile)

Reason:

- current commands only require BOPTEST to exist
- they do not require that BOPTEST be launched from this repository

If your friend wants the same local container workflow, then these become useful, but they are not logically mandatory if the service already exists elsewhere.

## 5. What is mandatory for exact current-state reproduction

If the goal is:

**"I want to reproduce all our results up to the current frozen state exactly as we now present them"**

then the mandatory set is:

### Mandatory code

- [surrogate](C:/Users/user/Desktop/HVAC_DRL_MORL/surrogate)
- [training](C:/Users/user/Desktop/HVAC_DRL_MORL/training)
- [evaluation](C:/Users/user/Desktop/HVAC_DRL_MORL/evaluation)
- [envs](C:/Users/user/Desktop/HVAC_DRL_MORL/envs)
- [configs](C:/Users/user/Desktop/HVAC_DRL_MORL/configs)
- [data](C:/Users/user/Desktop/HVAC_DRL_MORL/data)
- [reports](C:/Users/user/Desktop/HVAC_DRL_MORL/reports)

### Mandatory frozen artifacts

- [models](C:/Users/user/Desktop/HVAC_DRL_MORL/models)
- [outputs](C:/Users/user/Desktop/HVAC_DRL_MORL/outputs)
- [paper_canonical_bundle](C:/Users/user/Desktop/HVAC_DRL_MORL/paper_canonical_bundle)
- [draft/legacy_archive](C:/Users/user/Desktop/HVAC_DRL_MORL/draft/legacy_archive)

### Mandatory runtime

- Python environment with the required packages
- BOPTEST
- testcase `bestest_air`

## 6. Practical Recommendation for Your Friend

If your friend wants the highest probability of success, the correct instruction is:

1. copy the full repository
2. do **not** strip it down to the seven active code folders
3. keep:
   - `models/`
   - `outputs/`
   - `paper_canonical_bundle/`
   - `draft/legacy_archive/`
4. then follow:
   - [reproduce_current_state_runbook.md](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/reproduce_current_state_runbook.md)

## Final Verdict

For exact current-state reproduction:

- `minimal_active_contour` is **insufficient**
- `full_reproduction_contour` is **necessary**

That is the correct operational answer.
