# Legacy Archive

This folder contains materials that are no longer on the active execution path.

## Structure

- `code_archive/`
  Legacy utility scripts, smoke tests, Sinergym debug helpers, and older reproduction entrypoints.
- `output_archive/`
  Historical outputs, temporary runs, failed branches, legacy Sinergym results, smoke sweeps, and older presentation artifacts.
- `root_archive/`
  Historical root-level files that were removed from the main workspace to reduce clutter.

## Newly archived in this restructuring pass

- `outputs/surrogate_v35_rollout24_fixedczon_full`
  This branch was explicitly rolled back after worsening rollout realism, so it was moved into:
  `draft/legacy_archive/output_archive/failed_rollout_branches/surrogate_v35_rollout24_fixedczon_full`

## What stays active on purpose

The archive does not include the current active research path:

- Block 1 canonical baseline
- Block 1.2 15-minute surrogate improvement path
- current controller benchmark outputs
- current MORL pretrain outputs
- current `results/` and `reports/`

## Rule for future cleanup

Move an artifact here only if at least one of these is true:

1. The branch is explicitly rolled back or rejected.
2. The output is a smoke run or temporary debug run.
3. The script is tied to a legacy Sinergym path no longer on the paper-critical route.
4. The artifact is historical context, not an active benchmark dependency.
