from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "block1_surrogate_fidelity_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-folder bundle for the current Block 1 surrogate-fidelity state.")
    parser.add_argument("--bundle-dir", default=str(BUNDLE_ROOT))
    return parser.parse_args()


def copy_file(src: Path, dst: Path, bundle_dir: Path, manifest: list[dict[str, str]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({"type": "file", "source": str(src.relative_to(REPO_ROOT)), "bundle_path": str(dst.relative_to(bundle_dir))})


def copy_dir(src: Path, dst: Path, bundle_dir: Path, manifest: list[dict[str, str]]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    manifest.append({"type": "dir", "source": str(src.relative_to(REPO_ROOT)), "bundle_path": str(dst.relative_to(bundle_dir))})


def write_readme(bundle_dir: Path) -> None:
    text = """# Block 1 Surrogate Fidelity Bundle

This folder is a non-destructive convenience bundle for the current frozen Block 1 state.

It contains:

- `code/`: canonical code paths used to build, calibrate, and validate Block 1
- `data/`: canonical data files used by the Block 1 story
- `outputs/`: canonical Block 1 output folders
- `reports/`: Block 1 reports, tables, and methodology packaging files
- `manifests/`: bundle manifest

The source repository paths remain unchanged. This folder is a copy bundle only.
"""
    (bundle_dir / "README.md").write_text(text, encoding="utf-8")


def build_bundle(bundle_dir: Path) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []

    code_files = [
        REPO_ROOT / "data" / "collect_tsupply_data.py",
        REPO_ROOT / "surrogate" / "rc_node_v2.py",
        REPO_ROOT / "surrogate" / "rc_node_v35.py",
        REPO_ROOT / "surrogate" / "train_surrogate_backbone.py",
        REPO_ROOT / "surrogate" / "train_surrogate_v2.py",
        REPO_ROOT / "surrogate" / "inverse_problem_boptest_v3.py",
        REPO_ROOT / "surrogate" / "inverse_problem_boptest_v35.py",
        REPO_ROOT / "evaluation" / "validate_surrogate_v35_rollout_prepared.py",
        REPO_ROOT / "evaluation" / "validate_closed_loop_transfer_thermostatic_live.py",
        REPO_ROOT / "evaluation" / "diagnose_thermostatic_obs_transfer_gap.py",
        REPO_ROOT / "evaluation" / "build_hou_evins_appendix_tables.py",
        REPO_ROOT / "evaluation" / "build_hou_evins_open_items_closure.py",
        REPO_ROOT / "evaluation" / "build_block1_fidelity_bundle.py",
        REPO_ROOT / "draft" / "legacy_archive" / "top_level" / "block_1_2_surrogate_rmse" / "data" / "prepare_block12_15min_dataset.py",
        REPO_ROOT / "draft" / "legacy_archive" / "top_level" / "block_1_2_surrogate_rmse" / "data" / "collect_block12_15min_dataset.py",
        REPO_ROOT / "draft" / "legacy_archive" / "top_level" / "block_1_2_surrogate_rmse" / "training" / "train_block12_backbone.py",
    ]

    data_files = [
        REPO_ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv",
        REPO_ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_prepared.csv",
        REPO_ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_collected.csv",
    ]

    output_dirs = [
        REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_block13_closed_loop",
        REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_power_head_only",
        REPO_ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only",
        REPO_ROOT / "outputs" / "block13_closed_loop_transfer_no_delta_t_powerlog_tzone",
        REPO_ROOT / "outputs" / "block13_obs_gap_no_delta_t_powerlog_tzone",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_dataset",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_dataset",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_focus_dataset",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_baseline",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_baseline",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_focus",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_rollout_select",
        REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_rollout_long",
    ]

    report_files = [
        REPO_ROOT / "reports" / "block1_surrogate_final_report.md",
        REPO_ROOT / "reports" / "block1_surrogate_final_metrics.csv",
        REPO_ROOT / "reports" / "hou_evins_compliance_matrix.md",
        REPO_ROOT / "reports" / "hou_evins_partial_closure.md",
        REPO_ROOT / "reports" / "hou_evins_sample_generation_table.csv",
        REPO_ROOT / "reports" / "hou_evins_stage_a_processing_table.csv",
        REPO_ROOT / "reports" / "hou_evins_feature_justification_table.csv",
        REPO_ROOT / "reports" / "hou_evins_architecture_justification_table.csv",
        REPO_ROOT / "reports" / "hou_evins_final_open_items_closure.md",
        REPO_ROOT / "reports" / "hou_evins_sample_size_justification_table.csv",
        REPO_ROOT / "reports" / "hou_evins_split_representativeness_table.csv",
        REPO_ROOT / "reports" / "hou_evins_targeted_sensitivity_table.csv",
        REPO_ROOT / "reports" / "reproduce_current_state_runbook.md",
        REPO_ROOT / "reports" / "reproduction_contours.md",
    ]

    for src in code_files:
        copy_file(src, bundle_dir / "code" / src.relative_to(REPO_ROOT), bundle_dir, manifest)

    for src in data_files:
        copy_file(src, bundle_dir / "data" / src.relative_to(REPO_ROOT / "data"), bundle_dir, manifest)

    for src in output_dirs:
        copy_dir(src, bundle_dir / "outputs" / src.relative_to(REPO_ROOT / "outputs"), bundle_dir, manifest)

    for src in report_files:
        copy_file(src, bundle_dir / "reports" / src.name, bundle_dir, manifest)

    write_readme(bundle_dir)

    manifest_path = bundle_dir / "manifests" / "bundle_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "bundle": "block1_surrogate_fidelity_bundle",
                "source_root": str(REPO_ROOT),
                "item_count": len(manifest),
                "items": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    build_bundle(Path(args.bundle_dir))


if __name__ == "__main__":
    main()
