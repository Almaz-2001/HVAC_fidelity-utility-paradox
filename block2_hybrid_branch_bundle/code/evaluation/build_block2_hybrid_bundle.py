from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "block2_hybrid_branch_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-folder bundle for the current Block 2 hybrid branch.")
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
    text = """# Block 2 Hybrid Branch Bundle

This folder is a non-destructive convenience bundle for the current frozen Block 2 hybrid branch.

It contains:

- `code/`: canonical training, adapter, benchmark, and transfer-diagnostic code
- `models/`: canonical and comparison controller checkpoints
- `outputs/`: canonical benchmark, warm-start baseline, and transfer-diagnostic outputs
- `reports/`: Block 2 reports, figures, and evidence-closure tables
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
        REPO_ROOT / "training" / "train_thermostatic.py",
        REPO_ROOT / "training" / "launch_thermostatic_warmstart_benchmark.py",
        REPO_ROOT / "training" / "finetune_tsup_policies_boptest.py",
        REPO_ROOT / "surrogate" / "direct_tsup_adapter.py",
        REPO_ROOT / "envs" / "tsup_features.py",
        REPO_ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py",
        REPO_ROOT / "evaluation" / "validate_closed_loop_transfer_thermostatic_live.py",
        REPO_ROOT / "evaluation" / "diagnose_thermostatic_obs_transfer_gap.py",
        REPO_ROOT / "evaluation" / "build_hybrid_surrogate_snapshot.py",
        REPO_ROOT / "evaluation" / "build_hybrid_evidence_closure.py",
        REPO_ROOT / "evaluation" / "build_block2_hybrid_bundle.py",
    ]

    model_files = [
        REPO_ROOT / "models" / "ppo_thermostatic.zip",
        REPO_ROOT / "models" / "ppo_thermostatic_hybrid_v3_v35_l005.zip",
        REPO_ROOT / "models" / "ppo_thermostatic_hybrid_v3_v35_l010.zip",
        REPO_ROOT / "models" / "ppo_thermostatic_hybrid_v3_v35_l015.zip",
        REPO_ROOT / "models" / "ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone.zip",
    ]

    output_dirs = [
        REPO_ROOT / "outputs" / "surrogate_v2",
        REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_power_head_only",
        REPO_ROOT / "outputs" / "bestest_air_article7_style_15min",
        REPO_ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l010",
        REPO_ROOT / "outputs" / "block2_thermostatic_warmstart_utility",
        REPO_ROOT / "outputs" / "block13_closed_loop_transfer_hybrid_l010",
        REPO_ROOT / "outputs" / "block13_obs_gap_hybrid_l010",
        REPO_ROOT / "outputs" / "block13_closed_loop_transfer_pure_v3",
        REPO_ROOT / "outputs" / "block13_obs_gap_pure_v3",
        REPO_ROOT / "outputs" / "block13_closed_loop_transfer_no_delta_t_powerlog_tzone",
        REPO_ROOT / "outputs" / "block13_obs_gap_no_delta_t_powerlog_tzone",
        REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_hybrid_v3_v35_l005",
        REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_hybrid_v3_v35_l015",
    ]

    report_files = [
        REPO_ROOT / "reports" / "block2_hybrid_surrogate_report.md",
        REPO_ROOT / "reports" / "block2_hybrid_surrogate_metrics.csv",
        REPO_ROOT / "reports" / "hybrid_evidence_closure.md",
        REPO_ROOT / "reports" / "hybrid_disagreement_summary.csv",
        REPO_ROOT / "reports" / "hybrid_transfer_comparison.csv",
        REPO_ROOT / "reports" / "hybrid_evidence_status.md",
        REPO_ROOT / "reports" / "structured_plan_status.md",
        REPO_ROOT / "results" / "minimum_paper_suite" / "docs" / "next_steps.md",
    ]

    figure_files = [
        REPO_ROOT / "reports" / "figures" / "hybrid_boptest_comfort_traces.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_boptest_power_energy_traces.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_vs_pi_ms.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_vs_pi_violation.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_vs_pi_energy.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_disagreement_summary.png",
        REPO_ROOT / "reports" / "figures" / "hybrid_transfer_gap_comparison.png",
    ]

    for src in code_files:
        copy_file(src, bundle_dir / "code" / src.relative_to(REPO_ROOT), bundle_dir, manifest)

    for src in model_files:
        copy_file(src, bundle_dir / "models" / src.name, bundle_dir, manifest)

    for src in output_dirs:
        if src.parts[-3:-1] == ("legacy_archive", "outputs"):
            rel = Path("legacy_archive_outputs") / src.name
            copy_dir(src, bundle_dir / "outputs" / rel, bundle_dir, manifest)
        else:
            copy_dir(src, bundle_dir / "outputs" / src.relative_to(REPO_ROOT / "outputs"), bundle_dir, manifest)

    for src in report_files:
        target = bundle_dir / "reports" / src.name
        if src.parts[-3:] == ("docs", "next_steps.md"):
            target = bundle_dir / "reports" / "next_steps.md"
        copy_file(src, target, bundle_dir, manifest)

    for src in figure_files:
        copy_file(src, bundle_dir / "reports" / "figures" / src.name, bundle_dir, manifest)

    write_readme(bundle_dir)

    manifest_path = bundle_dir / "manifests" / "bundle_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "bundle": "block2_hybrid_branch_bundle",
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
