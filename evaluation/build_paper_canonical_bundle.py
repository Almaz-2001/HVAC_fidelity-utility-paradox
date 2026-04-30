from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "paper_canonical_bundle"


@dataclass(frozen=True)
class Artifact:
    source: str
    destination: str
    category: str


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("reports/block1_surrogate_final_report.md", "docs/block1_surrogate_final_report.md", "docs"),
    Artifact("reports/block2_hybrid_surrogate_report.md", "docs/block2_hybrid_surrogate_report.md", "docs"),
    Artifact("reports/structured_plan_status.md", "docs/structured_plan_status.md", "docs"),
    Artifact("reports/project_workspace_map.md", "docs/project_workspace_map.md", "docs"),
    Artifact("reports/hybrid_evidence_status.md", "docs/hybrid_evidence_status.md", "docs"),
    Artifact("results/minimum_paper_suite/docs/next_steps.md", "docs/next_steps.md", "docs"),
    Artifact("reports/block1_surrogate_final_metrics.csv", "tables/block1_surrogate_final_metrics.csv", "tables"),
    Artifact("reports/block2_hybrid_surrogate_metrics.csv", "tables/block2_hybrid_surrogate_metrics.csv", "tables"),
    Artifact(
        "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/v35_prepared_compare_summary.csv",
        "tables/block1_rollout_compare_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv",
        "tables/block2_hybrid_l010_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/bestest_air_article7_style_15min/summary.csv",
        "tables/v3_article7_style_15min_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv",
        "tables/v35_warmstart_negative_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone/summary.csv",
        "tables/block1_zero_shot_transfer_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/block13_obs_gap_no_delta_t_powerlog_tzone/first_divergence_summary.csv",
        "tables/block1_first_divergence_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/block2_bestest_air_15min_thermostatic_v35/summary.csv",
        "tables/pi_and_failed_v35_summary.csv",
        "tables",
    ),
    Artifact(
        "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json",
        "manifests/block1_canonical_v35_calibration_summary.json",
        "manifests",
    ),
    Artifact(
        "outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop/calibration_summary_boptest_v35.json",
        "manifests/block1_best_temp_alignment_summary.json",
        "manifests",
    ),
    Artifact(
        "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_trace_21_24_vs_boptest.png",
        "figures/block1_comfort_trace_21_24_vs_boptest.png",
        "figures",
    ),
    Artifact(
        "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/hvac_power_trace_vs_boptest.png",
        "figures/block1_hvac_power_trace_vs_boptest.png",
        "figures",
    ),
    Artifact(
        "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/cumulative_energy_trace_vs_boptest.png",
        "figures/block1_cumulative_energy_trace_vs_boptest.png",
        "figures",
    ),
    Artifact(
        "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/comfort_violation_comparison.png",
        "figures/block1_comfort_violation_comparison.png",
        "figures",
    ),
    Artifact("reports/figures/hybrid_boptest_comfort_traces.png", "figures/hybrid_boptest_comfort_traces.png", "figures"),
    Artifact("reports/figures/hybrid_boptest_power_energy_traces.png", "figures/hybrid_boptest_power_energy_traces.png", "figures"),
    Artifact("reports/figures/hybrid_vs_pi_ms.png", "figures/hybrid_vs_pi_ms.png", "figures"),
    Artifact("reports/figures/hybrid_vs_pi_violation.png", "figures/hybrid_vs_pi_violation.png", "figures"),
    Artifact("reports/figures/hybrid_vs_pi_energy.png", "figures/hybrid_vs_pi_energy.png", "figures"),
    Artifact("models/ppo_thermostatic.zip", "models/ppo_thermostatic_v3_baseline.zip", "models"),
    Artifact("models/ppo_thermostatic_hybrid_v3_v35_l010.zip", "models/ppo_thermostatic_hybrid_v3_v35_l010.zip", "models"),
    Artifact("models/ppo_winter_final.zip", "models/ppo_hdrl_winter_v3_baseline.zip", "models"),
    Artifact("models/ppo_summer_final.zip", "models/ppo_hdrl_summer_v3_baseline.zip", "models"),
    Artifact(
        "outputs/morl_surrogate_ppo_v35_calibrated/seed42/finetune_boptest/models/ppo_model.zip",
        "models/ppo_morl_v35_calibrated_finetuned.zip",
        "models",
    ),
)


def build_bundle() -> list[dict[str, str]]:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for artifact in ARTIFACTS:
        src = REPO_ROOT / artifact.source
        dst = BUNDLE_ROOT / artifact.destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            status = "copied"
        else:
            status = "missing"
        records.append(
            {
                "category": artifact.category,
                "source": artifact.source,
                "destination": artifact.destination,
                "status": status,
            }
        )
    return records


def write_manifest(records: list[dict[str, str]]) -> None:
    manifest_csv = BUNDLE_ROOT / "artifact_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source", "destination", "status"])
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "bundle_root": str(BUNDLE_ROOT.relative_to(REPO_ROOT)),
        "copied": sum(1 for r in records if r["status"] == "copied"),
        "missing": [r for r in records if r["status"] == "missing"],
    }
    (BUNDLE_ROOT / "bundle_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_readme(records: list[dict[str, str]]) -> None:
    copied = sum(1 for r in records if r["status"] == "copied")
    missing = [r for r in records if r["status"] == "missing"]
    lines = [
        "# Paper Canonical Bundle",
        "",
        "This folder is the single article-facing bundle.",
        "",
        "## Structure",
        "",
        "- `docs/`",
        "- `figures/`",
        "- `tables/`",
        "- `models/`",
        "- `manifests/`",
        "",
        "## Status",
        "",
        f"- copied artifacts: `{copied}`",
        f"- missing artifacts: `{len(missing)}`",
        "",
        "## Canonical Intent",
        "",
        "- `docs/` is the narrative layer for writing and presentation.",
        "- `figures/` is the canonical source for article plots.",
        "- `tables/` contains the CSVs used for claims and comparisons.",
        "- `models/` contains the current zip models needed for controller comparison.",
        "- original experiments remain in `outputs/` and `models/` for reproducibility.",
    ]
    if missing:
        lines.extend(["", "## Missing Artifacts", ""])
        for item in missing:
            lines.append(f"- `{item['source']}`")
    (BUNDLE_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = build_bundle()
    write_manifest(records)
    write_readme(records)


if __name__ == "__main__":
    main()
