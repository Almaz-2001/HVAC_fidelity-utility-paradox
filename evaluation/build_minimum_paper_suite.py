from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "minimum_paper_suite"
TABLE_DIR = OUT_ROOT / "tables"
DOC_DIR = OUT_ROOT / "docs"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def load_csv_if_exists(rel_path: str) -> pd.DataFrame | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    return pd.read_csv(path)


def surrogate_fidelity_status() -> tuple[str, str, str]:
    df = load_csv_if_exists("results/research_benchmark/tables/surrogate_rollout_benchmark.csv")
    required_columns = [
        "one_step_rmse_c",
        "rollout_4h_rmse_c",
        "rollout_8h_rmse_c",
        "rollout_24h_rmse_c",
        "mean_episode_bias_c",
        "mean_episode_power_rmse_w",
        "c_zon_error_pct",
    ]

    if df is None or df.empty:
        return (
            "pending",
            "no canonical surrogate rollout table found yet",
            "run live v3.5 rollout validation and rebuild the research benchmark",
        )

    existing_bits: list[str] = []
    missing_bits: list[str] = []
    for column in required_columns:
        if column in df.columns and df[column].notna().all():
            existing_bits.append(column)
        else:
            missing_bits.append(column)

    status = "implemented" if not missing_bits else "partial"
    what_exists = ", ".join(existing_bits) if existing_bits else "table exists but required metrics are missing"
    what_missing = ", ".join(missing_bits) if missing_bits else "none"
    return status, what_exists, what_missing


def current_status_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    surrogate_status, surrogate_exists, surrogate_missing = surrogate_fidelity_status()

    rows.append(
        {
            "block": "1_surrogate_fidelity",
            "requirement": "raw surrogate + calibrated surrogate + one-step/4h/8h/24h + bias + power RMSE + C_zon error",
            "status": surrogate_status,
            "what_exists": surrogate_exists,
            "what_missing": surrogate_missing,
            "primary_output": "results/research_benchmark/tables/surrogate_rollout_benchmark.csv",
        }
    )

    rows.append(
        {
            "block": "2_main_controller_benchmark",
            "requirement": "PI + thermostatic + HDRL on peak heat + typical heat, ideally at 15 min",
            "status": "partial",
            "what_exists": "15 min peak/typical heat benchmark for thermostatic and HDRL; PI available in yearly benchmark family",
            "what_missing": "one single canonical 15 min comparison table including PI for the same protocol",
            "primary_output": "outputs/bestest_air_article7_style_15min/summary.csv",
        }
    )

    rows.append(
        {
            "block": "3_predictive_information_ablation",
            "requirement": "M.1 vs M.2 vs M.3 at least on thermostatic task",
            "status": "partial",
            "what_exists": "M.2 proxy exists (ppo_thermostatic.zip); M.3 exists (GRU article22 model); M.1 infrastructure exists but no benchmarked model artifact",
            "what_missing": "train/eval/benchmark M.1, then unify M.1/M.2/M.3 in one comparison table",
            "primary_output": "models/ppo_thermostatic.zip + models/ppo_thermostatic_article22_gru.zip",
        }
    )

    rows.append(
        {
            "block": "4_transfer_ablation",
            "requirement": "trained on surrogate -> fine-tuned on BOPTEST -> evaluated on BOPTEST",
            "status": "partial",
            "what_exists": "MORL surrogate-pretrain pipeline exists; thermostatic 15 min BOPTEST fine-tune path exists",
            "what_missing": "canonical before/after transfer comparison table for one controller family",
            "primary_output": "training/run_morl_surrogate_pipeline.py + training/finetune_tsup_policies_boptest.py",
        }
    )

    rows.append(
        {
            "block": "5_time_step_ablation",
            "requirement": "1h vs 15 min on same controller family",
            "status": "partial",
            "what_exists": "1h yearly thermostatic benchmark exists; 15 min peak/typical heat benchmark exists",
            "what_missing": "one canonical ablation table with same controller family and aligned metrics",
            "primary_output": "outputs/thermostatic_yearly_summary.csv + outputs/bestest_air_article7_style_15min/summary.csv",
        }
    )

    return rows


def next_command_rows() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "experiment": "Canonical 4h/8h/24h surrogate rollout table",
            "command": "python evaluation/validate_surrogate_v35_rollout_live.py --horizons 1 4 8 24 --out-dir outputs/surrogate_v35_rollout_live",
            "why": "Adds the missing 8h horizon to the canonical surrogate fidelity source output.",
        },
        {
            "priority": "2",
            "experiment": "Refresh canonical surrogate benchmark snapshot",
            "command": "python evaluation/build_research_benchmark.py",
            "why": "Rebuilds the paper-ready surrogate table and figures after the new rollout validation.",
        },
        {
            "priority": "3",
            "experiment": "Train honest M.1 thermostatic baseline",
            "command": "python training/train_thermostatic.py --article22-variant m1 --save-name ppo_thermostatic_article22_m1",
            "why": "Predictive-information ablation cannot start without M.1.",
        },
        {
            "priority": "4",
            "experiment": "Benchmark M.1 at 15 min",
            "command": "python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers thermostatic --thermostatic-model models/ppo_thermostatic_article22_m1.zip --output-dir outputs/article22_m1_benchmark_15min",
            "why": "Creates the first comparable M.1 result on the same heating windows.",
        },
        {
            "priority": "5",
            "experiment": "Fine-tune M.1 on BOPTEST 15 min",
            "command": "python training/finetune_tsup_policies_boptest.py --agents thermostatic --step-sec 900 --thermostatic-model models/ppo_thermostatic_article22_m1.zip --out-dir outputs/boptest_15min_policy_finetune_m1",
            "why": "Needed for transfer ablation on the same controller family.",
        },
        {
            "priority": "6",
            "experiment": "Benchmark fine-tuned M.1 at 15 min",
            "command": "python evaluation/benchmark_bestest_air_article7_style.py --step-sec 900 --controllers thermostatic --thermostatic-model outputs/boptest_15min_policy_finetune_m1/thermostatic_step900_finetuned.zip --output-dir outputs/article22_m1_benchmark_15min_finetuned",
            "why": "Closes the surrogate->BOPTEST transfer comparison for M.1.",
        },
        {
            "priority": "7",
            "experiment": "Refresh canonical benchmark outputs",
            "command": "python evaluation/build_research_benchmark.py",
            "why": "Rebuilds the unified paper-ready tables and figures after new controller artifacts are produced.",
        },
        {
            "priority": "8",
            "experiment": "Refresh minimum paper suite status",
            "command": "python evaluation/build_minimum_paper_suite.py",
            "why": "Updates the project status sheet after the canonical benchmark is rebuilt.",
        },
    ]


def write_next_steps_md(rows: list[dict[str, str]]) -> Path:
    lines = ["# Minimum Paper Suite Next Steps", ""]
    for row in rows:
        lines.append(f"## Step {row['priority']}: {row['experiment']}")
        lines.append("")
        lines.append(f"Why: {row['why']}")
        lines.append("")
        lines.append("```powershell")
        lines.append(row["command"])
        lines.append("```")
        lines.append("")

    path = DOC_DIR / "next_steps.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manifest(status_df: pd.DataFrame, commands_df: pd.DataFrame, next_steps_md: Path) -> Path:
    manifest = {
        "status_table": str((TABLE_DIR / "minimum_experiment_status.csv").relative_to(REPO_ROOT)),
        "commands_table": str((TABLE_DIR / "next_commands.csv").relative_to(REPO_ROOT)),
        "next_steps_markdown": str(next_steps_md.relative_to(REPO_ROOT)),
        "artifacts_checked": {
            "controller_15min": exists("outputs/bestest_air_article7_style_15min/summary.csv"),
            "controller_15min_gru": exists("outputs/bestest_air_article7_style_15min_article22_gru/summary.csv"),
            "thermostatic_m2_model": exists("models/ppo_thermostatic.zip"),
            "thermostatic_m3_model": exists("models/ppo_thermostatic_article22_gru.zip"),
            "thermostatic_m1_model": exists("models/ppo_thermostatic_article22_m1.zip"),
        },
    }
    path = DOC_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()

    status_df = pd.DataFrame(current_status_rows())
    commands_df = pd.DataFrame(next_command_rows())

    status_path = TABLE_DIR / "minimum_experiment_status.csv"
    commands_path = TABLE_DIR / "next_commands.csv"
    status_df.to_csv(status_path, index=False)
    commands_df.to_csv(commands_path, index=False)

    next_steps_md = write_next_steps_md(next_command_rows())
    manifest_path = write_manifest(status_df, commands_df, next_steps_md)

    print("MINIMUM PAPER SUITE STATUS COMPLETE")
    print(f"Status table: {status_path}")
    print(f"Command table:{commands_path}")
    print(f"Next steps:   {next_steps_md}")
    print(f"Manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
