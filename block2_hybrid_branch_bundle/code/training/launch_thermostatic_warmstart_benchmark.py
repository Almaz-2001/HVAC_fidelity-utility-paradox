from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.block2_v35_paths import V35_15MIN_SUMMARY_JSON


def run_command(cmd: list[str]) -> None:
    print("=" * 80)
    print("COMMAND")
    print("=" * 80)
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_single_controller_summary(summary_csv: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df["mode"] = mode
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the minimal Block 2 thermostatic scratch-vs-warmstart benchmark."
    )
    parser.add_argument("--artifact-root", default="outputs/block2_thermostatic_warmstart_utility")
    parser.add_argument("--save-name", default="ppo_thermostatic_v35_block2_warmstart")
    parser.add_argument("--step-sec", type=int, default=900)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--steps-thermostatic", type=int, default=120000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--boptest-url", default="http://web:8000")
    parser.add_argument("--testcase-id", default="bestest_air")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-scratch", action="store_true")
    parser.add_argument("--skip-warmstart", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    artifact_root = ROOT / args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    pretrained_model = ROOT / "models" / f"{args.save_name}.zip"
    scratch_out = artifact_root / "scratch_finetune"
    warmstart_out = artifact_root / "warmstart_finetune"
    scratch_eval = artifact_root / "scratch_eval"
    warmstart_eval = artifact_root / "warmstart_eval"

    if not args.skip_pretrain:
        pretrain_cmd = [
            sys.executable,
            str(ROOT / "training" / "launch_thermostatic_retrain_on_calibrated_twin.py"),
            "--save-name",
            args.save_name,
        ]
        run_command(pretrain_cmd)

    if not args.skip_scratch:
        scratch_cmd = [
            sys.executable,
            str(ROOT / "training" / "finetune_tsup_policies_boptest.py"),
            "--agents",
            "thermostatic",
            "--scratch-thermostatic",
            "--boptest-url",
            args.boptest_url,
            "--testcase-id",
            args.testcase_id,
            "--step-sec",
            str(args.step_sec),
            "--episode-days",
            str(args.episode_days),
            "--steps-thermostatic",
            str(args.steps_thermostatic),
            "--learning-rate",
            str(args.learning_rate),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--out-dir",
            str(scratch_out),
        ]
        run_command(scratch_cmd)

    if not args.skip_warmstart:
        warmstart_cmd = [
            sys.executable,
            str(ROOT / "training" / "finetune_tsup_policies_boptest.py"),
            "--agents",
            "thermostatic",
            "--boptest-url",
            args.boptest_url,
            "--testcase-id",
            args.testcase_id,
            "--step-sec",
            str(args.step_sec),
            "--episode-days",
            str(args.episode_days),
            "--steps-thermostatic",
            str(args.steps_thermostatic),
            "--learning-rate",
            str(args.learning_rate),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--thermostatic-model",
            str(pretrained_model),
            "--out-dir",
            str(warmstart_out),
        ]
        run_command(warmstart_cmd)

    if not args.skip_benchmark:
        scratch_model = scratch_out / f"thermostatic_step{args.step_sec}_scratch.zip"
        warmstart_model = warmstart_out / f"thermostatic_step{args.step_sec}_finetuned.zip"

        if not args.skip_scratch:
            scratch_benchmark_cmd = [
                sys.executable,
                str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
                "--boptest-url",
                args.boptest_url,
                "--testcase-id",
                args.testcase_id,
                "--step-sec",
                str(args.step_sec),
                "--controllers",
                "thermostatic",
                "--thermostatic-model",
                str(scratch_model),
                "--output-dir",
                str(scratch_eval),
            ]
            run_command(scratch_benchmark_cmd)

        if not args.skip_warmstart:
            warmstart_benchmark_cmd = [
                sys.executable,
                str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
                "--boptest-url",
                args.boptest_url,
                "--testcase-id",
                args.testcase_id,
                "--step-sec",
                str(args.step_sec),
                "--controllers",
                "thermostatic",
                "--thermostatic-model",
                str(warmstart_model),
                "--output-dir",
                str(warmstart_eval),
            ]
            run_command(warmstart_benchmark_cmd)

    compare_frames: list[pd.DataFrame] = []
    scratch_summary_csv = scratch_eval / "summary.csv"
    warmstart_summary_csv = warmstart_eval / "summary.csv"
    if scratch_summary_csv.exists():
        compare_frames.append(load_single_controller_summary(scratch_summary_csv, "scratch"))
    if warmstart_summary_csv.exists():
        compare_frames.append(load_single_controller_summary(warmstart_summary_csv, "warmstart"))

    combined_summary_path = artifact_root / "comparison_summary.csv"
    if compare_frames:
        comparison_df = pd.concat(compare_frames, ignore_index=True)
        comparison_df.to_csv(combined_summary_path, index=False)
    else:
        comparison_df = pd.DataFrame()

    manifest = {
        "artifact_root": str(artifact_root),
        "surrogate_summary_json": str(V35_15MIN_SUMMARY_JSON),
        "pretrained_model": str(pretrained_model),
        "scratch_finetune_dir": str(scratch_out),
        "warmstart_finetune_dir": str(warmstart_out),
        "scratch_eval_dir": str(scratch_eval),
        "warmstart_eval_dir": str(warmstart_eval),
        "comparison_summary_csv": str(combined_summary_path),
        "step_sec": args.step_sec,
        "episode_days": args.episode_days,
        "steps_thermostatic": args.steps_thermostatic,
        "learning_rate": args.learning_rate,
        "ppo_epochs": args.ppo_epochs,
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 80)
    print("BLOCK 2 THERMOSTATIC WARM-START UTILITY")
    print("=" * 80)
    print(f"Artifact root: {artifact_root}")
    if not comparison_df.empty:
        print(f"Comparison summary: {combined_summary_path}")
        print(
            comparison_df[
                ["mode", "scenario", "m_s", "violation_pct", "rmse_center_c", "mean_power_w", "energy_kwh"]
            ].to_string(index=False)
        )
    else:
        print("Comparison summary not available yet.")


if __name__ == "__main__":
    main()
