from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[RUN] {printable}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=ROOT)


def _guarded_ms_paths(output_root: str) -> tuple[str, str]:
    finetune_dir = f"{output_root}/boptest_15min_policy_finetune_ms_guarded"
    benchmark_dir = f"{output_root}/bestest_air_article7_style_15min_ms_guarded"
    return finetune_dir, benchmark_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact Python entrypoint for the 15-minute bestest_air transition path."
    )
    parser.add_argument("--preset", default="guarded_ms", choices=["guarded_ms"])
    parser.add_argument("--mode", default="both", choices=["finetune", "benchmark", "both"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--thermostatic-model", default="models/ppo_thermostatic.zip")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finetune_dir, benchmark_dir = _guarded_ms_paths(args.output_root)
    thermostatic_model = f"{finetune_dir}/thermostatic_step900_finetuned.zip"

    if args.mode in ("finetune", "both"):
        _run(
            [
                args.python,
                "training/finetune_tsup_policies_boptest.py",
                "--agents",
                "thermostatic",
                "--step-sec",
                "900",
                "--episode-days",
                "14",
                "--jitter-days",
                "0.5",
                "--learning-rate",
                "3e-5",
                "--ppo-epochs",
                "3",
                "--steps-thermostatic",
                "30000",
                "--thermostatic-model",
                args.thermostatic_model,
                "--out-dir",
                finetune_dir,
            ],
            args.dry_run,
        )

    if args.mode in ("benchmark", "both"):
        _run(
            [
                args.python,
                "evaluation/benchmark_bestest_air_article7_style.py",
                "--step-sec",
                "900",
                "--controllers",
                "thermostatic",
                "--thermostatic-model",
                thermostatic_model,
                "--output-dir",
                benchmark_dir,
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
