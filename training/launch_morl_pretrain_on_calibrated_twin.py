from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_summary_boptest_v35.json"
ARTIFACT_ROOT = ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated"


def build_command(extra_args: list[str]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "training" / "run_morl_surrogate_pipeline.py"),
        "--mode",
        "pretrain",
        "--surrogate-kind",
        "v35_calibrated",
        "--surrogate-summary-json",
        str(SUMMARY_JSON),
        "--artifact-root",
        str(ARTIFACT_ROOT),
    ] + extra_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch MORL surrogate pretraining on the calibrated v3.5 direct-TSup surrogate."
    )
    _, extra_args = parser.parse_known_args()

    cmd = build_command(extra_args)
    print("MORL pretrain on v35_calibrated")
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
