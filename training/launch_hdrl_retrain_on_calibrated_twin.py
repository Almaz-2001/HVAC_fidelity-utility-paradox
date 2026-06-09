from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.block2_v35_paths import V35_15MIN_SUMMARY_JSON

SUMMARY_JSON = V35_15MIN_SUMMARY_JSON


def build_command(extra_args: list[str]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "training" / "train_hdrl.py"),
        "--surrogate-kind",
        "v35_calibrated",
        "--surrogate-summary-json",
        str(SUMMARY_JSON),
    ] + extra_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch HDRL retraining on the calibrated 15-minute v3.5 direct-TSup surrogate."
    )
    _, extra_args = parser.parse_known_args()

    cmd = build_command(extra_args)
    print("HDRL retrain on 15-minute v35_calibrated")
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
