from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_1_2_surrogate_rmse.workflow_config import (  # noqa: E402
    DEFAULT_COLLECTED_OUTPUT_DIR,
    DEFAULT_COLLECTED_TRAIN_RUN,
    DEFAULT_COLLECTED_TRAIN_SUBSET_CSV,
    DEFAULT_HYBRID_TRAIN_RUN,
    DEFAULT_PREPARED_DATASET_CSV,
    DEFAULT_PREPARED_ROLLOUT_LONG_TRAIN_RUN,
    DEFAULT_PREPARED_ROLLOUT_TRAIN_RUN,
    DEFAULT_PREPARED_TRAIN_RUN,
    LEGACY_V35_STEP_SEC,
    SURROGATE_STEP_SEC,
)


def run_command(args: list[str]) -> None:
    print("=" * 88)
    print("RUNNING")
    print("=" * 88)
    print(" ".join(args))
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short launcher for the active Block 1.2 15-minute surrogate workflow."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "prepare",
            "collect",
            "build_hybrid",
            "train_prepared",
            "train_prepared_rollout",
            "train_prepared_rollout_long",
            "train_collected",
            "train_hybrid",
            "calibrate_15min",
        ],
    )
    parser.add_argument(
        "--calibration-data",
        default=str(DEFAULT_COLLECTED_OUTPUT_DIR / "episodes" / "winter__heat__seed42.csv"),
        help="Episode CSV used for the 15-minute v3.5 calibration stage.",
    )
    parser.add_argument(
        "--calibration-output",
        default="outputs/surrogate_v35_inverse_boptest_15min_winter_heat",
    )
    args = parser.parse_args()

    python = sys.executable

    if args.stage == "prepare":
        run_command([python, "block_1_2_surrogate_rmse/data/prepare_surrogate_15min_dataset.py"])
        return

    if args.stage == "collect":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/data/collect_surrogate_15min_boptest_data.py",
                "--profile",
                "heating_focus",
                "--write-train-subset",
            ]
        )
        return

    if args.stage == "train_prepared":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py",
                "--preset",
                "prepared_15min",
                "--data",
                str(DEFAULT_PREPARED_DATASET_CSV),
                "--run-name",
                DEFAULT_PREPARED_TRAIN_RUN,
                "--validate-safety",
            ]
        )
        return

    if args.stage == "train_prepared_rollout":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py",
                "--preset",
                "prepared_15min_rollout_select",
                "--data",
                str(DEFAULT_PREPARED_DATASET_CSV),
                "--run-name",
                DEFAULT_PREPARED_ROLLOUT_TRAIN_RUN,
                "--validate-safety",
            ]
        )
        return

    if args.stage == "train_prepared_rollout_long":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py",
                "--preset",
                "prepared_15min_rollout_long",
                "--data",
                str(DEFAULT_PREPARED_DATASET_CSV),
                "--run-name",
                DEFAULT_PREPARED_ROLLOUT_LONG_TRAIN_RUN,
                "--validate-safety",
            ]
        )
        return

    if args.stage == "build_hybrid":
        run_command([python, "block_1_2_surrogate_rmse/data/build_surrogate_15min_hybrid_dataset.py"])
        return

    if args.stage == "train_collected":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py",
                "--preset",
                "collected_15min_focus",
                "--data",
                str(DEFAULT_COLLECTED_TRAIN_SUBSET_CSV),
                "--run-name",
                DEFAULT_COLLECTED_TRAIN_RUN,
                "--validate-safety",
            ]
        )
        return

    if args.stage == "train_hybrid":
        run_command(
            [
                python,
                "block_1_2_surrogate_rmse/training/train_surrogate_15min_backbone.py",
                "--preset",
                "hybrid_15min_anchor",
                "--run-name",
                DEFAULT_HYBRID_TRAIN_RUN,
                "--validate-safety",
            ]
        )
        return

    if args.stage == "calibrate_15min":
        run_command(
            [
                python,
                "surrogate/calibrate_surrogate_v35.py",
                "--preset",
                "full",
                "--data",
                args.calibration_data,
                "--output_dir",
                args.calibration_output,
                "--step-sec",
                str(SURROGATE_STEP_SEC),
                "--legacy-step-sec",
                str(LEGACY_V35_STEP_SEC),
                "--temp-latency-steps",
                "8",
                "--max-latency-search",
                "24",
                "--smooth-window",
                "20",
                "--rollout-horizons",
                "16,32",
                "--stage-c-mode",
                "heads_only",
            ]
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
