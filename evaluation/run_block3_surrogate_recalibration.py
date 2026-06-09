from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "outputs/surrogate_v2/rc_node_v3_tsupply.pt"
DEFAULT_PARTIAL_INIT_SUMMARY = (
    "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/"
    "calibration_summary_boptest_v35.json"
)


def default_data(testcase: str) -> str:
    return f"data/block3_{testcase}/hydronic_adapter_stage_c_15min.csv"


def default_output_dir(testcase: str, regime: str) -> str:
    suffix = {
        "partial": "surrogate_v35_partial_stage_c_allrows_heads",
        "full": "surrogate_v35_full_stage_abc_allrows_heads",
    }[regime]
    return f"outputs/block3_{testcase}/{suffix}"


def build_command(args: argparse.Namespace) -> list[str]:
    data = args.data or default_data(args.testcase)
    output_dir = args.output_dir or default_output_dir(args.testcase, args.regime)

    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "surrogate" / "inverse_problem_boptest_v35.py"),
        "--data",
        data,
        "--model",
        args.model,
        "--output_dir",
        output_dir,
        "--no-artifact-injection",
        "--target-mode",
        "clean",
        "--step-sec",
        "900",
        "--legacy-step-sec",
        "3600",
        "--stage-c-mode",
        "heads_only",
        "--stage-c-selection-metric",
        "val_loss",
        "--excitation-quantile",
        "0.0",
        "--excitation-mix-ratio",
        "1.0",
        "--excitation-mode",
        "dt_only",
        "--calib-lr",
        "1e-3",
        "--lambda-temp-reg",
        "0.01",
        "--seed",
        str(args.seed),
    ]

    if args.regime == "partial":
        cmd.extend(
            [
                "--stage-b-epochs",
                "0",
                "--stage-b-patience",
                "0",
                "--stage-c-epochs",
                "120",
                "--stage-c-patience",
                "20",
                "--backbone-lr",
                "1e-4",
                "--czon-lr",
                "1e-2",
                "--init-summary-json",
                args.init_summary_json,
            ]
        )
    else:
        cmd.extend(
            [
                "--stage-b-epochs",
                "120",
                "--stage-b-patience",
                "20",
                "--stage-c-epochs",
                "180",
                "--stage-c-patience",
                "30",
                "--backbone-lr",
                "2e-5",
                "--czon-lr",
                "1e-3",
            ]
        )

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Block 3 v3.5 surrogate recalibration with testcase-aware paths."
    )
    parser.add_argument(
        "--testcase",
        "--testcase-id",
        dest="testcase",
        choices=("bestest_hydronic_heat_pump", "bestest_hydronic", "singlezone_commercial_hydronic"),
        required=True,
    )
    parser.add_argument("--regime", choices=("partial", "full"), required=True)
    parser.add_argument("--data", default=None, help="Defaults to data/block3_<testcase>/hydronic_adapter_stage_c_15min.csv.")
    parser.add_argument("--output-dir", default=None, help="Defaults to outputs/block3_<testcase>/surrogate_v35_<regime>...")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-summary-json", default=DEFAULT_PARTIAL_INIT_SUMMARY)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    args = parser.parse_args()

    data_path = ROOT / (args.data or default_data(args.testcase))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Telemetry CSV not found: {data_path}. Run collect_block3_hydronic_adapter_telemetry.py first."
        )
    if args.regime == "partial":
        init_path = ROOT / args.init_summary_json
        if not init_path.exists():
            raise FileNotFoundError(f"Partial-regime init summary not found: {init_path}")

    cmd = build_command(args)
    print("Resolved Block 3 recalibration command:")
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
