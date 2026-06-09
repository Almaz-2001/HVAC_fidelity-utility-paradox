"""Block 1 short-command runner.

Wraps the v3/v3.5 surrogate, calibration, validation, and article-facing
artifact pipelines behind a single CLI so the roadmap stays terse.  The
wrapper does not change underlying scripts; it only fixes canonical paths,
presets, and validation defaults.

Sub-commands
------------
collect-data        Collect the direct-TSup BOPTEST dataset for v3 training.
v3-train            Train the v3 backbone with canonical hyperparameters
                    (also renames the saved checkpoint to the canonical name).
prepare-15min       Prepare the 15-minute calibration corpus consumed by v3.5.
v35-calibrate       Run v3.5 Stage A/B/C inverse calibration.  The default
                    `--preset canonical` runs the two-step pipeline
                    (episodeaware → power_head_only) in sequence, which is
                    what produces the canonical artifact used downstream.
validate-rollouts   Validate prepared offline rollouts.
                    `--variant v3|v35|all` selects which surrogate(s) to run.
build-reports       Build the eleven Hou-and-Evins tables and the article
                    figures in a single pass.
speed-benchmark     Run the BOPTEST RTE-vs-surrogate speed comparison.
all                 Run the complete Block 1 pipeline end-to-end.

Add `--dry-run` before the sub-command to print the resolved commands without
executing them.

Examples
--------
    python3 -B evaluation/run_block1.py --dry-run v3-train
    python3 -B evaluation/run_block1.py v35-calibrate --preset canonical
    python3 -B evaluation/run_block1.py validate-rollouts --variant all
    python3 -B evaluation/run_block1.py build-reports
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Canonical artifacts (paths used downstream by Block 2 / Block 3 / paper).
TSUPPLY_DATA = "data/surrogate_v2/boptest_v2_tsupply.csv"
V3_OUTPUT_DIR = "outputs/surrogate_v2"
V3_COMPAT_PT = f"{V3_OUTPUT_DIR}/rc_node_v2_best.pt"
V3_CANONICAL_PT = f"{V3_OUTPUT_DIR}/rc_node_v3_tsupply.pt"

V35_DATA = "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv"
V35_EPISODEAWARE_DIR = "outputs/surrogate_v35_inverse_boptest_15min_episodeaware"
V35_POWER_HEAD_DIR = "outputs/surrogate_v35_inverse_boptest_15min_power_head_only"
V35_CANONICAL_SUMMARY = f"{V35_POWER_HEAD_DIR}/calibration_summary_boptest_v35.json"

V3_ROLLOUT_OUT = "outputs/surrogate_v3_rollout_prepared_15min"
V35_ROLLOUT_OUT = "outputs/surrogate_v35_rollout_prepared_15min_power_head_only"

# Section 1.5 — corpus-matched v3 retraining (Tactic B).
# v3 is normally trained on the 51,200-row hourly corpus, while v3.5 uses the
# 10,744-row 15-min corpus.  A reviewer can object that the v3-vs-v3.5
# predictive-validity comparison is corpus-confounded.  The v3_15min branch
# retrains v3 on EXACTLY the v3.5 corpus so the predictive-validity comparison
# becomes apples-to-apples (same data, same architecture, only the
# Stage A/B/C inverse calibration differs).
V3_15MIN_OUTPUT_DIR = "outputs/surrogate_v3_15min_matched"
V3_15MIN_COMPAT_PT = f"{V3_15MIN_OUTPUT_DIR}/rc_node_v2_best.pt"
V3_15MIN_CANONICAL_PT = f"{V3_15MIN_OUTPUT_DIR}/rc_node_v3_15min_matched.pt"
V3_15MIN_ROLLOUT_OUT = "outputs/surrogate_v3_15min_matched_rollout_prepared"
CORPUS_MATCHED_REPORT = "reports/block1_corpus_matched_comparison.csv"

LEGACY_PREPARE = (
    "draft/legacy_archive/top_level/block_1_2_surrogate_rmse/data/"
    "prepare_surrogate_15min_dataset.py"
)


# ---------------------------------------------------------------------------


def cmd(*parts) -> list[str]:
    return [str(part) for part in parts]


def run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    """Print each command then run it sequentially (Block 2-style)."""
    for command in commands:
        print("=" * 88, flush=True)
        print(" ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


# ---------------------------------------------------------------------------
# Section 1 — v3 surrogate


def collect_data_command() -> list[str]:
    return cmd(PY, "-B", "data/collect_tsupply_data.py")


def v3_train_command(*, epochs: int, batch_size: int, lr: float,
                     hidden_dim: int, patience: int) -> list[str]:
    return cmd(
        PY, "-B", "surrogate/train_surrogate_backbone.py",
        "--data", TSUPPLY_DATA,
        "--output_dir", V3_OUTPUT_DIR,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--lr", lr,
        "--hidden_dim", hidden_dim,
        "--patience", patience,
        "--multi_horizons", 2, 4,
    )


def v3_rename_checkpoint(*, dry_run: bool) -> None:
    """Copy rc_node_v2_best.pt → rc_node_v3_tsupply.pt (canonical name)."""
    src = ROOT / V3_COMPAT_PT
    dst = ROOT / V3_CANONICAL_PT
    print("=" * 88, flush=True)
    print(f"cp {V3_COMPAT_PT} {V3_CANONICAL_PT}", flush=True)
    if dry_run:
        return
    if not src.exists():
        raise FileNotFoundError(
            f"Expected v3 checkpoint at {src}.  Did v3 training finish?")
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Section 1.5 — corpus-matched v3 retraining (Tactic B reviewer mitigation)


def v3_train_15min_command(*, epochs: int, batch_size: int, lr: float,
                           hidden_dim: int, patience: int) -> list[str]:
    """Train the v3 backbone on the 15-min v3.5 corpus.

    This is the reviewer-mitigation experiment: it isolates the effect of
    Stage A/B/C inverse calibration from the effect of corpus differences.
    The v3 model has no hard-coded timestep in its forward pass
    (t_next = t_zone + dT(x)), so training on 15-min data is methodologically
    sound — it simply learns a "15-min delta" rather than a "1-hour delta".
    """
    return cmd(
        PY, "-B", "surrogate/train_surrogate_backbone.py",
        "--data", V35_DATA,
        "--output_dir", V3_15MIN_OUTPUT_DIR,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--lr", lr,
        "--hidden_dim", hidden_dim,
        "--patience", patience,
        "--multi_horizons", 2, 4,
    )


def v3_15min_rename_checkpoint(*, dry_run: bool) -> None:
    """Copy outputs/surrogate_v3_15min_matched/rc_node_v2_best.pt to a
    canonical filename that downstream rollout validation can resolve."""
    src = ROOT / V3_15MIN_COMPAT_PT
    dst = ROOT / V3_15MIN_CANONICAL_PT
    print("=" * 88, flush=True)
    print(f"cp {V3_15MIN_COMPAT_PT} {V3_15MIN_CANONICAL_PT}", flush=True)
    if dry_run:
        return
    if not src.exists():
        raise FileNotFoundError(
            f"Expected v3_15min checkpoint at {src}.  Did v3-train-15min finish?")
    shutil.copy2(src, dst)


def validate_v3_15min_rollout_command() -> list[str]:
    """Validate the corpus-matched v3 checkpoint on the same prepared 15-min
    rollouts used by v3.5 — this is the apples-to-apples reference run."""
    return cmd(
        PY, "-B", "evaluation/validate_surrogate_v3_rollout_prepared.py",
        "--model", V3_15MIN_CANONICAL_PT,
        "--out-dir", V3_15MIN_ROLLOUT_OUT,
    )


def build_corpus_matched_report_command() -> list[str]:
    """Aggregate the matched-corpus 24h rollout RMSEs into one CSV/table."""
    return cmd(PY, "-B", "evaluation/build_block1_corpus_matched_report.py")


# ---------------------------------------------------------------------------
# Section 2 — v3.5 inverse calibration


def prepare_15min_command() -> list[str]:
    return cmd(PY, "-B", LEGACY_PREPARE)


def v35_calibrate_command(preset: str) -> list[str]:
    # surrogate/calibrate_surrogate_v35.py thin-wraps inverse_problem_boptest_v35
    return cmd(
        PY, "-B", "surrogate/calibrate_surrogate_v35.py",
        "--preset", preset,
    )


def v35_calibrate_canonical_commands() -> list[list[str]]:
    """The canonical v3.5 artifact is built by TWO sequential presets.

    Step 1 (`block1_15min_episodeaware`) runs Stage A + B + C with the
    rollout-heads selection metric.  This is where C_zon is identified
    (120 Stage-B epochs) and the temperature head is calibrated.

    Step 2 (`block1_15min_power_head_only`) reads `init_summary_json` from
    Step 1, freezes C_zon, and re-runs Stage C with power_head_only mode
    (80 epochs) for a tighter power calibration.

    Skipping Step 1 is a silent correctness bug — Step 2's preset hard-codes
    the init JSON path from Step 1's output_dir, so calling Step 2 on a fresh
    repo will fail with a missing-file error.
    """
    return [
        v35_calibrate_command("block1_15min_episodeaware"),
        v35_calibrate_command("block1_15min_power_head_only"),
    ]


# ---------------------------------------------------------------------------
# Section 2.5 — prepared rollout validation


def validate_v3_rollout_command() -> list[str]:
    return cmd(
        PY, "-B", "evaluation/validate_surrogate_v3_rollout_prepared.py",
        "--model", V3_CANONICAL_PT,
        "--out-dir", V3_ROLLOUT_OUT,
    )


def validate_v35_rollout_command() -> list[str]:
    return cmd(
        PY, "-B", "evaluation/validate_surrogate_v35_rollout_prepared.py",
        "--summary-json", V35_CANONICAL_SUMMARY,
        "--out-dir", V35_ROLLOUT_OUT,
    )


# ---------------------------------------------------------------------------
# Section 3 — article-facing artifacts


def hou_evins_command() -> list[str]:
    return cmd(PY, "-B", "evaluation/build_hou_evins_q1_gap_tables.py")


def article_figures_command() -> list[str]:
    return cmd(PY, "-B", "evaluation/build_article_real_figures.py")


def speed_benchmark_command(*, boptest_url: str, episodes: int,
                            steps_per_episode: int, step_sec: int) -> list[str]:
    return cmd(
        PY, "-B", "evaluation/build_speed_benchmark.py",
        "--boptest-url", boptest_url,
        "--episodes", episodes,
        "--steps-per-episode", steps_per_episode,
        "--step-sec", step_sec,
    )


def build_reports_commands() -> list[list[str]]:
    """The two pure-build commands that produce the Hou-Evins tables and the
    real-data article figures.  Speed benchmark is intentionally separate
    because it touches BOPTEST RTE and has its own knobs."""
    return [hou_evins_command(), article_figures_command()]


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 1 short-command runner for v3/v3.5 surrogates, "
                    "calibration, rollout validation, and article artifacts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved commands without executing.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect-data")

    p = sub.add_parser("v3-train")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--skip-rename", action="store_true",
                   help="Skip cp rc_node_v2_best.pt → rc_node_v3_tsupply.pt.")

    # Section 1.5 — corpus-matched v3 retraining (Tactic B reviewer mitigation)
    p = sub.add_parser(
        "v3-train-15min",
        help="Retrain v3 on the 10,744-row 15-min v3.5 corpus for the "
             "apples-to-apples comparison reported in Section 1.5.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--skip-rename", action="store_true",
                   help="Skip the canonical-name copy step.")

    sub.add_parser("prepare-15min")

    p = sub.add_parser("v35-calibrate")
    p.add_argument(
        "--preset",
        choices=["canonical", "episodeaware", "power_head_only"],
        default="canonical",
        help="`canonical` (default) runs episodeaware then power_head_only.")

    p = sub.add_parser("validate-rollouts")
    p.add_argument(
        "--variant",
        choices=["v3", "v35", "v3_15min", "matched", "all"],
        default="all",
        help="`v3_15min` validates only the corpus-matched checkpoint; "
             "`matched` validates the trio (v3, v3_15min, v3.5) used in §1.5.")

    sub.add_parser(
        "build-corpus-matched-report",
        help="Aggregate the v3, v3_15min, raw v3.5, calibrated v3.5 24h RMSEs "
             "into the matched-comparison CSV table.")

    sub.add_parser("build-reports")

    p = sub.add_parser("speed-benchmark")
    p.add_argument("--boptest-url", default="http://web:8000")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps-per-episode", type=int, default=96)
    p.add_argument("--step-sec", type=int, default=900)

    sub.add_parser("all", help="Run the complete Block 1 pipeline.")

    args = parser.parse_args()
    dry_run = bool(args.dry_run)
    commands: list[list[str]] = []

    if args.command == "collect-data":
        commands = [collect_data_command()]

    elif args.command == "v3-train":
        commands = [v3_train_command(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            hidden_dim=args.hidden_dim, patience=args.patience)]
        run_commands(commands, dry_run=dry_run)
        if not args.skip_rename:
            v3_rename_checkpoint(dry_run=dry_run)
        return

    elif args.command == "v3-train-15min":
        commands = [v3_train_15min_command(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            hidden_dim=args.hidden_dim, patience=args.patience)]
        run_commands(commands, dry_run=dry_run)
        if not args.skip_rename:
            v3_15min_rename_checkpoint(dry_run=dry_run)
        return

    elif args.command == "prepare-15min":
        commands = [prepare_15min_command()]

    elif args.command == "v35-calibrate":
        if args.preset == "canonical":
            commands = v35_calibrate_canonical_commands()
        elif args.preset == "episodeaware":
            commands = [v35_calibrate_command("block1_15min_episodeaware")]
        else:  # power_head_only
            commands = [v35_calibrate_command("block1_15min_power_head_only")]

    elif args.command == "validate-rollouts":
        if args.variant == "matched":
            commands.append(validate_v3_rollout_command())
            commands.append(validate_v3_15min_rollout_command())
            commands.append(validate_v35_rollout_command())
        else:
            if args.variant in ("v3", "all"):
                commands.append(validate_v3_rollout_command())
            if args.variant in ("v3_15min", "all"):
                commands.append(validate_v3_15min_rollout_command())
            if args.variant in ("v35", "all"):
                commands.append(validate_v35_rollout_command())

    elif args.command == "build-corpus-matched-report":
        commands = [build_corpus_matched_report_command()]

    elif args.command == "build-reports":
        commands = build_reports_commands()

    elif args.command == "speed-benchmark":
        commands = [speed_benchmark_command(
            boptest_url=args.boptest_url,
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            step_sec=args.step_sec,
        )]

    elif args.command == "all":
        commands = [
            collect_data_command(),
            v3_train_command(epochs=500, batch_size=256, lr=1e-3,
                             hidden_dim=64, patience=30),
        ]
        run_commands(commands, dry_run=dry_run)
        v3_rename_checkpoint(dry_run=dry_run)
        commands = [
            prepare_15min_command(),
            *v35_calibrate_canonical_commands(),
            validate_v3_rollout_command(),
            validate_v35_rollout_command(),
            *build_reports_commands(),
            speed_benchmark_command(
                boptest_url="http://web:8000", episodes=100,
                steps_per_episode=96, step_sec=900),
        ]

    else:
        raise ValueError(f"Unsupported command: {args.command}")

    run_commands(commands, dry_run=dry_run)


if __name__ == "__main__":
    main()
