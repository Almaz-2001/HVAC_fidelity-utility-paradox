from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

SURROGATE_V3 = "outputs/surrogate_v2/rc_node_v3_tsupply.pt"
V35_SUMMARY = "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json"

THERMOSTATIC_HYBRID = {
    "hybrid_l005": ("0.05", "l005"),
    "hybrid_l010": ("0.10", "l010"),
    "hybrid_l015": ("0.15", "l015"),
}

HDRL_SWEEP = {
    "l000": "0.00",
    "l003": "0.03",
    "l005": "0.05",
    "l010": "0.10",
}

MORL_POINTS = {
    "comfort_100_energy_000": "1.00,0.00,0.00",
    "comfort_075_energy_025": "0.75,0.25,0.00",
    "comfort_050_energy_050": "0.50,0.50,0.00",
    "comfort_025_energy_075": "0.25,0.75,0.00",
    "comfort_000_energy_100": "0.00,1.00,0.00",
}

CANONICALS = {
    "neutral": ("comfort_050_energy_050", "0.50,0.50,0.00"),
    "practical": ("comfort_075_energy_025", "0.75,0.25,0.00"),
}


def cmd(*parts: str | Path) -> list[str]:
    return [str(part) for part in parts]


def run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        print("=" * 88, flush=True)
        print(" ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def thermostatic_train_command(variant: str) -> list[str]:
    base = cmd(PY, "-B", ROOT / "training" / "train_thermostatic.py", "--step-sec", "900", "--comfort-low", "21", "--comfort-high", "24")
    if variant == "pure":
        return base + cmd("--surrogate-kind", "legacy_v3", "--surrogate-path", SURROGATE_V3, "--save-name", "ppo_thermostatic")
    if variant == "v35_direct":
        return base + cmd(
            "--surrogate-kind",
            "v35_calibrated",
            "--surrogate-summary-json",
            V35_SUMMARY,
            "--obs-ablation",
            "no_delta_t",
            "--power-feature-mode",
            "clipped_log",
            "--t-zone-feature-mode",
            "comfort_centered",
            "--save-name",
            "ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone",
        )
    if variant in THERMOSTATIC_HYBRID:
        lambda_temp, tag = THERMOSTATIC_HYBRID[variant]
        return base + cmd(
            "--surrogate-kind",
            "hybrid_v3_v35",
            "--surrogate-path",
            SURROGATE_V3,
            "--surrogate-summary-json",
            V35_SUMMARY,
            "--obs-ablation",
            "no_delta_t",
            "--power-feature-mode",
            "clipped_log",
            "--t-zone-feature-mode",
            "raw",
            "--lambda-temp-disagree",
            lambda_temp,
            "--lambda-power-disagree",
            "5e-5",
            "--save-name",
            f"ppo_thermostatic_hybrid_v3_v35_{tag}",
        )
    raise ValueError(f"Unknown thermostatic variant: {variant}")


def thermostatic_benchmark_command(variant: str) -> list[str]:
    model = {
        "pure": "models/ppo_thermostatic.zip",
        "hybrid_l005": "models/ppo_thermostatic_hybrid_v3_v35_l005.zip",
        "hybrid_l010": "models/ppo_thermostatic_hybrid_v3_v35_l010.zip",
        "hybrid_l015": "models/ppo_thermostatic_hybrid_v3_v35_l015.zip",
    }[variant]
    out = "outputs/bestest_air_article7_style_15min" if variant == "pure" else f"outputs/block2_thermostatic_hybrid_v3_v35_{THERMOSTATIC_HYBRID[variant][1]}"
    return cmd(
        PY,
        "-B",
        ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py",
        "--step-sec",
        "900",
        "--controllers",
        "thermostatic",
        "--thermostatic-model",
        model,
        "--output-dir",
        out,
    )


def transfer_variant_config(variant: str) -> tuple[str, list[str], str]:
    if variant == "pure":
        return "models/ppo_thermostatic.zip", [], "outputs/block13_closed_loop_transfer_pure_v3"
    if variant == "hybrid_l010":
        return (
            "models/ppo_thermostatic_hybrid_v3_v35_l010.zip",
            ["--obs-ablation", "no_delta_t", "--power-feature-mode", "clipped_log", "--t-zone-feature-mode", "raw"],
            "outputs/block13_closed_loop_transfer_hybrid_l010",
        )
    if variant == "v35_direct":
        return (
            "models/ppo_thermostatic_v35_15min_no_delta_t_powerlog_tzone.zip",
            ["--obs-ablation", "no_delta_t", "--power-feature-mode", "clipped_log", "--t-zone-feature-mode", "comfort_centered"],
            "outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone",
        )
    raise ValueError(f"Unknown transfer variant: {variant}")


def thermostatic_transfer_command(variant: str) -> list[str]:
    model, feature_args, out = transfer_variant_config(variant)
    return cmd(
        PY,
        "-B",
        ROOT / "evaluation" / "validate_closed_loop_transfer_thermostatic_live.py",
        "--thermostatic-model",
        model,
        "--summary-json",
        V35_SUMMARY,
        "--step-sec",
        "900",
        "--duration-days",
        "14",
        "--output-dir",
        out,
    ) + feature_args


def thermostatic_diagnose_command(variant: str) -> list[str]:
    model, feature_args, transfer_out = transfer_variant_config(variant)
    out = transfer_out.replace("block13_closed_loop_transfer", "block13_obs_gap")
    return cmd(
        PY,
        "-B",
        ROOT / "evaluation" / "diagnose_thermostatic_obs_transfer_gap.py",
        "--thermostatic-model",
        model,
        "--summary-json",
        V35_SUMMARY,
        "--step-sec",
        "900",
        "--duration-days",
        "14",
        "--output-dir",
        out,
    ) + feature_args


def warmstart_command() -> list[str]:
    return cmd(
        PY,
        "-B",
        ROOT / "training" / "launch_thermostatic_warmstart_benchmark.py",
        "--artifact-root",
        "outputs/block2_thermostatic_warmstart_utility",
        "--step-sec",
        "900",
        "--episode-days",
        "14",
        "--steps-thermostatic",
        "120000",
    )


def hdrl_train_command(variant: str) -> list[str]:
    lambda_temp = HDRL_SWEEP[variant]
    return cmd(
        PY,
        "-B",
        ROOT / "training" / "train_hdrl.py",
        "--surrogate-kind",
        "hybrid_v3_v35",
        "--surrogate-path",
        SURROGATE_V3,
        "--surrogate-summary-json",
        V35_SUMMARY,
        "--step-sec",
        "900",
        "--episode-days",
        "14",
        "--temp-low",
        "21",
        "--temp-high",
        "24",
        "--obs-ablation",
        "no_delta_t",
        "--power-feature-mode",
        "clipped_log",
        "--t-zone-feature-mode",
        "raw",
        "--lambda-temp-disagree",
        lambda_temp,
        "--lambda-power-disagree",
        "5e-5",
        "--save-prefix",
        f"hdrl_hybrid_v3_v35_{variant}",
    )


def hdrl_benchmark_command(variant: str) -> list[str]:
    return cmd(
        PY,
        "-B",
        ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py",
        "--step-sec",
        "900",
        "--controllers",
        "hdrl",
        "--hdrl-winter-model",
        f"models/hdrl_hybrid_v3_v35_{variant}_winter_final.zip",
        "--hdrl-summer-model",
        f"models/hdrl_hybrid_v3_v35_{variant}_summer_final.zip",
        "--output-dir",
        f"outputs/block2_hdrl_hybrid_v3_v35_{variant}",
    )


def morl_pipeline_command(*, artifact_root: str, weights: str, seed: int, config_dir: str = "configs/morl_surrogate_ppo") -> list[str]:
    return cmd(
        PY,
        "-B",
        ROOT / "training" / "run_morl_surrogate_pipeline.py",
        "--config-dir",
        config_dir,
        "--mode",
        "full",
        "--seed",
        str(seed),
        "--artifact-root",
        artifact_root,
        "--morl-weights",
        weights,
        "--surrogate-kind",
        "hybrid_v3_v35",
        "--surrogate-path",
        SURROGATE_V3,
        "--surrogate-summary-json",
        V35_SUMMARY,
        "--lambda-temp-disagree",
        "0.00",
        "--lambda-power-disagree",
        "5e-5",
        "--power-feature-mode",
        "clipped_log",
        "--t-zone-feature-mode",
        "raw",
        "--eval-step-sec",
        "900",
        "--eval-scenario-days",
        "14",
    )


def pi_yearly_command() -> list[str]:
    return cmd(PY, "-B", ROOT / "evaluation" / "yearly_validation_pi.py", "--output_dir", "outputs/pi_baseline_15min_yearly", "--step-sec", "900", "--scenario-days", "14")


def build_reports_commands() -> list[list[str]]:
    scripts = [
        "build_hou_evins_q1_gap_tables.py",
        "build_hybrid_evidence_closure.py",
        "build_morl_pareto_table.py",
        "build_morl_canonical_variance_diagnostics.py",
        "build_morl_seasonal_variance_inversion.py",
        "build_article_real_figures.py",
    ]
    return [cmd(PY, "-B", ROOT / "evaluation" / script) for script in scripts]


def build_hybrid_evidence_commands() -> list[list[str]]:
    return [[PY, "-B", str(ROOT / "evaluation" / "build_hybrid_evidence_closure.py")]]


def build_morl_5d_comparison_commands() -> list[list[str]]:
    return [[PY, "-B", str(ROOT / "evaluation" / "build_morl_5d_reconstructed_comparison.py")]]


def expand_variants(value: str, variants: list[str]) -> list[str]:
    return variants if value == "all" else [value]


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Block 2 short-command runner for thermostatic, HDRL, MORL, PI, and reports.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without executing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("thermostatic-train")
    p.add_argument("--variant", choices=["pure", "v35_direct", "hybrid_l005", "hybrid_l010", "hybrid_l015", "hybrid_sweep"], required=True)

    p = sub.add_parser("thermostatic-benchmark")
    p.add_argument("--variant", choices=["pure", "hybrid_l005", "hybrid_l010", "hybrid_l015", "hybrid_sweep"], required=True)

    p = sub.add_parser("thermostatic-transfer")
    p.add_argument("--variant", choices=["pure", "v35_direct", "hybrid_l010", "all"], required=True)

    p = sub.add_parser("thermostatic-diagnose")
    p.add_argument("--variant", choices=["pure", "v35_direct", "hybrid_l010", "all"], required=True)

    sub.add_parser("warmstart")

    p = sub.add_parser("hdrl-train")
    p.add_argument("--variant", choices=["l000", "l003", "l005", "l010", "sweep"], required=True)

    p = sub.add_parser("hdrl-benchmark")
    p.add_argument("--variant", choices=["l000", "l003", "l005", "l010", "sweep"], required=True)

    p = sub.add_parser("morl-pareto")
    p.add_argument("--point", choices=[*MORL_POINTS.keys(), "all"], default="all")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("morl-17d")
    p.add_argument("--point", choices=[*MORL_POINTS.keys(), "all"], default="all")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("morl-5d")
    p.add_argument("--point", choices=[*MORL_POINTS.keys(), "all"], default="comfort_075_energy_025")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("morl-canonical")
    p.add_argument("--canonical", choices=["neutral", "practical", "all"], required=True)
    p.add_argument("--seeds", default="42,43,44,45,46")

    sub.add_parser("pi-yearly")
    sub.add_parser("build-hybrid-evidence")
    sub.add_parser("build-morl-5d-comparison")
    sub.add_parser("build-reports")

    args = parser.parse_args()
    commands: list[list[str]] = []

    if args.command == "thermostatic-train":
        variants = ["hybrid_l005", "hybrid_l010", "hybrid_l015"] if args.variant == "hybrid_sweep" else [args.variant]
        commands = [thermostatic_train_command(v) for v in variants]
    elif args.command == "thermostatic-benchmark":
        variants = ["hybrid_l005", "hybrid_l010", "hybrid_l015"] if args.variant == "hybrid_sweep" else [args.variant]
        commands = [thermostatic_benchmark_command(v) for v in variants]
    elif args.command == "thermostatic-transfer":
        commands = [thermostatic_transfer_command(v) for v in expand_variants(args.variant, ["v35_direct", "pure", "hybrid_l010"])]
    elif args.command == "thermostatic-diagnose":
        commands = [thermostatic_diagnose_command(v) for v in expand_variants(args.variant, ["v35_direct", "pure", "hybrid_l010"])]
    elif args.command == "warmstart":
        commands = [warmstart_command()]
    elif args.command == "hdrl-train":
        variants = list(HDRL_SWEEP) if args.variant == "sweep" else [args.variant]
        commands = [hdrl_train_command(v) for v in variants]
    elif args.command == "hdrl-benchmark":
        variants = list(HDRL_SWEEP) if args.variant == "sweep" else [args.variant]
        commands = [hdrl_benchmark_command(v) for v in variants]
    elif args.command == "morl-pareto":
        points = list(MORL_POINTS) if args.point == "all" else [args.point]
        commands = [morl_pipeline_command(artifact_root=f"outputs/morl_pareto_hybrid_power_only/{p}", weights=MORL_POINTS[p], seed=args.seed) for p in points]
    elif args.command == "morl-17d":
        points = list(MORL_POINTS) if args.point == "all" else [args.point]
        commands = [
            morl_pipeline_command(
                artifact_root=f"outputs/morl_pareto_hybrid_power_only/{p}",
                weights=MORL_POINTS[p],
                seed=args.seed,
                config_dir="configs/morl_surrogate_ppo",
            )
            for p in points
        ]
    elif args.command == "morl-5d":
        points = list(MORL_POINTS) if args.point == "all" else [args.point]
        commands = [
            morl_pipeline_command(
                artifact_root=f"outputs/morl_5d_legacy_rerun/{p}",
                weights=MORL_POINTS[p],
                seed=args.seed,
                config_dir="configs/morl_surrogate_ppo_5d",
            )
            for p in points
        ]
    elif args.command == "morl-canonical":
        canonicals = list(CANONICALS) if args.canonical == "all" else [args.canonical]
        for canonical in canonicals:
            tag, weights = CANONICALS[canonical]
            for seed in parse_seeds(args.seeds):
                commands.append(morl_pipeline_command(artifact_root=f"outputs/morl_pareto_hybrid_power_only_seedfix/{tag}", weights=weights, seed=seed))
    elif args.command == "pi-yearly":
        commands = [pi_yearly_command()]
    elif args.command == "build-hybrid-evidence":
        commands = build_hybrid_evidence_commands()
    elif args.command == "build-morl-5d-comparison":
        commands = build_morl_5d_comparison_commands()
    elif args.command == "build-reports":
        commands = build_reports_commands()
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    run_commands(commands, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
