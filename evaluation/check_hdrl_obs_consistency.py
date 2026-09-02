"""A/B check: does the HDRL train/eval observation mismatch change the numbers?

Background
----------
evaluation/run_block2.py::hdrl_train_command trains the HDRL agents with

    --obs-ablation no_delta_t --power-feature-mode clipped_log --t-zone-feature-mode raw

but evaluation/run_block2.py::hdrl_benchmark_command does not forward any of those
flags to the benchmark, whose defaults are obs_ablation=none and
power_feature_mode=raw. The ablations zero out slots rather than removing them
(see envs/tsup_features.apply_tsup_obs_ablation), so the observation dimension
still matches and nothing raises -- the policy is simply fed a different encoding
at evaluation than the one it was trained on.

reports/reproduce_current_state_runbook.md records the same omission for the
thermostatic hybrid benchmarks, so this is worth resolving before the multi-seed
sweep is run, not after.

This script benchmarks ONE already-trained HDRL policy twice against live
BOPTEST -- once with the matched encoding, once with the legacy one -- and prints
the difference. It trains nothing. Two windows of 14 days at a 900 s step is
about 2700 environment steps per arm, so both arms together take a few minutes.

Nothing existing is overwritten: results go to a dedicated directory.

Usage
-----
    python evaluation/check_hdrl_obs_consistency.py \
        --winter-model models/hdrl_hybrid_l000_winter_final.zip \
        --summer-model models/hdrl_hybrid_l000_summer_final.zip
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

METRICS = ("m_s", "violation_pct", "rmse_center_c", "energy_kwh")


def benchmark(controller: str, models: dict[str, str], out_dir: Path, boptest_url: str,
              matched: bool, dry_run: bool, duration_days: int = 14) -> None:
    command = [
        PY, "-B", str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
        "--boptest-url", boptest_url,
        "--step-sec", "900",
        "--duration-days", str(int(duration_days)),
        "--controllers", controller,
        "--output-dir", str(out_dir.relative_to(ROOT)),
    ]
    if controller == "hdrl":
        command += ["--hdrl-winter-model", models["winter"],
                    "--hdrl-summer-model", models["summer"]]
    else:
        command += ["--thermostatic-model", models["thermostatic"]]
    if matched:
        command += [
            "--obs-ablation", "no_delta_t",
            "--power-feature-mode", "clipped_log",
            "--t-zone-feature-mode", "raw",
        ]
    print("=" * 88, flush=True)
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def read_summary(path: Path, controller: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("controller") != controller:
                continue
            rows[row["scenario"]] = {
                k: float(row[k]) for k in METRICS if row.get(k) not in (None, "")
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one HDRL policy with matched vs legacy evaluation observations."
    )
    parser.add_argument("--controller", choices=["hdrl", "thermostatic"], default="hdrl",
                        help="thermostatic checks the same omission on the hybrid PPO benchmark.")
    parser.add_argument("--winter-model", default="models/hdrl_hybrid_l000_winter_final.zip")
    parser.add_argument("--summer-model", default="models/hdrl_hybrid_l000_summer_final.zip")
    parser.add_argument("--thermostatic-model",
                        default="models/ppo_thermostatic_hybrid_v3_v35_l010.zip")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--boptest-url", default="http://web:8000")
    parser.add_argument("--duration-days", type=int, default=14,
                        help="Both arms use the same window, so a short one (e.g. 3) answers "
                             "'do the encodings differ' in a few minutes. Use 14 to quote numbers.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.controller == "hdrl":
        models = {"winter": args.winter_model, "summer": args.summer_model}
    else:
        models = {"thermostatic": args.thermostatic_model}
    for rel in models.values():
        if not (ROOT / rel).exists():
            parser.error(f"Model not found: {rel}\nAvailable checkpoints: "
                         f"{sorted(p.name for p in (ROOT / 'models').glob('*.zip'))}")

    suffix = "" if args.duration_days == 14 else f"_{args.duration_days}d"
    root = ROOT / (args.output_root
                   or f"outputs/block2_{args.controller}_obs_consistency{suffix}")
    matched_dir, legacy_dir = root / "matched_obs", root / "legacy_obs"

    benchmark(args.controller, models, matched_dir, args.boptest_url,
              matched=True, dry_run=args.dry_run, duration_days=args.duration_days)
    benchmark(args.controller, models, legacy_dir, args.boptest_url,
              matched=False, dry_run=args.dry_run, duration_days=args.duration_days)

    if args.dry_run:
        return

    matched = read_summary(matched_dir / "summary.csv", args.controller)
    legacy = read_summary(legacy_dir / "summary.csv", args.controller)

    print("\n" + "=" * 88)
    print(f"{args.controller} evaluation-observation A/B")
    print("  matched = obs_ablation no_delta_t, power clipped_log (as trained)")
    print("  legacy  = obs_ablation none, power raw (as the frozen sweep was benchmarked)")
    print("=" * 88)
    header = f"{'scenario':<22}{'metric':<16}{'matched':>12}{'legacy':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))

    material = False
    for scenario in sorted(set(matched) | set(legacy)):
        for metric in METRICS:
            a = matched.get(scenario, {}).get(metric)
            b = legacy.get(scenario, {}).get(metric)
            if a is None or b is None:
                continue
            delta = a - b
            print(f"{scenario:<22}{metric:<16}{a:>12.4f}{b:>12.4f}{delta:>+12.4f}")
            if metric == "m_s" and abs(delta) > 0.01:
                material = True
        print("-" * len(header))

    if material:
        print("\nThe two encodings give materially different m_s.")
        print("The frozen single-seed sweep was produced with the legacy encoding, so it")
        print("evaluated the policies on observations they were not trained on. Re-run the")
        print("sweep with matched observations before quoting the lambda trend.")
    else:
        print("\nNo material difference in m_s: the frozen sweep's conclusion is not an")
        print("artifact of the encoding mismatch. Record this check and proceed.")


if __name__ == "__main__":
    main()
