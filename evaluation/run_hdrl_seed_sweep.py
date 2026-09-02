"""Multi-seed HDRL lambda_temp_disagree sweep.

Why this exists
---------------
The published HDRL sweep (reports/block2_hdrl_lambda_sweep_summary.csv) is
single-seed. Our own manuscript flags it as the weakest evidence in the paper,
and IEEE Access allows exactly one revision round, so the seed band has to be in
the submission rather than promised in a response letter.

This driver repeats the whole sweep across seeds and writes everything into a
NEW artifact namespace (default outputs/block2_hdrl_seed_sweep/). Nothing under
outputs/block2_hdrl_hybrid_v3_v35_* or models/hdrl_hybrid_* is read, moved or
overwritten, so the frozen single-seed results stay exactly as published and can
be compared against the new band.

Train/eval observation consistency
----------------------------------
The HDRL agents are trained with obs_ablation=no_delta_t and
power_feature_mode=clipped_log. evaluation/run_block2.py::hdrl_benchmark_command
does not forward those flags to the benchmark, so the frozen sweep was evaluated
with obs_ablation=none and power_feature_mode=raw -- a different observation
encoding from the one the policy was trained on. This driver forwards the
matching flags by default. Pass --legacy-eval-obs to reproduce the old
(unmatched) evaluation instead; see evaluation/check_hdrl_obs_consistency.py for
a cheap A/B on an already-trained policy.

Usage
-----
    # 0. validate the whole chain in a couple of minutes
    python evaluation/run_hdrl_seed_sweep.py --stage all --smoke

    # 1. train every (lambda, seed) cell, resumable
    python evaluation/run_hdrl_seed_sweep.py --stage train --skip-existing

    # 2. benchmark them on live BOPTEST
    python evaluation/run_hdrl_seed_sweep.py --stage benchmark --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Same surrogate wiring as evaluation/run_block2.py::hdrl_train_command.
SURROGATE_V3 = "outputs/surrogate_v2/rc_node_v3_tsupply.pt"
V35_SUMMARY = "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json"

HDRL_SWEEP = {
    "l000": "0.00",
    "l003": "0.03",
    "l005": "0.05",
    "l010": "0.10",
}

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ARTIFACT_ROOT = "outputs/block2_hdrl_seed_sweep"
MODEL_PREFIX = "hdrl_seedsweep"

# Frozen protocol of the published sweep. Changing any of these makes the new
# band non-comparable with the single-seed result it is meant to qualify.
STEP_SEC = "900"
EPISODE_DAYS = "14"
TEMP_LOW = "21"
TEMP_HIGH = "24"
OBS_ABLATION = "no_delta_t"
POWER_FEATURE_MODE = "clipped_log"
T_ZONE_FEATURE_MODE = "raw"
LAMBDA_POWER = "5e-5"
WINTER_STEPS = 5_000_000
SUMMER_STEPS = 7_000_000
NUM_ENVS = 16
DURATION_DAYS = "14"


def cell_tag(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def save_prefix(variant: str, seed: int, model_prefix: str = MODEL_PREFIX) -> str:
    return f"{model_prefix}_{cell_tag(variant, seed)}"


def model_paths(variant: str, seed: int, model_prefix: str = MODEL_PREFIX) -> tuple[Path, Path]:
    prefix = save_prefix(variant, seed, model_prefix)
    return (
        ROOT / "models" / f"{prefix}_winter_final.zip",
        ROOT / "models" / f"{prefix}_summer_final.zip",
    )


def out_dir(artifact_root: str, variant: str, seed: int) -> Path:
    return ROOT / artifact_root / cell_tag(variant, seed)


def train_command(variant: str, seed: int, *, winter_steps: int, summer_steps: int,
                  num_envs: int, model_prefix: str = MODEL_PREFIX) -> list[str]:
    return [
        PY, "-B", str(ROOT / "training" / "train_hdrl.py"),
        "--surrogate-kind", "hybrid_v3_v35",
        "--surrogate-path", SURROGATE_V3,
        "--surrogate-summary-json", V35_SUMMARY,
        "--step-sec", STEP_SEC,
        "--episode-days", EPISODE_DAYS,
        "--temp-low", TEMP_LOW,
        "--temp-high", TEMP_HIGH,
        "--obs-ablation", OBS_ABLATION,
        "--power-feature-mode", POWER_FEATURE_MODE,
        "--t-zone-feature-mode", T_ZONE_FEATURE_MODE,
        "--lambda-temp-disagree", HDRL_SWEEP[variant],
        "--lambda-power-disagree", LAMBDA_POWER,
        "--winter-steps", str(int(winter_steps)),
        "--summer-steps", str(int(summer_steps)),
        "--num-envs", str(int(num_envs)),
        "--seed", str(int(seed)),
        "--save-prefix", save_prefix(variant, seed, model_prefix),
    ]


def benchmark_command(variant: str, seed: int, *, artifact_root: str, boptest_url: str,
                      legacy_eval_obs: bool, model_prefix: str = MODEL_PREFIX) -> list[str]:
    winter, summer = model_paths(variant, seed, model_prefix)
    command = [
        PY, "-B", str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
        "--boptest-url", boptest_url,
        "--step-sec", STEP_SEC,
        "--duration-days", DURATION_DAYS,
        "--controllers", "hdrl",
        "--hdrl-winter-model", str(winter.relative_to(ROOT)),
        "--hdrl-summer-model", str(summer.relative_to(ROOT)),
        "--output-dir", str(out_dir(artifact_root, variant, seed).relative_to(ROOT)),
    ]
    if not legacy_eval_obs:
        # Match the encoding the policy was actually trained on.
        command += [
            "--obs-ablation", OBS_ABLATION,
            "--power-feature-mode", POWER_FEATURE_MODE,
            "--t-zone-feature-mode", T_ZONE_FEATURE_MODE,
        ]
    return command


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=15
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_logged(command: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, float]:
    """Run a command, streaming to stdout and to a log file. Returns (rc, seconds)."""
    print("=" * 88, flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return 0, 0.0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Piping stdout flips the child's Python to block buffering, which hides
    # progress until 8 KB has accumulated. PPO's tables fill that quickly enough
    # to have masked it here, but a quieter child looks like a hang.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
    return rc, time.time() - started


def load_status(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cells": {}}


def save_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the HDRL lambda sweep across random seeds, into a fresh artifact namespace."
    )
    parser.add_argument("--stage", choices=["train", "benchmark", "all"], default="all")
    parser.add_argument("--lambdas", default=",".join(HDRL_SWEEP),
                        help=f"Comma-separated subset of {sorted(HDRL_SWEEP)}")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--winter-steps", type=int, default=WINTER_STEPS)
    parser.add_argument("--summer-steps", type=int, default=SUMMER_STEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a cell whose artifacts already exist (resume after an interruption).")
    parser.add_argument("--legacy-eval-obs", action="store_true",
                        help="Evaluate with obs_ablation=none/power=raw, reproducing the frozen sweep's "
                             "train/eval observation mismatch instead of correcting it.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny training budget to validate the chain end to end (results are meaningless).")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue with the remaining cells if one fails.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = [v.strip() for v in args.lambdas.split(",") if v.strip()]
    unknown = [v for v in variants if v not in HDRL_SWEEP]
    if unknown:
        parser.error(f"Unknown lambda variant(s): {unknown}. Known: {sorted(HDRL_SWEEP)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    winter_steps, summer_steps, num_envs = args.winter_steps, args.summer_steps, args.num_envs
    model_prefix = MODEL_PREFIX
    if args.smoke:
        winter_steps, summer_steps, num_envs = 20_000, 20_000, 4
        # Smoke checkpoints must never occupy the real filenames, or a later
        # --skip-existing run would treat 20k-step garbage as a finished cell.
        model_prefix = f"{MODEL_PREFIX}_smoke"
        if args.artifact_root == DEFAULT_ARTIFACT_ROOT:
            args.artifact_root = DEFAULT_ARTIFACT_ROOT + "_smoke"

    artifact_root = ROOT / args.artifact_root
    status_path = artifact_root / "sweep_status.json"
    status = load_status(status_path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "artifact_root": args.artifact_root,
        "stage": args.stage,
        "lambdas": {v: HDRL_SWEEP[v] for v in variants},
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "model_prefix": model_prefix,
        "legacy_eval_obs": bool(args.legacy_eval_obs),
        "protocol": {
            "surrogate_kind": "hybrid_v3_v35",
            "surrogate_path": SURROGATE_V3,
            "surrogate_summary_json": V35_SUMMARY,
            "step_sec": int(STEP_SEC),
            "episode_days": float(EPISODE_DAYS),
            "temp_band_c": [float(TEMP_LOW), float(TEMP_HIGH)],
            "obs_ablation": OBS_ABLATION,
            "power_feature_mode": POWER_FEATURE_MODE,
            "t_zone_feature_mode": T_ZONE_FEATURE_MODE,
            "lambda_power_disagree": LAMBDA_POWER,
            "winter_steps": int(winter_steps),
            "summer_steps": int(summer_steps),
            "num_envs": int(num_envs),
            "eval_duration_days": int(DURATION_DAYS),
        },
    }
    if not args.dry_run:
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    cells = [(v, s) for v in variants for s in seeds]
    stages = ["train", "benchmark"] if args.stage == "all" else [args.stage]

    print(f"HDRL multi-seed sweep: {len(cells)} cells x {len(stages)} stage(s)")
    print(f"  lambdas: {variants}")
    print(f"  seeds:   {seeds}")
    print(f"  root:    {args.artifact_root}")
    print(f"  eval obs: {'LEGACY (mismatched)' if args.legacy_eval_obs else 'matched to training'}")
    if args.smoke:
        print("  SMOKE MODE: results are not usable, this only validates the pipeline")

    failures: list[str] = []
    t_all = time.time()

    for stage in stages:
        for variant, seed in cells:
            tag = cell_tag(variant, seed)
            key = f"{tag}:{stage}"
            winter, summer = model_paths(variant, seed, model_prefix)
            summary = out_dir(args.artifact_root, variant, seed) / "summary.csv"

            if args.skip_existing:
                done = (winter.exists() and summer.exists()) if stage == "train" else summary.exists()
                if done:
                    print(f"[skip] {key} (artifacts present)")
                    continue

            if stage == "benchmark" and not (winter.exists() and summer.exists()):
                msg = f"[missing] {tag}: no trained models, run --stage train first"
                print(msg)
                failures.append(key)
                if not args.keep_going:
                    break
                continue

            if stage == "train":
                command = train_command(variant, seed, winter_steps=winter_steps,
                                        summer_steps=summer_steps, num_envs=num_envs,
                                        model_prefix=model_prefix)
            else:
                command = benchmark_command(variant, seed, artifact_root=args.artifact_root,
                                            boptest_url=args.boptest_url,
                                            legacy_eval_obs=args.legacy_eval_obs,
                                            model_prefix=model_prefix)

            log_path = ROOT / "logs" / "hdrl_seed_sweep" / f"{tag}_{stage}.log"
            rc, secs = run_logged(command, log_path, dry_run=args.dry_run)

            status.setdefault("cells", {})[key] = {
                "returncode": rc,
                "seconds": round(secs, 1),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "log": str(log_path.relative_to(ROOT)),
            }
            if not args.dry_run:
                save_status(status_path, status)

            if rc != 0:
                print(f"[FAILED] {key} (exit {rc}) -- see {log_path}")
                failures.append(key)
                if not args.keep_going:
                    print("Stopping. Re-run with --skip-existing to resume, or --keep-going to continue.")
                    break
            else:
                print(f"[ok] {key} in {secs / 60:.1f} min")
        else:
            continue
        break

    elapsed = (time.time() - t_all) / 60
    print("=" * 88)
    print(f"Sweep stage(s) {stages} finished in {elapsed:.1f} min")
    if failures:
        print(f"FAILED cells: {failures}")
        sys.exit(1)
    print("Next: python evaluation/build_hdrl_seed_band.py")


if __name__ == "__main__":
    main()
