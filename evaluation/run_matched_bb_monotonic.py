"""Can a sign-correct matched-resolution surrogate be trained, and does it still
fail as a training environment?

Background
----------
`reports/block1_matched_bb_seed_audit.json` established that training this
backbone on the 15-minute corpus reliably produces sign-INVERTED models: four
independent draws, 3.2-14.0% correct response sign, while 24 h rollout RMSE
*improved* over the canonical checkpoint (0.696-0.750 against 0.876). The
controller trained on such a model saturates at the minimum supply command while
the building sits below band.

That leaves the paper's central question half-answered for this data point. The
matched-resolution backend failed, but because its sign was inverted -- not
because of the per-step increment mechanism the paper argues for. So:

    Q1  Can the monotonicity penalty produce a matched-resolution surrogate that
        is sign-correct AND still more accurate than the hourly one?
    Q2  If yes, does a controller trained on it still fail?

Q2 is the one that matters. A sign-correct, accurate, 15-minute surrogate that
still trains a failing controller isolates the step-size mechanism cleanly at
matched resolution, and restores the data point to the central test on the
paper's own terms. A sign-correct surrogate that trains a WORKING controller
would instead show the matched-resolution failure was never about step size, and
the paper must drop that point and say so.

Both outcomes are reportable. The verdict rule is fixed here, before the run.

Stages
------
    surrogate  train with --lambda-mono on N seeds, rollout-validate, audit sign
    control    train thermostatic PPO on the best sign-correct surrogate,
               benchmark zero-shot on live BOPTEST (needs the emulator)
    all        both

Usage
-----
    python evaluation/run_matched_bb_monotonic.py --stage surrogate --seeds 42,43,44
    python evaluation/run_matched_bb_monotonic.py --stage control
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CORPUS = "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv"
HP = dict(epochs=500, batch_size=256, lr=1e-3, hidden_dim=64, patience=30,
          multi_horizons=(2, 4))

# Reference points, all frozen and already in the manuscript.
REF = {
    "hourly BB (usable controller)": dict(rmse=1.557, sign=100.0, m_s_typical=0.095),
    "matched BB canonical (fails)": dict(rmse=0.8761, sign=9.8, m_s_typical=1.211),
    "calibrated GB (fails)": dict(rmse=0.644, sign=100.0, m_s_typical=1.102),
}
SIGN_PASS_PCT = 95.0          # what "sign-correct" means; the twin scores 100
N_AUDIT_STATES = 400


def sdir(seed: int, lambda_mono_fan: float = 0.0) -> Path:
    """Artifact directory. The fan weight is part of the name: without it a sweep
    over that weight either skips every cell (--skip-existing sees the previous
    run) or overwrites the earlier checkpoints in place."""
    if lambda_mono_fan and lambda_mono_fan > 0.0:
        return ROOT / f"outputs/surrogate_v3_15min_mono_seed{seed}_fan{lambda_mono_fan:g}"
    return ROOT / f"outputs/surrogate_v3_15min_mono_seed{seed}"


def ckpt(seed: int, lambda_mono_fan: float = 0.0) -> Path:
    return sdir(seed, lambda_mono_fan) / "rc_node_v2_best.pt"


def train_cmd(seed: int, lambda_mono: float, mono_margin: float,
               mono_jitter: float, lambda_mono_fan: float) -> list[str]:
    return [
        PY, "-B", str(ROOT / "surrogate" / "train_surrogate_backbone.py"),
        "--data", CORPUS,
        "--output_dir", str(sdir(seed, lambda_mono_fan).relative_to(ROOT)).replace("\\", "/"),
        "--epochs", str(HP["epochs"]),
        "--batch_size", str(HP["batch_size"]),
        "--lr", str(HP["lr"]),
        "--hidden_dim", str(HP["hidden_dim"]),
        "--patience", str(HP["patience"]),
        "--multi_horizons", *[str(h) for h in HP["multi_horizons"]],
        "--seed", str(int(seed)),
        "--lambda-mono", str(lambda_mono),
        "--mono-margin", str(mono_margin),
        "--mono-jitter", str(mono_jitter),
        "--lambda-mono-fan", str(lambda_mono_fan),
    ]


def rollout_cmd(seed: int, lambda_mono_fan: float = 0.0) -> list[str]:
    return [
        PY, "-B", str(ROOT / "evaluation" / "validate_surrogate_v3_rollout_prepared.py"),
        "--model", str(ckpt(seed, lambda_mono_fan).relative_to(ROOT)).replace("\\", "/"),
        "--out-dir", str((sdir(seed, lambda_mono_fan) / "rollout_prepared").relative_to(ROOT)).replace("\\", "/"),
    ]


def ppo_cmd(seed: int, model_ckpt: Path, save_name: str) -> list[str]:
    """Same thermostatic recipe as every other backend in the central test."""
    return [
        PY, "-B", str(ROOT / "training" / "train_thermostatic.py"),
        "--surrogate-kind", "legacy_v3",
        "--surrogate-path", str(model_ckpt.relative_to(ROOT)).replace("\\", "/"),
        "--step-sec", "900",
        "--comfort-low", "21", "--comfort-high", "24",
        "--seed", str(int(seed)),
        "--save-name", save_name,
    ]


def bench_cmd(save_name: str, out_dir: str, boptest_url: str) -> list[str]:
    return [
        PY, "-B", str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
        "--boptest-url", boptest_url,
        "--step-sec", "900",
        "--duration-days", "14",
        "--controllers", "thermostatic",
        "--thermostatic-model", f"models/{save_name}.zip",
        "--output-dir", out_dir,
    ]


# --------------------------------------------------------------------------- #
def audit_sign(model_path: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

    ad = load_direct_tsup_adapter(kind="legacy_v3", legacy_model_path=str(model_path),
                                  device="cpu").eval()
    rng = np.random.default_rng(42)
    st = list(zip(rng.uniform(16, 30, N_AUDIT_STATES), rng.uniform(-20, 35, N_AUDIT_STATES),
                  rng.uniform(0, 24, N_AUDIT_STATES), rng.uniform(0, 365, N_AUDIT_STATES)))
    ok, d = 0, []
    for z, a, h, dd in st:
        kw = dict(t_zone=float(z), t_amb=float(a), hour=float(h), day=float(dd), a1=0.5)
        lo = ad.step_with_aux_numpy(**kw, a0=-0.9)["t_next"]
        hi = ad.step_with_aux_numpy(**kw, a0=+0.9)["t_next"]
        d.append(hi - lo)
        ok += hi >= lo - 1e-6
    return {"correct_sign_pct": round(100.0 * ok / N_AUDIT_STATES, 1),
            "median_delta_c": round(float(np.median(d)), 4)}


def rmse24(seed: int, lambda_mono_fan: float = 0.0) -> float | None:
    p = sdir(seed, lambda_mono_fan) / "rollout_prepared" / "v3" / "horizon_metrics.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(p.open(encoding="utf-8")):
        if abs(float(r["horizon_h"]) - 24.0) < 1e-6:
            return float(r["temp_rmse_c"])
    return None


def run(cmd: list[str], log: Path, dry_run: bool) -> int:
    print("=" * 88, flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}     # else the child block-buffers
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        return proc.wait()


def report_surrogates(seeds: list[int], lambda_mono_fan: float = 0.0) -> list[dict]:
    rows = []
    for seed in seeds:
        if not ckpt(seed, lambda_mono_fan).exists():
            continue
        rows.append({"seed": seed, "rollout_rmse_c": rmse24(seed, lambda_mono_fan),
                     **audit_sign(ckpt(seed, lambda_mono_fan))})
    if not rows:
        print("No monotonic checkpoints yet.")
        return rows

    print("\n" + "=" * 78)
    print("Matched-resolution BB trained WITH the monotonicity penalty")
    print("=" * 78)
    hdr = f"{'model':<34}{'24 h RMSE':>12}{'correct sign':>15}{'median dT':>12}"
    print(hdr); print("-" * len(hdr))
    for name, r in REF.items():
        print(f"{name:<34}{r['rmse']:>12.4f}{r['sign']:>14.1f}%{'':>12}")
    print("-" * len(hdr))
    for r in rows:
        rm = "n/a" if r["rollout_rmse_c"] is None else f"{r['rollout_rmse_c']:.4f}"
        print(f"{'mono seed ' + str(r['seed']):<34}{rm:>12}"
              f"{r['correct_sign_pct']:>14.1f}%{r['median_delta_c']:>+12.3f}")

    good = [r for r in rows
            if r["correct_sign_pct"] >= SIGN_PASS_PCT
            and r["rollout_rmse_c"] is not None
            and r["rollout_rmse_c"] < REF["hourly BB (usable controller)"]["rmse"]]
    print()
    if good:
        best = min(good, key=lambda r: r["rollout_rmse_c"])
        print(f"Q1 ANSWERED YES: seed {best['seed']} is sign-correct "
              f"({best['correct_sign_pct']:.1f}%) and more accurate than the hourly BB "
              f"({best['rollout_rmse_c']:.4f} vs 1.557 C).")
        print("    Proceed to --stage control to answer Q2.")
    else:
        print(f"Q1 ANSWERED NO: no seed reached {SIGN_PASS_PCT}% correct sign while beating "
              "the hourly BB on rollout error.")
        print("    The matched-resolution point cannot be rescued and must leave the "
              "central test, reported as a sign-inversion failure instead.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Monotonicity-penalised matched-BB experiment.")
    ap.add_argument("--stage", choices=["surrogate", "control", "all"], default="surrogate")
    ap.add_argument("--seeds", default="42,43,44")
    # Calibrated on 2026-08-19, seed 42, full-length runs. Sign validity against
    # penalty strength: 9.8% unpenalised -> 60.0% (L=1) -> 78.0% (L=50, no jitter)
    # -> 88.0% (L=50, jitter 2) -> 100% at the settings below, with 24 h rollout
    # error 0.808 C, i.e. better than the canonical matched checkpoint (0.876).
    # The jitter is what carries it past ~80%: without it the constraint only
    # holds on corpus states, and a policy explores well outside them.
    ap.add_argument("--lambda-mono", type=float, default=200.0)
    ap.add_argument("--mono-margin", type=float, default=0.1)
    ap.add_argument("--mono-jitter", type=float, default=3.0)
    # Fan channel. 200 reached 89% validity, below the 95% admission gate of
    # H5b; sweep upward until the gate is met or it stops improving.
    ap.add_argument("--lambda-mono-fan", type=float, default=200.0)
    ap.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    ap.add_argument("--control-seed", type=int, default=None,
                    help="Surrogate seed to train the controller on. Default: the "
                         "sign-correct one with the lowest rollout error.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    logs = ROOT / "logs" / "matched_bb_mono"

    if args.stage in ("surrogate", "all"):
        print(f"Monotonic matched-BB: seeds {seeds}, lambda_mono={args.lambda_mono}, "
              f"margin={args.mono_margin} C, jitter={args.mono_jitter} C")
        for seed in seeds:
            if args.skip_existing and ckpt(seed, args.lambda_mono_fan).exists():
                print(f"[skip] seed {seed}")
                continue
            t0 = time.time()
            if run(train_cmd(seed, args.lambda_mono, args.mono_margin, args.mono_jitter,
                          args.lambda_mono_fan),
                   logs / f"train_seed{seed}.log", args.dry_run) != 0:
                print(f"[FAILED] training seed {seed}")
                if not args.keep_going:
                    sys.exit(1)
                continue
            print(f"[ok] seed {seed} trained in {(time.time() - t0) / 60:.1f} min")
            if run(rollout_cmd(seed, args.lambda_mono_fan),
                   logs / f"rollout_seed{seed}.log", args.dry_run) != 0:
                print(f"[FAILED] rollout seed {seed}")
                if not args.keep_going:
                    sys.exit(1)

        if not args.dry_run:
            rows = report_surrogates(seeds, args.lambda_mono_fan)
            out = ROOT / "reports" / "block1_matched_bb_monotonic_audit.json"
            out.write_text(json.dumps({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "corpus": CORPUS, "recipe": HP,
                "lambda_mono": args.lambda_mono, "mono_margin": args.mono_margin,
                "mono_jitter": args.mono_jitter,
                "sign_pass_pct": SIGN_PASS_PCT, "reference": REF, "rows": rows,
            }, indent=2), encoding="utf-8")
            print(f"\nWrote {out.relative_to(ROOT)}")

    if args.stage in ("control", "all"):
        rows = [{"seed": s, "rollout_rmse_c": rmse24(s), **audit_sign(ckpt(s))}
                for s in seeds if ckpt(s).exists()] if not args.dry_run else []
        if args.control_seed is not None:
            pick = args.control_seed
        else:
            good = [r for r in rows if r["correct_sign_pct"] >= SIGN_PASS_PCT
                    and r["rollout_rmse_c"] is not None]
            if not good and not args.dry_run:
                sys.exit("No sign-correct surrogate to train a controller on. "
                         "Run --stage surrogate first, or pass --control-seed explicitly.")
            pick = min(good, key=lambda r: r["rollout_rmse_c"])["seed"] if good else seeds[0]

        save_name = f"ppo_thermostatic_mono15min_seed{pick}"
        out_dir = f"outputs/block2_thermostatic_mono15min_seed{pick}"
        print(f"\nControl stage on surrogate seed {pick}")

        if run(ppo_cmd(pick, ckpt(pick), save_name),
               logs / f"ppo_seed{pick}.log", args.dry_run) != 0:
            sys.exit("[FAILED] PPO training")
        if run(bench_cmd(save_name, out_dir, args.boptest_url),
               logs / f"bench_seed{pick}.log", args.dry_run) != 0:
            sys.exit("[FAILED] live benchmark")

        if not args.dry_run:
            summary = ROOT / out_dir / "summary.csv"
            print("\n" + "=" * 78)
            print("Q2: does a sign-correct, accurate 15-minute surrogate still fail?")
            print("=" * 78)
            if summary.exists():
                for r in csv.DictReader(summary.open(encoding="utf-8")):
                    if r.get("controller") != "thermostatic":
                        continue
                    ms = float(r["m_s"])
                    verdict = "STILL FAILS (m_s > 1)" if ms > 1.0 else \
                              ("USABLE (m_s < 0.1)" if ms < 0.1 else "marginal")
                    print(f"  {r['scenario']:<22} m_s {ms:6.3f}   "
                          f"violation {float(r['violation_pct']):5.1f}%   [{verdict}]")
                print("\n  STILL FAILS on both windows -> the step-size mechanism holds at "
                      "matched resolution; the data point returns to the central test.")
                print("  USABLE -> the matched-resolution failure was never about step size; "
                      "drop the point and say so.")
            else:
                print(f"  {summary.relative_to(ROOT)} not found")


if __name__ == "__main__":
    main()
