"""Run the receding-horizon MPC baseline on each planning surrogate.

Protocol and hypothesis H5 are fixed in configs/mpc_baseline_preregistration.yaml;
this driver only executes it. The planner settings below are read from that file
rather than restated, so the two cannot drift apart.

Three backends spanning the same fidelity axis as the RL test:

    mpc_bb_hourly     coarse black box, 1.557 C  -- the one RL succeeds on
    mpc_bb_matched    finer black box,  0.876 C  -- RL fails
    mpc_gb_calibrated calibrated twin,  0.644 C  -- RL fails worst

If the planner's ordering follows predictive accuracy while the learner's is
inverted, the fidelity-utility inversion is a property of policy-gradient
learning rather than of the surrogates.

Usage
-----
    python evaluation/run_mpc_baseline.py --smoke          # planner self-test, no BOPTEST
    python evaluation/run_mpc_baseline.py --dry-run
    python evaluation/run_mpc_baseline.py --skip-existing
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
PREREG = ROOT / "configs" / "mpc_baseline_preregistration.yaml"


def load_prereg() -> dict:
    """Minimal reader for the fields this driver needs.

    Deliberately not a full YAML parse: the file is the human-readable record and
    a missing key here should fail loudly rather than be silently defaulted.
    """
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml is required: pip install pyyaml")
    with PREREG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def benchmark_command(backend: dict, planner: dict, *, artifact_root: str,
                      boptest_url: str, duration_days: int) -> list[str]:
    cmd = [
        PY, "-B", str(ROOT / "evaluation" / "benchmark_bestest_air_article7_style.py"),
        "--boptest-url", boptest_url,
        "--step-sec", "900",
        "--duration-days", str(int(duration_days)),
        "--controllers", "mpc",
        "--mpc-surrogate-kind", str(backend["surrogate_kind"]),
        "--mpc-model-step-sec", str(backend["model_step_sec"]),
        "--mpc-horizon-hours", str(planner["horizon_hours"]),
        "--mpc-iters", str(planner["n_iters"]),
        "--mpc-lr", str(planner["lr"]),
        "--mpc-lambda-comfort", str(planner["lambda_comfort"]),
        "--mpc-lambda-energy", str(planner["lambda_energy"]),
        "--output-dir", f"{artifact_root}/{backend['id']}",
    ]
    if backend.get("path"):
        cmd += ["--mpc-surrogate-path", str(backend["path"])]
    if backend.get("summary_json"):
        cmd += ["--mpc-summary-json", str(backend["summary_json"])]
    return cmd


def smoke_test(prereg: dict) -> None:
    """Exercise the planner directly on each surrogate. No BOPTEST, ~1 min.

    Checks the two things that would silently ruin a 3-hour benchmark run: that
    the adapter loads and is differentiable, and that the planner responds in the
    physically correct direction (cold zone -> heating command).
    """
    sys.path.insert(0, str(ROOT))
    from surrogate.direct_tsup_adapter import load_direct_tsup_adapter
    from envs.tsup_features import WeatherLookup, action_to_t_supply
    from evaluation.mpc_baseline import RecedingHorizonMPC

    weather_csv = ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv"
    weather = WeatherLookup(str(weather_csv) if weather_csv.exists() else None)
    planner_cfg = prereg["protocol"]["planner"]

    ok = True
    for backend in prereg["protocol"]["backends"]:
        adapter = load_direct_tsup_adapter(
            kind=backend["surrogate_kind"],
            legacy_model_path=str(ROOT / backend["path"]) if backend.get("path") else None,
            summary_json=str(ROOT / backend["summary_json"]) if backend.get("summary_json") else None,
            device="cpu",
        )
        mpc = RecedingHorizonMPC(
            adapter, weather,
            model_step_sec=backend["model_step_sec"],
            horizon_hours=planner_cfg["horizon_hours"],
            n_iters=planner_cfg["n_iters"],
            lr=planner_cfg["lr"],
            lambda_comfort=planner_cfg["lambda_comfort"],
            lambda_energy=planner_cfg["lambda_energy"],
        )
        t0 = time.time()
        cold = mpc.compute({"t_zone": 18.0, "t_amb": -10.0, "hour": 6.0, "day": 10.0})
        mpc._prev = None                       # do not warm-start the second probe
        hot = mpc.compute({"t_zone": 27.0, "t_amb": 30.0, "hour": 14.0, "day": 200.0})
        dt = (time.time() - t0) / 2.0

        t_cold, t_hot = action_to_t_supply(float(cold[0])), action_to_t_supply(float(hot[0]))
        direction = "OK" if t_cold > t_hot else "WRONG"
        if direction != "OK":
            ok = False
        print(f"  {backend['id']:<18} H={mpc.horizon:>2} steps  "
              f"cold->T_sup {t_cold:5.1f} C   hot->T_sup {t_hot:5.1f} C   "
              f"[{direction}]  {dt * 1000:.0f} ms/decision")

    if not ok:
        sys.exit("!! planner responded in the wrong direction on at least one backend")
    print("\nSmoke test passed. Estimated live cost per window is printed above "
          "times 1344 control steps.")


def main() -> None:
    prereg = load_prereg()
    planner = prereg["protocol"]["planner"]
    backends = prereg["protocol"]["backends"]
    default_root = prereg["outputs"]["artifact_root"]

    parser = argparse.ArgumentParser(description="Run the pre-registered MPC baseline.")
    parser.add_argument("--artifact-root", default=default_root)
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--duration-days", type=int,
                        default=int(prereg["protocol"]["evaluation"]["duration_days"]))
    parser.add_argument("--backends", default=",".join(b["id"] for b in backends))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Planner self-test on each surrogate; does not touch BOPTEST.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("MPC planner smoke test")
        smoke_test(prereg)
        return

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [w for w in wanted if w not in {b["id"] for b in backends}]
    if unknown:
        parser.error(f"unknown backend(s): {unknown}")
    selected = [b for b in backends if b["id"] in wanted]

    print(f"MPC baseline: {len(selected)} backend(s), horizon {planner['horizon_hours']} h, "
          f"{planner['n_iters']} iters")
    print(f"  pre-registration: {PREREG.relative_to(ROOT)}")
    print(f"  artifact root:    {args.artifact_root}")

    failures: list[str] = []
    for backend in selected:
        summary = ROOT / args.artifact_root / backend["id"] / "summary.csv"
        if args.skip_existing and summary.exists():
            print(f"[skip] {backend['id']} (summary.csv present)")
            continue

        cmd = benchmark_command(backend, planner, artifact_root=args.artifact_root,
                                boptest_url=args.boptest_url,
                                duration_days=args.duration_days)
        print("=" * 88, flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue

        log_path = ROOT / "logs" / "mpc_baseline" / f"{backend['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            # Without PYTHONUNBUFFERED the child block-buffers once stdout is a
            # pipe, and a multi-hour run shows nothing until 8 KB accumulates.
            child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=child_env)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()          # else the log stays empty until the run ends,
                                     # which makes a long run impossible to observe
            rc = proc.wait()
        mins = (time.time() - started) / 60

        if rc != 0:
            print(f"[FAILED] {backend['id']} (exit {rc}) -- see {log_path}")
            failures.append(backend["id"])
            if not args.keep_going:
                break
        else:
            print(f"[ok] {backend['id']} in {mins:.1f} min")

    if not args.dry_run:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration": str(PREREG.relative_to(ROOT)),
            "hypothesis": prereg["hypothesis"]["id"],
            "planner": planner,
            "backends": [b["id"] for b in selected],
            "duration_days": args.duration_days,
        }
        root = ROOT / args.artifact_root
        root.mkdir(parents=True, exist_ok=True)
        (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.dry_run:
        report(prereg, selected, args.artifact_root)

    if failures:
        sys.exit(f"failed: {failures}")


def read_ms(artifact_root: str, backend_id: str) -> dict[str, float]:
    """m_s per window from one backend's live-benchmark summary."""
    import csv
    out: dict[str, float] = {}
    p = ROOT / artifact_root / backend_id / "summary.csv"
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("controller") == "mpc":
                out[r["scenario"]] = float(r["m_s"])
    return out


def report(prereg: dict, selected: list[dict], artifact_root: str) -> None:
    """Print the table and the pre-registered H5 verdict."""
    windows = ("peak_heat_window", "typical_heat_window")
    print("\n" + "=" * 84)
    print("MPC baseline against the RL result on the same surrogate")
    print("=" * 84)
    hdr = (f"{'backend':<20}{'RMSE':>7}{'sign':>7}{'RL m_s':>9}"
           f"{'MPC peak':>11}{'MPC typ.':>11}")
    print(hdr); print("-" * len(hdr))
    got: dict[str, dict[str, float]] = {}
    for b in selected:
        ms = read_ms(artifact_root, b["id"])
        got[b["id"]] = ms
        peak = f"{ms[windows[0]]:.3f}" if windows[0] in ms else "--"
        typ = f"{ms[windows[1]]:.3f}" if windows[1] in ms else "--"
        print(f"{b['id']:<20}{b['rollout_rmse_c']:>7.3f}{b['sign_valid_pct']:>6.1f}%"
              f"{b['rl_m_s_typical']:>9.3f}{peak:>11}{typ:>11}")

    print("\nReference: BOPTEST built-in PI m_s 0.910; role-separated hybrid RL "
          "0.087 peak / 0.041 typical.")

    primary = got.get("mpc_bb_mono", {})
    if len(primary) < 2:
        print("\nH5 verdict pending: mpc_bb_mono has not completed both windows.")
        return
    vals = [primary[w] for w in windows]
    n_ok = sum(v < 1.0 for v in vals)
    verdict = {2: "SUPPORTED", 1: "NOT SUPPORTED", 0: "FALSIFIED"}[n_ok]
    print(f"\nH5 ({prereg['hypothesis']['formal']}): {verdict}")
    print(f"    MPC on bb_mono: {vals[0]:.3f} peak, {vals[1]:.3f} typical, "
          f"against RL's 1.426 / 1.597 on the same surrogate.")

    neg = got.get("mpc_bb_matched", {})
    if len(neg) == 2 and all(neg[w] < 1.0 for w in windows):
        print("    !! NEGATIVE CONTROL PASSED WHEN IT SHOULD FAIL: the planner succeeded "
              "on the sign-inverted surrogate. Check the harness before believing H5.")


if __name__ == "__main__":
    main()
