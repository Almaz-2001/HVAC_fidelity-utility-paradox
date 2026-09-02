"""Retrain the matched-resolution black-box surrogate on N seeds and audit each.

The question
-----------
The canonical matched-resolution checkpoint
(`outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt`, sha1
def373d3, the one that produces the frozen 0.8761 C rollout RMSE) responds to
the supply-temperature command with an INVERTED sign: raising the command lowers
the predicted zone temperature in about 88% of sampled states, on both the
`forward` and the `step_with_aux_numpy` call paths. The policy trained on it
saturates at the minimum supply command (18.0 C, p10 = p90) while the building
sits below the comfort band for the whole evaluation.

Two readings fit, and they lead to opposite manuscript changes:

  A. defect  -- this particular checkpoint is broken. The matched-resolution
                data point must come out of the paper's central test.
  B. property -- training the same backbone on the 15-minute corpus reliably
                produces sign-inverted models. Then it is a finding: a surrogate
                can cut its multi-step rollout error by 44% while losing the
                control-relevant sign, and rollout RMSE does not notice.

Retraining on several seeds separates them. This driver runs the retraining with
the canonical hyperparameters, changing only the seed, then reports for each
checkpoint: 24 h rollout RMSE (is it still "more accurate"?) and directional
validity (does it keep the sign?).

Nothing existing is touched: checkpoints go to
`outputs/surrogate_v3_15min_matched_seed<N>/`, and the canonical
`outputs/surrogate_v3_15min_matched/` is read for reference only.

Usage
-----
    python evaluation/run_matched_bb_seed_audit.py --dry-run
    python evaluation/run_matched_bb_seed_audit.py --seeds 42,43,44 --skip-existing
    python evaluation/run_matched_bb_seed_audit.py --audit-only
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

# Canonical matched-corpus training recipe, from roadmap.md section 2.5 and
# evaluation/run_block1.py::v3_train_15min_command. Only --seed is added.
CORPUS = "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv"
HP = dict(epochs=500, batch_size=256, lr=1e-3, hidden_dim=64, patience=30,
          multi_horizons=(2, 4))

CANONICAL_DIR = "outputs/surrogate_v3_15min_matched"
CANONICAL_PT = f"{CANONICAL_DIR}/rc_node_v3_15min_matched.pt"
REFERENCE_RMSE_C = 0.8761          # frozen Block 1 value for the canonical checkpoint

N_AUDIT_STATES = 400


def out_dir(seed: int) -> Path:
    return ROOT / f"outputs/surrogate_v3_15min_matched_seed{seed}"


def checkpoint(seed: int) -> Path:
    # train_surrogate_backbone.py writes rc_node_v2_best.pt inside --output_dir
    return out_dir(seed) / "rc_node_v2_best.pt"


def train_command(seed: int) -> list[str]:
    return [
        PY, "-B", str(ROOT / "surrogate" / "train_surrogate_backbone.py"),
        "--data", CORPUS,
        "--output_dir", str(out_dir(seed).relative_to(ROOT)),
        "--epochs", str(HP["epochs"]),
        "--batch_size", str(HP["batch_size"]),
        "--lr", str(HP["lr"]),
        "--hidden_dim", str(HP["hidden_dim"]),
        "--patience", str(HP["patience"]),
        "--multi_horizons", *[str(h) for h in HP["multi_horizons"]],
        "--seed", str(int(seed)),
    ]


def rollout_command(seed: int) -> list[str]:
    return [
        PY, "-B", str(ROOT / "evaluation" / "validate_surrogate_v3_rollout_prepared.py"),
        "--model", str(checkpoint(seed).relative_to(ROOT)),
        "--out-dir", str((out_dir(seed) / "rollout_prepared").relative_to(ROOT)),
    ]


# --------------------------------------------------------------------------- #
def audit_sign(model_path: Path) -> dict:
    """Directional validity of one checkpoint, same probe used on the twin."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

    adapter = load_direct_tsup_adapter(kind="legacy_v3",
                                       legacy_model_path=str(model_path),
                                       device="cpu").eval()
    rng = np.random.default_rng(42)
    states = list(zip(rng.uniform(16, 30, N_AUDIT_STATES),
                      rng.uniform(-20, 35, N_AUDIT_STATES),
                      rng.uniform(0, 24, N_AUDIT_STATES),
                      rng.uniform(0, 365, N_AUDIT_STATES)))
    ok, deltas = 0, []
    for z, a, h, d in states:
        kw = dict(t_zone=float(z), t_amb=float(a), hour=float(h), day=float(d), a1=0.5)
        lo = adapter.step_with_aux_numpy(**kw, a0=-0.9)["t_next"]
        hi = adapter.step_with_aux_numpy(**kw, a0=+0.9)["t_next"]
        deltas.append(hi - lo)
        if hi >= lo - 1e-6:
            ok += 1
    return {"correct_sign_pct": round(100.0 * ok / N_AUDIT_STATES, 1),
            "median_delta_c": round(float(np.median(deltas)), 4)}


def read_rollout_rmse(seed: int) -> float | None:
    import csv
    p = out_dir(seed) / "rollout_prepared" / "v3" / "horizon_metrics.csv"
    if not p.exists():
        return None
    best = None
    with p.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if int(float(row.get("horizon_h", row.get("horizon", 0)))) == 24:
                    best = float(row.get("temp_rmse_c", row.get("rmse_c", "nan")))
            except (TypeError, ValueError):
                continue
    return best


def run(cmd: list[str], log: Path, dry_run: bool) -> int:
    print("=" * 88, flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    # Piping the child's stdout flips its Python from line- to block-buffering,
    # so a long training run shows nothing for minutes and reads as a hang.
    # PYTHONUNBUFFERED forces it back; bufsize=1 alone only affects our side.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain matched-resolution BB on N seeds and audit.")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--audit-only", action="store_true",
                    help="Skip training; audit whatever checkpoints already exist.")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Matched-resolution BB seed audit: seeds {seeds}")
    print(f"  corpus: {CORPUS}")
    print(f"  recipe: {HP}  (canonical, only --seed added)")

    if not args.audit_only:
        for seed in seeds:
            if args.skip_existing and checkpoint(seed).exists():
                print(f"[skip] seed {seed} (checkpoint present)")
                continue
            t0 = time.time()
            rc = run(train_command(seed), ROOT / "logs" / "matched_bb_seeds" / f"train_seed{seed}.log",
                     args.dry_run)
            if rc != 0:
                print(f"[FAILED] training seed {seed} (exit {rc})")
                if not args.keep_going:
                    sys.exit(1)
                continue
            print(f"[ok] seed {seed} trained in {(time.time() - t0) / 60:.1f} min")

            rc = run(rollout_command(seed), ROOT / "logs" / "matched_bb_seeds" / f"rollout_seed{seed}.log",
                     args.dry_run)
            if rc != 0:
                print(f"[FAILED] rollout validation seed {seed} (exit {rc})")
                if not args.keep_going:
                    sys.exit(1)

    if args.dry_run:
        return

    # ----------------------------------------------------------------- audit
    rows = []
    canonical = ROOT / CANONICAL_PT
    if canonical.exists():
        rows.append({"seed": "canonical", "rollout_rmse_c": REFERENCE_RMSE_C,
                     **audit_sign(canonical)})
    for seed in seeds:
        if not checkpoint(seed).exists():
            print(f"[warn] no checkpoint for seed {seed}; skipping audit")
            continue
        rows.append({"seed": seed, "rollout_rmse_c": read_rollout_rmse(seed),
                     **audit_sign(checkpoint(seed))})

    if not rows:
        print("Nothing to audit yet.")
        return

    print("\n" + "=" * 72)
    print("Matched-resolution BB: accuracy and directional validity per seed")
    print("=" * 72)
    hdr = f"{'seed':<12}{'24 h RMSE':>12}{'correct sign':>15}{'median dT':>12}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        rmse = "n/a" if r["rollout_rmse_c"] is None else f"{r['rollout_rmse_c']:.4f}"
        print(f"{str(r['seed']):<12}{rmse:>12}{r['correct_sign_pct']:>14.1f}%"
              f"{r['median_delta_c']:>+12.3f}")

    trained = [r for r in rows if r["seed"] != "canonical"]
    inverted = [r for r in trained if r["correct_sign_pct"] < 50.0]
    print()
    if not trained:
        verdict = "no retrained seeds yet"
    elif len(inverted) == len(trained):
        verdict = ("REPRODUCIBLE (reading B): every retrained seed is sign-inverted. "
                   "Training this backbone on the 15-minute corpus reliably produces "
                   "models that lower rollout RMSE while losing the control sign.")
    elif not inverted:
        verdict = ("NOT REPRODUCED (reading A): no retrained seed is inverted, so the "
                   "canonical checkpoint is a defective draw. The matched-resolution "
                   "point cannot stay in the paper's central test as it stands.")
    else:
        verdict = (f"MIXED: {len(inverted)} of {len(trained)} retrained seeds inverted. "
                   "Report as a frequency, not as a property, and say so.")
    print("Verdict:", verdict)

    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "corpus": CORPUS, "recipe": HP, "seeds": seeds,
               "reference_rmse_c": REFERENCE_RMSE_C,
               "rows": rows, "verdict": verdict}
    out = ROOT / "reports" / "block1_matched_bb_seed_audit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
