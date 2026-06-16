"""Aggregate the multi-seed thermostatic runs (pure v3 and hybrid) into a
mean +/- std band for the Block 2 controller table.

Background
----------
The canonical pure-v3 and hybrid controllers were reported single-seed (seed 42),
which is the most common reviewer objection. After running

    run_block2.py thermostatic-train     --variant {pure,hybrid_l010} --seed {43,44}
    run_block2.py thermostatic-benchmark  --variant {pure,hybrid_l010} --seed {43,44}

this script reads every available seed's live-BOPTEST benchmark summary and reports
m_s / violation / energy as mean +/- std over the seeds, per targeted window. Seeds
that have not been run yet are skipped (with a note), so it can be run incrementally.

Outputs
-------
  reports/block2_thermostatic_seed_band.csv   (controller, window, n_seeds, m_s_mean, m_s_std, ...)
  reports/block2_thermostatic_seed_band.json  (same + a ready-to-quote sentence)
"""

from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"peak_heat_window": "peak", "typical_heat_window": "typical"}
CONTROLLER = "thermostatic"  # both pure-v3 and hybrid are thermostatic-PPO rows


def out_dir(variant: str, seed: int) -> Path:
    sfx = "" if seed == 42 else f"_seed{seed}"
    base = {
        "pure": "outputs/bestest_air_article7_style_15min",
        "hybrid_l010": "outputs/block2_thermostatic_hybrid_v3_v35_l010",
        "matched_v3": "outputs/bestest_air_pure_v3_15min",
    }[variant]
    return ROOT / (base + sfx) / "summary.csv"


def read_window(path: Path, scenario: str):
    if not path.exists():
        return None
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r.get("controller") == CONTROLLER and r.get("scenario") == scenario:
            return {k: float(r[k]) for k in ("m_s", "violation_pct", "energy_kwh") if r.get(k) not in (None, "")}
    return None


def agg(vals):
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds to aggregate (default 42,43,44).")
    ap.add_argument("--out-csv", default="reports/block2_thermostatic_seed_band.csv")
    ap.add_argument("--out-json", default="reports/block2_thermostatic_seed_band.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows, payload = [], {}
    for variant, label in [("pure", "pure v3"), ("matched_v3", "matched v3 (15-min)"), ("hybrid_l010", "hybrid (lambda_T=0.10)")]:
        payload[variant] = {}
        for scenario, wlabel in WINDOWS.items():
            found, vals = [], {"m_s": [], "violation_pct": [], "energy_kwh": []}
            for s in seeds:
                rec = read_window(out_dir(variant, s), scenario)
                if rec:
                    found.append(s)
                    for k in vals:
                        if k in rec:
                            vals[k].append(rec[k])
            if not found:
                print(f"[skip] {label} {wlabel}: no seeds run yet")
                continue
            ms, vi = agg(vals["m_s"]), agg(vals["violation_pct"])
            rows.append({
                "controller": label, "window": wlabel, "seeds": "/".join(map(str, found)), "n_seeds": ms["n"],
                "m_s_mean": ms["mean"], "m_s_std": ms["std"],
                "violation_pct_mean": vi["mean"], "violation_pct_std": vi["std"],
            })
            payload[variant][wlabel] = {"seeds": found, **{"m_s": ms, "violation_pct": vi}}
            print(f"{label:22s} {wlabel:8s} seeds={found}  m_s={ms['mean']:.3f} ± {ms['std']:.3f}  viol={vi['mean']:.1f} ± {vi['std']:.1f}%")

    if rows:
        oc = ROOT / args.out_csv
        oc.parent.mkdir(parents=True, exist_ok=True)
        with oc.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        sent = ("Over the reported seeds, the pure-v3 and hybrid controllers are stated as "
                "mean +/- standard deviation per window; see reports/block2_thermostatic_seed_band.csv.")
        (ROOT / args.out_json).write_text(json.dumps({"seeds": seeds, "bands": payload, "sentence": sent}, indent=2), encoding="utf-8")
        print(f"\nWrote {oc}\nWrote {ROOT / args.out_json}")
    else:
        print("\nNo seed results found yet — run the train+benchmark commands first.")


if __name__ == "__main__":
    main()
