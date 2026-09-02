"""Aggregate the multi-seed HDRL lambda sweep into a mean +/- std band.

Reads every cell produced by evaluation/run_hdrl_seed_sweep.py and writes:

    reports/block2_hdrl_lambda_sweep_seed_band.csv
    reports/block2_hdrl_lambda_sweep_seed_band.json

The JSON also carries the two claims a reviewer will actually test, evaluated
against the seed band rather than asserted:

  1. monotonicity -- does m_s rise with lambda on every window, using seed means?
  2. separation   -- is the lambda=0.00 vs lambda=0.10 gap larger than the seed
                     noise? Reported as a gap-to-pooled-sd ratio and as a
                     non-overlap check on mean +/- 1 sd. With three seeds this is
                     a descriptive separation statistic, not a significance test,
                     and it is labelled as such.

Cells that have not been run yet are skipped with a note, so this can be run
while the sweep is still in progress.

Usage
-----
    python evaluation/build_hdrl_seed_band.py
    python evaluation/build_hdrl_seed_band.py --artifact-root outputs/block2_hdrl_seed_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAMBDAS = {"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10}
WINDOWS = {"peak_heat_window": "peak", "typical_heat_window": "typical"}
METRICS = ("m_s", "violation_pct", "rmse_center_c", "energy_kwh")
CONTROLLER = "hdrl"

LEGACY_SUMMARY = ROOT / "reports" / "block2_hdrl_lambda_sweep_summary.csv"


def read_cell(summary_csv: Path, scenario: str) -> dict[str, float] | None:
    if not summary_csv.exists():
        return None
    with summary_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("controller") == CONTROLLER and row.get("scenario") == scenario:
                return {k: float(row[k]) for k in METRICS if row.get(k) not in (None, "")}
    return None


def agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def read_legacy() -> dict[tuple[str, str], float]:
    """Frozen single-seed m_s, for the 'what changed' column."""
    out: dict[tuple[str, str], float] = {}
    if not LEGACY_SUMMARY.exists():
        return out
    with LEGACY_SUMMARY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(row["variant"], row["scenario"])] = float(row["m_s"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the multi-seed HDRL lambda sweep.")
    parser.add_argument("--artifact-root", default="outputs/block2_hdrl_seed_sweep")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-csv", default="reports/block2_hdrl_lambda_sweep_seed_band.csv")
    parser.add_argument("--out-json", default="reports/block2_hdrl_lambda_sweep_seed_band.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    root = ROOT / args.artifact_root
    legacy = read_legacy()

    rows: list[dict] = []
    per_cell: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    missing: list[str] = []

    for variant in LAMBDAS:
        for scenario, short in WINDOWS.items():
            found: dict[int, dict[str, float]] = {}
            for seed in seeds:
                cell = read_cell(root / f"{variant}_seed{seed}" / "summary.csv", scenario)
                if cell is None:
                    missing.append(f"{variant}_seed{seed}:{scenario}")
                else:
                    found[seed] = cell
            if not found:
                continue
            per_cell[(variant, scenario)] = found

            row: dict = {
                "variant": variant,
                "lambda_temp_disagree": LAMBDAS[variant],
                "window": short,
                "scenario": scenario,
                "seeds": "/".join(str(s) for s in sorted(found)),
                "n_seeds": len(found),
            }
            for metric in METRICS:
                vals = [v[metric] for v in found.values() if metric in v]
                mean, sd = agg(vals)
                row[f"{metric}_mean"] = round(mean, 4)
                row[f"{metric}_std"] = round(sd, 4)
            frozen = legacy.get((variant, scenario))
            row["m_s_single_seed_frozen"] = round(frozen, 4) if frozen is not None else ""
            rows.append(row)

    if not rows:
        print(f"No completed cells under {root}.")
        print("Run: python evaluation/run_hdrl_seed_sweep.py --stage all --skip-existing")
        return

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- the two claims a reviewer will test -------------------------------
    verdicts: dict[str, dict] = {}
    for scenario, short in WINDOWS.items():
        ordered = [v for v in LAMBDAS if (v, scenario) in per_cell]
        means = {
            v: statistics.fmean([c["m_s"] for c in per_cell[(v, scenario)].values()])
            for v in ordered
        }
        monotone = all(
            means[a] <= means[b] for a, b in zip(ordered, ordered[1:])
        ) if len(ordered) > 1 else None

        entry: dict = {
            "lambdas_available": ordered,
            "m_s_mean_by_lambda": {v: round(means[v], 4) for v in ordered},
            "monotone_increasing_in_lambda": monotone,
        }

        if "l000" in per_cell.get("l000", {}) or ("l000", scenario) in per_cell:
            lo = [c["m_s"] for c in per_cell.get(("l000", scenario), {}).values()]
            hi = [c["m_s"] for c in per_cell.get(("l010", scenario), {}).values()]
            if lo and hi:
                lo_m, lo_sd = agg(lo)
                hi_m, hi_sd = agg(hi)
                pooled = ((lo_sd ** 2 + hi_sd ** 2) / 2) ** 0.5
                entry["separation_l000_vs_l010"] = {
                    "l000_mean": round(lo_m, 4), "l000_std": round(lo_sd, 4), "l000_n": len(lo),
                    "l010_mean": round(hi_m, 4), "l010_std": round(hi_sd, 4), "l010_n": len(hi),
                    "gap": round(hi_m - lo_m, 4),
                    "pooled_std": round(pooled, 4),
                    "gap_over_pooled_std": round((hi_m - lo_m) / pooled, 2) if pooled > 0 else None,
                    "bands_disjoint_at_1sd": bool((lo_m + lo_sd) < (hi_m - hi_sd)),
                    "note": "Descriptive separation over 3 seeds, not a significance test.",
                }
        verdicts[short] = entry

    payload = {
        "artifact_root": args.artifact_root,
        "seeds_requested": seeds,
        "missing_cells": missing,
        "rows": rows,
        "verdicts": verdicts,
    }
    out_json = ROOT / args.out_json
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- console report -----------------------------------------------------
    print(f"Wrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    if missing:
        print(f"\n{len(missing)} cell(s) not yet available, e.g. {missing[:4]}")

    print("\nm_s, mean +/- sd over seeds (frozen single-seed value in brackets)")
    header = f"{'window':<10}{'lambda':<9}{'n':>3}{'m_s mean':>12}{'sd':>9}{'frozen':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        frozen = row["m_s_single_seed_frozen"]
        frozen_s = f"{frozen:.4f}" if frozen != "" else "--"
        print(f"{row['window']:<10}{row['lambda_temp_disagree']:<9.2f}{row['n_seeds']:>3}"
              f"{row['m_s_mean']:>12.4f}{row['m_s_std']:>9.4f}{frozen_s:>10}")

    print("\nClaim checks")
    for short, entry in verdicts.items():
        mono = entry["monotone_increasing_in_lambda"]
        print(f"  [{short}] m_s increases monotonically with lambda: {mono}")
        sep = entry.get("separation_l000_vs_l010")
        if sep:
            print(f"          lambda 0.00 -> 0.10 gap {sep['gap']:+.4f}, "
                  f"pooled sd {sep['pooled_std']:.4f}, "
                  f"gap/sd {sep['gap_over_pooled_std']}, "
                  f"1-sd bands disjoint: {sep['bands_disjoint_at_1sd']}")

    print("\nNext: python evaluation/build_hdrl_seed_band_figure.py")


if __name__ == "__main__":
    main()
