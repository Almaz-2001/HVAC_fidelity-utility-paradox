"""Build the closed-loop comparison between the canonical hourly-trained v3 and
the corpus-matched 15-minute v3, both used as PPO *training* environments.

Motivation (reviewer mitigation, §8.5 of the manuscript)
--------------------------------------------------------
The canonical pure-v3 controller is trained on a v3 surrogate that was *fit* at a
1-hour step but *used* at the 15-minute control step.  A reviewer can ask whether
v3's training utility comes from its black-box smoothness or partly from this
timestep mismatch.  The matched-resolution v3 (``rc_node_v3_15min_matched.pt``)
was already validated as a *predictor* (24 h rollout RMSE ~0.876 C); this script
closes the loop by comparing the *downstream control* score of a PPO controller
trained on it against the canonical hourly-trained controller, on the identical
two targeted 14-day live-BOPTEST windows.

Inputs
------
Two ``benchmark_bestest_air_article7_style.py`` summary files (``summary.json``),
each a list of per-window records with at least ``controller``, ``scenario``,
``m_s``, ``violation_pct`` and ``energy_kwh`` fields:

  --hourly-summary  : canonical pure-v3 (hourly-trained) benchmark output
                      (default: outputs/bestest_air_article7_style_15min/summary.json)
  --matched-summary : new v3-15min-trained benchmark output
                      (default: outputs/bestest_air_pure_v3_15min/summary.json)

Outputs
-------
  reports/block2_v3_15min_closed_loop_comparison.csv
  reports/block2_v3_15min_closed_loop_comparison.json

The JSON payload carries per-window deltas and a single-sentence ``interpretation``
suitable for quoting verbatim in the manuscript.  The CSV is a tidy table with one
row per (variant, window) ready to be appended to the Block 2 controller table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical scenario keys produced by the benchmark, mapped to display labels.
WINDOWS = {
    "peak_heat_window": "Peak heat window",
    "typical_heat_window": "Typical heat window",
}
CONTROLLER = "thermostatic"  # the pure-v3 thermostatic PPO row

# Decision thresholds for the automatic interpretation.  These mirror the
# manuscript's own usability criterion (sub-5% comfort violation) and a tolerance
# band on the maintenance score.
VIOLATION_USABLE_PCT = 5.0      # both controllers are "usable" below this
MS_REL_TOLERANCE = 0.25         # |delta_ms| within 25% of the hourly score => "matched"


def _load_rows(summary_path: Path) -> dict[str, dict]:
    """Return {scenario: record} for the thermostatic controller."""
    if not summary_path.exists():
        raise SystemExit(
            f"[ERROR] summary not found: {summary_path}\n"
            f"        Run the benchmark step for this variant first (see the runbook)."
        )
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {
        rec["scenario"]: rec
        for rec in data
        if rec.get("controller") == CONTROLLER and rec.get("scenario") in WINDOWS
    }
    missing = set(WINDOWS) - set(rows)
    if missing:
        raise SystemExit(
            f"[ERROR] {summary_path} is missing thermostatic rows for: {sorted(missing)}"
        )
    return rows


def _classify(hourly: dict[str, dict], matched: dict[str, dict]) -> tuple[str, str]:
    """Return (verdict, one-sentence interpretation)."""
    matched_usable = all(matched[w]["violation_pct"] < VIOLATION_USABLE_PCT for w in WINDOWS)
    within_tol = all(
        abs(matched[w]["m_s"] - hourly[w]["m_s"]) <= MS_REL_TOLERANCE * max(hourly[w]["m_s"], 1e-9)
        for w in WINDOWS
    )
    worse = any(
        matched[w]["m_s"] > hourly[w]["m_s"] * (1.0 + MS_REL_TOLERANCE) for w in WINDOWS
    )

    if matched_usable and within_tol:
        verdict = "confound_rejected"
        msg = (
            "A PPO controller trained on the corpus-matched 15-minute v3 remains usable "
            "(sub-5% comfort violation on both windows) with a maintenance score comparable "
            "to the hourly-trained controller, so v3's training utility is attributable to its "
            "black-box rollout smoothness rather than to the training/control timestep mismatch."
        )
    elif matched_usable and worse:
        verdict = "partial_confound"
        msg = (
            "The corpus-matched 15-minute v3 still trains a usable controller but with a "
            "measurably higher maintenance score than the hourly-trained v3, indicating that "
            "temporal coarse-graining contributes part of v3's training utility alongside its "
            "black-box architecture."
        )
    elif not matched_usable:
        verdict = "timestep_driven"
        msg = (
            "A PPO controller trained on the corpus-matched 15-minute v3 fails the sub-5% "
            "comfort-violation usability bar on at least one window, indicating that v3's "
            "training utility depends substantially on the coarser hourly-step dynamics rather "
            "than on its black-box architecture alone; the paradox framing is then refined to a "
            "temporal-coarse-graining mechanism."
        )
    else:
        verdict = "inconclusive"
        msg = (
            "The corpus-matched 15-minute v3 yields a controller whose score is neither clearly "
            "matched to nor clearly worse than the hourly-trained baseline; report both rows and "
            "let the per-window numbers speak."
        )
    return verdict, msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hourly-summary",
        default="outputs/bestest_air_article7_style_15min/summary.json",
        help="Canonical pure-v3 (hourly-trained) benchmark summary.json",
    )
    ap.add_argument(
        "--matched-summary",
        default="outputs/bestest_air_pure_v3_15min/summary.json",
        help="New v3-15min-trained benchmark summary.json",
    )
    ap.add_argument("--out-csv", default="reports/block2_v3_15min_closed_loop_comparison.csv")
    ap.add_argument("--out-json", default="reports/block2_v3_15min_closed_loop_comparison.json")
    args = ap.parse_args()

    hourly = _load_rows((ROOT / args.hourly_summary).resolve())
    matched = _load_rows((ROOT / args.matched_summary).resolve())

    variants = [
        ("pure_v3_hourly", "v3 hourly-trained (canonical)", hourly),
        ("pure_v3_15min", "v3 15-min matched-resolution", matched),
    ]

    # Tidy per-(variant, window) table.
    table_rows: list[dict] = []
    for key, label, rows in variants:
        for scenario, wlabel in WINDOWS.items():
            rec = rows[scenario]
            table_rows.append(
                {
                    "variant": key,
                    "variant_label": label,
                    "window": scenario,
                    "window_label": wlabel,
                    "m_s": round(rec["m_s"], 4),
                    "violation_pct": round(rec["violation_pct"], 3),
                    "energy_kwh": round(rec["energy_kwh"], 2),
                    "rmse_22_c": round(rec.get("rmse_22_c", float("nan")), 4),
                }
            )

    # Per-window deltas (matched minus hourly).
    deltas = {
        scenario: {
            "m_s_hourly": round(hourly[scenario]["m_s"], 4),
            "m_s_matched": round(matched[scenario]["m_s"], 4),
            "delta_m_s": round(matched[scenario]["m_s"] - hourly[scenario]["m_s"], 4),
            "violation_pct_hourly": round(hourly[scenario]["violation_pct"], 3),
            "violation_pct_matched": round(matched[scenario]["violation_pct"], 3),
            "energy_kwh_hourly": round(hourly[scenario]["energy_kwh"], 2),
            "energy_kwh_matched": round(matched[scenario]["energy_kwh"], 2),
        }
        for scenario in WINDOWS
    }

    verdict, interpretation = _classify(hourly, matched)

    out_csv = (ROOT / args.out_csv).resolve()
    out_json = (ROOT / args.out_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    payload = {
        "experiment": "v3_15min_closed_loop_control_utility",
        "windows": deltas,
        "verdict": verdict,
        "interpretation": interpretation,
        "thresholds": {
            "violation_usable_pct": VIOLATION_USABLE_PCT,
            "m_s_rel_tolerance": MS_REL_TOLERANCE,
        },
        "sources": {
            "hourly_summary": args.hourly_summary,
            "matched_summary": args.matched_summary,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Console digest.
    print("=" * 78)
    print("v3-15min closed-loop control-utility comparison")
    print("=" * 78)
    for scenario, wlabel in WINDOWS.items():
        d = deltas[scenario]
        print(
            f"{wlabel:<22} m_s: hourly={d['m_s_hourly']:.3f}  "
            f"15min={d['m_s_matched']:.3f}  delta={d['delta_m_s']:+.3f}  "
            f"(viol% {d['violation_pct_hourly']:.1f} -> {d['violation_pct_matched']:.1f})"
        )
    print("-" * 78)
    print(f"VERDICT: {verdict}")
    print(interpretation)
    print("-" * 78)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
