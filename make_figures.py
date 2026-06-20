#!/usr/bin/env python3
"""Regenerate every paper figure + the graphical abstract from the committed artefacts,
in one command. No BOPTEST and no GPU required (Level B of REPRODUCE.md).

The driver reads ``paper_artifacts/figure_manifest.yaml`` and runs each generator listed
there, so it stays in sync with the manifest automatically. A small dependency ordering
is applied: the two scripts that (re)write the reports/*.csv consumed by
``_figstyle.paper_numbers()`` run first, the block builders run in the middle, and the
number-consuming figures (Fig 1, Fig 4, Fig 8, graphical abstract) run last.

Usage:
    python make_figures.py              # regenerate all figures
    python make_figures.py --list       # print the ordered script list and exit
    python make_figures.py --only fig5 mechanism   # only scripts matching a substring
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "paper_artifacts" / "figure_manifest.yaml"

# Scripts that write reports/*.csv read by _figstyle.paper_numbers() must run first.
RUN_FIRST = [
    "evaluation/build_fidelity_utility_scatter.py",       # -> reports/block2_fidelity_utility_scatter.csv
    "evaluation/build_mechanism_surface_diagnostic.py",   # -> reports/block2_mechanism_surface_sharpness.csv
]
# Figures that read those numbers via paper_numbers() must run last.
RUN_LAST = [
    "evaluation/build_conceptual_overview.py",            # Fig 1
    "evaluation/build_runtime_fidelity_scatter.py",       # Fig 4
    "evaluation/build_lambda_specificity.py",             # Fig 8
    "evaluation/build_graphical_abstract.py",             # graphical abstract
]


def manifest_scripts() -> list[str]:
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    scripts: list[str] = []

    def add(s: str | None) -> None:
        if s and s not in scripts and (ROOT / s).exists():
            scripts.append(s)

    for section in ("main_figures", "supplementary_figures"):
        for info in (m.get(section) or {}).values():
            if isinstance(info, dict):
                add(info.get("script"))
    add((m.get("graphical_abstract") or {}).get("script"))
    return scripts


def ordered(scripts: list[str]) -> list[str]:
    first = [s for s in RUN_FIRST if s in scripts]
    last = [s for s in RUN_LAST if s in scripts]
    middle = [s for s in scripts if s not in first and s not in last]
    return first + middle + last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the ordered script list and exit")
    ap.add_argument("--only", nargs="*", metavar="SUBSTR",
                    help="run only scripts whose path contains any of these substrings")
    args = ap.parse_args()

    scripts = ordered(manifest_scripts())
    if args.only:
        scripts = [s for s in scripts if any(o in s for o in args.only)]

    if args.list:
        print("\n".join(scripts))
        return 0

    print(f"Regenerating {len(scripts)} figure generators from committed artefacts "
          f"(no BOPTEST, no GPU)\n")
    results: list[tuple[str, int, float]] = []
    for i, s in enumerate(scripts, 1):
        t0 = time.time()
        print(f"[{i:2d}/{len(scripts)}] {s}", flush=True)
        rc = subprocess.run([sys.executable, s], cwd=ROOT).returncode
        dt = time.time() - t0
        results.append((s, rc, dt))
        print(f"        {'OK' if rc == 0 else 'FAIL (exit ' + str(rc) + ')'}  ({dt:.1f}s)\n")

    passed = [s for s, rc, _ in results if rc == 0]
    failed = [s for s, rc, _ in results if rc != 0]
    print(f"=== {len(passed)}/{len(results)} generators passed "
          f"({sum(d for _, _, d in results):.0f}s total) ===")
    if failed:
        print("FAILED:")
        for s in failed:
            print("  -", s)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
