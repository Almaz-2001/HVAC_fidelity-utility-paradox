"""Seed-robustness figure for the Block 2 controllers (N = 3 seeds).

The most common reviewer objection to a single-seed RL result is that the effect
is a seed artefact. This figure answers it directly: every number is the mean +/-
standard deviation over seeds 42/43/44 read straight from the committed
``reports/block2_thermostatic_seed_band.csv`` (no hand entry), shown against the two
pre-specified engineering references -- the m_s = 1 closed-loop-collapse line and the
5 % comfort-violation bar. The usable hourly-v3 and hybrid controllers sit below both
references on every seed, while the matched-resolution v3 collapses on every seed with
negligible seed variance, so the fidelity-utility paradox is seed-robust.

Output: docs/results2_control_overleaf/figures/block2_seed_band.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

SRC = ROOT / "reports/block2_thermostatic_seed_band.csv"
FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_seed_band.pdf"

# controller label in the CSV -> (display name, unified colour); narrative order
CONTROLLERS = [
    ("pure v3", "Coarse v3\n(hourly)", fs.V3),
    ("matched v3 (15-min)", "Matched-res. v3\n(15-min)", fs.MATCHED),
    ("hybrid (lambda_T=0.10)", "Hybrid\n(v3 + v3.5 censor)", fs.HYBRID),
]
# window -> (marker, filled?, x-offset, legend label)
WINDOWS = [("peak", "o", True, -0.12, "peak-heat window"),
           ("typical", "s", False, 0.12, "typical-heat window")]


def _row(df, ctrl, window):
    m = df[(df.controller == ctrl) & (df.window == window)]
    return None if m.empty else m.iloc[0]


def main() -> None:
    fs.apply()
    df = pd.read_csv(SRC)
    n_seeds = int(df["n_seeds"].max())
    seeds = df["seeds"].iloc[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = list(range(len(CONTROLLERS)))

    panels = [
        (axA, "m_s_mean", "m_s_std", "Live maintenance score $m_s$ (lower is better)",
         [(fs.MS_COLLAPSE, "$m_s=1$ closed-loop collapse", "--"),
          (fs.MS_USABLE, f"usable band $m_s<{fs.MS_USABLE:g}$", ":")]),
        (axB, "violation_pct_mean", "violation_pct_std", "Comfort violation (% of time outside 21-24 °C)",
         [(fs.VIOLATION_BAR, f"{fs.VIOLATION_BAR:g}% violation reference", "--")]),
    ]

    for ax, mcol, scol, ylab, refs in panels:
        for window, marker, filled, dx, _ in WINDOWS:
            for xi, (ctrl, _name, color) in zip(x, CONTROLLERS):
                r = _row(df, ctrl, window)
                if r is None:
                    continue
                kw = dict(color=color, ms=8, mew=1.6, capsize=4, elinewidth=1.6, lw=0)
                if filled:
                    ax.errorbar(xi + dx, r[mcol], yerr=r[scol], marker=marker, mfc=color, **kw)
                else:
                    ax.errorbar(xi + dx, r[mcol], yerr=r[scol], marker=marker, mfc="white", **kw)
        # engineering reference lines / bands (pre-specified, from _figstyle)
        for val, lab, ls in refs:
            ax.axhline(val, color="0.35", lw=1.1, ls=ls, zorder=0)
            ax.text(len(x) - 0.5, val, f" {lab}", va="bottom", ha="right",
                    fontsize=8, color="0.35", zorder=5)
        if mcol.startswith("m_s"):
            ax.axhspan(0, fs.MS_USABLE, color=fs.V3, alpha=0.06, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n, _ in CONTROLLERS], fontsize=9)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.margins(y=0.12)
        ax.set_ylim(bottom=0)

    # window legend (marker shape encodes the targeted window; colour encodes controller)
    handles = [Line2D([0], [0], marker=m, color="0.3", lw=0, ms=8,
                      mfc=("0.3" if f else "white"), mew=1.6, label=lab)
               for _, m, f, _, lab in WINDOWS]
    axA.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle(f"Seed robustness of the fidelity–utility paradox "
                 f"(mean ± s.d. over N={n_seeds} seeds: {seeds})",
                 fontsize=12, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT}  [N={n_seeds} seeds {seeds}; all values from {SRC.name}]")


if __name__ == "__main__":
    main()
