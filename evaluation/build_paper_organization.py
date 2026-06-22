"""Figure 2 -- 'Organization of paper' section-flow diagram (schematic, no data).

Mirrors the RINENG convention (a visual roadmap of the section structure):
  I. Introduction -> II. Related Work -> III. Proposed Work
    -> IV. Experimental Results {Results I digital-twin fidelity / II control utility /
       III transferability} -> V. Conclusion

Output: docs/results2_control_overleaf/figures/block2_paper_organization.pdf (+ .png)
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_paper_organization.pdf"


def main() -> None:
    fs.apply()
    fig = plt.figure(figsize=(12.2, 3.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, head, sub, color, fc=None, hfs=9.6, sfs=7.8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=1.7, edgecolor=color, facecolor=(fc if fc else color + "12")))
        ax.text(x, y + (h / 2 - 0.085 if sub else 0), head, ha="center",
                va=("top" if sub else "center"), fontsize=hfs, weight="bold", color=color)
        if sub:
            ax.text(x, y - 0.045, sub, ha="center", va="center", fontsize=sfs, color="0.2")

    def arr(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=13, color="0.45", lw=1.5))

    y = 0.55
    box(0.085, y, 0.15, 0.30, "I. Introduction", "paradox,\nhypotheses H1–H4", fs.EDGE, "#f5f5f5")
    box(0.255, y, 0.15, 0.30, "II. Related\nWork", "surrogates,\nDRL, transfer", fs.EDGE, "#f5f5f5")
    box(0.43, y, 0.16, 0.30, "III. Proposed\nWork", "v3 / v3.5 / hybrid,\nreward censor", fs.HYBRID)
    # IV. Experimental Results -- expanded into the three evidence blocks
    bx, bw = 0.665, 0.20
    ax.add_patch(FancyBboxPatch((bx - bw / 2, 0.16), bw, 0.78,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=1.8, edgecolor=fs.V3, facecolor=fs.V3 + "0e"))
    ax.text(bx, 0.90, "IV. Experimental Results", ha="center", va="top", fontsize=9.6, weight="bold", color=fs.V3)
    for yy, t in [(0.70, "Results I — digital-twin fidelity"),
                  (0.50, "Results II — control utility (paradox + hybrid)"),
                  (0.30, "Results III — transferability")]:
        ax.add_patch(FancyBboxPatch((bx - bw / 2 + 0.012, yy - 0.055), bw - 0.024, 0.10,
                     boxstyle="round,pad=0.004", linewidth=1.0, edgecolor=fs.V3, facecolor="white"))
        ax.text(bx, yy, t, ha="center", va="center", fontsize=7.4, color="0.15")
    box(0.90, y, 0.13, 0.30, "V. Conclusion", "boundary,\nBlock 4 outlook", fs.EDGE, "#f5f5f5")

    arr(0.085 + 0.075, 0.255 - 0.075, y)
    arr(0.255 + 0.075, 0.43 - 0.08, y)
    arr(0.43 + 0.08, bx - bw / 2, y)
    arr(bx + bw / 2, 0.90 - 0.065, y)

    ax.text(0.5, 0.045, "Organization of the paper.", ha="center", fontsize=8.2, style="italic", color="0.4")
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)")


if __name__ == "__main__":
    main()
