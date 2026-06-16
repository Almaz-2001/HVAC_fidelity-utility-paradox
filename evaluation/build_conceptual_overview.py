"""Conceptual overview figure: the fidelity-utility causal chain in three lanes.

A schematic (no data) that frames the whole paper before any result:

  v3 (coarse)        -> smooth surface         -> PPO stable         -> usable
  v3.5 / matched v3  -> sharp / rough surface  -> PPO bang-bang       -> collapse
  hybrid             -> smooth dyn + censor    -> stable + grounded   -> robust

Output: docs/results2_control_overleaf/figures/block2_conceptual_overview.pdf
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_conceptual_overview.pdf"

COLS = ["Surrogate training\nenvironment", "Action response\nsurface", "PPO policy\nbehaviour", "Live BOPTEST\noutcome"]
XS = [0.135, 0.385, 0.635, 0.885]
LANES = [
    ("#1b7837", 0.74, ["v3 black-box\n(coarse, 1 h step)", "smooth, monotone\n(rel. roughness 1.0×)",
                        "stable control", "usable\n($m_s\\approx0.08$)"]),
    ("#b2182b", 0.46, ["v3.5 calibrated /\nmatched v3 (accurate)", "sharp, non-monotone\n(7.9–9.4× rougher)",
                       "near bang-bang\nsaturation", "collapse\n($m_s>1$, >77% viol.)"]),
    ("#2166ac", 0.18, ["hybrid: v3 rollout +\nfrozen v3.5 censor", "smooth dynamics +\nplausibility censor",
                       "stable & physically\ngrounded", "robust\n($m_s\\approx0.04$)"]),
]
BW, BH = 0.20, 0.16


def box(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 linewidth=1.6, edgecolor=color, facecolor=color + "1A"))  # ~10% alpha hex
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="black")


def arrow(ax, x0, x1, y, color):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=15,
                 linewidth=1.6, color=color, shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    for x, label in zip(XS, COLS):
        ax.text(x, 0.95, label, ha="center", va="center", fontsize=10.5, weight="bold", color="0.25")

    for color, y, texts in LANES:
        for j, (x, t) in enumerate(zip(XS, texts)):
            box(ax, x, y, t, color)
            if j < len(XS) - 1:
                arrow(ax, x + BW / 2 + 0.005, XS[j + 1] - BW / 2 - 0.005, y, color)

    ax.text(0.02, LANES[0][1], "more accurate $\\rightarrow$ worse", rotation=90, va="center",
            ha="center", fontsize=8, color="0.5")
    ax.set_title("The fidelity–utility paradox and its resolution: more predictive accuracy can mean a worse "
                 "training environment", fontsize=11.5, pad=14)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
