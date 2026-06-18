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

COLS = ["Surrogate (fidelity)", "Action response surface", "PPO policy", "Live BOPTEST transfer"]
XS = [0.13, 0.38, 0.63, 0.88]
LANES = [
    ("#1b7837", 0.72, ["v3 black-box · 1 h step\nRMSE 1.557 °C", "smooth · monotone\nroughness 1.0×",
                        "modulating\n24% saturation", "stable transfer\n$m_s$≈0.08"]),
    ("#b2182b", 0.45, ["v3.5 / matched v3\nRMSE 0.644/0.876 °C", "rough · non-monotone\n7.9 to 9.4× rougher",
                       "near bang-bang\n100% saturation", "closed-loop collapse\n$m_s$>1, >77% viol."]),
    ("#2166ac", 0.18, ["hybrid: v3 rollout\n+ frozen v3.5 censor", "smooth (v3) +\nplausibility censor",
                       "modulating\n25% saturation", "robust transfer\n$m_s$=0.041, viol <5%"]),
]
BW, BH = 0.205, 0.17


def box(ax, x, y, text, color, status=False):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                 boxstyle="round,pad=0.010,rounding_size=0.02",
                 linewidth=(2.0 if status else 1.4), edgecolor=color, facecolor=color + "14"))
    head, _, tail = text.partition("\n")
    ax.text(x, y + 0.028, head, ha="center", va="center", fontsize=8.6,
            weight=("bold" if status else "normal"), color=(color if status else "black"))
    ax.text(x, y - 0.034, tail, ha="center", va="center", fontsize=7.8, color="0.3")


def arrow(ax, x0, x1, y, color):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10,
                 linewidth=1.2, color=color, shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    for x, label in zip(XS, COLS):
        ax.text(x, 0.95, label, ha="center", va="center", fontsize=10.5, weight="bold", color="0.25")

    for color, y, texts in LANES:
        for j, (x, t) in enumerate(zip(XS, texts)):
            box(ax, x, y, t, color, status=(j == len(XS) - 1))   # last column = formal status
            if j < len(XS) - 1:
                arrow(ax, x + BW / 2 + 0.004, XS[j + 1] - BW / 2 - 0.004, y, color)

    ax.set_title("The fidelity–utility paradox and its resolution",
                 fontsize=12.5, weight="bold", pad=12)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
