"""Methodology Figure 1 -- consolidated data-driven architecture (schematic, no data).

Self-describing: surrogate roles (incl. Stage A/B/C calibration of GB), the three
controller families (incl. what MORL 5D vs 17D means), and the live-evaluation targets
are all labelled inside the figure so the caption can stay short.

  BB = black-box surrogate (v3)   GB = grey-box surrogate (v3.5)

Output: docs/results2_control_overleaf/figures/block2_system_overview.pdf (+ .png)
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

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_system_overview.pdf"


def main() -> None:
    fs.apply()
    fig = plt.figure(figsize=(12.8, 5.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for x, t in [(0.18, "Surrogate training environments"),
                 (0.52, "Controller families (all PPO-based)"),
                 (0.85, "Live BOPTEST evaluation")]:
        ax.text(x, 0.975, t, ha="center", fontsize=10.5, weight="bold", color="0.2")

    def box(x, y, w, h, head, sub, color, fc=None, hfs=9.0, sfs=7.0):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=1.6, edgecolor=color, facecolor=(fc if fc else color + "14")))
        ax.text(x, y + h / 2 - 0.045, head, ha="center", va="top",
                fontsize=hfs, weight="bold", color=color)
        ax.text(x, y - 0.028, sub, ha="center", va="center", fontsize=sfs, color="0.22")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, color="0.45", lw=1.4, shrinkA=2, shrinkB=2))

    bw, bh = 0.31, 0.215
    by = [0.755, 0.495, 0.235]
    box(0.18, by[0], bw, bh, "Black-box surrogate (BB)",
        "hourly step · smooth, lower-fidelity rollouts\ntrained on a scripted-excitation corpus", fs.V3)
    box(0.18, by[1], bw, bh, "Grey-box surrogate (GB)",
        "RC physics + neural-residual heat-flow head\ncalibrated via Stage A$\\to$B$\\to$C inverse ID", fs.ACCURATE)
    box(0.18, by[2], bw, bh, "Hybrid",
        "BB supplies rollouts;\nfrozen GB = reward-shaping censor only", fs.HYBRID)

    cw, ch = 0.29, 0.215
    cy = by
    box(0.52, cy[0], cw, ch, "Thermostatic PPO",
        "single supply-temperature setpoint policy", fs.EDGE, fc="#f5f5f5")
    box(0.52, cy[1], cw, ch, "Hierarchical RL (HDRL)",
        "seasonal gate $\\to$ winter / summer sub-policies", fs.EDGE, fc="#f5f5f5")
    box(0.52, cy[2], cw, ch, "Multi-objective RL (MORL)",
        "comfort–energy preference policy\n5D compact obs. (fails) $\\to$ 17D forecast-augmented (works)",
        fs.EDGE, fc="#f5f5f5", sfs=6.6)

    box(0.85, 0.625, 0.28, 0.215, "bestest_air (source case)",
        # Escaped: usetex is on for this figure and a raw & is read as an
        # alignment tab, which halts latex rather than rendering an ampersand.
        "peak (cold January) \\& typical (February)\n14-day live closed-loop windows",
        fs.NEUTRAL, sfs=6.8)
    box(0.85, 0.345, 0.28, 0.215, "Hydronic family (3 cases)",
        "calibration transfer vs frozen-policy transfer", fs.NEUTRAL)

    for y in by:
        arrow(0.18 + bw / 2, y, 0.52 - cw / 2, cy[1] if y == by[1] else y)
    ax.text(0.35, 0.88, "train on\nsurrogate only", ha="center", va="center",
            fontsize=7.4, color="0.4", style="italic")
    for y in cy:
        arrow(0.52 + cw / 2, y, 0.85 - 0.14, 0.625 if y > by[1] else 0.345)
    ax.text(0.70, 0.88, "deploy on live emulator\n(MORL adds a short live finetune)",
            ha="center", va="center", fontsize=7.4, color="0.4", style="italic")

    ax.text(0.5, 0.045,
            "Live metric: maintenance score $m_s$ = time-in-violation + worst-case severity ($m_s\\!>\\!1$ = unusable); "
            "also comfort-violation %, energy, 24 h rollout RMSE$_T$.",
            ha="center", fontsize=7.2, color="0.35")
    ax.text(0.5, 0.010, "Schematic (no data).", ha="center", fontsize=7.0, color="0.55", style="italic")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)")


if __name__ == "__main__":
    main()
