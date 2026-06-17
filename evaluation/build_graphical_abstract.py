"""Elsevier graphical abstract (single landscape panel, ~1328x531 px).

Iconographic, cause->effect layout in three lanes. The centre column shows each
surrogate's REAL one-step action->next-temperature response curve (computed exactly
as in build_mechanism_surface_diagnostic.py), so "smooth vs rough" is visible at a
glance and is grounded in measured data, not a cartoon:

  coarse v3            smooth, monotone surface   -> usable controller
  accurate twin        rough, non-monotone        -> collapse
  hybrid (v3+censor)    smooth (v3 dynamics)       -> robust

Output: docs/paper_combined/figures/graphical_abstract.{pdf,png}
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import build_mechanism_surface_diagnostic as msd  # reuse the real surrogate curves

OUT = ROOT / "docs/paper_combined/figures/graphical_abstract"
GREEN, RED, BLUE = "#1b7837", "#b2182b", "#2166ac"


def real_curve(kind_kwargs):
    ad = msd.load_direct_tsup_adapter(**kind_kwargs)
    y = msd.curve(ad, msd.REP_TZ, msd.REP_TA)
    return (y - y.mean()) / (y.max() - y.min())  # normalised shape


def main() -> None:
    # real response-surface shapes (canonical checkpoints)
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    y_smooth = real_curve(v3_kw)        # coarse v3 (and hybrid rollout dynamics)
    y_rough = real_curve(v35_kw)        # accurate twin
    a0 = msd.A0

    fig = plt.figure(figsize=(13.28, 5.31))
    plt.rcParams.update({"font.size": 12})
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    bg.text(0.5, 0.965, "The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
            ha="center", fontsize=15, weight="bold")
    bg.text(0.5, 0.90, "A more accurate surrogate can be a worse RL training environment — it exposes a rougher "
            "action→temperature surface that the policy\nexploits into failure on the real building. "
            "A role-separating hybrid keeps the surface smooth and the controller robust.",
            ha="center", fontsize=11.5, style="italic", color="0.25")

    # column headers
    for x, t in [(0.135, "Surrogate training\nenvironment"),
                 (0.475, "Action → next-temperature\nresponse surface (measured)"),
                 (0.83, "Controller on the\nlive building")]:
        bg.text(x, 0.79, t, ha="center", fontsize=11, weight="bold", color="0.3")

    lanes = [
        (0.625, GREEN, y_smooth, "Coarse black-box v3\n(1 h step — less accurate)",
         "smooth · monotone", "✓  USABLE", "<5% comfort violation"),
        (0.405, RED, y_rough, "Accurate twin\n(calibrated v3.5 / fine-res v3)",
         "rough · non-monotone", "✗  COLLAPSE", ">77% violation,  $m_s>1$"),
        (0.185, BLUE, y_smooth, "Hybrid\n(v3 rollout + frozen\nv3.5 censor)",
         "smooth (v3 dynamics)\n+ plausibility censor", "✓  ROBUST", "<5% violation,  ~85× faster"),
    ]

    def lbox(x, y, w, h, text, color, fs=10.5, weight="normal"):
        bg.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.8, edgecolor=color, facecolor=color + "14"))
        bg.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color="black", weight=weight)

    for yc, color, ycurve, surro, shape, verdict, metric in lanes:
        # left: surrogate label
        lbox(0.02, yc, 0.235, 0.16, surro, color)
        bg.add_patch(FancyArrowPatch((0.258, yc), (0.318, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # centre: real response curve
        ax = fig.add_axes([0.345, yc - 0.072, 0.235, 0.145])
        ax.plot(a0, ycurve, color=color, lw=2.6)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.set_xlabel("action $a_0$", fontsize=8, labelpad=1)
        ax.text(0.5, 1.04, shape, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic")
        bg.add_patch(FancyArrowPatch((0.60, yc), (0.66, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # right: outcome badge (verdict line + metric line)
        bg.add_patch(FancyBboxPatch((0.665, yc - 0.08), 0.315, 0.16,
                     boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.8, edgecolor=color, facecolor=color + "14"))
        bg.text(0.8225, yc + 0.028, verdict, ha="center", va="center", fontsize=13, weight="bold", color=color)
        bg.text(0.8225, yc - 0.04, metric, ha="center", va="center", fontsize=10, color="black")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
