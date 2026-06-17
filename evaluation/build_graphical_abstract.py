"""Elsevier graphical abstract (single landscape panel, ~1328x531 px).

Three measured, data-driven lanes (cause -> effect), no stylised cartoons:

  surrogate -> measured action->next-temperature response dT_hat(a0) [deg C, real,
  swept over a grid of states] -> real closed-loop zone-temperature trace on the live
  BOPTEST building (24 h, with the comfort band).

The collapsing "accurate twin" lane is drawn in the style of the paper's warm-start
negative-control figure: a thin ORANGE trace over a 24 h window, so the bang-bang
oscillation reads as a fine sawtooth rather than a heavy band.

Numbers/traces from committed artifacts. Output: docs/paper_combined/figures/graphical_abstract.{pdf,png}
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import build_mechanism_surface_diagnostic as msd

OUT = ROOT / "docs/paper_combined/figures/graphical_abstract"
GREEN, ORANGE, BLUE = "#1b7837", "#d6604d", "#2166ac"   # orange = warm-start/collapse colour
COMFORT_LO, COMFORT_HI = 21.0, 24.0


def state_curves(kw):
    return msd.per_state_curves(msd.load_direct_tsup_adapter(**kw))


def zone_trace(run_dir):
    df = pd.read_csv(ROOT / "outputs" / run_dir / "traces" / "peak_heat_window_thermostatic.csv")
    h = df["sim_time_sec"].to_numpy() / 3600.0
    return h - h.min(), df["t_zone_c"].to_numpy()


def main() -> None:
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    c_v3, c_v35 = state_curves(v3_kw), state_curves(v35_kw)
    a0 = msd.A0
    tr_v3 = zone_trace("bestest_air_article7_style_15min")
    tr_acc = zone_trace("block2_bestest_air_15min_thermostatic_v35")
    tr_hyb = zone_trace("block2_thermostatic_hybrid_v3_v35_l010")

    fig = plt.figure(figsize=(12.8, 5.4))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.5, 0.965, "The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
            ha="center", fontsize=15, weight="bold")
    bg.text(0.5, 0.90, "A more accurate surrogate exposes a rougher action$\\rightarrow$temperature response, so the policy "
            "collapses on the real building;\na role-separating hybrid keeps the response smooth and the controller robust.",
            ha="center", fontsize=10.5, style="italic", color="0.3")
    for x, t in [(0.135, "Surrogate training\nenvironment"),
                 (0.475, "Action → next-temperature\nresponse surface (measured)"),
                 (0.84, "Zone temperature on the\nlive building (measured, 24 h)")]:
        bg.text(x, 0.79, t, ha="center", fontsize=10.5, weight="bold", color="0.3")

    lanes = [
        (0.625, GREEN, c_v3, "Coarse black-box v3\n(1 h step — less accurate)", "smooth · monotone", "✓ USABLE", tr_v3),
        (0.405, ORANGE, c_v35, "Accurate twin\n(calibrated v3.5 / fine-res v3)", "rough · non-monotone", "✗ COLLAPSE", tr_acc),
        (0.185, BLUE, c_v3, "Hybrid\n(v3 rollout + frozen\nv3.5 censor)", "smooth (v3 dynamics)", "✓ ROBUST", tr_hyb),
    ]

    for i, (yc, color, cs, surro, shape, verdict, tr) in enumerate(lanes):
        bottom = (i == len(lanes) - 1)
        # left: surrogate label
        bg.add_patch(FancyBboxPatch((0.02, yc - 0.08), 0.235, 0.16, boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.8, edgecolor=color, facecolor=color + "14"))
        bg.text(0.1375, yc, surro, ha="center", va="center", fontsize=10, color="black")
        bg.add_patch(FancyArrowPatch((0.258, yc), (0.318, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # centre: measured response surface
        ax = fig.add_axes([0.345, yc - 0.072, 0.235, 0.145])
        for c in cs:
            ax.plot(a0, c, color=color, lw=0.6, alpha=0.25, zorder=1)
        ax.plot(a0, np.mean(cs, axis=0), color=color, lw=2.4, zorder=3)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
        ax.set_xticks([-1, 0, 1]); ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylabel(r"$\Delta\hat T$ (°C)", fontsize=7.5, labelpad=1)
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.text(0.5, 1.06, shape, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=color, style="italic")
        if bottom:
            ax.set_xlabel(r"action $a_0$", fontsize=7.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        bg.add_patch(FancyArrowPatch((0.60, yc), (0.66, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # right: real 24 h zone-temperature trace, thin line (warm-start / Fig-8 style)
        axr = fig.add_axes([0.675, yc - 0.072, 0.235, 0.145])
        th, tz = tr
        w = th <= 24.0
        axr.axhspan(COMFORT_LO, COMFORT_HI, color="#1b7837", alpha=0.13, zorder=0)
        axr.plot(th[w], tz[w], color=color, lw=0.9, zorder=2)
        axr.set_ylim(14, 32); axr.set_xlim(0, 24)
        axr.set_xticks([0, 12, 24]); axr.set_yticks([15, 21, 24, 30])
        axr.tick_params(labelsize=6.5, length=2)
        for s in axr.spines.values():
            s.set_color("0.6")
        axr.set_ylabel("zone T (°C)", fontsize=7.5, labelpad=1)
        axr.text(0.5, 1.06, verdict, transform=axr.transAxes, ha="center", va="bottom", fontsize=11, weight="bold", color=color)
        if bottom:
            axr.set_xlabel("hours", fontsize=7.5, labelpad=1)
        else:
            axr.set_xticklabels([])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
