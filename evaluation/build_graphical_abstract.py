"""Elsevier graphical abstract (single landscape panel, ~1328x531 px).

Three measured, data-driven lanes (cause -> effect):
  surrogate (with its key numbers) -> NORMALISED action->next-temperature response
  shape (smooth vs rough) -> real closed-loop zone-temperature trace on the live
  BOPTEST runtime (24 h, with the comfort band). The collapsing lane is drawn in the
  warm-start negative-control style: a thin orange sawtooth.

Output: docs/paper_combined/figures/graphical_abstract.{pdf,png}
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
GREEN, ORANGE, BLUE = "#1b7837", "#d6604d", "#2166ac"
COMFORT_LO, COMFORT_HI = 21.0, 24.0


def norm_mean_curve(kw):
    cs = msd.per_state_curves(msd.load_direct_tsup_adapter(**kw))
    m = np.mean(cs, axis=0)
    return (m - m.mean()) / (m.max() - m.min())   # normalised shape, comparable across backends


def zone_trace(run_dir):
    df = pd.read_csv(ROOT / "outputs" / run_dir / "traces" / "peak_heat_window_thermostatic.csv")
    h = df["sim_time_sec"].to_numpy() / 3600.0
    return h - h.min(), df["t_zone_c"].to_numpy()


def main() -> None:
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    n_v3, n_v35 = norm_mean_curve(v3_kw), norm_mean_curve(v35_kw)
    a0 = msd.A0
    tr_v3 = zone_trace("bestest_air_article7_style_15min")
    tr_acc = zone_trace("block2_bestest_air_15min_thermostatic_v35")
    tr_hyb = zone_trace("block2_thermostatic_hybrid_v3_v35_l010")

    fig = plt.figure(figsize=(12.8, 5.4))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.5, 0.965, "The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
            ha="center", fontsize=15, weight="bold")
    bg.text(0.5, 0.905, "More accurate surrogate $\\rightarrow$ rougher action surface $\\rightarrow$ PPO collapse.   "
            "Hybrid separates rollout smoothness from physical censoring.",
            ha="center", fontsize=10.5, style="italic", color="0.3")
    for x, t in [(0.15, "Surrogate training\nenvironment"),
                 (0.48, "Normalised action → next-\ntemperature response"),
                 (0.84, "Zone temperature on the\nlive BOPTEST runtime (24 h)")]:
        bg.text(x, 0.80, t, ha="center", fontsize=10.5, weight="bold", color="0.3")

    # rows: (y, colour, normalised-response, name, numeric badge, shape-label, verdict, trace)
    rows = [
        (0.625, GREEN, n_v3, "Coarse black-box v3", "24 h RMSE 1.557 °C · $m_s$≈0.08",
         "smooth · monotone", "✓ USABLE", tr_v3),
        (0.405, ORANGE, n_v35, "Accurate twin (v3.5 / matched-v3)", "RMSE 0.644/0.876 °C · roughness 7.9–9.4× · $m_s$>1",
         "rough · non-monotone", "✗ COLLAPSE", tr_acc),
        (0.185, BLUE, n_v3, "Hybrid (v3 rollout + v3.5 censor)", "$m_s$=0.041 (typ.) · violation <5%",
         "smooth (v3 dynamics)", "✓ ROBUST", tr_hyb),
    ]

    for i, (yc, color, ncurve, name, badge, shape, verdict, tr) in enumerate(rows):
        bottom = (i == len(rows) - 1)
        # left: surrogate name + numeric badge
        bg.add_patch(FancyBboxPatch((0.015, yc - 0.085), 0.265, 0.17, boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.8, edgecolor=color, facecolor=color + "14"))
        bg.text(0.1475, yc + 0.035, name, ha="center", va="center", fontsize=9.5, weight="bold", color="black")
        bg.text(0.1475, yc - 0.035, badge, ha="center", va="center", fontsize=7.6, color="0.25")
        bg.add_patch(FancyArrowPatch((0.285, yc), (0.335, yc), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.7))
        # centre: NORMALISED response shape (same scale for all backends)
        ax = fig.add_axes([0.36, yc - 0.07, 0.215, 0.14])
        ax.plot(a0, ncurve, color=color, lw=2.4)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
        ax.set_ylim(-0.62, 0.62); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0, 0.5])
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylabel("norm.\nresponse", fontsize=7, labelpad=1)
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.text(0.5, 1.06, shape, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3, color=color, style="italic")
        ax.set_xlabel(r"action $a_0$", fontsize=7.5, labelpad=1) if bottom else ax.set_xticklabels([])
        bg.add_patch(FancyArrowPatch((0.59, yc), (0.64, yc), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.7))
        # right: real 24 h zone-temperature trace, thin line (warm-start style)
        axr = fig.add_axes([0.665, yc - 0.07, 0.235, 0.14])
        th, tz = tr; w = th <= 24.0
        axr.axhspan(COMFORT_LO, COMFORT_HI, color="#1b7837", alpha=0.13, zorder=0)
        axr.plot(th[w], tz[w], color=color, lw=0.9, zorder=2)
        axr.set_ylim(14, 32); axr.set_xlim(0, 24)
        axr.set_xticks([0, 12, 24]); axr.set_yticks([15, 21, 24, 30])
        axr.tick_params(labelsize=6.5, length=2)
        for s in axr.spines.values():
            s.set_color("0.6")
        axr.set_ylabel("zone T (°C)", fontsize=7, labelpad=1)
        axr.text(0.5, 1.06, verdict, transform=axr.transAxes, ha="center", va="bottom", fontsize=10.5, weight="bold", color=color)
        axr.set_xlabel("hours", fontsize=7.5, labelpad=1) if bottom else axr.set_xticklabels([])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
