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


import pandas as pd
COMFORT_LO, COMFORT_HI = 21.0, 24.0


def state_curves(kind_kwargs):
    """Real centred response curves dT_hat(a0) in deg C, one per state in the grid."""
    return msd.per_state_curves(msd.load_direct_tsup_adapter(**kind_kwargs))


def zone_trace(run_dir):
    """Real closed-loop zone temperature (deg C) vs hours from a committed BOPTEST trace."""
    df = pd.read_csv(ROOT / "outputs" / run_dir / "traces" / "peak_heat_window_thermostatic.csv")
    return df["sim_time_sec"].to_numpy() / 3600.0, df["t_zone_c"].to_numpy()


def main() -> None:
    # real response-surface curves (canonical checkpoints), in physical deg C
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    c_smooth = state_curves(v3_kw)      # coarse v3 (and hybrid rollout dynamics)
    c_rough = state_curves(v35_kw)      # accurate twin
    a0 = msd.A0
    # real closed-loop zone-temperature traces (peak window) from committed BOPTEST runs
    tr_v3 = zone_trace("bestest_air_article7_style_15min")
    tr_acc = zone_trace("block2_bestest_air_15min_thermostatic_v35")   # accurate twin (v3.5) collapse
    tr_hyb = zone_trace("block2_thermostatic_hybrid_v3_v35_l010")

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
                 (0.84, "Zone temperature on the\nlive building (measured)")]:
        bg.text(x, 0.79, t, ha="center", fontsize=11, weight="bold", color="0.3")

    lanes = [
        (0.625, GREEN, c_smooth, "Coarse black-box v3\n(1 h step — less accurate)",
         "smooth · monotone", "✓ USABLE", tr_v3),
        (0.405, RED, c_rough, "Accurate twin\n(calibrated v3.5 / fine-res v3)",
         "rough · non-monotone", "✗ COLLAPSE", tr_acc),
        (0.185, BLUE, c_smooth, "Hybrid\n(v3 rollout + frozen\nv3.5 censor)",
         "smooth (v3 dynamics)\n+ plausibility censor", "✓ ROBUST", tr_hyb),
    ]

    def lbox(x, y, w, h, text, color, fs=10.5, weight="normal"):
        bg.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.8, edgecolor=color, facecolor=color + "14"))
        bg.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color="black", weight=weight)

    import numpy as np
    for i, (yc, color, ccurves, surro, shape, verdict, tzone) in enumerate(lanes):
        bottom = (i == len(lanes) - 1)
        # left: surrogate label
        lbox(0.02, yc, 0.235, 0.16, surro, color)
        bg.add_patch(FancyArrowPatch((0.258, yc), (0.318, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # centre: real measured response in deg C (faint = per state, bold = mean)
        ax = fig.add_axes([0.345, yc - 0.072, 0.235, 0.145])
        for c in ccurves:
            ax.plot(a0, c, color=color, lw=0.6, alpha=0.28, zorder=1)
        ax.plot(a0, np.mean(ccurves, axis=0), color=color, lw=2.6, zorder=3)
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_xticks([-1, 0, 1])
        if not bottom:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("action $a_0$", fontsize=7.5, labelpad=1)
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.set_ylabel(r"$\Delta\hat T$ (°C)", fontsize=7.5, labelpad=1)
        ax.text(0.5, 1.07, shape, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic")
        bg.add_patch(FancyArrowPatch((0.60, yc), (0.66, yc), arrowstyle="-|>", mutation_scale=15, color=color, lw=1.8))
        # right: REAL closed-loop zone-temperature trace + comfort band (measured BOPTEST)
        axr = fig.add_axes([0.675, yc - 0.072, 0.235, 0.145])
        th, tz = tzone
        th = th - th.min()          # window-relative hours (traces are time-stamped from year start)
        win = th <= 120.0           # first 5 days: behaviour is daily-periodic, so this shows
        th, tz = th[win], tz[win]   # the real oscillation instead of a 14-day dense blur
        axr.axhspan(COMFORT_LO, COMFORT_HI, color="#1b7837", alpha=0.13, zorder=0)
        # the collapse trace is a dense daily sawtooth -> very thin line so it reads as a
        # fine oscillation rather than a heavy red band; smooth traces stay normal weight
        axr.plot(th, tz, color=color, lw=(0.3 if color == RED else 0.9),
                 alpha=(0.85 if color == RED else 1.0), zorder=2)
        axr.set_ylim(14, 35); axr.set_xlim(0, 120)
        axr.tick_params(labelsize=6.5, length=2)
        axr.set_xticks([0, 48, 96])
        axr.set_yticks([15, 21, 24, 30])
        if not bottom:
            axr.set_xticklabels([])
        else:
            axr.set_xticklabels(["0", "2", "4"])
            axr.set_xlabel("day", fontsize=7.5, labelpad=1)
        for s in axr.spines.values():
            s.set_color("0.6")
        axr.set_ylabel("zone T (°C)", fontsize=7.5, labelpad=1)
        axr.text(0.5, 1.07, verdict, transform=axr.transAxes, ha="center", va="bottom",
                 fontsize=11, weight="bold", color=color)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
