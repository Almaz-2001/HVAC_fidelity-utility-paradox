"""Elsevier graphical abstract (single landscape panel, ~1328x531 px).

Engineering, data-driven, two panels (no stylised cartoons):

  (A) the measured CAUSE -- each surrogate's real one-step action->next-temperature
      response dT_hat(a0) in deg C, swept over a grid of states (faint = per state,
      bold = mean). Smooth (usable) vs rough/non-monotone (collapse) is visible.
  (B) the measured EFFECT as a deployment plane in the style of the Block 3
      comfort-energy quadrants: surrogate roughness (x) vs live maintenance score m_s
      (y), with the m_s = 1 usability threshold and shaded usable / collapse regions.
      Every controller is a real measured point (v3, matched v3, v3.5, hybrid).

All numbers from committed artifacts:
  rel_roughness -> reports/block2_mechanism_surface_sharpness.csv
  live m_s      -> reports/block2_fidelity_utility_scatter.csv
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import build_mechanism_surface_diagnostic as msd  # reuse the real surrogate curves

OUT = ROOT / "docs/paper_combined/figures/graphical_abstract"
GREEN, ORANGE, RED, BLUE = "#1b7837", "#d6604d", "#b2182b", "#2166ac"
EDGE = "#222222"


def state_curves(kw):
    return msd.per_state_curves(msd.load_direct_tsup_adapter(**kw))


def main() -> None:
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    c_v3 = state_curves(v3_kw)
    c_v35 = state_curves(v35_kw)
    a0 = msd.A0

    rough = pd.read_csv(ROOT / "reports/block2_mechanism_surface_sharpness.csv")
    rr = {r.surrogate: float(r.rel_roughness) for r in rough.itertuples()}
    sc = pd.read_csv(ROOT / "reports/block2_fidelity_utility_scatter.csv")
    ms = {c: float(sc[sc.controller.str.startswith(c)].iloc[0]["m_s_mean"]) for c in ["v3 (", "matched", "v3.5", "hybrid"]}

    fig = plt.figure(figsize=(12.6, 5.3))
    fig.suptitle("The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
                 fontsize=15, weight="bold", y=0.99)
    fig.text(0.5, 0.905, "A more accurate surrogate exposes a rougher action$\\rightarrow$temperature response (A); "
             "a policy trained on it collapses on the live building (B).\nA role-separating hybrid keeps the response "
             "smooth and the controller deployable.", ha="center", fontsize=10.5, style="italic", color="0.3")

    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.45], left=0.065, right=0.975,
                          top=0.80, bottom=0.12, hspace=0.55, wspace=0.22)

    # ---- (A) measured response surfaces (cause) ----
    panelA = [("v3 hourly (1h)", c_v3, GREEN, "v3 (coarse) — smooth"),
              ("v3.5 calibrated", c_v35, RED, "accurate twin — rough"),
              ("v3 hourly (1h)", c_v3, BLUE, "hybrid (v3 dynamics) — smooth")]
    for i, (key, cs, col, lab) in enumerate(panelA):
        ax = fig.add_subplot(gs[i, 0])
        for c in cs:
            ax.plot(a0, c, color=col, lw=0.6, alpha=0.25, zorder=1)
        ax.plot(a0, np.mean(cs, axis=0), color=col, lw=2.2, zorder=3)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_ylabel(r"$\Delta\hat T$ (°C)", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([-1, 0, 1])
        ax.text(0.5, 1.04, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color=col, style="italic")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"action $a_0$", fontsize=8)
    fig.text(0.20, 0.83, "(A) Measured response surface", ha="center", fontsize=10, weight="bold", color="0.25")

    # ---- (B) deployment plane (effect), Fig-8 engineering style ----
    axB = fig.add_subplot(gs[:, 1])
    xmax = max(rr.values()) * 1.18
    ymax = max(ms.values()) * 1.18
    # shaded interpreted regions split by the m_s = 1 usability threshold
    axB.axhspan(0, 1.0, color=GREEN, alpha=0.09)
    axB.axhspan(1.0, ymax, color=RED, alpha=0.09)
    axB.axhline(1.0, color=EDGE, ls="--", lw=1.2)
    axB.text(xmax * 0.40, 1.02, "$m_s=1$ usability threshold", ha="center", va="bottom", fontsize=8, color=EDGE)
    axB.text(xmax * 0.04, ymax * 0.93, "collapse\n(rough surrogate, $m_s>1$)", fontsize=9, color="#8a1c1c", weight="bold")
    axB.text(xmax * 0.60, 0.12, "deployable\n(smooth surrogate, $m_s<1$)", fontsize=9, color="#135f55", weight="bold")

    pts = [(rr["v3 hourly (1h)"], ms["v3 ("], GREEN, "o", "v3 (coarse)"),
           (rr["v3 matched (15min)"], ms["matched"], ORANGE, "o", "matched v3"),
           (rr["v3.5 calibrated"], ms["v3.5"], RED, "o", "v3.5 calibrated"),
           (rr["v3 hourly (1h)"], ms["hybrid"], BLUE, "D", "hybrid")]
    for x, y, c, m, lab in pts:
        axB.scatter([x], [y], s=130, color=c, marker=m, edgecolor=EDGE, linewidth=1.0, zorder=4)
    lab_off = {"v3 (coarse)": (10, 6), "matched v3": (-8, 8), "v3.5 calibrated": (8, -4), "hybrid": (10, -14)}
    for x, y, c, m, lab in pts:
        dx, dy = lab_off[lab]
        axB.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                     weight="bold", color=c, ha="left" if dx >= 0 else "right")
    axB.set_xlim(0, xmax); axB.set_ylim(-0.03, ymax)
    axB.set_xlabel("surrogate response-surface roughness  (rel. roughness)", fontsize=10)
    axB.set_ylabel("live maintenance score $m_s$  (BOPTEST)", fontsize=10)
    axB.set_title("(B) Fidelity–utility deployment plane (measured)", fontsize=11, weight="bold")
    axB.grid(alpha=0.18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
