"""Elsevier graphical abstract (single landscape panel, 1328x531 px).

Left: the paradox -- across single-model surrogates, a more accurate twin (lower 24h
rollout RMSE) gives a WORSE live controller (higher maintenance score m_s).
Right: the resolution -- a role-separating hybrid (v3 smooth rollout + frozen v3.5
plausibility censor) escapes the trade-off (robust, sub-5% violation).

Numbers are read from reports/block2_fidelity_utility_scatter.csv (data-driven).
Output: docs/paper_combined/figures/graphical_abstract.{pdf,png}
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper_combined/figures/graphical_abstract"
CSV = ROOT / "reports/block2_fidelity_utility_scatter.csv"

GREEN, RED, BLUE = "#1b7837", "#b2182b", "#2166ac"


def main() -> None:
    df = pd.read_csv(CSV)

    def row(controller_startswith):
        return df[df.controller.str.startswith(controller_startswith)].iloc[0]

    v3, mv, v35, hy = row("v3 ("), row("matched"), row("v3.5"), row("hybrid")

    fig = plt.figure(figsize=(13.28, 5.31))
    plt.rcParams.update({"font.size": 12})
    fig.suptitle("The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
                 fontsize=15, weight="bold", y=0.985)

    # ---- LEFT: the paradox (compact scatter) ----
    axL = fig.add_axes([0.055, 0.10, 0.46, 0.74])
    pts = [(v3, GREEN, "o", "v3 (hourly)\nusable"),
           (mv, RED, "o", "matched v3\ncollapse"),
           (v35, RED, "o", "v3.5 calibrated\ncollapse")]
    axL.axhspan(1.0, 1.4, color=RED, alpha=0.06)
    axL.plot([v3.rmse_24h_c, mv.rmse_24h_c, v35.rmse_24h_c],
             [v3.m_s_mean, mv.m_s_mean, v35.m_s_mean], "--", color="0.55", lw=1.6, zorder=1)
    for r, c, m, lab in pts:
        axL.scatter(r.rmse_24h_c, r.m_s_mean, s=170, color=c, marker=m, zorder=3,
                    edgecolor="white", linewidth=1.2)
    axL.scatter(hy.rmse_24h_c, hy.m_s_mean, s=210, color=BLUE, marker="D", zorder=4,
                edgecolor="white", linewidth=1.3)
    axL.annotate("hybrid\n(robust)", (hy.rmse_24h_c, hy.m_s_mean), xytext=(-4, 30),
                 textcoords="offset points", ha="right", color=BLUE, fontsize=11, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    axL.annotate("v3 (usable)", (v3.rmse_24h_c, v3.m_s_mean), xytext=(8, -4),
                 textcoords="offset points", color=GREEN, fontsize=10)
    axL.annotate("matched v3", (mv.rmse_24h_c, mv.m_s_mean), xytext=(6, 6),
                 textcoords="offset points", color=RED, fontsize=10)
    axL.annotate("v3.5", (v35.rmse_24h_c, v35.m_s_mean), xytext=(6, 6),
                 textcoords="offset points", color=RED, fontsize=10)
    axL.text(0.62, 1.0, "more accurate surrogate\n$\\longrightarrow$ worse controller",
             color="0.3", fontsize=11, ha="center", style="italic")
    axL.set_xlim(1.65, 0.55)
    axL.set_ylim(-0.05, 1.35)
    axL.set_xlabel("24-h rollout RMSE (°C)  →  more accurate", fontsize=11)
    axL.set_ylabel("live $m_s$  →  worse", fontsize=11)
    axL.set_title("The paradox", fontsize=12, weight="bold")
    axL.grid(alpha=0.18)

    # ---- RIGHT: the resolution (role-separating hybrid) ----
    axR = fig.add_axes([0.56, 0.06, 0.42, 0.80]); axR.axis("off")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    axR.set_title("The resolution", fontsize=12, weight="bold")

    def box(x, y, w, h, text, color):
        axR.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                      linewidth=1.8, edgecolor=color, facecolor=color + "1A"))
        axR.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="black")

    box(0.02, 0.62, 0.44, 0.26, "v3 black-box\nsmooth rollout\ndynamics", GREEN)
    box(0.54, 0.62, 0.44, 0.26, "frozen v3.5 twin\nper-step\nplausibility censor", RED)
    box(0.27, 0.30, 0.46, 0.20, "Hybrid controller", BLUE)
    box(0.20, 0.02, 0.60, 0.18,
        "Robust on live BOPTEST:\n<5% comfort violation on both windows, ~85× faster", BLUE)
    axR.add_patch(FancyArrowPatch((0.24, 0.62), (0.42, 0.50), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.8))
    axR.add_patch(FancyArrowPatch((0.76, 0.62), (0.58, 0.50), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.8))
    axR.add_patch(FancyArrowPatch((0.5, 0.30), (0.5, 0.20), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.8))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
