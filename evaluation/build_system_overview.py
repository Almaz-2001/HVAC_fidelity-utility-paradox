"""Methodology Figure -- consolidated system / method overview (schematic, no data).

One figure that frames the whole study so the Methodology prose can stay short:

  Surrogate training backends      Controller families            Live BOPTEST evaluation
  -----------------------------    -------------------------      ------------------------
  v3   (coarse, 1 h, black-box) -> thermostatic PPO            -> bestest_air  (Blocks 1-2)
  v3.5 (calibrated RC+NeuralODE)   HDRL (seasonal hierarchy)      hydronic family (Block 3)
  hybrid (v3 rollout + frozen      MORL (preference-conditioned)  windows: peak / typical / yearly
          v3.5 reward censor)                                     metrics: m_s, violation, energy, RMSE_T

Backends are trained surrogate-only, then transferred to the live emulator (PPO/HDRL
zero-shot; MORL adds a short live finetune). Colours follow the paper-wide scheme
(_figstyle): v3 green, accurate single-model red, hybrid blue.

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
    fig = plt.figure(figsize=(12.2, 5.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    cols = [(0.17, "Surrogate training backends"),
            (0.52, "Controller families (PPO-based)"),
            (0.85, "Live BOPTEST evaluation")]
    for x, t in cols:
        ax.text(x, 0.95, t, ha="center", fontsize=11, weight="bold", color="0.2")

    def box(x, y, w, h, head, sub, color, fc=None, lw=1.6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=lw, edgecolor=color, facecolor=(fc if fc else color + "14")))
        ax.text(x, y + (0.028 if sub else 0), head, ha="center", va="center",
                fontsize=9.2, weight="bold", color=color)
        if sub:
            ax.text(x, y - 0.030, sub, ha="center", va="center", fontsize=7.6, color="0.25")

    def arrow(x0, y0, x1, y1, color="0.45", lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, color=color, lw=lw, shrinkA=2, shrinkB=2))

    # --- Stage 1: backends ---
    bw, bh = 0.30, 0.165
    by = [0.74, 0.50, 0.26]
    box(0.17, by[0], bw, bh, "v3  (coarse black-box)", "1 h step · smooth rollout dynamics", fs.V3)
    box(0.17, by[1], bw, bh, "v3.5  (calibrated)", "RC + NeuralODE · $C_{zon}$ Stage A/B/C", fs.ACCURATE)
    box(0.17, by[2], bw, bh, "hybrid", "v3 rollout + frozen v3.5 reward censor", fs.HYBRID)

    # --- Stage 2: controllers ---
    cw, ch = 0.27, 0.165
    cy = [0.74, 0.50, 0.26]
    box(0.52, cy[0], cw, ch, "Thermostatic PPO", "single-level setpoint policy", fs.EDGE, fc="#f5f5f5")
    box(0.52, cy[1], cw, ch, "HDRL", "seasonal hierarchical controller", fs.EDGE, fc="#f5f5f5")
    box(0.52, cy[2], cw, ch, "MORL", "preference-conditioned (5D$\\to$17D)", fs.EDGE, fc="#f5f5f5")

    # --- Stage 3: evaluation ---
    box(0.85, 0.62, 0.27, 0.20, "bestest_air", "Blocks 1–2 · peak / typical windows", fs.NEUTRAL)
    box(0.85, 0.34, 0.27, 0.20, "hydronic family", "Block 3 · 3 testcases · pre-registered\n$C_{zon}$ transfer", fs.NEUTRAL)

    # --- arrows: backends -> controllers (surrogate-only pretrain) ---
    for y in by:
        arrow(0.17 + bw / 2, y, 0.52 - cw / 2, cy[1] if y == 0.50 else y)
    ax.text(0.345, 0.86, "surrogate-only\npretrain", ha="center", va="center", fontsize=7.8,
            color="0.4", style="italic")
    # --- arrows: controllers -> live eval (zero-shot / finetune) ---
    for y in cy:
        arrow(0.52 + cw / 2, y, 0.85 - 0.135, 0.62 if y > 0.5 else 0.34)
    ax.text(0.70, 0.86, "zero-shot transfer\n(MORL: + live finetune)", ha="center", va="center",
            fontsize=7.8, color="0.4", style="italic")

    ax.text(0.5, 0.055,
            "Evaluation metrics: live maintenance score $m_s$ (= time-in-violation + severity), "
            "comfort-violation %, energy (kWh), 24 h rollout RMSE$_T$ · "
            "thresholds: $m_s=1$ collapse, 5% violation, transfer $\\tau_k=1.25\\,m_s^{PI}$.",
            ha="center", fontsize=7.8, color="0.35")
    ax.text(0.5, 0.015, "Schematic (no data).", ha="center", fontsize=7.2, color="0.55", style="italic")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)")


if __name__ == "__main__":
    main()
