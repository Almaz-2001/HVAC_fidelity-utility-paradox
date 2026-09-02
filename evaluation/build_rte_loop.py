"""Figure (supplementary) -- BOPTEST RTE closed-loop control interface.

Schematic of how the trained policy actuates the live BOPTEST Run-Time Environment via
the supply-temperature overwrite signal. Topology follows the official BOPTEST paper
(Blum et al. 2021, J. Building Performance Simulation), Figs. 1-3 and Table 1.

Output: docs/paper_combined/figures/rte_control_loop.pdf (+ .png).
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

OUT = ROOT / "docs/paper_combined/figures/rte_control_loop.pdf"
ACT = fs.ACCURATE      # action / overwrite path (single highlight)
MEAS = "#404040"       # measurement / observation path
HYB = fs.HYBRID


def box(ax, x, y, w, h, lines, edge, face, hfs=9.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 lw=1.9, edgecolor=edge, facecolor=face))
    head = lines[0]
    ax.text(x, y + h / 2 - 0.085, head, ha="center", va="top",
            fontsize=hfs, weight="bold", color=edge)
    if len(lines) > 1:
        ax.text(x, y - 0.02, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=7.5, color="0.2")


def main():
    fs.apply()
    fig = plt.figure(figsize=(11.0, 4.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # left: RL controller (host) ; right: RTE (Docker)
    box(ax, 0.20, 0.50, 0.30, 0.52,
        ["RL controller (host)", "policy $\\pi_\\theta(a_t\\,|\\,o_t)$",
         "$a_t\\in[-1,1]$"], HYB, HYB + "10")
    box(ax, 0.78, 0.50, 0.34, 0.66,
        ["BOPTEST RTE  (Docker container)", "Modelica emulator FMU",
         "+ embedded baseline loops", "forecaster  ·  KPI calculator"], "0.25", "#f5f5f5")
    ax.text(0.78, 0.135, "RESTful HTTP API  (localhost:5000)", ha="center",
            fontsize=7.3, style="italic", color="0.4")

    # top path: action -> overwrite -> advance
    ax.add_patch(FancyArrowPatch((0.355, 0.66), (0.605, 0.66), arrowstyle="-|>",
                 mutation_scale=14, color=ACT, lw=2.1))
    ax.text(0.48, 0.93, "control input $u_c$ (overwrite)", ha="center", fontsize=8.0,
            weight="bold", color=ACT)
    ax.text(0.48, 0.855, "$T_{\\mathrm{sup}}=18+\\frac{a_t+1}{2}(35-18)\\,^\\circ$C", ha="center",
            fontsize=7.6, color=ACT)
    ax.text(0.48, 0.745, "oveTSetSup\\_activate = 1,   oveTSetSup\\_u = $T_{\\mathrm{sup}}$"
            "\nvia  POST /advance", ha="center", va="center",
            family="monospace", fontsize=6.7, color=ACT)

    # bottom path: measurements + forecast -> observation
    ax.add_patch(FancyArrowPatch((0.605, 0.34), (0.355, 0.34), arrowstyle="-|>",
                 mutation_scale=14, color=MEAS, lw=1.9))
    ax.text(0.48, 0.30, "measurements $y_t$ ($T_{\\mathrm{zon}}$, CO$_2$, power)  +  forecast $\\omega_t$",
            ha="center", fontsize=7.6, color=MEAS)
    ax.text(0.48, 0.225, "$\\rightarrow$ 17D observation $o_t$  ·  live KPI / $m_s$", ha="center",
            fontsize=7.4, color=MEAS)

    ax.text(0.48, 0.045, "one control step $\\Delta t=900\\,\\mathrm{s}$ (Option-1 synchronisation)",
            ha="center", fontsize=7.6, style="italic", color="0.45")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} (+ .png)")


if __name__ == "__main__":
    main()
