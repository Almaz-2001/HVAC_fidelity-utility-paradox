"""Figure -- BOPTEST test-case schematics (primary + hydronic transfer family).

Schematic only (no measured data); component topology and numbers are taken from the
official BOPTEST test-case documentation (https://ibpsa.github.io/project1-boptest/):
  bestest_air, bestest_hydronic, bestest_hydronic_heat_pump, singlezone_commercial_hydronic.

One highlighted colour marks the actuator the RL controller commands; everything else is
neutral. Output: docs/paper_combined/figures/testcase_schematics.pdf (+ .png).
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

OUT = ROOT / "docs/paper_combined/figures/testcase_schematics.pdf"

ACT = fs.ACCURATE      # actuator the controller commands (single highlight)
SRC = fs.HYBRID        # heat-source block
NEU = "#444444"        # neutral structure
AMB = fs.PI            # ambient / disturbance

CASES = [
    dict(tag="bestest_air", role="primary source case", area="48 m$^2$ office",
         climate="Denver TMY", src="Gas boiler ($\\eta$=0.9)\n+ chiller (COP 3)",
         emit="4-pipe fan-coil", act="supply-air\ntemperature setpoint"),
    dict(tag="bestest_hydronic", role="transfer", area="48 m$^2$ residential",
         climate="Brussels", src="Gas water heater\n(5 kW)",
         emit="radiator +\nthermostatic valve", act="supply-water\ntemperature setpoint"),
    dict(tag="bestest_hydronic_heat_pump", role="transfer", area="192 m$^2$ residential",
         climate="Brussels", src="Air-to-water\nheat pump (15 kW)",
         emit="floor-heating loop", act="heat-pump\nmodulation signal"),
    dict(tag="singlezone_commercial_hydronic", role="transfer", area="8500 m$^2$ commercial",
         climate="Copenhagen", src="District heating\n(65 $^\\circ$C)",
         emit="AHU + radiator circuit", act="coil / radiator\nvalve setpoints"),
]


def panel(ax, c):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # title
    ax.text(0.5, 0.965, c["tag"].replace("_", "\\_"), ha="center", va="top",
            fontsize=9.3, weight="bold", family="monospace", color="0.1")
    ax.text(0.5, 0.895, f"{c['area']}  ·  {c['role']}  ·  {c['climate']}",
            ha="center", va="top", fontsize=7.6, color="0.35")
    # zone
    ax.add_patch(FancyBboxPatch((0.30, 0.30), 0.40, 0.34,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 lw=1.8, edgecolor=NEU, facecolor="#f5f5f5"))
    ax.text(0.50, 0.585, "Zone", ha="center", fontsize=8.6, weight="bold", color="0.1")
    ax.text(0.50, 0.485, "$C_{\\mathrm{zon}}\\,\\dot T = \\dot Q_{\\mathrm{hyd}}$"
            "\n$+ (T_{\\mathrm{amb}}{-}T)/R$", ha="center", va="center",
            fontsize=7.2, color="0.25")
    ax.text(0.50, 0.355, "sensors: $T_{\\mathrm{zon}}$, CO$_2$, power",
            ha="center", fontsize=6.8, style="italic", color="0.4")
    # ambient through envelope (left)
    ax.add_patch(FancyArrowPatch((0.08, 0.47), (0.295, 0.47), arrowstyle="-|>",
                 mutation_scale=11, color=AMB, lw=1.6))
    ax.text(0.085, 0.52, "$T_{\\mathrm{amb}}$, solar", ha="left", fontsize=7.0, color=AMB)
    ax.text(0.20, 0.43, "$R_{\\mathrm{env}}$", ha="center", fontsize=7.0, color="0.45")
    # heat source (bottom-left) -> emitter (bottom) -> zone
    ax.add_patch(FancyBboxPatch((0.05, 0.07), 0.32, 0.15,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 lw=1.6, edgecolor=SRC, facecolor=SRC + "12"))
    ax.text(0.21, 0.145, c["src"], ha="center", va="center", fontsize=7.0, color=SRC)
    ax.add_patch(FancyBboxPatch((0.46, 0.07), 0.32, 0.15,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 lw=1.6, edgecolor=NEU, facecolor="white"))
    ax.text(0.62, 0.145, c["emit"], ha="center", va="center", fontsize=7.0, color="0.15")
    ax.add_patch(FancyArrowPatch((0.37, 0.145), (0.46, 0.145), arrowstyle="-|>",
                 mutation_scale=10, color=SRC, lw=1.5))
    ax.add_patch(FancyArrowPatch((0.62, 0.22), (0.62, 0.30), arrowstyle="-|>",
                 mutation_scale=11, color=SRC, lw=1.6))
    ax.text(0.655, 0.26, "$\\dot Q_{\\mathrm{hyd}}$", ha="left", fontsize=7.0, color=SRC)
    # controller action (highlighted)
    ax.add_patch(FancyArrowPatch((0.90, 0.46), (0.78, 0.20), arrowstyle="-|>",
                 mutation_scale=12, color=ACT, lw=2.0,
                 connectionstyle="arc3,rad=-0.25"))
    ax.text(0.955, 0.55, "controller\naction", ha="center", va="center",
            fontsize=7.4, weight="bold", color=ACT)
    ax.text(0.955, 0.40, c["act"], ha="center", va="center", fontsize=6.9, color=ACT)


def main():
    fs.apply()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0))
    for ax, c in zip(axes.ravel(), CASES):
        panel(ax, c)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} (+ .png)")


if __name__ == "__main__":
    main()
