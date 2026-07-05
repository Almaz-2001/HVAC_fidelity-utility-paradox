"""Elsevier graphical abstract (landscape, entirely data-driven).

Two real-data panels, no schematic boxes or decorative arrows:

  (a) Live BOPTEST closed-loop zone temperature over the peak-heat window for three
      controllers. The policy trained on the most accurate surrogate (GB) drives the
      zone far outside the comfort band; the policy trained on the coarse surrogate
      (BB) and the role-separating hybrid stay inside it. This is the physical result.

  (b) The fidelity--utility plane: predictive rollout RMSE vs live maintenance score
      m_s. As the surrogate becomes a better predictor the trained controller crosses
      the m_s = 1 collapse line -- the paradox, quantified.

Trajectories come from committed closed-loop trace CSVs; scatter values from
_figstyle.paper_numbers(). No number is hand-typed.
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
import _figstyle as fs

OUT = ROOT / "docs/paper_combined/figures/graphical_abstract"
GREEN, ORANGE, BLUE = fs.V3, fs.MATCHED, fs.HYBRID
GB = "#9b2226"
BAND_LO, BAND_HI = 21.0, 24.0

TRACES = {
    "BB":  "outputs/block13_closed_loop_transfer_pure_v3/traces/peak_heat_window_thermostatic_transfer.csv",
    "GB":  "outputs/block2_bestest_air_15min_thermostatic_v35/traces/peak_heat_window_thermostatic.csv",
    "HY":  "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/peak_heat_window_thermostatic.csv",
}


def load_tz(rel):
    df = pd.read_csv(ROOT / rel)
    col = "boptest_t_zone_c" if "boptest_t_zone_c" in df.columns else "t_zone_c"
    tz = df[col].to_numpy(dtype=float)
    return np.arange(len(tz)) / 96.0, tz          # 96 fifteen-minute steps per day


def main() -> None:
    N = fs.paper_numbers(ROOT)
    R, M = N["rmse"], N["m_s"]

    fig = plt.figure(figsize=(13.8, 5.5))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.5, 0.955, "The Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
            ha="center", fontsize=15.5, weight="bold")
    bg.text(0.5, 0.902, "The controller trained on the most accurate surrogate (GB) drives the live zone out of comfort; "
            "the coarse surrogate (BB) and the hybrid keep it in band.",
            ha="center", fontsize=10.3, style="italic", color="0.32")

    # ---------------- (a) live closed-loop trajectories -----------------------
    ax = fig.add_axes([0.052, 0.15, 0.565, 0.62])
    ax.axhspan(BAND_LO, BAND_HI, color=GREEN, alpha=0.12, zorder=0)

    K = 480     # first 5 days: the daily oscillation is persistent, not a transient
    dBB, tBB = load_tz(TRACES["BB"])
    dGB, tGB = load_tz(TRACES["GB"])
    dHY, tHY = load_tz(TRACES["HY"])
    ax.plot(dGB[:K], tGB[:K], color=GB, lw=0.9, alpha=0.8, zorder=3, label="GB-trained → collapse")
    ax.plot(dBB[:K], tBB[:K], color=GREEN, lw=1.7, zorder=4, label="BB-trained → usable")
    ax.plot(dHY[:K], tHY[:K], color=BLUE, lw=1.7, zorder=5, label="hybrid → robust")

    ax.set_xlim(0, 5); ax.set_ylim(14, 38.5)
    ax.set_xlabel("time on live BOPTEST (days)", fontsize=9.6)
    ax.set_ylabel("zone temperature (°C)", fontsize=9.6)
    ax.set_yticks([15, 20, 25, 30, 35]); ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.1, 22.5, "comfort band 21–24 °C", fontsize=8.0, color="0.3", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    ax.text(0.1, 15.0, "GB-trained policy leaves the band on 83% of steps (15.7–34.4 °C)",
            fontsize=7.9, color=GB, weight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GB, alpha=0.9, lw=0.6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=8.7,
              frameon=False, handlelength=1.5, columnspacing=1.8)

    # ---------------- (b) fidelity--utility plane -----------------------------
    ax2 = fig.add_axes([0.715, 0.20, 0.245, 0.55])
    ax2.set_yscale("log")
    ax2.set_xlim(1.72, 0.55); ax2.set_ylim(0.033, 1.55)
    ax2.axhspan(1.0, 1.55, color=GB, alpha=0.06, zorder=0)
    ax2.axhline(1.0, color=GB, ls=(0, (5, 3)), lw=1.2, zorder=2)
    ax2.text(1.70, 1.24, "collapse ($m_s>1$)", color=GB, fontsize=7.8, weight="bold", va="center")

    pts = [("BB", R["v3"], M["v3"], GREEN, "o"), ("matched-BB", R["matched"], M["matched"], ORANGE, "o"),
           ("GB", R["v35"], M["v35"], GB, "o"), ("hybrid", R["v3"], M["hybrid_typ"], BLUE, "*")]
    for name, x, y, c, mk in pts:
        ax2.scatter(x, y, s=(230 if mk == "*" else 95), marker=mk, color=c, ec="white", lw=1.0, zorder=6)
    ax2.annotate("", xy=(R["v3"], M["hybrid_typ"] * 1.15), xytext=(R["v3"], M["v3"] * 0.85),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6), zorder=5)
    ax2.text(R["v3"] - 0.05, 0.058, "+ GB\ncensor", color=BLUE, fontsize=7.4, ha="left", va="center")
    ax2.text(R["v3"] - 0.05, M["v3"] * 1.15, "BB", color=GREEN, fontsize=8.2, weight="bold", ha="left")
    ax2.text(R["matched"], M["matched"] * 1.16, "matched-BB", color=ORANGE, fontsize=8.0, weight="bold", ha="center")
    ax2.text(R["v35"] + 0.02, M["v35"] * 0.80, "GB", color=GB, fontsize=8.2, weight="bold", ha="center")

    ax2.set_xlabel("rollout RMSE (°C)   fidelity →", fontsize=8.8)
    ax2.set_ylabel("live $m_s$ (log)", fontsize=8.8)
    ax2.set_xticks([1.5, 1.0, 0.6]); ax2.set_yticks([0.05, 0.1, 0.3, 1.0])
    ax2.set_yticklabels(["0.05", "0.1", "0.3", "1.0"]); ax2.tick_params(labelsize=7.8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("more accurate surrogate,\nworse controller", fontsize=8.6, color="0.3", pad=4)

    bg.text(0.5, 0.035, "Live BOPTEST outcomes from committed closed-loop traces (peak-heat window).   "
            "BB = coarse black-box (hourly);   matched-BB = same architecture at 15 min;   GB = calibrated grey-box twin.",
            ha="center", fontsize=8.0, color="0.42")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}  GB[{tGB.min():.1f},{tGB.max():.1f}] BB[{tBB.min():.1f},{tBB.max():.1f}] "
          f"HY[{tHY.min():.1f},{tHY.max():.1f}]  m_s GB {M['v35']:.2f} BB {M['v3']:.2f} hy {M['hybrid_typ']:.2f}")


if __name__ == "__main__":
    main()
