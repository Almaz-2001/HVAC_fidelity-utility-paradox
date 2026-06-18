"""Elsevier graphical abstract (single landscape panel, ~1328x531 px).

Four-column causal chain (engineering, data-driven):
  Training backend -> Action-response surface -> PPO behaviour -> Live BOPTEST outcome

  v3 (coarse)            smooth                modulating, 24% sat.   USABLE  (m_s~0.08)
  v3.5 / matched v3      rough (7.9-9.4x)      bang-bang, 100% sat.   COLLAPSE (m_s>1)
  hybrid                 smooth + censored     modulating, ~25% sat.  ROBUST  (m_s 0.041, 85x)

Numbers: roughness from reports/block2_mechanism_surface_sharpness.csv; m_s from
reports/block2_fidelity_utility_scatter.csv; action saturation from the committed
BOPTEST traces; RMSE / 85x are the canonical values cited in the paper.
Footer legend defines RMSE_T, m_s, violation so the figure is self-contained.
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


def norm_mean_curve(kw):
    cs = msd.per_state_curves(msd.load_direct_tsup_adapter(**kw))
    m = np.mean(cs, axis=0)
    return (m - m.mean()) / (m.max() - m.min())


def sat_pct(run_dir):
    a = pd.read_csv(ROOT / "outputs" / run_dir / "traces" / "peak_heat_window_thermostatic.csv")["a0"]
    return 100.0 * float((a.abs() > 0.9).mean())


def main() -> None:
    v3_kw = dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    v35_kw = dict(kind="v35_calibrated", summary_json=msd.V35_SUMMARY)
    n_v3, n_v35 = norm_mean_curve(v3_kw), norm_mean_curve(v35_kw)
    a0 = msd.A0
    sat_v3 = sat_pct("bestest_air_article7_style_15min")
    sat_acc = sat_pct("block2_bestest_air_15min_thermostatic_v35")
    sat_hyb = sat_pct("block2_thermostatic_hybrid_v3_v35_l010")

    fig = plt.figure(figsize=(13.6, 5.6))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.5, 0.965, "Fidelity–Utility Paradox in Surrogate-Based RL for HVAC Control",
            ha="center", fontsize=15, weight="bold")
    bg.text(0.5, 0.908, "More accurate surrogates expose rougher action surfaces and collapse PPO; "
            "hybrid separates smooth rollout from physical censoring.",
            ha="center", fontsize=10.5, style="italic", color="0.3")

    cols = [(0.135, "Training backend"), (0.40, "Action-response\nsurface"),
            (0.605, "PPO behaviour"), (0.84, "Live BOPTEST outcome")]
    for x, t in cols:
        bg.text(x, 0.84, t, ha="center", fontsize=10.5, weight="bold", color="0.3")

    rows = [
        dict(y=0.66, c=GREEN, ncurve=n_v3, shape="smooth",
             backend="Coarse black-box v3\n24 h RMSE$_T$ 1.557 °C",
             ppo=f"modulating\n{sat_v3:.0f}% saturation",
             status="USABLE", outcome="$m_s\\approx0.08$"),
        dict(y=0.42, c=ORANGE, ncurve=n_v35, shape="rough",
             backend="Accurate single-model\nv3.5: RMSE$_T$ 0.644 °C, 7.9×\nmatched-v3: RMSE$_T$ 0.876 °C, 9.4×",
             ppo=f"bang-bang\n{sat_acc:.0f}% saturation",
             status="COLLAPSE", outcome="$m_s>1$"),
        dict(y=0.18, c=BLUE, ncurve=n_v3, shape="smooth + censored",
             backend="Hybrid\nv3 rollout + frozen v3.5 censor",
             ppo=f"modulating\n~{sat_hyb:.0f}% saturation",
             status="ROBUST", outcome="$m_s$=0.041 typ. · viol <5% · 85× faster"),
    ]

    def box(x, w, y, h, text, color, fs=8.6, status=False):
        bg.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=(2.0 if status else 1.5), edgecolor=color, facecolor=color + "14"))
        if status:
            head, _, tail = text.partition("\n")
            bg.text(x + w / 2, y + 0.03, head, ha="center", va="center", fontsize=11, weight="bold", color=color)
            bg.text(x + w / 2, y - 0.032, tail, ha="center", va="center", fontsize=8.4, color="0.2")
        else:
            bg.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color="black")

    def arr(x0, x1, y, color):
        bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.6))

    bg.text(0.285, 0.42, "normalised response", rotation=90, va="center", ha="center", fontsize=8, color="0.45")
    for r in rows:
        y, c = r["y"], r["c"]
        box(0.01, 0.255, y, 0.165, r["backend"], c, fs=8.3)             # col 1 backend
        arr(0.272, 0.305, y, c)
        ax = fig.add_axes([0.315, y - 0.058, 0.115, 0.12])             # col 2 response curve
        ax.plot(a0, r["ncurve"], color=c, lw=2.2)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_ylim(-0.62, 0.62); ax.set_xticks([-1, 0, 1]); ax.set_yticks([])
        ax.tick_params(labelsize=6.5, length=2)
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.text(0.5, 1.04, r["shape"], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, color=c, style="italic")
        arr(0.45, 0.485, y, c)
        box(0.49, 0.205, y, 0.155, r["ppo"], c, fs=8.6)                # col 3 PPO behaviour
        arr(0.70, 0.735, y, c)
        box(0.74, 0.25, y, 0.165, f"{r['status']}\n{r['outcome']}", c, status=True)  # col 4 outcome

    bg.text(0.5, 0.035, "RMSE$_T$ = 24 h predictive temperature error   ·   "
            "$m_s$ = live maintenance score (lower is better)   ·   "
            "violation = fraction of time outside the 21-24 °C comfort band",
            ha="center", fontsize=8.3, color="0.35")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}  [sat v3={sat_v3:.0f} acc={sat_acc:.0f} hyb={sat_hyb:.0f}]")


if __name__ == "__main__":
    main()
