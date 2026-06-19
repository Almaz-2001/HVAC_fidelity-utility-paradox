"""Figure 4 -- runtime vs predictive-fidelity feasibility.

Surrogates are what make policy-gradient training feasible: the in-process surrogate
backends run 85-220x faster than the BOPTEST RTE HTTP testbed, turning a days-long
training run into hours. The figure plots throughput (env steps/s, log) against 24 h
rollout RMSE_T for the ground-truth emulator and the three surrogate backends. All
numbers come from committed artefacts: reports/speed_benchmark_table.csv (throughput)
and the rollout-RMSE artefacts via _figstyle.paper_numbers().

Output: docs/results2_control_overleaf/figures/block2_runtime_fidelity.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_runtime_fidelity.pdf"


def main() -> None:
    fs.apply()
    spd = pd.read_csv(ROOT / "reports/speed_benchmark_table.csv").set_index("backend")
    n = fs.paper_numbers(ROOT)
    sps = lambda b: float(spd.loc[b, "env_steps_per_sec"])
    spx = lambda b: float(spd.loc[b, "speedup_vs_boptest_rte"])

    # (label, 24h RMSE_T, steps/s, model key, speedup vs BOPTEST, note)
    pts = [
        ("BOPTEST emulator", 0.0, sps("boptest_rte_http"), "pi", 1.0, "ground truth (slowest)"),
        ("v3 hourly", n["rmse"]["v3"], sps("v3_surrogate"), "v3", spx("v3_surrogate"), ""),
        ("direct v3.5", n["rmse"]["v35"], sps("v35_calibrated_surrogate"), "v35", spx("v35_calibrated_surrogate"), ""),
        ("hybrid", n["rmse"]["v3"], sps("hybrid_v3_v35_surrogate"), "hybrid", spx("hybrid_v3_v35_surrogate"),
         "rolls out on v3"),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    ax.set_yscale("log")
    base = sps("boptest_rte_http")
    ax.axhspan(base, sps("v3_surrogate") * 3, color=fs.V3, alpha=0.05, zorder=0)
    fs.threshold(ax, base, "BOPTEST throughput", color=fs.PI, ls="--", pos=0.02, fontsize=8)
    ax.text(0.02, base * 1.6, "RL-feasible region (surrogate-accelerated)", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color=fs.V3, va="bottom")

    for label, rmse, y, key, spd_x, note in pts:
        ax.scatter(rmse, y, s=170, color=fs.COLOR[key], marker=fs.MARKER[key],
                   edgecolor="white", linewidth=1.3, zorder=4)
        tag = f"{label}\n{y:,.0f} steps/s" + (f"  ({spd_x:.0f}× BOPTEST)" if spd_x > 1 else "") + (f"\n{note}" if note else "")
        dx = -10 if key == "hybrid" else 10
        ha = "right" if key == "hybrid" else "left"
        ax.annotate(tag, (rmse, y), xytext=(dx, -2), textcoords="offset points",
                    fontsize=8.3, va="center", ha=ha, color=fs.COLOR[key])

    ax.set_xlim(-0.12, 1.95)
    ax.set_xlabel(r"24-h rollout RMSE$_T$ (°C)   (← lower = more predictive fidelity)")
    ax.set_ylabel("environment throughput (env steps/s, log scale)")
    ax.set_title("Surrogates make PPO training feasible: 85–220× the BOPTEST throughput")
    ax.grid(alpha=0.18, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)  [steps/s: BOPTEST {base:.0f}, v3 {sps('v3_surrogate'):.0f}, "
          f"v3.5 {sps('v35_calibrated_surrogate'):.0f}, hybrid {sps('hybrid_v3_v35_surrogate'):.0f}]")


if __name__ == "__main__":
    main()
