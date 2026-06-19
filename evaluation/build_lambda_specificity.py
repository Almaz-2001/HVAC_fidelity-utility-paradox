"""Figure 8 -- controller-family specificity of the censor weight lambda_T.

The optimal disagreement-penalty weight lambda_T is NOT universal: it depends on the
controller family. Using the live-BOPTEST maintenance score m_s averaged over the peak
and typical windows (cross-window mean), each family's optimum sits at a different
lambda_T:

  * thermostatic PPO : optimum at lambda_T = 0.10 (the canonical hybrid)
  * HDRL             : optimum at lambda_T = 0.00 (penalty only hurts; full sweep)
  * MORL             : adopted at lambda_T = 0.00

Every point is read from a committed artefact (no hand entry); only HDRL has a dense
sweep, so PPO is shown with its two measured endpoints and MORL with its single adopted
configuration. The y-axis compares the *location of the optimum*, not absolute m_s
across families (the families have different observation/action interfaces).

Sources: reports/block2_hdrl_lambda_sweep_summary.csv (HDRL),
         reports/block2_fidelity_utility_scatter.csv (PPO: pure v3 = lambda 0, hybrid = 0.10),
         reports/block2_morl_comparison_summary.csv (MORL adopted config).

Output: docs/results2_control_overleaf/figures/block2_lambda_specificity.pdf
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

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_lambda_specificity.pdf"

FAM = {  # family -> (colour, marker)
    "thermostatic PPO": ("#2166ac", "o"),
    "HDRL": ("#6f4e7c", "s"),
    "MORL": ("#b25f2c", "D"),
}


def collect():
    # HDRL: dense lambda sweep -> cross-window mean m_s per lambda
    h = pd.read_csv(ROOT / "reports/block2_hdrl_lambda_sweep_summary.csv")
    lam = {"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10}
    hdrl = []
    for v, lv in lam.items():
        d = h[h.variant == v]
        if len(d) >= 2:
            hdrl.append((lv, float(d["m_s"].mean())))
    hdrl.sort()

    # PPO: pure v3 (lambda_T = 0) and hybrid (lambda_T = 0.10), cross-window mean m_s
    sc = pd.read_csv(ROOT / "reports/block2_fidelity_utility_scatter.csv")
    def ms_mean(prefix):
        return float(sc[sc.controller.str.startswith(prefix)].iloc[0]["m_s_mean"])
    ppo = [(0.00, ms_mean("v3 (")), (0.10, ms_mean("hybrid"))]

    # MORL: adopted lambda_temp_disagree = 0.00; usable 17D canonical m_s
    m = pd.read_csv(ROOT / "reports/block2_morl_comparison_summary.csv")
    m17 = m[m.variant.str.contains("17", case=False)].iloc[0]
    morl = [(float(m17["lambda_temp_disagree"]), float(m17["m_s"]))]
    return {"thermostatic PPO": ppo, "HDRL": hdrl, "MORL": morl}


def main() -> None:
    fs.apply()
    data = collect()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axhspan(0, fs.MS_USABLE, color=fs.V3, alpha=0.10, zorder=0)
    fs.threshold(ax, fs.MS_USABLE, f"usable $m_s<{fs.MS_USABLE:g}$", color=fs.V3, ls=":", pos=0.99, fontsize=8)

    for fam, pts in data.items():
        c, mk = FAM[fam]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, color=c, lw=1.8, marker=mk, ms=9, mec="white", mew=1.0, label=fam, zorder=3)
        else:
            ax.scatter(xs, ys, color=c, marker=mk, s=95, edgecolor="white", linewidth=1.0, label=f"{fam} (adopted)", zorder=3)
        # star the family optimum (lowest cross-window m_s)
        opt = min(pts, key=lambda p: p[1])
        ax.scatter([opt[0]], [opt[1]], marker="*", s=260, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        ax.annotate(f"opt $\\lambda_T$={opt[0]:.2f}", (opt[0], opt[1]), xytext=(6, 10),
                    textcoords="offset points", fontsize=8.5, color=c, weight="bold")

    ax.set_xlim(-0.012, 0.118)
    ax.set_ylim(0, max(max(y for _, y in pts) for pts in data.values()) * 1.15)
    ax.set_xlabel(r"disagreement-penalty weight $\lambda_T$")
    ax.set_ylabel(r"live $m_s$ (cross-window mean of peak & typical)")
    ax.set_title("Controller-family specificity: the optimal $\\lambda_T$ is not universal\n"
                 "(★ = per-family optimum: PPO 0.10, HDRL 0.00, MORL 0.00)", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.grid(alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)  [{ {k: [(round(a,2), round(b,3)) for a,b in v] for k,v in data.items()} }]")


if __name__ == "__main__":
    main()
