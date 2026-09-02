"""Figure 1 -- quantitative evidence-chain matrix for the fidelity-utility paradox.

NOT a cartoon: every cell is a measured quantity loaded from committed artefacts via
_figstyle.paper_numbers(). Rows are the four training backends in canonical order
(v3 hourly, matched-v3, direct v3.5, hybrid); columns are the four measured links of
the chain that the paper claims:

    predictive fidelity  ->  action-surface roughness  ->  policy saturation  ->  live utility
    (24 h RMSE_T, deg C)     (relative roughness, x v3)    (|a0|>=0.9, %)         (live m_s)

The figure makes the paradox visible as a *measured contradiction*: the RMSE column
orders v3.5 < matched < v3 (v3 least accurate), but the live-m_s column reverses it
(v3 best, both accurate backends collapse past m_s = 1). The hybrid trains on v3's
rollout dynamics, so it inherits v3's fidelity and surface roughness (marked *), while
the frozen v3.5 enters only as a reward censor -- yet it lands in the usable band.

Output: docs/results2_control_overleaf/figures/block2_conceptual_overview.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_conceptual_overview.pdf"

KEYS = ["v3", "matched", "v35", "hybrid"]      # canonical order (PI is not a surrogate backend)
INHERITED = {"hybrid"}                          # hybrid inherits v3's fidelity + surface


def assemble():
    """All four measured columns per backend, from committed artefacts (no hand entry)."""
    n = fs.paper_numbers(ROOT)
    rmse = {"v3": n["rmse"]["v3"], "matched": n["rmse"]["matched"],
            "v35": n["rmse"]["v35"], "hybrid": n["rmse"]["v3"]}            # hybrid rolls out on v3
    rough = {"v3": 1.0, "matched": n["rough_fold"]["matched"],
             "v35": n["rough_fold"]["v35"], "hybrid": 1.0}                 # hybrid sees v3's surface
    sat = {k: n["saturation"][k] for k in KEYS}
    ms = {"v3": n["m_s"]["v3"], "matched": n["m_s"]["matched"],
          "v35": n["m_s"]["v35"], "hybrid": n["m_s"]["hybrid_mean"]}
    return rmse, rough, sat, ms


def main() -> None:
    fs.apply()
    rmse, rough, sat, ms = assemble()
    ypos = {k: len(KEYS) - 1 - i for i, k in enumerate(KEYS)}              # v3 on top

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.3), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.16, wspace=0.28)

    cols = [
        ("rmse",  rmse,  r"24 h rollout RMSE$_T$ ($^{\circ}$C)", "(A) predictive fidelity", "{:.3f}", 1.72, None),
        ("rough", rough, r"relative roughness ($\times$ v3)", "(B) action-surface geometry", r"{:.1f}$\times$", 10.6, None),
        ("sat",   sat,   r"policy saturation $|a_0|\geq0.9$ (\%)", "(C) policy pathology", r"{:.0f}\%", 112, 90.0),
        ("ms",    ms,    r"live maintenance score $m_s$", "(D) live BOPTEST utility", "{:.3f}", 1.34, fs.MS_COLLAPSE),
    ]

    for ax, (key, vals, xlabel, title, fmt, xmax, thr) in zip(axes, cols):
        for k in KEYS:
            y, v = ypos[k], vals[k]
            hatch = "////" if k in INHERITED and key in ("rmse", "rough") else None
            ax.barh(y, v, height=0.62, color=fs.COLOR[k], alpha=0.88,
                    edgecolor="white", linewidth=0.8, hatch=hatch, zorder=2)
            ax.plot(v, y, marker=fs.MARKER[k], color=fs.COLOR[k], ms=8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            star = "*" if (k in INHERITED and key in ("rmse", "rough")) else ""
            ax.annotate(f" {fmt.format(v)}{star}", (v, y), va="center", ha="left",
                        fontsize=8.4, color="0.15",
                        xytext=(3, 0), textcoords="offset points")
        # column-specific engineering references / shading
        if key == "ms":
            ax.axvspan(0, fs.MS_USABLE, color=fs.V3, alpha=0.10, zorder=0)
            fs.threshold(ax, fs.MS_COLLAPSE, "collapse", axis="v", color=fs.ACCURATE, pos=0.97)
            ax.text(fs.MS_USABLE, len(KEYS) - 0.35, " usable", color=fs.V3, fontsize=7.5, va="top")
        elif key == "sat":
            fs.threshold(ax, thr, "near bang-bang", axis="v", color="0.45", ls=":", pos=0.5)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=9.5, weight="bold", color="0.25", pad=6)
        ax.tick_params(length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_yticks([ypos[k] for k in KEYS])
    axes[0].set_yticklabels([fs.LABEL[k] for k in KEYS], fontsize=9.5)
    for tick, k in zip(axes[0].get_yticklabels(), KEYS):
        tick.set_color(fs.COLOR[k]); tick.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(KEYS) - 0.4)

    fig.suptitle("Quantified evidence chain: lower predictive RMSE$_T$ does not imply lower live $m_s$ "
                 r"(RMSE$_T\!\downarrow\;\Rightarrow\!\!\!\!\!/\;\;m_s\!\downarrow$)",
                 fontsize=12.5, weight="bold", y=0.965)
    fig.text(0.5, 0.025,
             r"Reading left$\rightarrow$right is the paradox: the RMSE$_T$ order (v3.5 $<$ matched-v3 $<$ v3) reverses in live $m_s$ "
             r"(v3, hybrid usable; matched-v3, v3.5 collapse).  "
             r"*hybrid inherits v3's rollout fidelity and surface; the frozen v3.5 enters only as a reward censor.",
             ha="center", fontsize=7.8, color="0.4")

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)  [RMSE {rmse} | rough {rough} | sat {sat} | m_s {ms}]")


if __name__ == "__main__":
    main()
