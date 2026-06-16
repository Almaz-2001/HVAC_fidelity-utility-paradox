"""Fidelity-utility paradox scatter: 24h rollout RMSE (x) vs live maintenance score (y).

This is the single clearest view of the paradox: across the three single-model
surrogates, a *lower* predictive RMSE (a more accurate twin) goes with a *higher* live
maintenance score m_s (a worse controller). The role-separating hybrid breaks the
trade-off: it rolls out on v3 (so its environment fidelity equals v3's, RMSE 1.557 C)
yet the frozen v3.5 censor pulls m_s down to the usable regime.

All numbers are read from the committed artifacts (no hand-entry):
  * 24h rollout RMSE  -> reports/block1_corpus_matched_comparison.csv (v3, matched v3)
                         reports/hou_evins_architecture_justification_table.csv (v3.5)
  * live m_s (peak/typ) -> reports/block2_v3_15min_closed_loop_comparison.csv (v3, matched v3)
                           reports/hou_evins_architecture_justification_table.csv (v3.5)
                           outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv (hybrid)

Outputs: docs/results2_control_overleaf/figures/block2_fidelity_utility_scatter.pdf
         reports/block2_fidelity_utility_scatter.csv (provenance)
"""

from __future__ import annotations
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_fidelity_utility_scatter.pdf"
CSV_OUT = ROOT / "reports/block2_fidelity_utility_scatter.csv"


def _rd(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def collect() -> list[dict]:
    matched = _rd("reports/block1_corpus_matched_comparison.csv").set_index("variant")
    arch = _rd("reports/hou_evins_architecture_justification_table.csv").set_index("variant")
    cl = _rd("reports/block2_v3_15min_closed_loop_comparison.csv")
    hyb = _rd("outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv")

    def cl_ms(variant, window):
        return float(cl[(cl.variant == variant) & (cl.window == window)].iloc[0]["m_s"])

    def hyb_ms(scenario):
        return float(hyb[hyb.scenario == scenario].iloc[0]["m_s"])

    v35 = arch.loc["v35_calibrated"]
    rows = [
        {"controller": "v3 (hourly)", "rmse_24h_c": float(matched.loc["v3_hourly"]["rmse_24h_c"]),
         "m_s_peak": cl_ms("pure_v3_hourly", "peak_heat_window"),
         "m_s_typ": cl_ms("pure_v3_hourly", "typical_heat_window"),
         "verdict": "usable", "is_single": True},
        {"controller": "matched v3 (15-min)", "rmse_24h_c": float(matched.loc["v3_15min_matched"]["rmse_24h_c"]),
         "m_s_peak": cl_ms("pure_v3_15min", "peak_heat_window"),
         "m_s_typ": cl_ms("pure_v3_15min", "typical_heat_window"),
         "verdict": "collapse", "is_single": True},
        {"controller": "v3.5 (calibrated)", "rmse_24h_c": float(v35["block1_rollout_24h_rmse_c"]),
         "m_s_peak": float(v35["peak_control_m_s"]), "m_s_typ": float(v35["typical_control_m_s"]),
         "verdict": "collapse", "is_single": True},
        # hybrid has no single rollout RMSE: it rolls out on v3, so it is placed at v3's RMSE.
        {"controller": "hybrid (v3 rollout + v3.5 censor)", "rmse_24h_c": float(matched.loc["v3_hourly"]["rmse_24h_c"]),
         "m_s_peak": hyb_ms("peak_heat_window"), "m_s_typ": hyb_ms("typical_heat_window"),
         "verdict": "robust", "is_single": False},
    ]
    for r in rows:
        r["m_s_mean"] = (r["m_s_peak"] + r["m_s_typ"]) / 2.0
    return rows


def make_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"usable": "#1b7837", "collapse": "#b2182b", "robust": "#2166ac"}
    fig, ax = plt.subplots(figsize=(6.4, 4.7))

    # paradox trend through the three single-model surrogates (ordered by RMSE)
    singles = sorted([r for r in rows if r["is_single"]], key=lambda r: r["rmse_24h_c"])
    ax.plot([r["rmse_24h_c"] for r in singles], [r["m_s_mean"] for r in singles],
            "--", color="0.5", lw=1.2, zorder=1)

    for r in rows:
        c = color[r["verdict"]]
        lo, hi = min(r["m_s_peak"], r["m_s_typ"]), max(r["m_s_peak"], r["m_s_typ"])
        marker = "D" if not r["is_single"] else "o"
        ax.errorbar(r["rmse_24h_c"], r["m_s_mean"], yerr=[[r["m_s_mean"] - lo], [hi - r["m_s_mean"]]],
                    fmt=marker, ms=11, color=c, ecolor=c, elinewidth=1.3, capsize=4, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)

    # usable / collapse divider (m_s = 1 is a clear failure; usable controllers are well below)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":", zorder=0)
    ax.text(ax.get_xlim()[1], 1.0, " collapse (m_s>1) ", va="bottom", ha="right", color="0.45", fontsize=7)

    # annotations
    ann = {"v3 (hourly)": (8, 10), "matched v3 (15-min)": (8, 8), "v3.5 (calibrated)": (-8, 10),
           "hybrid (v3 rollout + v3.5 censor)": (-10, -22)}
    for r in rows:
        dx, dy = ann.get(r["controller"], (6, 6))
        ax.annotate(r["controller"], (r["rmse_24h_c"], r["m_s_mean"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, ha="left" if dx >= 0 else "right",
                    color=color[r["verdict"]])

    ax.set_xlabel(r"surrogate predictive fidelity: 24 h rollout RMSE ($^\circ$C)  $\rightarrow$ more accurate")
    ax.set_ylabel(r"downstream control utility: live $m_s$  $\rightarrow$ worse")
    ax.set_title("Fidelity–utility paradox: lower RMSE does not imply lower $m_s$")
    ax.invert_xaxis()  # so "more accurate" is to the right, matching the arrow
    ax.grid(alpha=0.2)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT}")


def main() -> None:
    rows = collect()
    print(f"{'controller':36s} {'RMSE':>6s} {'m_s_mean':>9s} {'verdict':>9s}")
    for r in rows:
        print(f"{r['controller']:36s} {r['rmse_24h_c']:6.3f} {r['m_s_mean']:9.3f} {r['verdict']:>9s}")
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["controller", "rmse_24h_c", "m_s_peak", "m_s_typ", "m_s_mean", "verdict", "is_single"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {CSV_OUT}")
    make_figure(rows)


if __name__ == "__main__":
    main()
