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

    plt.rcParams.update({"font.size": 11})
    color = {"usable": "#1b7837", "collapse": "#b2182b", "robust": "#2166ac"}
    legend_label = {"v3 (hourly)": "v3 hourly — usable",
                    "matched v3 (15-min)": "matched v3 (15-min) — collapse",
                    "v3.5 (calibrated)": "v3.5 calibrated — collapse",
                    "hybrid (v3 rollout + v3.5 censor)": "hybrid (v3 rollout + v3.5 censor) — robust"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ymax = max(r["m_s_mean"] for r in rows) * 1.18
    # shaded collapse / usable zones + threshold lines (engineering reference levels)
    ax.axhspan(1.0, ymax, color="#b2182b", alpha=0.06, zorder=0)
    ax.axhspan(0.0, 1.0, color="#1b7837", alpha=0.05, zorder=0)
    ax.axhspan(0.0, 0.1, color="#1b7837", alpha=0.16, zorder=0)   # tight "usable" band m_s < 0.1
    ax.axhline(1.0, color="#b2182b", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.1, color="#1b7837", lw=0.8, ls=":", zorder=1)

    # paradox trend through the three single-model surrogates (ordered by RMSE)
    singles = sorted([r for r in rows if r["is_single"]], key=lambda r: r["rmse_24h_c"])
    ax.plot([r["rmse_24h_c"] for r in singles], [r["m_s_mean"] for r in singles],
            "--", color="0.55", lw=1.4, zorder=2)

    from matplotlib.lines import Line2D
    handles = []
    for r in rows:
        c = color[r["verdict"]]
        lo, hi = min(r["m_s_peak"], r["m_s_typ"]), max(r["m_s_peak"], r["m_s_typ"])
        marker = "D" if not r["is_single"] else "o"
        ax.errorbar(r["rmse_24h_c"], r["m_s_mean"], yerr=[[r["m_s_mean"] - lo], [hi - r["m_s_mean"]]],
                    fmt=marker, ms=13, color=c, ecolor=c, elinewidth=1.4, capsize=4, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.0)
        handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=c,
                              markeredgecolor="white", markersize=11, label=legend_label[r["controller"]]))

    # zone / threshold labels (normal, non-inverted axis)
    xlo, xhi = 0.55, 1.70
    ax.text(xlo + 0.02, ymax * 0.97, "collapse  ($m_s>1$)", color="#b2182b", fontsize=10,
            va="top", ha="left", weight="bold")
    ax.text(xhi - 0.02, 1.02, "$m_s=1$ collapse threshold", color="#b2182b", fontsize=8, va="bottom", ha="right")
    ax.text(xhi - 0.02, 0.135, "usable region ($m_s<0.1$)", color="#1b7837", fontsize=9, va="bottom", ha="right", weight="bold")

    # callout for the hybrid: it has no single rollout RMSE -> placed at v3's RMSE
    hy = next(r for r in rows if not r["is_single"])
    ax.annotate("hybrid — plotted at v3 RMSE\n(rollout dynamics are v3)", (hy["rmse_24h_c"], hy["m_s_mean"]),
                textcoords="offset points", xytext=(-8, 30), fontsize=8.5, ha="right",
                color=color["robust"], arrowprops=dict(arrowstyle="->", color=color["robust"], lw=1.2))

    ax.set_xlim(xlo, xhi)  # normal axis: lower RMSE (more accurate) on the left
    ax.set_ylim(-0.02, ymax)
    ax.set_xlabel(r"24-h rollout RMSE$_T$ ($^\circ$C)   (lower = more predictive fidelity)")
    ax.set_ylabel(r"live maintenance score $m_s$   (higher = worse controller)")
    ax.set_title("Fidelity–utility paradox: lower RMSE does not imply lower $m_s$")
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)
    ax.grid(alpha=0.18)
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
