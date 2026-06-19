"""Figure 5 -- fidelity-utility paradox scatter, two panels.

(A) Single-model backends only: across v3 -> matched-v3 -> v3.5 a *lower* 24 h rollout
    RMSE_T (a more accurate twin) goes with a *higher* live maintenance score m_s
    (a worse controller). The accurate backends sit above the m_s = 1 collapse line.
(B) The role-separated hybrid breaks the trade-off: it is plotted at v3's rollout RMSE
    because v3 supplies the transition dynamics, while the frozen v3.5 enters only as a
    per-step reward censor -- yet it lands in the usable band with v3.

Peak and typical heat windows are shown as separate markers (filled = peak, open =
typical), never a single vertical segment. Where N = 3 seeds are available (v3,
matched-v3, hybrid) the bars are +/- one seed s.d. from reports/block2_thermostatic_seed_band.csv
(direct v3.5 is single-seed and is marked). All numbers come from committed artefacts.

Outputs: docs/results2_control_overleaf/figures/block2_fidelity_utility_scatter.pdf
         reports/block2_fidelity_utility_scatter.csv (provenance)
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_fidelity_utility_scatter.pdf"
CSV_OUT = ROOT / "reports/block2_fidelity_utility_scatter.csv"


def _rd(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def collect() -> list[dict]:
    matched = _rd("reports/block1_corpus_matched_comparison.csv").set_index("variant")
    arch = _rd("reports/hou_evins_architecture_justification_table.csv").set_index("variant")
    cl = _rd("reports/block2_v3_15min_closed_loop_comparison.csv")
    hyb = _rd("outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv")
    sb = _rd("reports/block2_thermostatic_seed_band.csv")

    def cl_ms(variant, window):
        return float(cl[(cl.variant == variant) & (cl.window == window)].iloc[0]["m_s"])

    def hyb_ms(scenario):
        return float(hyb[hyb.scenario == scenario].iloc[0]["m_s"])

    def std(label, window):  # seed s.d. where N=3 seeds exist, else None
        m = sb[(sb.controller == label) & (sb.window == window)]
        return float(m.iloc[0]["m_s_std"]) if not m.empty else None

    v35 = arch.loc["v35_calibrated"]
    rows = [
        {"key": "v3", "controller": "v3 (hourly)", "rmse_24h_c": float(matched.loc["v3_hourly"]["rmse_24h_c"]),
         "m_s_peak": cl_ms("pure_v3_hourly", "peak_heat_window"), "m_s_typ": cl_ms("pure_v3_hourly", "typical_heat_window"),
         "std_peak": std("pure v3", "peak"), "std_typ": std("pure v3", "typical"), "is_single": True},
        {"key": "matched", "controller": "matched v3 (15-min)", "rmse_24h_c": float(matched.loc["v3_15min_matched"]["rmse_24h_c"]),
         "m_s_peak": cl_ms("pure_v3_15min", "peak_heat_window"), "m_s_typ": cl_ms("pure_v3_15min", "typical_heat_window"),
         "std_peak": std("matched v3 (15-min)", "peak"), "std_typ": std("matched v3 (15-min)", "typical"), "is_single": True},
        {"key": "v35", "controller": "v3.5 (calibrated)", "rmse_24h_c": float(v35["block1_rollout_24h_rmse_c"]),
         "m_s_peak": float(v35["peak_control_m_s"]), "m_s_typ": float(v35["typical_control_m_s"]),
         "std_peak": None, "std_typ": None, "is_single": True},
        {"key": "hybrid", "controller": "hybrid (v3 rollout + v3.5 censor)", "rmse_24h_c": float(matched.loc["v3_hourly"]["rmse_24h_c"]),
         "m_s_peak": hyb_ms("peak_heat_window"), "m_s_typ": hyb_ms("typical_heat_window"),
         "std_peak": std("hybrid (lambda_T=0.10)", "peak"), "std_typ": std("hybrid (lambda_T=0.10)", "typical"), "is_single": False},
    ]
    for r in rows:
        r["m_s_mean"] = (r["m_s_peak"] + r["m_s_typ"]) / 2.0
    return rows


def _draw_point(ax, r, *, ghost=False):
    """Peak (filled) and typical (open) markers for one model, with seed error bars."""
    c = "0.78" if ghost else fs.COLOR[r["key"]]
    mk = fs.MARKER[r["key"]]
    z = 2 if ghost else 4
    for window, mfc, std in [("peak", c, r["std_peak"]), ("typical", "white", r["std_typ"])]:
        y = r["m_s_peak"] if window == "peak" else r["m_s_typ"]
        ax.errorbar(r["rmse_24h_c"], y, yerr=(std if std else None), fmt=mk, ms=11,
                    color=c, mfc=(c if window == "peak" else "white"), mec=c, mew=1.4,
                    ecolor=c, elinewidth=1.3, capsize=3, zorder=z)


def _zones(ax, ymax):
    ax.axhspan(0, fs.MS_USABLE, color=fs.V3, alpha=0.14, zorder=0)
    ax.axhspan(fs.MS_COLLAPSE, ymax, color=fs.ACCURATE, alpha=0.06, zorder=0)
    fs.threshold(ax, fs.MS_COLLAPSE, "$m_s=1$ collapse", color=fs.ACCURATE, pos=0.985)
    fs.threshold(ax, fs.MS_USABLE, "usable $m_s<0.1$", color=fs.V3, ls=":", pos=0.985)


def make_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fs.apply()

    byk = {r["key"]: r for r in rows}
    ymax = max(max(r["m_s_peak"], r["m_s_typ"]) for r in rows) * 1.15
    xlo, xhi = 0.50, 1.75
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)

    # ---- Panel A: single-model paradox ----
    singles = sorted([byk[k] for k in ("v3", "matched", "v35")], key=lambda r: r["rmse_24h_c"])
    axA.plot([r["rmse_24h_c"] for r in singles], [r["m_s_mean"] for r in singles],
             "--", color="0.55", lw=1.3, zorder=1)
    for r in singles:
        _draw_point(axA, r)
    _zones(axA, ymax)
    axA.set_title("(A) Single-model paradox: more accurate → worse live $m_s$", fontsize=10.5, weight="bold")
    axA.annotate("more accurate twin", (0.62, ymax * 0.5), fontsize=8.5, color="0.4",
                 ha="left", rotation=0)
    axA.annotate("", xy=(0.55, ymax * 0.42), xytext=(0.95, ymax * 0.42),
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=1.1))

    # ---- Panel B: hybrid breaks the trade-off ----
    for k in ("matched", "v35"):              # ghost the collapsed single-models for context
        _draw_point(axB, byk[k], ghost=True)
    _draw_point(axB, byk["v3"])
    _draw_point(axB, byk["hybrid"])
    _zones(axB, ymax)
    hy = byk["hybrid"]
    axB.annotate("hybrid: plotted at v3's RMSE\n(v3 supplies rollout dynamics;\nfrozen v3.5 = reward censor only)",
                 (hy["rmse_24h_c"], hy["m_s_mean"]), xytext=(-10, 60), textcoords="offset points",
                 fontsize=8, ha="right", color=fs.HYBRID,
                 arrowprops=dict(arrowstyle="->", color=fs.HYBRID, lw=1.2))
    axB.set_title("(B) Role-separated hybrid recovers usable control", fontsize=10.5, weight="bold")

    for ax in (axA, axB):
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-0.03, ymax)
        ax.set_xlabel(r"24-h rollout RMSE$_T$ (°C)   (← lower = more predictive fidelity)")
        ax.grid(alpha=0.16)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axA.set_ylabel(r"live maintenance score $m_s$   (↑ worse controller)")

    # shared legends: model identity (colour+marker) and window encoding
    mh = fs.legend_handles(["v3", "matched", "v35", "hybrid"], with_role=True, markersize=9)
    wh = [Line2D([0], [0], lw=0, marker="o", color="0.3", mfc="0.3", mec="0.3", ms=9, label="peak window"),
          Line2D([0], [0], lw=0, marker="o", color="0.3", mfc="white", mec="0.3", ms=9, label="typical window")]
    axA.legend(handles=mh, loc="lower left", fontsize=8, frameon=True, framealpha=0.92)
    axB.legend(handles=wh, loc="lower left", fontsize=8, frameon=True, framealpha=0.92,
               title="markers", title_fontsize=8)

    fig.suptitle("Fidelity–utility paradox: lower RMSE$_T$ does not imply lower live $m_s$",
                 fontsize=12.5, weight="bold", y=0.99)
    fig.text(0.5, 0.005, "Error bars = ±1 s.d. over N=3 seeds (v3, matched-v3, hybrid; "
             "reports/block2_thermostatic_seed_band.csv); direct v3.5 is single-seed.",
             ha="center", fontsize=7.6, color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)")


def main() -> None:
    rows = collect()
    print(f"{'controller':36s} {'RMSE':>6s} {'m_s_pk':>7s} {'m_s_typ':>8s}")
    for r in rows:
        print(f"{r['controller']:36s} {r['rmse_24h_c']:6.3f} {r['m_s_peak']:7.3f} {r['m_s_typ']:8.3f}")
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["controller", "rmse_24h_c", "m_s_peak", "m_s_typ", "m_s_mean",
                                           "std_peak", "std_typ", "is_single"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"Wrote {CSV_OUT}")
    make_figure(rows)


if __name__ == "__main__":
    main()
