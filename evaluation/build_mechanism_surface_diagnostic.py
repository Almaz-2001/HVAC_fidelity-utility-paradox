"""Mechanism diagnostic: measure the response-surface sharpness of each surrogate.

The paper hypothesises that the fidelity-utility paradox is driven by the *sharper*
response surface of the more accurate surrogates: a policy-gradient learner exploits
that sharpness into a near-bang-bang law that does not transfer. The action-gap
(g_a ~ 2.0) already measures the resulting *policy* behaviour; this script measures
the *cause* directly on the surrogates themselves, with no RL and no BOPTEST.

For each surrogate we hold a representative state (zone temp, ambient, time) fixed,
sweep the supply-temperature action a0 over [-1, 1], and record the predicted next
zone temperature T_hat(a0). From that 1-D response curve we compute, averaged over a
grid of states:

  * sensitivity   = mean |dT_hat/da0|              (slope magnitude, C per unit action)
  * roughness     = mean |d2 T_hat/da0^2|          (curvature magnitude)
  * rel_roughness = roughness / sensitivity        (SCALE-FREE non-smoothness)
  * range         = max T_hat - min T_hat over the sweep

The raw slope/range are confounded by the step duration (the hourly v3 is a 1-h step,
the others 15-min), which is exactly the coarse-graining variable -- so they are *not*
a clean measure of the cause. The scale-free `rel_roughness` (curvature normalised by
slope, both in C/action) is independent of step length and of response magnitude: it
measures how non-smooth / non-monotone the action -> next-temperature map is, which is
precisely what a policy-gradient learner exploits into a bang-bang law.

Prediction (mechanism): the usable hourly v3 has the *lowest* rel_roughness (a smooth,
near-monotone, optimisation-friendly landscape), while both surrogates that collapse as
training environments (matched-resolution v3, calibrated v3.5) expose *rougher*
landscapes -- the measured counterpart of the action-gap policy symptom (g_a ~ 2.0).

Output: reports/block2_mechanism_surface_sharpness.csv  (+ console digest).
"""

from __future__ import annotations
import csv
import sys
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter
V35_SUMMARY = "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json"

SURROGATES = [
    ("v3 hourly (1h)", dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v2/rc_node_v3_tsupply.pt")),
    ("v3 matched (15min)", dict(kind="legacy_v3", legacy_model_path="outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt")),
    ("v3.5 calibrated", dict(kind="v35_calibrated", summary_json=V35_SUMMARY)),
]

# Representative state grid: comfort-band zone temps x the two windows' daily-mean
# ambients; mid-day, mid-month, fan off. a0 is the swept control action.
T_ZONE = [19.0, 21.0, 22.0, 23.0, 25.0]
T_AMB = [-24.4, 2.4]          # peak / typical window daily-mean ambient
HOUR, DAY, A1 = 12.0, 15.0, 0.0
A0 = np.linspace(-1.0, 1.0, 41)


def curve(adapter, tz, ta):
    return np.array([adapter.step_numpy(tz, ta, HOUR, DAY, float(a), A1)[0] for a in A0])


def metrics(adapter):
    sens, rough, rng = [], [], []
    da = A0[1] - A0[0]
    for tz in T_ZONE:
        for ta in T_AMB:
            y = curve(adapter, tz, ta)
            sens.append(float(np.mean(np.abs(np.gradient(y, da)))))
            rough.append(float(np.mean(np.abs(np.diff(y, 2) / (da * da)))))
            rng.append(float(y.max() - y.min()))
    return st.mean(sens), st.mean(rough), st.mean(rng)


FIG_OUT = ROOT / "docs/results2_control_overleaf/figures/block2_surface_response_curves.pdf"
# Representative state for the Panel-A response curves (typical-window ambient, mid comfort band).
REP_TZ, REP_TA = 22.0, 2.4


def per_state_curves(adapter):
    """Centred response curve dT_hat(a0) in deg C for every state in the grid."""
    return [curve(adapter, tz, ta) - curve(adapter, tz, ta).mean()
            for tz in T_ZONE for ta in T_AMB]


# Closed-loop traces (peak window) for the three single-model surrogate controllers.
TRACE_DIRS = {"v3 hourly (1h)": "bestest_air_article7_style_15min",
              "v3 matched (15min)": "bestest_air_pure_v3_15min",
              "v3.5 calibrated": "block2_bestest_air_15min_thermostatic_v35"}


def per_state_rel(adapter):
    """Per-state relative roughness (rough_s / sens_s) across the grid."""
    da = A0[1] - A0[0]
    out = []
    for tz in T_ZONE:
        for ta in T_AMB:
            y = curve(adapter, tz, ta)
            s = float(np.mean(np.abs(np.gradient(y, da))))
            r = float(np.mean(np.abs(np.diff(y, 2) / (da * da))))
            out.append(r / s if s else np.nan)
    return np.array(out)


def _saturation_pct(run_dir):
    import pandas as pd
    a = pd.read_csv(ROOT / "outputs" / run_dir / "traces" / "peak_heat_window_thermostatic.csv")["a0"]
    return 100.0 * float((a.abs() > 0.9).mean())


def make_figure(adapters: list[tuple[str, object]], rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"[figure skipped: matplotlib unavailable: {exc}]")
        return

    order = ["v3 hourly (1h)", "v3 matched (15min)", "v3.5 calibrated"]
    short = {"v3 hourly (1h)": "v3 hourly (usable)",
             "v3 matched (15min)": "matched-resolution v3 (collapse)",
             "v3.5 calibrated": "calibrated v3.5 (collapse)"}
    tick = {"v3 hourly (1h)": "v3\nhourly", "v3 matched (15min)": "matched\nv3", "v3.5 calibrated": "v3.5"}
    colors = {"v3 hourly (1h)": "#1b7837", "v3 matched (15min)": "#d6604d", "v3.5 calibrated": "#b2182b"}
    admap = dict(adapters)
    rr = {r["surrogate"]: r["rel_roughness"] for r in rows}
    base = rr["v3 hourly (1h)"]
    xs = list(range(len(order)))

    fig = plt.figure(figsize=(10.6, 6.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.7, wspace=0.42)

    # (A) measured response over the state grid (faint = per state, bold = mean)
    last = None
    for i, name in enumerate(order):
        ax = fig.add_subplot(gs[i, 0]); last = ax
        cs = per_state_curves(admap[name])
        for c in cs:
            ax.plot(A0, c, color=colors[name], lw=0.7, alpha=0.28, zorder=1)
        ax.plot(A0, np.mean(cs, axis=0), color=colors[name], lw=2.4, zorder=3)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(f"{short[name]} — rel. roughness {rr[name]:.2f} "
                     f"(${'1.0' if name == order[0] else f'{rr[name]/base:.1f}'}\\times$)",
                     fontsize=8.4, color=colors[name])
        ax.set_ylabel(r"$\Delta\hat T$ ($^\circ$C)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        if i < 2:
            ax.set_xticklabels([])
    last.set_xlabel(r"supply-temperature action $a_0$", fontsize=9.5)
    fig.text(0.285, 0.995, "(A) Measured action$\\rightarrow$next-temperature response over a grid of states",
             fontsize=9, ha="center", va="top", weight="bold")

    # (B) canonical relative roughness (matches the main table) with per-state spread
    axB = fig.add_subplot(gs[0, 1])
    vals = [rr[n] for n in order]
    errs = [float(np.nanstd(per_state_rel(admap[n]))) for n in order]
    axB.bar(xs, vals, yerr=errs, capsize=3, color=[colors[n] for n in order], edgecolor="0.3", linewidth=0.6,
            error_kw=dict(elinewidth=0.9, ecolor="0.35"))
    for x, v, e in zip(xs, vals, errs):
        axB.text(x, v + e + 0.05, f"$\\times${v/base:.1f}", ha="center", va="bottom", fontsize=8.5, weight="bold")
    axB.set_xticks(xs); axB.set_xticklabels([tick[n] for n in order], fontsize=8)
    axB.set_ylabel(r"rel. roughness $\overline{|\partial^2\hat T|}/\overline{|\partial\hat T|}$", fontsize=8.5)
    axB.set_title("(B) Scale-free non-smoothness\n(bar = canonical; whisker = per-state std)", fontsize=8.6, weight="bold")
    axB.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.2)
    axB.grid(axis="y", alpha=0.2)

    # (C) measured policy-side consequence: closed-loop action saturation (|a0|>0.9)
    axC = fig.add_subplot(gs[1, 1])
    sat = [_saturation_pct(TRACE_DIRS[n]) for n in order]
    axC.bar(xs, sat, color=[colors[n] for n in order], edgecolor="0.3", linewidth=0.6)
    for x, s in zip(xs, sat):
        axC.text(x, s + 1.5, f"{s:.0f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")
    axC.set_xticks(xs); axC.set_xticklabels([tick[n] for n in order], fontsize=8)
    axC.set_ylabel("action saturation\n($|a_0|>0.9$, % steps)", fontsize=8.5)
    axC.set_title("(C) Live policy bang-bang\n(measured on BOPTEST)", fontsize=8.6, weight="bold")
    axC.set_ylim(0, 112); axC.grid(axis="y", alpha=0.2)

    # (D) cause vs effect: surrogate roughness vs live controller score
    axD = fig.add_subplot(gs[2, 1])
    sc = pd.read_csv(ROOT / "reports/block2_fidelity_utility_scatter.csv")
    ms = {c: float(sc[sc.controller.str.startswith(c)].iloc[0]["m_s_mean"])
          for c in ["v3 (", "matched", "v3.5", "hybrid"]}
    pts = [(rr["v3 hourly (1h)"], ms["v3 ("], colors["v3 hourly (1h)"], "o", "v3"),
           (rr["v3 matched (15min)"], ms["matched"], colors["v3 matched (15min)"], "o", "matched v3"),
           (rr["v3.5 calibrated"], ms["v3.5"], colors["v3.5 calibrated"], "o", "v3.5"),
           (base, ms["hybrid"], "#2166ac", "D", "hybrid")]
    axD.axhline(1.0, color="0.7", lw=0.8, ls=":")
    for x, y, c, m, lab in pts:
        axD.scatter(x, y, s=70, color=c, marker=m, edgecolor="white", linewidth=0.8, zorder=3)
    # v3 and hybrid sit at the same point (hybrid rolls out on v3) -> one shared label
    axD.annotate("v3, hybrid\n(usable)", (base, ms["hybrid"]), xytext=(12, 2), textcoords="offset points",
                 fontsize=7.2, color="0.25", ha="left", va="center")
    axD.annotate("matched v3", (rr["v3 matched (15min)"], ms["matched"]), xytext=(-6, 6),
                 textcoords="offset points", fontsize=7.2, color=colors["v3 matched (15min)"], ha="right", weight="bold")
    axD.annotate("v3.5", (rr["v3.5 calibrated"], ms["v3.5"]), xytext=(-6, -13),
                 textcoords="offset points", fontsize=7.2, color=colors["v3.5 calibrated"], ha="right", weight="bold")
    axD.set_xlim(-0.05, max(rr.values()) * 1.30)
    axD.set_ylim(-0.1, 1.45)
    axD.set_xlabel("surrogate rel. roughness", fontsize=8.5)
    axD.set_ylabel("live $m_s$", fontsize=8.5)
    axD.set_title("(D) Rougher surrogate $\\rightarrow$ worse\ncontroller (measured)", fontsize=8.6, weight="bold")
    axD.grid(alpha=0.2)

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT}  [saturation %: " + ", ".join(f"{n.split()[0]} {s:.0f}" for n, s in zip(order, sat)) + "]")


def main() -> None:
    rows = []
    _adapters: list[tuple[str, object]] = []
    print(f"{'surrogate':22s} {'sensitivity':>12s} {'roughness':>11s} {'rel_rough':>10s} {'range(C)':>9s}")
    for name, kw in SURROGATES:
        try:
            ad = load_direct_tsup_adapter(**kw)
        except Exception as exc:
            print(f"{name:22s} LOAD ERROR: {exc}")
            continue
        s, r, g = metrics(ad)
        rel = r / s if s else float("nan")
        rows.append({"surrogate": name, "sensitivity_C_per_action": round(s, 4),
                     "roughness_C_per_action2": round(r, 4), "rel_roughness": round(rel, 4),
                     "response_range_C": round(g, 4)})
        _adapters.append((name, ad))
        print(f"{name:22s} {s:12.4f} {r:11.4f} {rel:10.4f} {g:9.3f}")

    if rows:
        out = ROOT / "reports/block2_mechanism_surface_sharpness.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        v3 = next((x for x in rows if x["surrogate"].startswith("v3 hourly")), None)
        others = [x for x in rows if x is not v3]
        if v3 and others:
            smoothest = all(v3["rel_roughness"] < x["rel_roughness"] for x in others)
            ratios = ", ".join(f"{x['surrogate']} {x['rel_roughness']/v3['rel_roughness']:.1f}x" for x in others)
            print("\nVERDICT:", f"hourly v3 has the SMOOTHEST action->next-T landscape "
                  f"(rel_roughness {v3['rel_roughness']}); the surrogates that collapse as training "
                  f"environments are rougher ({ratios}) -> measured mechanism for the paradox."
                  if smoothest else
                  "hourly v3 is NOT the smoothest by rel_roughness -> report measured values as-is, do not overclaim.")
        print(f"Wrote {out}")
        make_figure(_adapters, rows)


if __name__ == "__main__":
    main()
