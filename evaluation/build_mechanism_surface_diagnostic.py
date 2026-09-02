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
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"[figure skipped: matplotlib unavailable: {exc}]")
        return
    sys.path.insert(0, str(ROOT / "evaluation"))
    import _figstyle as fs
    fs.apply()

    order = ["v3 hourly (1h)", "v3 matched (15min)", "v3.5 calibrated"]
    KEY = {"v3 hourly (1h)": "v3", "v3 matched (15min)": "matched", "v3.5 calibrated": "v35"}
    style = {"v3": "-", "matched": "--", "v35": "-."}             # colour-blind safety
    tick = {"v3": "BB\nhourly", "matched": "BB\n15 min", "v35": "GB"}
    admap = dict(adapters)
    rr = {r["surrogate"]: r["rel_roughness"] for r in rows}
    base = rr["v3 hourly (1h)"]
    xs = list(range(len(order)))
    col = [fs.COLOR[KEY[n]] for n in order]
    ms = fs.paper_numbers(ROOT)["m_s"]

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.24, wspace=0.42)
    axA, axB, axC, axD = axes

    # (A) measured action->next-temperature shape: per-state curves (thin) + median (thick)
    for name in order:
        k = KEY[name]
        curves = np.array(per_state_curves(admap[name]))        # centred dT (deg C), one per state
        med = np.median(curves, axis=0)
        rng = med.max() - med.min()
        for c in curves:
            axA.plot(A0, c / rng, color=fs.COLOR[k], lw=0.6, alpha=0.16, zorder=1)
        axA.plot(A0, med / rng, color=fs.COLOR[k], ls=style[k], lw=2.4, zorder=3,
                 label=f"{fs.LABEL[k]} ({'1.0' if k == 'v3' else f'{rr[name]/base:.1f}'}$\\times$)")
    axA.axhline(0, color="0.85", lw=0.6, zorder=0)
    axA.set_xlabel(r"supply-temperature action $a_0$", fontsize=10)
    axA.set_ylabel("normalized response (shape)", fontsize=10)
    axA.set_title(r"(A) Action$\rightarrow$next-T response shape", fontsize=10, weight="bold")
    axA.tick_params(labelsize=9); axA.grid(alpha=0.18)

    # (B) scale-free relative-roughness DISTRIBUTION over the state grid (box + jittered points)
    data = [per_state_rel(admap[n]) for n in order]
    bp = axB.boxplot(data, positions=xs, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="0.15", lw=1.3))
    for patch, c in zip(bp["boxes"], col):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    rng_state = np.random.default_rng(0)
    for x, d, c, n in zip(xs, data, col, order):
        axB.scatter(x + rng_state.uniform(-0.12, 0.12, len(d)), d, s=14, color=c,
                    edgecolor="white", linewidth=0.4, zorder=3)
        axB.text(x, max(d) * 1.04 + 0.05, f"$\\times${rr[n]/base:.1f}", ha="center", va="bottom",
                 fontsize=10, weight="bold", color=c)
    axB.set_xticks(xs); axB.set_xticklabels([tick[KEY[n]] for n in order], fontsize=9.5)
    axB.set_ylabel(r"rel. roughness $\overline{|\partial^2\hat T|}/\overline{|\partial\hat T|}$", fontsize=9.5)
    axB.set_title("(B) Roughness over states", fontsize=10, weight="bold")
    axB.set_ylim(0, max(max(d) for d in data) * 1.18); axB.grid(axis="y", alpha=0.2)

    # (C) measured policy-side symptom: closed-loop action saturation (|a0|>0.9)
    sat = [_saturation_pct(TRACE_DIRS[n]) for n in order]
    axC.bar(xs, sat, color=col, edgecolor="0.3", linewidth=0.6, width=0.62)
    for x, s in zip(xs, sat):
        axC.text(x, s + 1.5, rf"{s:.0f}\%", ha="center", va="bottom", fontsize=10, weight="bold")
    fs.threshold(axC, 90, "near bang-bang", axis="h", color="0.45", ls=":", pos=0.5, fontsize=9.5)
    axC.set_xticks(xs); axC.set_xticklabels([tick[KEY[n]] for n in order], fontsize=9.5)
    axC.set_ylabel(r"action saturation ($|a_0|>0.9$, \% steps)", fontsize=9.5)
    axC.set_title("(C) Policy saturation (live)", fontsize=10, weight="bold")
    axC.set_ylim(0, 112); axC.grid(axis="y", alpha=0.2)

    # (D) live BOPTEST outcome: maintenance score m_s with the m_s=1 collapse threshold
    msv = [ms["v3"], ms["matched"], ms["v35"]]
    axD.bar(xs, msv, color=col, edgecolor="0.3", linewidth=0.6, width=0.62)
    for x, v in zip(xs, msv):
        axD.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")
    axD.axhspan(0, fs.MS_USABLE, color=fs.V3, alpha=0.12, zorder=0)
    fs.threshold(axD, fs.MS_COLLAPSE, "$m_s=1$ collapse", axis="h", color=fs.ACCURATE, pos=0.97, fontsize=9.5)
    axD.set_xticks(xs); axD.set_xticklabels([tick[KEY[n]] for n in order], fontsize=9.5)
    axD.set_ylabel("live maintenance score $m_s$", fontsize=9.5)
    axD.set_title("(D) Live BOPTEST outcome", fontsize=10, weight="bold")
    axD.set_ylim(0, max(msv) * 1.22); axD.grid(axis="y", alpha=0.2)

    # shared legend (panel-A response curves) at the figure bottom
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.6)

    fig.suptitle(r"Measured roughness mechanism: rougher action surface $\rightarrow$ policy saturation $\rightarrow$ live collapse "
                 r"(association, not a proven causal law)", fontsize=12, weight="bold", y=0.99)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(FIG_OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT} (+ .png)  [sat %: " + ", ".join(f"{KEY[n]} {s:.0f}" for n, s in zip(order, sat)) +
          " | m_s: " + ", ".join(f"{KEY[n]} {v:.2f}" for n, v in zip(order, msv)) + "]")


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
