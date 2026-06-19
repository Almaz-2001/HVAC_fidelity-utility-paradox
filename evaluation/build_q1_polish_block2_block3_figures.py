"""Build Q1-polished Block 2/3 figures from existing project artifacts.

The script implements the review-style visualization recommendations without
fabricating uncertainty. It uses existing closed-loop traces, MORL per-seed
tables, and the Block 3 transfer matrix.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "article_real"

import sys as _sys
_sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

BLUE = "#2f5d8c"
TEAL = "#21867a"
ORANGE = "#b25f2c"
PURPLE = "#6f4e7c"
RED = "#c44e52"
GREY = "#5c6470"
SKY = "#4a9ecf"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def style(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


TRACE_PATHS = {
    "pure v3": "outputs/bestest_air_article7_style_15min/traces/typical_heat_window_thermostatic.csv",
    "direct v3.5": "outputs/block2_bestest_air_15min_thermostatic_v35/traces/typical_heat_window_thermostatic.csv",
    "hybrid": "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/typical_heat_window_thermostatic.csv",
}

# unified paper colour scheme: v3 = green (usable), v3.5 = red (collapse), hybrid = blue
TRACE_COLORS = {"pure v3": "#1b7837", "direct v3.5": "#b2182b", "hybrid": "#2166ac"}


def fig_block2_closed_loop_disturbance() -> None:
    """Closed-loop trace with ambient disturbance, comfort band and actuator limits."""
    traces = {name: read(path).iloc[: 96 * 3].copy() for name, path in TRACE_PATHS.items()}
    base = next(iter(traces.values()))
    t = (base["sim_time_sec"] - base["sim_time_sec"].iloc[0]) / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True)

    axes[0].plot(t, base["t_amb_c"], color=SKY, linewidth=1.8)
    style(axes[0], "(a) Ambient disturbance available to the 17D interface", ylabel="$T_{amb}$ (degC)")

    for name, df in traces.items():
        axes[1].plot(t, df["t_zone_c"], label=name, color=TRACE_COLORS[name], linewidth=1.5)
    axes[1].axhspan(21, 24, color="#1b7837", alpha=0.07, label="comfort band 21-24 degC")
    style(axes[1], "(b) Zone temperature response", ylabel="$T_{zone}$ (degC)")
    axes[1].legend(ncol=4, frameon=False, fontsize=10)

    for name, df in traces.items():
        axes[2].plot(t, df["t_supply_cmd_c"], label=name, color=TRACE_COLORS[name], linewidth=1.3)
    axes[2].axhspan(18, 35, color="#f2f2f2", alpha=0.7, label="actuator range 18-35 degC")
    axes[2].axhline(18, color="#999999", linestyle="--", linewidth=0.8)
    axes[2].axhline(35, color="#999999", linestyle="--", linewidth=0.8)
    style(axes[2], "(c) Supply-temperature command and physical actuator limits", ylabel="$T_{sup}$ command (degC)")

    for name, df in traces.items():
        axes[3].plot(t, df["p_total_w"] / 1000.0, label=name, color=TRACE_COLORS[name], linewidth=1.2)
    style(axes[3], "(d) HVAC power", xlabel="Time since start of typical window (h)", ylabel="Power (kW)")

    fig.suptitle("Block 2 closed-loop traces with physical bounds and ambient disturbance", fontsize=14, weight="bold")
    save(fig, "block2_q1_polish_closed_loop_disturbance")


def fig_block2_phase_density() -> None:
    """Action-density phase portrait (hexbin) of policy action vs thermal error,
    with a marginal action histogram and saturation share, identical axes per backend."""
    from matplotlib.colors import LinearSegmentedColormap
    keys = {"pure v3": "v3", "direct v3.5": "v35", "hybrid": "hybrid"}
    T_SET = 22.5
    xlim, ylim = (-4.5, 4.5), (-1.08, 1.08)
    fig = plt.figure(figsize=(12.4, 4.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[4, 1, 4, 1, 4, 1], wspace=0.10)
    for i, (name, rel) in enumerate(TRACE_PATHS.items()):
        k = keys[name]; c = fs.COLOR[k]
        df = read(rel)
        err = df["t_zone_c"].astype(float) - T_SET
        act = df["a0"].astype(float)
        cmap = LinearSegmentedColormap.from_list("", ["#ffffff", c])
        axh = fig.add_subplot(gs[0, 2 * i])
        axm = fig.add_subplot(gs[0, 2 * i + 1], sharey=axh)
        axh.hexbin(err, act, gridsize=34, cmap=cmap, mincnt=1, linewidths=0, extent=(*xlim, *ylim))
        axh.axhspan(0.9, ylim[1], color=fs.ACCURATE, alpha=0.09, zorder=0)        # saturation bands
        axh.axhspan(ylim[0], -0.9, color=fs.ACCURATE, alpha=0.09, zorder=0)
        for yb in (0.9, -0.9):
            axh.axhline(yb, color=fs.ACCURATE, ls=":", lw=0.8)
        axh.axhline(0, color="0.6", lw=0.7); axh.axvline(0, color="0.6", lw=0.7)
        axh.set_xlim(*xlim); axh.set_ylim(*ylim)
        sat = float((act.abs() >= 0.9).mean() * 100.0)
        style(axh, fs.LABEL[k], xlabel=r"$T_{zone}-T_{set}$ (°C)",
              ylabel=("normalised action $a_0$" if i == 0 else None))
        axh.text(0.04, 0.5, f"saturation\n{sat:.0f}%", transform=axh.transAxes, fontsize=8.5,
                 va="center", ha="left", weight="bold", color=c,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.8))
        # marginal action distribution (shares the a0 axis)
        axm.hist(act, bins=44, orientation="horizontal", color=c, alpha=0.78, range=ylim)
        axm.axhspan(0.9, ylim[1], color=fs.ACCURATE, alpha=0.09)
        axm.axhspan(ylim[0], -0.9, color=fs.ACCURATE, alpha=0.09)
        axm.set_ylim(*ylim); axm.axis("off")
    fig.suptitle("Action-density phase portrait: policy action vs thermal error "
                 "(identical axes; red bands = saturation $|a_0|\\geq0.9$; right strip = marginal $a_0$)",
                 fontsize=12, weight="bold")
    save(fig, "block2_q1_polish_phase_density")


def add_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    # 95% chi-square quantile for 2D Gaussian.
    scale = math.sqrt(5.991)
    width, height = 2 * scale * np.sqrt(np.maximum(vals, 1e-12))
    ell = Ellipse((x.mean(), y.mean()), width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.5)
    ax.add_patch(ell)
    ax.scatter(x, y, s=35, color=color, edgecolor="#222222", linewidth=0.4, label=label)
    ax.scatter([x.mean()], [y.mean()], s=90, color=color, edgecolor="#222222", marker="D", linewidth=0.7)


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def fig_morl_pareto_ellipses() -> None:
    pareto = read("reports/morl_pareto_front_table.csv")
    per_seed = read("reports/morl_canonical_seedfix_yearly_per_seed.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    seed42 = pareto[(pareto["complete"] == True) & (pareto["seed"].astype(str) == "42")].copy()
    seed42 = seed42[seed42["kind"].isin(["morl_pareto", "morl_reference"])]
    ax.scatter(seed42["energy_kwh_mean"], seed42["ms_mean"], s=70, color=GREY, edgecolor="#222222", label="seed-42 Pareto diagnostics")
    for _, r in seed42.iterrows():
        txt = f"{float(r['w_comfort']):.2g}/{float(r['w_energy']):.2g}" if str(r["w_comfort"]) != "nan" else str(r["label"])
        ax.text(float(r["energy_kwh_mean"]) + 3, float(r["ms_mean"]), txt, fontsize=7)

    # Canonical N=5 ellipses. Convert yearly sum to monthly mean to match pareto table axis.
    for canonical, color, label in [
        ("comfort_050_energy_050", BLUE, "50/50 N=5"),
        ("comfort_075_energy_025", TEAL, "75/25 N=5"),
    ]:
        d = per_seed[per_seed["canonical"] == canonical].copy()
        x = d["energy_kwh_sum"].astype(float).to_numpy() / 12.0
        y = d["ms_mean"].astype(float).to_numpy()
        add_confidence_ellipse(ax, x, y, color, label)

    points = [(float(r["energy_kwh_mean"]), float(r["ms_mean"])) for _, r in seed42.iterrows()]
    front = pareto_front(points)
    if len(front) >= 2:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color=PURPLE, linestyle="--", linewidth=1.6, label="empirical Pareto envelope")

    style(ax, "MORL comfort-energy Pareto front with N=5 confidence ellipses", "Energy per monthly window (kWh)", "$m_s$")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "block2_q1_polish_morl_pareto_ellipses")


def fig_block3_czon_hypothesis_box() -> None:
    tm = read("reports/block3_transfer_matrix.csv")
    vals = tm["c_zon_ratio_vs_bestest_air"].astype(float).to_numpy()
    labels = ["heat pump", "hydronic", "commercial"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axvspan(1.7, 2.2, color=TEAL, alpha=0.15, label="Hypothesis A: uniform hydronic 1.7-2.2x")
    ax.axvspan(3.0, 10.0, color=ORANGE, alpha=0.12, label="Hypothesis B: scale-dependent 3-10x")
    ax.boxplot(vals, vert=False, positions=[1], widths=0.22, patch_artist=True, boxprops=dict(facecolor="#eeeeee", color="#333333"), medianprops=dict(color=PURPLE, linewidth=2))
    y_jitter = np.array([0.92, 1.0, 1.08])
    ax.scatter(vals, y_jitter, s=70, color=BLUE, edgecolor="#222222", zorder=3)
    for x, y, lab in zip(vals, y_jitter, labels):
        ax.text(x + 0.035, y, f"{lab}: {x:.3f}x", va="center", fontsize=8)
    ax.axvline(vals.mean(), color=PURPLE, linestyle="--", linewidth=1.4, label=f"mean {vals.mean():.3f}x")
    ax.set_yticks([])
    ax.set_xlim(0.8, 4.2)
    style(ax, "$C_{zon}$ hydronic-family consistency against version-locked hypothesis intervals", "$C_{zon}$ ratio vs bestest_air", "")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "block3_q1_polish_czon_hypothesis_box")


def fig_block3_deployment_quadrants() -> None:
    """Deployment plane with the comfort/safety margin M_k = tau_k - m_s^RL on the x-axis
    (pass <=> M_k > 0), energy delta on y, and four interpreted quadrants."""
    tm = read("reports/block3_transfer_matrix.csv")
    margin = (tm["pass_threshold_m_s"].astype(float) - tm["m_s_rl"].astype(float)).to_numpy()
    y = tm["energy_delta_pct_vs_pi"].astype(float).to_numpy()
    labels = ["heat pump", "hydronic", "commercial"]
    pal = [BLUE, PURPLE, TEAL]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    xpad = max(0.15, float(np.abs(margin).max()) * 0.5)
    xlo, xhi = float(margin.min()) - xpad, float(margin.max()) + xpad
    ypad = max(6.0, float(y.max() - y.min()) * 0.3)
    ylo, yhi = min(-8.0, float(y.min()) - ypad), max(12.0, float(y.max()) + ypad)
    fy = (0 - ylo) / (yhi - ylo)
    # quadrant tints: right = pass (M>0); bottom = energy saving (y<0)
    ax.axvspan(0, xhi, ymin=0, ymax=fy, color=fs.V3, alpha=0.10)          # deployable
    ax.axvspan(0, xhi, ymin=fy, ymax=1, color="#d9b44a", alpha=0.16)      # safe but inefficient
    ax.axvspan(xlo, 0, ymin=0, ymax=fy, color=ORANGE, alpha=0.12)         # unsafe energy saving
    ax.axvspan(xlo, 0, ymin=fy, ymax=1, color=fs.ACCURATE, alpha=0.10)    # reject
    ax.axvline(0, color="0.2", ls="--", lw=1.3)
    ax.axhline(0, color="0.2", ls="--", lw=1.3)

    for xi, yi, lab, c in zip(margin, y, labels, pal):
        ax.scatter([xi], [yi], s=120, color=c, edgecolor="white", linewidth=1.0, zorder=4)
        ax.annotate(lab, (xi, yi), xytext=(7, 7), textcoords="offset points", fontsize=9, weight="bold")
    # quadrant captions centred in each region
    xr, xl = (0 + xhi) / 2, (xlo + 0) / 2
    yb, yt = (ylo + 0) / 2, (0 + yhi) / 2
    ax.text(xr, yb, "DEPLOYABLE\npass + energy saving", ha="center", va="center", fontsize=8.5, color="#135f55", weight="bold")
    ax.text(xr, yt, "safe but inefficient\npass + energy penalty", ha="center", va="center", fontsize=8, color="#7a5d00")
    ax.text(xl, yb, "unsafe energy saving\nfail + energy saving", ha="center", va="center", fontsize=8, color="#8a4d17")
    ax.text(xl, yt, "REJECT\nfail + energy penalty", ha="center", va="center", fontsize=8.5, color="#7d1f1f", weight="bold")

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    style(ax, "Block 3 deployment plane: comfort/safety margin vs energy",
          r"comfort/safety margin $M_k=\tau_k-m_s^{RL}$   ($M_k>0$ = pass; $\tau_k=1.25\,m_s^{PI}$)",
          "energy $\\Delta$ vs PI (%)")
    save(fig, "block3_q1_polish_deployment_quadrants")


def fig_block3_radar_transfer_profiles() -> None:
    tm = read("reports/block3_transfer_matrix.csv")
    labels = ["RMSE gain", "comfort pass", "energy parity"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    colors = [BLUE, PURPLE, TEAL]
    names = ["heat pump", "hydronic", "commercial"]
    for (_, r), name, color in zip(tm.iterrows(), names, colors):
        rmse_gain = float(r["rmse_improvement_pct"]) / 100.0
        comfort_score = min(1.0, float(r["pass_threshold_m_s"]) / float(r["m_s_rl"]))
        # 1.0 means no energy penalty or saving; values below 1 indicate penalty.
        ed = float(r["energy_delta_pct_vs_pi"])
        energy_score = 1.0 if ed <= 0 else max(0.0, 1.0 - ed / 50.0)
        values = [rmse_gain, comfort_score, energy_score]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.0, label=name)
        ax.fill(angles, values, color=color, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("Block 3 transfer profile radar: surrogate gain vs controller deployability", pad=20, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=False)
    save(fig, "block3_q1_polish_transfer_radar")


def main() -> None:
    fig_block2_closed_loop_disturbance()
    fig_block2_phase_density()
    fig_morl_pareto_ellipses()
    fig_block3_czon_hypothesis_box()
    fig_block3_deployment_quadrants()
    fig_block3_radar_transfer_profiles()
    print("Wrote Q1-polished Block 2/3 figures to", OUT)


if __name__ == "__main__":
    main()
