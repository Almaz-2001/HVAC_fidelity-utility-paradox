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

TRACE_COLORS = {"pure v3": GREY, "direct v3.5": RED, "hybrid": TEAL}


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
    axes[1].axhspan(21, 24, color=TEAL, alpha=0.12, label="comfort band 21-24 degC")
    style(axes[1], "(b) Zone temperature response", ylabel="$T_{zone}$ (degC)")
    axes[1].legend(ncol=4, frameon=False, fontsize=8)

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
    """Empirical phase portraits as density maps: action versus temperature error."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharex=True, sharey=True)
    for ax, (name, rel) in zip(axes, TRACE_PATHS.items()):
        df = read(rel)
        err = df["t_zone_c"].astype(float) - 22.5
        act = df["a0"].astype(float)
        hb = ax.hexbin(err, act, gridsize=38, cmap="viridis", mincnt=1, linewidths=0, alpha=0.95)
        ax.axhline(-1, color="#333333", linestyle=":", linewidth=0.8)
        ax.axhline(1, color="#333333", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.axvline(0, color="#333333", linewidth=0.7)
        ax.axhspan(0.92, 1.0, color=RED, alpha=0.12)
        ax.axhspan(-1.0, -0.92, color=RED, alpha=0.12)
        style(ax, name, xlabel="$T_{zone}-22.5$ degC", ylabel="normalized action $a_0$")
        sat = float(((act.abs() > 0.92).mean()) * 100.0)
        ax.text(0.03, 0.94, f"saturation share: {sat:.1f}%", transform=ax.transAxes, fontsize=8, va="top")
    fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.85, label="empirical state-action density")
    fig.suptitle("Block 2 phase portrait: empirical density of policy action versus thermal error", fontsize=14, weight="bold")
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
    tm = read("reports/block3_transfer_matrix.csv")
    x = tm["m_s_rl"].astype(float) / tm["pass_threshold_m_s"].astype(float)
    y = tm["energy_delta_pct_vs_pi"].astype(float)
    labels = ["heat pump", "hydronic", "commercial"]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    xmax = max(1.35, float(x.max()) + 0.12)
    ymin = min(-12, float(y.min()) - 5)
    ymax = max(42, float(y.max()) + 7)
    ax.axvspan(0, 1, ymin=0, ymax=(0 - ymin) / (ymax - ymin), color=TEAL, alpha=0.10)
    ax.axvspan(1, xmax, ymin=0, ymax=(0 - ymin) / (ymax - ymin), color=ORANGE, alpha=0.10)
    ax.axvspan(0, 1, ymin=(0 - ymin) / (ymax - ymin), ymax=1, color="#d9b44a", alpha=0.16)
    ax.axvspan(1, xmax, ymin=(0 - ymin) / (ymax - ymin), ymax=1, color=RED, alpha=0.08)
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    colors = [BLUE, PURPLE, TEAL]
    for xi, yi, lab, c in zip(x, y, labels, colors):
        ax.scatter([xi], [yi], s=95, color=c, edgecolor="#222222", zorder=3)
        ax.text(float(xi) + 0.025, float(yi) + 1.2, lab, fontsize=9, weight="bold")
    ax.text(0.08, ymin + 2, "deployment-ready\nthreshold pass + energy saving", fontsize=8, color="#135f55")
    ax.text(1.03, ymin + 2, "comfort/safety fail\nbut energy saving", fontsize=8, color="#8a4d17")
    ax.text(0.08, ymax - 5, "threshold pass\nbut energy penalty", fontsize=8, color="#7a5d00")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    style(ax, "Block 3 comfort-energy deployment plane with interpreted quadrants", "$m_s^{RL} / (1.25 m_s^{PI})$", "Energy delta vs PI (%)")
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
