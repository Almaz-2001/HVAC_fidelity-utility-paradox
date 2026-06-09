"""Build the final Block 1 Q1 figure set from project artifacts.

The figures are written to reports/figures/article_real with stable names:
block1_q1_fig01_... through block1_q1_fig15_....
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "article_real"

BLUE = "#2f5d8c"
TEAL = "#21867a"
ORANGE = "#b25f2c"
PURPLE = "#6f4e7c"
RED = "#c44e52"
GREY = "#5c6470"
LIGHT = "#f4f7fb"


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def box(ax: plt.Axes, xy, wh, text, color=BLUE, fc="#ffffff", fontsize=9) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold", color="#1f2933")


def arrow(ax: plt.Axes, start, end, color=GREY, lw=1.5) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color))


def fig01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ((0.03, 0.58), (0.17, 0.22), "BOPTEST\n15-min traces\n10,744 rows", BLUE),
        ((0.25, 0.58), (0.17, 0.22), "Stage A\nalignment + bias\npreprocessing", TEAL),
        ((0.47, 0.58), (0.17, 0.22), "Stage B\nC_zon inverse\nidentification", TEAL),
        ((0.69, 0.58), (0.17, 0.22), "Stage C\nresidual heads\ncalibration", TEAL),
        ((0.25, 0.16), (0.17, 0.22), "v3 compact\ncontrol surrogate\n8,482 params", BLUE),
        ((0.47, 0.16), (0.17, 0.22), "v3.5 physical twin\nC_zon = 4.413e5\nJ/K", ORANGE),
        ((0.69, 0.16), (0.17, 0.22), "Hybrid backend\nv3 rollout +\nv3.5 regularizer", PURPLE),
        ((0.87, 0.36), (0.11, 0.22), "Live\nBOPTEST\nvalidation", RED),
    ]
    for xy, wh, text, color in boxes:
        box(ax, xy, wh, text, color=color, fc="#ffffff")
    for x0, x1 in [(0.20, 0.25), (0.42, 0.47), (0.64, 0.69)]:
        arrow(ax, (x0, 0.69), (x1, 0.69))
    arrow(ax, (0.335, 0.58), (0.335, 0.38))
    arrow(ax, (0.555, 0.58), (0.555, 0.38))
    arrow(ax, (0.64, 0.27), (0.69, 0.27))
    arrow(ax, (0.86, 0.27), (0.87, 0.43))
    arrow(ax, (0.86, 0.69), (0.90, 0.58), color=RED)
    ax.text(0.02, 0.95, "Block 1 experimental pipeline: from surrogate calibration to live BOPTEST validation", fontsize=15, weight="bold")
    ax.text(0.02, 0.04, "Core claim: predictive fidelity and RL training utility are evaluated separately, then recombined through a role-separated hybrid backend.", fontsize=10, color="#333333")
    save(fig, "block1_q1_fig01_pipeline")


def fig02_v3_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box(ax, (0.03, 0.38), (0.18, 0.24), "8-D input\nT_zone, T_amb,\ntime sin/cos,\nprev actions", BLUE, "#eef5fb")
    box(ax, (0.27, 0.58), (0.20, 0.24), "Shared feature\nnormalization\n_build_features()", GREY, "#ffffff")
    box(ax, (0.54, 0.62), (0.25, 0.24), "HeatFlowNetV2\nLinear + LayerNorm + Tanh\nresidual block\n→ dT", TEAL, "#eefaf7")
    box(ax, (0.54, 0.24), (0.25, 0.24), "PowerNetV2\nLinear + Tanh\nSoftplus output\n→ HVAC power", ORANGE, "#fff6ec")
    box(ax, (0.84, 0.62), (0.12, 0.20), "T_next", TEAL, "#ffffff")
    box(ax, (0.84, 0.24), (0.12, 0.20), "P_HVAC", ORANGE, "#ffffff")
    arrow(ax, (0.21, 0.50), (0.27, 0.68))
    arrow(ax, (0.47, 0.70), (0.54, 0.74))
    arrow(ax, (0.47, 0.62), (0.54, 0.36))
    arrow(ax, (0.79, 0.74), (0.84, 0.72), TEAL)
    arrow(ax, (0.79, 0.36), (0.84, 0.34), ORANGE)
    ax.text(0.03, 0.93, "Dual-head architecture of the control-oriented v3 surrogate", fontsize=15, weight="bold")
    ax.text(0.03, 0.08, "Compact by design: 8,482 trainable parameters. The goal is stable PPO rollout curvature, not maximum long-horizon forecasting fidelity.", fontsize=10, color="#333333")
    save(fig, "block1_q1_fig02_v3_dual_head")


def fig03_stage_improvement() -> None:
    labels = ["1-step RMSE_T\n(C)", "24h RMSE_T\n(C)", "Power MAE\n(kW)"]
    before = [0.3839, 1.4665, 0.8103]
    after = [0.2348, 0.6441, 0.4820]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.18, before, 0.36, label="before calibration", color=ORANGE, edgecolor="#222222", linewidth=0.4)
    ax.bar(x + 0.18, after, 0.36, label="after Stage A/B/C", color=TEAL, edgecolor="#222222", linewidth=0.4)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i, max(b, a) * 1.06, f"-{100*(1-a/b):.0f}%", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    style(ax, "Effect of Stage A/B/C inverse calibration on v3.5 predictive fidelity", ylabel="metric value")
    save(fig, "block1_q1_fig03_stage_abc_improvement")


def fig04_czon_trajectory() -> None:
    df = read_csv("outputs/surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(df["epoch"], df["c_zon_j_per_k"] / 1e5, color=TEAL, linewidth=2.6)
    ax.axhline(4.2, color=GREY, linestyle="--", linewidth=1.2, label="prior 4.200e5 J/K")
    ax.axhline(df["c_zon_j_per_k"].iloc[-1] / 1e5, color=ORANGE, linestyle=":", linewidth=1.8, label=f"identified {df['c_zon_j_per_k'].iloc[-1]/1e5:.3f}e5 J/K")
    style(ax, "Bayesian inverse identification trajectory of C_zon during Stage B", "epoch", "C_zon (1e5 J/K)")
    ax.legend(frameon=False)
    save(fig, "block1_q1_fig04_czon_identification")


def fig05_matched_corpus() -> None:
    df = read_csv("reports/block1_corpus_matched_comparison.csv")
    order = ["v3_hourly", "v3_15min_matched", "v35_raw", "v35_calibrated"]
    labels = ["v3 hourly\nlegacy", "v3 15-min\nmatched", "raw v3.5\n15-min", "calibrated v3.5\n15-min"]
    vals = [df.loc[df.variant == k, "rmse_24h_c"].iloc[0] for k in order]
    colors = [BLUE, BLUE, ORANGE, TEAL]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, vals, color=colors, edgecolor="#222222", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
    style(ax, "Corpus-controlled decomposition of 24h rollout RMSE", ylabel="24h rollout RMSE_T (C)")
    save(fig, "block1_q1_fig05_matched_corpus_rmse")


def fig06_waterfall() -> None:
    start = 1.5572
    mid = 0.8761
    end = 0.6441
    corpus = start - mid
    calib = mid - end
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar([0], [start], color=BLUE, width=0.55, label="starting RMSE")
    ax.bar([1], [-corpus], bottom=[start], color="#76a7d8", width=0.55, label="15-min corpus shift")
    ax.bar([2], [-calib], bottom=[mid], color=TEAL, width=0.55, label="Stage A/B/C calibration")
    ax.bar([3], [end], color=PURPLE, width=0.55, label="final RMSE")
    ax.plot([0.28, 0.72], [start, start], color="#777777", linewidth=1)
    ax.plot([1.28, 1.72], [mid, mid], color="#777777", linewidth=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["v3 hourly\n1.557", "corpus shift\n-0.681\n(74.6%)", "calibration\n-0.232\n(25.4%)", "calibrated v3.5\n0.644"])
    style(ax, "Attribution of the v3-to-v3.5 predictive-fidelity gain", ylabel="24h RMSE_T (C)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "block1_q1_fig06_fidelity_gain_waterfall")


def fig07_fidelity_vs_utility() -> None:
    arch = read_csv("reports/hou_evins_architecture_justification_table.csv")
    points = [
        ("v3", 1.5572, arch.loc[arch.variant == "v3", "typical_control_m_s"].iloc[0], BLUE),
        ("calibrated v3.5", 0.6441, arch.loc[arch.variant == "v35_calibrated", "typical_control_m_s"].iloc[0], ORANGE),
        ("hybrid\n(role separated)", 0.6441, arch.loc[arch.variant == "hybrid_l010", "typical_control_m_s"].iloc[0], PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for label, x, y, c in points:
        ax.scatter([x], [y], s=180, color=c, edgecolor="#222222", linewidth=0.7)
        ax.text(x + 0.035, y + 0.025, label, fontsize=9, weight="bold")
    style(ax, "Predictive fidelity does not imply RL training utility", "24h predictive RMSE_T (C), lower is better", "live BOPTEST m_s, lower is better")
    ax.annotate("better predictor\nworse controller", xy=(0.6441, 1.10), xytext=(0.95, 0.82), arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=10)
    ax.set_ylim(0, 1.25)
    save(fig, "block1_q1_fig07_fidelity_vs_rl_utility")


def fig08_live_performance() -> None:
    arch = read_csv("reports/hou_evins_architecture_justification_table.csv")
    labels = ["v3", "v3.5", "hybrid"]
    variants = ["v3", "v35_calibrated", "hybrid_l010"]
    colors = [BLUE, ORANGE, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(labels))
    w = 0.34
    peak = [arch.loc[arch.variant == v, "peak_control_m_s"].iloc[0] for v in variants]
    typ = [arch.loc[arch.variant == v, "typical_control_m_s"].iloc[0] for v in variants]
    axes[0].bar(x - w / 2, peak, w, color=colors, alpha=0.85, label="peak")
    axes[0].bar(x + w / 2, typ, w, color=colors, alpha=0.45, label="typical")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    style(axes[0], "Maintenance score", ylabel="m_s")
    axes[0].legend(frameon=False)
    peak_e = [arch.loc[arch.variant == v, "peak_energy_kwh"].iloc[0] for v in variants]
    typ_e = [arch.loc[arch.variant == v, "typical_energy_kwh"].iloc[0] for v in variants]
    axes[1].bar(x - w / 2, peak_e, w, color=colors, alpha=0.85, label="peak")
    axes[1].bar(x + w / 2, typ_e, w, color=colors, alpha=0.45, label="typical")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    style(axes[1], "Energy use", ylabel="kWh")
    fig.suptitle("Live closed-loop BOPTEST performance of PPO controllers trained on different backends", fontsize=14, weight="bold")
    save(fig, "block1_q1_fig08_live_boptest_performance")


def fig09_speed() -> None:
    df = read_csv("reports/speed_benchmark_table.csv")
    labels = ["BOPTEST\nHTTP", "v3", "v3.5", "hybrid"]
    keys = ["boptest_rte_http", "v3_surrogate", "v35_calibrated_surrogate", "hybrid_v3_v35_surrogate"]
    vals = [df.loc[df.backend == k, "env_steps_per_sec"].iloc[0] for k in keys]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.bar(labels, vals, color=[GREY, BLUE, ORANGE, PURPLE], edgecolor="#222222", linewidth=0.5)
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.0f}", ha="center", fontsize=9)
    style(ax, "Simulation throughput of BOPTEST and surrogate backends", ylabel="environment steps/s (log scale)")
    save(fig, "block1_q1_fig09_backend_speed")


def fig10_transfer_gap() -> None:
    df = read_csv("reports/hybrid_transfer_comparison.csv")
    variants = ["pure_v3", "direct_v35", "hybrid_l010"]
    labels = ["v3", "direct v3.5", "hybrid"]
    colors = [BLUE, ORANGE, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    metrics = [("boptest_violation_pct", "Violation %", "%"), ("action_gap_norm", "Action gap norm", ""), ("first_divergence_step", "First divergence step", "step")]
    for ax, (col, title, ylabel) in zip(axes, metrics):
        vals = [df[df.variant == v][col].mean() for v in variants]
        ax.bar(labels, vals, color=colors, edgecolor="#222222", linewidth=0.4)
        style(ax, title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Transfer-gap diagnostics reveal bang-bang saturation in standalone v3.5 training", fontsize=14, weight="bold")
    save(fig, "block1_q1_fig10_transfer_gap_diagnostics")


def fig11_action_saturation() -> None:
    paths = {
        "direct v3.5": ROOT / "outputs/block2_bestest_air_15min_thermostatic_v35/traces/peak_heat_window_thermostatic.csv",
        "hybrid": ROOT / "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/peak_heat_window_thermostatic.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    bins = np.linspace(-1, 1, 21)
    for ax, (label, path), color in zip(axes, paths.items(), [ORANGE, PURPLE]):
        df = pd.read_csv(path)
        ax.hist(df["a0"], bins=bins, color=color, alpha=0.85, edgecolor="white")
        sat = ((df["a0"].abs() > 0.95).mean() * 100)
        style(ax, f"{label}: a0 distribution", "raw action a0", "count")
        ax.text(0.04, 0.92, f"|a0| > 0.95: {sat:.1f}%", transform=ax.transAxes, fontsize=10, weight="bold")
    fig.suptitle("Policy action saturation under direct v3.5 training", fontsize=14, weight="bold")
    save(fig, "block1_q1_fig11_action_saturation")


def fig12_per_episode() -> None:
    # Reuse the same aggregate source as the existing replicative validity bars, but show episode-wise RMSE.
    rollouts = {
        "v3": pd.read_csv(ROOT / "outputs/surrogate_v3_rollout_prepared_15min/v3/all_full_rollouts.csv"),
        "raw v3.5": pd.read_csv(ROOT / "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/raw_v35/all_full_rollouts.csv"),
        "calibrated v3.5": pd.read_csv(ROOT / "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/calibrated_v35/all_full_rollouts.csv"),
    }
    records = []
    for name, df in rollouts.items():
        for ep, g in df.groupby("episode_id"):
            err = g["temp_error_c"].to_numpy(float)
            records.append({"model": name, "episode": str(ep).split("__")[-1][:18], "rmse": np.sqrt(np.mean(err**2))})
    data = pd.DataFrame(records)
    eps = sorted(data["episode"].unique())
    models = list(rollouts.keys())
    x = np.arange(len(eps))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, (model, color) in enumerate(zip(models, [BLUE, ORANGE, TEAL])):
        vals = [data[(data.episode == ep) & (data.model == model)]["rmse"].iloc[0] for ep in eps]
        ax.bar(x + (i - 1) * w, vals, w, label=model, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(eps, rotation=35, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style(ax, "Replicative validity across held-out BOPTEST episodes", ylabel="24h RMSE_T (C)")
    save(fig, "block1_q1_fig12_per_episode_rmse")


def fig13_residuals() -> None:
    # Keep the established residual plot under the requested final numbering.
    import shutil

    src_png = OUT / "block1_temperature_residual_histograms.png"
    src_pdf = OUT / "block1_temperature_residual_histograms.pdf"
    if src_png.exists():
        shutil.copyfile(src_png, OUT / "block1_q1_fig13_residual_distributions.png")
    if src_pdf.exists():
        shutil.copyfile(src_pdf, OUT / "block1_q1_fig13_residual_distributions.pdf")


def fig14_hybrid_loss() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, (0.04, 0.55), (0.18, 0.24), "Policy\nπθ(o)", BLUE, "#eef5fb")
    box(ax, (0.30, 0.55), (0.18, 0.24), "v3 rollout\ndynamics\n(train env)", BLUE, "#eef5fb")
    box(ax, (0.56, 0.55), (0.18, 0.24), "PPO reward\ncomfort +\nenergy", TEAL, "#eefaf7")
    box(ax, (0.30, 0.18), (0.18, 0.24), "frozen v3.5\nphysical twin\n(teacher)", ORANGE, "#fff6ec")
    box(ax, (0.78, 0.36), (0.18, 0.24), "L_total = L_PPO\n+ λT||ΔT||²\n+ λP||ΔP||²", PURPLE, "#f7f0fa")
    arrow(ax, (0.22, 0.67), (0.30, 0.67))
    arrow(ax, (0.48, 0.67), (0.56, 0.67))
    arrow(ax, (0.48, 0.30), (0.78, 0.43), ORANGE)
    arrow(ax, (0.74, 0.67), (0.78, 0.53), TEAL)
    arrow(ax, (0.39, 0.55), (0.39, 0.42), GREY)
    ax.text(0.04, 0.92, "Hybrid backend: v3 rollout dynamics with frozen-v3.5 soft regularization", fontsize=15, weight="bold")
    ax.text(0.04, 0.06, "Role separation: v3 supplies smooth trainable rollouts; v3.5 supplies physically calibrated soft targets without becoming the environment.", fontsize=10)
    save(fig, "block1_q1_fig14_hybrid_loss")


def fig15_positioning() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    style(ax, "Positioning of this study against prior HVAC DRL and surrogate-model literature", "surrogate physical structure / interpretability", "closed-loop RL/control validation")
    points = [
        ("Benchmarking\nBOPTEST/Gym", 0.25, 0.75, BLUE),
        ("Grey-box / MPC\nmodels", 0.78, 0.48, TEAL),
        ("Physics-constrained\nthermal DL", 0.82, 0.42, TEAL),
        ("HVAC DRL\ncontrollers", 0.32, 0.78, ORANGE),
        ("Transfer learning\nHVAC", 0.55, 0.62, PURPLE),
        ("This study:\nrole-separated\nhybrid surrogate", 0.82, 0.82, RED),
    ]
    for label, x, y, c in points:
        ax.scatter([x], [y], s=170, color=c, edgecolor="#222222", linewidth=0.7)
        ax.text(x + 0.025, y + 0.015, label, fontsize=9, weight="bold" if "This study" in label else "normal")
    ax.axvspan(0.68, 1.0, color=TEAL, alpha=0.06)
    ax.axhspan(0.68, 1.0, color=BLUE, alpha=0.05)
    ax.text(0.04, 0.05, "Source families: docs/related_works/block1_Surrogate_Fidelity and project Related Works section.", fontsize=8, color="#444444")
    save(fig, "block1_q1_fig15_literature_positioning")


def main() -> None:
    # Existing residual plot is needed for fig13 copy; build the canonical article figures first if absent.
    if not (OUT / "block1_temperature_residual_histograms.png").exists():
        raise FileNotFoundError("Run evaluation/build_article_real_figures.py first to create residual histograms.")
    for fn in [
        fig01_pipeline,
        fig02_v3_architecture,
        fig03_stage_improvement,
        fig04_czon_trajectory,
        fig05_matched_corpus,
        fig06_waterfall,
        fig07_fidelity_vs_utility,
        fig08_live_performance,
        fig09_speed,
        fig10_transfer_gap,
        fig11_action_saturation,
        fig12_per_episode,
        fig13_residuals,
        fig14_hybrid_loss,
        fig15_positioning,
    ]:
        fn()
    print(f"Wrote Block 1 Q1 figures to {OUT}")


if __name__ == "__main__":
    main()
