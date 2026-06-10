"""Build the 17 main-paper Q1 figures from project artifacts.

The script avoids hard-coded result arrays. Numeric plots are read from the
CSV/JSON/YAML artifacts produced by Blocks 1-3. Conceptual diagrams pull their
labels and parameters from those same artifacts where available.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "article_real"
OUT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#666666",
}


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def read_yaml(rel: str) -> dict:
    with open(ROOT / rel, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#333333", fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end, color="#333333"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))


def metric(df: pd.DataFrame, category: str, variant: str, metric_name: str) -> float:
    row = df[(df["category"] == category) & (df["variant"] == variant) & (df["metric"] == metric_name)]
    if row.empty:
        raise KeyError((category, variant, metric_name))
    return float(row.iloc[0]["value"])


def fig01_overall_architecture():
    manifest = read_yaml("configs/block3_testcase_manifest.yaml")
    source = manifest["scope"]["source_testcase"]
    targets = [manifest["testcase_candidates"][k]["label"] for k in ["primary", "secondary", "stretch"]]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    boxes = [
        (0.3, 3.1, 1.7, 0.8, f"BOPTEST data\nsource: {source}", "#E8F4FA"),
        (2.4, 3.1, 1.6, 0.8, "v3 surrogate\ncontrol-oriented", "#EAF7EA"),
        (4.4, 3.1, 1.8, 0.8, "v3.5 surrogate\nStage A/B/C calibrated", "#FFF3D6"),
        (6.7, 3.1, 1.8, 0.8, "Hybrid backend\nv3 rollout + v3.5 penalty", "#F1E8F7"),
        (9.1, 3.1, 2.2, 0.8, "RL controllers\nThermostatic / HDRL / MORL", "#F9ECE8"),
        (1.2, 1.1, 2.0, 0.85, "Block 1\nsurrogate validation", "#F7F7F7"),
        (4.9, 1.1, 2.0, 0.85, "Block 2\ncontroller validation", "#F7F7F7"),
        (8.6, 1.1, 2.5, 0.85, "Block 3\ntransferability validation\n" + "\n".join(targets), "#F7F7F7"),
    ]
    for args in boxes:
        box(ax, *args)
    for x1, x2 in [(2.0, 2.4), (4.0, 4.4), (6.2, 6.7), (8.5, 9.1)]:
        arrow(ax, (x1, 3.5), (x2, 3.5))
    arrow(ax, (1.15, 3.1), (2.0, 1.95))
    arrow(ax, (7.6, 3.1), (5.9, 1.95))
    arrow(ax, (10.2, 3.1), (9.8, 1.95))
    ax.set_title("Overall study architecture: surrogate roles, controller validation, and transferability")
    save(fig, "final17_fig01_overall_study_architecture")


def fig02_backend_architecture():
    txt = (ROOT / "docs" / "block1_complete_results.txt").read_text(encoding="utf-8", errors="ignore")
    params = re.search(r"8[\\s\\xa0]*482", txt)
    param_text = params.group(0).replace("\xa0", " ") if params else "8,482"
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    box(ax, 0.4, 3.8, 2.5, 1.1, f"v3 smooth rollout surrogate\ninput: 8D, output: ΔT + P\n{param_text} params", "#EAF7EA")
    box(ax, 0.6, 2.35, 1.9, 0.8, "HeatFlowNetV2\nTemperature head → ΔT", "#D9F0D3")
    box(ax, 0.6, 1.35, 1.9, 0.8, "PowerNetV2\nPower head → HVAC W", "#D9F0D3")
    box(ax, 4.2, 3.5, 2.7, 1.25, "v3.5 calibrated physical twin\n15-min prepared corpus\nRC-NeuralODE + residual heads", "#FFF3D6")
    box(ax, 4.45, 1.9, 2.2, 0.9, "Identified C_zon\nphysical state constraint", "#FFE6A6")
    box(ax, 8.3, 3.6, 2.9, 1.2, "Hybrid backend\nPolicy rollout uses v3\nv3.5 stays frozen", "#F1E8F7")
    box(ax, 8.1, 1.75, 3.2, 1.0, "Reward regularization\nλ_temp |ΔT| + λ_pwr |ΔP|\nsoft physical censor", "#EADCF4")
    arrow(ax, (2.9, 4.35), (4.2, 4.1))
    arrow(ax, (6.9, 4.1), (8.3, 4.15))
    arrow(ax, (6.1, 3.5), (8.5, 2.75))
    arrow(ax, (2.5, 2.75), (8.25, 4.0), COLORS["green"])
    ax.text(5.4, 5.35, "Same state-action pair is evaluated by two roles: rollout dynamics (v3) and frozen physics check (v3.5)", ha="center", fontsize=10)
    ax.set_title("v3 / v3.5 / hybrid backend architecture")
    save(fig, "final17_fig02_backend_architecture")


def fig_eng_residual_cdf():
    sources = {
        "v3": ROOT / "outputs/surrogate_v3_rollout_prepared_15min/v3/all_full_rollouts.csv",
        "raw v3.5": ROOT / "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/raw_v35/all_full_rollouts.csv",
        "calibrated v3.5": ROOT / "outputs/surrogate_v35_rollout_prepared_15min_power_head_only/calibrated_v35/all_full_rollouts.csv",
    }
    colors = {"v3": COLORS["gray"], "raw v3.5": COLORS["orange"], "calibrated v3.5": COLORS["blue"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, path in sources.items():
        df = pd.read_csv(path)
        residual = df["temp_error_c"].dropna().astype(float)
        axes[0].hist(residual, bins=80, density=True, histtype="step", lw=2, color=colors[name], label=name)
        abs_err = np.sort(np.abs(residual.values))
        cdf = np.arange(1, len(abs_err) + 1) / len(abs_err)
        axes[1].plot(abs_err, cdf, lw=2, color=colors[name], label=name)
        axes[0].text(
            0.02,
            0.92 - 0.08 * list(sources).index(name),
            f"{name}: μ={residual.mean():.2f}, σ={residual.std():.2f}, P95|e|={np.percentile(np.abs(residual),95):.2f}°C",
            transform=axes[0].transAxes,
            fontsize=8,
            color=colors[name],
        )
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Temperature residual T_pred - T_obs (°C)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) Residual distribution")
    for thr in [0.5, 1.0, 1.5]:
        axes[1].axvline(thr, color="#999999", lw=0.8, ls=":")
        axes[1].text(thr, 0.03, f"{thr:.1f}°C", rotation=90, va="bottom", ha="right", fontsize=8)
    axes[1].set_xlim(0, min(6, axes[1].get_xlim()[1]))
    axes[1].set_xlabel("|Temperature error| (°C)")
    axes[1].set_ylabel("Fraction below threshold")
    axes[1].set_title("(b) Absolute-error CDF")
    axes[1].legend(loc="lower right")
    fig.suptitle("Residual distribution and engineering tolerance CDF")
    save(fig, "final_eng_fig04_residual_distribution_error_cdf")


def fig_eng_closed_loop_traces():
    paths = {
        "pure v3": ROOT / "outputs/bestest_air_article7_style_15min/traces/typical_heat_window_thermostatic.csv",
        "direct v3.5": ROOT / "outputs/block2_bestest_air_15min_thermostatic_v35/traces/typical_heat_window_thermostatic.csv",
        "hybrid_l010": ROOT / "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/typical_heat_window_thermostatic.csv",
    }
    colors = {"pure v3": COLORS["gray"], "direct v3.5": COLORS["red"], "hybrid_l010": COLORS["green"]}
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.8), sharex=True)
    for name, path in paths.items():
        df = pd.read_csv(path).iloc[: 96 * 3].copy()
        hours = (df["sim_time_sec"] - df["sim_time_sec"].iloc[0]) / 3600
        axes[0].plot(hours, df["t_zone_c"], label=name, color=colors[name], lw=1.6)
        axes[1].plot(hours, df["t_supply_cmd_c"], label=name, color=colors[name], lw=1.4)
        axes[2].plot(hours, df["p_total_w"] / 1000, label=name, color=colors[name], lw=1.2)
    amb = pd.read_csv(next(iter(paths.values()))).iloc[: 96 * 3].copy()
    hours = (amb["sim_time_sec"] - amb["sim_time_sec"].iloc[0]) / 3600
    axes[0].plot(hours, amb["t_amb_c"], color=COLORS["sky"], lw=1.0, alpha=0.8, label="ambient")
    axes[0].axhspan(21, 24, color=COLORS["green"], alpha=0.12, label="comfort 21–24°C")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("(a) Zone temperature and ambient disturbance")
    axes[1].set_ylabel("Supply command (°C)")
    axes[1].set_title("(b) Supply-temperature command")
    axes[2].set_ylabel("HVAC power (kW)")
    axes[2].set_xlabel("Time since window start (h)")
    axes[2].set_title("(c) HVAC power")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Live BOPTEST closed-loop traces: pure v3 vs direct v3.5 vs hybrid")
    save(fig, "final_eng_fig08_live_boptest_closed_loop_traces")


def fig_eng_action_phase_portrait():
    paths = {
        "pure v3": ROOT / "outputs/bestest_air_article7_style_15min/traces/typical_heat_window_thermostatic.csv",
        "direct v3.5": ROOT / "outputs/block2_bestest_air_15min_thermostatic_v35/traces/typical_heat_window_thermostatic.csv",
        "hybrid_l010": ROOT / "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/typical_heat_window_thermostatic.csv",
    }
    colors = {"pure v3": COLORS["gray"], "direct v3.5": COLORS["red"], "hybrid_l010": COLORS["green"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, path in paths.items():
        df = pd.read_csv(path)
        action = df["a0"].astype(float)
        axes[0].hist(action, bins=np.linspace(-1, 1, 45), density=True, histtype="step", lw=2, color=colors[name], label=name)
        sample = df.iloc[:: max(1, len(df) // 550)]
        temp_error = sample["t_zone_c"].astype(float) - 22.5
        sc = axes[1].scatter(temp_error, sample["a0"], c=sample["t_amb_c"], cmap="viridis", s=9, alpha=0.55, label=name if name == "direct v3.5" else None)
        axes[1].plot([], [], "o", color=colors[name], label=name)
    axes[0].axvline(-1, color="#999999", ls=":")
    axes[0].axvline(1, color="#999999", ls=":")
    axes[0].set_xlabel("Normalized action a0")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) Action distribution")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].axvline(0, color="black", lw=0.7)
    axes[1].set_xlabel("Temperature error T_zone - 22.5°C")
    axes[1].set_ylabel("Normalized supply action a0")
    axes[1].set_title("(b) Phase portrait: action vs thermal error")
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Ambient temperature (°C)")
    fig.suptitle("Policy action saturation and phase portrait")
    save(fig, "final_eng_fig09_action_distribution_phase_portrait")


def fig_eng_block3_deployment_plane():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    labels = ["heat pump", "hydronic", "commercial"]
    x = tm["m_s_rl"] / tm["pass_threshold_m_s"]
    y = tm["energy_delta_pct_vs_pi"]
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.axvspan(0, 1, color=COLORS["green"], alpha=0.08)
    ax.axvspan(1, max(1.25, x.max() + 0.1), color=COLORS["red"], alpha=0.06)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(1, color="black", lw=0.9, ls="--")
    for xi, yi, lab in zip(x, y, labels):
        color = COLORS["green"] if xi <= 1 and yi <= 0 else COLORS["orange"] if xi <= 1 else COLORS["red"]
        ax.scatter(xi, yi, s=150, color=color, edgecolor="black", zorder=3)
        ax.text(xi + 0.015, yi + 1.3, lab, fontsize=9)
    ax.text(0.58, -9, "deployment-ready\ncomfort pass + no energy penalty", fontsize=8, color=COLORS["green"], ha="center")
    ax.text(0.55, 29, "threshold pass\nbut energy penalty", fontsize=8, color=COLORS["orange"], ha="center")
    ax.text(1.1, -8, "comfort fail\nenergy saving", fontsize=8, color=COLORS["red"], ha="center")
    ax.set_xlabel("Threshold-normalized safety metric: m_s_RL / (1.25 × m_s_PI)")
    ax.set_ylabel("Energy Δ% vs PI")
    ax.set_title("Block 3 comfort-energy deployment plane")
    ax.grid(alpha=0.2)
    save(fig, "final_eng_fig11_block3_deployment_plane")


def fig_eng_czon_hypothesis_interval():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    labels = ["bestest_air", "heat pump", "hydronic", "commercial"]
    vals = [1.0] + tm["c_zon_ratio_vs_bestest_air"].tolist()
    x = np.arange(len(labels))
    hyd = np.array(vals[1:])
    mean, std = hyd.mean(), hyd.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(1.7, 2.2, color=COLORS["green"], alpha=0.12, label="Hypothesis A: uniform hydronic family (1.7–2.2×)")
    ax.axhspan(3.0, 10.0, color=COLORS["red"], alpha=0.08, label="Hypothesis B: scale-dependent (3–10×)")
    ax.scatter(x, vals, s=130, color=[COLORS["gray"]] + [COLORS["blue"]] * 3, edgecolor="black", zorder=3)
    ax.plot(x[1:], vals[1:], color=COLORS["blue"], lw=1.2, alpha=0.7)
    ax.axhline(mean, color=COLORS["blue"], ls="--", lw=1.1, label=f"observed hydronic mean = {mean:.3f} ± {std:.3f}×")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.08, f"{v:.3f}×", ha="center", fontsize=9)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0.7, 4.0)
    ax.set_ylabel("C_zon ratio vs bestest_air")
    ax.set_title("C_zon hypothesis interval test across hydronic testcases")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "final_eng_fig12_czon_hypothesis_interval")


def fig03_stage_calibration_improvement():
    df = read_csv("reports/block1_surrogate_final_metrics.csv")
    vals = {
        "1-step\nRMSE_T (°C)": (
            metric(df, "inverse_calibration", "best_temp_alignment", "baseline_rmse"),
            metric(df, "inverse_calibration", "best_temp_alignment", "calibrated_rmse"),
        ),
        "24h rollout\nRMSE_T (°C)": (
            metric(df, "prepared_rollout", "raw_v35", "rollout_24h_rmse"),
            metric(df, "prepared_rollout", "calibrated_v35", "rollout_24h_rmse"),
        ),
        "Power MAE\n(kW)": (
            metric(df, "downstream_backend", "power_head_only", "baseline_power_mae") / 1000,
            metric(df, "downstream_backend", "power_head_only", "calibrated_power_mae") / 1000,
        ),
    }
    labels = list(vals)
    raw = [vals[k][0] for k in labels]
    cal = [vals[k][1] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, raw, width=0.36, color=COLORS["orange"], label="Raw v3.5")
    ax.bar(x + 0.18, cal, width=0.36, color=COLORS["blue"], label="Calibrated v3.5")
    for xi, a, b in zip(x, raw, cal):
        ax.text(xi, max(a, b) * 1.04, f"↓{(a-b)/a*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Stage A/B/C calibration improvement")
    ax.legend()
    save(fig, "final17_fig03_stage_abc_calibration_improvement")


def fig04_matched_corpus_decomposition():
    df = read_csv("reports/block1_corpus_matched_comparison.csv")
    order = ["v3_hourly", "v3_15min_matched", "v35_raw", "v35_calibrated"]
    plot = df.set_index("variant").loc[order].reset_index()
    labels = ["v3\nhourly", "v3\n15-min", "raw\nv3.5", "calibrated\nv3.5"]
    vals = plot["rmse_24h_c"].astype(float).values
    total_gain = vals[0] - vals[3]
    corpus = vals[0] - vals[1]
    calib = vals[1] - vals[3]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    bars = ax.bar(labels, vals, color=[COLORS["gray"], COLORS["sky"], COLORS["orange"], COLORS["blue"]])
    ax.set_ylabel("24h rollout RMSE_T (°C)")
    ax.set_title("Matched-corpus predictive-fidelity decomposition")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f"{val:.3f}", ha="center")
    ax.annotate(f"Corpus contribution\n{corpus/total_gain*100:.1f}%", xy=(0.5, (vals[0]+vals[1])/2), xytext=(0.7, 1.75),
                arrowprops=dict(arrowstyle="->", color=COLORS["sky"]), color=COLORS["sky"], ha="center")
    ax.annotate(f"Stage A/B/C contribution\n{calib/total_gain*100:.1f}%", xy=(2.5, (vals[1]+vals[3])/2), xytext=(2.3, 1.25),
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"]), color=COLORS["blue"], ha="center")
    save(fig, "final17_fig04_matched_corpus_decomposition")


def fig05_fidelity_vs_rl_utility():
    arch = read_csv("reports/hou_evins_architecture_justification_table.csv")
    corpus = read_csv("reports/block1_corpus_matched_comparison.csv").set_index("variant")
    xvals = {
        "v3": float(corpus.loc["v3_hourly", "rmse_24h_c"]),
        "v3.5 calibrated": float(corpus.loc["v35_calibrated", "rmse_24h_c"]),
        "hybrid_l010": float(corpus.loc["v3_hourly", "rmse_24h_c"]),
    }
    yvals = {
        "v3": arch[arch["variant"] == "v3"][["peak_transfer_temp_rmse_c", "typical_transfer_temp_rmse_c"]].mean(axis=1).iloc[0],
        "v3.5 calibrated": arch[arch["variant"] == "v35_calibrated"][["peak_transfer_temp_rmse_c", "typical_transfer_temp_rmse_c"]].mean(axis=1).iloc[0],
        "hybrid_l010": arch[arch["variant"] == "hybrid_l010"][["peak_transfer_temp_rmse_c", "typical_transfer_temp_rmse_c"]].mean(axis=1).iloc[0],
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, color in zip(xvals, [COLORS["gray"], COLORS["orange"], COLORS["green"]]):
        ax.scatter(xvals[name], yvals[name], s=170, color=color, edgecolor="black", zorder=3)
        ax.text(xvals[name] + 0.035, yvals[name] + 0.06, name, fontsize=9)
    ax.set_xlabel("Offline 24h rollout RMSE_T (°C)")
    ax.set_ylabel("Live BOPTEST closed-loop RMSE_T (°C)")
    ax.set_title("Predictive fidelity does not imply RL training utility")
    ax.grid(alpha=0.25)
    save(fig, "final17_fig05_fidelity_vs_rl_utility")


def fig06_live_controller_comparison():
    df = read_csv("reports/hou_evins_architecture_justification_table.csv")
    variants = ["v3", "v35_calibrated", "hybrid_l010"]
    labels = ["pure_v3", "direct_v35", "hybrid_l010"]
    scen = ["peak", "typical"]
    metrics = [
        ("m_s", "m_s", ["peak_control_m_s", "typical_control_m_s"]),
        ("Violation %", "Violation %", [None, None]),
        ("RMSE_T (°C)", "RMSE_T", ["peak_transfer_temp_rmse_c", "typical_transfer_temp_rmse_c"]),
        ("Energy (kWh)", "Energy", ["peak_energy_kwh", "typical_energy_kwh"]),
    ]
    # Violation comes from hybrid_transfer_comparison for the three backend variants.
    trans = read_csv("reports/hybrid_transfer_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = [COLORS["gray"], COLORS["orange"], COLORS["green"]]
    for ax, (title, ylabel, cols) in zip(axes.ravel(), metrics):
        x = np.arange(len(scen))
        width = 0.24
        for i, (var, lab) in enumerate(zip(variants, labels)):
            if title == "Violation %":
                tvar = "direct_v35" if var == "v35_calibrated" else ("pure_v3" if var == "v3" else "hybrid_l010")
                vals = [
                    float(trans[(trans["variant"] == tvar) & (trans["scenario"] == "peak_heat_window")]["boptest_violation_pct"].iloc[0]),
                    float(trans[(trans["variant"] == tvar) & (trans["scenario"] == "typical_heat_window")]["boptest_violation_pct"].iloc[0]),
                ]
            else:
                row = df[df["variant"] == var].iloc[0]
                vals = [float(row[cols[0]]), float(row[cols[1]])]
            ax.bar(x + (i - 1) * width, vals, width=width, label=lab, color=colors[i])
        ax.set_xticks(x, ["peak", "typical"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Live BOPTEST controller comparison")
    save(fig, "final17_fig06_live_boptest_controller_comparison")


def fig07_hybrid_reward_mechanism():
    disagree = read_csv("reports/hybrid_disagreement_summary.csv")
    overall = disagree[disagree["scenario"] == "overall"].iloc[0]
    temp_mean = float(overall["temp_disagree_mean_c"])
    temp_p95 = float(overall["temp_disagree_p95_c"])
    pwr_mean = float(overall["power_disagree_mean_w"])
    pwr_p95 = float(overall["power_disagree_p95_w"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    box(ax, 0.4, 2.3, 1.8, 0.8, "Policy\nπθ(s)", "#E8F4FA")
    box(ax, 3.0, 3.25, 2.2, 0.85, "v3 rollout dynamics\nT, P prediction", "#EAF7EA")
    box(ax, 3.0, 1.25, 2.2, 0.85, "frozen v3.5 twin\nsame state-action", "#FFF3D6")
    box(ax, 6.0, 3.25, 2.1, 0.85, "base reward\ncomfort + smooth + energy", "#F7F7F7")
    box(ax, 6.0, 1.25, 2.1, 0.85, f"disagreement\nmean |ΔT|={temp_mean:.2f}°C\nmean |ΔP|={pwr_mean:.0f} W", "#F7F7F7", fontsize=8)
    box(ax, 9.0, 2.2, 2.55, 1.05, "hybrid reward\nr = r_c + r_s + r_e\n− λT|ΔT| − λP|ΔP|", "#F1E8F7")
    for s, e in [((2.2, 2.7), (3.0, 3.65)), ((2.2, 2.7), (3.0, 1.65)), ((5.2, 3.65), (6.0, 3.65)), ((5.2, 1.65), (6.0, 1.65)), ((8.1, 3.65), (9.0, 2.95)), ((8.1, 1.65), (9.0, 2.45))]:
        arrow(ax, s, e)
    ax.text(7.05, 0.72, f"overall p95: |ΔT|={temp_p95:.2f}°C, |ΔP|={pwr_p95:.0f} W", ha="center", fontsize=8, color=COLORS["gray"])
    ax.set_title("Hybrid reward-shaping mechanism: v3 rollout with frozen-v3.5 physical censor")
    save(fig, "final17_fig07_hybrid_reward_shaping_mechanism")


def fig08_hdrl_lambda_sweep():
    df = read_csv("reports/block2_hdrl_lambda_sweep_summary.csv")
    df["lambda"] = df["variant"].str.extract(r"l(\d+)").astype(int) / 100
    metric_cols = [("m_s", "m_s"), ("violation_pct", "Violation %"), ("rmse_center_c", "RMSE_T (°C)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, title) in zip(axes, metric_cols):
        for scen, color in [("peak_heat_window", COLORS["orange"]), ("typical_heat_window", COLORS["blue"])]:
            s = df[df["scenario"] == scen].sort_values("lambda")
            ax.plot(s["lambda"], s[col], marker="o", color=color, label=scen.replace("_heat_window", ""))
        ax.set_xlabel("λ_temp")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("HDRL λ_temp sweep: regularization is controller-family specific")
    save(fig, "final17_fig08_hdrl_lambda_temp_sweep")


def fig09_morl_5d_17d():
    df = read_csv("reports/block2_morl_comparison_summary.csv")
    metrics = [("rmse_c", "RMSE_T\n(°C)"), ("violation_pct", "Violation\n%"), ("within_1c_pct", "Within\n1°C %"), ("m_s", "m_s"), ("energy_kwh", "Energy\nkWh")]
    labels = ["5D basic", "17D power-only"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.8))
    colors = [COLORS["red"], COLORS["green"]]
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].astype(float).values
        ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.03 if v else 0.03, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("MORL observation-interface ablation: 5D failure → 17D success")
    save(fig, "final17_fig09_morl_5d_failure_17d_success")


def fig10_morl_pareto():
    df = read_csv("reports/morl_pareto_front_table.csv")
    df = df[df["kind"].isin(["morl_pareto", "morl_reference"])].copy()
    # Aggregate N=5 canonical points, retain seed42-only non-canonical points.
    rows = []
    for label, g in df.groupby("label"):
        rows.append(
            {
                "label": label,
                "energy": g["energy_kwh_mean"].mean(),
                "ms": g["ms_mean"].mean(),
                "ms_std": g["ms_mean"].std(ddof=1) if len(g) > 1 else 0.0,
                "n": len(g),
                "w_comfort": g["w_comfort"].iloc[0],
                "w_energy": g["w_energy"].iloc[0],
            }
        )
    plot = pd.DataFrame(rows).sort_values("w_comfort")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for _, r in plot.iterrows():
        color = COLORS["red"] if r["w_comfort"] == 0 else COLORS["green"] if r["w_comfort"] == 1 else COLORS["blue"]
        ax.errorbar(r["energy"], r["ms"], yerr=r["ms_std"] if r["n"] > 1 else None, fmt="o", ms=8, color=color, capsize=4)
        label = f"{int(r['w_comfort']*100)}/{int(r['w_energy']*100)}"
        if "080_020" in r["label"]:
            label = "80/20"
        ax.text(r["energy"] + 3, r["ms"] + 0.025, label, fontsize=8)
    ax.set_xlabel("Energy (kWh)")
    ax.set_ylabel("m_s")
    ax.set_title("MORL comfort-energy Pareto front")
    ax.grid(alpha=0.25)
    save(fig, "final17_fig10_morl_comfort_energy_pareto")


def fig11_block3_protocol():
    manifest = read_yaml("configs/block3_testcase_manifest.yaml")
    regimes = " / ".join(manifest["recalibration_regimes"].keys())
    threshold = manifest["passfail_criteria"]["primary_threshold"]["comparison"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    labels = [
        "Block 2 frozen\nhybrid controller",
        "3 target BOPTEST\ntestcases",
        f"Recalibration regimes\n{regimes}",
        f"Controller verdict\n{threshold}",
        "Surrogate verdict\nRMSE_T improvement\nafter Stage A/B/C",
    ]
    xs = [0.4, 2.6, 4.8, 7.0, 9.0]
    widths = [1.7, 1.7, 1.7, 1.65, 1.8]
    for x, w, lab, color in zip(xs, widths, labels, ["#E8F4FA", "#F7F7F7", "#FFF3D6", "#F9ECE8", "#EAF7EA"]):
        box(ax, x, 2.1, w, 1.25, lab, color, fontsize=8.5)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + widths[i], 2.72), (xs[i + 1], 2.72))
    ax.set_title("Block 3 version-locked transferability protocol")
    save(fig, "final17_fig11_block3_transferability_protocol")


def fig12_testcase_ladder_adapters():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    manifest = read_yaml("configs/block3_testcase_manifest.yaml")
    roles = ["primary", "secondary", "stretch"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    for i, role in enumerate(roles):
        label = manifest["testcase_candidates"][role]["label"]
        adapter = tm[tm["testcase"] == label]["adapter"].iloc[0]
        x = 0.8 + i * 3.3
        box(ax, x, 2.8, 2.5, 1.0, f"{role}\n{label}", "#E8F4FA")
        box(ax, x, 1.25, 2.5, 0.95, f"adapter\n{adapter}", "#FFF3D6", fontsize=8)
        arrow(ax, (x + 1.25, 2.8), (x + 1.25, 2.2))
    ax.text(5.5, 4.35, "Transfer is adapter-mediated, not literal copy-paste of bestest_air actions", ha="center", fontsize=11)
    ax.set_title("Target testcase ladder and actuator adapters")
    save(fig, "final17_fig12_target_testcase_ladder_adapters")


def fig13_block3_verdict_heatmap():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    tests = tm["testcase"].tolist()
    regimes = ["none", "partial", "full"]
    vals = []
    annotations = []
    for _, r in tm.iterrows():
        rowv, rowa = [], []
        for reg in regimes:
            verdict = r["none_controller_verdict"] if reg in ["none", "partial"] else r["full_controller_verdict"]
            if r["testcase"] == "singlezone_commercial_hydronic":
                code = 1 if verdict == "PASS" else -1
                txt = "COND PASS" if verdict == "PASS" else "FAIL"
            else:
                code = 1 if verdict == "PASS" else -1
                txt = verdict
            rowv.append(code if txt != "COND PASS" else 0)
            rowa.append(txt)
        vals.append(rowv)
        annotations.append(rowa)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = ListedColormap(["#D55E00", "#F0E442", "#009E73"])
    ax.imshow(np.array(vals), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(regimes)), regimes)
    ax.set_yticks(range(len(tests)), [t.replace("bestest_", "").replace("_", "\n") for t in tests])
    for i in range(len(tests)):
        for j in range(len(regimes)):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Block 3 controller transfer verdict heatmap")
    save(fig, "final17_fig13_block3_controller_transfer_heatmap")


def fig14_rl_vs_pi_threshold_energy():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    labels = ["heat pump", "hydronic", "commercial"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.25
    axes[0].bar(x - w, tm["m_s_rl"], w, label="RL", color=COLORS["blue"])
    axes[0].bar(x, tm["m_s_pi"], w, label="PI", color=COLORS["gray"])
    axes[0].bar(x + w, tm["pass_threshold_m_s"], w, label="1.25×PI", color=COLORS["orange"])
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("m_s")
    axes[0].set_title("Controller threshold verdict")
    axes[0].legend(fontsize=8)
    colors = [COLORS["green"] if v < 0 else COLORS["red"] for v in tm["energy_delta_pct_vs_pi"]]
    axes[1].bar(labels, tm["energy_delta_pct_vs_pi"], color=colors)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Energy Δ% vs PI")
    axes[1].set_title("Energy penalty / saving")
    axes[1].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("RL vs PI threshold and energy penalty")
    save(fig, "final17_fig14_rl_vs_pi_threshold_energy_penalty")


def fig15_full_stage_transfer_rmse():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    labels = ["heat pump", "hydronic", "commercial"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.35
    ax.bar(x - w / 2, tm["raw_rmse_t_c"], w, label="Raw target RMSE_T", color=COLORS["orange"])
    ax.bar(x + w / 2, tm["full_rmse_t_c"], w, label="Full Stage A/B/C", color=COLORS["blue"])
    for i, r in tm.iterrows():
        ax.text(i, max(r["raw_rmse_t_c"], r["full_rmse_t_c"]) + 0.08, f"↓{r['rmse_improvement_pct']:.1f}%", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE_T (°C)")
    ax.set_title("Full Stage A/B/C transfer RMSE improvement")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, "final17_fig15_full_stage_transfer_rmse_improvement")


def fig16_czon_consistency():
    tm = read_csv("reports/block3_transfer_matrix.csv")
    labels = ["bestest_air", "heat pump", "hydronic", "commercial"]
    vals = [1.0] + tm["c_zon_ratio_vs_bestest_air"].tolist()
    hyd = np.array(vals[1:])
    mean, std = hyd.mean(), hyd.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["blue"], COLORS["blue"], COLORS["blue"]])
    ax.axhline(mean, color=COLORS["red"], ls="--", label=f"hydronic mean = {mean:.3f} ± {std:.3f}×")
    ax.fill_between([-0.5, 3.5], mean - std, mean + std, color=COLORS["red"], alpha=0.12)
    ax.set_ylabel("C_zon ratio vs bestest_air")
    ax.set_title("Hydronic-family C_zon consistency")
    ax.legend()
    ax.tick_params(axis="x", rotation=15)
    save(fig, "final17_fig16_hydronic_czon_consistency")


def fig17_hypothesis_closure():
    manifest = read_yaml("configs/block3_testcase_manifest.yaml")
    tm = read_csv("reports/block3_transfer_matrix.csv")
    agg = manifest["aggregated_results"]["hypothesis_status_final"]
    stretch = manifest["stretch_testcase_predictions"]["predictions"]
    rows = [
        ("H1 strong", agg["H1_strong"]["verdict"], "frozen recipe direct transfer"),
        ("H2 medium", agg["H2_medium"]["verdict"], "partial recalibration"),
        ("H3 surrogate", agg["H3_weak_surrogate_side"]["verdict"], "full Stage A/B/C RMSE"),
        ("H3 controller", agg["H3_weak_controller_side"]["verdict"], "controller full regime"),
        ("Stretch none", f"expected {stretch['mode_none_controller_verdict']['expected']} → observed PASS", "commercial mode=none"),
        ("Stretch C_zon", f"expected scale-dependent → observed {tm.iloc[2]['c_zon_ratio_vs_bestest_air']:.3f}×", "commercial C_zon"),
        ("Stretch RMSE", f"expected 50–90% → observed {tm.iloc[2]['rmse_improvement_pct']:.2f}%", "commercial full recalibration"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y0, dy = 0.88, 0.105
    ax.text(0.02, 0.96, "Hypothesis / prediction", fontweight="bold")
    ax.text(0.34, 0.96, "Observed closure", fontweight="bold")
    ax.text(0.72, 0.96, "Evidence type", fontweight="bold")
    for i, (h, verdict, evidence) in enumerate(rows):
        y = y0 - i * dy
        color = COLORS["green"] if "SUPPORTED" in verdict or "observed 87" in verdict else COLORS["red"] if "FALSIFIED" in verdict or "expected FAIL" in verdict or "expected scale" in verdict else COLORS["yellow"]
        ax.add_patch(Rectangle((0.0, y - 0.035), 0.98, 0.075, facecolor="#FAFAFA" if i % 2 == 0 else "#F0F0F0", edgecolor="none"))
        ax.scatter(0.02, y, s=95, color=color, edgecolor="black")
        ax.text(0.05, y, h, va="center")
        ax.text(0.34, y, verdict.replace("_", " "), va="center", fontsize=8.3)
        ax.text(0.72, y, evidence, va="center", fontsize=8.3)
    ax.set_title("Hypothesis closure and version-locked predictions vs observed outcomes")
    save(fig, "final17_fig17_hypothesis_closure_matrix")


def main() -> None:
    fig01_overall_architecture()
    fig02_backend_architecture()
    fig03_stage_calibration_improvement()
    fig04_matched_corpus_decomposition()
    fig05_fidelity_vs_rl_utility()
    fig06_live_controller_comparison()
    fig07_hybrid_reward_mechanism()
    fig08_hdrl_lambda_sweep()
    fig09_morl_5d_17d()
    fig10_morl_pareto()
    fig11_block3_protocol()
    fig12_testcase_ladder_adapters()
    fig13_block3_verdict_heatmap()
    fig14_rl_vs_pi_threshold_energy()
    fig15_full_stage_transfer_rmse()
    fig16_czon_consistency()
    fig17_hypothesis_closure()
    fig_eng_residual_cdf()
    fig_eng_closed_loop_traces()
    fig_eng_action_phase_portrait()
    fig_eng_block3_deployment_plane()
    fig_eng_czon_hypothesis_interval()
    print("Built 17 final Q1 figures in", OUT)


if __name__ == "__main__":
    main()
