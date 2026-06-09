"""Build paper figures from real CSV artifacts only.

The v3 prepared-rollout CSV is expected at
outputs/surrogate_v3_rollout_prepared_15min/v3/all_full_rollouts.csv.
Generate it with evaluation/validate_surrogate_v3_rollout_prepared.py if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "reports" / "figures" / "article_real"
MANIFEST_DEFAULT = ROOT / "reports" / "article_real_figures_manifest.csv"

COLORS = {
    "v3": "#2f5d8c",
    "raw_v35": "#b25f2c",
    "calibrated_v35": "#21867a",
    "hybrid_l010": "#6f4e7c",
    "pi": "#333333",
    "morl": "#c44e52",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save(fig: plt.Figure, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def manifest(rows: list[dict], figure: str, status: str, sources: list[Path], note: str, png: Path | None, pdf: Path | None) -> None:
    rows.append(
        {
            "figure": figure,
            "status": status,
            "sources": " | ".join(rel(p) for p in sources),
            "note": note,
            "png": rel(png) if png else "",
            "pdf": rel(pdf) if pdf else "",
        }
    )


def style_ax(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, weight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    ymax = ax.get_ylim()[1]
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                h + ymax * 0.015,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )


def load_surrogate_rollouts() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rollout_paths = {
        "v3": ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "all_full_rollouts.csv",
        "raw_v35": ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "all_full_rollouts.csv",
        "calibrated_v35": ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "all_full_rollouts.csv",
    }
    horizon_paths = {
        "v3": ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "horizon_metrics.csv",
        "raw_v35": ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "horizon_metrics.csv",
        "calibrated_v35": ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "horizon_metrics.csv",
    }
    return ({k: read_csv(v) for k, v in rollout_paths.items()}, {k: read_csv(v) for k, v in horizon_paths.items()})


def common_episode(rollouts: dict[str, pd.DataFrame], preferred: str = "bestest_air_article7_style_15min__peak_heat_window_thermostatic") -> str:
    common = set.intersection(*(set(df["episode_id"].unique()) for df in rollouts.values()))
    if preferred in common:
        return preferred
    return sorted(common)[0]


def fig1_replicative(out_dir: Path, rows: list[dict]) -> None:
    rollouts, horizons = load_surrogate_rollouts()
    sources = [
        ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "all_full_rollouts.csv",
    ]
    labels = ["v3", "raw v3.5", "calibrated v3.5"]
    keys = ["v3", "raw_v35", "calibrated_v35"]
    temp_rmse = [np.sqrt(np.mean(rollouts[k]["temp_error_c"].to_numpy(float) ** 2)) for k in keys]
    power_mae = [np.mean(np.abs(rollouts[k]["power_error_w"].to_numpy(float))) for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    axes[0].bar(labels, temp_rmse, color=[COLORS[k] for k in keys], edgecolor="#222222", linewidth=0.5)
    style_ax(axes[0], "Recursive 15-min temperature error", ylabel="RMSE_T (C)")
    annotate_bars(axes[0], "{:.2f}")
    axes[1].bar(labels, power_mae, color=[COLORS[k] for k in keys], edgecolor="#222222", linewidth=0.5)
    style_ax(axes[1], "Recursive 15-min power error", ylabel="MAE_P (W)")
    annotate_bars(axes[1], "{:.0f}")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Block 1. Surrogate fidelity from full prepared-rollout CSVs", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block1_replicative_validity_bars")
    manifest(rows, "block1_replicative_validity_bars", "complete", sources, "v3, raw_v35, and calibrated_v35 are computed from full rollout CSVs.", png, pdf)


def fig2_predictive(out_dir: Path, rows: list[dict]) -> None:
    _, horizons = load_surrogate_rollouts()
    sources = [
        ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "horizon_metrics.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "horizon_metrics.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "horizon_metrics.csv",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.25), sharex=True)
    display = {"v3": "v3", "raw_v35": "raw v3.5", "calibrated_v35": "calibrated v3.5"}
    for key, df in horizons.items():
        df = df.sort_values("horizon_h")
        x = df["horizon_h"].to_numpy(float)
        y = df["temp_rmse_c"].to_numpy(float)
        axes[0].plot(x, y, marker="o", linewidth=2.4, color=COLORS[key], label=display[key])
        if {"temp_rmse_ci_low", "temp_rmse_ci_high"}.issubset(df.columns):
            axes[0].fill_between(x, df["temp_rmse_ci_low"], df["temp_rmse_ci_high"], color=COLORS[key], alpha=0.14)
        axes[1].plot(x, df["temp_mae_c"], marker="s", linewidth=2.1, color=COLORS[key], label=display[key])
    style_ax(axes[0], "Rollout RMSE by horizon", "horizon (h)", "RMSE_T (C)")
    style_ax(axes[1], "Rollout MAE by horizon", "horizon (h)", "MAE_T (C)")
    axes[0].set_xticks([1, 4, 8, 24])
    axes[0].legend(frameon=False)
    fig.suptitle("Block 1. Predictive validity on the same prepared 15-min traces", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block1_predictive_validity_horizon_lines")
    manifest(rows, "block1_predictive_validity_horizon_lines", "complete", sources, "All three models use their own real horizon_metrics.csv.", png, pdf)


def fig3_trace(out_dir: Path, rows: list[dict]) -> None:
    rollouts, _ = load_surrogate_rollouts()
    cal = rollouts["calibrated_v35"]

    def scenario_of(episode_id: str) -> str:
        if "peak_heat_window" in episode_id:
            return "peak_heat_window"
        if "typical_heat_window" in episode_id:
            return "typical_heat_window"
        return "other"

    def per_episode_rmse(df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for episode_id, group in df.groupby("episode_id"):
            sub = group.sort_values("step").head(96)
            err = sub["pred_t_zone"].to_numpy(float) - sub["actual_t_zone"].to_numpy(float)
            records.append(
                {
                    "episode_id": episode_id,
                    "scenario": scenario_of(str(episode_id)),
                    "rmse_c": float(np.sqrt(np.mean(err**2))),
                    "bias_c": float(np.mean(err)),
                }
            )
        return pd.DataFrame(records)

    rmse_df = per_episode_rmse(cal)

    def median_episode_for(scenario: str) -> tuple[str, float, float, int]:
        subset = rmse_df[rmse_df["scenario"] == scenario].copy().sort_values(["rmse_c", "episode_id"]).reset_index(drop=True)
        if subset.empty:
            raise RuntimeError(f"No calibrated-v35 rollout episodes found for scenario: {scenario}")
        selected = subset.iloc[(len(subset) - 1) // 2]
        return str(selected["episode_id"]), float(selected["rmse_c"]), float(selected["bias_c"]), len(subset)

    selected = {
        "peak_heat_window": median_episode_for("peak_heat_window"),
        "typical_heat_window": median_episode_for("typical_heat_window"),
    }

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.0], hspace=0.34, wspace=0.22)
    trace_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    dist_ax = fig.add_subplot(gs[1, :])
    model_labels = [("v3", "v3"), ("raw_v35", "raw v3.5"), ("calibrated_v35", "calibrated v3.5")]

    for ax, scenario, title in [
        (trace_axes[0], "peak_heat_window", "Median peak-heat 24h rollout"),
        (trace_axes[1], "typical_heat_window", "Median typical-heat 24h rollout"),
    ]:
        episode, rmse_c, bias_c, count = selected[scenario]
        base = rollouts["v3"][rollouts["v3"]["episode_id"] == episode].sort_values("step").head(96)
        hours = np.arange(len(base)) * 0.25
        ax.axhspan(21, 24, color="#d8ead7", alpha=0.8, label="comfort 21-24 C")
        ax.plot(hours, base["actual_t_zone"], color="#111111", linewidth=2.5, label="BOPTEST")
        for key, label in model_labels:
            sub = rollouts[key][rollouts[key]["episode_id"] == episode].sort_values("step").head(96)
            ax.plot(hours[: len(sub)], sub["pred_t_zone"], linewidth=2.0, color=COLORS[key], label=label)
        style_ax(ax, title, "hours", "zone temperature (C)")
        ax.text(
            0.02,
            0.04,
            f"programmatic lower median of {count} episodes\ncal. v3.5 RMSE={rmse_c:.2f} C, bias={bias_c:+.2f} C",
            transform=ax.transAxes,
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
        )
    trace_axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    rmse_df = rmse_df.sort_values(["scenario", "rmse_c", "episode_id"]).reset_index(drop=True)
    bar_colors = [
        COLORS["calibrated_v35"] if ep in {selected["peak_heat_window"][0], selected["typical_heat_window"][0]} else "#b7c3c0"
        for ep in rmse_df["episode_id"]
    ]
    x = np.arange(len(rmse_df))
    dist_ax.bar(x, rmse_df["rmse_c"], color=bar_colors, edgecolor="#333333", linewidth=0.4)
    aggregate_rmse = float(np.sqrt(np.mean(cal["temp_error_c"].to_numpy(float) ** 2)))
    dist_ax.axhline(aggregate_rmse, color="#111111", linestyle="--", linewidth=1.4, label=f"aggregate RMSE={aggregate_rmse:.2f} C")
    for scenario, marker in [("peak_heat_window", "P"), ("typical_heat_window", "T")]:
        episode = selected[scenario][0]
        idx = int(rmse_df.index[rmse_df["episode_id"] == episode][0])
        dist_ax.text(idx, rmse_df.loc[idx, "rmse_c"] + 0.05, marker, ha="center", va="bottom", fontsize=9, weight="bold")
    short_labels = [
        ("peak" if row.scenario == "peak_heat_window" else "typ") + f" {i+1}"
        for i, row in enumerate(rmse_df.itertuples())
    ]
    dist_ax.set_xticks(x, short_labels, rotation=0)
    style_ax(dist_ax, "Calibrated v3.5 per-episode 24h RMSE distribution", "held-out episode", "RMSE_T (C)")
    dist_ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Block 1. Programmatic 24h rollout realism: median traces plus RMSE distribution", fontsize=14, weight="bold")
    sources = [
        ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "all_full_rollouts.csv",
    ]
    png, pdf = save(fig, out_dir, "block1_rollout_24h_temperature_trace")
    manifest(
        rows,
        "block1_rollout_24h_temperature_trace",
        "complete",
        sources,
        "Peak and typical traces are selected programmatically as calibrated-v3.5 lower-median RMSE episodes; bottom panel shows all per-episode RMSE values.",
        png,
        pdf,
    )


def fig4_residuals(out_dir: Path, rows: list[dict]) -> None:
    rollouts, _ = load_surrogate_rollouts()
    sources = [
        ROOT / "outputs" / "surrogate_v3_rollout_prepared_15min" / "v3" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "raw_v35" / "all_full_rollouts.csv",
        ROOT / "outputs" / "surrogate_v35_rollout_prepared_15min_power_head_only" / "calibrated_v35" / "all_full_rollouts.csv",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    bins = np.linspace(-4, 4, 80)
    for key, label in [("v3", "v3"), ("raw_v35", "raw v3.5"), ("calibrated_v35", "calibrated v3.5")]:
        temp = rollouts[key]["temp_error_c"].dropna().to_numpy(float)
        power = rollouts[key]["power_error_w"].dropna().to_numpy(float)
        axes[0].hist(temp, bins=bins, density=True, histtype="step", linewidth=2.2, color=COLORS[key], label=label)
        axes[0].axvline(np.median(temp), color=COLORS[key], linestyle="--", linewidth=1.2)
        axes[1].boxplot(
            np.abs(power),
            positions=[{"v3": 1, "raw_v35": 2, "calibrated_v35": 3}[key]],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": COLORS[key], "alpha": 0.35, "edgecolor": COLORS[key]},
            medianprops={"color": "#111111"},
        )
    axes[0].axvline(0, color="#111111", linewidth=1)
    style_ax(axes[0], "Temperature residual distribution", "predicted - actual (C)", "density")
    style_ax(axes[1], "Absolute power residual distribution", ylabel="|error| (W)")
    axes[1].set_xticks([1, 2, 3], ["v3", "raw v3.5", "cal. v3.5"], rotation=10)
    axes[0].legend(frameon=False)
    png, pdf = save(fig, out_dir, "block1_temperature_residual_histograms")
    manifest(rows, "block1_temperature_residual_histograms", "complete", sources, "Residuals are computed from all three full rollout CSVs.", png, pdf)


def fig4b_stage_abc(out_dir: Path, rows: list[dict]) -> None:
    stage_b_path = ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_episodeaware" / "stage_b_history_v35.csv"
    stage_c_path = ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_episodeaware" / "stage_c_history_v35.csv"
    summary_path = ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_episodeaware" / "calibration_summary_boptest_v35.json"
    stage_b = read_csv(stage_b_path)
    stage_c = read_csv(stage_c_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    axes[0].plot(stage_b["epoch"], stage_b["c_zon_j_per_k"] / 1e5, color="#21867a", linewidth=2.5)
    axes[0].axhline(stage_b["c_zon_j_per_k"].iloc[0] / 1e5, color="#777777", linestyle="--", linewidth=1.2, label="initial")
    axes[0].axhline(stage_b["c_zon_j_per_k"].iloc[-1] / 1e5, color="#111111", linestyle=":", linewidth=1.4, label="identified")
    style_ax(axes[0], "Stage B identifies C_zon", "epoch", "C_zon (1e5 J/K)")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(stage_c["epoch"], stage_c["val_rmse_temp"], color="#2f5d8c", linewidth=2.2, label="one-step val RMSE")
    if "rollout_rmse_val" in stage_c.columns:
        axes[1].plot(stage_c["epoch"], stage_c["rollout_rmse_val"], color="#b25f2c", linewidth=2.0, label="rollout val RMSE")
    best_idx = int(stage_c["val_rmse_temp"].astype(float).idxmin())
    axes[1].scatter([stage_c.loc[best_idx, "epoch"]], [stage_c.loc[best_idx, "val_rmse_temp"]], color="#111111", s=45, zorder=5)
    style_ax(axes[1], "Stage C residual-head refinement", "epoch", "RMSE_T (C)")
    axes[1].legend(frameon=False)

    fig.suptitle("Block 1. Stage A/B/C calibration diagnostics", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block1_stage_abc_calibration_diagnostics")
    manifest(
        rows,
        "block1_stage_abc_calibration_diagnostics",
        "complete",
        [stage_b_path, stage_c_path, summary_path],
        "Stage B C_zon convergence and Stage C residual-head validation curves from real calibration histories.",
        png,
        pdf,
    )


def fig5_warmstart(out_dir: Path, rows: list[dict]) -> None:
    scratch_summary_path = ROOT / "outputs" / "block2_thermostatic_warmstart_utility" / "scratch_eval" / "summary.csv"
    warm_summary_path = ROOT / "outputs" / "block2_thermostatic_warmstart_utility" / "warmstart_eval" / "summary.csv"
    scratch_trace_path = ROOT / "outputs" / "block2_thermostatic_warmstart_utility" / "scratch_eval" / "traces" / "peak_heat_window_thermostatic.csv"
    warm_trace_path = ROOT / "outputs" / "block2_thermostatic_warmstart_utility" / "warmstart_eval" / "traces" / "peak_heat_window_thermostatic.csv"
    scratch_summary = read_csv(scratch_summary_path)
    warm_summary = read_csv(warm_summary_path)
    scratch_trace = read_csv(scratch_trace_path)
    warm_trace = read_csv(warm_trace_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1.0]})
    h = np.arange(min(96, len(scratch_trace), len(warm_trace))) * 0.25
    axes[0].axhspan(21, 24, color="#d8ead7", alpha=0.8)
    axes[0].plot(h, scratch_trace["t_zone_c"].head(len(h)), color="#2f5d8c", linewidth=2, label="scratch")
    axes[0].plot(h, warm_trace["t_zone_c"].head(len(h)), color="#b25f2c", linewidth=2, label="v3.5 warm-start")
    style_ax(axes[0], "Peak-heat live trace after fine-tune", "hours", "zone temperature (C)")
    axes[0].legend(frameon=False)
    summary = pd.concat([scratch_summary.assign(mode="scratch"), warm_summary.assign(mode="warm-start")], ignore_index=True)
    peak = summary[summary["scenario"] == "peak_heat_window"]
    x = np.arange(3)
    width = 0.35
    metrics = [("m_s", "m_s"), ("violation_pct", "viol %"), ("energy_kwh", "kWh")]
    for i, mode in enumerate(["scratch", "warm-start"]):
        vals = [float(peak[peak["mode"] == mode][col].iloc[0]) for col, _ in metrics]
        axes[1].bar(x + (i - 0.5) * width, vals, width=width, label=mode, color=["#2f5d8c", "#b25f2c"][i])
    axes[1].set_xticks(x, [label for _, label in metrics])
    style_ax(axes[1], "Peak-heat KPIs", ylabel="raw metric")
    axes[1].legend(frameon=False)
    png, pdf = save(fig, out_dir, "block2_warmstart_negative_eval_kpis")
    manifest(rows, "block2_warmstart_negative_eval_kpis", "complete", [scratch_summary_path, warm_summary_path, scratch_trace_path, warm_trace_path], "Uses real scratch/warm-start BOPTEST eval traces and summaries; no training reward logs were available.", png, pdf)


def fig6_thermostatic(out_dir: Path, rows: list[dict]) -> None:
    pure_path = ROOT / "outputs" / "bestest_air_article7_style_15min" / "summary.csv"
    hybrid_path = ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l010" / "summary.csv"
    pure = read_csv(pure_path)
    pure = pure[pure["controller"] == "thermostatic"].assign(model="pure v3")
    hybrid = read_csv(hybrid_path).assign(model="hybrid l010")
    df = pd.concat([pure, hybrid], ignore_index=True)
    scenarios = ["peak_heat_window", "typical_heat_window"]
    metrics = [("m_s", "m_s", "{:.3f}"), ("violation_pct", "violation (%)", "{:.1f}"), ("energy_kwh", "energy (kWh)", "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, (col, title, form) in zip(axes, metrics):
        x = np.arange(len(scenarios))
        for i, model in enumerate(["pure v3", "hybrid l010"]):
            vals = [float(df[(df["scenario"] == s) & (df["model"] == model)][col].iloc[0]) for s in scenarios]
            ax.bar(x + (i - 0.5) * 0.36, vals, width=0.36, color=[COLORS["v3"], COLORS["hybrid_l010"]][i], label=model, edgecolor="#222", linewidth=0.4)
        ax.set_xticks(x, ["peak", "typical"])
        style_ax(ax, title)
        annotate_bars(ax, form)
    axes[0].legend(frameon=False)
    fig.suptitle("Block 2. Thermostatic PPO transfer to live BOPTEST", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block2_thermostatic_pure_v3_vs_hybrid_kpis")
    manifest(rows, "block2_thermostatic_pure_v3_vs_hybrid_kpis", "complete", [pure_path, hybrid_path], "Pure v3 and hybrid_l010 values come from live BOPTEST summary.csv files.", png, pdf)


def fig7_hdrl_lambda(out_dir: Path, rows: list[dict]) -> None:
    df = read_csv(ROOT / "reports" / "block2_hdrl_lambda_sweep_summary.csv")
    df["lambda_temp"] = df["variant"].map({"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    for scenario, sub in df.sort_values("lambda_temp").groupby("scenario"):
        label = scenario.replace("_", " ")
        axes[0].plot(sub["lambda_temp"], sub["m_s"], marker="o", linewidth=2.4, label=label)
        axes[1].plot(sub["lambda_temp"], sub["violation_pct"], marker="o", linewidth=2.4, label=label)
    style_ax(axes[0], "Safety metric worsens with lambda_temp", "lambda_temp", "m_s")
    style_ax(axes[1], "Comfort violations worsen with lambda_temp", "lambda_temp", "violation (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Block 2. HDRL sensitivity to temperature-physics regularization", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block2_hdrl_lambda_sweep_sensitivity")
    manifest(rows, "block2_hdrl_lambda_sweep_sensitivity", "complete", [ROOT / "reports" / "block2_hdrl_lambda_sweep_summary.csv"], "Real HDRL lambda sweep.", png, pdf)


def fig8_hdrl_trace(out_dir: Path, rows: list[dict]) -> None:
    path = ROOT / "outputs" / "block2_hdrl_hybrid_v3_v35_l000" / "traces" / "peak_heat_window_hdrl.csv"
    df = read_csv(path).head(96)
    h = np.arange(len(df)) * 0.25
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.axhspan(21, 24, color="#d8ead7", alpha=0.75, label="comfort 21-24 C")
    ax.plot(h, df["t_zone_c"], color="#111111", linewidth=2.3, label="zone temperature")
    ax2 = ax.twinx()
    ax2.plot(h, df["t_supply_cmd_c"], color="#b25f2c", linewidth=1.9, linestyle="--", label="supply command")
    style_ax(ax, "HDRL lambda_temp=0 live winter day", "hours", "zone temperature (C)")
    ax2.set_ylabel("supply command (C)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    png, pdf = save(fig, out_dir, "block2_hdrl_l000_winter_tracking")
    manifest(rows, "block2_hdrl_l000_winter_tracking", "complete", [path], "Trace contains actual zone temperature and commanded supply; explicit high-level setpoint is not logged.", png, pdf)


def fig9_morl_pareto(out_dir: Path, rows: list[dict]) -> None:
    pareto_path = ROOT / "reports" / "morl_pareto_front_table.csv"
    summary_path = ROOT / "reports" / "morl_canonical_seedfix_yearly_summary.csv"
    df = read_csv(pareto_path)
    pareto = df[(df["kind"] == "morl_pareto") & (df["complete"].astype(str).str.lower() == "true")].copy()
    seed42_points = pareto[pareto["canonical_designation"].eq("pareto_point")].sort_values("w_comfort")
    canonical = pareto[pareto["canonical_designation"].isin(["pre_registered", "practical_deployment"])].copy()
    reference = df[df["kind"].eq("morl_reference")].copy()
    tcrit_95_df4 = 2.7764451051977987
    canonical_stats = []
    for label, sub in canonical.groupby("label", sort=False):
        n = len(sub)
        designation = str(sub["canonical_designation"].iloc[0])
        canonical_stats.append(
            {
                "label": label,
                "designation": designation,
                "w_comfort": float(sub["w_comfort"].mean()),
                "w_energy": float(sub["w_energy"].mean()),
                "energy_mean": float(sub["energy_kwh_mean"].mean()),
                "energy_ci95": float(tcrit_95_df4 * sub["energy_kwh_mean"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "ms_mean": float(sub["ms_mean"].mean()),
                "ms_ci95": float(tcrit_95_df4 * sub["ms_mean"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "n": n,
            }
        )
    canonical_stats = pd.DataFrame(canonical_stats).sort_values("w_comfort")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.15), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax, zoom = axes
    front_x = pd.concat([seed42_points["energy_kwh_mean"], canonical_stats["energy_mean"]], ignore_index=True)
    front_y = pd.concat([seed42_points["ms_mean"], canonical_stats["ms_mean"]], ignore_index=True)
    front_w = pd.concat([seed42_points["w_comfort"], canonical_stats["w_comfort"]], ignore_index=True)
    front = pd.DataFrame({"energy": front_x, "ms": front_y, "w": front_w}).sort_values("w")
    sc = ax.scatter(seed42_points["energy_kwh_mean"], seed42_points["ms_mean"], c=seed42_points["w_comfort"], cmap="viridis", s=95, marker="o", edgecolor="#222", linewidth=0.8, label="single-seed sweep (seed42)")
    ax.scatter(reference["energy_kwh_mean"], reference["ms_mean"], c=reference["w_comfort"], cmap="viridis", s=105, marker="D", edgecolor="#222", linewidth=0.8, label="legacy seed42 reference")
    ax.plot(front["energy"], front["ms"], color="#777", linewidth=1.1, alpha=0.55)
    zoom.scatter(seed42_points["energy_kwh_mean"], seed42_points["ms_mean"], c=seed42_points["w_comfort"], cmap="viridis", s=105, marker="o", edgecolor="#222", linewidth=0.8, label="single-seed sweep (seed42)")
    zoom.scatter(reference["energy_kwh_mean"], reference["ms_mean"], c=reference["w_comfort"], cmap="viridis", s=105, marker="D", edgecolor="#222", linewidth=0.8, label="legacy seed42 reference")
    zoom.plot(front["energy"], front["ms"], color="#777", linewidth=1.1, alpha=0.55)
    for axis in (ax, zoom):
        for _, r in canonical_stats.iterrows():
            color = "#0072B2" if r["designation"] == "pre_registered" else "#D55E00"
            axis.errorbar(
                r["energy_mean"],
                r["ms_mean"],
                xerr=r["energy_ci95"],
                yerr=r["ms_ci95"],
                fmt="s",
                markersize=8.5,
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=4,
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                label=f"N=5 canonical {r['w_comfort']:.2f}/{r['w_energy']:.2f} (95% CI)" if axis is ax else None,
                zorder=5,
            )
    base = df[df["kind"] == "baseline"]
    if not base.empty:
        ax.scatter(base["energy_kwh_mean"], base["ms_mean"], marker="X", s=140, color=COLORS["pi"], label="PI baseline")
        zoom.scatter(base["energy_kwh_mean"], base["ms_mean"], marker="X", s=120, color=COLORS["pi"], label="PI baseline")
    if not seed42_points.empty:
        ax.annotate("energy-only\ncollapse", (seed42_points.iloc[0]["energy_kwh_mean"], seed42_points.iloc[0]["ms_mean"]), xytext=(15, -5), textcoords="offset points", fontsize=8)
    ax.annotate("N=5 canonical\nuncertainty", (canonical_stats.iloc[-1]["energy_mean"], canonical_stats.iloc[-1]["ms_mean"]), xytext=(-130, 42), textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.8})
    label_offsets = {
        "comfort_025_energy_075": (-10, 12),
        "comfort_050_energy_050": (8, 4),
        "comfort_075_energy_025": (-52, -2),
        "comfort_100_energy_000": (-48, -18),
    }
    for _, r in pd.concat([seed42_points, reference], ignore_index=True).query("w_energy < 1.0").iterrows():
        dx, dy = label_offsets.get(str(r["label"]), (6, 6))
        zoom.annotate(
            f"{r['w_comfort']:.2f}/{r['w_energy']:.2f}",
            (r["energy_kwh_mean"], r["ms_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5},
        )
    for _, r in canonical_stats.iterrows():
        dx, dy = label_offsets.get(str(r["label"]), (8, 4))
        zoom.annotate(
            f"{r['w_comfort']:.2f}/{r['w_energy']:.2f}\nN={int(r['n'])}",
            (r["energy_mean"], r["ms_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5},
        )
    ax.set_xlim(-12, 280)
    ax.set_ylim(-0.06, max(1.72, float(seed42_points["ms_mean"].max()) * 1.05))
    zoom.set_xlim(214, 268)
    zoom.set_ylim(-0.02, 0.34)
    style_ax(ax, "Full front incl. safety collapse", "mean monthly energy (kWh)", "m_s, lower is better")
    style_ax(zoom, "Zoom: practical operating region", "mean monthly energy (kWh)", "m_s")
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
    cb.set_label("comfort preference weight")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("MORL yearly Pareto front with N=5 canonical uncertainty", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block2_morl_pareto_energy_vs_ms")
    manifest(rows, "block2_morl_pareto_energy_vs_ms", "complete", [pareto_path, summary_path], "Seed42 sweep points plus N=5 canonical mean +/- 95% CI from seedfix runs; PI baseline included.", png, pdf)


def fig10_morl_radar(out_dir: Path, rows: list[dict]) -> None:
    comparison_path = ROOT / "reports" / "block2_morl_comparison_summary.csv"
    pareto_path = ROOT / "reports" / "morl_pareto_front_table.csv"
    df = read_csv(comparison_path)
    pareto = read_csv(pareto_path)
    pi = pareto[pareto["label"] == "pi_yearly_builtin"]
    if not pi.empty:
        pi_row = pi.iloc[0]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "variant": "PI_baseline",
                            "obs_path": "boptest_builtin",
                            "backend": "boptest",
                            "lambda_temp_disagree": np.nan,
                            "lambda_power_disagree": np.nan,
                            "rmse_c": float(pi_row["rmse_mean"]),
                            "mae_c": float(pi_row["mae_mean"]),
                            "within_1c_pct": float(pi_row["within_1c_pct_mean"]),
                            "within_05c_pct": float(pi_row["within_05c_pct_mean"]),
                            "violation_pct": float(pi_row["violation_pct_mean"]),
                            "energy_kwh": float(pi_row["energy_kwh_mean"]),
                            "m_s": float(pi_row["ms_mean"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    metrics = [
        ("rmse_c", "Tracking\nscore"),
        ("mae_c", "Error\nscore"),
        ("violation_pct", "Comfort\nscore"),
        ("m_s", "Safety\nscore"),
        ("energy_kwh", "Energy\nefficiency"),
    ]
    scores = []
    for _, r in df.iterrows():
        row = {"variant": r["variant"]}
        for col, label in metrics:
            vals = df[col].astype(float)
            lo, hi = vals.min(), vals.max()
            row[label] = 1.0 if hi == lo else 1.0 - (float(r[col]) - lo) / (hi - lo)
        scores.append(row)
    labels = [m[1] for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = plt.subplot(111, polar=True)
    color_map = {
        "MORL_5D_basic": "#c44e52",
        "MORL_17D_power_only": "#2f5d8c",
        "PI_baseline": "#666666",
    }
    fill_alpha = {
        "MORL_5D_basic": 0.10,
        "MORL_17D_power_only": 0.18,
        "PI_baseline": 0.06,
    }
    for row in scores:
        color = color_map.get(row["variant"], "#888888")
        vals = [row[label] for label in labels] + [row[labels[0]]]
        ax.plot(angles, vals, linewidth=2.6, color=color, label=row["variant"])
        ax.fill(angles, vals, color=color, alpha=fill_alpha.get(row["variant"], 0.10))
    ax.set_xticks(angles[:-1], labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_title("MORL observation interface ablation\nmin-max KPI scores, outer ring = best", fontsize=12, weight="bold", pad=20)
    ax.text(
        0.5,
        -0.12,
        "Scores use lower-is-better min-max normalization for RMSE, MAE, violation, m_s, and energy.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), frameon=False)
    png, pdf = save(fig, out_dir, "block2_morl_5d_vs_17d_radar")
    manifest(rows, "block2_morl_5d_vs_17d_radar", "complete", [comparison_path, pareto_path], "Real 5D/17D MORL metrics plus PI baseline; all axes are min-max scores with 1.0 as best.", png, pdf)


def fig11_morl_heatmap(out_dir: Path, rows: list[dict]) -> None:
    df = read_csv(ROOT / "outputs" / "morl_hybrid_v3_v35_power_only_17d" / "seed42" / "yearly_eval" / "morl_yearly_summary.csv")
    months = df[~df["name"].str.upper().eq("MEAN")].copy()
    labels = [str(x).split("_")[0] for x in months["name"]]
    mat = np.vstack([months["viol_pct"].to_numpy(float), months["energy_kwh"].to_numpy(float), months["ms"].to_numpy(float)])
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 4.6), sharex=True)
    cmaps = ["Reds", "YlOrBr", "Purples"]
    row_names = ["violation (%)", "energy (kWh)", "m_s"]
    for i, ax in enumerate(axes):
        im = ax.imshow(mat[i : i + 1], aspect="auto", cmap=cmaps[i])
        ax.set_yticks([0], [row_names[i]])
        for j, val in enumerate(mat[i]):
            ax.text(j, 0, f"{val:.1f}" if i < 2 else f"{val:.2f}", ha="center", va="center", fontsize=8, color="#111")
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    axes[-1].set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    fig.suptitle("MORL 17D yearly seasonal validation", fontsize=14, weight="bold")
    png, pdf = save(fig, out_dir, "block2_morl_17d_seasonal_heatmap")
    manifest(rows, "block2_morl_17d_seasonal_heatmap", "complete", [ROOT / "outputs" / "morl_hybrid_v3_v35_power_only_17d" / "seed42" / "yearly_eval" / "morl_yearly_summary.csv"], "Real monthly MORL 17D yearly validation table.", png, pdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.facecolor": "white", "figure.facecolor": "white"})
    rows: list[dict] = []
    fig1_replicative(args.output_dir, rows)
    fig2_predictive(args.output_dir, rows)
    fig3_trace(args.output_dir, rows)
    fig4_residuals(args.output_dir, rows)
    fig4b_stage_abc(args.output_dir, rows)
    fig5_warmstart(args.output_dir, rows)
    fig6_thermostatic(args.output_dir, rows)
    fig7_hdrl_lambda(args.output_dir, rows)
    fig8_hdrl_trace(args.output_dir, rows)
    fig9_morl_pareto(args.output_dir, rows)
    fig10_morl_radar(args.output_dir, rows)
    fig11_morl_heatmap(args.output_dir, rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(f"Saved figures to: {args.output_dir}")
    print(f"Saved manifest to: {args.manifest}")


if __name__ == "__main__":
    main()
