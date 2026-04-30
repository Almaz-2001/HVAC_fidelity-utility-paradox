"""
evaluation/diagnose_thermostatic_obs_transfer_gap.py

Diagnose the thermostatic closed-loop transfer gap at the observation level.

For each benchmark scenario, this script runs one live BOPTEST rollout and one
surrogate rollout in parallel, logs:
  - BOPTEST observation vector
  - surrogate observation vector
  - action chosen from each observation

Outputs:
  - per-step observation/action trace CSVs
  - per-feature drift summary table
  - first-divergence summary showing which feature first dominated the policy gap
  - compact plots for feature drift and action-gap timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import (
    EXTENDED_TSUP_OBS_DIM,
    FORECAST_HORIZONS,
    NO_FORECAST_TSUP_OBS_DIM,
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    resolve_weather_csv,
)
from evaluation.benchmark_bestest_air_article7_style import (
    BOPTESTClient,
    THERMOSTATIC_MODEL_CANDIDATES,
    ThermostaticController,
    build_bestest_air_command,
    derive_article7_style_scenarios,
    make_tsup_obs,
    resolve_existing_path,
)
from evaluation.validate_closed_loop_transfer_thermostatic_live import (
    DEFAULT_SUMMARY,
    SurrogateThermostaticRollout,
)

DEFAULT_OUT_DIR = "outputs/block_1_3_thermostatic_obs_transfer_gap"
HISTORY_FEATURES = ("t_supply_prev_norm", "prev_a0", "prev_a1", "delta_t_zone_norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose BOPTEST-vs-surrogate observation drift for the thermostatic PPO policy."
    )
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--testcase-id", default="bestest_air")
    parser.add_argument("--thermostatic-model", default=resolve_existing_path(THERMOSTATIC_MODEL_CANDIDATES))
    parser.add_argument("--surrogate-kind", default="v35_calibrated")
    parser.add_argument("--surrogate-legacy-model", default=None)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--surrogate-checkpoint", default=None)
    parser.add_argument("--surrogate-base-model", default=None)
    parser.add_argument("--step-sec", type=int, default=900)
    parser.add_argument("--duration-days", type=int, default=14)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--heating-threshold-c", type=float, default=12.0)
    parser.add_argument("--action-gap-threshold", type=float, default=0.25)
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default="none")
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default="raw")
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default="raw")
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default="raw")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def obs_feature_names(obs_dim: int) -> list[str]:
    base = [
        "t_zone_norm",
        "co2_norm",
        "p_total_norm",
        "t_supply_prev_norm",
        "t_amb_norm",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]
    history = ["prev_a0", "prev_a1", "delta_t_zone_norm"]
    if obs_dim == EXTENDED_TSUP_OBS_DIM:
        forecast = [f"forecast_t_amb_{h}h_norm" for h in FORECAST_HORIZONS]
        return [*base, *forecast, *history]
    if obs_dim == NO_FORECAST_TSUP_OBS_DIM:
        return [*base, *history]
    raise ValueError(f"Unsupported thermostatic observation dim: {obs_dim}")


def action_gap_norm(bop_action: np.ndarray, surr_action: np.ndarray) -> float:
    diff = np.asarray(surr_action, dtype=float) - np.asarray(bop_action, dtype=float)
    return float(np.sqrt(np.sum(diff ** 2)))


def run_obs_diagnostic_scenario(
    *,
    client: BOPTESTClient,
    controller: ThermostaticController,
    surrogate_rollout: SurrogateThermostaticRollout,
    scenario: Any,
    weather: WeatherLookup,
    warmup_sec: float,
    step_sec: int,
    delta_feature_mode: str,
    t_zone_feature_mode: str,
    power_feature_mode: str,
) -> pd.DataFrame:
    total_steps = int(scenario.duration_days * 86400 / step_sec)
    feature_names = obs_feature_names(controller.obs_dim)
    testid = client.select_testcase()
    client.initialize(testid, scenario.start_time_sec, warmup_sec)
    payload = client.advance(testid, {})

    prev_action_bop = np.zeros(2, dtype=np.float32)
    prev_action_surr = np.zeros(2, dtype=np.float32)
    bop_obs, bop_state = make_tsup_obs(
        payload,
        prev_action_bop,
        None,
        weather,
        controller.obs_dim,
        obs_ablation=controller.obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    surrogate_rollout.reset(bop_state)
    surr_obs = surrogate_rollout.build_obs(
        prev_action_surr,
        controller.obs_dim,
        weather,
        obs_ablation=controller.obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )

    rows: list[dict[str, Any]] = []
    try:
        for step in range(total_steps):
            bop_action, _ = controller.act(bop_obs, bop_state)
            surr_action = np.asarray(controller.model.predict(surr_obs, deterministic=True)[0], dtype=np.float32)
            gap = action_gap_norm(bop_action, surr_action)
            next_payload = client.advance(testid, build_bestest_air_command(np.asarray(bop_action, dtype=np.float32)))
            next_bop_obs, next_bop_state = make_tsup_obs(
                next_payload,
                np.asarray(bop_action, dtype=np.float32),
                bop_state["t_zone"],
                weather,
                controller.obs_dim,
                obs_ablation=controller.obs_ablation,
                delta_feature_mode=delta_feature_mode,
                t_zone_feature_mode=t_zone_feature_mode,
                power_feature_mode=power_feature_mode,
            )
            next_surr_state = surrogate_rollout.step(surr_action, next_bop_state)
            next_surr_obs = surrogate_rollout.build_obs(
                surr_action,
                controller.obs_dim,
                weather,
                obs_ablation=controller.obs_ablation,
                delta_feature_mode=delta_feature_mode,
                t_zone_feature_mode=t_zone_feature_mode,
                power_feature_mode=power_feature_mode,
            )

            row = {
                "scenario": str(scenario.name),
                "label": str(scenario.label),
                "step": int(step),
                "sim_time_sec": float(next_bop_state["time"]),
                "elapsed_days": float((float(next_bop_state["time"]) - float(scenario.start_time_sec)) / 86400.0),
                "t_amb_c": float(next_bop_state["t_amb"]),
                "boptest_t_zone_c": float(next_bop_state["t_zone"]),
                "surrogate_t_zone_c": float(next_surr_state["t_zone"]),
                "boptest_p_total_w": float(next_bop_state["p_total_w"]),
                "surrogate_p_total_w": float(next_surr_state["p_total_w"]),
                "boptest_a0": float(bop_action[0]),
                "boptest_a1": float(bop_action[1]),
                "surrogate_a0": float(surr_action[0]),
                "surrogate_a1": float(surr_action[1]),
                "action_gap_norm": float(gap),
                "boptest_t_supply_cmd_c": action_to_t_supply(float(bop_action[0])),
                "surrogate_t_supply_cmd_c": action_to_t_supply(float(surr_action[0])),
                "boptest_fan_cmd_u": action_to_fan(float(bop_action[1])),
                "surrogate_fan_cmd_u": action_to_fan(float(surr_action[1])),
            }
            for idx, name in enumerate(feature_names):
                bop_val = float(bop_obs[idx])
                surr_val = float(surr_obs[idx])
                row[f"boptest_{name}"] = bop_val
                row[f"surrogate_{name}"] = surr_val
                row[f"drift_{name}"] = float(surr_val - bop_val)
                row[f"abs_drift_{name}"] = float(abs(surr_val - bop_val))
            rows.append(row)

            bop_obs, bop_state = next_bop_obs, next_bop_state
            surr_obs = next_surr_obs
            prev_action_bop = np.asarray(bop_action, dtype=np.float32)
            prev_action_surr = np.asarray(surr_action, dtype=np.float32)
    finally:
        client.stop(testid)

    return pd.DataFrame(rows)


def summarize_feature_drift(trace_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, group in trace_df.groupby("scenario", sort=False):
        for name in feature_names:
            diff = group[f"drift_{name}"].to_numpy(dtype=float)
            rows.append(
                {
                    "scenario": str(scenario_name),
                    "feature": name,
                    "bias": float(np.mean(diff)),
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff ** 2))),
                    "max_abs": float(np.max(np.abs(diff))),
                }
            )
    summary_df = pd.DataFrame(rows)
    overall = (
        summary_df.groupby("feature", as_index=False)[["bias", "mae", "rmse", "max_abs"]]
        .mean()
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )
    overall["scenario"] = "overall"
    return pd.concat([summary_df, overall], ignore_index=True)


def summarize_selected_feature_drift(feature_summary_df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    return feature_summary_df[feature_summary_df["feature"].isin(selected_features)].copy().reset_index(drop=True)


def summarize_first_divergence(
    trace_df: pd.DataFrame,
    feature_names: list[str],
    action_gap_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, group in trace_df.groupby("scenario", sort=False):
        mask = group["action_gap_norm"].to_numpy(dtype=float) >= float(action_gap_threshold)
        if not np.any(mask):
            rows.append(
                {
                    "scenario": str(scenario_name),
                    "first_divergence_step": None,
                    "first_divergence_day": None,
                    "action_gap_norm": None,
                    "top_feature": None,
                    "top_feature_abs_drift": None,
                }
            )
            continue
        idx = int(np.flatnonzero(mask)[0])
        row = group.iloc[idx]
        top_feature = max(feature_names, key=lambda name: abs(float(row[f"drift_{name}"])))
        rows.append(
            {
                "scenario": str(scenario_name),
                "first_divergence_step": int(row["step"]),
                "first_divergence_day": float(row["elapsed_days"]),
                "action_gap_norm": float(row["action_gap_norm"]),
                "top_feature": top_feature,
                "top_feature_abs_drift": float(abs(row[f"drift_{top_feature}"])),
            }
        )
    return pd.DataFrame(rows)


def summarize_first_divergence_focus(
    trace_df: pd.DataFrame,
    focus_features: list[str],
    action_gap_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, group in trace_df.groupby("scenario", sort=False):
        mask = group["action_gap_norm"].to_numpy(dtype=float) >= float(action_gap_threshold)
        if not np.any(mask):
            row = {
                "scenario": str(scenario_name),
                "first_divergence_step": None,
                "first_divergence_day": None,
                "action_gap_norm": None,
                "top_focus_feature": None,
                "top_focus_feature_abs_drift": None,
            }
            for name in focus_features:
                row[f"abs_drift_{name}"] = None
            rows.append(row)
            continue
        idx = int(np.flatnonzero(mask)[0])
        focus_row = group.iloc[idx]
        top_feature = max(focus_features, key=lambda name: abs(float(focus_row[f"drift_{name}"])))
        row = {
            "scenario": str(scenario_name),
            "first_divergence_step": int(focus_row["step"]),
            "first_divergence_day": float(focus_row["elapsed_days"]),
            "action_gap_norm": float(focus_row["action_gap_norm"]),
            "top_focus_feature": top_feature,
            "top_focus_feature_abs_drift": float(abs(focus_row[f"drift_{top_feature}"])),
        }
        for name in focus_features:
            row[f"abs_drift_{name}"] = float(abs(focus_row[f"drift_{name}"]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_feature_drift_heatmap(feature_summary_df: pd.DataFrame, out_path: Path) -> None:
    heat_df = feature_summary_df[feature_summary_df["scenario"] != "overall"].pivot(
        index="scenario",
        columns="feature",
        values="mae",
    )
    if heat_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(heat_df.columns)), 4.5))
    im = ax.imshow(heat_df.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    ax.set_title("Thermostatic observation drift MAE: surrogate vs BOPTEST")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_focus_feature_bars_at_first_divergence(
    trace_df: pd.DataFrame,
    scenario_name: str,
    focus_features: list[str],
    action_gap_threshold: float,
    out_path: Path,
) -> None:
    df = trace_df[trace_df["scenario"] == scenario_name].copy()
    if df.empty:
        return
    mask = df["action_gap_norm"].to_numpy(dtype=float) >= float(action_gap_threshold)
    if not np.any(mask):
        return
    row = df.iloc[int(np.flatnonzero(mask)[0])]
    labels = list(focus_features)
    values = [float(abs(row[f"drift_{name}"])) for name in focus_features]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#ff7f0e")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("|obs_surrogate - obs_boptest|")
    ax.set_title(
        f"History drift at first action divergence: {scenario_name}\n"
        f"step={int(row['step'])}, action_gap={float(row['action_gap_norm']):.3f}"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_action_gap_trace(trace_df: pd.DataFrame, scenario_name: str, out_path: Path) -> None:
    df = trace_df[trace_df["scenario"] == scenario_name].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["elapsed_days"], df["action_gap_norm"], linewidth=1.8, color="#d62728")
    ax.set_title(f"Thermostatic action-gap norm over time: {scenario_name}")
    ax.set_xlabel("Elapsed time, days")
    ax.set_ylabel("||a_surrogate - a_boptest||")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_first_divergence_top_features(
    trace_df: pd.DataFrame,
    scenario_name: str,
    feature_names: list[str],
    action_gap_threshold: float,
    out_path: Path,
) -> None:
    df = trace_df[trace_df["scenario"] == scenario_name].copy()
    if df.empty:
        return
    mask = df["action_gap_norm"].to_numpy(dtype=float) >= float(action_gap_threshold)
    if not np.any(mask):
        return
    row = df.iloc[int(np.flatnonzero(mask)[0])]
    top_items = sorted(
        ((name, float(abs(row[f"drift_{name}"]))) for name in feature_names),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    labels = [item[0] for item in top_items]
    values = [item[1] for item in top_items]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(labels)), values, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("|obs_surrogate - obs_boptest|")
    ax.set_title(
        f"Top feature drifts at first action divergence: {scenario_name}\n"
        f"step={int(row['step'])}, action_gap={float(row['action_gap_norm']):.3f}"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    traces_dir = output_dir / "traces"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    weather_csv = resolve_weather_csv()
    weather = WeatherLookup(weather_csv)
    scenarios = derive_article7_style_scenarios(
        weather_csv=weather_csv,
        duration_days=args.duration_days,
        heating_threshold_c=args.heating_threshold_c,
    )
    controller = ThermostaticController(
        args.thermostatic_model,
        obs_ablation=args.obs_ablation,
        delta_feature_mode=args.delta_feature_mode,
        t_zone_feature_mode=args.t_zone_feature_mode,
        power_feature_mode=args.power_feature_mode,
    )
    surrogate_rollout = SurrogateThermostaticRollout(
        surrogate_kind=args.surrogate_kind,
        legacy_model_path=args.surrogate_legacy_model,
        summary_json=args.summary_json,
        checkpoint_path=args.surrogate_checkpoint,
        base_model_path=args.surrogate_base_model,
        step_sec=int(args.step_sec),
        device="cpu",
    )
    client = BOPTESTClient(
        base_url=args.boptest_url,
        testcase_id=args.testcase_id,
        step_sec=int(args.step_sec),
        timeout_sec=60.0,
        select_timeout_sec=300.0,
        retries=3,
        backoff_base_sec=1.0,
    )
    version_payload = client.check_connectivity()
    print(f"[BOPTEST] Connected: {version_payload}")

    feature_names = obs_feature_names(controller.obs_dim)
    history_features = [name for name in HISTORY_FEATURES if name in feature_names]
    all_traces: list[pd.DataFrame] = []
    for scenario in scenarios:
        print("\n" + "=" * 88)
        print(f"OBS TRANSFER DIAGNOSTIC: {scenario.name} | {scenario.label}")
        print("=" * 88)
        trace_df = run_obs_diagnostic_scenario(
            client=client,
            controller=controller,
            surrogate_rollout=surrogate_rollout,
            scenario=scenario,
            weather=weather,
            warmup_sec=float(args.warmup_sec),
            step_sec=int(args.step_sec),
            delta_feature_mode=args.delta_feature_mode,
            t_zone_feature_mode=args.t_zone_feature_mode,
            power_feature_mode=args.power_feature_mode,
        )
        trace_path = traces_dir / f"{scenario.name}_obs_action_trace.csv"
        trace_df.to_csv(trace_path, index=False)
        plot_action_gap_trace(trace_df, str(scenario.name), plots_dir / f"{scenario.name}_action_gap_trace.png")
        plot_first_divergence_top_features(
            trace_df,
            str(scenario.name),
            feature_names,
            float(args.action_gap_threshold),
            plots_dir / f"{scenario.name}_first_divergence_top_features.png",
        )
        plot_focus_feature_bars_at_first_divergence(
            trace_df,
            str(scenario.name),
            history_features,
            float(args.action_gap_threshold),
            plots_dir / f"{scenario.name}_first_divergence_history_features.png",
        )
        print(f"  steps={len(trace_df)} | saved={trace_path}")
        all_traces.append(trace_df)

    combined_df = pd.concat(all_traces, ignore_index=True)
    feature_summary_df = summarize_feature_drift(combined_df, feature_names)
    history_summary_df = summarize_selected_feature_drift(feature_summary_df, history_features)
    first_div_df = summarize_first_divergence(combined_df, feature_names, float(args.action_gap_threshold))
    first_div_history_df = summarize_first_divergence_focus(
        combined_df,
        history_features,
        float(args.action_gap_threshold),
    )
    feature_summary_path = output_dir / "feature_drift_summary.csv"
    history_summary_path = output_dir / "history_feature_drift_summary.csv"
    first_div_path = output_dir / "first_divergence_summary.csv"
    first_div_history_path = output_dir / "first_divergence_history_summary.csv"
    combined_path = output_dir / "combined_obs_action_trace.csv"
    combined_df.to_csv(combined_path, index=False)
    feature_summary_df.to_csv(feature_summary_path, index=False)
    history_summary_df.to_csv(history_summary_path, index=False)
    first_div_df.to_csv(first_div_path, index=False)
    first_div_history_df.to_csv(first_div_history_path, index=False)
    plot_feature_drift_heatmap(feature_summary_df, plots_dir / "feature_drift_heatmap.png")
    plot_feature_drift_heatmap(history_summary_df, plots_dir / "history_feature_drift_heatmap.png")

    manifest = {
        "boptest_url": args.boptest_url,
        "testcase_id": args.testcase_id,
        "thermostatic_model": args.thermostatic_model,
        "surrogate_kind": args.surrogate_kind,
        "surrogate_legacy_model": args.surrogate_legacy_model,
        "summary_json": args.summary_json,
        "step_sec": int(args.step_sec),
        "duration_days": int(args.duration_days),
        "action_gap_threshold": float(args.action_gap_threshold),
        "obs_dim": int(controller.obs_dim),
        "delta_feature_mode": args.delta_feature_mode,
        "t_zone_feature_mode": args.t_zone_feature_mode,
        "power_feature_mode": args.power_feature_mode,
        "feature_names": feature_names,
        "history_features": history_features,
        "weather_csv": weather_csv,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("THERMOSTATIC OBS TRANSFER DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print("\nTop overall feature drift (MAE):")
    overall = feature_summary_df[feature_summary_df["scenario"] == "overall"].head(10)
    if not overall.empty:
        print(overall[["feature", "mae", "rmse", "bias", "max_abs"]].to_string(index=False, justify="center"))
    print("\nFirst divergence summary:")
    print(first_div_df.to_string(index=False, justify="center"))
    print("\nFirst divergence history summary:")
    print(first_div_history_df.to_string(index=False, justify="center"))
    print("\nSaved:")
    print(f"  {combined_path}")
    print(f"  {feature_summary_path}")
    print(f"  {first_div_path}")
    print(f"  {output_dir / 'manifest.json'}")
    print(f"  {plots_dir}")


if __name__ == "__main__":
    main()
