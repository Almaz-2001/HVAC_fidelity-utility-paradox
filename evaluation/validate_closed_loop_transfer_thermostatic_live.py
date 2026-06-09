"""
evaluation/validate_closed_loop_transfer_thermostatic_live.py

Live closed-loop transfer-gap validation for the thermostatic PPO policy.

The script runs the same thermostatic controller in parallel on:
1. live BOPTEST
2. the calibrated direct-TSup surrogate backend

Both rollouts share the same scenario schedule and exogenous weather timeline.
This isolates the closed-loop transfer gap between the real simulator and the
current calibrated twin.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import (
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    build_tsup_obs,
    resolve_weather_csv,
)
from evaluation.benchmark_bestest_air_article7_style import (
    BOPTESTClient,
    T_HIGH,
    T_LOW,
    THERMOSTATIC_MODEL_CANDIDATES,
    ThermostaticController,
    build_bestest_air_command,
    compute_safety_metrics,
    derive_article7_style_scenarios,
    make_tsup_obs,
    resolve_existing_path,
)
from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

DEFAULT_SUMMARY = "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json"
DEFAULT_OUT_DIR = "outputs/block_1_3_closed_loop_transfer_thermostatic_live"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate thermostatic PPO closed-loop transfer gap between live BOPTEST and the calibrated v3.5 twin."
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
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default="none")
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default="raw")
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default="raw")
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default="raw")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


class SurrogateThermostaticRollout:
    def __init__(
        self,
        *,
        surrogate_kind: str,
        legacy_model_path: str | None,
        summary_json: str,
        checkpoint_path: str | None,
        base_model_path: str | None,
        step_sec: int,
        device: torch.device,
    ) -> None:
        self.step_sec = int(step_sec)
        self.device = device
        self.adapter = load_direct_tsup_adapter(
            kind=surrogate_kind,
            legacy_model_path=legacy_model_path,
            summary_json=summary_json,
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            device=device,
            runtime_step_sec=float(step_sec),
        )
        self.state: dict[str, float] | None = None

    def reset(self, initial_state: dict[str, float]) -> None:
        self.state = {
            "t_zone": float(initial_state["t_zone"]),
            "co2_ppm": float(initial_state["co2_ppm"]),
            "p_total_w": float(initial_state["p_total_w"]),
            "t_amb": float(initial_state["t_amb"]),
            "time": float(initial_state["time"]),
            "hour": float(initial_state["hour"]),
            "day": float(initial_state["day"]),
            "delta_t_zone": float(initial_state["delta_t_zone"]),
        }

    def build_obs(
        self,
        prev_action: np.ndarray,
        obs_dim: int,
        weather: WeatherLookup,
        obs_ablation: str = "none",
        delta_feature_mode: str = "raw",
        t_zone_feature_mode: str = "raw",
        power_feature_mode: str = "raw",
    ) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Surrogate rollout must be reset before calling build_obs.")
        prev_t_supply_c = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5
        return build_tsup_obs(
            self.state["t_zone"],
            self.state["co2_ppm"],
            self.state["p_total_w"],
            prev_t_supply_c,
            self.state["t_amb"],
            self.state["hour"],
            self.state["day"],
            np.asarray(prev_action, dtype=np.float32),
            self.state["delta_t_zone"],
            weather,
            include_forecast=(obs_dim == 17),
            obs_ablation=obs_ablation,
            delta_feature_mode=delta_feature_mode,
            t_zone_feature_mode=t_zone_feature_mode,
            power_feature_mode=power_feature_mode,
        )

    def step(self, action: np.ndarray, next_exogenous_state: dict[str, float]) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Surrogate rollout must be reset before stepping.")
        a0 = float(action[0])
        a1 = float(action[1])
        with torch.no_grad():
            t_next, p_total = self.adapter(
                torch.tensor([self.state["t_zone"]], dtype=torch.float32, device=self.device),
                torch.tensor([self.state["t_amb"]], dtype=torch.float32, device=self.device),
                torch.tensor([self.state["hour"]], dtype=torch.float32, device=self.device),
                torch.tensor([self.state["day"]], dtype=torch.float32, device=self.device),
                torch.tensor([a0], dtype=torch.float32, device=self.device),
                torch.tensor([a1], dtype=torch.float32, device=self.device),
            )
        t_next_c = float(t_next[0].detach().cpu())
        p_total_w = float(p_total[0].detach().cpu())
        fan_u = action_to_fan(a1)
        prev_t_zone = float(self.state["t_zone"])
        self.state = {
            "t_zone": t_next_c,
            "co2_ppm": float(np.clip(self.state["co2_ppm"] - 50.0 * fan_u + 10.0, 400.0, 2000.0)),
            "p_total_w": p_total_w,
            "t_amb": float(next_exogenous_state["t_amb"]),
            "time": float(next_exogenous_state["time"]),
            "hour": float(next_exogenous_state["hour"]),
            "day": float(next_exogenous_state["day"]),
            "delta_t_zone": float(t_next_c - prev_t_zone),
        }
        return dict(self.state)


def _variant_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": df["step"],
            "sim_time_sec": df["sim_time_sec"],
            "t_zone_c": df[f"{prefix}_t_zone_c"],
            "p_total_w": df[f"{prefix}_p_total_w"],
        }
    )


def _compare_metrics(df: pd.DataFrame, step_sec: int) -> dict[str, float]:
    temp_err = df["surrogate_t_zone_c"].to_numpy(dtype=float) - df["boptest_t_zone_c"].to_numpy(dtype=float)
    power_err = df["surrogate_p_total_w"].to_numpy(dtype=float) - df["boptest_p_total_w"].to_numpy(dtype=float)
    bop_metrics = compute_safety_metrics(_variant_frame(df, "boptest"), step_sec)
    surr_metrics = compute_safety_metrics(_variant_frame(df, "surrogate"), step_sec)
    return {
        "temp_rmse_c": float(np.sqrt(np.mean(temp_err ** 2))),
        "temp_bias_c": float(np.mean(temp_err)),
        "power_rmse_w": float(np.sqrt(np.mean(power_err ** 2))),
        "power_bias_w": float(np.mean(power_err)),
        "action_rmse": float(
            np.sqrt(
                np.mean(
                    (df["surrogate_a0"].to_numpy(dtype=float) - df["boptest_a0"].to_numpy(dtype=float)) ** 2
                    + (df["surrogate_a1"].to_numpy(dtype=float) - df["boptest_a1"].to_numpy(dtype=float)) ** 2
                )
            )
        ),
        "boptest_m_s": float(bop_metrics["m_s"]),
        "surrogate_m_s": float(surr_metrics["m_s"]),
        "ms_gap": float(surr_metrics["m_s"] - bop_metrics["m_s"]),
        "boptest_violation_pct": float(bop_metrics["violation_pct"]),
        "surrogate_violation_pct": float(surr_metrics["violation_pct"]),
        "violation_gap_pct": float(surr_metrics["violation_pct"] - bop_metrics["violation_pct"]),
        "boptest_energy_kwh": float(bop_metrics["energy_kwh"]),
        "surrogate_energy_kwh": float(surr_metrics["energy_kwh"]),
        "energy_gap_kwh": float(surr_metrics["energy_kwh"] - bop_metrics["energy_kwh"]),
    }


def plot_transfer_trace(trace_df: pd.DataFrame, title: str, out_path: Path) -> None:
    x_days = trace_df["elapsed_days"].to_numpy(dtype=float)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    ax1.fill_between(x_days, T_LOW, T_HIGH, color="#dff3e4", alpha=0.7, label="comfort band")
    ax1.plot(x_days, trace_df["boptest_t_zone_c"], label="BOPTEST", linewidth=2.0, color="#1f77b4")
    ax1.plot(x_days, trace_df["surrogate_t_zone_c"], label="Surrogate v3.5", linewidth=1.8, color="#d62728")
    ax1.set_ylabel("Zone temperature, C")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncol=3)

    ax2.plot(x_days, trace_df["boptest_p_total_w"], label="BOPTEST", linewidth=2.0, color="#1f77b4")
    ax2.plot(x_days, trace_df["surrogate_p_total_w"], label="Surrogate v3.5", linewidth=1.8, color="#d62728")
    ax2.set_ylabel("HVAC power, W")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, ncol=2)

    ax3.plot(x_days, trace_df["boptest_a0"], label="BOPTEST a0", linewidth=1.5, color="#1f77b4")
    ax3.plot(x_days, trace_df["surrogate_a0"], label="Surrogate a0", linewidth=1.5, color="#d62728")
    ax3.plot(x_days, trace_df["boptest_a1"], label="BOPTEST a1", linewidth=1.5, linestyle="--", color="#2ca02c")
    ax3.plot(x_days, trace_df["surrogate_a1"], label="Surrogate a1", linewidth=1.5, linestyle="--", color="#9467bd")
    ax3.set_ylabel("Action")
    ax3.set_xlabel("Elapsed time, days")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, ncol=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_transfer_scenario(
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
) -> tuple[pd.DataFrame, dict[str, float]]:
    total_steps = int(scenario.duration_days * 86400 / step_sec)
    testid = client.select_testcase()
    client.initialize(testid, scenario.start_time_sec, warmup_sec)
    payload = client.advance(testid, {})
    bop_obs, bop_state = make_tsup_obs(
        payload,
        np.zeros(2, dtype=np.float32),
        None,
        weather,
        controller.obs_dim,
        obs_ablation=controller.obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    surrogate_rollout.reset(bop_state)

    prev_action_bop = np.zeros(2, dtype=np.float32)
    prev_action_surr = np.zeros(2, dtype=np.float32)
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

            elapsed_days = (float(next_bop_state["time"]) - float(scenario.start_time_sec)) / 86400.0
            rows.append(
                {
                    "step": step,
                    "sim_time_sec": float(next_bop_state["time"]),
                    "elapsed_days": float(elapsed_days),
                    "t_amb_c": float(next_bop_state["t_amb"]),
                    "boptest_t_zone_c": float(next_bop_state["t_zone"]),
                    "surrogate_t_zone_c": float(next_surr_state["t_zone"]),
                    "boptest_p_total_w": float(next_bop_state["p_total_w"]),
                    "surrogate_p_total_w": float(next_surr_state["p_total_w"]),
                    "boptest_a0": float(bop_action[0]),
                    "boptest_a1": float(bop_action[1]),
                    "surrogate_a0": float(surr_action[0]),
                    "surrogate_a1": float(surr_action[1]),
                    "boptest_t_supply_cmd_c": action_to_t_supply(float(bop_action[0])),
                    "surrogate_t_supply_cmd_c": action_to_t_supply(float(surr_action[0])),
                    "boptest_fan_cmd_u": action_to_fan(float(bop_action[1])),
                    "surrogate_fan_cmd_u": action_to_fan(float(surr_action[1])),
                }
            )

            bop_obs, bop_state = next_bop_obs, next_bop_state
            surr_obs = next_surr_obs
            prev_action_bop = np.asarray(bop_action, dtype=np.float32)
            prev_action_surr = np.asarray(surr_action, dtype=np.float32)
    finally:
        client.stop(testid)

    trace_df = pd.DataFrame(rows)
    metrics = _compare_metrics(trace_df, step_sec)
    metrics.update(
        {
            "scenario": str(scenario.name),
            "label": str(scenario.label),
            "start_day_index": int(scenario.start_day_index),
            "duration_days": int(scenario.duration_days),
            "step_sec": int(step_sec),
        }
    )
    return trace_df, metrics


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    traces_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    weather_csv = resolve_weather_csv()
    scenarios = derive_article7_style_scenarios(
        weather_csv=weather_csv,
        duration_days=args.duration_days,
        heating_threshold_c=args.heating_threshold_c,
    )
    weather = WeatherLookup(weather_csv)
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
        device=torch.device("cpu"),
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
    summary_rows: list[dict[str, float]] = []

    for scenario in scenarios:
        print("\n" + "=" * 88)
        print(f"CLOSED-LOOP TRANSFER SCENARIO: {scenario.name} | {scenario.label}")
        print("=" * 88)
        trace_df, metrics = run_transfer_scenario(
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
        trace_path = traces_dir / f"{scenario.name}_thermostatic_transfer.csv"
        trace_df.to_csv(trace_path, index=False)
        plot_transfer_trace(
            trace_df=trace_df,
            title=f"{scenario.label} | thermostatic closed-loop transfer gap",
            out_path=output_dir / f"{scenario.name}_thermostatic_transfer.png",
        )
        summary_rows.append(metrics)
        print(
            f"  temp_rmse={metrics['temp_rmse_c']:.3f} C | power_rmse={metrics['power_rmse_w']:.1f} W | "
            f"m_s gap={metrics['ms_gap']:.4f} | violation gap={metrics['violation_gap_pct']:.1f}% | "
            f"energy gap={metrics['energy_gap_kwh']:.2f} kWh"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    manifest = {
        "boptest_url": args.boptest_url,
        "testcase_id": args.testcase_id,
        "step_sec": int(args.step_sec),
        "duration_days": int(args.duration_days),
        "thermostatic_model": args.thermostatic_model,
        "surrogate_kind": args.surrogate_kind,
        "surrogate_legacy_model": args.surrogate_legacy_model,
        "summary_json": args.summary_json,
        "weather_csv": weather_csv,
        "scenarios": [asdict(s) for s in scenarios],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("THERMOSTATIC CLOSED-LOOP TRANSFER VALIDATION COMPLETE")
    print("=" * 88)
    if not summary_df.empty:
        print(summary_df.to_string(index=False, justify="center"))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {output_dir / 'manifest.json'}")
    print(f"  {traces_dir}")


if __name__ == "__main__":
    main()
