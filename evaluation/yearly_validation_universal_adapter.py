from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import (
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    WeatherLookup,
    resolve_weather_csv,
)
from evaluation.benchmark_bestest_air_article7_style import (
    BOPTESTClient,
    THERMOSTATIC_MODEL_CANDIDATES,
    ThermostaticController,
    resolve_existing_path,
)
from evaluation.block3_testcase_adapters import SUPPORTED_TESTCASES, get_adapter, make_tsup_observation, parse_common_state


DEFAULT_BOPTEST_URL = "http://web:8000"
DEFAULT_STEP_SEC = 900
DEFAULT_SCENARIO_DAYS = 14
DEFAULT_TEMP_LOW_C = 21.0
DEFAULT_TEMP_HIGH_C = 24.0
DEFAULT_TEMP_TARGET_C = 22.0

SCENARIOS = {
    "Jan_Winter": 0,
    "Feb_Winter": 2678400,
    "Mar_Spring": 5097600,
    "Apr_Spring": 7776000,
    "May_Spring": 10368000,
    "Jun_Summer": 13132800,
    "Jul_Summer": 15552000,
    "Aug_Summer": 18316800,
    "Sep_Autumn": 20995200,
    "Oct_Autumn": 23587200,
    "Nov_Autumn": 26265600,
    "Dec_Winter": 28857600,
}

DEFAULT_ADAPTER_CONFIGS = {
    "bestest_air": None,
    "bestest_hydronic_heat_pump": "configs/block3_actuator_mapping_bestest_hydronic_heat_pump.yaml",
    "bestest_hydronic": "configs/block3_actuator_mapping_bestest_hydronic.yaml",
    "singlezone_commercial_hydronic": "configs/block3_actuator_mapping_singlezone_commercial_hydronic.yaml",
}

FEATURE_PRESETS = {
    "bestest_air_article": {
        "obs_ablation": "none",
        "delta_feature_mode": "raw",
        "power_feature_mode": "raw",
        "t_zone_feature_mode": "raw",
    },
    "block3_hydronic": {
        "obs_ablation": "no_delta_t",
        "delta_feature_mode": "raw",
        "power_feature_mode": "clipped_log",
        "t_zone_feature_mode": "raw",
    },
}


def default_preset_for_testcase(testcase: str) -> str:
    return "bestest_air_article" if testcase == "bestest_air" else "block3_hydronic"


def apply_feature_preset(args: argparse.Namespace) -> str:
    preset_name = default_preset_for_testcase(args.testcase) if args.preset == "auto" else args.preset
    preset = FEATURE_PRESETS[preset_name]
    if args.obs_ablation is None:
        args.obs_ablation = preset["obs_ablation"]
    if args.delta_feature_mode is None:
        args.delta_feature_mode = preset["delta_feature_mode"]
    if args.power_feature_mode is None:
        args.power_feature_mode = preset["power_feature_mode"]
    if args.t_zone_feature_mode is None:
        args.t_zone_feature_mode = preset["t_zone_feature_mode"]
    return preset_name


def default_output_dir(testcase: str, controller: str, preset_name: str) -> str:
    return f"outputs/universal_validation/{testcase}/{controller}_{preset_name}"


def compute_metrics(
    temps: np.ndarray,
    powers: np.ndarray,
    *,
    step_sec: int,
    temp_low: float,
    temp_high: float,
    temp_target: float,
) -> dict[str, float]:
    errors = temps - temp_target
    below = temps < temp_low
    above = temps > temp_high
    violation = below | above
    r_time = float(np.mean(violation))
    over = np.where(above, (temps - temp_high) / temp_high, 0.0)
    under = np.where(below, (temp_low - temps) / temp_low, 0.0)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "within_1c_pct": float(np.mean(np.abs(errors) < 1.0) * 100.0),
        "within_05c_pct": float(np.mean(np.abs(errors) < 0.5) * 100.0),
        "viol_pct": float(r_time * 100.0),
        "energy_kwh": float(np.sum(powers) * (step_sec / 3600.0) / 1000.0),
        "ms": float(r_time + max(float(np.max(under)), float(np.max(over)))),
        "t_min": float(np.min(temps)),
        "t_max": float(np.max(temps)),
        "t_mean": float(np.mean(temps)),
    }


def load_existing_scenario(
    *,
    name: str,
    output_dir: Path,
    controller_name: str,
    scenario_days: float,
    step_sec: int,
    temp_low: float,
    temp_high: float,
    temp_target: float,
) -> dict[str, float | str] | None:
    path = output_dir / f"{controller_name}_scenario_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"t_zone_c", "p_total_w"}
    if not required.issubset(df.columns):
        print(f"  Existing {path} is missing required columns; rerunning.")
        return None
    expected_steps = int(round(float(scenario_days) * 86400 / int(step_sec)))
    if len(df) < expected_steps:
        print(f"  Existing {path} has {len(df)}/{expected_steps} rows; rerunning.")
        return None
    metrics = compute_metrics(
        df["t_zone_c"].to_numpy(dtype=float),
        df["p_total_w"].to_numpy(dtype=float),
        step_sec=int(step_sec),
        temp_low=float(temp_low),
        temp_high=float(temp_high),
        temp_target=float(temp_target),
    )
    print(f"  Reusing existing scenario {name}: {path}")
    return {"name": name, **metrics}


def run_pi_scenario(
    *,
    client: BOPTESTClient,
    name: str,
    start_time: float,
    scenario_days: float,
    step_sec: int,
    warmup_sec: float,
    temp_low: float,
    temp_high: float,
    temp_target: float,
    output_dir: Path,
) -> dict[str, float | str]:
    print("\n" + "=" * 72)
    print(f"UNIVERSAL PI SCENARIO: {name} (start={start_time:.0f}s)")
    print("=" * 72)
    testid = client.select_testcase()
    print(f"  Selected testid={testid}")
    client.initialize(testid, start_time, warmup_sec)

    rows: list[dict[str, Any]] = []
    steps = int(round(float(scenario_days) * 86400 / int(step_sec)))
    try:
        for step in range(steps):
            payload = client.advance(testid, {})
            state = parse_common_state(payload, None, rows[-1]["t_zone_c"] if rows else None)
            rows.append(
                {
                    "step": step,
                    "time_s": float(state["time"]),
                    "t_zone_c": float(state["t_zone"]),
                    "t_amb_c": float(state["t_amb"]),
                    "p_total_w": float(state["p_total_w"]),
                    "rea_t_supply_c": float(state["rea_t_supply_c"]),
                    "controller_source": "built_in_pi",
                }
            )
            if step % max(1, int(round(86400 / int(step_sec)))) == 0:
                print(f"  Step {step:4d} | T={state['t_zone']:.2f}C | P={state['p_total_w']:.0f}W")
    finally:
        client.stop(testid)

    return save_scenario_rows(
        rows=rows,
        controller_name="pi",
        name=name,
        output_dir=output_dir,
        step_sec=step_sec,
        temp_low=temp_low,
        temp_high=temp_high,
        temp_target=temp_target,
    )


def run_thermostatic_scenario(
    *,
    client: BOPTESTClient,
    controller: ThermostaticController,
    weather: WeatherLookup,
    adapter,
    name: str,
    start_time: float,
    scenario_days: float,
    step_sec: int,
    warmup_sec: float,
    obs_ablation: str,
    delta_feature_mode: str,
    t_zone_feature_mode: str,
    power_feature_mode: str,
    temp_low: float,
    temp_high: float,
    temp_target: float,
    output_dir: Path,
) -> dict[str, float | str]:
    print("\n" + "=" * 72)
    print(f"UNIVERSAL THERMOSTATIC SCENARIO: {name} (start={start_time:.0f}s)")
    print("=" * 72)
    testid = client.select_testcase()
    print(f"  Selected testid={testid}")
    client.initialize(testid, start_time, warmup_sec)

    payload = client.advance(testid, {})
    prev_action = np.zeros(2, dtype=np.float32)
    obs, state = make_tsup_observation(
        payload,
        prev_action,
        None,
        weather,
        controller.obs_dim,
        obs_ablation=obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )

    rows: list[dict[str, Any]] = []
    steps = int(round(float(scenario_days) * 86400 / int(step_sec)))
    try:
        for step in range(steps):
            action, info = controller.act(obs, state)
            action = np.asarray(action, dtype=np.float32)
            command, adapter_info = adapter.build_command(action)
            payload = client.advance(testid, command)
            next_obs, next_state = make_tsup_observation(
                payload,
                action,
                state["t_zone"],
                weather,
                controller.obs_dim,
                obs_ablation=obs_ablation,
                delta_feature_mode=delta_feature_mode,
                t_zone_feature_mode=t_zone_feature_mode,
                power_feature_mode=power_feature_mode,
            )
            rows.append(
                {
                    "step": step,
                    "time_s": float(next_state["time"]),
                    "t_zone_c": float(next_state["t_zone"]),
                    "t_amb_c": float(next_state["t_amb"]),
                    "p_total_w": float(next_state["p_total_w"]),
                    "rea_t_supply_c": float(next_state["rea_t_supply_c"]),
                    "rea_heat_power_w": float(next_state["rea_heat_power_w"]),
                    "rea_cool_power_w": float(next_state["rea_cool_power_w"]),
                    "rea_fan_power_w": float(next_state["rea_fan_power_w"]),
                    "rea_pump_power_w": float(next_state["rea_pump_power_w"]),
                    "action_a0": float(action[0]),
                    "action_a1": float(action[1]),
                    "controller_source": info.get("source", "ppo_thermostatic"),
                    **adapter_info,
                }
            )
            obs, state = next_obs, next_state
            if step % max(1, int(round(86400 / int(step_sec)))) == 0:
                print(
                    f"  Step {step:4d} | T={state['t_zone']:.2f}C | P={state['p_total_w']:.0f}W | "
                    f"h={adapter_info['adapter_heat_intensity']:.2f}"
                )
    finally:
        client.stop(testid)

    return save_scenario_rows(
        rows=rows,
        controller_name="thermostatic",
        name=name,
        output_dir=output_dir,
        step_sec=step_sec,
        temp_low=temp_low,
        temp_high=temp_high,
        temp_target=temp_target,
    )


def save_scenario_rows(
    *,
    rows: list[dict[str, Any]],
    controller_name: str,
    name: str,
    output_dir: Path,
    step_sec: int,
    temp_low: float,
    temp_high: float,
    temp_target: float,
) -> dict[str, float | str]:
    trace = pd.DataFrame(rows)
    trace_path = output_dir / f"{controller_name}_scenario_{name}.csv"
    trace.to_csv(trace_path, index=False)
    metrics = compute_metrics(
        trace["t_zone_c"].to_numpy(dtype=float),
        trace["p_total_w"].to_numpy(dtype=float),
        step_sec=step_sec,
        temp_low=temp_low,
        temp_high=temp_high,
        temp_target=temp_target,
    )
    metrics["name"] = name
    print(
        f"  RESULT: RMSE={metrics['rmse']:.2f}C | Viol={metrics['viol_pct']:.1f}% | "
        f"E={metrics['energy_kwh']:.1f}kWh | m_s={metrics['ms']:.3f}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal yearly validation runner for bestest_air and Block 3 hydronic testcases."
    )
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", DEFAULT_BOPTEST_URL))
    parser.add_argument("--testcase", "--testcase-id", dest="testcase", choices=sorted(SUPPORTED_TESTCASES), required=True)
    parser.add_argument("--controller", choices=("thermostatic", "pi"), required=True)
    parser.add_argument("--adapter-config", default=None, help="Optional testcase adapter YAML. If omitted, a default is selected for hydronic testcases.")
    parser.add_argument("--output-dir", default=None, help="Defaults to outputs/universal_validation/<testcase>/<controller>_<preset>.")
    parser.add_argument("--model", "--thermostatic-model", dest="model", default=resolve_existing_path(THERMOSTATIC_MODEL_CANDIDATES))
    parser.add_argument("--step-sec", type=int, default=DEFAULT_STEP_SEC)
    parser.add_argument("--scenario-days", type=float, default=DEFAULT_SCENARIO_DAYS)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--temp-low", type=float, default=DEFAULT_TEMP_LOW_C)
    parser.add_argument("--temp-high", type=float, default=DEFAULT_TEMP_HIGH_C)
    parser.add_argument("--temp-target", type=float, default=DEFAULT_TEMP_TARGET_C)
    parser.add_argument("--select-timeout", type=float, default=300.0)
    parser.add_argument("--advance-timeout", type=float, default=60.0)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse complete per-scenario CSVs already in output_dir.")
    parser.add_argument(
        "--preset",
        choices=("auto", "bestest_air_article", "block3_hydronic"),
        default="auto",
        help="Feature preset. auto uses bestest_air_article for bestest_air and block3_hydronic otherwise.",
    )
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default=None)
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default=None)
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default=None)
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default=None)
    args = parser.parse_args()
    resolved_preset = apply_feature_preset(args)

    output_dir_arg = args.output_dir or default_output_dir(args.testcase, args.controller, resolved_preset)
    output_dir = ROOT / output_dir_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = args.adapter_config if args.adapter_config else DEFAULT_ADAPTER_CONFIGS[args.testcase]
    adapter_path = ROOT / adapter_config if adapter_config else None
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_path}")
    adapter = get_adapter(args.testcase, adapter_path)

    client = BOPTESTClient(
        base_url=args.boptest_url,
        testcase_id=args.testcase,
        step_sec=int(args.step_sec),
        timeout_sec=float(args.advance_timeout),
        select_timeout_sec=float(args.select_timeout),
        retries=int(args.http_retries),
        backoff_base_sec=1.0,
    )

    print("Checking BOPTEST...")
    version = client.check_connectivity()
    print(f"BOPTEST version: {version.get('payload', {}).get('version', version)}")
    print(f"Testcase: {args.testcase}")
    print(f"Controller: {args.controller}")
    print(f"Output dir: {output_dir}")
    print(f"Adapter: {adapter.adapter_name}")
    print(f"Claim boundary: {adapter.transfer_claim}")
    if args.controller == "thermostatic":
        print(
            "Feature preset: "
            f"{resolved_preset} "
            f"(obs_ablation={args.obs_ablation}, power_feature_mode={args.power_feature_mode}, "
            f"t_zone_feature_mode={args.t_zone_feature_mode}, delta_feature_mode={args.delta_feature_mode})"
        )

    weather = WeatherLookup(resolve_weather_csv())
    controller = None
    if args.controller == "thermostatic":
        controller = ThermostaticController(
            args.model,
            obs_ablation=args.obs_ablation,
            delta_feature_mode=args.delta_feature_mode,
            t_zone_feature_mode=args.t_zone_feature_mode,
            power_feature_mode=args.power_feature_mode,
        )
        print(f"Model: {args.model}")

    results = []
    start_all = time.time()
    for name, start_time in SCENARIOS.items():
        if args.skip_existing:
            existing = load_existing_scenario(
                name=name,
                output_dir=output_dir,
                controller_name=args.controller,
                scenario_days=float(args.scenario_days),
                step_sec=int(args.step_sec),
                temp_low=float(args.temp_low),
                temp_high=float(args.temp_high),
                temp_target=float(args.temp_target),
            )
            if existing is not None:
                results.append(existing)
                continue

        if args.controller == "pi":
            results.append(
                run_pi_scenario(
                    client=client,
                    name=name,
                    start_time=float(start_time),
                    scenario_days=float(args.scenario_days),
                    step_sec=int(args.step_sec),
                    warmup_sec=float(args.warmup_sec),
                    temp_low=float(args.temp_low),
                    temp_high=float(args.temp_high),
                    temp_target=float(args.temp_target),
                    output_dir=output_dir,
                )
            )
        else:
            assert controller is not None
            results.append(
                run_thermostatic_scenario(
                    client=client,
                    controller=controller,
                    weather=weather,
                    adapter=adapter,
                    name=name,
                    start_time=float(start_time),
                    scenario_days=float(args.scenario_days),
                    step_sec=int(args.step_sec),
                    warmup_sec=float(args.warmup_sec),
                    obs_ablation=args.obs_ablation,
                    delta_feature_mode=args.delta_feature_mode,
                    t_zone_feature_mode=args.t_zone_feature_mode,
                    power_feature_mode=args.power_feature_mode,
                    temp_low=float(args.temp_low),
                    temp_high=float(args.temp_high),
                    temp_target=float(args.temp_target),
                    output_dir=output_dir,
                )
            )

    elapsed = (time.time() - start_all) / 60.0
    summary = pd.DataFrame(results)
    metric_cols = ["rmse", "mae", "within_1c_pct", "within_05c_pct", "viol_pct", "energy_kwh", "ms", "t_min", "t_max", "t_mean"]
    mean_row = {"name": "MEAN"}
    mean_row.update({col: float(summary[col].mean()) for col in metric_cols})
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    summary_path = output_dir / f"{args.controller}_universal_yearly_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "boptest_url": args.boptest_url,
        "testcase": args.testcase,
        "controller": args.controller,
        "output_dir": str(output_dir),
        "model": args.model if args.controller == "thermostatic" else None,
        "adapter_config": str(adapter_path) if adapter_path is not None else None,
        "adapter_name": adapter.adapter_name,
        "transfer_claim": adapter.transfer_claim,
        "step_sec": int(args.step_sec),
        "scenario_days": float(args.scenario_days),
        "temp_low": float(args.temp_low),
        "temp_high": float(args.temp_high),
        "temp_target": float(args.temp_target),
        "feature_preset": resolved_preset if args.controller == "thermostatic" else None,
        "obs_ablation": args.obs_ablation if args.controller == "thermostatic" else None,
        "delta_feature_mode": args.delta_feature_mode if args.controller == "thermostatic" else None,
        "power_feature_mode": args.power_feature_mode if args.controller == "thermostatic" else None,
        "t_zone_feature_mode": args.t_zone_feature_mode if args.controller == "thermostatic" else None,
    }
    (output_dir / f"{args.controller}_universal_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 86)
    print(f"UNIVERSAL YEARLY VALIDATION COMPLETE ({elapsed:.1f} min)")
    print("=" * 86)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
