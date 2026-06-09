from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import (
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    WeatherLookup,
    action_to_t_supply,
    build_tsup_obs,
    resolve_weather_csv,
)
from evaluation.benchmark_bestest_air_article7_style import (
    BOPTESTClient,
    THERMOSTATIC_MODEL_CANDIDATES,
    ThermostaticController,
    resolve_existing_path,
)


DEFAULT_BOPTEST_URL = "http://web:8000"
DEFAULT_TESTCASE = "bestest_hydronic_heat_pump"
DEFAULT_OUTPUT_DIR = "outputs/block3_bestest_hydronic_heat_pump/thermostatic_hybrid_l010_adapter_none"
DEFAULT_ADAPTER_CONFIG = "configs/block3_actuator_mapping_bestest_hydronic_heat_pump.yaml"
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


def k_to_c(value: float) -> float:
    value = float(value)
    return value - 273.15 if value > 200.0 else value


def get_val(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return float(value)


def get_first_val(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in payload:
            return get_val(payload, key)
    raise KeyError(f"None of the expected BOPTEST keys were found: {keys}")


def get_first_val_or_default(payload: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in payload:
            return get_val(payload, key)
    return float(default)


def sum_existing_vals(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    found = [key for key in keys if key in payload]
    if not found:
        raise KeyError(f"None of the expected BOPTEST power keys were found: {keys}")
    return float(sum(get_val(payload, key) for key in found))


def parse_hydronic_state(
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
) -> dict[str, float]:
    t_zone = k_to_c(get_first_val(payload, ("reaTZon_y", "reaTRoo_y", "reaTRooAir_y", "zon_reaTRooAir_y")))
    co2_ppm = get_first_val_or_default(payload, ("reaCO2RooAir_y", "zon_reaCO2RooAir_y"), 400.0)
    p_total_w = sum_existing_vals(
        payload,
        (
            "reaPHeaPum_y",
            "reaPFan_y",
            "reaPPumEmi_y",
            "ahu_reaPFanExt_y",
            "ahu_reaPFanSup_y",
            "reaPHea_y",
            "reaQHea_y",
            "reaPCoo_y",
            "reaPPum_y",
            "fcu_reaPCoo_y",
            "fcu_reaPFan_y",
            "fcu_reaPHea_y",
        ),
    )
    t_amb = k_to_c(get_first_val(payload, ("weaSta_reaWeaTDryBul_y", "zon_weaSta_reaWeaTDryBul_y")))
    sim_time_sec = get_val(payload, "time")
    hour = (sim_time_sec / 3600.0) % 24.0
    day = (sim_time_sec / 86400.0) % 365.0
    delta_t_zone = 0.0 if prev_t_zone is None else float(t_zone - prev_t_zone)
    return {
        "t_zone": float(t_zone),
        "co2_ppm": float(co2_ppm),
        "p_total_w": float(p_total_w),
        "t_amb": float(t_amb),
        "time": float(sim_time_sec),
        "hour": float(hour),
        "day": float(day),
        "delta_t_zone": float(delta_t_zone),
        "prev_t_supply_c": action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5,
        "rea_t_supply_c": k_to_c(get_first_val(payload, ("reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y"))) if any(k in payload for k in ("reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y")) else np.nan,
        "rea_heat_pump_power_w": get_first_val(payload, ("reaPHeaPum_y", "reaQHea_y")) if any(k in payload for k in ("reaPHeaPum_y", "reaQHea_y")) else 0.0,
        "rea_fan_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPFan_y", "ahu_reaPFanExt_y", "ahu_reaPFanSup_y")),
        "rea_pump_power_w": get_first_val(payload, ("reaPPumEmi_y", "reaPPum_y")) if any(k in payload for k in ("reaPPumEmi_y", "reaPPum_y")) else 0.0,
    }


def make_obs(
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
    weather: WeatherLookup,
    obs_dim: int,
    *,
    obs_ablation: str,
    delta_feature_mode: str,
    t_zone_feature_mode: str,
    power_feature_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    state = parse_hydronic_state(payload, prev_action, prev_t_zone)
    obs = build_tsup_obs(
        state["t_zone"],
        state["co2_ppm"],
        state["p_total_w"],
        state["prev_t_supply_c"],
        state["t_amb"],
        state["hour"],
        state["day"],
        prev_action if prev_action is not None else np.zeros(2, dtype=np.float32),
        state["delta_t_zone"],
        weather,
        include_forecast=(obs_dim == 17),
        obs_ablation=obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    return obs, state


def load_adapter_name(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("adapter_mapping", {}).get("name", "hydronic_setpoint_enable_adapter_v1"))


def hydronic_adapter_command(action: np.ndarray, adapter_name: str) -> tuple[dict[str, float], dict[str, float]]:
    policy_t_like_c = action_to_t_supply(float(action[0]))
    h = float(np.clip((policy_t_like_c - 18.0) / (35.0 - 18.0), 0.0, 1.0))
    enabled = 1.0 if h > 0.05 else 0.0
    if adapter_name == "hydronic_direct_supply_setpoint_adapter_v1":
        setpoint_k = policy_t_like_c + 273.15
        payload = {
            "oveTSetSup_activate": 1,
            "oveTSetSup_u": float(setpoint_k),
            "oveTSetHea_activate": 1,
            "oveTSetHea_u": 294.15,
            "oveTSetCoo_activate": 1,
            "oveTSetCoo_u": 297.15,
            "ovePum_activate": 1,
            "ovePum_u": enabled,
        }
        adapter_info = {
            "policy_temperature_like_command_c": float(policy_t_like_c),
            "adapter_heat_intensity": h,
            "adapter_supply_setpoint_c": float(policy_t_like_c),
            "adapter_zone_setpoint_c": np.nan,
            "adapter_plant_enabled": enabled,
        }
        return payload, adapter_info
    if adapter_name == "commercial_hydronic_supply_valve_adapter_v1":
        setpoint_k = policy_t_like_c + 273.15
        payload = {
            "dh_oveTSupSetHea_activate": 1,
            "dh_oveTSupSetHea_u": float(setpoint_k),
            "oveTZonSet_activate": 1,
            "oveTZonSet_u": 294.15,
            "oveTSupSetAir_activate": 1,
            "oveTSupSetAir_u": float(setpoint_k),
            "ovePum_activate": 1,
            "ovePum_u": 50000.0 if enabled else 0.0,
            "oveValCoi_activate": 1,
            "oveValCoi_u": enabled,
            "oveValRad_activate": 1,
            "oveValRad_u": enabled,
        }
        adapter_info = {
            "policy_temperature_like_command_c": float(policy_t_like_c),
            "adapter_heat_intensity": h,
            "adapter_supply_setpoint_c": float(policy_t_like_c),
            "adapter_zone_setpoint_c": 21.0,
            "adapter_plant_enabled": enabled,
        }
        return payload, adapter_info
    setpoint_k = 288.15 + h * (297.15 - 288.15)
    payload = {
        "oveTSet_activate": 1,
        "oveTSet_u": float(setpoint_k),
        "oveHeaPumY_activate": 1,
        "oveHeaPumY_u": enabled,
        "ovePum_activate": 1,
        "ovePum_u": enabled,
        "oveFan_activate": 1,
        "oveFan_u": enabled,
    }
    adapter_info = {
        "policy_temperature_like_command_c": float(policy_t_like_c),
        "adapter_heat_intensity": h,
        "adapter_zone_setpoint_c": float(setpoint_k - 273.15),
        "adapter_plant_enabled": enabled,
    }
    return payload, adapter_info


def compute_metrics(
    temps: np.ndarray,
    powers: np.ndarray,
    *,
    step_sec: int,
    temp_low: float,
    temp_high: float,
    temp_target: float,
) -> dict[str, float]:
    below = temps < temp_low
    above = temps > temp_high
    violation = below | above
    r_time = float(np.mean(violation))
    under = np.where(below, (temp_low - temps) / temp_low, 0.0)
    over = np.where(above, (temps - temp_high) / temp_high, 0.0)
    errors = temps - temp_target
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


def run_scenario(
    *,
    client: BOPTESTClient,
    controller: ThermostaticController,
    weather: WeatherLookup,
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
    adapter_name: str,
) -> dict[str, float | str]:
    print("\n" + "=" * 72)
    print(f"THERMOSTATIC HYDRONIC ADAPTER SCENARIO: {name} (start={start_time:.0f}s)")
    print("=" * 72)
    testid = client.select_testcase()
    print(f"  Selected testid={testid}")
    client.initialize(testid, start_time, warmup_sec)
    payload = client.advance(testid, {})
    prev_action = np.zeros(2, dtype=np.float32)
    obs, state = make_obs(
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
    steps = int(scenario_days * 86400 / step_sec)
    try:
        for step in range(steps):
            action, info = controller.act(obs, state)
            action = np.asarray(action, dtype=np.float32)
            command, adapter_info = hydronic_adapter_command(action, adapter_name)
            payload = client.advance(testid, command)
            next_obs, next_state = make_obs(
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
                    "rea_heat_pump_power_w": float(next_state["rea_heat_pump_power_w"]),
                    "rea_fan_power_w": float(next_state["rea_fan_power_w"]),
                    "rea_pump_power_w": float(next_state["rea_pump_power_w"]),
                    "action_a0": float(action[0]),
                    "action_a1": float(action[1]),
                    "controller_source": info.get("source", "ppo_thermostatic"),
                    **adapter_info,
                }
            )
            obs, state = next_obs, next_state
            prev_action = action
            if step % 96 == 0:
                print(
                    f"  Step {step:4d} | T={state['t_zone']:.2f}C | P={state['p_total_w']:.0f}W | "
                    f"h={adapter_info['adapter_heat_intensity']:.2f} | set={adapter_info['adapter_zone_setpoint_c']:.2f}C"
                )
    finally:
        client.stop(testid)

    trace = pd.DataFrame(rows)
    trace_path = output_dir / f"thermostatic_hydronic_adapter_scenario_{name}.csv"
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
        description="Yearly validation for frozen thermostatic PPO on hydronic BOPTEST through the pre-registered adapter."
    )
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", DEFAULT_BOPTEST_URL))
    parser.add_argument("--testcase-id", "--testcase", dest="testcase_id", default=DEFAULT_TESTCASE)
    parser.add_argument("--model", "--thermostatic-model", dest="model", default=resolve_existing_path(THERMOSTATIC_MODEL_CANDIDATES))
    parser.add_argument("--adapter-config", default=DEFAULT_ADAPTER_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-sec", type=int, default=DEFAULT_STEP_SEC)
    parser.add_argument("--scenario-days", type=float, default=DEFAULT_SCENARIO_DAYS)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--temp-low", type=float, default=DEFAULT_TEMP_LOW_C)
    parser.add_argument("--temp-high", type=float, default=DEFAULT_TEMP_HIGH_C)
    parser.add_argument("--temp-target", type=float, default=DEFAULT_TEMP_TARGET_C)
    parser.add_argument("--select-timeout", type=float, default=300.0)
    parser.add_argument("--advance-timeout", type=float, default=60.0)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default="no_delta_t")
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default="raw")
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default="clipped_log")
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default="raw")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = ROOT / args.adapter_config
    if not adapter_config.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config}")
    adapter_name = load_adapter_name(adapter_config)

    weather = WeatherLookup(resolve_weather_csv())
    controller = ThermostaticController(
        args.model,
        obs_ablation=args.obs_ablation,
        delta_feature_mode=args.delta_feature_mode,
        t_zone_feature_mode=args.t_zone_feature_mode,
        power_feature_mode=args.power_feature_mode,
    )
    client = BOPTESTClient(
        base_url=args.boptest_url,
        testcase_id=args.testcase_id,
        step_sec=int(args.step_sec),
        timeout_sec=float(args.advance_timeout),
        select_timeout_sec=float(args.select_timeout),
        retries=int(args.http_retries),
        backoff_base_sec=1.0,
    )
    print("Checking BOPTEST...")
    version = client.check_connectivity()
    print(f"BOPTEST version: {version.get('payload', {}).get('version', version)}")
    print(f"Model: {args.model}")
    print(f"Adapter config: {adapter_config}")
    print(f"Adapter name: {adapter_name}")

    summary_rows = []
    for name, start_time in SCENARIOS.items():
        summary_rows.append(
            run_scenario(
                client=client,
                controller=controller,
                weather=weather,
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
                adapter_name=adapter_name,
            )
        )

    summary = pd.DataFrame(summary_rows)
    cols = ["rmse", "mae", "within_1c_pct", "within_05c_pct", "viol_pct", "energy_kwh", "ms", "t_min", "t_max", "t_mean"]
    mean_row = {"name": "MEAN"}
    mean_row.update({col: float(summary[col].mean()) for col in cols})
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    summary_path = output_dir / "thermostatic_hydronic_adapter_yearly_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "boptest_url": args.boptest_url,
        "testcase_id": args.testcase_id,
        "model": args.model,
        "adapter_config": str(adapter_config),
        "step_sec": int(args.step_sec),
        "scenario_days": float(args.scenario_days),
        "temp_low": float(args.temp_low),
        "temp_high": float(args.temp_high),
        "obs_ablation": args.obs_ablation,
        "delta_feature_mode": args.delta_feature_mode,
        "power_feature_mode": args.power_feature_mode,
        "t_zone_feature_mode": args.t_zone_feature_mode,
        "adapter_name": adapter_name,
        "transfer_claim": "direct supply-setpoint transfer" if adapter_name == "hydronic_direct_supply_setpoint_adapter_v1" else "adapter-mediated, not literal direct-TSup transfer",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 86)
    print("THERMOSTATIC HYDRONIC ADAPTER YEARLY VALIDATION COMPLETE")
    print("=" * 86)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
