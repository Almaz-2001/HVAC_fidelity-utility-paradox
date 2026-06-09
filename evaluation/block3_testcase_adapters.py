from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from envs.tsup_features import WeatherLookup, action_to_fan, action_to_t_supply, build_tsup_obs


COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0

SUPPORTED_TESTCASES = frozenset(
    {
        "bestest_air",
        "bestest_hydronic_heat_pump",
        "bestest_hydronic",
        "singlezone_commercial_hydronic",
    }
)


@dataclass(frozen=True)
class TestcaseAdapter:
    testcase: str
    adapter_name: str
    transfer_claim: str
    build_command: Callable[[np.ndarray], tuple[dict[str, float], dict[str, float]]]


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


def parse_common_state(
    payload: dict[str, Any],
    prev_action: np.ndarray | None = None,
    prev_t_zone: float | None = None,
) -> dict[str, float]:
    t_zone = k_to_c(get_first_val(payload, ("zon_reaTRooAir_y", "reaTZon_y", "reaTRoo_y", "reaTRooAir_y")))
    co2_ppm = get_first_val_or_default(payload, ("zon_reaCO2RooAir_y", "reaCO2RooAir_y"), 400.0)
    p_total_w = sum_existing_vals(
        payload,
        (
            "fcu_reaPCoo_y",
            "fcu_reaPFan_y",
            "fcu_reaPHea_y",
            "reaPHeaPum_y",
            "reaPFan_y",
            "reaPPumEmi_y",
            "ahu_reaPFanExt_y",
            "ahu_reaPFanSup_y",
            "reaPHea_y",
            "reaQHea_y",
            "reaPCoo_y",
            "reaPPum_y",
        ),
    )
    t_amb = k_to_c(get_first_val(payload, ("zon_weaSta_reaWeaTDryBul_y", "weaSta_reaWeaTDryBul_y", "reaWeaTDryBul_y")))
    sim_time_sec = get_val(payload, "time")
    delta_t_zone = 0.0 if prev_t_zone is None else float(t_zone - prev_t_zone)
    return {
        "t_zone": float(t_zone),
        "co2_ppm": float(co2_ppm),
        "p_total_w": float(p_total_w),
        "t_amb": float(t_amb),
        "time": float(sim_time_sec),
        "hour": float((sim_time_sec / 3600.0) % 24.0),
        "day": float((sim_time_sec / 86400.0) % 365.0),
        "delta_t_zone": float(delta_t_zone),
        "prev_t_supply_c": action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5,
        "rea_t_supply_c": k_to_c(get_first_val(payload, ("reaTSup_y", "fcu_reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y", "ahu_reaTSupAir_y"))) if any(k in payload for k in ("reaTSup_y", "fcu_reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y", "ahu_reaTSupAir_y")) else np.nan,
        "rea_heat_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPHeaPum_y", "reaPHea_y", "reaQHea_y", "fcu_reaPHea_y")),
        "rea_cool_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPCoo_y", "fcu_reaPCoo_y")),
        "rea_fan_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPFan_y", "fcu_reaPFan_y", "ahu_reaPFanExt_y", "ahu_reaPFanSup_y")),
        "rea_pump_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPPumEmi_y", "reaPPum_y")),
    }


def make_tsup_observation(
    payload: dict[str, Any],
    prev_action: np.ndarray | None,
    prev_t_zone: float | None,
    weather: WeatherLookup,
    obs_dim: int,
    *,
    obs_ablation: str,
    delta_feature_mode: str,
    t_zone_feature_mode: str,
    power_feature_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    state = parse_common_state(payload, prev_action, prev_t_zone)
    prev = prev_action if prev_action is not None else np.zeros(2, dtype=np.float32)
    obs = build_tsup_obs(
        state["t_zone"],
        state["co2_ppm"],
        state["p_total_w"],
        state["prev_t_supply_c"],
        state["t_amb"],
        state["hour"],
        state["day"],
        prev,
        state["delta_t_zone"],
        weather,
        include_forecast=(obs_dim == 17),
        obs_ablation=obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    return obs, state


def _intensity(policy_t_like_c: float) -> float:
    return float(np.clip((policy_t_like_c - 18.0) / (35.0 - 18.0), 0.0, 1.0))


def _info(policy_t_like_c: float, h: float, *, supply_c: float | None, zone_c: float | None, enabled: float) -> dict[str, float]:
    return {
        "policy_temperature_like_command_c": float(policy_t_like_c),
        "adapter_heat_intensity": float(h),
        "adapter_supply_setpoint_c": float("nan") if supply_c is None else float(supply_c),
        "adapter_zone_setpoint_c": float("nan") if zone_c is None else float(zone_c),
        "adapter_plant_enabled": float(enabled),
    }


def build_bestest_air_command(action: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    t_supply_c = action_to_t_supply(float(action[0]))
    fan_u = action_to_fan(float(action[1]))
    payload = {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": float(fan_u),
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": float(t_supply_c + 273.15),
    }
    return payload, _info(t_supply_c, _intensity(t_supply_c), supply_c=t_supply_c, zone_c=None, enabled=fan_u)


def build_hydronic_heat_pump_command(action: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    policy_t_like_c = action_to_t_supply(float(action[0]))
    h = _intensity(policy_t_like_c)
    enabled = 1.0 if h > 0.05 else 0.0
    zone_setpoint_c = 15.0 + h * 9.0
    payload = {
        "oveTSet_activate": 1,
        "oveTSet_u": float(zone_setpoint_c + 273.15),
        "oveHeaPumY_activate": 1,
        "oveHeaPumY_u": enabled,
        "ovePum_activate": 1,
        "ovePum_u": enabled,
        "oveFan_activate": 1,
        "oveFan_u": enabled,
    }
    return payload, _info(policy_t_like_c, h, supply_c=None, zone_c=zone_setpoint_c, enabled=enabled)


def build_bestest_hydronic_command(action: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    policy_t_like_c = action_to_t_supply(float(action[0]))
    h = _intensity(policy_t_like_c)
    enabled = 1.0 if h > 0.05 else 0.0
    payload = {
        "oveTSetSup_activate": 1,
        "oveTSetSup_u": float(policy_t_like_c + 273.15),
        "oveTSetHea_activate": 1,
        "oveTSetHea_u": 294.15,
        "oveTSetCoo_activate": 1,
        "oveTSetCoo_u": 297.15,
        "ovePum_activate": 1,
        "ovePum_u": enabled,
    }
    return payload, _info(policy_t_like_c, h, supply_c=policy_t_like_c, zone_c=None, enabled=enabled)


def build_commercial_hydronic_command(action: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    policy_t_like_c = action_to_t_supply(float(action[0]))
    h = _intensity(policy_t_like_c)
    enabled = 1.0 if h > 0.05 else 0.0
    payload = {
        "dh_oveTSupSetHea_activate": 1,
        "dh_oveTSupSetHea_u": float(policy_t_like_c + 273.15),
        "oveTZonSet_activate": 1,
        "oveTZonSet_u": 294.15,
        "oveTSupSetAir_activate": 1,
        "oveTSupSetAir_u": float(policy_t_like_c + 273.15),
        "ovePum_activate": 1,
        "ovePum_u": 50000.0 if enabled else 0.0,
        "oveValCoi_activate": 1,
        "oveValCoi_u": enabled,
        "oveValRad_activate": 1,
        "oveValRad_u": enabled,
    }
    return payload, _info(policy_t_like_c, h, supply_c=policy_t_like_c, zone_c=21.0, enabled=enabled)


def load_adapter_name(path: str | Path | None) -> str | None:
    if path is None:
        return None
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return str(data.get("adapter_mapping", {}).get("name", "")).strip() or None


def get_adapter(testcase: str, adapter_config: str | Path | None = None) -> TestcaseAdapter:
    testcase = str(testcase)
    adapter_name = load_adapter_name(adapter_config)
    if testcase not in SUPPORTED_TESTCASES:
        raise ValueError(f"Unsupported testcase {testcase!r}. Supported: {sorted(SUPPORTED_TESTCASES)}")

    if testcase == "bestest_air":
        return TestcaseAdapter(
            testcase=testcase,
            adapter_name=adapter_name or "bestest_air_native_tsup_adapter",
            transfer_claim="native direct supply-temperature control",
            build_command=build_bestest_air_command,
        )
    if testcase == "bestest_hydronic_heat_pump":
        return TestcaseAdapter(
            testcase=testcase,
            adapter_name=adapter_name or "hydronic_setpoint_enable_adapter_v1",
            transfer_claim="adapter-mediated, not literal direct-TSup transfer",
            build_command=build_hydronic_heat_pump_command,
        )
    if testcase == "bestest_hydronic":
        return TestcaseAdapter(
            testcase=testcase,
            adapter_name=adapter_name or "hydronic_direct_supply_setpoint_adapter_v1",
            transfer_claim="direct supply-setpoint transfer with documented pump wrapper",
            build_command=build_bestest_hydronic_command,
        )
    return TestcaseAdapter(
        testcase=testcase,
        adapter_name=adapter_name or "commercial_hydronic_supply_valve_adapter_v1",
        transfer_claim="documented commercial-hydronic adapter transfer",
        build_command=build_commercial_hydronic_command,
    )
