from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import action_to_t_supply


DEFAULT_BOPTEST_URL = "http://web:8000"
DEFAULT_TESTCASE = "bestest_hydronic_heat_pump"
DEFAULT_STEP_SEC = 900

DEFAULT_ADAPTER_CONFIGS = {
    "bestest_hydronic_heat_pump": "configs/block3_actuator_mapping_bestest_hydronic_heat_pump.yaml",
    "bestest_hydronic": "configs/block3_actuator_mapping_bestest_hydronic.yaml",
    "singlezone_commercial_hydronic": "configs/block3_actuator_mapping_singlezone_commercial_hydronic.yaml",
}


def default_adapter_config(testcase_id: str) -> str:
    try:
        return DEFAULT_ADAPTER_CONFIGS[testcase_id]
    except KeyError as exc:
        raise ValueError(f"No default Block 3 hydronic adapter config for testcase: {testcase_id}") from exc


def default_output_csv(testcase_id: str) -> str:
    return f"data/block3_{testcase_id}/hydronic_adapter_stage_c_15min.csv"

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
    return {
        "t_zone": float(t_zone),
        "p_total_w": float(p_total_w),
        "t_amb": float(t_amb),
        "time": float(sim_time_sec),
        "hour": float((sim_time_sec / 3600.0) % 24.0),
        "day": float((sim_time_sec / 86400.0) % 365.0),
        "delta_t_zone": 0.0 if prev_t_zone is None else float(t_zone - prev_t_zone),
        "prev_t_supply_c": action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5,
        "rea_t_supply_c": k_to_c(get_first_val(payload, ("reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y"))) if any(k in payload for k in ("reaTSup_y", "oveTSetSup_y", "dh_reaTSupHyd_y", "dh_oveTSupSetHea_y")) else np.nan,
        "rea_heat_pump_power_w": get_first_val(payload, ("reaPHeaPum_y", "reaQHea_y")) if any(k in payload for k in ("reaPHeaPum_y", "reaQHea_y")) else 0.0,
        "rea_fan_power_w": sum(get_val(payload, key, 0.0) for key in ("reaPFan_y", "ahu_reaPFanExt_y", "ahu_reaPFanSup_y")),
        "rea_pump_power_w": get_first_val(payload, ("reaPPumEmi_y", "reaPPum_y")) if any(k in payload for k in ("reaPPumEmi_y", "reaPPum_y")) else 0.0,
    }


def load_adapter_name(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("adapter_mapping", {}).get("name", "hydronic_setpoint_enable_adapter_v1"))


def hydronic_adapter_command(action: np.ndarray, adapter_name: str) -> tuple[dict[str, float], dict[str, float]]:
    policy_t_like_c = action_to_t_supply(float(action[0]))
    h = float(np.clip((policy_t_like_c - 18.0) / (35.0 - 18.0), 0.0, 1.0))
    enabled = 1.0 if h > 0.05 else 0.0
    if adapter_name == "hydronic_direct_supply_setpoint_adapter_v1":
        return (
            {
                "oveTSetSup_activate": 1,
                "oveTSetSup_u": float(policy_t_like_c + 273.15),
                "oveTSetHea_activate": 1,
                "oveTSetHea_u": 294.15,
                "oveTSetCoo_activate": 1,
                "oveTSetCoo_u": 297.15,
                "ovePum_activate": 1,
                "ovePum_u": enabled,
            },
            {
                "policy_temperature_like_command_c": float(policy_t_like_c),
                "adapter_heat_intensity": h,
                "adapter_supply_setpoint_c": float(policy_t_like_c),
                "adapter_zone_setpoint_c": np.nan,
                "adapter_plant_enabled": enabled,
            },
        )
    if adapter_name == "commercial_hydronic_supply_valve_adapter_v1":
        return (
            {
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
            },
            {
                "policy_temperature_like_command_c": float(policy_t_like_c),
                "adapter_heat_intensity": h,
                "adapter_supply_setpoint_c": float(policy_t_like_c),
                "adapter_zone_setpoint_c": 21.0,
                "adapter_plant_enabled": enabled,
            },
        )
    setpoint_k = 288.15 + h * (297.15 - 288.15)
    return (
        {
            "oveTSet_activate": 1,
            "oveTSet_u": float(setpoint_k),
            "oveHeaPumY_activate": 1,
            "oveHeaPumY_u": enabled,
            "ovePum_activate": 1,
            "ovePum_u": enabled,
            "oveFan_activate": 1,
            "oveFan_u": enabled,
        },
        {
            "policy_temperature_like_command_c": float(policy_t_like_c),
            "adapter_heat_intensity": h,
            "adapter_zone_setpoint_c": float(setpoint_k - 273.15),
            "adapter_plant_enabled": enabled,
        },
    )


class BOPTESTClient:
    def __init__(
        self,
        *,
        base_url: str,
        testcase_id: str,
        step_sec: int,
        timeout_sec: float,
        select_timeout_sec: float,
        retries: int,
        backoff_base_sec: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.testcase_id = testcase_id
        self.step_sec = int(step_sec)
        self.timeout_sec = float(timeout_sec)
        self.select_timeout_sec = float(select_timeout_sec)
        self.retries = int(retries)
        self.backoff_base_sec = float(backoff_base_sec)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        timeout = float(timeout if timeout is not None else self.timeout_sec)
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if method == "GET":
                    response = self.session.get(url, timeout=timeout)
                elif method == "POST":
                    response = self.session.post(url, json=payload or {}, timeout=timeout)
                elif method == "PUT":
                    response = self.session.put(url, json=payload or {}, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                if response.status_code in (500, 502, 503, 504):
                    import time

                    time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
                last_error = exc
                import time

                time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))
        raise RuntimeError(f"BOPTEST request failed: {url}") from last_error

    def check_connectivity(self) -> dict[str, Any]:
        return self._request_json("GET", "/version", timeout=min(self.timeout_sec, 10.0))

    def select_testcase(self) -> str:
        data = self._request_json(
            "POST",
            f"/testcases/{self.testcase_id}/select",
            payload={},
            timeout=self.select_timeout_sec,
        )
        testid = data.get("testid")
        if not testid:
            raise RuntimeError(f"Could not obtain testid from BOPTEST response: {data}")
        return str(testid)

    def initialize(self, testid: str, start_time_sec: float, warmup_sec: float) -> None:
        self._request_json("PUT", f"/step/{testid}", payload={"step": self.step_sec}, timeout=30.0)
        self._request_json(
            "PUT",
            f"/initialize/{testid}",
            payload={"start_time": float(start_time_sec), "warmup_period": float(warmup_sec)},
            timeout=self.select_timeout_sec,
        )

    def advance(self, testid: str, actions: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._request_json("POST", f"/advance/{testid}", payload=actions or {})
        return data.get("payload", data)

    def stop(self, testid: str) -> None:
        try:
            self._request_json("PUT", f"/stop/{testid}", payload={}, timeout=10.0)
        except Exception:
            pass


def choose_action(policy: str, step: int, rng: np.random.Generator, state: dict[str, float]) -> np.ndarray:
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    if policy == "low_high":
        phase = (step // 8) % 2
        return np.array([1.0 if phase else -1.0, 1.0], dtype=np.float32)
    if policy == "comfort_probe":
        t_zone = float(state["t_zone"])
        if t_zone < 21.0:
            a0 = 1.0
        elif t_zone > 24.0:
            a0 = -1.0
        else:
            a0 = 0.25 * np.sin(step / 6.0)
        return np.array([np.clip(a0, -1.0, 1.0), 1.0], dtype=np.float32)
    raise ValueError(f"Unsupported policy: {policy}")


def run_episode(
    *,
    client: BOPTESTClient,
    scenario_name: str,
    start_time: float,
    episode_id: str,
    policy: str,
    seed: int,
    steps: int,
    adapter_name: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    testid = client.select_testcase()
    print(f"[COLLECT] scenario={scenario_name} episode={episode_id} policy={policy} testid={testid}")
    rows: list[dict[str, Any]] = []
    prev_action = np.zeros(2, dtype=np.float32)
    try:
        client.initialize(testid, start_time, 0.0)
        payload = client.advance(testid, {})
        state = parse_hydronic_state(payload, prev_action, None)
        for step in range(steps):
            action = choose_action(policy, step, rng, state)
            command, adapter_info = hydronic_adapter_command(action, adapter_name)
            payload_next = client.advance(testid, command)
            next_state = parse_hydronic_state(payload_next, action, state["t_zone"])
            rows.append(
                {
                    "episode_id": episode_id,
                    "scenario": scenario_name,
                    "policy": policy,
                    "season": scenario_name,
                    "step": step,
                    "time": float(state["time"]),
                    "t_zone": float(state["t_zone"]),
                    "t_amb": float(state["t_amb"]),
                    "hour": float(state["hour"]),
                    "day": float(state["day"]),
                    "a0_raw": float(action[0]),
                    "a1_raw": float(action[1]),
                    "t_zone_next": float(next_state["t_zone"]),
                    "delta_t": float(next_state["t_zone"] - state["t_zone"]),
                    "p_total": float(next_state["p_total_w"]),
                    "rea_t_supply_c": float(next_state["rea_t_supply_c"]),
                    "rea_heat_pump_power_w": float(next_state["rea_heat_pump_power_w"]),
                    "rea_fan_power_w": float(next_state["rea_fan_power_w"]),
                    "rea_pump_power_w": float(next_state["rea_pump_power_w"]),
                    **adapter_info,
                }
            )
            state = next_state
            prev_action = action
            if step % 96 == 0:
                print(
                    f"  step={step:4d}/{steps} T={state['t_zone']:.2f}C "
                    f"P={state['p_total_w']:.0f}W h={adapter_info['adapter_heat_intensity']:.2f}"
                )
    finally:
        client.stop(testid)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Block 3 hydronic adapter telemetry in the v3/v3.5 Stage-C CSV schema."
    )
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", DEFAULT_BOPTEST_URL))
    parser.add_argument("--testcase-id", "--testcase", dest="testcase_id", default=DEFAULT_TESTCASE)
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--step-sec", type=int, default=DEFAULT_STEP_SEC)
    parser.add_argument("--steps-per-episode", type=int, default=96)
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--policies", default="random,low_high,comfort_probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-timeout", type=float, default=300.0)
    parser.add_argument("--advance-timeout", type=float, default=60.0)
    parser.add_argument("--http-retries", type=int, default=3)
    args = parser.parse_args()

    adapter_config_arg = args.adapter_config or default_adapter_config(args.testcase_id)
    output_csv_arg = args.output_csv or default_output_csv(args.testcase_id)

    adapter_config = ROOT / adapter_config_arg
    if not adapter_config.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config}")
    adapter_name = load_adapter_name(adapter_config)

    output_csv = ROOT / output_csv_arg
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    client = BOPTESTClient(
        base_url=args.boptest_url,
        testcase_id=args.testcase_id,
        step_sec=int(args.step_sec),
        timeout_sec=float(args.advance_timeout),
        select_timeout_sec=float(args.select_timeout),
        retries=int(args.http_retries),
        backoff_base_sec=1.0,
    )
    print("BOPTEST:", client.check_connectivity())

    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    episode_index = 0
    for scenario_name, start_time in SCENARIOS.items():
        for policy in policies:
            for rep in range(int(args.episodes_per_scenario)):
                episode_seed = int(args.seed + episode_index)
                episode_id = f"{scenario_name}_{policy}_{rep:02d}"
                rows.extend(
                    run_episode(
                        client=client,
                        scenario_name=scenario_name,
                        start_time=float(start_time),
                        episode_id=episode_id,
                        policy=policy,
                        seed=episode_seed,
                        steps=int(args.steps_per_episode),
                        adapter_name=adapter_name,
                    )
                )
                episode_index += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    manifest = {
        "boptest_url": args.boptest_url,
        "testcase_id": args.testcase_id,
        "adapter_config": str(adapter_config),
        "adapter_name": adapter_name,
        "output_csv": str(output_csv),
        "step_sec": int(args.step_sec),
        "steps_per_episode": int(args.steps_per_episode),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "policies": policies,
        "rows": int(len(df)),
        "transfer_regime": "partial",
        "stage": "target telemetry collection for Stage C-only recalibration",
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved telemetry: {output_csv}")
    print(f"Saved manifest:  {manifest_path}")
    print(f"Rows: {len(df)}")
    print(df[["t_zone", "t_zone_next", "t_amb", "a0_raw", "a1_raw", "p_total"]].describe().to_string())


if __name__ == "__main__":
    main()
