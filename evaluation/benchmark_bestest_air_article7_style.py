from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.tsup_features import (
    EXTENDED_TSUP_OBS_DIM,
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    SUPPORTED_TSUP_OBS_DIMS,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    build_tsup_obs,
    infer_tsup_model_obs_dim,
    resolve_weather_csv,
)
from layers.safety.fallback import SurrogateMPCFallback


T_LOW = 21.0
T_HIGH = 24.0
T_TARGET = 22.5
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0
WINTER_ENTER_T_AMB = 10.0
WINTER_EXIT_T_AMB = 15.0
EMERGENCY_T_AMB = 5.0
EMERGENCY_T_ZONE = 20.0
EMERGENCY_ACTION = np.array([1.0, 1.0], dtype=np.float32)

SURROGATE_MPC_MODEL_CANDIDATES = [
    REPO_ROOT / "outputs" / "surrogate_v2" / "rc_node_v3_tsupply.pt",
    Path("/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt"),
]

THERMOSTATIC_MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "ppo_thermostatic.zip",
    Path("/app/models/ppo_thermostatic.zip"),
]

HDRL_WINTER_MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "ppo_winter_final.zip",
    Path("/app/models/ppo_winter_final.zip"),
]

HDRL_SUMMER_MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "ppo_summer_final.zip",
    Path("/app/models/ppo_summer_final.zip"),
]

MORL_MODEL_CANDIDATES = [
    REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_15min" / "seed42" / "finetune_boptest" / "models" / "ppo_model.zip",
    REPO_ROOT / "outputs" / "morl_boptest_finetune_seed42" / "models" / "ppo_model.zip",
    Path("/app/outputs/morl_surrogate_ppo_v35_15min/seed42/finetune_boptest/models/ppo_model.zip"),
]


def resolve_existing_path(candidates: list[Path]) -> str:
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def flatten_boptest_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return float(value)


def k_to_c(value: float) -> float:
    value = float(value)
    return value - 273.15 if value > 200.0 else value


@dataclass
class Scenario:
    name: str
    label: str
    start_day_index: int
    start_time_sec: float
    duration_days: int
    daily_mean_t_amb_c: float


class BOPTESTClient:
    def __init__(
        self,
        base_url: str,
        testcase_id: str,
        step_sec: int,
        timeout_sec: float = 60.0,
        select_timeout_sec: float = 300.0,
        retries: int = 3,
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
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code in (500, 502, 503, 504):
                    time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))
                    continue

                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = exc
                time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"BOPTEST request failed without explicit exception: {url}")

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


def load_ppo_model(path: str) -> tuple[PPO, int]:
    obs_dim = infer_tsup_model_obs_dim(path)
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        ),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    return PPO.load(path, device="cpu", custom_objects=custom_objects), obs_dim


def load_morl_model(path: str) -> tuple[PPO, int]:
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    obs_shape = data.get("observation_space", {}).get("_shape", [5])
    obs_dim = obs_shape[0] if isinstance(obs_shape, list) else int(obs_shape)
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    return PPO.load(path, device="cpu", custom_objects=custom_objects), obs_dim


def build_bestest_air_command(action: np.ndarray) -> dict[str, Any]:
    t_supply = action_to_t_supply(float(action[0]))
    fan_u = action_to_fan(float(action[1]))
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": t_supply + 273.15,
    }


def parse_payload_state(
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
) -> dict[str, float]:
    t_zone_c = k_to_c(flatten_boptest_value(payload, "zon_reaTRooAir_y"))
    co2_ppm = flatten_boptest_value(payload, "zon_reaCO2RooAir_y")
    p_cool_w = flatten_boptest_value(payload, "fcu_reaPCoo_y")
    p_fan_w = flatten_boptest_value(payload, "fcu_reaPFan_y")
    p_heat_w = flatten_boptest_value(payload, "fcu_reaPHea_y")
    t_amb_c = k_to_c(flatten_boptest_value(payload, "zon_weaSta_reaWeaTDryBul_y"))
    sim_time_sec = flatten_boptest_value(payload, "time")
    hour = (sim_time_sec / 3600.0) % 24.0
    day = (sim_time_sec / 86400.0) % 365.0
    p_total_w = p_cool_w + p_fan_w + p_heat_w
    prev_t_supply_c = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5
    delta_t_zone_c = 0.0 if prev_t_zone is None else (t_zone_c - prev_t_zone)

    state = {
        "t_zone": float(t_zone_c),
        "co2_ppm": float(co2_ppm),
        "p_total_w": float(p_total_w),
        "t_amb": float(t_amb_c),
        "time": float(sim_time_sec),
        "hour": float(hour),
        "day": float(day),
        "delta_t_zone": float(delta_t_zone_c),
    }
    return state


def make_tsup_obs(
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
    weather: WeatherLookup,
    obs_dim: int,
    obs_ablation: str = "none",
    delta_feature_mode: str = "raw",
    t_zone_feature_mode: str = "raw",
    power_feature_mode: str = "raw",
) -> tuple[np.ndarray, dict[str, float]]:
    if obs_dim not in SUPPORTED_TSUP_OBS_DIMS:
        raise RuntimeError(f"Unsupported observation dim {obs_dim}. Supported dims: {sorted(SUPPORTED_TSUP_OBS_DIMS)}.")

    state = parse_payload_state(payload, prev_action, prev_t_zone)
    prev_t_supply_c = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 26.5
    obs = build_tsup_obs(
        state["t_zone"],
        state["co2_ppm"],
        state["p_total_w"],
        prev_t_supply_c,
        state["t_amb"],
        state["hour"],
        state["day"],
        prev_action if prev_action is not None else np.zeros(2, dtype=np.float32),
        state["delta_t_zone"],
        weather,
        include_forecast=(obs_dim == EXTENDED_TSUP_OBS_DIM),
        obs_ablation=obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    return obs, state


def make_morl_obs(
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    state = parse_payload_state(payload, prev_action, prev_t_zone)
    prev_t_supply_c = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 0.5 * (
        T_SUPPLY_LOW + T_SUPPLY_HIGH
    )
    raw = np.array(
        [
            state["t_zone"],
            state["co2_ppm"],
            state["p_total_w"],
            prev_t_supply_c,
            state["t_amb"],
        ],
        dtype=np.float32,
    )
    obs = 2.0 * (raw - STATE_LOW) / (STATE_HIGH - STATE_LOW) - 1.0
    return np.clip(obs, -1.0, 1.0), state


def compute_safety_metrics(trace_df: pd.DataFrame, step_sec: int) -> dict[str, float]:
    temps = trace_df["t_zone_c"].to_numpy(dtype=float)
    below = temps < T_LOW
    above = temps > T_HIGH
    violation = below | above

    r_time = float(np.mean(violation))
    under = np.where(below, (T_LOW - temps) / T_LOW, 0.0)
    over = np.where(above, (temps - T_HIGH) / T_HIGH, 0.0)
    r_sev = float(max(np.max(under), np.max(over)))
    rmse_center = float(np.sqrt(np.mean((temps - T_TARGET) ** 2)))
    mean_power_w = float(trace_df["p_total_w"].mean())
    energy_kwh = float(trace_df["p_total_w"].sum() * (step_sec / 3600.0) / 1000.0)
    within_band_pct = float(np.mean((temps >= T_LOW) & (temps <= T_HIGH)) * 100.0)

    return {
        "r_time": r_time,
        "r_sev": r_sev,
        "m_s": float(r_time + r_sev),
        "violation_pct": float(r_time * 100.0),
        "rmse_center_c": rmse_center,
        "rmse_22_c": rmse_center,
        "mean_power_w": mean_power_w,
        "energy_kwh": energy_kwh,
        "within_band_pct": within_band_pct,
        "t_min_c": float(np.min(temps)),
        "t_max_c": float(np.max(temps)),
        "t_mean_c": float(np.mean(temps)),
    }


def derive_article7_style_scenarios(
    weather_csv: str,
    duration_days: int,
    heating_threshold_c: float,
) -> list[Scenario]:
    df = pd.read_csv(weather_csv)
    grouped = (
        df.groupby("day", as_index=False)["t_amb"]
        .mean()
        .rename(columns={"t_amb": "daily_mean_t_amb_c"})
        .sort_values("day")
        .reset_index(drop=True)
    )

    grouped = grouped[(grouped["day"] >= 0) & (grouped["day"] <= 365 - duration_days)].copy()
    if grouped.empty:
        raise RuntimeError(f"Could not derive daily weather windows from {weather_csv}")

    peak_row = grouped.loc[grouped["daily_mean_t_amb_c"].idxmin()]
    heating_days = grouped[grouped["daily_mean_t_amb_c"] <= heating_threshold_c].copy()
    if heating_days.empty:
        heating_days = grouped.copy()

    typical_target = float(heating_days["daily_mean_t_amb_c"].median())
    heating_days["distance"] = np.abs(heating_days["daily_mean_t_amb_c"] - typical_target)
    typical_row = heating_days.sort_values(["distance", "day"]).iloc[0]

    scenarios = [
        Scenario(
            name="peak_heat_window",
            label="Peak heat window",
            start_day_index=int(peak_row["day"]),
            start_time_sec=float(int(peak_row["day"]) * 86400),
            duration_days=duration_days,
            daily_mean_t_amb_c=float(peak_row["daily_mean_t_amb_c"]),
        ),
        Scenario(
            name="typical_heat_window",
            label="Typical heat window",
            start_day_index=int(typical_row["day"]),
            start_time_sec=float(int(typical_row["day"]) * 86400),
            duration_days=duration_days,
            daily_mean_t_amb_c=float(typical_row["daily_mean_t_amb_c"]),
        ),
    ]
    return scenarios


class BaseController:
    name: str
    obs_mode: str = "tsup"

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class PIController(BaseController):
    name = "pi"
    obs_mode = "none"

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[None, dict[str, Any]]:
        return None, {"source": "built_in_pi"}


class ThermostaticController(BaseController):
    name = "thermostatic"
    obs_mode = "tsup"

    def __init__(
        self,
        model_path: str,
        obs_ablation: str = "none",
        delta_feature_mode: str = "raw",
        t_zone_feature_mode: str = "raw",
        power_feature_mode: str = "raw",
    ) -> None:
        self.model, self.obs_dim = load_ppo_model(model_path)
        self.obs_ablation = str(obs_ablation).strip().lower()
        self.delta_feature_mode = str(delta_feature_mode).strip().lower()
        self.t_zone_feature_mode = str(t_zone_feature_mode).strip().lower()
        self.power_feature_mode = str(power_feature_mode).strip().lower()

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        return action, {"source": "ppo_thermostatic"}


class HDRLController(BaseController):
    name = "hdrl"
    obs_mode = "tsup"

    def __init__(
        self,
        winter_model_path: str,
        summer_model_path: str,
        obs_ablation: str = "none",
        delta_feature_mode: str = "raw",
        t_zone_feature_mode: str = "raw",
        power_feature_mode: str = "raw",
    ) -> None:
        self.winter_model, self.winter_obs_dim = load_ppo_model(winter_model_path)
        self.summer_model, self.summer_obs_dim = load_ppo_model(summer_model_path)
        if self.winter_obs_dim != self.summer_obs_dim:
            raise RuntimeError(
                "HDRL benchmark currently expects winter and summer models to use the same observation dim, "
                f"got {self.winter_obs_dim} and {self.summer_obs_dim}."
            )
        self.obs_dim = self.winter_obs_dim
        self.gate_mode = "winter"
        self.obs_ablation = obs_ablation
        self.delta_feature_mode = delta_feature_mode
        self.t_zone_feature_mode = t_zone_feature_mode
        self.power_feature_mode = power_feature_mode

    def reset(self, initial_t_amb: float) -> None:
        self.gate_mode = "winter" if initial_t_amb < WINTER_ENTER_T_AMB else "summer"

    def _update_gate_mode(self, t_amb: float) -> None:
        if self.gate_mode == "winter":
            if t_amb > WINTER_EXIT_T_AMB:
                self.gate_mode = "summer"
        else:
            if t_amb < WINTER_ENTER_T_AMB:
                self.gate_mode = "winter"

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        t_zone = float(state["t_zone"])
        t_amb = float(state["t_amb"])

        if t_amb < EMERGENCY_T_AMB and t_zone < EMERGENCY_T_ZONE:
            return EMERGENCY_ACTION.copy(), {"source": "emergency"}

        self._update_gate_mode(t_amb)
        if self.gate_mode == "winter":
            action, _ = self.winter_model.predict(obs, deterministic=True)
            source = "winter"
        else:
            action, _ = self.summer_model.predict(obs, deterministic=True)
            source = "summer"
        return np.asarray(action, dtype=np.float32), {"source": source}


class SurrogateMPCController(BaseController):
    name = "surrogate_mpc"
    obs_mode = "tsup"

    def __init__(
        self,
        surrogate_path: str,
        horizon: int,
        n_iters: int,
        lr: float,
        safety_margin_c: float,
        lambda_safety: float,
        lambda_energy: float,
    ) -> None:
        self.obs_dim = EXTENDED_TSUP_OBS_DIM
        self.policy = SurrogateMPCFallback(
            model_path=surrogate_path,
            horizon=horizon,
            n_iters=n_iters,
            lr=lr,
            t_safe_low=T_LOW + safety_margin_c,
            t_safe_high=T_HIGH - safety_margin_c,
            lambda_safety=lambda_safety,
            lambda_energy=lambda_energy,
        )

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        action = self.policy.compute(state)
        return np.asarray(action, dtype=np.float32), {"source": "surrogate_mpc"}


class MORLController(BaseController):
    name = "morl"
    obs_mode = "morl"

    def __init__(self, model_path: str) -> None:
        self.model, self.obs_dim = load_morl_model(model_path)

    def act(self, obs: np.ndarray, state: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32), {"source": "morl"}


def build_controller_obs(
    controller: BaseController,
    payload: dict[str, Any],
    prev_action: np.ndarray,
    prev_t_zone: float | None,
    weather: WeatherLookup,
) -> tuple[np.ndarray | None, dict[str, float]]:
    if getattr(controller, "obs_mode", "tsup") == "none":
        return None, parse_payload_state(payload, prev_action, prev_t_zone)
    if controller.obs_mode == "morl":
        return make_morl_obs(payload, prev_action, prev_t_zone)
    return make_tsup_obs(
        payload,
        prev_action,
        prev_t_zone,
        weather,
        getattr(controller, "obs_dim", EXTENDED_TSUP_OBS_DIM),
        obs_ablation=getattr(controller, "obs_ablation", "none"),
        delta_feature_mode=getattr(controller, "delta_feature_mode", "raw"),
        t_zone_feature_mode=getattr(controller, "t_zone_feature_mode", "raw"),
        power_feature_mode=getattr(controller, "power_feature_mode", "raw"),
    )


def plot_scenario_trace(
    traces: dict[str, pd.DataFrame],
    scenario: Scenario,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for controller_name, df in traces.items():
        x_days = (df["sim_time_sec"].to_numpy(dtype=float) - scenario.start_time_sec) / 86400.0
        ax1.plot(x_days, df["t_zone_c"], linewidth=1.9, label=controller_name)
        ax2.plot(x_days, df["p_total_w"], linewidth=1.5, label=controller_name)

    x_ref = (next(iter(traces.values()))["sim_time_sec"].to_numpy(dtype=float) - scenario.start_time_sec) / 86400.0
    ax1.fill_between(x_ref, T_LOW, T_HIGH, color="#dff3e4", alpha=0.7, label="comfort band")
    ax1.set_ylabel("Zone temperature, C")
    ax1.set_title(f"{scenario.label} | start_day={scenario.start_day_index} | daily_mean_t_amb={scenario.daily_mean_t_amb_c:.2f} C")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncol=2)

    ax2.set_ylabel("HVAC power, W")
    ax2.set_xlabel("Elapsed time, days")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    scenarios = list(summary_df["scenario"].unique())
    controllers = list(summary_df["controller"].unique())
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(controllers), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, controller in enumerate(controllers):
        subset = summary_df[summary_df["controller"] == controller].set_index("scenario").reindex(scenarios)
        ax.bar(x - 0.4 + width / 2 + idx * width, subset["m_s"], width=width, label=controller)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("m_s safety metric")
    ax.set_title("Article 7 style benchmark on bestest_air")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_controller_on_scenario(
    client: BOPTESTClient,
    controller: BaseController,
    scenario: Scenario,
    warmup_sec: float,
    step_sec: int,
    weather: WeatherLookup,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total_steps = int(scenario.duration_days * 86400 / step_sec)
    testid = client.select_testcase()
    client.initialize(testid, scenario.start_time_sec, warmup_sec)
    payload = client.advance(testid, {})

    prev_action = np.zeros(2, dtype=np.float32)
    prev_t_zone = None
    obs, state = build_controller_obs(controller, payload, prev_action, prev_t_zone, weather)

    if isinstance(controller, HDRLController):
        controller.reset(state["t_amb"])

    rows: list[dict[str, Any]] = []
    try:
        for step in range(total_steps):
            action, meta = controller.act(obs, state)
            action_arr = None if action is None else np.asarray(action, dtype=np.float32)
            command = {} if action_arr is None else build_bestest_air_command(action_arr)
            next_payload = client.advance(testid, command)
            next_prev_action = prev_action if action_arr is None else action_arr
            next_obs, next_state = build_controller_obs(
                controller,
                next_payload,
                next_prev_action,
                state["t_zone"],
                weather,
            )

            rows.append(
                {
                    "step": step,
                    "sim_time_sec": next_state["time"],
                    "t_zone_c": next_state["t_zone"],
                    "t_amb_c": next_state["t_amb"],
                    "p_total_w": next_state["p_total_w"],
                    "a0": float(action_arr[0]) if action_arr is not None else float("nan"),
                    "a1": float(action_arr[1]) if action_arr is not None else float("nan"),
                    "t_supply_cmd_c": action_to_t_supply(float(action_arr[0])) if action_arr is not None else float("nan"),
                    "fan_cmd_u": action_to_fan(float(action_arr[1])) if action_arr is not None else float("nan"),
                    "controller_source": str(meta.get("source", controller.name)),
                    "prev_t_zone_c": state["t_zone"],
                }
            )

            prev_t_zone = state["t_zone"]
            prev_action = next_prev_action
            obs, state = next_obs, next_state
    finally:
        client.stop(testid)

    trace_df = pd.DataFrame(rows)
    metrics = compute_safety_metrics(trace_df, step_sec)
    metrics.update(
        {
            "controller": controller.name,
            "scenario": scenario.name,
            "label": scenario.label,
            "start_day_index": scenario.start_day_index,
            "start_time_sec": scenario.start_time_sec,
            "duration_days": scenario.duration_days,
            "step_sec": step_sec,
            "daily_mean_t_amb_c": scenario.daily_mean_t_amb_c,
        }
    )
    return trace_df, metrics


def parse_controller_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one controller must be specified.")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Windowed benchmark on bestest_air for PI, thermostatic PPO, HDRL, MORL, and surrogate MPC."
    )
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--testcase-id", default="bestest_air")
    parser.add_argument("--output-dir", default="outputs/bestest_air_article7_style")
    parser.add_argument("--step-sec", type=int, default=3600)
    parser.add_argument("--duration-days", type=int, default=14)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--heating-threshold-c", type=float, default=12.0)
    parser.add_argument(
        "--controllers",
        default="pi,thermostatic,hdrl,morl",
        help="Comma-separated list: pi, thermostatic, hdrl, morl, surrogate_mpc",
    )
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default="none")
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default="raw")
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default="raw")
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default="raw")
    parser.add_argument("--thermostatic-model", default=resolve_existing_path(THERMOSTATIC_MODEL_CANDIDATES))
    parser.add_argument("--hdrl-winter-model", default=resolve_existing_path(HDRL_WINTER_MODEL_CANDIDATES))
    parser.add_argument("--hdrl-summer-model", default=resolve_existing_path(HDRL_SUMMER_MODEL_CANDIDATES))
    parser.add_argument("--morl-model", default=resolve_existing_path(MORL_MODEL_CANDIDATES))
    parser.add_argument("--surrogate-model", default=resolve_existing_path(SURROGATE_MPC_MODEL_CANDIDATES))
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-iters", type=int, default=25)
    parser.add_argument("--mpc-lr", type=float, default=0.08)
    parser.add_argument("--mpc-safety-margin-c", type=float, default=0.75)
    parser.add_argument("--mpc-lambda-safety", type=float, default=80.0)
    parser.add_argument("--mpc-lambda-energy", type=float, default=0.4)
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
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

    client = BOPTESTClient(
        base_url=args.boptest_url,
        testcase_id=args.testcase_id,
        step_sec=args.step_sec,
        timeout_sec=60.0,
        select_timeout_sec=300.0,
        retries=3,
        backoff_base_sec=1.0,
    )
    version_payload = client.check_connectivity()
    print(f"[BOPTEST] Connected: {version_payload}")
    print("[SCENARIOS]")
    for scenario in scenarios:
        print(
            f"  {scenario.name}: start_day={scenario.start_day_index}, "
            f"t_amb_mean={scenario.daily_mean_t_amb_c:.2f} C, duration={scenario.duration_days} days"
        )

    controller_names = parse_controller_names(args.controllers)
    controllers: list[BaseController] = []
    for name in controller_names:
        if name == "pi":
            controllers.append(PIController())
        elif name == "thermostatic":
            controllers.append(
                ThermostaticController(
                    args.thermostatic_model,
                    obs_ablation=args.obs_ablation,
                    delta_feature_mode=args.delta_feature_mode,
                    t_zone_feature_mode=args.t_zone_feature_mode,
                    power_feature_mode=args.power_feature_mode,
                )
            )
        elif name == "hdrl":
            controllers.append(
                HDRLController(
                    args.hdrl_winter_model,
                    args.hdrl_summer_model,
                    obs_ablation=args.obs_ablation,
                    delta_feature_mode=args.delta_feature_mode,
                    t_zone_feature_mode=args.t_zone_feature_mode,
                    power_feature_mode=args.power_feature_mode,
                )
            )
        elif name == "morl":
            controllers.append(MORLController(args.morl_model))
        elif name == "surrogate_mpc":
            controllers.append(
                SurrogateMPCController(
                    surrogate_path=args.surrogate_model,
                    horizon=args.mpc_horizon,
                    n_iters=args.mpc_iters,
                    lr=args.mpc_lr,
                    safety_margin_c=args.mpc_safety_margin_c,
                    lambda_safety=args.mpc_lambda_safety,
                    lambda_energy=args.mpc_lambda_energy,
                )
            )
        else:
            raise ValueError(f"Unknown controller: {name}")

    summary_rows: list[dict[str, Any]] = []
    scenarios_manifest: list[dict[str, Any]] = [scenario.__dict__ for scenario in scenarios]
    (output_dir / "scenario_manifest.json").write_text(
        json.dumps(
            {
                "boptest_url": args.boptest_url,
                "testcase_id": args.testcase_id,
                "weather_csv": weather_csv,
                "duration_days": args.duration_days,
                "step_sec": args.step_sec,
                "scenarios": scenarios_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for scenario in scenarios:
        scenario_traces: dict[str, pd.DataFrame] = {}
        print(f"\n{'=' * 72}")
        print(f"SCENARIO: {scenario.name} | {scenario.label}")
        print(f"{'=' * 72}")
        for controller in controllers:
            print(f"[RUN] controller={controller.name}")
            trace_df, metrics = run_controller_on_scenario(
                client=client,
                controller=controller,
                scenario=scenario,
                warmup_sec=args.warmup_sec,
                step_sec=args.step_sec,
                weather=weather,
            )
            trace_path = traces_dir / f"{scenario.name}_{controller.name}.csv"
            trace_df.to_csv(trace_path, index=False)
            scenario_traces[controller.name] = trace_df
            summary_rows.append(metrics)
            print(
                f"  m_s={metrics['m_s']:.4f} | viol={metrics['violation_pct']:.1f}% | "
                f"rmse_center={metrics['rmse_center_c']:.3f} C | mean_power={metrics['mean_power_w']:.1f} W"
            )

        plot_scenario_trace(
            scenario_traces,
            scenario,
            output_dir / f"{scenario.name}_trace_compare.png",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    plot_summary(summary_df, output_dir / "summary_ms_comparison.png")

    print(f"\nSaved summary: {output_dir / 'summary.csv'}")
    print(
        summary_df[
            [
                "controller",
                "scenario",
                "m_s",
                "violation_pct",
                "rmse_center_c",
                "mean_power_w",
                "energy_kwh",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
