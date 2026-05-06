from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from gymnasium import spaces

from envs.base_env import HVACBaseEnv
from envs.tsup_features import (
    BASIC_TSUP_OBS_DIM,
    EXTENDED_TSUP_OBS_DIM,
    WeatherLookup,
    build_basic_tsup_obs,
    build_tsup_obs,
    resolve_weather_csv,
)


def c_to_k(c: float) -> float:
    return float(c) + 273.15


_OBS_THERM_LOW = np.array([5.0, 400.0, 0.0, 0.0], dtype=np.float32)
_OBS_THERM_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0], dtype=np.float32)
_OBS_TSUP_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
_OBS_TSUP_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)


def _parse_comfort_shaping(config: Dict[str, Any]) -> Dict[str, float]:
    shaping = config.get("comfort_shaping", {}) or {}
    undershoot_weight = float(shaping.get("undershoot_weight", 1.0))
    overshoot_weight = float(shaping.get("overshoot_weight", 1.0))
    return {
        "deadband_c": float(shaping.get("deadband_c", 0.0)),
        "band_bonus": float(shaping.get("band_bonus", 0.0)),
        "undershoot_weight": undershoot_weight,
        "overshoot_weight": overshoot_weight,
        "cold_amb_threshold_c": float(shaping.get("cold_amb_threshold_c", 8.0)),
        "hot_amb_threshold_c": float(shaping.get("hot_amb_threshold_c", 24.0)),
        "cold_undershoot_weight": float(shaping.get("cold_undershoot_weight", undershoot_weight)),
        "hot_overshoot_weight": float(shaping.get("hot_overshoot_weight", overshoot_weight)),
        "heating_action_bonus": float(shaping.get("heating_action_bonus", 0.0)),
        "cooling_action_bonus": float(shaping.get("cooling_action_bonus", 0.0)),
        "heating_t_supply_c": float(shaping.get("heating_t_supply_c", 29.0)),
        "cooling_t_supply_c": float(shaping.get("cooling_t_supply_c", 21.0)),
        "action_fan_threshold": float(shaping.get("action_fan_threshold", 0.55)),
    }


class BOPTESTBackend(HVACBaseEnv):
    KEY_T_ROOM = "zon_reaTRooAir_y"
    KEY_CO2 = "zon_reaCO2RooAir_y"
    KEY_P_COO = "fcu_reaPCoo_y"
    KEY_P_FAN = "fcu_reaPFan_y"
    KEY_P_HEA = "fcu_reaPHea_y"
    KEY_T_AMB = "zon_weaSta_reaWeaTDryBul_y"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        self.control_mode = str(config.get("control_mode", "thermostat")).lower()
        if self.control_mode == "tsup_direct":
            self.obs_mode = str(config.get("obs_mode", "basic")).lower()
            self.obs_ablation = str(config.get("obs_ablation", "none")).lower()
            self.delta_feature_mode = str(config.get("delta_feature_mode", "raw")).lower()
            self.power_feature_mode = str(config.get("power_feature_mode", "raw")).lower()
            self.t_zone_feature_mode = str(config.get("t_zone_feature_mode", "raw")).lower()
            obs_dim = EXTENDED_TSUP_OBS_DIM if self.obs_mode == "extended" else BASIC_TSUP_OBS_DIM
            self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self._obs_low = _OBS_TSUP_LOW
            self._obs_high = _OBS_TSUP_HIGH
        elif self.control_mode == "thermostat":
            self.obs_mode = "basic"
            self.obs_ablation = "none"
            self.delta_feature_mode = "raw"
            self.power_feature_mode = "raw"
            self.t_zone_feature_mode = "raw"
            self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self._obs_low = _OBS_THERM_LOW
            self._obs_high = _OBS_THERM_HIGH
        else:
            raise ValueError(f"Unsupported control_mode for BOPTESTBackend: {self.control_mode}")

        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.base_url = config.get("boptest_url", "http://web:8000").rstrip("/")
        self.testcase_id = config.get("testcase_id", "bestest_air")
        self.step_sec = int(config.get("step_sec", 3600))
        self.start_time_base = float(config.get("boptest_start_time", 0.0))
        self.start_time_choices = [
            float(v) for v in (config.get("boptest_start_time_choices", []) or [])
        ]
        self.start_time_jitter_sec = float(config.get("boptest_start_jitter_sec", 0.0))
        self.warmup_sec = float(config.get("boptest_warmup_sec", 604800.0))

        self.timeout = float(config.get("boptest_timeout", config.get("http_timeout", 120.0)))
        self.max_retries = int(config.get("boptest_retries", config.get("http_retries", 5)))
        self.backoff_base = float(config.get("boptest_backoff", config.get("http_backoff_base", 0.5)))
        self.recover_on_fail = bool(config.get("recover_on_fail", True))
        self.select_timeout = float(
            config.get(
                "boptest_select_timeout",
                config.get("select_timeout", max(600.0, self.timeout)),
            )
        )

        morl = config.get("morl", {})
        self.temp_low = float(morl.get("temp_low", 20.0))
        self.temp_high = float(morl.get("temp_high", 26.0))
        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.w_comfort = float(morl.get("w_comfort", 0.8))
        self.w_energy = float(morl.get("w_energy", 0.2))
        self.w_safety = float(morl.get("w_safety", 0.0))
        self.comfort_shaping = _parse_comfort_shaping(config)

        self.t_supply_low = float(config.get("t_supply_low", 18.0))
        self.t_supply_high = float(config.get("t_supply_high", 35.0))
        self.cooling_setpoint_c = float(config.get("cooling_setpoint_c", 40.0))
        self.heating_setpoint_c = float(config.get("heating_setpoint_c", 15.0))
        self.fixed_t_supply_c = float(config.get("fixed_t_supply_c", 18.0))

        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0

        self.testid: Optional[str] = None
        self.t = 0.0
        self._prev_t_supply = 0.5 * (self.t_supply_low + self.t_supply_high)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_t_zone = 22.0
        self._delta_t_zone = 0.0
        self.weather = WeatherLookup(config.get("weather_csv") or resolve_weather_csv())

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2**attempt)
        time.sleep(min(delay, 8.0))

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        timeout = float(timeout if timeout is not None else self.timeout)
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    res = self.session.get(url, timeout=timeout)
                elif method.upper() == "POST":
                    res = self.session.post(url, json=(payload or {}), timeout=timeout)
                elif method.upper() == "PUT":
                    res = self.session.put(url, json=(payload or {}), timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if res.status_code in (500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue

                res.raise_for_status()
                return res.json()
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
                last_err = e
                self._sleep_backoff(attempt)

        if last_err:
            raise last_err
        raise RuntimeError("HTTP request failed without exception (unexpected)")

    def _select_and_warmup(self) -> None:
        url_sel = f"{self.base_url}/testcases/{self.testcase_id}/select"
        data = self._request_json("POST", url_sel, payload={}, timeout=self.select_timeout)

        self.testid = data.get("testid")
        if not self.testid:
            raise RuntimeError(f"Failed to obtain testid from select response: {data}")

        try:
            self._request_json("PUT", f"{self.base_url}/step/{self.testid}", payload={"step": self.step_sec})
        except Exception as e:
            print(f"[BOPTEST] Warning (step): {e}")

        start_time = self._sample_start_time()
        warmup = self.warmup_sec
        print(f"[BOPTEST] Initializing building (warmup={warmup:.0f}s)...")
        self._request_json(
            "PUT",
            f"{self.base_url}/initialize/{self.testid}",
            payload={"start_time": start_time, "warmup_period": warmup},
            timeout=self.select_timeout,
        )

        self.t = start_time
        self._prev_t_supply = 0.5 * (self.t_supply_low + self.t_supply_high)
        print(f"[BOPTEST] Initialized: testid={self.testid}")

    def _sample_start_time(self) -> float:
        if self.start_time_choices:
            start_time = float(np.random.choice(self.start_time_choices))
        else:
            start_time = self.start_time_base

        if self.start_time_jitter_sec > 0.0:
            jitter = float(np.random.uniform(-self.start_time_jitter_sec, self.start_time_jitter_sec))
            start_time += jitter

        return max(0.0, start_time)

    def _stop_current(self) -> None:
        if not self.testid:
            return
        try:
            self._request_json("PUT", f"{self.base_url}/stop/{self.testid}", payload={})
        except Exception:
            pass

    def _recover(self) -> None:
        if not self.recover_on_fail:
            return
        self._stop_current()
        self.testid = None
        self._select_and_warmup()

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0
        self._prev_t_supply = 0.5 * (self.t_supply_low + self.t_supply_high)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._delta_t_zone = 0.0

        try:
            self._select_and_warmup()
            payload = self.advance({})
        except Exception:
            self._recover()
            payload = self.advance({})

        self._last_t_zone = float(self._make_obs_raw(payload)[0])
        obs = self._make_obs(payload)
        return obs, {"testid": self.testid, "time": self.t}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0
        fan_u = float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))

        if self.control_mode == "tsup_direct":
            t_supply = self.t_supply_low + (a0 + 1.0) * 0.5 * (self.t_supply_high - self.t_supply_low)
            u = {
                "con_oveTSetCoo_activate": 1,
                "con_oveTSetCoo_u": c_to_k(self.cooling_setpoint_c),
                "con_oveTSetHea_activate": 1,
                "con_oveTSetHea_u": c_to_k(self.heating_setpoint_c),
                "fcu_oveFan_activate": 1,
                "fcu_oveFan_u": fan_u,
                "fcu_oveTSup_activate": 1,
                "fcu_oveTSup_u": c_to_k(t_supply),
            }
        else:
            t_target = self.temp_low + (a0 + 1.0) * 0.5 * (self.temp_high - self.temp_low)
            t_supply = self.fixed_t_supply_c
            u = {
                "con_oveTSetCoo_activate": 1,
                "con_oveTSetCoo_u": c_to_k(t_target + 0.5),
                "con_oveTSetHea_activate": 1,
                "con_oveTSetHea_u": c_to_k(t_target - 0.5),
                "fcu_oveFan_activate": 1,
                "fcu_oveFan_u": fan_u,
                "fcu_oveTSup_activate": 1,
                "fcu_oveTSup_u": c_to_k(t_supply),
            }

        payload = self.advance(u)
        self.t = self._get_val(payload, "time")
        if self.control_mode == "tsup_direct":
            self._prev_t_supply = t_supply
            current_t_zone = float(self._make_obs_raw(payload)[0])
            self._delta_t_zone = current_t_zone - self._last_t_zone
            self._last_t_zone = current_t_zone
            self._prev_action = np.array([a0, a1], dtype=np.float32)

        obs = self._make_obs(payload)
        rv = self._make_reward_vector(payload, t_supply=t_supply, fan_u=fan_u)
        reward = (
            self.w_comfort * rv["comfort"]
            + self.w_energy * rv["energy"]
            + self.w_safety * rv["safety"]
        )
        self._update_safety(rv["zone_temp"])

        t_amb_k = self._get_val(payload, self.KEY_T_AMB)
        t_amb_c = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

        info = {
            "reward_vector": rv,
            "time": self.t,
            "safety": self.get_safety_metric(),
            "t_amb": t_amb_c,
            "hour": (self.t / 3600.0) % 24.0,
            "day": (self.t / 86400.0) % 365.0,
            "control_mode": self.control_mode,
        }
        if self.control_mode == "tsup_direct":
            info["t_supply_cmd"] = t_supply
        return obs, float(reward), False, False, info

    def advance(self, actions: Dict[str, Any]) -> dict:
        if not self.testid:
            self._recover()

        url = f"{self.base_url}/advance/{self.testid}"
        try:
            data = self._request_json("POST", url, payload=actions)
        except Exception:
            self._recover()
            url = f"{self.base_url}/advance/{self.testid}"
            data = self._request_json("POST", url, payload=actions)

        return data.get("payload", data)

    def _get_val(self, values: dict, key: str) -> float:
        v = values.get(key, 0.0)
        return float(v.get("value", v) if isinstance(v, dict) else v)

    def _make_obs_raw(self, values: dict) -> np.ndarray:
        t_c = self._get_val(values, self.KEY_T_ROOM) - 273.15
        co2 = self._get_val(values, self.KEY_CO2)
        p_cool = self._get_val(values, self.KEY_P_COO)
        p_fan = self._get_val(values, self.KEY_P_FAN)

        if self.control_mode == "thermostat":
            return np.array([t_c, co2, p_cool, p_fan], dtype=np.float32)

        p_heat = self._get_val(values, self.KEY_P_HEA)
        t_amb_k = self._get_val(values, self.KEY_T_AMB)
        t_amb_c = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k
        p_total = p_cool + p_fan + p_heat
        return np.array([t_c, co2, p_total, self._prev_t_supply, t_amb_c], dtype=np.float32)

    def _make_obs(self, values: dict) -> np.ndarray:
        raw = self._make_obs_raw(values)
        if self.control_mode == "tsup_direct":
            if self.obs_mode == "extended":
                sim_time = self._get_val(values, "time")
                hour = (sim_time / 3600.0) % 24.0
                day = (sim_time / 86400.0) % 365.0
                return build_tsup_obs(
                    float(raw[0]),
                    float(raw[1]),
                    float(raw[2]),
                    float(raw[3]),
                    float(raw[4]),
                    float(hour),
                    float(day),
                    self._prev_action,
                    float(self._delta_t_zone),
                    self.weather,
                    include_forecast=True,
                    obs_ablation=self.obs_ablation,
                    delta_feature_mode=self.delta_feature_mode,
                    t_zone_feature_mode=self.t_zone_feature_mode,
                    power_feature_mode=self.power_feature_mode,
                )
            return build_basic_tsup_obs(
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
                float(raw[3]),
                float(raw[4]),
                t_zone_feature_mode=self.t_zone_feature_mode,
                power_feature_mode=self.power_feature_mode,
            )
        obs = 2.0 * (raw - self._obs_low) / (self._obs_high - self._obs_low) - 1.0
        return np.clip(obs, -1.0, 1.0)

    def _make_reward_vector(self, values: dict, t_supply: Optional[float] = None, fan_u: Optional[float] = None) -> dict:
        raw = self._make_obs_raw(values)
        t_c = float(raw[0])
        if self.control_mode == "thermostat":
            p_total = float(raw[2] + raw[3])
        else:
            p_total = float(raw[2])
        t_amb = float(raw[-1]) if self.control_mode == "tsup_direct" else 10.0

        shaping = self.comfort_shaping
        comfort = 0.0
        if t_c < self.temp_low:
            weight = shaping["cold_undershoot_weight"] if t_amb <= shaping["cold_amb_threshold_c"] else shaping["undershoot_weight"]
            comfort = -weight * (self.temp_low - t_c)
            if t_supply is not None and fan_u is not None:
                if t_supply >= shaping["heating_t_supply_c"] and fan_u >= shaping["action_fan_threshold"]:
                    comfort += shaping["heating_action_bonus"]
        elif t_c > self.temp_high:
            weight = shaping["hot_overshoot_weight"] if t_amb >= shaping["hot_amb_threshold_c"] else shaping["overshoot_weight"]
            comfort = -weight * (t_c - self.temp_high)
            if t_supply is not None and fan_u is not None:
                if t_supply <= shaping["cooling_t_supply_c"] and fan_u >= shaping["action_fan_threshold"]:
                    comfort += shaping["cooling_action_bonus"]
        else:
            inner_low = self.temp_low + shaping["deadband_c"]
            inner_high = self.temp_high - shaping["deadband_c"]
            if shaping["band_bonus"] > 0.0 and inner_low <= t_c <= inner_high:
                comfort = shaping["band_bonus"]

        energy = -self.energy_scale * p_total
        safety = self._step_safety_reward(t_c)
        return {
            "comfort": float(comfort),
            "energy": float(energy),
            "safety": float(safety),
            "zone_temp": float(t_c),
            "hvac_power": float(p_total),
            "w_comfort": float(self.w_comfort),
            "w_energy": float(self.w_energy),
            "w_safety": float(self.w_safety),
        }

    def _step_safety_reward(self, t_c: float) -> float:
        if t_c > self.temp_high:
            severity = (t_c - self.temp_high) / self.temp_high
            return float(-(1.0 + severity))
        if t_c < self.temp_low:
            severity = (self.temp_low - t_c) / self.temp_low
            return float(-(1.0 + severity))
        return 0.0

    def set_objective_weights(self, comfort: float, energy: float, safety: float | None = None) -> None:
        weights = np.array(
            [
                max(float(comfort), 0.0),
                max(float(energy), 0.0),
                max(float(0.0 if safety is None else safety), 0.0),
            ],
            dtype=np.float32,
        )
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("Objective weights must contain at least one positive value.")
        weights /= total
        self.w_comfort = float(weights[0])
        self.w_energy = float(weights[1])
        self.w_safety = float(weights[2])

    def _update_safety(self, t_c: float) -> None:
        self._total_steps += 1
        if t_c > self.temp_high:
            self._violation_steps += 1
            overshoot = (t_c - self.temp_high) / self.temp_high
            self._max_overshoot = max(self._max_overshoot, overshoot)
        elif t_c < self.temp_low:
            self._violation_steps += 1
            undershoot = (self.temp_low - t_c) / self.temp_low
            self._max_undershoot = max(self._max_undershoot, undershoot)

    def get_safety_metric(self) -> dict:
        if self._total_steps == 0:
            return {
                "r_time": 0.0,
                "r_sev": 0.0,
                "m_s": 0.0,
                "violation_steps": 0,
                "total_steps": 0,
            }

        r_time = self._violation_steps / self._total_steps
        r_sev = max(self._max_overshoot, self._max_undershoot)
        m_s = r_time + r_sev
        return {
            "r_time": r_time,
            "r_sev": r_sev,
            "m_s": m_s,
            "violation_steps": self._violation_steps,
            "total_steps": self._total_steps,
        }

    def close(self):
        self._stop_current()
        self.session.close()
