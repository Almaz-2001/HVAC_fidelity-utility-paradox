from __future__ import annotations

import os
from time import thread_time_ns
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from envs.base_env import HVACBaseEnv
from envs.tsup_features import (
    BASIC_TSUP_OBS_DIM,
    EXTENDED_TSUP_OBS_DIM,
    WeatherLookup,
    action_to_t_supply,
    build_basic_tsup_obs,
    build_extended_tsup_obs,
)
from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_OBS_THERM_LOW = np.array([5.0, 400.0, 0.0, 0.0, -15.0], dtype=np.float32)
_OBS_THERM_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0, 35.0], dtype=np.float32)
_OBS_TSUP_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
_OBS_TSUP_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)

_T_LOW = 21.0
_T_HIGH = 25.0
_T_SUPPLY_LOW = 18.0
_T_SUPPLY_HIGH = 35.0

_SURROGATE_V2_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "outputs", "surrogate_v2", "rc_node_v2_best.pt"),
    "/app/outputs/surrogate_v2/rc_node_v2_best.pt",
]
_SURROGATE_V3_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "outputs", "surrogate_v2", "rc_node_v3_tsupply.pt"),
    "/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt",
]
_SURROGATE_V35_SUMMARY_CANDIDATES = [
    os.path.join(
        _PROJECT_ROOT,
        "outputs",
        "surrogate_v35_inverse_boptest_prior420_heads_only",
        "calibration_summary_boptest_v35.json",
    ),
    "/app/outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json",
]
_WEATHER_V2_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "data", "surrogate_v2", "boptest_v2_all.csv"),
    "/app/data/surrogate_v2/boptest_v2_all.csv",
]
_WEATHER_V3_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "data", "surrogate_v2", "boptest_v2_tsupply.csv"),
    os.path.join(_PROJECT_ROOT, "data", "surrogate_v2", "boptest_v2_all.csv"),
    "/app/data/surrogate_v2/boptest_v2_tsupply.csv",
    "/app/data/surrogate_v2/boptest_v2_all.csv",
]


def _resolve_existing_path(candidates: list[str]) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def _resolve_torch_device(value: str | None, default: str = "cpu") -> torch.device:
    raw = (value or default).strip().lower()
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raw = "cpu"
    return torch.device(raw)


def _action_to_t_supply(a0: float, t_supply_low: float, t_supply_high: float) -> float:
    return t_supply_low + (a0 + 1.0) * 0.5 * (t_supply_high - t_supply_low)


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


class SurrogateBackend(HVACBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cfg = config
        self.max_episode_steps = int(config.get("max_episode_steps", 2000))
        self.step_count = 0
        self.dt = float(config.get("step_sec", 3600))

        morl = config.get("morl", {})
        self.w_comfort = float(morl.get("w_comfort", 0.8))
        self.w_energy = float(morl.get("w_energy", 0.2))
        self.w_safety = float(morl.get("w_safety", 0.0))
        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.temp_low = float(morl.get("temp_low", _T_LOW))
        self.temp_high = float(morl.get("temp_high", _T_HIGH))

        self.t_supply_low = float(config.get("t_supply_low", _T_SUPPLY_LOW))
        self.t_supply_high = float(config.get("t_supply_high", _T_SUPPLY_HIGH))
        self.comfort_shaping = _parse_comfort_shaping(config)
        self.obs_mode = str(config.get("obs_mode", "basic")).lower()

        dr = config.get("domain_randomization", {})
        self.dr_enabled = bool(dr.get("enabled", True))
        self.dr_t_init_low = float(dr.get("t_init_low", 10.0))
        self.dr_t_init_high = float(dr.get("t_init_high", 30.0))
        self.dr_noise_std = float(dr.get("weather_noise_std", 1.5))
        self.start_day_ranges = dr.get("start_day_ranges")
        self.start_day_low = dr.get("day_low")
        self.start_day_high = dr.get("day_high")

        surrogate_path_cfg = config.get("surrogate_path")
        surrogate_path = surrogate_path_cfg or _resolve_existing_path(_SURROGATE_V3_CANDIDATES)
        surrogate_kind = str(config.get("surrogate_kind", "legacy_v3")).lower()
        surrogate_summary_json = config.get("surrogate_summary_json")
        if surrogate_summary_json is None and surrogate_kind in {"v35_raw", "v35_calibrated", "raw_v35", "calibrated_v35"}:
            surrogate_summary_json = _resolve_existing_path(_SURROGATE_V35_SUMMARY_CANDIDATES)
        self.torch_device = _resolve_torch_device(
            str(config.get("surrogate_device")) if config.get("surrogate_device") is not None else None,
            "cpu",
        )

        control_mode_cfg = config.get("control_mode")
        if control_mode_cfg is not None:
            self.control_mode = str(control_mode_cfg).lower()
        else:
            self.control_mode = "tsup_direct" if "tsupply" in os.path.basename(surrogate_path).lower() else "thermostat"

        if self.control_mode == "thermostat":
            obs_low = _OBS_THERM_LOW
            obs_high = _OBS_THERM_HIGH
            obs_shape = (5,)
        elif self.control_mode == "tsup_direct":
            obs_low = _OBS_TSUP_LOW
            obs_high = _OBS_TSUP_HIGH
            obs_shape = (EXTENDED_TSUP_OBS_DIM,) if self.obs_mode == "extended" else (BASIC_TSUP_OBS_DIM,)
        else:
            raise ValueError(f"Unsupported control_mode for SurrogateBackend: {self.control_mode}")

        self._obs_low = obs_low
        self._obs_high = obs_high
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.model = self._load_surrogate(
            surrogate_path=surrogate_path,
            surrogate_kind=surrogate_kind,
            summary_json=surrogate_summary_json,
            checkpoint_path=config.get("surrogate_checkpoint"),
            base_model_path=config.get("surrogate_base_model"),
            c_zon_min=float(config.get("surrogate_c_zon_min", 5.0e4)),
            q_scale=float(config.get("surrogate_q_scale", 3000.0)),
        )

        weather_path = config.get("weather_csv")
        if not weather_path:
            weather_candidates = _WEATHER_V3_CANDIDATES if self.control_mode == "tsup_direct" else _WEATHER_V2_CANDIDATES
            weather_path = _resolve_existing_path(weather_candidates)
        self.weather = WeatherLookup(weather_path)

        self._t_zone = 18.0
        self._t_amb = 5.0
        self._co2 = 800.0
        self._p_cool = 0.0
        self._p_fan = 0.0
        self._p_total = 0.0
        self._t_supply_prev = 0.5 * (self.t_supply_low + self.t_supply_high)
        self._time = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._delta_t_zone = 0.0

        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0

        model_meta = self.model.describe() if hasattr(self.model, "describe") else {}
        print(f"[SURROGATE] Kind: {model_meta.get('kind', surrogate_kind)}")
        print(f"[SURROGATE] Loaded: {model_meta.get('checkpoint_path') or model_meta.get('summary_json') or model_meta.get('legacy_model_path') or surrogate_path}")
        if model_meta.get("base_model_path"):
            print(f"[SURROGATE] Base model: {model_meta['base_model_path']}")
        print(f"[SURROGATE] Control mode: {self.control_mode}")
        print(f"[SURROGATE] Obs shape: {self._observation_space.shape}")
        print(f"[SURROGATE] Weather: {'REAL' if self.weather.available else 'SYNTHETIC'} ({weather_path})")
        print(f"[SURROGATE] Torch device: {self.torch_device}")

    def _load_surrogate(
        self,
        surrogate_path: str,
        surrogate_kind: str,
        summary_json: str | None,
        checkpoint_path: str | None,
        base_model_path: str | None,
        c_zon_min: float,
        q_scale: float,
    ):
        return load_direct_tsup_adapter(
            kind=surrogate_kind,
            legacy_model_path=surrogate_path,
            summary_json=summary_json,
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            device=self.torch_device,
            c_zon_min=c_zon_min,
            q_scale=q_scale,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def _hour(self):
        return (self._time / 3600.0) % 24.0

    @property
    def _day(self):
        return (self._time / 86400.0) % 365.0

    def _get_t_amb(self, hour, day):
        return self.weather.get(hour, day, self.dr_noise_std if self.dr_enabled else 0.5)

    def _sample_start_day(self) -> float:
        if isinstance(self.start_day_ranges, list) and self.start_day_ranges:
            cleaned = []
            widths = []
            for item in self.start_day_ranges:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                low = float(item[0])
                high = float(item[1])
                if high <= low:
                    continue
                cleaned.append((low, high))
                widths.append(high - low)
            if cleaned:
                idx = int(np.random.choice(len(cleaned), p=np.array(widths) / np.sum(widths)))
                low, high = cleaned[idx]
                return float(np.random.uniform(low, high))

        if self.start_day_low is not None and self.start_day_high is not None:
            low = float(self.start_day_low)
            high = float(self.start_day_high)
            if high > low:
                return float(np.random.uniform(low, high))

        return float(np.random.uniform(0.0, 365.0))

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if self.dr_enabled:
            self._t_zone = float(np.random.uniform(self.dr_t_init_low, self.dr_t_init_high))
        else:
            self._t_zone = float(np.clip(np.random.normal(18.0, 2.0), 15.0, 35.0))

        self._co2 = 800.0 + np.random.normal(0.0, 50.0)
        self._p_cool = 0.0
        self._p_fan = 0.0
        self._p_total = 0.0
        self._t_supply_prev = 0.5 * (self.t_supply_low + self.t_supply_high)
        self._start_day = self._sample_start_day()
        self._time = self._start_day * 86400.0
        self._t_amb = self._get_t_amb(self._hour, self._day)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._delta_t_zone = 0.0
        self.step_count = 0
        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0
        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        self._time += self.dt
        self._t_amb = self._get_t_amb(self._hour, self._day)
        t_next, p_total = self._surrogate_step(a0, a1)
        fan_signal = float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))
        self._co2 = float(np.clip(self._co2 - 50.0 * fan_signal + 10.0, 400.0, 2000.0))

        if self.control_mode == "tsup_direct":
            self._p_total = float(p_total)
            self._t_supply_prev = _action_to_t_supply(a0, self.t_supply_low, self.t_supply_high)
        else:
            self._p_cool = float(p_total * 0.85)
            self._p_fan = float(p_total * 0.15)

        prev_t_zone = self._t_zone
        self._t_zone = float(t_next)
        self._delta_t_zone = self._t_zone - prev_t_zone
        r_comfort = self._comfort_reward(self._t_zone, self._t_amb, self._t_supply_prev, fan_signal)
        r_energy = -self.energy_scale * float(p_total)
        r_safety = self._safety_reward(self._t_zone)
        reward = (
            self.w_comfort * r_comfort
            + self.w_energy * r_energy
            + self.w_safety * r_safety
        )
        self._update_safety(self._t_zone)
        self.step_count += 1
        self._prev_action = np.array([a0, a1], dtype=np.float32)

        info = {
            "reward_vector": {
                "comfort": r_comfort,
                "energy": r_energy,
                "safety": r_safety,
                "zone_temp": self._t_zone,
                "hvac_power": float(p_total),
                "w_comfort": self.w_comfort,
                "w_energy": self.w_energy,
                "w_safety": self.w_safety,
            },
            "safety": self.get_safety_metric(),
            "t_amb": self._t_amb,
            "hour": self._hour,
            "day": self._day,
            "control_mode": self.control_mode,
            "delta_t_zone": self._delta_t_zone,
        }
        if self.control_mode == "tsup_direct":
            info["t_supply_cmd"] = self._t_supply_prev

        return self._make_obs(), float(reward), False, self.step_count >= self.max_episode_steps, info

    def close(self):
        pass

    def _surrogate_step(self, a0, a1):
        with torch.no_grad():
            t_next, p_total = self.model(
                torch.tensor([self._t_zone], device=self.torch_device),
                torch.tensor([self._t_amb], device=self.torch_device),
                torch.tensor([self._hour], device=self.torch_device),
                torch.tensor([self._day], device=self.torch_device),
                torch.tensor([a0], device=self.torch_device),
                torch.tensor([a1], device=self.torch_device),
            )
        return float(t_next[0]), float(p_total[0])

    def _make_obs(self):
        if self.control_mode == "tsup_direct":
            if self.obs_mode == "extended":
                return build_extended_tsup_obs(
                    self._t_zone,
                    self._co2,
                    self._p_total,
                    self._t_supply_prev,
                    self._t_amb,
                    self._hour,
                    self._day,
                    self._prev_action,
                    self._delta_t_zone,
                    self.weather,
                )
            return build_basic_tsup_obs(
                self._t_zone,
                self._co2,
                self._p_total,
                self._t_supply_prev,
                self._t_amb,
            )
        else:
            raw = np.array(
                [self._t_zone, self._co2, self._p_cool, self._p_fan, self._t_amb],
                dtype=np.float32,
            )
        obs = 2.0 * (raw - self._obs_low) / (self._obs_high - self._obs_low) - 1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _comfort_reward(self, t_c, t_amb, t_supply, fan_u):
        shaping = self.comfort_shaping
        if t_c < self.temp_low:
            weight = shaping["cold_undershoot_weight"] if t_amb <= shaping["cold_amb_threshold_c"] else shaping["undershoot_weight"]
            comfort = -weight * (self.temp_low - t_c)
            if t_supply >= shaping["heating_t_supply_c"] and fan_u >= shaping["action_fan_threshold"]:
                comfort += shaping["heating_action_bonus"]
            return float(comfort)
        if t_c > self.temp_high:
            weight = shaping["hot_overshoot_weight"] if t_amb >= shaping["hot_amb_threshold_c"] else shaping["overshoot_weight"]
            comfort = -weight * (t_c - self.temp_high)
            if t_supply <= shaping["cooling_t_supply_c"] and fan_u >= shaping["action_fan_threshold"]:
                comfort += shaping["cooling_action_bonus"]
            return float(comfort)

        inner_low = self.temp_low + shaping["deadband_c"]
        inner_high = self.temp_high - shaping["deadband_c"]
        if shaping["band_bonus"] > 0.0 and inner_low <= t_c <= inner_high:
            return float(shaping["band_bonus"])
        return 0.0

    def _safety_reward(self, t_c: float) -> float:
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

    def _update_safety(self, t_c):
        self._total_steps += 1
        if t_c > self.temp_high:
            self._violation_steps += 1
            self._max_overshoot = max(self._max_overshoot, (t_c - self.temp_high) / self.temp_high)
        elif t_c < self.temp_low:
            self._violation_steps += 1
            self._max_undershoot = max(self._max_undershoot, (self.temp_low - t_c) / self.temp_low)

    def get_safety_metric(self):
        if self._total_steps == 0:
            return {"r_time": 0, "r_sev": 0, "m_s": 0, "violation_steps": 0, "total_steps": 0}
        r_time = self._violation_steps / self._total_steps
        r_sev = max(self._max_overshoot, self._max_undershoot)
        return {
            "r_time": r_time,
            "r_sev": r_sev,
            "m_s": r_time + r_sev,
            "violation_steps": self._violation_steps,
            "total_steps": self._total_steps,
        }
