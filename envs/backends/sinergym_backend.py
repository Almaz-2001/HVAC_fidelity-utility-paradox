# envs/backends/sinergym_backend.py
from __future__ import annotations

from typing import Any, Dict
import socket as pysocket
import math

import gymnasium as gym
import numpy as np
import sinergym  # noqa: F401

from envs.base_env import HVACBaseEnv
from envs.rewards.morl_reward import MORLReward, MORLWeights

from envs.wrappers import (
    ActionSafetyWrapper,
    ActionNormalizeWrapper,
    ActionRateLimitWrapper,
)


class SinergymBackend(HVACBaseEnv):
    metadata = {"render_modes": []}

    DEFAULT_TEMP_NAME = "Zone Air Temperature(SPACE1-1)"
    DEFAULT_POWER_NAME = "Facility Total HVAC Electricity Demand Rate(Whole Building)"

    def __init__(self, config: dict):
        HVACBaseEnv.__init__(self, config)

        self.env_id = config.get("env_id", "Eplus-5Zone-hot-continuous-v1")

        pysocket.gethostname = lambda: "127.0.0.1"

        env = gym.make(self.env_id, disable_env_checker=True)

        while hasattr(env, "env"):
            env = env.env

        normalize_action = bool(config.get("normalize_action", True))
        action_parameterization = str(config.get("action_parameterization", "direct"))
        rate_limit = bool(config.get("rate_limit", True))
        max_delta = float(config.get("max_delta", 0.15))
        deadband = float(config.get("deadband", 0.5))

        env = ActionSafetyWrapper(env, deadband=deadband)

        if normalize_action:
            env = ActionNormalizeWrapper(
                env,
                parameterization=action_parameterization,
                min_deadband=deadband,
            )

        if rate_limit:
            env = ActionRateLimitWrapper(env, max_delta=max_delta)

        self.env: gym.Env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        morl_cfg: Dict[str, Any] = config.get("morl", {}) or {}

        configured_temp_index = int(morl_cfg.get("temp_index", 10))
        configured_energy_index = int(morl_cfg.get("energy_index", 16))
        self.temp_name = str(morl_cfg.get("temp_name", self.DEFAULT_TEMP_NAME))
        self.energy_name = str(morl_cfg.get("energy_name", self.DEFAULT_POWER_NAME))
        self.observation_variables = self._discover_observation_variables()
        self.temp_index, self.temp_resolve_source = self._resolve_observation_index(
            preferred_name=self.temp_name,
            fallback_index=configured_temp_index,
        )
        self.energy_index, self.energy_resolve_source = self._resolve_observation_index(
            preferred_name=self.energy_name,
            fallback_index=configured_energy_index,
        )
        self.temp_observation_name = self._lookup_observation_name(self.temp_index)
        self.energy_observation_name = self._lookup_observation_name(self.energy_index)

        self.morl = MORLReward(
            temp_low=float(morl_cfg.get("temp_low", 20.0)),
            temp_high=float(morl_cfg.get("temp_high", 26.0)),
            energy_scale=float(morl_cfg.get("energy_scale", 1e-6)),
            weights=MORLWeights(
                comfort=float(morl_cfg.get("w_comfort", 0.6)),
                energy=float(morl_cfg.get("w_energy", 0.4)),
            ),
            cold_penalty_mult=float(morl_cfg.get("cold_penalty_mult", 1.5)),
            quadratic_threshold=float(morl_cfg.get("quadratic_threshold", 2.0)),
            comfort_bonus=float(morl_cfg.get("comfort_bonus", 0.02)),
            inband_temp_penalty_scale=float(morl_cfg.get("inband_temp_penalty_scale", 0.25)),
            recovery_energy_mult=float(morl_cfg.get("recovery_energy_mult", 0.60)),
            peak_power_threshold=float(morl_cfg.get("peak_power_threshold", 700.0)),
            peak_power_scale=float(morl_cfg.get("peak_power_scale", 8e-5)),
            action_penalty_scale=float(morl_cfg.get("action_penalty_scale", 0.03)),
            reward_shape=str(morl_cfg.get("reward_shape", "linear")),
            target_temp=morl_cfg.get("target_temp"),
            exp_alpha=float(morl_cfg.get("exp_alpha", 2.5)),
            exp_scale=float(morl_cfg.get("exp_scale", 0.04)),
            gaussian_sigma=float(morl_cfg.get("gaussian_sigma", 1.0)),
            gaussian_peak=float(morl_cfg.get("gaussian_peak", 1.0)),
            gaussian_offset=float(morl_cfg.get("gaussian_offset", 0.2)),
            cubic_scale=float(morl_cfg.get("cubic_scale", 0.60)),
        )

        self._dbg_steps = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def unwrapped(self):
        return self.env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    @staticmethod
    def _safe_scalar(value: Any) -> float | None:
        try:
            scalar = float(value)
        except Exception:
            return None
        return scalar if math.isfinite(scalar) else None

    def _extract_from_info(self, info: Dict[str, Any], kind: str) -> tuple[float | None, str | None]:
        exact_candidates = {
            "temp": (
                "zone_temp",
                "zone_temperature",
                "air_temperature",
                "indoor_temperature",
                "temperature",
            ),
            "power": (
                "hvac_power",
                "total_hvac_power",
                "facility_total_hvac_electric_demand_power",
                "electric_demand_power",
                "power",
            ),
        }[kind]

        keyword_groups = {
            "temp": (("zone", "temp"), ("air", "temp"), ("indoor", "temp")),
            "power": (("hvac", "power"), ("facility", "power"), ("electric", "power")),
        }[kind]

        for key in exact_candidates:
            scalar = self._safe_scalar(info.get(key))
            if scalar is not None:
                return scalar, key

        for key, value in info.items():
            lowered = str(key).lower()
            if any(all(token in lowered for token in group) for group in keyword_groups):
                scalar = self._safe_scalar(value)
                if scalar is not None:
                    return scalar, key
        return None, None

    def _find_wrapper_instance(self, cls: type) -> Any | None:
        cursor = self.env
        visited: set[int] = set()
        while cursor is not None and id(cursor) not in visited:
            visited.add(id(cursor))
            if isinstance(cursor, cls):
                return cursor
            cursor = getattr(cursor, "env", None)
        return None

    def _discover_observation_variables(self) -> list[str]:
        cursor = self.env
        visited: set[int] = set()
        while cursor is not None and id(cursor) not in visited:
            visited.add(id(cursor))
            variables = getattr(cursor, "variables", None)
            if isinstance(variables, dict):
                obs_names = variables.get("observation")
                if isinstance(obs_names, (list, tuple)) and obs_names and all(
                    isinstance(item, str) for item in obs_names
                ):
                    return list(obs_names)
            original_obs = getattr(cursor, "original_obs", None)
            if isinstance(original_obs, (list, tuple)) and original_obs and all(
                isinstance(item, str) for item in original_obs
            ):
                return list(original_obs)
            cursor = getattr(cursor, "env", None)
        return []

    @staticmethod
    def _normalize_obs_name(name: str) -> str:
        return " ".join(str(name).strip().lower().split())

    def _resolve_observation_index(self, *, preferred_name: str, fallback_index: int) -> tuple[int, str]:
        if self.observation_variables:
            normalized_preferred = self._normalize_obs_name(preferred_name)
            for idx, name in enumerate(self.observation_variables):
                if self._normalize_obs_name(name) == normalized_preferred:
                    return idx, f"name:{name}"

            for idx, name in enumerate(self.observation_variables):
                normalized_name = self._normalize_obs_name(name)
                if normalized_preferred in normalized_name or normalized_name in normalized_preferred:
                    return idx, f"partial-name:{name}"

        return fallback_index, f"fallback-index:{fallback_index}"

    def _lookup_observation_name(self, index: int) -> str | None:
        if 0 <= index < len(self.observation_variables):
            return self.observation_variables[index]
        return None

    def step(self, action: Any):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        if info is None:
            info = {}
        else:
            info = dict(info)

        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        info["action_raw"] = raw_action.copy()

        rate_wrapper = self._find_wrapper_instance(ActionRateLimitWrapper)
        normalize_wrapper = self._find_wrapper_instance(ActionNormalizeWrapper)
        safety_wrapper = self._find_wrapper_instance(ActionSafetyWrapper)

        if rate_wrapper is not None and getattr(rate_wrapper, "last_output_action", None) is not None:
            info["action_rate_limited"] = np.asarray(rate_wrapper.last_output_action, dtype=np.float32).reshape(-1).copy()
        if normalize_wrapper is not None and getattr(normalize_wrapper, "last_output_action", None) is not None:
            info["action_physical_pre_safety"] = np.asarray(
                normalize_wrapper.last_output_action, dtype=np.float32
            ).reshape(-1).copy()
        if safety_wrapper is not None and getattr(safety_wrapper, "last_output_action", None) is not None:
            info["action_physical_final"] = np.asarray(
                safety_wrapper.last_output_action, dtype=np.float32
            ).reshape(-1).copy()

        zone_temp_obs = float(obs[self.temp_index])
        hvac_power_obs = float(obs[self.energy_index])
        zone_temp_info, zone_temp_key = self._extract_from_info(info, "temp")
        hvac_power_info, hvac_power_key = self._extract_from_info(info, "power")

        zone_temp = zone_temp_info if zone_temp_info is not None else zone_temp_obs
        hvac_power = hvac_power_info if hvac_power_info is not None else hvac_power_obs

        if zone_temp < -40 or zone_temp > 80:
            zone_temp = zone_temp_info if zone_temp_info is not None else zone_temp_obs

        action_vec = info.get("action_rate_limited")
        action_mag = None
        if action_vec is not None:
            action_arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
            if action_arr.size:
                action_mag = float(np.mean(np.abs(action_arr)))

        scalar, vec = self.morl.compute(zone_temp=zone_temp, hvac_power=hvac_power, action_mag=action_mag)

        info["reward_vector"] = vec
        info["zone_temp"] = zone_temp
        info["hvac_power"] = hvac_power
        info["zone_temp_from_obs"] = zone_temp_obs
        info["hvac_power_from_obs"] = hvac_power_obs
        info["zone_temp_from_info"] = zone_temp_info
        info["hvac_power_from_info"] = hvac_power_info
        info["zone_temp_obs_name"] = self.temp_observation_name
        info["hvac_power_obs_name"] = self.energy_observation_name
        info["zone_temp_index"] = self.temp_index
        info["hvac_power_index"] = self.energy_index
        info["zone_temp_index_resolve_source"] = self.temp_resolve_source
        info["hvac_power_index_resolve_source"] = self.energy_resolve_source
        info["zone_temp_source"] = f"info:{zone_temp_key}" if zone_temp_key else f"obs[{self.temp_index}]"
        info["hvac_power_source"] = f"info:{hvac_power_key}" if hvac_power_key else f"obs[{self.energy_index}]"

        return obs, scalar, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
