from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _sample_action(env, rng: np.random.Generator):
    space = env.action_space
    if hasattr(space, "seed"):
        try:
            space.seed(int(rng.integers(0, 2**31 - 1)))
        except Exception:
            pass
    try:
        return space.sample()
    except Exception:
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return rng.uniform(low=low, high=high).astype(np.float32)


def _find_wrapper(obj: Any, attr_name: str) -> Any | None:
    cursor = obj
    visited: set[int] = set()

    while cursor is not None and id(cursor) not in visited:
        visited.add(id(cursor))
        if hasattr(cursor, attr_name):
            return cursor
        cursor = getattr(cursor, "env", None)
    return None


def _extract_zone_temp(env, obs, info: dict[str, Any] | None = None) -> float:
    info = info or {}

    if isinstance(info, dict):
        for key in ("zone_temp", "temperature"):
            value = info.get(key)
            if value is not None:
                return float(value)

    try:
        idx = int(getattr(env, "temp_index"))
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1 and idx < arr.shape[0]:
            return float(arr[idx])
    except Exception:
        pass

    if hasattr(obs, "__len__") and len(obs):
        try:
            return float(np.asarray(obs, dtype=np.float32)[0])
        except Exception:
            pass
    return np.nan


def _extract_band(env) -> tuple[float, float]:
    morl = getattr(env, "morl", None)
    if morl is not None:
        low = getattr(morl, "temp_low", None)
        high = getattr(morl, "temp_high", None)
        if low is not None and high is not None:
            return float(low), float(high)
    cfg = getattr(env, "config", {}) or {}
    morl_cfg = cfg.get("morl", {}) if isinstance(cfg, dict) else {}
    return float(morl_cfg.get("temp_low", 20.0)), float(morl_cfg.get("temp_high", 26.0))


def _safe_physical_setpoints(heat: float, cool: float, deadband: float, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    heat = float(np.clip(heat, low[0], high[0]))
    cool = float(np.clip(cool, low[1], high[1]))
    if heat + deadband > cool:
        midpoint = 0.5 * (heat + cool)
        heat = midpoint - 0.5 * deadband
        cool = midpoint + 0.5 * deadband
        heat = float(np.clip(heat, low[0], high[0]))
        cool = float(np.clip(cool, low[1], high[1]))
        if heat + deadband > cool:
            heat = min(heat, cool - deadband)
            heat = float(np.clip(heat, low[0], high[0]))
            cool = max(cool, heat + deadband)
            cool = float(np.clip(cool, low[1], high[1]))
    return np.asarray([heat, cool], dtype=np.float32)


class RuleBasedSinergymPolicy:
    """
    A simple thermostat-style baseline for the legacy Sinergym branch.

    It tries to keep the zone inside the configured comfort band using a small
    hysteresis window. Actions are generated in physical setpoint space and then
    mapped back into the normalized action space exposed by the wrapped env.
    """

    def __init__(self, env, hysteresis_c: float = 0.5):
        self.env = env
        self.temp_low, self.temp_high = _extract_band(env)
        self.hysteresis_c = float(hysteresis_c)
        self.mid = 0.5 * (self.temp_low + self.temp_high)

        normalize = _find_wrapper(getattr(env, "unwrapped", env), "reverse_action")
        self._normalize = normalize
        self._low = None
        self._high = None

        if normalize is not None and hasattr(normalize, "low") and hasattr(normalize, "high"):
            self._low = np.asarray(normalize.low, dtype=np.float32)
            self._high = np.asarray(normalize.high, dtype=np.float32)

        self.deadband = float(getattr(_find_wrapper(getattr(env, "unwrapped", env), "deadband"), "deadband", 0.5))
        self.mode = "hold"

    def reset(self) -> None:
        self.mode = "hold"

    def _physical_action(self, zone_temp: float) -> np.ndarray:
        if self._low is None or self._high is None or self._low.shape[0] < 2:
            return np.asarray([1.0, 1.0], dtype=np.float32)

        heat_low, cool_low = float(self._low[0]), float(self._low[1])
        heat_high, cool_high = float(self._high[0]), float(self._high[1])

        if np.isnan(zone_temp):
            zone_temp = self.mid

        enter_heat = self.temp_low + 0.25
        exit_heat = self.temp_low + self.hysteresis_c
        enter_cool = self.temp_high - 0.25
        exit_cool = self.temp_high - self.hysteresis_c

        if self.mode != "heat" and zone_temp < enter_heat:
            self.mode = "heat"
        elif self.mode == "heat" and zone_temp >= exit_heat:
            self.mode = "hold"

        if self.mode != "cool" and zone_temp > enter_cool:
            self.mode = "cool"
        elif self.mode == "cool" and zone_temp <= exit_cool:
            self.mode = "hold"

        if self.mode == "heat":
            heat = heat_high
            cool = cool_high
        elif self.mode == "cool":
            heat = heat_low
            cool = cool_low
        else:
            heat = float(np.clip(self.temp_low, heat_low, heat_high))
            cool = float(np.clip(self.temp_high, cool_low, cool_high))

        return _safe_physical_setpoints(heat, cool, self.deadband, self._low, self._high)

    def __call__(self, obs, info: dict[str, Any] | None = None):
        physical = self._physical_action(_extract_zone_temp(self.env, obs, info))
        if self._normalize is not None:
            try:
                return self._normalize.reverse_action(physical)
            except Exception:
                pass
        return physical.astype(np.float32)


def run_baseline(
    env,
    name: str,
    n_steps: int,
    out_dir: str | Path,
    fixed_action: Any = None,
    action_fn: Callable[[Any, dict[str, Any] | None], Any] | None = None,
    seed: int = 42,
) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"

    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    rows: list[dict[str, Any]] = []
    last_info: dict[str, Any] = dict(info or {})

    def _action_component(container, idx: int) -> float:
        if container is None:
            return np.nan
        try:
            arr = np.asarray(container, dtype=np.float32).reshape(-1)
            return float(arr[idx]) if idx < arr.size else np.nan
        except Exception:
            return np.nan

    if action_fn is not None and hasattr(action_fn, "reset"):
        try:
            action_fn.reset()
        except Exception:
            pass

    for step in range(int(n_steps)):
        action = fixed_action
        if action_fn is not None:
            action = action_fn(obs, last_info)
        elif action is None:
            action = _sample_action(env, rng)

        obs, reward, terminated, truncated, info = env.step(action)
        info = dict(info or {})
        rv = info.get("reward_vector") if isinstance(info, dict) else None

        row = {
            "step": step,
            "reward_scalar": float(reward),
            "a0": float(action[0]) if hasattr(action, "__len__") else float(action),
            "a1": float(action[1]) if hasattr(action, "__len__") and len(action) > 1 else np.nan,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        if isinstance(rv, dict):
            row.update(
                {
                    "comfort": rv.get("comfort"),
                    "energy": rv.get("energy"),
                    "zone_temp": rv.get("zone_temp"),
                    "hvac_power": rv.get("hvac_power"),
                    "zone_temp_from_obs": info.get("zone_temp_from_obs"),
                    "zone_temp_from_info": info.get("zone_temp_from_info"),
                    "zone_temp_source": info.get("zone_temp_source"),
                    "hvac_power_from_obs": info.get("hvac_power_from_obs"),
                    "hvac_power_from_info": info.get("hvac_power_from_info"),
                    "hvac_power_source": info.get("hvac_power_source"),
                    "raw_a0": _action_component(info.get("action_raw"), 0),
                    "raw_a1": _action_component(info.get("action_raw"), 1),
                    "limited_a0": _action_component(info.get("action_rate_limited"), 0),
                    "limited_a1": _action_component(info.get("action_rate_limited"), 1),
                    "physical_pre_safety_a0": _action_component(info.get("action_physical_pre_safety"), 0),
                    "physical_pre_safety_a1": _action_component(info.get("action_physical_pre_safety"), 1),
                    "physical_final_a0": _action_component(info.get("action_physical_final"), 0),
                    "physical_final_a1": _action_component(info.get("action_physical_final"), 1),
                    "w_comfort": rv.get("w_comfort"),
                    "w_energy": rv.get("w_energy"),
                }
            )

        rows.append(row)
        last_info = info

        if terminated or truncated:
            obs, info = env.reset(seed=seed)
            last_info = dict(info or {})
            if action_fn is not None and hasattr(action_fn, "reset"):
                try:
                    action_fn.reset()
                except Exception:
                    pass

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)
