"""
envs/backends/surrogate_backend.py

Surrogate backend with REAL weather data from BOPTEST.

Key change: instead of synthetic sinusoidal T_amb, loads actual Belgian
weather data collected from BOPTEST (51,200 steps across 4 seasons).
This eliminates the weather forecast gap identified as root cause of
sim-to-real gap in Phase 3 experiments.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, Tuple

from envs.base_env import HVACBaseEnv


_OBS_LOW  = np.array([5.0,   400.0,    0.0,   0.0], dtype=np.float32)
_OBS_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0], dtype=np.float32)

_T_LOW  = 21.0
_T_HIGH = 25.0

# Path to real weather data
WEATHER_CSV = "/app/data/surrogate_v2/boptest_v2_all.csv"


class RealWeatherProvider:
    """
    Provides real T_amb values from BOPTEST data.

    Builds a lookup table: (day_of_year, hour) → T_amb
    with random noise for domain randomization.
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            print(f"[WEATHER] Real weather not found: {csv_path}, using synthetic fallback")
            self.available = False
            return

        df = pd.read_csv(csv_path)
        self.available = True

        # Build hourly weather profile: 365 days × 24 hours
        # Average across all episodes for each (day, hour) pair
        self.weather_grid = np.full((366, 24), 10.0, dtype=np.float32)
        self.weather_count = np.zeros((366, 24), dtype=np.int32)

        for _, row in df.iterrows():
            day = int(row['day']) % 366
            hour = int(row['hour']) % 24
            t_amb = float(row['t_amb'])
            if -30 < t_amb < 50:  # filter out zeros/defaults
                self.weather_grid[day, hour] += t_amb
                self.weather_count[day, hour] += 1

        # Average
        mask = self.weather_count > 0
        self.weather_grid[mask] /= self.weather_count[mask]

        # Fill gaps with interpolation from neighbors
        for d in range(366):
            for h in range(24):
                if self.weather_count[d, h] == 0:
                    # Use nearest available day
                    for offset in range(1, 30):
                        d_prev = (d - offset) % 366
                        d_next = (d + offset) % 366
                        if self.weather_count[d_prev, h] > 0:
                            self.weather_grid[d, h] = self.weather_grid[d_prev, h]
                            break
                        elif self.weather_count[d_next, h] > 0:
                            self.weather_grid[d, h] = self.weather_grid[d_next, h]
                            break

        coverage = mask.sum() / (366 * 24) * 100
        print(f"[WEATHER] Real weather loaded: {csv_path}")
        print(f"[WEATHER] Coverage: {coverage:.1f}% of (day,hour) grid")
        print(f"[WEATHER] T_amb range: [{self.weather_grid.min():.1f}, {self.weather_grid.max():.1f}]°C")

    def get_t_amb(self, hour: float, day: float, noise_std: float = 1.0) -> float:
        """Get T_amb from real data with small noise for DR."""
        if not self.available:
            return 10.0

        d = int(day) % 366
        h = int(hour) % 24

        # Base value from real data
        t_base = float(self.weather_grid[d, h])

        # Interpolate between hours for smoother transitions
        h_next = (h + 1) % 24
        frac = hour - int(hour)
        t_next = float(self.weather_grid[d, h_next])
        t_interp = t_base * (1 - frac) + t_next * frac

        # Small noise for domain randomization
        noise = np.random.normal(0, noise_std)

        return float(np.clip(t_interp + noise, -30.0, 45.0))


class SurrogateBackend(HVACBaseEnv):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cfg = config

        self.max_episode_steps = int(config.get("max_episode_steps", 2000))
        self.step_count = 0
        self.dt = float(config.get("step_sec", 3600))

        morl = config.get("morl", {})
        self.w_comfort    = float(morl.get("w_comfort", 0.8))
        self.w_energy     = float(morl.get("w_energy", 0.2))
        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.temp_low     = float(morl.get("temp_low", _T_LOW))
        self.temp_high    = float(morl.get("temp_high", _T_HIGH))

        # Domain randomization
        dr = config.get("domain_randomization", {})
        self.dr_enabled     = bool(dr.get("enabled", True))
        self.dr_t_init_low  = float(dr.get("t_init_low", 10.0))
        self.dr_t_init_high = float(dr.get("t_init_high", 30.0))
        self.dr_noise_std   = float(dr.get("weather_noise_std", 1.5))

        # Spaces
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Load surrogate
        surrogate_path = config.get(
            "surrogate_path",
            "/app/outputs/surrogate_v2/rc_node_v2_best.pt"
        )
        self.model = self._load_surrogate(surrogate_path)

        # Load REAL weather data
        weather_path = config.get("weather_csv", WEATHER_CSV)
        self.weather = RealWeatherProvider(weather_path)

        # State
        self._t_zone = 18.0
        self._t_amb  = 5.0
        self._co2    = 800.0
        self._p_cool = 0.0
        self._p_fan  = 0.0
        self._time   = 0.0
        self._start_day = 0.0

        # Safety
        self._total_steps     = 0
        self._violation_steps = 0
        self._max_overshoot   = 0.0
        self._max_undershoot  = 0.0

        print(f"[SURROGATE_V2] Loaded: {surrogate_path}")
        print(f"[SURROGATE_V2] Weather: {'REAL (BOPTEST)' if self.weather.available else 'SYNTHETIC'}")
        print(f"[SURROGATE_V2] Domain randomization: {'ON' if self.dr_enabled else 'OFF'}")

    def _load_surrogate(self, path: str):
        from surrogate.rc_node_v2 import RCNeuralODEv2
        if not os.path.exists(path):
            raise FileNotFoundError(f"Surrogate v2 not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = RCNeuralODEv2(hidden_dim=checkpoint.get("hidden_dim", 64))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def _get_t_amb(self, hour: float, day: float) -> float:
        """Get T_amb from real weather data (with noise for DR)."""
        noise_std = self.dr_noise_std if self.dr_enabled else 0.5
        return self.weather.get_t_amb(hour, day, noise_std=noise_std)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        if self.dr_enabled:
            self._t_zone = float(np.random.uniform(self.dr_t_init_low, self.dr_t_init_high))
        else:
            self._t_zone = float(np.clip(np.random.normal(18.0, 2.0), 15.0, 35.0))

        self._co2    = 800.0 + np.random.normal(0, 50)
        self._p_cool = 0.0
        self._p_fan  = 0.0

        # Random start day (full year coverage)
        self._start_day = np.random.uniform(0, 365)
        self._time = self._start_day * 86400.0
        self._t_amb = self._get_t_amb(self._hour, self._day)

        self.step_count = 0
        self._total_steps     = 0
        self._violation_steps = 0
        self._max_overshoot   = 0.0
        self._max_undershoot  = 0.0

        return self._make_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        self._time += self.dt
        self._t_amb = self._get_t_amb(self._hour, self._day)

        t_next, p_total = self._surrogate_step(a0, a1)

        fan_signal = float(np.clip((a1 + 1.0) / 2.0, 0.0, 1.0))
        self._co2 = float(np.clip(self._co2 - 50.0 * fan_signal + 10.0, 400, 2000))

        self._p_cool = float(p_total * 0.85)
        self._p_fan  = float(p_total * 0.15)
        self._t_zone = float(t_next)

        r_comfort = self._comfort_reward(self._t_zone)
        r_energy  = -self.energy_scale * p_total
        reward    = self.w_comfort * r_comfort + self.w_energy * r_energy

        self._update_safety(self._t_zone)
        self.step_count += 1

        obs = self._make_obs()
        info = {
            "reward_vector": {
                "comfort": r_comfort, "energy": r_energy,
                "zone_temp": self._t_zone, "hvac_power": p_total,
                "w_comfort": self.w_comfort, "w_energy": self.w_energy,
            },
            "safety": self.get_safety_metric(),
            "t_amb": self._t_amb, "hour": self._hour, "day": self._day,
        }
        return obs, float(reward), False, self.step_count >= self.max_episode_steps, info

    def close(self) -> None:
        pass

    @property
    def _hour(self) -> float:
        return (self._time / 3600.0) % 24.0

    @property
    def _day(self) -> float:
        return (self._time / 86400.0) % 365.0

    def _surrogate_step(self, a0: float, a1: float) -> Tuple[float, float]:
        with torch.no_grad():
            t_next, p_total = self.model(
                torch.tensor([self._t_zone], dtype=torch.float32),
                torch.tensor([self._t_amb], dtype=torch.float32),
                torch.tensor([self._hour], dtype=torch.float32),
                torch.tensor([self._day], dtype=torch.float32),
                torch.tensor([a0], dtype=torch.float32),
                torch.tensor([a1], dtype=torch.float32),
            )
        return float(t_next[0]), float(p_total[0])

    def _make_obs(self) -> np.ndarray:
        raw = np.array([self._t_zone, self._co2, self._p_cool, self._p_fan],
                       dtype=np.float32)
        obs = 2.0 * (raw - _OBS_LOW) / (_OBS_HIGH - _OBS_LOW) - 1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _comfort_reward(self, t_c: float) -> float:
        if t_c < self.temp_low:
            return -(self.temp_low - t_c)
        elif t_c > self.temp_high:
            return -(t_c - self.temp_high)
        return 0.0

    def _update_safety(self, t_c: float) -> None:
        self._total_steps += 1
        if t_c > self.temp_high:
            self._violation_steps += 1
            self._max_overshoot = max(self._max_overshoot,
                                      (t_c - self.temp_high) / self.temp_high)
        elif t_c < self.temp_low:
            self._violation_steps += 1
            self._max_undershoot = max(self._max_undershoot,
                                       (self.temp_low - t_c) / self.temp_low)

    def get_safety_metric(self) -> dict:
        if self._total_steps == 0:
            return {"r_time": 0.0, "r_sev": 0.0, "m_s": 0.0,
                    "violation_steps": 0, "total_steps": 0}
        r_time = self._violation_steps / self._total_steps
        r_sev = max(self._max_overshoot, self._max_undershoot)
        return {"r_time": r_time, "r_sev": r_sev, "m_s": r_time + r_sev,
                "violation_steps": self._violation_steps,
                "total_steps": self._total_steps}