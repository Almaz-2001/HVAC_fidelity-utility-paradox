

from __future__ import annotations

import os
import numpy as np
import torch
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from envs.base_env import HVACBaseEnv


_OBS_LOW  = np.array([15.0,  400.0,    0.0,   0.0], dtype=np.float32)
_OBS_HIGH = np.array([35.0, 2000.0, 5000.0, 500.0], dtype=np.float32)

_T_LOW  = 21.0
_T_HIGH = 25.0


class SurrogateBackend(HVACBaseEnv):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cfg = config

        self.max_episode_steps = int(config.get("max_episode_steps", 2000))
        self.step_count = 0
        self.dt = float(config.get("step_sec", 3600))

        # MORL
        morl = config.get("morl", {})
        self.w_comfort    = float(morl.get("w_comfort", 0.8))
        self.w_energy     = float(morl.get("w_energy", 0.2))
        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.temp_low     = float(morl.get("temp_low", _T_LOW))
        self.temp_high    = float(morl.get("temp_high", _T_HIGH))

        # Domain randomization settings
        dr = config.get("domain_randomization", {})
        self.dr_enabled     = bool(dr.get("enabled", True))
        self.dr_t_init_low  = float(dr.get("t_init_low", 15.0))
        self.dr_t_init_high = float(dr.get("t_init_high", 28.0))
        self.dr_t_amb_base_range  = (float(dr.get("t_amb_base_low", 4.0)),
                                     float(dr.get("t_amb_base_high", 14.0)))
        self.dr_t_amb_amp_range   = (float(dr.get("t_amb_amp_low", 5.0)),
                                     float(dr.get("t_amb_amp_high", 15.0)))
        self.dr_t_amb_diurnal_range = (float(dr.get("diurnal_low", 2.0)),
                                       float(dr.get("diurnal_high", 6.0)))
        self.dr_t_amb_noise_range = (float(dr.get("noise_low", 0.5)),
                                     float(dr.get("noise_high", 2.0)))

        # Spaces
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Load surrogate v2
        surrogate_path = config.get(
            "surrogate_path",
            "/app/outputs/surrogate_v2/rc_node_v2_best.pt"
        )
        self.model = self._load_surrogate(surrogate_path)

        # State
        self._t_zone = 18.0
        self._t_amb  = 5.0
        self._co2    = 800.0
        self._p_cool = 0.0
        self._p_fan  = 0.0
        self._time   = 0.0
        self._start_day = 0.0

        # Domain randomization params (set per episode)
        self._dr_t_amb_base = 8.0
        self._dr_t_amb_amp = 10.0
        self._dr_t_amb_diurnal = 4.0
        self._dr_t_amb_noise = 1.0

        
        self._total_steps     = 0
        self._violation_steps = 0
        self._max_overshoot   = 0.0
        self._max_undershoot  = 0.0

        print(f"[SURROGATE_V2] Loaded: {surrogate_path}")
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
        """
        
        """
        seasonal = self._dr_t_amb_base + self._dr_t_amb_amp * np.sin(
            2 * np.pi * (day - 80) / 365.0
        )
        diurnal = self._dr_t_amb_diurnal * np.sin(
            2 * np.pi * (hour - 6) / 24.0
        )
        noise = np.random.normal(0, self._dr_t_amb_noise)
        return float(np.clip(seasonal + diurnal + noise, -25.0, 40.0))

    def _randomize_domain(self) -> None:
        """Sample new domain parameters for this episode."""
        if not self.dr_enabled:
            self._dr_t_amb_base = 8.0
            self._dr_t_amb_amp = 10.0
            self._dr_t_amb_diurnal = 4.0
            self._dr_t_amb_noise = 1.0
            return

        self._dr_t_amb_base = np.random.uniform(*self.dr_t_amb_base_range)
        self._dr_t_amb_amp = np.random.uniform(*self.dr_t_amb_amp_range)
        self._dr_t_amb_diurnal = np.random.uniform(*self.dr_t_amb_diurnal_range)
        self._dr_t_amb_noise = np.random.uniform(*self.dr_t_amb_noise_range)

    # -----------------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        # Domain randomization — new weather params each episode
        self._randomize_domain()

        # Random initial conditions
        if self.dr_enabled:
            self._t_zone = float(np.random.uniform(self.dr_t_init_low, self.dr_t_init_high))
        else:
            self._t_zone = float(np.clip(np.random.normal(18.0, 2.0), 15.0, 35.0))

        self._co2    = 800.0 + np.random.normal(0, 50)
        self._p_cool = 0.0
        self._p_fan  = 0.0

        # Random start day
        self._start_day = np.random.uniform(0, 365)
        self._time = self._start_day * 86400.0
        self._t_amb = self._get_t_amb(self._hour, self._day)

        self.step_count = 0
        self._total_steps     = 0
        self._violation_steps = 0
        self._max_overshoot   = 0.0
        self._max_undershoot  = 0.0

        obs = self._make_obs()
        return obs, {}

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
        terminated = False
        truncated  = self.step_count >= self.max_episode_steps

        obs = self._make_obs()
        info = {
            "reward_vector": {
                "comfort":    r_comfort,
                "energy":     r_energy,
                "zone_temp":  self._t_zone,
                "hvac_power": p_total,
                "w_comfort":  self.w_comfort,
                "w_energy":   self.w_energy,
            },
            "safety": self.get_safety_metric(),
            "t_amb": self._t_amb,
            "hour":  self._hour,
            "day":   self._day,
        }

        return obs, float(reward), terminated, truncated, info

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
            t_zone = torch.tensor([self._t_zone], dtype=torch.float32)
            t_amb  = torch.tensor([self._t_amb],  dtype=torch.float32)
            hour   = torch.tensor([self._hour],   dtype=torch.float32)
            day    = torch.tensor([self._day],    dtype=torch.float32)
            a0_t   = torch.tensor([a0], dtype=torch.float32)
            a1_t   = torch.tensor([a1], dtype=torch.float32)
            t_next, p_total = self.model(t_zone, t_amb, hour, day, a0_t, a1_t)
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
        r_sev  = max(self._max_overshoot, self._max_undershoot)
        return {
            "r_time": r_time, "r_sev": r_sev, "m_s": r_time + r_sev,
            "violation_steps": self._violation_steps,
            "total_steps": self._total_steps,
        }