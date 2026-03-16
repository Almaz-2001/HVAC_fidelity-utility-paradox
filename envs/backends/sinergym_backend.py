# envs/backends/sinergym_backend.py
from __future__ import annotations

from typing import Any, Dict
import socket as pysocket

import gymnasium as gym
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

    def __init__(self, config: dict):
        HVACBaseEnv.__init__(self, config)

        self.env_id = config.get("env_id", "Eplus-5Zone-hot-continuous-v1")

        
        pysocket.gethostname = lambda: "127.0.0.1"

        
        env = gym.make(self.env_id, disable_env_checker=True)

        
        while hasattr(env, "env"):
            env = env.env

        
        normalize_action = bool(config.get("normalize_action", True))
        rate_limit = bool(config.get("rate_limit", True))
        max_delta = float(config.get("max_delta", 0.15))
        deadband = float(config.get("deadband", 0.5))  

       
        env = ActionSafetyWrapper(env, deadband=deadband)

       
        if normalize_action:
            env = ActionNormalizeWrapper(env)

        
        if rate_limit:
            env = ActionRateLimitWrapper(env, max_delta=max_delta)

        # финальная среда
        self.env: gym.Env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        # ---------- MORL ----------
        morl_cfg: Dict[str, Any] = config.get("morl", {}) or {}

        self.temp_index = int(morl_cfg.get("temp_index", 10))
        self.energy_index = int(morl_cfg.get("energy_index", 16))

        self.morl = MORLReward(
            temp_low=float(morl_cfg.get("temp_low", 20.0)),
            temp_high=float(morl_cfg.get("temp_high", 26.0)),
            energy_scale=float(morl_cfg.get("energy_scale", 1e-6)),
            weights=MORLWeights(
                comfort=float(morl_cfg.get("w_comfort", 0.6)),
                energy=float(morl_cfg.get("w_energy", 0.4)),
            ),
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

    def step(self, action: Any):
        """
        
        
        """
        obs, _reward, terminated, truncated, info = self.env.step(action)

        if info is None:
            info = {}
        else:
            info = dict(info)

        
        info["action_final"] = action

        
        zone_temp = float(obs[self.temp_index])
        hvac_power = float(obs[self.energy_index])

        
        if zone_temp < -40 or zone_temp > 80:
            fb = info.get("zone_temp") or info.get("temperature")
            if fb is not None:
                zone_temp = float(fb)

        scalar, vec = self.morl.compute(zone_temp=zone_temp, hvac_power=hvac_power)

        info["reward_vector"] = vec
        info["zone_temp"] = zone_temp
        info["hvac_power"] = hvac_power

        return obs, scalar, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
