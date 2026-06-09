from __future__ import annotations

import numpy as np
import gymnasium as gym


class ActionSafetyWrapper(gym.ActionWrapper):
    """
    Enforce a minimum deadband between heating and cooling setpoints.

    This wrapper targets the legacy Sinergym continuous control interface where
    the first two action dimensions are typically:
    - action[0]: heating setpoint
    - action[1]: cooling setpoint
    """

    def __init__(
        self,
        env: gym.Env,
        deadband: float = 0.5,
        heating_idx: int = 0,
        cooling_idx: int = 1,
    ):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Only Box action_space supported"
        self.deadband = float(deadband)
        self.heating_idx = int(heating_idx)
        self.cooling_idx = int(cooling_idx)
        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)
        self.last_input_action = None
        self.last_output_action = None

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).copy()
        a = np.clip(a, self.low, self.high)
        self.last_input_action = a.copy()

        max_idx = max(self.heating_idx, self.cooling_idx)
        if a.ndim == 1 and a.shape[0] > max_idx:
            heat = float(a[self.heating_idx])
            cool = float(a[self.cooling_idx])

            if heat + self.deadband > cool:
                midpoint = 0.5 * (heat + cool)
                half_gap = 0.5 * self.deadband
                heat = midpoint - half_gap
                cool = midpoint + half_gap

                heat = np.clip(heat, self.low[self.heating_idx], self.high[self.heating_idx])
                cool = np.clip(cool, self.low[self.cooling_idx], self.high[self.cooling_idx])

                if heat + self.deadband > cool:
                    heat_low = self.low[self.heating_idx]
                    heat_high = min(self.high[self.heating_idx], cool - self.deadband)
                    if heat_high >= heat_low:
                        heat = heat_high
                    cool_low = max(self.low[self.cooling_idx], heat + self.deadband)
                    cool_high = self.high[self.cooling_idx]
                    if cool_high >= cool_low:
                        cool = cool_low

                a[self.heating_idx] = np.float32(heat)
                a[self.cooling_idx] = np.float32(cool)

        self.last_output_action = a.copy()
        return a
