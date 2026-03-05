from __future__ import annotations
import numpy as np
import gymnasium as gym

class ActionSafetyWrapper(gym.ActionWrapper):
    """
    Гарантирует физическую корректность setpoints:
      - heat <= cool - deadband
      - клиппинг в bounds env.action_space
    """
    def __init__(self, env: gym.Env, deadband: float = 1.0):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,), "Expected 2D action: [heat, cool]"
        self.deadband = float(deadband)

        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)

    def action(self, act):
        a = np.asarray(act, dtype=np.float32).copy()
        # clip first
        a = np.clip(a, self.low, self.high)

        heat, cool = float(a[0]), float(a[1])

        # enforce ordering
        if heat > cool - self.deadband:
            mid = 0.5 * (heat + cool)
            heat = mid - self.deadband / 2.0
            cool = mid + self.deadband / 2.0

        # clip again after adjustment
        heat = float(np.clip(heat, self.low[0], self.high[0]))
        cool = float(np.clip(cool, self.low[1], self.high[1]))

        # if still violates because of hard bounds, push cool up / heat down
        if heat > cool - self.deadband:
            # try push cool up
            cool = float(np.clip(heat + self.deadband, self.low[1], self.high[1]))
            # if impossible, push heat down
            heat = float(np.clip(cool - self.deadband, self.low[0], self.high[0]))

        return np.array([heat, cool], dtype=np.float32)
