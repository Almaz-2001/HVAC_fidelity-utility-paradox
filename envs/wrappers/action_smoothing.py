from __future__ import annotations
import numpy as np
import gymnasium as gym


class ActionSmoothingWrapper(gym.ActionWrapper):
    """
    Ограничивает:
    1) скорость изменения действия (rate limit)
    2) мелкие колебания (deadband)
    """

    def __init__(
        self,
        env: gym.Env,
        max_delta: float = 0.1,
        deadband: float = 0.05,
    ):
        super().__init__(env)
        self.max_delta = float(max_delta)
        self.deadband = float(deadband)
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)

        if self.prev_action is None:
            self.prev_action = action.copy()
            return action

        # --- Deadband ---
        delta = action - self.prev_action
        delta[np.abs(delta) < self.deadband] = 0.0

        # --- Rate limit ---
        delta = np.clip(delta, -self.max_delta, self.max_delta)

        smoothed_action = self.prev_action + delta
        self.prev_action = smoothed_action.copy()

        return smoothed_action
