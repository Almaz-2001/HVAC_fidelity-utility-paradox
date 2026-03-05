from __future__ import annotations
import numpy as np
import gymnasium as gym

class ActionRateLimitWrapper(gym.ActionWrapper):
    """
    Ограничивает изменение action между шагами.
    Работает в текущем action_space env (например [-1, 1] после NormalizeWrapper).
    """
    def __init__(self, env: gym.Env, max_delta: float = 0.2):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.max_delta = float(max_delta)
        self._prev_action = None

        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        return obs, info

    def action(self, act):
        a = np.asarray(act, dtype=np.float32)
        a = np.clip(a, self.low, self.high)

        if self._prev_action is None:
            self._prev_action = a
            return a

        delta = np.clip(a - self._prev_action, -self.max_delta, self.max_delta)
        limited = self._prev_action + delta
        limited = np.clip(limited, self.low, self.high)

        self._prev_action = limited
        return limited
