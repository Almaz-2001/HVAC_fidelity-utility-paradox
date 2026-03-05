from __future__ import annotations
import numpy as np
import gymnasium as gym

class ActionNormalizeWrapper(gym.ActionWrapper):
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Only Box action_space supported"
        self._orig_space: gym.spaces.Box = env.action_space

        self.low = self._orig_space.low.astype(np.float32)
        self.high = self._orig_space.high.astype(np.float32)

        # Новый action_space для агента
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._orig_space.shape, dtype=np.float32
        )

    def action(self, act):
        a = np.asarray(act, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        
        scaled = self.low + (a + 1.0) * 0.5 * (self.high - self.low)
        return scaled.astype(np.float32)

    def reverse_action(self, act):
        x = np.asarray(act, dtype=np.float32)
        
        a = 2.0 * (x - self.low) / (self.high - self.low) - 1.0
        return np.clip(a, -1.0, 1.0).astype(np.float32)
