from __future__ import annotations

import numpy as np
import gymnasium as gym


class ActionNormalizeWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        parameterization: str = "direct",
        min_deadband: float = 0.5,
    ):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Only Box action_space supported"
        self._orig_space: gym.spaces.Box = env.action_space
        self.parameterization = str(parameterization or "direct").lower()
        self.min_deadband = float(min_deadband)
        self.last_input_action = None
        self.last_output_action = None

        self.low = self._orig_space.low.astype(np.float32)
        self.high = self._orig_space.high.astype(np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._orig_space.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _scale_unit_to_range(value: float, low: float, high: float) -> float:
        return float(low + (value + 1.0) * 0.5 * (high - low))

    @staticmethod
    def _scale_range_to_unit(value: float, low: float, high: float) -> float:
        if np.isclose(high, low):
            return 0.0
        return float(2.0 * (value - low) / (high - low) - 1.0)

    def _action_direct(self, a: np.ndarray) -> np.ndarray:
        return self.low + (a + 1.0) * 0.5 * (self.high - self.low)

    def _action_midpoint_gap(self, a: np.ndarray) -> np.ndarray:
        if a.size < 2:
            return self._action_direct(a)

        heat_low, cool_low = float(self.low[0]), float(self.low[1])
        heat_high, cool_high = float(self.high[0]), float(self.high[1])

        gap_min = max(float(self.min_deadband), cool_low - heat_high, 0.0)
        gap_max = max(gap_min, cool_high - heat_low)

        midpoint_base_low = max(heat_low + 0.5 * gap_min, cool_low - 0.5 * gap_min)
        midpoint_base_high = min(heat_high + 0.5 * gap_min, cool_high - 0.5 * gap_min)

        desired_midpoint = self._scale_unit_to_range(float(a[0]), midpoint_base_low, midpoint_base_high)
        desired_gap = self._scale_unit_to_range(float(a[1]), gap_min, gap_max)
        half_gap = 0.5 * desired_gap

        midpoint_low = max(heat_low + half_gap, cool_low - half_gap)
        midpoint_high = min(heat_high + half_gap, cool_high - half_gap)
        midpoint = float(np.clip(desired_midpoint, midpoint_low, midpoint_high))

        heat = float(np.clip(midpoint - half_gap, heat_low, heat_high))
        cool = float(np.clip(midpoint + half_gap, cool_low, cool_high))

        out = np.asarray(a, dtype=np.float32).copy()
        out[0] = np.float32(heat)
        out[1] = np.float32(cool)
        return out

    def action(self, act):
        a = np.asarray(act, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        self.last_input_action = a.copy()

        if self.parameterization == "midpoint_gap":
            scaled = self._action_midpoint_gap(a)
        else:
            scaled = self._action_direct(a)

        self.last_output_action = scaled.astype(np.float32)
        return self.last_output_action

    def reverse_action(self, act):
        x = np.asarray(act, dtype=np.float32)
        if self.parameterization != "midpoint_gap" or x.size < 2:
            a = 2.0 * (x - self.low) / (self.high - self.low) - 1.0
            return np.clip(a, -1.0, 1.0).astype(np.float32)

        heat_low, cool_low = float(self.low[0]), float(self.low[1])
        heat_high, cool_high = float(self.high[0]), float(self.high[1])

        gap_min = max(float(self.min_deadband), cool_low - heat_high, 0.0)
        gap_max = max(gap_min, cool_high - heat_low)
        midpoint_base_low = max(heat_low + 0.5 * gap_min, cool_low - 0.5 * gap_min)
        midpoint_base_high = min(heat_high + 0.5 * gap_min, cool_high - 0.5 * gap_min)

        heat = float(np.clip(x[0], heat_low, heat_high))
        cool = float(np.clip(x[1], cool_low, cool_high))
        gap = float(np.clip(cool - heat, gap_min, gap_max))
        midpoint = 0.5 * (heat + cool)

        out = np.asarray(x, dtype=np.float32).copy()
        out[0] = np.float32(
            np.clip(self._scale_range_to_unit(midpoint, midpoint_base_low, midpoint_base_high), -1.0, 1.0)
        )
        out[1] = np.float32(np.clip(self._scale_range_to_unit(gap, gap_min, gap_max), -1.0, 1.0))
        return out
