from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter


class RuntimeShield:
    def __init__(
        self,
        model_path: str,
        surrogate_kind: str = "legacy_v3",
        surrogate_summary_json: str | None = None,
        surrogate_checkpoint: str | None = None,
        surrogate_base_model: str | None = None,
        horizon: int = 3,
        t_low: float = 21.0,
        t_high: float = 25.0,
        margin: float = 0.82,
        delta_a: float = 0.05,
        max_iterations: int = 20,
    ):
        self.model = load_direct_tsup_adapter(
            kind=surrogate_kind,
            legacy_model_path=model_path,
            summary_json=surrogate_summary_json,
            checkpoint_path=surrogate_checkpoint,
            base_model_path=surrogate_base_model,
            device="cpu",
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.horizon = int(horizon)
        self.t_low = float(t_low)
        self.t_high = float(t_high)
        self.margin = float(margin)
        self.t_safe_low = self.t_low + self.margin
        self.t_safe_high = self.t_high - self.margin
        self.delta_a = float(delta_a)
        self.max_iterations = int(max_iterations)

        self.total_calls = 0
        self.accepted_calls = 0
        self.corrected_calls = 0
        self.emergency_calls = 0
        self.total_corrections = 0

    def _predict_trajectory(
        self,
        action: np.ndarray,
        state: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_zone = float(state["t_zone"])
        t_amb = float(state.get("t_amb", 10.0))
        hour = float(state.get("hour", 12.0))
        day = float(state.get("day", 180.0))
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        temps = np.zeros(self.horizon, dtype=np.float32)
        powers = np.zeros(self.horizon, dtype=np.float32)
        t_curr = float(t_zone)

        with torch.no_grad():
            for idx in range(self.horizon):
                h = (hour + idx) % 24.0
                d = day + idx / 24.0
                t_next, p_total = self.model(
                    torch.tensor([t_curr], dtype=torch.float32),
                    torch.tensor([t_amb], dtype=torch.float32),
                    torch.tensor([h], dtype=torch.float32),
                    torch.tensor([d], dtype=torch.float32),
                    torch.tensor([a0], dtype=torch.float32),
                    torch.tensor([a1], dtype=torch.float32),
                )
                t_curr = float(t_next[0])
                temps[idx] = t_curr
                powers[idx] = float(p_total[0])

        return temps, powers

    def _check_safety(self, temps: np.ndarray) -> Tuple[bool, str]:
        min_t = float(np.min(temps))
        max_t = float(np.max(temps))
        if min_t < self.t_safe_low:
            return False, "cold"
        if max_t > self.t_safe_high:
            return False, "hot"
        return True, "none"

    def _get_worst_case_violation(self, temps: np.ndarray) -> float:
        cold_violation = max(0.0, self.t_safe_low - float(np.min(temps)))
        hot_violation = max(0.0, float(np.max(temps)) - self.t_safe_high)
        if cold_violation > hot_violation:
            return cold_violation
        if hot_violation > cold_violation:
            return -hot_violation
        return 0.0

    def shield(
        self,
        action_ppo: np.ndarray,
        state: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict]:
        self.total_calls += 1
        action = action_ppo.copy().astype(np.float32)

        temps, powers = self._predict_trajectory(action, state)
        is_safe, _ = self._check_safety(temps)

        if is_safe:
            self.accepted_calls += 1
            return action, {
                "safe": True,
                "source": "ppo",
                "iterations": 0,
                "predicted_temps": temps,
            }

        for iteration in range(self.max_iterations):
            violation = self._get_worst_case_violation(temps)
            if violation > 0.0:
                action[0] = min(action[0] + self.delta_a, 1.0)
                if violation > 2.0:
                    action[1] = min(action[1] + self.delta_a * 0.5, 1.0)
            else:
                action[0] = max(action[0] - self.delta_a, -1.0)

            temps, powers = self._predict_trajectory(action, state)
            is_safe, _ = self._check_safety(temps)
            if is_safe:
                self.corrected_calls += 1
                self.total_corrections += iteration + 1
                return action, {
                    "safe": True,
                    "source": "corrected",
                    "iterations": iteration + 1,
                    "predicted_temps": temps,
                    "original_action": action_ppo.copy(),
                }

        self.emergency_calls += 1
        t_zone = float(state.get("t_zone", 22.0))
        t_mid = 0.5 * (self.t_low + self.t_high)
        if t_zone < t_mid:
            emergency = np.array([1.0, 1.0], dtype=np.float32)
        else:
            emergency = np.array([-1.0, 0.5], dtype=np.float32)

        return emergency, {
            "safe": False,
            "source": "emergency",
            "iterations": self.max_iterations,
            "predicted_temps": temps,
            "original_action": action_ppo.copy(),
        }

    def get_stats(self) -> Dict:
        total = max(self.total_calls, 1)
        avg_corrections = self.total_corrections / max(self.corrected_calls, 1)
        return {
            "total_calls": self.total_calls,
            "accepted": self.accepted_calls,
            "accepted_pct": self.accepted_calls / total * 100.0,
            "corrected": self.corrected_calls,
            "corrected_pct": self.corrected_calls / total * 100.0,
            "emergency": self.emergency_calls,
            "emergency_pct": self.emergency_calls / total * 100.0,
            "avg_corrections": avg_corrections,
        }

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.accepted_calls = 0
        self.corrected_calls = 0
        self.emergency_calls = 0
        self.total_corrections = 0
