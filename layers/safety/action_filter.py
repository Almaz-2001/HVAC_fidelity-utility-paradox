from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter


class SurrogateSafetyFilter:
    def __init__(
        self,
        model_path: str,
        surrogate_kind: str = "legacy_v3",
        surrogate_summary_json: Optional[str] = None,
        surrogate_checkpoint: Optional[str] = None,
        surrogate_base_model: Optional[str] = None,
        horizon: int = 4,
        t_low: float = 21.0,
        t_high: float = 25.0,
        margin: float = 1.11,
        fallback=None,
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

        self.horizon = int(horizon)
        self.t_low = float(t_low)
        self.t_high = float(t_high)
        self.margin = float(margin)
        self.t_safe_low = self.t_low + self.margin
        self.t_safe_high = self.t_high - self.margin

        from layers.safety.fallback import SurrogateMPCFallback

        self.fallback = fallback or SurrogateMPCFallback(
            model_path=model_path,
            surrogate_kind=surrogate_kind,
            surrogate_summary_json=surrogate_summary_json,
            surrogate_checkpoint=surrogate_checkpoint,
            surrogate_base_model=surrogate_base_model,
            horizon=self.horizon,
            t_safe_low=self.t_safe_low,
            t_safe_high=self.t_safe_high,
        )

        self._total_calls = 0
        self._accepted = 0
        self._rejected = 0

        print("[SAFETY] Initialized SurrogateSafetyFilter:")
        print(f"  Horizon:       {self.horizon} steps")
        print(f"  Comfort band:  [{self.t_low}, {self.t_high}] C")
        print(f"  Safety margin: {self.margin:.2f} C")
        print(f"  Effective band:[{self.t_safe_low:.1f}, {self.t_safe_high:.1f}] C")

    def filter(
        self,
        action_ppo: np.ndarray,
        state: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._total_calls += 1

        t_zone = float(state["t_zone"])
        t_amb = float(state.get("t_amb", 10.0))
        hour = float(state.get("hour", 12.0))
        day = float(state.get("day", 180.0))

        is_safe, t_trajectory = self._check_safety(
            t_zone,
            t_amb,
            hour,
            day,
            float(action_ppo[0]),
            float(action_ppo[1]),
        )

        if is_safe:
            self._accepted += 1
            return action_ppo, {
                "safe": True,
                "source": "ppo",
                "t_trajectory": t_trajectory,
                "min_t": min(t_trajectory),
                "max_t": max(t_trajectory),
            }

        self._rejected += 1
        action_safe = self.fallback.compute(state)
        _, t_traj_fallback = self._check_safety(
            t_zone,
            t_amb,
            hour,
            day,
            float(action_safe[0]),
            float(action_safe[1]),
        )

        return action_safe, {
            "safe": False,
            "source": "fallback",
            "t_trajectory_ppo": t_trajectory,
            "t_trajectory_fallback": t_traj_fallback,
            "min_t_ppo": min(t_trajectory),
            "max_t_ppo": max(t_trajectory),
            "violation_step": self._find_violation_step(t_trajectory),
        }

    def _check_safety(
        self,
        t_zone: float,
        t_amb: float,
        hour: float,
        day: float,
        a0: float,
        a1: float,
    ) -> Tuple[bool, list[float]]:
        trajectory: list[float] = []
        t_curr = torch.tensor([t_zone], dtype=torch.float32)

        for step in range(self.horizon):
            h = (hour + step) % 24.0
            d = day + (step / 24.0)

            with torch.no_grad():
                t_curr, _ = self.model(
                    t_curr,
                    torch.tensor([t_amb], dtype=torch.float32),
                    torch.tensor([h], dtype=torch.float32),
                    torch.tensor([d], dtype=torch.float32),
                    torch.tensor([a0], dtype=torch.float32),
                    torch.tensor([a1], dtype=torch.float32),
                )

            trajectory.append(float(t_curr.item()))

        is_safe = all(self.t_safe_low <= t <= self.t_safe_high for t in trajectory)
        return is_safe, trajectory

    def _find_violation_step(self, trajectory: list[float]) -> int:
        for idx, temp in enumerate(trajectory):
            if temp < self.t_safe_low or temp > self.t_safe_high:
                return idx + 1
        return -1

    def get_stats(self) -> Dict[str, Any]:
        total = max(self._total_calls, 1)
        return {
            "total_calls": self._total_calls,
            "accepted": self._accepted,
            "rejected": self._rejected,
            "acceptance_rate": self._accepted / total * 100.0,
            "rejection_rate": self._rejected / total * 100.0,
        }

    def reset_stats(self) -> None:
        self._total_calls = 0
        self._accepted = 0
        self._rejected = 0

    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\n[SAFETY] Filter Statistics:")
        print(f"  Total calls:     {stats['total_calls']}")
        print(f"  Accepted (PPO):  {stats['accepted']} ({stats['acceptance_rate']:.1f}%)")
        print(f"  Rejected (FB):   {stats['rejected']} ({stats['rejection_rate']:.1f}%)")
