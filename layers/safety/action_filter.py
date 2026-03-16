

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional


class SurrogateSafetyFilter:
    

    def __init__(
        self,
        model_path: str,
        horizon: int = 4,
        t_low: float = 21.0,
        t_high: float = 25.0,
        margin: float = 1.11,
        fallback=None,
    ):
        from surrogate.rc_node_v2 import RCNeuralODEv2

        # Загружаем surrogate
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = RCNeuralODEv2(hidden_dim=checkpoint["hidden_dim"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.horizon = horizon
        self.t_low = t_low
        self.t_high = t_high
        self.margin = margin

        # Эффективные границы с учётом margin
        self.t_safe_low = t_low + margin
        self.t_safe_high = t_high - margin

        # Fallback policy
        from layers.safety.fallback import SurrogateMPCFallback
        self.fallback = fallback or SurrogateMPCFallback(
            model_path=model_path,
            horizon=horizon,
            t_safe_low=self.t_safe_low,
            t_safe_high=self.t_safe_high,
        )

        # Статистика
        self._total_calls = 0
        self._accepted = 0
        self._rejected = 0

        print(f"[SAFETY] Initialized SurrogateSafetyFilter:")
        print(f"  Horizon:       {horizon} steps ({horizon}h)")
        print(f"  Comfort band:  [{t_low}, {t_high}] °C")
        print(f"  Safety margin: {margin:.2f} °C")
        print(f"  Effective band:[{self.t_safe_low:.1f}, {self.t_safe_high:.1f}] °C")

    def filter(
        self,
        action_ppo: np.ndarray,
        state: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        self._total_calls += 1

        # Текущее состояние
        t_zone = state['t_zone']
        t_amb = state.get('t_amb', 10.0)
        hour = state.get('hour', 12.0)
        day = state.get('day', 180.0)

        # Rollout с действием PPO (повторяем одно действие на весь горизонт)
        is_safe, t_trajectory = self._check_safety(
            t_zone, t_amb, hour, day,
            float(action_ppo[0]), float(action_ppo[1])
        )

        if is_safe:
            self._accepted += 1
            return action_ppo, {
                'safe': True,
                'source': 'ppo',
                't_trajectory': t_trajectory,
                'min_t': min(t_trajectory),
                'max_t': max(t_trajectory),
            }
        else:
            # PPO действие небезопасно → fallback
            self._rejected += 1
            action_safe = self.fallback.compute(state)

            # Проверяем fallback тоже (для логирования)
            _, t_traj_fallback = self._check_safety(
                t_zone, t_amb, hour, day,
                float(action_safe[0]), float(action_safe[1])
            )

            return action_safe, {
                'safe': False,
                'source': 'fallback',
                't_trajectory_ppo': t_trajectory,
                't_trajectory_fallback': t_traj_fallback,
                'min_t_ppo': min(t_trajectory),
                'max_t_ppo': max(t_trajectory),
                'violation_step': self._find_violation_step(t_trajectory),
            }

    def _check_safety(
        self,
        t_zone: float,
        t_amb: float,
        hour: float,
        day: float,
        a0: float,
        a1: float,
    ) -> Tuple[bool, list]:
        
        trajectory = []
        t_curr = torch.tensor([t_zone], dtype=torch.float32)

        for step in range(self.horizon):
            # Время сдвигается на каждом шаге (1 час = 1 step)
            h = (hour + step) % 24.0
            d = day + (step / 24.0)

            t_amb_t = torch.tensor([t_amb], dtype=torch.float32)
            hour_t = torch.tensor([h], dtype=torch.float32)
            day_t = torch.tensor([d], dtype=torch.float32)
            a0_t = torch.tensor([a0], dtype=torch.float32)
            a1_t = torch.tensor([a1], dtype=torch.float32)

            with torch.no_grad():
                t_curr, _ = self.model(t_curr, t_amb_t, hour_t, day_t, a0_t, a1_t)

            trajectory.append(t_curr.item())

        # Проверяем все точки траектории
        is_safe = all(
            self.t_safe_low <= t <= self.t_safe_high
            for t in trajectory
        )

        return is_safe, trajectory

    def _find_violation_step(self, trajectory: list) -> int:
        """Находит первый шаг, где температура выходит за границы."""
        for i, t in enumerate(trajectory):
            if t < self.t_safe_low or t > self.t_safe_high:
                return i + 1
        return -1

    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы фильтра."""
        total = max(self._total_calls, 1)
        return {
            'total_calls': self._total_calls,
            'accepted': self._accepted,
            'rejected': self._rejected,
            'acceptance_rate': self._accepted / total * 100,
            'rejection_rate': self._rejected / total * 100,
        }

    def reset_stats(self) -> None:
        self._total_calls = 0
        self._accepted = 0
        self._rejected = 0

    def print_stats(self) -> None:
        s = self.get_stats()
        print(f"\n[SAFETY] Filter Statistics:")
        print(f"  Total calls:     {s['total_calls']}")
        print(f"  Accepted (PPO):  {s['accepted']} ({s['acceptance_rate']:.1f}%)")
        print(f"  Rejected (FB):   {s['rejected']} ({s['rejection_rate']:.1f}%)")