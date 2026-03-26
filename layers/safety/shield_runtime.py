

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class RuntimeShield:
    

    def __init__(
        self,
        model_path: str,
        horizon: int = 3,
        t_low: float = 21.0,
        t_high: float = 25.0,
        margin: float = 0.82,
        delta_a: float = 0.05,
        max_iterations: int = 20,
    ):
        from surrogate.rc_node_v2 import RCNeuralODEv2

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = RCNeuralODEv2(hidden_dim=checkpoint.get("hidden_dim", 64))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.horizon = horizon
        self.t_low = t_low
        self.t_high = t_high
        self.margin = margin
        self.t_safe_low = t_low + margin
        self.t_safe_high = t_high - margin
        self.delta_a = delta_a
        self.max_iterations = max_iterations

        # Statistics
        self.total_calls = 0
        self.accepted_calls = 0
        self.corrected_calls = 0
        self.emergency_calls = 0
        self.total_corrections = 0  # sum of iterations needed

    def _predict_trajectory(
        self, action: np.ndarray, state: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict temperature and power trajectory for given action.
        Returns (temps[H], powers[H]).
        """
        t_zone = state['t_zone']
        t_amb = state.get('t_amb', 10.0)
        hour = state.get('hour', 12.0)
        day = state.get('day', 180.0)

        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        temps = np.zeros(self.horizon)
        powers = np.zeros(self.horizon)
        t_curr = t_zone

        with torch.no_grad():
            for i in range(self.horizon):
                h = (hour + i) % 24.0
                d = day + i / 24.0
                t_next, p = self.model(
                    torch.tensor([t_curr], dtype=torch.float32),
                    torch.tensor([t_amb], dtype=torch.float32),
                    torch.tensor([h], dtype=torch.float32),
                    torch.tensor([d], dtype=torch.float32),
                    torch.tensor([a0], dtype=torch.float32),
                    torch.tensor([a1], dtype=torch.float32),
                )
                t_curr = float(t_next[0])
                temps[i] = t_curr
                powers[i] = float(p[0])

        return temps, powers

    def _check_safety(self, temps: np.ndarray) -> Tuple[bool, str]:
        """
        Check if trajectory is safe.
        Returns (is_safe, violation_type).
        violation_type: 'none', 'cold', 'hot'
        """
        # Worst-case check (any step violates → unsafe)
        min_t = temps.min()
        max_t = temps.max()

        if min_t < self.t_safe_low:
            return False, 'cold'
        if max_t > self.t_safe_high:
            return False, 'hot'
        return True, 'none'

    def _get_worst_case_violation(self, temps: np.ndarray) -> float:
        """
        Compute worst-case violation magnitude.
        Positive = too cold, negative = too hot.
        """
        cold_violation = max(0, self.t_safe_low - temps.min())
        hot_violation = max(0, temps.max() - self.t_safe_high)

        if cold_violation > hot_violation:
            return cold_violation   # positive = need to heat more
        elif hot_violation > cold_violation:
            return -hot_violation   # negative = need to cool more
        return 0.0

    def shield(
        self, action_ppo: np.ndarray, state: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main shielding function.

        Args:
            action_ppo: PPO's proposed action [a0, a1] in [-1, 1]
            state: dict with t_zone, t_amb, hour, day

        Returns:
            (corrected_action, info_dict)
        """
        self.total_calls += 1
        action = action_ppo.copy().astype(np.float32)

        # Step 1: Check PPO's original action
        temps, powers = self._predict_trajectory(action, state)
        is_safe, viol_type = self._check_safety(temps)

        if is_safe:
            self.accepted_calls += 1
            return action, {
                'safe': True,
                'source': 'ppo',
                'iterations': 0,
                'predicted_temps': temps,
            }

        # Step 2: Iteratively correct
        for iteration in range(self.max_iterations):
            violation = self._get_worst_case_violation(temps)

            # Adjust a0 (setpoint): positive violation → increase a0 (heat more)
            if violation > 0:
                # Too cold: increase heating
                action[0] = min(action[0] + self.delta_a, 1.0)
                # Also increase fan if very cold
                if violation > 2.0:
                    action[1] = min(action[1] + self.delta_a * 0.5, 1.0)
            else:
                # Too hot: decrease heating
                action[0] = max(action[0] - self.delta_a, -1.0)

            # Re-predict
            temps, powers = self._predict_trajectory(action, state)
            is_safe, viol_type = self._check_safety(temps)

            if is_safe:
                self.corrected_calls += 1
                self.total_corrections += (iteration + 1)
                return action, {
                    'safe': True,
                    'source': 'corrected',
                    'iterations': iteration + 1,
                    'predicted_temps': temps,
                    'original_action': action_ppo.copy(),
                }

        # Step 3: Emergency — max heating if cold, min if hot
        self.emergency_calls += 1
        t_zone = state.get('t_zone', 22.0)
        t_mid = (self.t_low + self.t_high) / 2.0

        if t_zone < t_mid:
            emergency = np.array([1.0, 1.0], dtype=np.float32)
        else:
            emergency = np.array([-1.0, 0.5], dtype=np.float32)

        return emergency, {
            'safe': False,
            'source': 'emergency',
            'iterations': self.max_iterations,
            'predicted_temps': temps,
            'original_action': action_ppo.copy(),
        }

    def get_stats(self) -> Dict:
        """Return shielding statistics."""
        total = max(self.total_calls, 1)
        avg_corrections = (self.total_corrections / max(self.corrected_calls, 1))
        return {
            'total_calls': self.total_calls,
            'accepted': self.accepted_calls,
            'accepted_pct': self.accepted_calls / total * 100,
            'corrected': self.corrected_calls,
            'corrected_pct': self.corrected_calls / total * 100,
            'emergency': self.emergency_calls,
            'emergency_pct': self.emergency_calls / total * 100,
            'avg_corrections': avg_corrections,
        }

    def reset_stats(self):
        self.total_calls = 0
        self.accepted_calls = 0
        self.corrected_calls = 0
        self.emergency_calls = 0
        self.total_corrections = 0