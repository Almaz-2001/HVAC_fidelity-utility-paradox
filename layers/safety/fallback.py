"""
layers/safety/fallback.py

Fallback policies for Phase 3 Safe MORL.

v2 improvements:
  - ComfortFallback: aggressive heating when T_zone << T_low
  - ComfortFallback: temperature-aware fan (stronger when further from comfort)
  - SurrogateMPCFallback: higher lambda_safety, comfort-biased initial action
  - HardClampFallback: full heat mode instead of mid-point
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional


class ComfortFallback:
    """
    Proportional fallback with aggressive winter heating.

    Key change vs v1:
      v1: proportional to error, moderate fan → T_zone=10°C gets a0=1.0, a1=-0.1
      v2: quadratic urgency + full fan when far from comfort → fast recovery
    """

    def __init__(
        self,
        t_low: float = 21.0,
        t_high: float = 25.0,
        fan_default: float = 0.3,
    ):
        self.t_low = t_low
        self.t_high = t_high
        self.t_mid = (t_low + t_high) / 2.0
        self.fan_default = fan_default
        self.t_range = t_high - t_low

    def compute(self, state: Dict[str, float]) -> np.ndarray:
        t_zone = state.get('t_zone', self.t_mid)
        t_amb = state.get('t_amb', 10.0)

        error = self.t_mid - t_zone  # positive = too cold, negative = too hot
        abs_error = abs(error)

        # --- a0: setpoint ---
        # Base: proportional control
        a0 = np.clip(error / (self.t_range / 2.0), -1.0, 1.0)

        # Emergency boost: if far from comfort, push to extreme
        if abs_error > 3.0:
            # More than 3°C from midpoint → full power in that direction
            a0 = 1.0 if error > 0 else -1.0
        elif abs_error > 1.5:
            # 1.5-3°C → amplify signal
            a0 = np.clip(a0 * 1.5, -1.0, 1.0)

        # Cold ambient boost: if T_amb < 5°C and building is cold, max heat
        if t_amb < 5.0 and t_zone < self.t_low:
            a0 = 1.0

        # --- a1: fan ---
        # Fan should be HIGH when we need to move temperature quickly
        if abs_error > 3.0:
            # Emergency: full fan
            a1 = 1.0 if error > 0 else 0.8
        elif abs_error > 1.0:
            # Active correction
            a1_raw = 0.5 + 0.3 * (abs_error / self.t_range)
            a1 = np.clip(2.0 * a1_raw - 1.0, -1.0, 1.0)  # convert to [-1,1]
        else:
            # Near comfort: moderate fan
            a1_base = 2.0 * self.fan_default - 1.0
            a1 = np.clip(a1_base + 0.2 * abs_error, -1.0, 1.0)

        return np.array([a0, a1], dtype=np.float32)


class SurrogateMPCFallback:
    """
    Gradient-based optimization through differentiable surrogate.

    v2 improvements:
      - lambda_safety: 10 → 50 (prioritize safety over energy)
      - Initial action: biased toward heating if T_zone < T_mid
      - More iterations: 30 → 50
    """

    def __init__(
        self,
        model_path: str,
        surrogate_kind: str = "legacy_v3",
        surrogate_summary_json: Optional[str] = None,
        surrogate_checkpoint: Optional[str] = None,
        surrogate_base_model: Optional[str] = None,
        horizon: int = 4,
        n_iters: int = 50,
        lr: float = 0.1,
        t_safe_low: float = 22.1,
        t_safe_high: float = 23.9,
        lambda_safety: float = 50.0,
        lambda_energy: float = 1.0,
    ):
        from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

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

        self.horizon = horizon
        self.n_iters = n_iters
        self.lr = lr
        self.t_safe_low = t_safe_low
        self.t_safe_high = t_safe_high
        self.lambda_safety = lambda_safety
        self.lambda_energy = lambda_energy
        self.p_max = self.model.P_MAX
        self.t_mid = (t_safe_low + t_safe_high) / 2.0

    def compute(self, state: Dict[str, float]) -> np.ndarray:
        t_zone = state.get('t_zone', 22.0)
        t_amb = state.get('t_amb', 10.0)
        hour = state.get('hour', 12.0)
        day = state.get('day', 180.0)

        # Smart initial action: bias toward heating if cold, cooling if hot
        if t_zone < self.t_mid:
            init_a0 = min((self.t_mid - t_zone) / 3.0, 0.8)
            init_a1 = 0.3
        elif t_zone > self.t_mid:
            init_a0 = max(-(t_zone - self.t_mid) / 3.0, -0.8)
            init_a1 = 0.3
        else:
            init_a0, init_a1 = 0.0, 0.0

        # Use atanh to map initial action to unconstrained space
        init_a0_unc = np.arctanh(np.clip(init_a0, -0.99, 0.99))
        init_a1_unc = np.arctanh(np.clip(init_a1, -0.99, 0.99))

        a_opt = torch.tensor([init_a0_unc, init_a1_unc],
                             dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([a_opt], lr=self.lr)

        best_action = np.array([init_a0, init_a1], dtype=np.float32)
        best_cost = float('inf')

        for iteration in range(self.n_iters):
            optimizer.zero_grad()

            a_clamped = torch.tanh(a_opt)
            a0 = a_clamped[0:1]
            a1 = a_clamped[1:2]

            t_curr = torch.tensor([t_zone], dtype=torch.float32)
            total_power = torch.tensor(0.0)
            safety_penalty = torch.tensor(0.0)

            for step in range(self.horizon):
                h = (hour + step) % 24.0
                d = day + (step / 24.0)

                t_amb_t = torch.tensor([t_amb], dtype=torch.float32)
                hour_t = torch.tensor([h], dtype=torch.float32)
                day_t = torch.tensor([d], dtype=torch.float32)

                t_curr, p = self.model(t_curr, t_amb_t, hour_t, day_t, a0, a1)
                total_power = total_power + p.sum()

                # Quadratic safety penalty (stronger push away from boundaries)
                penalty_low = torch.relu(self.t_safe_low - t_curr) ** 2
                penalty_high = torch.relu(t_curr - self.t_safe_high) ** 2
                safety_penalty = safety_penalty + penalty_low.sum() + penalty_high.sum()

            cost = (self.lambda_energy * total_power / self.p_max
                    + self.lambda_safety * safety_penalty)

            cost.backward()
            optimizer.step()

            with torch.no_grad():
                if cost.item() < best_cost:
                    best_cost = cost.item()
                    best_action = torch.tanh(a_opt).detach().numpy().copy()

        return best_action.astype(np.float32)


class HardClampFallback:
    """
    Last resort: full heating if cold, mid-point if warm.

    v2 change: context-aware instead of fixed mid-point.
    """

    def __init__(self, t_low: float = 21.0, t_high: float = 25.0):
        self.t_low = t_low
        self.t_high = t_high
        self.t_mid = (t_low + t_high) / 2.0

    def compute(self, state: Dict[str, float] = None) -> np.ndarray:
        if state is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        t_zone = state.get('t_zone', self.t_mid)

        if t_zone < self.t_low:
            # Cold: full heat, full fan
            return np.array([1.0, 1.0], dtype=np.float32)
        elif t_zone > self.t_high:
            # Hot: min heat, moderate fan for cooling
            return np.array([-1.0, 0.5], dtype=np.float32)
        else:
            # In range: gentle mid-point
            return np.array([0.0, -0.4], dtype=np.float32)
