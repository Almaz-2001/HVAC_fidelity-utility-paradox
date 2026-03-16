

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional


class ComfortFallback:
    

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

    def compute(self, state: Dict[str, float]) -> np.ndarray:
        
        t_zone = state.get('t_zone', self.t_mid)

        # Пропорциональный контроль к середине band
        error = self.t_mid - t_zone  # положительный если холодно
        t_range = self.t_high - self.t_low

        # a0: setpoint. Масштабируем ошибку в [-1, 1]
        # Если error > 0 (холодно) → a0 > 0 (греть)
        # Если error < 0 (жарко)  → a0 < 0 (охлаждать)
        a0 = np.clip(error / (t_range / 2.0), -1.0, 1.0)

        # a1: вентилятор. Умеренный уровень + пропорционально ошибке
        a1_base = 2.0 * self.fan_default - 1.0  # convert [0,1] → [-1,1]
        a1 = np.clip(a1_base + 0.3 * abs(error) / t_range, -1.0, 1.0)

        return np.array([a0, a1], dtype=np.float32)


class SurrogateMPCFallback:
    

    def __init__(
        self,
        model_path: str,
        horizon: int = 4,
        n_iters: int = 30,
        lr: float = 0.1,
        t_safe_low: float = 22.1,
        t_safe_high: float = 23.9,
        lambda_safety: float = 10.0,
        lambda_energy: float = 1.0,
    ):
        from surrogate.rc_node_v2 import RCNeuralODEv2

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = RCNeuralODEv2(hidden_dim=checkpoint["hidden_dim"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Замораживаем веса surrogate — оптимизируем только действия
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

    def compute(self, state: Dict[str, float]) -> np.ndarray:
        
        t_zone = state.get('t_zone', 22.0)
        t_amb = state.get('t_amb', 10.0)
        hour = state.get('hour', 12.0)
        day = state.get('day', 180.0)

        # Начальное действие: середина comfort band
        a_opt = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([a_opt], lr=self.lr)

        best_action = np.array([0.0, 0.0], dtype=np.float32)
        best_cost = float('inf')

        for iteration in range(self.n_iters):
            optimizer.zero_grad()

            # Клампируем в [-1, 1] дифференцируемо (через tanh)
            a_clamped = torch.tanh(a_opt)
            a0 = a_clamped[0:1]
            a1 = a_clamped[1:2]

            # Rollout на horizon шагов
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

                # Safety penalty: штраф за выход за границы
                penalty_low = torch.relu(self.t_safe_low - t_curr).sum()
                penalty_high = torch.relu(t_curr - self.t_safe_high).sum()
                safety_penalty = safety_penalty + penalty_low + penalty_high

            # Cost = energy + safety constraint
            cost = (self.lambda_energy * total_power / self.p_max
                    + self.lambda_safety * safety_penalty)

            cost.backward()
            optimizer.step()

            # Сохраняем лучшее
            with torch.no_grad():
                if cost.item() < best_cost:
                    best_cost = cost.item()
                    best_action = torch.tanh(a_opt).detach().numpy().copy()

        return best_action.astype(np.float32)


class HardClampFallback:
    

    def __init__(self, t_low: float = 21.0, t_high: float = 25.0):
        self.t_mid = (t_low + t_high) / 2.0
        t_range = t_high - t_low
        self.a0_mid = 2.0 * (self.t_mid - t_low) / t_range - 1.0

    def compute(self, state: Dict[str, float] = None) -> np.ndarray:
        return np.array([self.a0_mid, 0.0], dtype=np.float32)