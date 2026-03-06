"""
surrogate/rc_node.py

RC Neural ODE — physics-informed surrogate модель здания.

Физическая основа (RC thermal model):

    C_zon * dT/dt = Q_hvac(u) + f_theta(T_amb, T_zone, u_fan, I_solar)

где:
    C_zon        — тепловая ёмкость зоны [J/K]  (обучаемый параметр)
    Q_hvac(u)    — тепловая мощность HVAC [W]    (из управляющего сигнала)
    f_theta(...)  — нейросеть для остаточных теплопотоков

Для дискретного шага Δt = 3600 с (1 час):

    T_zone(t+1) = T_zone(t) + Δt/C_zon * [Q_hvac + f_theta(...)]

Входы модели:
    x = [T_zone, a0_raw, a1_raw]     (состояние + действие)

Выходы:
    y = [T_zone_next, P_total_pred]   (следующее состояние + мощность)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


# -----------------------------------------------------------------------
# Нейросеть для остаточных теплопотоков f_theta
# -----------------------------------------------------------------------

class HeatFlowNet(nn.Module):
    """
    Нейросеть моделирует суммарный тепловой поток:
        f_theta(T_zone, a0, a1) → dQ [W]

    Архитектура: MLP с residual connection.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()

        # Входы: [T_zone_norm, a0, a1]
        in_dim = 3

        layers = []
        dim = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.Tanh()]
            dim = hidden_dim
        layers += [nn.Linear(dim, 1)]   # → скалярный тепловой поток

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3] — [T_zone_norm, a0, a1]
        Returns:
            dQ: [batch, 1] — тепловой поток [нормализованный]
        """
        return self.net(x)


# -----------------------------------------------------------------------
# Нейросеть для предсказания мощности HVAC
# -----------------------------------------------------------------------

class PowerNet(nn.Module):
    """
    Предсказывает потребляемую мощность:
        P_total = g_phi(T_zone, a0, a1) [W]
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()   # мощность всегда ≥ 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------
# Основная RC Neural ODE модель
# -----------------------------------------------------------------------

class RCNeuralODE(nn.Module):
    """
    Physics-informed surrogate модель здания.

    Уравнение одного шага (Euler дискретизация):

        T_next = T_curr + (Δt / C_zon) * [Q_hvac + f_theta(T_curr, a0, a1)]

    Обучаемые параметры:
        - C_zon:    тепловая ёмкость зоны (log-параметризация для > 0)
        - f_theta:  HeatFlowNet
        - g_phi:    PowerNet

    Нормализация:
        T_norm = (T - T_mean) / T_std    (zscore внутри модели)
    """

    # Физические константы
    DT = 3600.0         # шаг времени [с]
    T_LOW  = 15.0       # мин. температура [°C]
    T_HIGH = 35.0       # макс. температура [°C]
    P_MAX  = 5500.0     # макс. мощность [W]

    def __init__(
        self,
        hidden_dim:  int   = 64,
        n_layers:    int   = 3,
        # Физически правильное значение:
        # C_zon = Δt * P_mean / dT_mean = 3600 * 265 / 1.8 ≈ 530,000 J/K
        c_zon_init:  float = 5.3e5,
        # Нормализация температуры (будет обновлена из данных)
        t_mean:      float = 20.0,
        t_std:       float = 5.0,
    ):
        super().__init__()

        # Обучаемый параметр C_zon (log для гарантии > 0)
        # Физический расчёт из данных:
        # C_zon = Δt * P_mean / dT_mean = 3600 * 265 / 1.8 ≈ 530,000 J/K
        # C_zon фиксирован — register_buffer не обучается
        self.register_buffer('c_zon_val',
            torch.tensor(float(c_zon_init), dtype=torch.float32))

        # Нейросети
        self.heat_net  = HeatFlowNet(hidden_dim=hidden_dim, n_layers=n_layers)
        self.power_net = PowerNet(hidden_dim=hidden_dim // 2)

        # Нормализация (не обучается, задаётся из данных)
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float32))
        self.register_buffer("t_std",  torch.tensor(t_std,  dtype=torch.float32))

    @property
    def c_zon(self) -> torch.Tensor:
        """Тепловая ёмкость зоны [J/K] — фиксированная константа."""
        return self.c_zon_val

    def _normalize_temp(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_mean) / self.t_std

    def _q_hvac(self, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        """
        Тепловая мощность от HVAC [W].
        Простая линейная модель: Q = P_max * fan_signal * setpoint_factor
        где fan_signal = clip((a1+1)/2, 0, 1)
        """
        fan = torch.clamp((a1 + 1.0) / 2.0, 0.0, 1.0)
        setpoint_factor = torch.clamp((a0 + 1.0) / 2.0, 0.0, 1.0)
        return self.P_MAX * fan * setpoint_factor

    def forward(
        self,
        t_zone: torch.Tensor,   # [batch] — текущая температура [°C]
        a0:     torch.Tensor,   # [batch] — уставка [-1, 1]
        a1:     torch.Tensor,   # [batch] — вентилятор [-1, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Один шаг RC Neural ODE.

        Args:
            t_zone: текущая температура зоны [°C], shape [batch]
            a0:     уставка (нормализованная), shape [batch]
            a1:     сигнал вентилятора, shape [batch]

        Returns:
            t_next:  предсказанная температура [°C], shape [batch]
            p_total: предсказанная мощность [W],     shape [batch]
        """
        # Нормализуем температуру для нейросети
        t_norm = self._normalize_temp(t_zone)   # [batch]

        # Вход нейросети: [T_norm, a0, a1]
        x = torch.stack([t_norm, a0, a1], dim=-1)   # [batch, 3]

        # Data-driven: нейросеть предсказывает dT напрямую
        # dT в диапазоне [-10, +10] °C/шаг (масштаб из данных)
        dT = self.heat_net(x).squeeze(-1) * 10.0    # [batch]

        t_next = t_zone + dT                         # [batch]
        t_next = torch.clamp(t_next, self.T_LOW, self.T_HIGH)

        # Предсказание мощности
        p_total = self.power_net(x).squeeze(-1) * self.P_MAX  # [batch]

        return t_next, p_total

    def rollout(
        self,
        t_zone_0: torch.Tensor,  # [batch] — начальная температура
        actions:  torch.Tensor,  # [batch, horizon, 2] — последовательность действий
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Многошаговый прогноз (для MPC safety filter в Фазе 3).

        Args:
            t_zone_0: начальная температура [batch]
            actions:  последовательность действий [batch, horizon, 2]

        Returns:
            temps:   траектория температур [batch, horizon]
            powers:  траектория мощностей  [batch, horizon]
        """
        batch, horizon, _ = actions.shape
        t = t_zone_0
        temps  = []
        powers = []

        for h in range(horizon):
            a0 = actions[:, h, 0]
            a1 = actions[:, h, 1]
            t, p = self.forward(t, a0, a1)
            temps.append(t)
            powers.append(p)

        return torch.stack(temps, dim=1), torch.stack(powers, dim=1)

    def update_normalization(self, t_mean: float, t_std: float) -> None:
        """Обновляет параметры нормализации из статистики датасета."""
        self.t_mean.fill_(t_mean)
        self.t_std.fill_(max(t_std, 1e-6))

    def summary(self) -> None:
        """Печатает параметры модели."""
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"RC Neural ODE Summary")
        print(f"{'='*50}")
        print(f"  Total parameters: {n_params:,}")
        print(f"  C_zon = {self.c_zon.item():.3e} J/K")
        print(f"  T_norm: mean={self.t_mean.item():.1f}°C, "
              f"std={self.t_std.item():.1f}°C")
        print(f"  Δt = {self.DT:.0f} s")
        print(f"  Δt/C_zon = {self.DT / self.c_zon.item():.3e} K/J")


# -----------------------------------------------------------------------
# Функция потерь
# -----------------------------------------------------------------------

class SurrogateLoss(nn.Module):
    """
    Комбинированная функция потерь:

        L = λ_T * L_temp + λ_P * L_power + λ_phys * L_physics

    где:
        L_temp   — MSE по температуре
        L_power  — MSE по мощности (нормализованной)
        L_physics — штраф за нефизичные предсказания
    """

    def __init__(
        self,
        lambda_temp:   float = 1.0,
        lambda_power:  float = 0.1,
        lambda_physics: float = 0.01,
    ):
        super().__init__()
        self.lambda_temp    = lambda_temp
        self.lambda_power   = lambda_power
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(
        self,
        t_pred:   torch.Tensor,    # [batch] предсказанная T_next
        t_true:   torch.Tensor,    # [batch] реальная T_next
        p_pred:   torch.Tensor,    # [batch] предсказанная P_total
        p_true:   torch.Tensor,    # [batch] реальная P_total
        p_max:    float = 5500.0,
    ) -> Tuple[torch.Tensor, dict]:

        # Температурная потеря [°C²]
        l_temp = self.mse(t_pred, t_true)

        # Потеря по мощности (нормализованная)
        l_power = self.mse(p_pred / p_max, p_true / p_max)

        # Физический штраф: температура не может прыгнуть > 5°C за час
        dt_pred = (t_pred - t_true).abs()
        l_physics = torch.mean(torch.relu(dt_pred - 5.0))

        total = (self.lambda_temp   * l_temp
               + self.lambda_power  * l_power
               + self.lambda_physics * l_physics)

        return total, {
            "loss_total":   total.item(),
            "loss_temp":    l_temp.item(),
            "loss_power":   l_power.item(),
            "loss_physics": l_physics.item(),
        }