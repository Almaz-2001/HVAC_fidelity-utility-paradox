"""

"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict




class HeatFlowNetV2(nn.Module):
    """
    
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64,
                 scale_factor: float = 10.0):
        super().__init__()
        self.scale_factor = scale_factor

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        """
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h2 = h2 + h1                    # residual connection
        h3 = self.block3(h2)
        dT = self.output(h3).squeeze(-1) * self.scale_factor
        return dT




class PowerNetV2(nn.Module):
    """Предсказывает P_total. Softplus гарантирует P ≥ 0."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32,
                 p_max: float = 5500.0):
        super().__init__()
        self.p_max = p_max
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * self.p_max




class RCNeuralODEv2(nn.Module):
    """
    
    """

    # Физические константы
    DT = 3600.0
    P_MAX = 5500.0

    # Границы нормализации
    T_ZONE_MIN, T_ZONE_MAX = 15.0, 35.0
    T_AMB_MIN,  T_AMB_MAX  = -10.0, 40.0

    def __init__(
        self,
        hidden_dim: int = 64,
        p_max: float = 5500.0,
        scale_factor: float = 10.0,
    ):
        super().__init__()
        self.P_MAX = p_max

        self.heat_net = HeatFlowNetV2(
            input_dim=8, hidden_dim=hidden_dim,
            scale_factor=scale_factor
        )
        self.power_net = PowerNetV2(
            input_dim=8, hidden_dim=hidden_dim // 2,
            p_max=p_max
        )

    # --- Нормализация ---

    def _norm_t_zone(self, t: torch.Tensor) -> torch.Tensor:
        """T_zone → [-1, 1]"""
        return 2.0 * (t - self.T_ZONE_MIN) / (self.T_ZONE_MAX - self.T_ZONE_MIN) - 1.0

    def _norm_t_amb(self, t: torch.Tensor) -> torch.Tensor:
        """T_amb → [-1, 1]"""
        return 2.0 * (t - self.T_AMB_MIN) / (self.T_AMB_MAX - self.T_AMB_MIN) - 1.0

    @staticmethod
    def _encode_hour(hour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Час суток → (sin, cos) для циклического кодирования."""
        rad = 2.0 * np.pi * hour / 24.0
        return torch.sin(rad), torch.cos(rad)

    @staticmethod
    def _encode_day(day: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """День года → (sin, cos) для сезонности."""
        rad = 2.0 * np.pi * day / 365.0
        return torch.sin(rad), torch.cos(rad)

    def _build_features(
        self,
        t_zone: torch.Tensor,  
        t_amb:  torch.Tensor,  
        hour:   torch.Tensor,  
        day:    torch.Tensor,  
        a0:     torch.Tensor,  
        a1:     torch.Tensor,  
    ) -> torch.Tensor:
        """Собирает вектор из 8 признаков."""
        t_zone_n = self._norm_t_zone(t_zone)
        t_amb_n  = self._norm_t_amb(t_amb)
        h_sin, h_cos = self._encode_hour(hour)
        d_sin, d_cos = self._encode_day(day)

        return torch.stack([
            t_zone_n, t_amb_n,
            h_sin, h_cos, d_sin, d_cos,
            a0, a1
        ], dim=-1)   # [batch, 8]

    # --- Forward ---

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb:  torch.Tensor,
        hour:   torch.Tensor,
        day:    torch.Tensor,
        a0:     torch.Tensor,
        a1:     torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        """
        x = self._build_features(t_zone, t_amb, hour, day, a0, a1)
        dT = self.heat_net(x)                    # [batch]
        t_next = t_zone + dT
        t_next = torch.clamp(t_next, self.T_ZONE_MIN, self.T_ZONE_MAX)
        p_total = self.power_net(x)              # [batch]
        return t_next, p_total

    

    def rollout(
        self,
        t_zone_0: torch.Tensor,    
        t_amb_seq: torch.Tensor,   
        hour_seq:  torch.Tensor,   
        day_seq:   torch.Tensor,   
        a0_seq:    torch.Tensor,   
        a1_seq:    torch.Tensor,   
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        """
        horizon = a0_seq.shape[1]
        t = t_zone_0
        temps = [t]
        powers = []

        for h in range(horizon):
            t, p = self.forward(
                t, t_amb_seq[:, h], hour_seq[:, h], day_seq[:, h],
                a0_seq[:, h], a1_seq[:, h]
            )
            temps.append(t)
            powers.append(p)

        return torch.stack(temps, dim=1), torch.stack(powers, dim=1)

    

    def forward_v1_compat(
        self,
        t_zone: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        t_amb: float = 10.0,
        hour: float = 12.0,
        day: float = 180.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        batch = t_zone.shape[0]
        device = t_zone.device
        t_amb_t = torch.full((batch,), t_amb, device=device)
        hour_t  = torch.full((batch,), hour,  device=device)
        day_t   = torch.full((batch,), day,   device=device)
        return self.forward(t_zone, t_amb_t, hour_t, day_t, a0, a1)

    def summary(self) -> None:
        n_total = sum(p.numel() for p in self.parameters())
        n_heat  = sum(p.numel() for p in self.heat_net.parameters())
        n_power = sum(p.numel() for p in self.power_net.parameters())
        print(f"\n{'='*55}")
        print(f"RC Neural ODE v2 Summary")
        print(f"{'='*55}")
        print(f"  Input features:   8 (T_zone, T_amb, h_sin, h_cos, "
              f"d_sin, d_cos, a0, a1)")
        print(f"  HeatFlowNetV2:    {n_heat:,} params (residual + LayerNorm)")
        print(f"  PowerNetV2:       {n_power:,} params")
        print(f"  Total:            {n_total:,} params")
        print(f"  T_zone range:     [{self.T_ZONE_MIN}, {self.T_ZONE_MAX}] °C")
        print(f"  T_amb range:      [{self.T_AMB_MIN}, {self.T_AMB_MAX}] °C")
        print(f"  P_max:            {self.P_MAX} W")




class SurrogateLossV2(nn.Module):
    """
    
    """

    def __init__(
        self,
        lambda_temp:    float = 1.0,
        lambda_power:   float = 0.1,
        lambda_multi:   float = 0.5,
        lambda_physics: float = 0.05,
        multi_horizons: list  = None,
        max_delta_T:    float = 5.0,
        p_max:          float = 5500.0,
    ):
        super().__init__()
        self.lambda_temp    = lambda_temp
        self.lambda_power   = lambda_power
        self.lambda_multi   = lambda_multi
        self.lambda_physics = lambda_physics
        self.multi_horizons = multi_horizons or [2, 4]
        self.max_delta_T    = max_delta_T
        self.p_max          = p_max

    def forward(
        self,
        t_pred:  torch.Tensor,    
        t_true:  torch.Tensor,    
        p_pred:  torch.Tensor,    
        p_true:  torch.Tensor,    
        multi_loss: torch.Tensor = None,  
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        
        l_temp = torch.mean((t_pred - t_true) ** 2)
        l_power = torch.mean((p_pred / self.p_max - p_true / self.p_max) ** 2)

        
        dT = (t_pred - t_true).abs()
        l_phys = torch.mean(torch.relu(dT - self.max_delta_T))

        total = (self.lambda_temp * l_temp
                 + self.lambda_power * l_power
                 + self.lambda_physics * l_phys)

        l_multi_val = 0.0
        if multi_loss is not None:
            total = total + self.lambda_multi * multi_loss
            l_multi_val = multi_loss.item()

        return total, {
            "loss_total":   total.item(),
            "loss_temp":    l_temp.item(),
            "loss_power":   l_power.item(),
            "loss_multi":   l_multi_val,
            "loss_physics": l_phys.item(),
        }

    def compute_multi_step_loss(
        self,
        model: RCNeuralODEv2,
        t_zone_seq:  torch.Tensor,  
        t_amb_seq:   torch.Tensor,  
        hour_seq:    torch.Tensor,  
        day_seq:     torch.Tensor,  
        a0_seq:      torch.Tensor,  
        a1_seq:      torch.Tensor,  
    ) -> torch.Tensor:
        """
        
        """
        loss = torch.tensor(0.0, device=t_zone_seq.device)
        max_h = t_amb_seq.shape[1]

        for k in self.multi_horizons:
            if k > max_h:
                continue

            t_current = t_zone_seq[:, 0]   

            for step in range(k):
                t_current, _ = model.forward(
                    t_current,
                    t_amb_seq[:, step],
                    hour_seq[:, step],
                    day_seq[:, step],
                    a0_seq[:, step],
                    a1_seq[:, step],
                )

            
            t_target = t_zone_seq[:, k]
            loss = loss + torch.mean((t_current - t_target) ** 2)

        return loss / len(self.multi_horizons)