from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetV35(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, q_scale: float = 3000.0):
        super().__init__()
        self.q_scale = q_scale
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h2 = h2 + h1
        h3 = self.block3(h2)
        return self.output(h3).squeeze(-1) * self.q_scale


class PowerNetV35(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, p_max: float = 5500.0):
        super().__init__()
        self.p_max = p_max
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * self.p_max


class RCNeuralODEv35(nn.Module):
    DT = 3600.0
    P_MAX = 5500.0
    T_ZONE_MIN, T_ZONE_MAX = 15.0, 35.0
    T_AMB_MIN, T_AMB_MAX = -10.0, 40.0

    def __init__(
        self,
        hidden_dim: int = 64,
        p_max: float = 5500.0,
        q_scale: float = 3000.0,
        c_zon_init: float = 5.3e5,
        c_zon_min: float = 5.0e4,
        c_zon_scale: float = 1.0e5,
        dt_seconds: float | None = None,
    ):
        super().__init__()
        self.P_MAX = p_max
        self.q_scale = q_scale
        self.c_zon_min = c_zon_min
        self.c_zon_scale = c_zon_scale
        self.dt_seconds = float(self.DT if dt_seconds is None else dt_seconds)
        self.q_net = QNetV35(input_dim=8, hidden_dim=hidden_dim, q_scale=q_scale)
        self.power_net = PowerNetV35(input_dim=8, hidden_dim=hidden_dim // 2, p_max=p_max)

        c_min_scaled = float(c_zon_min) / float(c_zon_scale)
        init_scaled = float(c_zon_init) / float(c_zon_scale)
        init_shifted = max(init_scaled - c_min_scaled, 1e-4)
        raw_init = init_shifted if init_shifted > 20.0 else math.log(math.expm1(init_shifted))
        self.log_c_zon = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @property
    def c_zon(self) -> torch.Tensor:
        c_min_scaled = self.c_zon_min / self.c_zon_scale
        return self.c_zon_scale * (F.softplus(self.log_c_zon) + c_min_scaled)

    def _norm_t_zone(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.T_ZONE_MIN) / (self.T_ZONE_MAX - self.T_ZONE_MIN) - 1.0

    def _norm_t_amb(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.T_AMB_MIN) / (self.T_AMB_MAX - self.T_AMB_MIN) - 1.0

    @staticmethod
    def _encode_hour(hour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rad = 2.0 * np.pi * hour / 24.0
        return torch.sin(rad), torch.cos(rad)

    @staticmethod
    def _encode_day(day: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rad = 2.0 * np.pi * day / 365.0
        return torch.sin(rad), torch.cos(rad)

    def _build_features(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> torch.Tensor:
        t_zone_n = self._norm_t_zone(t_zone)
        t_amb_n = self._norm_t_amb(t_amb)
        h_sin, h_cos = self._encode_hour(hour)
        d_sin, d_cos = self._encode_day(day)
        return torch.stack([t_zone_n, t_amb_n, h_sin, h_cos, d_sin, d_cos, a0, a1], dim=-1)

    def forward_with_aux(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._build_features(t_zone, t_amb, hour, day, a0, a1)
        q_net = self.q_net(x)
        c_zon = self.c_zon
        d_t = self.dt_seconds * q_net / c_zon
        t_next = torch.clamp(t_zone + d_t, self.T_ZONE_MIN, self.T_ZONE_MAX)
        p_total = self.power_net(x)
        return t_next, p_total, q_net, d_t, c_zon

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_next, p_total, _, _, _ = self.forward_with_aux(t_zone, t_amb, hour, day, a0, a1)
        return t_next, p_total

    def summary(self) -> None:
        n_total = sum(p.numel() for p in self.parameters())
        n_q = sum(p.numel() for p in self.q_net.parameters())
        n_power = sum(p.numel() for p in self.power_net.parameters())
        print(f"\n{'='*55}")
        print("RC Neural ODE v3.5 Summary")
        print(f"{'='*55}")
        print("  Input features:   8 (T_zone, T_amb, h_sin, h_cos, d_sin, d_cos, a0, a1)")
        print(f"  QNetV35:          {n_q:,} params")
        print(f"  PowerNetV35:      {n_power:,} params")
        print(f"  Total:            {n_total:,} params")
        print(f"  C_zon:            {float(self.c_zon.detach().cpu().item()):.3e} J/K")
        print(f"  DT:               {self.dt_seconds:.1f} s")
        print(f"  T_zone range:     [{self.T_ZONE_MIN}, {self.T_ZONE_MAX}] C")
        print(f"  T_amb range:      [{self.T_AMB_MIN}, {self.T_AMB_MAX}] C")
        print(f"  P_max:            {self.P_MAX} W")


def load_v35_from_v2_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    c_zon_init: float = 5.3e5,
    c_zon_min: float = 5.0e4,
    c_zon_scale: float = 1.0e5,
    q_scale: float = 3000.0,
    legacy_heat_scale_factor: float = 10.0,
    dt_seconds: float | None = None,
    legacy_step_seconds: float = 3600.0,
) -> RCNeuralODEv35:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = int(ckpt.get("hidden_dim", 64))
    model = RCNeuralODEv35(
        hidden_dim=hidden_dim,
        p_max=float(ckpt.get("p_max", RCNeuralODEv35.P_MAX)),
        q_scale=q_scale,
        c_zon_init=c_zon_init,
        c_zon_min=c_zon_min,
        c_zon_scale=c_zon_scale,
        dt_seconds=dt_seconds,
    ).to(device)

    src_state = ckpt["model_state"]
    dst_state = model.state_dict()

    for src_key, src_val in src_state.items():
        if src_key.startswith("power_net.") and src_key in dst_state:
            dst_state[src_key] = src_val
            continue

        if not src_key.startswith("heat_net."):
            continue

        new_key = "q_net." + src_key.split(".", 1)[1]
        if new_key not in dst_state:
            continue

        copied = src_val.clone()
        if src_key.startswith("heat_net.output."):
            scale = legacy_heat_scale_factor * c_zon_init / (float(legacy_step_seconds) * q_scale)
            copied = copied * scale
        dst_state[new_key] = copied

    model.load_state_dict(dst_state, strict=False)
    model.eval()
    return model
