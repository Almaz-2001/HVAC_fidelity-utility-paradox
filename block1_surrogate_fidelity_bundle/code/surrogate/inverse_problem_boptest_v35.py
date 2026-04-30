from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surrogate.inverse_problem_boptest_v3 import (
    ArtifactSpec,
    _frame_to_tensors,
    _load_clean_df,
    _metrics,
    inject_boptest_artifacts,
    preprocess_artifacts,
)
from surrogate.rc_node_v35 import RCNeuralODEv35, load_v35_from_v2_checkpoint

ROLLOUT_COMFORT_LOW_C = 21.0
ROLLOUT_COMFORT_HIGH_C = 24.0
ROLLOUT_EDGE_MARGIN_C = 0.5
ROLLOUT_BUCKET_ORDER = ("band_violation", "band_edge", "high_excitation", "ambient_extreme", "context")
ROLLOUT_STAGE_C_MODES = frozenset({"rollout_heads_only", "rollout_temp_head_only"})


class PowerCalibrationHeadV35(nn.Module):
    def __init__(
        self,
        p_max: float = 5500.0,
        t_zone_min: float = 15.0,
        t_zone_max: float = 35.0,
        t_amb_min: float = -10.0,
        t_amb_max: float = 40.0,
        hidden_dim: int = 32,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.p_max = float(p_max)
        self.t_zone_min = float(t_zone_min)
        self.t_zone_max = float(t_zone_max)
        self.t_amb_min = float(t_amb_min)
        self.t_amb_max = float(t_amb_max)
        self.residual_scale = float(residual_scale)
        self.scale_P = nn.Parameter(torch.tensor(1.0))
        self.bias_P = nn.Parameter(torch.tensor(0.0))
        hidden_dim_2 = max(hidden_dim // 2, 16)
        self.residual_net = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.Tanh(),
            nn.Linear(hidden_dim_2, 1),
        )
        final_layer = self.residual_net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def _norm_t_zone(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.t_zone_min) / (self.t_zone_max - self.t_zone_min) - 1.0

    def _norm_t_amb(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.t_amb_min) / (self.t_amb_max - self.t_amb_min) - 1.0

    def forward(
        self,
        p_surr: torch.Tensor,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        t_surr: torch.Tensor,
    ) -> torch.Tensor:
        h_rad = 2.0 * np.pi * hour / 24.0
        d_rad = 2.0 * np.pi * day / 365.0
        p_surr_norm = torch.clamp(p_surr / max(self.p_max, 1.0), min=0.0, max=2.0)
        delta_t = torch.clamp((t_surr - t_zone) / 5.0, min=-3.0, max=3.0)
        feats = torch.stack(
            [
                p_surr_norm,
                self._norm_t_zone(t_zone),
                self._norm_t_amb(t_amb),
                torch.sin(h_rad),
                torch.cos(h_rad),
                torch.sin(d_rad),
                torch.cos(d_rad),
                a0,
                a1,
                self._norm_t_zone(t_surr),
                delta_t,
            ],
            dim=-1,
        )
        power_residual_w = torch.tanh(self.residual_net(feats).squeeze(-1)) * (self.residual_scale * self.p_max)
        return torch.clamp(self.scale_P * p_surr + self.bias_P + power_residual_w, min=0.0)

    def regularization_loss(self) -> torch.Tensor:
        affine_reg = 0.2 * (self.scale_P - 1.0) ** 2 + (self.bias_P / 1000.0) ** 2
        net_reg = torch.zeros((), dtype=self.scale_P.dtype, device=self.scale_P.device)
        count = 0
        for param in self.residual_net.parameters():
            net_reg = net_reg + torch.mean(param ** 2)
            count += 1
        if count > 0:
            net_reg = net_reg / count
        return affine_reg + 0.01 * net_reg


class TempCalibrationHeadV35(nn.Module):
    def __init__(
        self,
        t_min: float = 15.0,
        t_max: float = 35.0,
        t_zone_min: float = 15.0,
        t_zone_max: float = 35.0,
        t_amb_min: float = -10.0,
        t_amb_max: float = 40.0,
        p_max: float = 5500.0,
        q_scale: float = 3000.0,
        hidden_dim: int = 32,
        residual_scale: float = 0.75,
        feature_set: str = "v1",
    ) -> None:
        super().__init__()
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.t_zone_min = float(t_zone_min)
        self.t_zone_max = float(t_zone_max)
        self.t_amb_min = float(t_amb_min)
        self.t_amb_max = float(t_amb_max)
        self.p_max = float(p_max)
        self.q_scale = float(q_scale)
        self.residual_scale = float(residual_scale)
        self.feature_set = str(feature_set)
        self.scale_T = nn.Parameter(torch.tensor(1.0))
        self.bias_T = nn.Parameter(torch.tensor(0.0))
        self.residual_net = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        final_layer = self.residual_net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def _norm_t_zone(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.t_zone_min) / (self.t_zone_max - self.t_zone_min) - 1.0

    def _norm_t_amb(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * (t - self.t_amb_min) / (self.t_amb_max - self.t_amb_min) - 1.0

    def _norm_p(self, p: torch.Tensor) -> torch.Tensor:
        return torch.clamp(p / max(self.p_max, 1.0), min=0.0, max=2.0)

    def _norm_q(self, q: torch.Tensor) -> torch.Tensor:
        return torch.clamp(q / max(self.q_scale, 1.0), min=-3.0, max=3.0)

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        t_surr: torch.Tensor,
        p_surr: torch.Tensor,
        q_net: torch.Tensor,
    ) -> torch.Tensor:
        h_rad = 2.0 * np.pi * hour / 24.0
        d_rad = 2.0 * np.pi * day / 365.0
        delta_t = torch.clamp((t_surr - t_zone) / 5.0, min=-3.0, max=3.0)
        if self.feature_set == "block13_rich":
            delta_zone_amb = torch.clamp((t_zone - t_amb) / 20.0, min=-3.0, max=3.0)
            delta_surr_amb = torch.clamp((t_surr - t_amb) / 20.0, min=-3.0, max=3.0)
            feats = torch.stack(
                [
                    self._norm_t_zone(t_zone),
                    self._norm_t_amb(t_amb),
                    torch.sin(h_rad),
                    torch.cos(h_rad),
                    torch.sin(d_rad),
                    torch.cos(d_rad),
                    delta_t,
                    self._norm_p(p_surr),
                    self._norm_q(q_net),
                    0.5 * (delta_zone_amb + delta_surr_amb),
                ],
                dim=-1,
            )
        else:
            feats = torch.stack(
                [
                    self._norm_t_zone(t_zone),
                    self._norm_t_amb(t_amb),
                    torch.sin(h_rad),
                    torch.cos(h_rad),
                    torch.sin(d_rad),
                    torch.cos(d_rad),
                    a0,
                    a1,
                    self._norm_t_zone(t_surr),
                    delta_t,
                ],
                dim=-1,
            )
        temp_residual = torch.tanh(self.residual_net(feats).squeeze(-1)) * self.residual_scale
        return torch.clamp(
            self.scale_T * t_surr + self.bias_T + temp_residual,
            min=self.t_min,
            max=self.t_max,
        )

    def regularization_loss(self) -> torch.Tensor:
        affine_reg = 0.2 * (self.scale_T - 1.0) ** 2 + (self.bias_T / 5.0) ** 2
        net_reg = torch.zeros((), dtype=self.scale_T.dtype, device=self.scale_T.device)
        count = 0
        for param in self.residual_net.parameters():
            net_reg = net_reg + torch.mean(param ** 2)
            count += 1
        if count > 0:
            net_reg = net_reg / count
        return affine_reg + 0.01 * net_reg


class StagedCalibratedSurrogateV35(nn.Module):
    def __init__(self, surrogate: RCNeuralODEv35, temp_head_feature_set: str = "v1"):
        super().__init__()
        self.surrogate = surrogate
        self.temp_head_feature_set = str(temp_head_feature_set)
        self.temp_head = TempCalibrationHeadV35(
            t_min=surrogate.T_ZONE_MIN,
            t_max=surrogate.T_ZONE_MAX,
            t_zone_min=surrogate.T_ZONE_MIN,
            t_zone_max=surrogate.T_ZONE_MAX,
            t_amb_min=surrogate.T_AMB_MIN,
            t_amb_max=surrogate.T_AMB_MAX,
            p_max=surrogate.P_MAX,
            q_scale=surrogate.q_scale,
            feature_set=self.temp_head_feature_set,
        )
        self.power_head = PowerCalibrationHeadV35(
            p_max=surrogate.P_MAX,
            t_zone_min=surrogate.T_ZONE_MIN,
            t_zone_max=surrogate.T_ZONE_MAX,
            t_amb_min=surrogate.T_AMB_MIN,
            t_amb_max=surrogate.T_AMB_MAX,
        )

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_surr, p_surr, q_net, _, c_zon = self.surrogate.forward_with_aux(t_zone, t_amb, hour, day, a0, a1)
        t_cal = self.temp_head(t_zone, t_amb, hour, day, a0, a1, t_surr, p_surr, q_net)
        p_cal = self.power_head(p_surr, t_zone, t_amb, hour, day, a0, a1, t_surr)
        return t_surr, t_cal, p_surr, p_cal, q_net, c_zon

    def freeze_backbone(self) -> None:
        for name, param in self.surrogate.named_parameters():
            param.requires_grad = name == "log_c_zon"
        for param in self.temp_head.parameters():
            param.requires_grad = False
        for param in self.power_head.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.surrogate.parameters():
            param.requires_grad = True
        for param in self.temp_head.parameters():
            param.requires_grad = True
        for param in self.power_head.parameters():
            param.requires_grad = True

    def stage_b_groups(self, czon_lr: float) -> list[dict]:
        return [{"params": [self.surrogate.log_c_zon], "lr": czon_lr, "name": "czon"}]

    def configure_stage_c(self, mode: str) -> None:
        if mode == "joint":
            self.unfreeze_all()
            return
        if mode in {"heads_only", "rollout_heads_only"}:
            for param in self.surrogate.parameters():
                param.requires_grad = False
            for param in self.temp_head.parameters():
                param.requires_grad = True
            for param in self.power_head.parameters():
                param.requires_grad = True
            return
        if mode == "rollout_temp_head_only":
            for param in self.surrogate.parameters():
                param.requires_grad = False
            for param in self.temp_head.parameters():
                param.requires_grad = True
            for param in self.power_head.parameters():
                param.requires_grad = False
            return
        if mode == "power_head_only":
            for param in self.surrogate.parameters():
                param.requires_grad = False
            for param in self.temp_head.parameters():
                param.requires_grad = False
            for param in self.power_head.parameters():
                param.requires_grad = True
            return
        raise ValueError(f"Unsupported Stage C mode: {mode}")

    def stage_c_groups(self, backbone_lr: float, calib_lr: float, mode: str) -> list[dict]:
        if mode == "joint":
            backbone_params = [p for n, p in self.surrogate.named_parameters() if n != "log_c_zon"]
            calib_params = [self.surrogate.log_c_zon, *self.temp_head.parameters(), *self.power_head.parameters()]
            return [
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                {"params": calib_params, "lr": calib_lr, "name": "calib"},
            ]

        if mode in {"heads_only", "rollout_heads_only"}:
            head_params = [*self.temp_head.parameters(), *self.power_head.parameters()]
            return [{"params": head_params, "lr": calib_lr, "name": "heads"}]

        if mode == "rollout_temp_head_only":
            return [{"params": list(self.temp_head.parameters()), "lr": calib_lr, "name": "temp_head"}]

        if mode == "power_head_only":
            return [{"params": list(self.power_head.parameters()), "lr": calib_lr, "name": "power_head"}]

        raise ValueError(f"Unsupported Stage C mode: {mode}")


class StageBLossV35(nn.Module):
    def __init__(self, c_prior: float, lambda_c_prior: float = 0.1) -> None:
        super().__init__()
        self.c_prior = float(c_prior)
        self.lambda_c_prior = float(lambda_c_prior)

    def forward(self, t_pred: torch.Tensor, t_true: torch.Tensor, c_zon: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        l_temp = torch.mean((t_pred - t_true) ** 2)
        log_c = torch.log(torch.clamp(c_zon, min=1.0))
        log_prior = torch.log(torch.tensor(self.c_prior, device=c_zon.device, dtype=c_zon.dtype))
        l_prior = (log_c - log_prior) ** 2
        total = l_temp + self.lambda_c_prior * l_prior
        return total, {
            "loss_total": float(total.item()),
            "loss_temp": float(l_temp.item()),
            "loss_c_prior": float(l_prior.item()),
            "rmse_temp": float(torch.sqrt(l_temp).item()),
        }


class StageCLossV35(nn.Module):
    def __init__(
        self,
        c_prior: float,
        lambda_temp_data: float = 1.0,
        lambda_power: float = 0.08,
        lambda_c_prior: float = 0.05,
        lambda_q_reg: float = 0.001,
        lambda_power_reg: float = 0.02,
        lambda_temp_reg: float = 0.02,
    ) -> None:
        super().__init__()
        self.c_prior = float(c_prior)
        self.lambda_temp_data = float(lambda_temp_data)
        self.lambda_power = float(lambda_power)
        self.lambda_c_prior = float(lambda_c_prior)
        self.lambda_q_reg = float(lambda_q_reg)
        self.lambda_power_reg = float(lambda_power_reg)
        self.lambda_temp_reg = float(lambda_temp_reg)

    def forward(
        self,
        t_pred: torch.Tensor,
        t_true: torch.Tensor,
        p_pred: torch.Tensor,
        p_true: torch.Tensor,
        q_net: torch.Tensor,
        c_zon: torch.Tensor,
        temp_head: TempCalibrationHeadV35,
        power_head: PowerCalibrationHeadV35,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        l_temp = torch.mean((t_pred - t_true) ** 2)
        p_scale = torch.clamp(p_true.max().detach(), min=1.0)
        l_power = torch.mean(((p_pred - p_true) / p_scale) ** 2)
        mae_power = torch.mean(torch.abs(p_pred - p_true))
        rmse_power = torch.sqrt(torch.mean((p_pred - p_true) ** 2))
        bias_power = torch.mean(p_pred - p_true)
        log_c = torch.log(torch.clamp(c_zon, min=1.0))
        log_prior = torch.log(torch.tensor(self.c_prior, device=c_zon.device, dtype=c_zon.dtype))
        l_prior = (log_c - log_prior) ** 2
        l_q = torch.mean((q_net / 3000.0) ** 2)
        l_temp_reg = temp_head.regularization_loss()
        l_power_reg = power_head.regularization_loss()
        total = (
            self.lambda_temp_data * l_temp
            + self.lambda_power * l_power
            + self.lambda_c_prior * l_prior
            + self.lambda_q_reg * l_q
            + self.lambda_temp_reg * l_temp_reg
            + self.lambda_power_reg * l_power_reg
        )
        return total, {
            "loss_total": float(total.item()),
            "loss_temp": float(l_temp.item()),
            "loss_power": float(l_power.item()),
            "loss_c_prior": float(l_prior.item()),
            "loss_q_reg": float(l_q.item()),
            "loss_temp_reg": float(l_temp_reg.item()),
            "loss_power_reg": float(l_power_reg.item()),
            "rmse_temp": float(torch.sqrt(l_temp).item()),
            "mae_power_w": float(mae_power.item()),
            "rmse_power_w": float(rmse_power.item()),
            "bias_power_w": float(bias_power.item()),
        }


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args

    parser_defaults = {
        "data": "data/surrogate_v2/boptest_v2_tsupply.csv",
        "model": "outputs/surrogate_v2/rc_node_v3_tsupply.pt",
        "output_dir": "outputs/surrogate_v35_inverse_boptest",
        "policy": None,
        "season": None,
        "limit_rows": None,
        "no_artifact_injection": False,
        "temp_bias_c": 0.5,
        "temp_noise_std": 0.08,
        "temp_latency_steps": 2,
        "power_scale": 1.04,
        "power_bias_w": 35.0,
        "power_noise_rel": 0.015,
        "c_zon_true": 4.2e5,
        "surrogate_czon_ref": 5.3e5,
        "max_latency_search": 6,
        "smooth_window": 5,
        "target_mode": "clean",
        "batch_size": 256,
        "val_split": 0.2,
        "stage_b_epochs": 120,
        "stage_b_patience": 20,
        "stage_c_epochs": 180,
        "stage_c_patience": 30,
        "czon_lr": 1e-2,
        "backbone_lr": 1e-4,
        "calib_lr": 1e-2,
        "lambda_c_prior_b": 0.1,
        "lambda_c_prior_c": 0.05,
        "lambda_power": 0.08,
        "lambda_q_reg": 0.001,
        "lambda_temp_reg": 0.02,
        "lambda_rollout": 0.0,
        "rollout_teacher_forced_epochs": 10,
        "rollout_free_run_final_ratio": 0.5,
        "excitation_quantile": 0.8,
        "excitation_mix_ratio": 0.6,
        "excitation_mode": "hybrid",
        "c_zon_prior": 5.3e5,
        "c_zon_min": 5.0e4,
        "q_scale": 3000.0,
        "temp_head_feature_set": "v1",
        "stage_c_mode": "joint",
        "stage_c_selection_metric": "auto",
        "rollout_horizons": "4,8",
        "step_sec": None,
        "legacy_step_sec": int(RCNeuralODEv35.DT),
        "init_summary_json": None,
        "init_checkpoint": None,
        "seed": None,
    }

    def _maybe_set(name: str, value) -> None:
        current = getattr(args, name)
        default = parser_defaults[name]
        if current is None or current == default:
            setattr(args, name, value)

    _maybe_set("seed", 42)

    if args.preset == "smoke":
        _maybe_set("policy", "mixed")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_smoke")
        _maybe_set("limit_rows", 600)
        _maybe_set("stage_b_epochs", 8)
        _maybe_set("stage_b_patience", 4)
        _maybe_set("stage_c_epochs", 10)
        _maybe_set("stage_c_patience", 5)
        _maybe_set("batch_size", 128)
        _maybe_set("excitation_quantile", 0.85)
        _maybe_set("excitation_mix_ratio", 0.6)
        _maybe_set("czon_lr", 1e-2)
        _maybe_set("backbone_lr", 1e-4)
        _maybe_set("calib_lr", 1e-2)
        _maybe_set("no_artifact_injection", False)
        _maybe_set("target_mode", "clean")
    elif args.preset == "full":
        _maybe_set("policy", "mixed")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest")
        _maybe_set("limit_rows", None)
        _maybe_set("stage_b_epochs", 120)
        _maybe_set("stage_b_patience", 20)
        _maybe_set("stage_c_epochs", 180)
        _maybe_set("stage_c_patience", 30)
        _maybe_set("excitation_quantile", 0.8)
        _maybe_set("excitation_mix_ratio", 0.6)
        _maybe_set("czon_lr", 1e-2)
        _maybe_set("backbone_lr", 1e-4)
        _maybe_set("calib_lr", 1e-2)
        _maybe_set("no_artifact_injection", False)
        _maybe_set("target_mode", "clean")
    elif args.preset == "realclean":
        _maybe_set("policy", "mixed")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_realclean")
        _maybe_set("limit_rows", None)
        _maybe_set("stage_b_epochs", 120)
        _maybe_set("stage_b_patience", 20)
        _maybe_set("stage_c_epochs", 180)
        _maybe_set("stage_c_patience", 30)
        _maybe_set("excitation_quantile", 0.8)
        _maybe_set("excitation_mix_ratio", 0.6)
        _maybe_set("czon_lr", 1e-2)
        _maybe_set("backbone_lr", 1e-4)
        _maybe_set("calib_lr", 1e-2)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
    elif args.preset == "block2_15min":
        _maybe_set("policy", None)
        _maybe_set("data", "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_15min")
        _maybe_set("limit_rows", None)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
        _maybe_set("step_sec", 900)
        _maybe_set("legacy_step_sec", int(RCNeuralODEv35.DT))
        _maybe_set("stage_b_epochs", 120)
        _maybe_set("stage_b_patience", 25)
        _maybe_set("stage_c_epochs", 20)
        _maybe_set("stage_c_patience", 6)
        _maybe_set("stage_c_mode", "heads_only")
        _maybe_set("excitation_quantile", 0.95)
        _maybe_set("excitation_mix_ratio", 1.0)
        _maybe_set("excitation_mode", "dt_only")
        _maybe_set("c_zon_prior", 4.2e5)
        _maybe_set("c_zon_true", 4.2e5)
        _maybe_set("czon_lr", 1e-3)
        _maybe_set("backbone_lr", 2e-5)
        _maybe_set("calib_lr", 1e-3)
        _maybe_set("lambda_c_prior_b", 0.35)
        _maybe_set("lambda_c_prior_c", 0.15)
        _maybe_set("lambda_power", 0.02)
        _maybe_set("lambda_q_reg", 5e-4)
        _maybe_set("lambda_temp_reg", 0.05)
    elif args.preset == "block2_15min_rollout":
        _maybe_set("policy", None)
        _maybe_set("data", "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_15min_rollout")
        _maybe_set("limit_rows", None)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
        _maybe_set("step_sec", 900)
        _maybe_set("legacy_step_sec", int(RCNeuralODEv35.DT))
        _maybe_set("stage_b_epochs", 120)
        _maybe_set("stage_b_patience", 25)
        _maybe_set("stage_c_epochs", 60)
        _maybe_set("stage_c_patience", 12)
        _maybe_set("stage_c_mode", "rollout_heads_only")
        _maybe_set("stage_c_selection_metric", "val_rollout_rmse_free")
        _maybe_set("lambda_rollout", 0.5)
        _maybe_set("rollout_teacher_forced_epochs", 8)
        _maybe_set("rollout_free_run_final_ratio", 0.5)
        _maybe_set("rollout_horizons", "4,8,24")
        _maybe_set("excitation_quantile", 0.95)
        _maybe_set("excitation_mix_ratio", 1.0)
        _maybe_set("excitation_mode", "dt_only")
        _maybe_set("c_zon_prior", 4.2e5)
        _maybe_set("c_zon_true", 4.2e5)
        _maybe_set("czon_lr", 1e-3)
        _maybe_set("backbone_lr", 2e-5)
        _maybe_set("calib_lr", 5e-4)
        _maybe_set("lambda_c_prior_b", 0.35)
        _maybe_set("lambda_c_prior_c", 0.15)
        _maybe_set("lambda_power", 0.02)
        _maybe_set("lambda_q_reg", 5e-4)
        _maybe_set("lambda_temp_reg", 0.05)
    elif args.preset == "block1_15min_episodeaware":
        _maybe_set("policy", None)
        _maybe_set("data", "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_15min_episodeaware")
        _maybe_set("limit_rows", None)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
        _maybe_set("step_sec", 900)
        _maybe_set("legacy_step_sec", int(RCNeuralODEv35.DT))
        _maybe_set("stage_b_epochs", 120)
        _maybe_set("stage_b_patience", 25)
        _maybe_set("stage_c_epochs", 60)
        _maybe_set("stage_c_patience", 12)
        _maybe_set("stage_c_mode", "rollout_heads_only")
        _maybe_set("stage_c_selection_metric", "val_rollout_rmse_free")
        _maybe_set("lambda_rollout", 0.5)
        _maybe_set("rollout_teacher_forced_epochs", 8)
        _maybe_set("rollout_free_run_final_ratio", 0.5)
        _maybe_set("rollout_horizons", "4,8,24")
        _maybe_set("excitation_quantile", 0.95)
        _maybe_set("excitation_mix_ratio", 1.0)
        _maybe_set("excitation_mode", "dt_only")
        _maybe_set("c_zon_prior", 4.2e5)
        _maybe_set("c_zon_true", 4.2e5)
        _maybe_set("czon_lr", 1e-3)
        _maybe_set("backbone_lr", 2e-5)
        _maybe_set("calib_lr", 5e-4)
        _maybe_set("lambda_c_prior_b", 0.35)
        _maybe_set("lambda_c_prior_c", 0.15)
        _maybe_set("lambda_power", 0.02)
        _maybe_set("lambda_q_reg", 5e-4)
        _maybe_set("lambda_temp_reg", 0.05)
    elif args.preset == "block1_15min_power_head_only":
        _maybe_set("policy", None)
        _maybe_set("data", "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_15min_power_head_only")
        _maybe_set("limit_rows", None)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
        _maybe_set("step_sec", 900)
        _maybe_set("legacy_step_sec", int(RCNeuralODEv35.DT))
        _maybe_set("stage_b_epochs", 0)
        _maybe_set("stage_b_patience", 0)
        _maybe_set("stage_c_epochs", 80)
        _maybe_set("stage_c_patience", 15)
        _maybe_set("stage_c_mode", "power_head_only")
        _maybe_set("stage_c_selection_metric", "val_power_mae_w")
        _maybe_set("init_summary_json", "outputs/surrogate_v35_inverse_boptest_15min_episodeaware/calibration_summary_boptest_v35.json")
        _maybe_set("excitation_quantile", 0.95)
        _maybe_set("excitation_mix_ratio", 1.0)
        _maybe_set("excitation_mode", "dt_only")
        _maybe_set("c_zon_prior", 4.2e5)
        _maybe_set("c_zon_true", 4.2e5)
        _maybe_set("czon_lr", 1e-3)
        _maybe_set("backbone_lr", 2e-5)
        _maybe_set("calib_lr", 1e-3)
        _maybe_set("lambda_c_prior_b", 0.35)
        _maybe_set("lambda_c_prior_c", 0.0)
        _maybe_set("lambda_power", 0.10)
        _maybe_set("lambda_q_reg", 0.0)
        _maybe_set("lambda_temp_reg", 0.0)
        _maybe_set("lambda_rollout", 0.0)
    elif args.preset == "block1_3_15min_closed_loop":
        _maybe_set("policy", None)
        _maybe_set("data", "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
        _maybe_set("output_dir", "outputs/surrogate_v35_inverse_boptest_15min_block13_closed_loop")
        _maybe_set("limit_rows", None)
        _maybe_set("no_artifact_injection", True)
        _maybe_set("target_mode", "clean")
        _maybe_set("step_sec", 900)
        _maybe_set("legacy_step_sec", int(RCNeuralODEv35.DT))
        _maybe_set("stage_b_epochs", 0)
        _maybe_set("stage_b_patience", 0)
        _maybe_set("stage_c_epochs", 80)
        _maybe_set("stage_c_patience", 16)
        _maybe_set("stage_c_mode", "rollout_temp_head_only")
        _maybe_set("stage_c_selection_metric", "val_rollout_rmse_free")
        _maybe_set("temp_head_feature_set", "block13_rich")
        _maybe_set("init_summary_json", "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json")
        _maybe_set("excitation_quantile", 0.95)
        _maybe_set("excitation_mix_ratio", 1.0)
        _maybe_set("excitation_mode", "dt_only")
        _maybe_set("c_zon_prior", 4.2e5)
        _maybe_set("c_zon_true", 4.2e5)
        _maybe_set("czon_lr", 1e-3)
        _maybe_set("backbone_lr", 2e-5)
        _maybe_set("calib_lr", 7.5e-4)
        _maybe_set("lambda_c_prior_b", 0.35)
        _maybe_set("lambda_c_prior_c", 0.10)
        _maybe_set("lambda_power", 0.0)
        _maybe_set("lambda_q_reg", 5e-4)
        _maybe_set("lambda_temp_reg", 0.05)
        _maybe_set("lambda_rollout", 0.6)
        _maybe_set("rollout_teacher_forced_epochs", 8)
        _maybe_set("rollout_free_run_final_ratio", 0.6)
        _maybe_set("rollout_horizons", "4,8,16,32,96")
    return args


def _compute_excitation_scores(df: pd.DataFrame, mode: str) -> np.ndarray:
    dt_abs = np.abs(df["t_zone_next"].to_numpy(dtype=float) - df["t_zone"].to_numpy(dtype=float))
    if mode == "dt_only":
        return dt_abs

    da0 = np.abs(np.diff(df["a0_raw"].to_numpy(dtype=float), prepend=df["a0_raw"].iloc[0]))
    da1 = np.abs(np.diff(df["a1_raw"].to_numpy(dtype=float), prepend=df["a1_raw"].iloc[0]))
    damb = np.abs(np.diff(df["t_amb"].to_numpy(dtype=float), prepend=df["t_amb"].iloc[0]))

    def _norm(x: np.ndarray) -> np.ndarray:
        q = float(np.quantile(x, 0.9))
        if q <= 1e-9:
            return np.zeros_like(x)
        return np.clip(x / q, 0.0, 3.0)

    return 1.0 * _norm(dt_abs) + 0.35 * _norm(da0) + 0.35 * _norm(da1) + 0.25 * _norm(damb)


def _parse_rollout_horizons(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)

    horizons = sorted({int(part) for part in parts if str(part).strip()})
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        raise ValueError("At least one positive rollout horizon is required")
    return horizons


def _infer_step_seconds(df: pd.DataFrame, fallback_step_sec: int) -> int:
    if "step_sec" in df.columns:
        values = sorted({int(round(float(v))) for v in df["step_sec"].dropna().unique() if float(v) > 0})
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            raise ValueError(f"Dataset has multiple step_sec values: {values}")

    if "sim_time_sec" in df.columns:
        diffs = np.diff(df["sim_time_sec"].to_numpy(dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if len(diffs) > 0:
            rounded = np.unique(np.round(diffs).astype(int))
            if len(rounded) == 1:
                return int(rounded[0])

    return int(fallback_step_sec)


def _build_rollout_window_starts(start_idx: int, end_idx: int, max_horizon: int) -> np.ndarray:
    if max_horizon <= 0:
        return np.array([], dtype=int)
    last_start = end_idx - max_horizon
    if last_start < start_idx:
        return np.array([], dtype=int)
    return np.arange(start_idx, last_start + 1, dtype=int)


def _episode_segments(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    if "episode_id" not in df.columns:
        return [("all_rows", 0, len(df))]
    segments: list[tuple[str, int, int]] = []
    for episode_id, group in df.groupby("episode_id", sort=False):
        start = int(group.index[0])
        end = int(group.index[-1]) + 1
        segments.append((str(episode_id), start, end))
    return segments


def _build_rollout_window_starts_from_segments(
    segments: list[tuple[str, int, int]],
    max_horizon: int,
) -> np.ndarray:
    if max_horizon <= 0:
        return np.array([], dtype=int)
    all_starts: list[np.ndarray] = []
    for _, start_idx, end_idx in segments:
        starts = _build_rollout_window_starts(start_idx=start_idx, end_idx=end_idx, max_horizon=max_horizon)
        if len(starts) > 0:
            all_starts.append(starts)
    if not all_starts:
        return np.array([], dtype=int)
    return np.concatenate(all_starts).astype(int)


def _compute_rollout_window_table(
    df: pd.DataFrame,
    start_idx: np.ndarray,
    max_horizon: int,
    comfort_low_c: float = ROLLOUT_COMFORT_LOW_C,
    comfort_high_c: float = ROLLOUT_COMFORT_HIGH_C,
    edge_margin_c: float = ROLLOUT_EDGE_MARGIN_C,
) -> pd.DataFrame:
    if len(start_idx) == 0 or max_horizon <= 0:
        return pd.DataFrame(columns=["start_idx", "bucket", "mean_abs_delta_t", "mean_t_amb"])

    target_col = "t_zone_next" if "t_zone_next" in df.columns else "t_zone"
    rows: list[dict] = []
    for start in start_idx.astype(int):
        window = df.iloc[start : start + max_horizon]
        if len(window) < max_horizon:
            continue
        temps = window[target_col].to_numpy(dtype=float)
        t_amb = window["t_amb"].to_numpy(dtype=float)
        t_zone_now = window["t_zone"].to_numpy(dtype=float)
        mean_abs_delta_t = float(np.mean(np.abs(temps - t_zone_now)))
        mean_t_amb = float(np.mean(t_amb))
        below = temps < comfort_low_c
        above = temps > comfort_high_c
        violation = bool(np.any(below | above))
        edge_distance = float(np.min(np.minimum(np.abs(temps - comfort_low_c), np.abs(temps - comfort_high_c))))
        rows.append(
            {
                "start_idx": int(start),
                "mean_abs_delta_t": mean_abs_delta_t,
                "mean_t_amb": mean_t_amb,
                "has_violation": violation,
                "near_edge": bool(edge_distance <= edge_margin_c),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["start_idx", "bucket", "mean_abs_delta_t", "mean_t_amb"])

    table = pd.DataFrame(rows)
    exc_threshold = float(table["mean_abs_delta_t"].quantile(0.75))
    cold_threshold = float(table["mean_t_amb"].quantile(0.20))
    hot_threshold = float(table["mean_t_amb"].quantile(0.80))

    def _bucket(row: pd.Series) -> str:
        if bool(row["has_violation"]):
            return "band_violation"
        if bool(row["near_edge"]):
            return "band_edge"
        if float(row["mean_abs_delta_t"]) >= exc_threshold:
            return "high_excitation"
        if float(row["mean_t_amb"]) <= cold_threshold or float(row["mean_t_amb"]) >= hot_threshold:
            return "ambient_extreme"
        return "context"

    table["bucket"] = table.apply(_bucket, axis=1)
    return table


def _build_rollout_bucket_pools(
    df: pd.DataFrame,
    segments: list[tuple[str, int, int]],
    max_horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    starts = _build_rollout_window_starts_from_segments(segments=segments, max_horizon=max_horizon)
    table = _compute_rollout_window_table(df=df, start_idx=starts, max_horizon=max_horizon)
    pools: dict[str, np.ndarray] = {}
    summary: dict[str, int] = {"starts_total": int(len(starts))}
    for bucket in ROLLOUT_BUCKET_ORDER:
        bucket_idx = table.loc[table["bucket"] == bucket, "start_idx"].to_numpy(dtype=int) if not table.empty else np.array([], dtype=int)
        pools[bucket] = bucket_idx
        summary[f"bucket_{bucket}"] = int(len(bucket_idx))
    return pools, summary


def _subset_epoch_rollout_batches(
    rollout_pools: dict[str, np.ndarray],
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = {
        bucket: rng.permutation(indices).astype(int)
        for bucket, indices in rollout_pools.items()
        if len(indices) > 0
    }
    if not shuffled:
        return []

    pointers = {bucket: 0 for bucket in shuffled}
    active = [bucket for bucket in ROLLOUT_BUCKET_ORDER if bucket in shuffled]
    batches: list[np.ndarray] = []
    while active:
        batch: list[int] = []
        while len(batch) < batch_size and active:
            next_active: list[str] = []
            for bucket in active:
                arr = shuffled[bucket]
                ptr = pointers[bucket]
                if ptr >= len(arr):
                    continue
                batch.append(int(arr[ptr]))
                pointers[bucket] = ptr + 1
                if pointers[bucket] < len(arr):
                    next_active.append(bucket)
                if len(batch) >= batch_size:
                    next_active.extend([b for b in active if b not in next_active and pointers[b] < len(shuffled[b])])
                    break
            active = next_active
        if batch:
            batches.append(np.asarray(batch, dtype=int))
    return batches


def _segments_from_indices(indices: np.ndarray, label_prefix: str) -> list[tuple[str, int, int]]:
    if len(indices) == 0:
        return []
    sorted_idx = np.sort(np.unique(indices.astype(int)))
    breakpoints = np.flatnonzero(np.diff(sorted_idx) != 1)
    segment_starts = np.concatenate(([0], breakpoints + 1))
    segment_ends = np.concatenate((breakpoints, [len(sorted_idx) - 1]))
    segments: list[tuple[str, int, int]] = []
    for seg_num, (start_pos, end_pos) in enumerate(zip(segment_starts, segment_ends)):
        start_idx = int(sorted_idx[start_pos])
        end_idx = int(sorted_idx[end_pos]) + 1
        segments.append((f"{label_prefix}_{seg_num}", start_idx, end_idx))
    return segments


def _scheduled_free_run_ratio(
    epoch: int,
    total_epochs: int,
    teacher_forced_epochs: int,
    free_run_final_ratio: float,
) -> float:
    free_run_final_ratio = float(np.clip(free_run_final_ratio, 0.0, 1.0))
    if free_run_final_ratio <= 0.0:
        return 0.0
    if epoch <= teacher_forced_epochs:
        return 0.0
    anneal_epochs = max(total_epochs - teacher_forced_epochs, 1)
    progress = min(max((epoch - teacher_forced_epochs) / anneal_epochs, 0.0), 1.0)
    return float(free_run_final_ratio * progress)


def _build_training_index_sets(
    df: pd.DataFrame,
    val_split: float,
    excitation_quantile: float,
    excitation_mix_ratio: float,
    excitation_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n = len(df)
    if "episode_id" in df.columns:
        episode_ids = [str(v) for v in pd.unique(df["episode_id"])]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(episode_ids))
        n_val_episodes = min(max(1, int(round(len(episode_ids) * val_split))), max(len(episode_ids) - 1, 1))
        if len(episode_ids) <= 1:
            val_episodes = set(episode_ids)
            train_episodes = set(episode_ids)
        else:
            val_episodes = {episode_ids[i] for i in perm[:n_val_episodes]}
            train_episodes = {episode_ids[i] for i in perm[n_val_episodes:]}
            if not train_episodes:
                train_episodes = {episode_ids[perm[-1]]}
                val_episodes = set(episode_ids) - train_episodes
        train_mask = df["episode_id"].astype(str).isin(train_episodes).to_numpy()
        val_mask = df["episode_id"].astype(str).isin(val_episodes).to_numpy()
        train_all = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        split_mode = "episode_id"
        episode_summary = {
            "episodes_total": int(len(episode_ids)),
            "episodes_train": int(len(train_episodes)),
            "episodes_val": int(len(val_episodes)),
        }
    else:
        n_train = max(1, int(n * (1.0 - val_split)))
        train_all = np.arange(0, n_train)
        val_idx = np.arange(n_train, n)
        split_mode = "row_index"
        episode_summary = {
            "episodes_total": 1,
            "episodes_train": 1,
            "episodes_val": 1 if len(val_idx) > 0 else 0,
        }

    if len(val_idx) == 0:
        val_idx = train_all.copy()

    scores = _compute_excitation_scores(df, mode=excitation_mode)
    train_scores = scores[train_all]
    threshold = float(np.quantile(train_scores, excitation_quantile))
    exc_idx = train_all[train_scores >= threshold]
    if len(exc_idx) == 0:
        exc_idx = train_all.copy()

    rng = np.random.default_rng(seed)
    context_pool = np.setdiff1d(train_all, exc_idx, assume_unique=False)
    context_count = int(len(exc_idx) * max(0.0, 1.0 - excitation_mix_ratio) / max(excitation_mix_ratio, 1e-6))
    if context_count > 0 and len(context_pool) > 0:
        context_count = min(context_count, len(context_pool))
        context_idx = np.sort(rng.choice(context_pool, size=context_count, replace=False))
        train_idx = np.sort(np.unique(np.concatenate([exc_idx, context_idx])))
    else:
        context_idx = np.array([], dtype=int)
        train_idx = np.sort(exc_idx)

    summary = {
        "rows_total": int(n),
        "rows_train_all": int(len(train_all)),
        "rows_train_selected": int(len(train_idx)),
        "rows_excitation": int(len(exc_idx)),
        "rows_context": int(len(context_idx)),
        "rows_val": int(len(val_idx)),
        "excitation_quantile": float(excitation_quantile),
        "excitation_threshold": float(threshold),
        "excitation_mix_ratio": float(excitation_mix_ratio),
        "excitation_mode": excitation_mode,
        "split_mode": split_mode,
        "score_mean_train": float(np.mean(train_scores)) if len(train_all) else 0.0,
        "score_mean_excitation": float(np.mean(scores[exc_idx])) if len(exc_idx) else 0.0,
    }
    summary.update(episode_summary)
    return train_idx, train_all, val_idx, summary


def _subset_epoch_batches(indices: np.ndarray, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    return [perm[i:i + batch_size] for i in range(0, len(perm), batch_size)]


def _compute_rollout_loss(
    model: StagedCalibratedSurrogateV35,
    data: Dict[str, torch.Tensor],
    start_idx_np: np.ndarray,
    rollout_horizons: list[int],
    free_run_mask_np: np.ndarray | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = data["t_zone"].device
    zero = torch.zeros((), dtype=data["t_zone"].dtype, device=device)
    if len(start_idx_np) == 0 or not rollout_horizons:
        return zero, {"loss_rollout_temp": 0.0, "rmse_rollout_temp": 0.0}

    horizons = sorted({int(h) for h in rollout_horizons if int(h) > 0})
    if not horizons:
        return zero, {"loss_rollout_temp": 0.0, "rmse_rollout_temp": 0.0}

    start_idx = torch.tensor(start_idx_np, dtype=torch.long, device=device)
    t_curr = data["t_zone"][start_idx]
    loss_terms: list[torch.Tensor] = []
    horizon_set = set(horizons)
    max_horizon = horizons[-1]
    if free_run_mask_np is None:
        free_run_mask = torch.ones(len(start_idx_np), dtype=torch.bool, device=device)
    else:
        free_run_mask = torch.tensor(free_run_mask_np, dtype=torch.bool, device=device)
        if free_run_mask.numel() != len(start_idx_np):
            raise ValueError("free_run_mask length must match rollout start count")

    for step in range(max_horizon):
        idx = start_idx + step
        _, t_pred, _, _, _, _ = model(
            t_curr,
            data["t_amb"][idx],
            data["hour"][idx],
            data["day"][idx],
            data["a0"][idx],
            data["a1"][idx],
        )
        if step + 1 in horizon_set:
            target = data["t_next_target"][idx]
            loss_terms.append(torch.mean((t_pred - target) ** 2))
        if step + 1 < max_horizon:
            next_truth = data["t_zone"][idx + 1]
            t_curr = torch.where(free_run_mask, t_pred, next_truth)

    if not loss_terms:
        return zero, {"loss_rollout_temp": 0.0, "rmse_rollout_temp": 0.0}

    rollout_mse = torch.mean(torch.stack(loss_terms))
    return rollout_mse, {
        "loss_rollout_temp": float(rollout_mse.item()),
        "rmse_rollout_temp": float(torch.sqrt(rollout_mse).item()),
    }


def _resolve_stage_c_selection_metric(
    policy: str,
    stage_c_mode: str,
    has_rollout_windows: bool,
) -> str:
    normalized = str(policy).strip().lower()
    if normalized == "auto":
        if stage_c_mode in ROLLOUT_STAGE_C_MODES and has_rollout_windows:
            return "val_rollout_rmse_free"
        if stage_c_mode == "power_head_only":
            return "val_power_mae_w"
        return "val_loss"
    allowed = {
        "val_loss",
        "val_rollout_rmse_free",
        "val_rollout_rmse_mixed",
        "val_rollout_loss_free",
        "val_rollout_loss_mixed",
        "val_power_mae_w",
        "val_power_rmse_w",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported Stage C selection metric policy: {policy}")
    return normalized


def _evaluate_stage(
    model: StagedCalibratedSurrogateV35,
    data: Dict[str, torch.Tensor],
    idx_np: np.ndarray,
    criterion: nn.Module,
    stage: str,
) -> tuple[float, Dict[str, float]]:
    device = data["t_zone"].device
    idx = torch.tensor(idx_np, dtype=torch.long, device=device)
    _, t_pred, _, p_pred, q_net, c_zon = model(
        data["t_zone"][idx],
        data["t_amb"][idx],
        data["hour"][idx],
        data["day"][idx],
        data["a0"][idx],
        data["a1"][idx],
    )
    if stage == "b":
        loss, metrics = criterion(t_pred, data["t_next_target"][idx], c_zon)
    else:
        loss, metrics = criterion(
            t_pred,
            data["t_next_target"][idx],
            p_pred,
            data["p_target"][idx],
            q_net,
            c_zon,
            model.temp_head,
            model.power_head,
        )
    return float(loss.item()), metrics


def _resolve_optional_path(raw: str | None, anchors: list[Path], default_name: str | None = None) -> str | None:
    if raw is None and default_name is None:
        return None
    candidates: list[Path] = []
    if raw is not None:
        raw_path = Path(raw)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            for anchor in anchors:
                candidates.append(anchor / raw_path)
    if default_name is not None:
        for anchor in anchors:
            candidates.append(anchor / default_name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    if candidates:
        return str(candidates[-1])
    return None


def calibrate_boptest_v35(
    data_path: str,
    model_path: str,
    output_dir: str,
    policy: str | None,
    season: str | None,
    limit_rows: int | None,
    inject_artifacts: bool,
    artifact_spec: ArtifactSpec,
    max_latency_search: int,
    smooth_window: int,
    target_mode: str,
    stage_b_epochs: int,
    stage_b_patience: int,
    stage_c_epochs: int,
    stage_c_patience: int,
    batch_size: int,
    val_split: float,
    czon_lr: float,
    backbone_lr: float,
    calib_lr: float,
    lambda_c_prior_b: float,
    lambda_c_prior_c: float,
    lambda_power: float,
    lambda_q_reg: float,
    lambda_temp_reg: float,
    lambda_rollout: float,
    rollout_teacher_forced_epochs: int,
    rollout_free_run_final_ratio: float,
    excitation_quantile: float,
    excitation_mix_ratio: float,
    excitation_mode: str,
    c_zon_prior: float,
    c_zon_min: float,
    q_scale: float,
    stage_c_mode: str,
    stage_c_selection_metric: str,
    rollout_horizons: str | list[int] | tuple[int, ...],
    temp_head_feature_set: str,
    step_sec: int | None,
    legacy_step_sec: int,
    init_summary_json: str | None,
    init_checkpoint: str | None,
    seed: int,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_summary: dict | None = None
    init_checkpoint_path: str | None = None
    if init_summary_json is not None:
        init_summary_path = Path(init_summary_json)
        if not init_summary_path.is_absolute():
            init_summary_path = (REPO_ROOT / init_summary_path).resolve()
        init_summary = json.loads(init_summary_path.read_text(encoding="utf-8"))
        anchors = [init_summary_path.parent, REPO_ROOT]
        init_checkpoint_path = _resolve_optional_path(
            raw=init_checkpoint,
            anchors=anchors,
            default_name="rc_node_v35_boptest_staged_calibrated.pt",
        )
    elif init_checkpoint is not None:
        init_checkpoint_path = _resolve_optional_path(raw=init_checkpoint, anchors=[REPO_ROOT])

    clean_df = _load_clean_df(data_path, policy, season, limit_rows)
    runtime_step_sec = _infer_step_seconds(
        clean_df,
        fallback_step_sec=int(step_sec if step_sec is not None else RCNeuralODEv35.DT),
    )
    warm_model = load_v35_from_v2_checkpoint(
        model_path,
        device=device,
        c_zon_init=c_zon_prior,
        c_zon_min=c_zon_min,
        q_scale=q_scale,
        dt_seconds=runtime_step_sec,
        legacy_step_seconds=legacy_step_sec,
    )
    init_checkpoint_state = None
    if init_checkpoint_path is not None:
        init_checkpoint_state = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        if "surrogate_state" in init_checkpoint_state:
            warm_model.load_state_dict(init_checkpoint_state["surrogate_state"])

    if inject_artifacts:
        observed_df = inject_boptest_artifacts(clean_df, warm_model, device, artifact_spec, seed)
    else:
        observed_df = clean_df.copy()
        observed_df["t_zone_clean"] = clean_df["t_zone"]
        observed_df["t_zone_next_clean"] = clean_df["t_zone_next"]
        observed_df["p_total_clean"] = clean_df["p_total"]
        observed_df["t_zone_next_surrogate"] = np.nan
        observed_df["p_total_surrogate"] = np.nan

    preprocessed_df, preprocess_summary = preprocess_artifacts(
        observed_df,
        warm_model,
        device,
        max_latency_search=max_latency_search,
        smooth_window=smooth_window,
    )
    observed_csv = out_dir / "artifact_observed.csv"
    preprocessed_csv = out_dir / "artifact_preprocessed.csv"
    observed_df.to_csv(observed_csv, index=False)
    preprocessed_df.to_csv(preprocessed_csv, index=False)

    tensors = _frame_to_tensors(preprocessed_df, target_mode=target_mode)
    data = {k: v.to(device) for k, v in tensors.items()}
    train_idx, train_all_idx, val_idx, excitation_summary = _build_training_index_sets(
        preprocessed_df,
        val_split=val_split,
        excitation_quantile=excitation_quantile,
        excitation_mix_ratio=excitation_mix_ratio,
        excitation_mode=excitation_mode,
        seed=seed,
    )
    rollout_horizons_list = _parse_rollout_horizons(rollout_horizons)
    train_rollout_segments = _segments_from_indices(train_all_idx, label_prefix="train")
    val_rollout_segments = _segments_from_indices(val_idx, label_prefix="val")

    def _valid_horizons(segments: list[tuple[str, int, int]]) -> list[int]:
        if not segments:
            return []
        max_segment_len = max(end_idx - start_idx for _, start_idx, end_idx in segments)
        return [h for h in rollout_horizons_list if h <= max_segment_len]

    train_rollout_horizons = _valid_horizons(train_rollout_segments)
    val_rollout_horizons = _valid_horizons(val_rollout_segments)
    train_rollout_pools, train_rollout_summary = _build_rollout_bucket_pools(
        df=preprocessed_df,
        segments=train_rollout_segments,
        max_horizon=max(train_rollout_horizons, default=0),
    )
    val_rollout_pools, val_rollout_summary = _build_rollout_bucket_pools(
        df=preprocessed_df,
        segments=val_rollout_segments,
        max_horizon=max(val_rollout_horizons, default=0),
    )
    train_rollout_starts = np.concatenate([arr for arr in train_rollout_pools.values() if len(arr) > 0]).astype(int) if any(
        len(arr) > 0 for arr in train_rollout_pools.values()
    ) else np.array([], dtype=int)
    val_rollout_starts = np.concatenate([arr for arr in val_rollout_pools.values() if len(arr) > 0]).astype(int) if any(
        len(arr) > 0 for arr in val_rollout_pools.values()
    ) else np.array([], dtype=int)

    if stage_c_mode in ROLLOUT_STAGE_C_MODES and len(train_rollout_starts) == 0:
        raise ValueError(
            f"{stage_c_mode} requires enough sequential training rows for the requested rollout horizons"
        )
    resolved_stage_c_selection_metric = _resolve_stage_c_selection_metric(
        policy=stage_c_selection_metric,
        stage_c_mode=stage_c_mode,
        has_rollout_windows=len(val_rollout_starts) > 0,
    )

    model = StagedCalibratedSurrogateV35(warm_model, temp_head_feature_set=temp_head_feature_set).to(device)
    if init_checkpoint_state is not None:
        init_temp_head_feature_set = str(init_summary.get("temp_head_feature_set", "v1")) if init_summary is not None else "v1"
        if "temp_head_state" in init_checkpoint_state and init_temp_head_feature_set == temp_head_feature_set:
            model.temp_head.load_state_dict(init_checkpoint_state["temp_head_state"], strict=False)
        if "power_head_state" in init_checkpoint_state:
            model.power_head.load_state_dict(init_checkpoint_state["power_head_state"], strict=False)

    with torch.no_grad():
        if init_checkpoint_state is None:
            t_base, p_base = warm_model(
                data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
            )
        else:
            _, t_base, _, p_base, _, _ = model(
                data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
            )
        baseline_metrics = _metrics(
            t_base.cpu().numpy(),
            data["t_next_target"].cpu().numpy(),
            p_base.cpu().numpy(),
            data["p_target"].cpu().numpy(),
        )

    lambda_temp_data = 0.0 if stage_c_mode == "power_head_only" else 1.0
    stage_b_criterion = StageBLossV35(c_prior=c_zon_prior, lambda_c_prior=lambda_c_prior_b)
    stage_c_criterion = StageCLossV35(
        c_prior=c_zon_prior,
        lambda_temp_data=lambda_temp_data,
        lambda_power=lambda_power,
        lambda_c_prior=lambda_c_prior_c,
        lambda_q_reg=lambda_q_reg,
        lambda_temp_reg=lambda_temp_reg,
    )

    stage_b_history: list[dict] = []
    stage_c_history: list[dict] = []
    ckpt_path = out_dir / "rc_node_v35_boptest_staged_calibrated.pt"
    best_stage_b = float("inf")
    best_stage_c = float("inf")
    best_stage_b_epoch = 0
    best_stage_c_epoch = 0
    best_stage_c_metric_name = resolved_stage_c_selection_metric
    if stage_b_epochs <= 0:
        if init_checkpoint_state is None:
            raise ValueError("stage_b_epochs <= 0 requires --init-summary-json or --init-checkpoint")
        best_stage_b_epoch = int(init_checkpoint_state.get("best_stage_b_epoch", 0))
        torch.save(
            {
                "stage": "b",
                "surrogate_state": model.surrogate.state_dict(),
                "temp_head_state": model.temp_head.state_dict(),
                "power_head_state": model.power_head.state_dict(),
                "artifact_spec": asdict(artifact_spec),
                "temp_head_feature_set": model.temp_head.feature_set,
                "best_stage_b_epoch": best_stage_b_epoch,
            },
            ckpt_path,
        )
        c_zon_after_stage_b = float(model.surrogate.c_zon.detach().cpu().item())
    else:
        model.freeze_backbone()
        opt_b = optim.Adam(model.stage_b_groups(czon_lr=czon_lr))
        wait_b = 0
        for epoch in range(1, stage_b_epochs + 1):
            model.train()
            batch_rows = []
            for batch_np in _subset_epoch_batches(train_idx, batch_size, seed + epoch):
                idx = torch.tensor(batch_np, dtype=torch.long, device=device)
                _, t_pred, _, _, _, c_zon = model(
                    data["t_zone"][idx],
                    data["t_amb"][idx],
                    data["hour"][idx],
                    data["day"][idx],
                    data["a0"][idx],
                    data["a1"][idx],
                )
                loss, metrics = stage_b_criterion(t_pred, data["t_next_target"][idx], c_zon)
                opt_b.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.surrogate.log_c_zon], max_norm=1.0)
                opt_b.step()
                batch_rows.append(metrics)

            train_avg = {k: float(np.mean([r[k] for r in batch_rows])) for k in batch_rows[0]}
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics = _evaluate_stage(model, data, val_idx, stage_b_criterion, stage="b")
            stage_b_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_avg["loss_total"],
                    "train_rmse_temp": train_avg["rmse_temp"],
                    "val_loss": val_loss,
                    "val_rmse_temp": float(val_metrics["rmse_temp"]),
                    "loss_c_prior_train": train_avg["loss_c_prior"],
                    "loss_c_prior_val": float(val_metrics["loss_c_prior"]),
                    "c_zon_j_per_k": float(model.surrogate.c_zon.detach().cpu().item()),
                    "czon_lr": float(opt_b.param_groups[0]["lr"]),
                }
            )
            if epoch == 1 or epoch % 20 == 0:
                print(
                    f"[INV_BOPTEST_V35][B] epoch={epoch:4d} train_rmse={train_avg['rmse_temp']:.4f} "
                    f"val_rmse={val_metrics['rmse_temp']:.4f} c_zon={stage_b_history[-1]['c_zon_j_per_k']:.3e}"
                )

            if val_loss < best_stage_b - 1e-6:
                best_stage_b = val_loss
                best_stage_b_epoch = epoch
                wait_b = 0
                torch.save(
                    {
                        "stage": "b",
                        "surrogate_state": model.surrogate.state_dict(),
                        "temp_head_state": model.temp_head.state_dict(),
                        "power_head_state": model.power_head.state_dict(),
                        "artifact_spec": asdict(artifact_spec),
                        "temp_head_feature_set": model.temp_head.feature_set,
                        "best_stage_b_epoch": best_stage_b_epoch,
                    },
                    ckpt_path,
                )
            else:
                wait_b += 1
                if wait_b >= stage_b_patience:
                    print(f"[INV_BOPTEST_V35][B] Early stopping at epoch {epoch}")
                    break

        stage_b_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.surrogate.load_state_dict(stage_b_ckpt["surrogate_state"])
        model.temp_head.load_state_dict(stage_b_ckpt["temp_head_state"], strict=False)
        model.power_head.load_state_dict(stage_b_ckpt["power_head_state"])
        c_zon_after_stage_b = float(model.surrogate.c_zon.detach().cpu().item())

    model.configure_stage_c(stage_c_mode)
    opt_c = optim.Adam(model.stage_c_groups(backbone_lr=backbone_lr, calib_lr=calib_lr, mode=stage_c_mode))
    sched_c = optim.lr_scheduler.ReduceLROnPlateau(opt_c, mode="min", factor=0.5, patience=15)
    wait_c = 0

    for epoch in range(1, stage_c_epochs + 1):
        model.train()
        batch_rows = []
        rollout_free_run_ratio = 0.0
        if stage_c_mode in ROLLOUT_STAGE_C_MODES and lambda_rollout > 0.0:
            rollout_free_run_ratio = _scheduled_free_run_ratio(
                epoch=epoch,
                total_epochs=stage_c_epochs,
                teacher_forced_epochs=rollout_teacher_forced_epochs,
                free_run_final_ratio=rollout_free_run_final_ratio,
            )
        batch_rng = np.random.default_rng(seed + 20_000 + epoch)
        if stage_c_mode in ROLLOUT_STAGE_C_MODES:
            epoch_batches = _subset_epoch_rollout_batches(train_rollout_pools, batch_size, seed + 10_000 + epoch)
        else:
            epoch_batches = _subset_epoch_batches(train_idx, batch_size, seed + 10_000 + epoch)

        for batch_np in epoch_batches:
            idx = torch.tensor(batch_np, dtype=torch.long, device=device)
            _, t_pred, _, p_pred, q_net, c_zon = model(
                data["t_zone"][idx],
                data["t_amb"][idx],
                data["hour"][idx],
                data["day"][idx],
                data["a0"][idx],
                data["a1"][idx],
            )
            base_loss, metrics = stage_c_criterion(
                t_pred,
                data["t_next_target"][idx],
                p_pred,
                data["p_target"][idx],
                q_net,
                c_zon,
                model.temp_head,
                model.power_head,
            )
            rollout_metrics = {"loss_rollout_temp": 0.0, "rmse_rollout_temp": 0.0}
            total_loss = base_loss
            if stage_c_mode in ROLLOUT_STAGE_C_MODES and lambda_rollout > 0.0:
                free_run_mask_np = batch_rng.random(len(batch_np)) < rollout_free_run_ratio
                rollout_loss, rollout_metrics = _compute_rollout_loss(
                    model,
                    data,
                    start_idx_np=batch_np,
                    rollout_horizons=train_rollout_horizons,
                    free_run_mask_np=free_run_mask_np,
                )
                total_loss = total_loss + lambda_rollout * rollout_loss
            opt_c.zero_grad()
            total_loss.backward()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt_c.step()
            metrics["loss_total"] = float(total_loss.item())
            metrics.update(rollout_metrics)
            batch_rows.append(metrics)

        train_avg = {k: float(np.mean([r[k] for r in batch_rows])) for k in batch_rows[0]}
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = _evaluate_stage(model, data, val_idx, stage_c_criterion, stage="c")
            if stage_c_mode in ROLLOUT_STAGE_C_MODES and len(val_rollout_starts) > 0:
                val_rollout_teacher_loss, val_rollout_teacher_metrics = _compute_rollout_loss(
                    model,
                    data,
                    start_idx_np=val_rollout_starts,
                    rollout_horizons=val_rollout_horizons,
                    free_run_mask_np=np.zeros(len(val_rollout_starts), dtype=bool),
                )
                val_rollout_free_loss, val_rollout_free_metrics = _compute_rollout_loss(
                    model,
                    data,
                    start_idx_np=val_rollout_starts,
                    rollout_horizons=val_rollout_horizons,
                    free_run_mask_np=np.ones(len(val_rollout_starts), dtype=bool),
                )
                val_rollout_mixed_loss = (
                    (1.0 - rollout_free_run_ratio) * val_rollout_teacher_loss
                    + rollout_free_run_ratio * val_rollout_free_loss
                )
                val_rollout_mixed_rmse = float(torch.sqrt(val_rollout_mixed_loss).item())
                val_metrics["loss_rollout_temp_teacher"] = float(val_rollout_teacher_loss.item())
                val_metrics["rmse_rollout_temp_teacher"] = float(val_rollout_teacher_metrics["rmse_rollout_temp"])
                val_metrics["loss_rollout_temp_free"] = float(val_rollout_free_loss.item())
                val_metrics["rmse_rollout_temp_free"] = float(val_rollout_free_metrics["rmse_rollout_temp"])
                val_metrics["loss_rollout_temp_mixed"] = float(val_rollout_mixed_loss.item())
                val_metrics["rmse_rollout_temp_mixed"] = float(val_rollout_mixed_rmse)
                if lambda_rollout > 0.0:
                    val_loss = val_loss + lambda_rollout * float(val_rollout_mixed_loss.item())
                    val_metrics["loss_rollout_temp"] = float(val_rollout_mixed_loss.item())
                    val_metrics["rmse_rollout_temp"] = float(val_rollout_mixed_rmse)
                else:
                    val_metrics["loss_rollout_temp"] = 0.0
                    val_metrics["rmse_rollout_temp"] = 0.0
            else:
                val_metrics["loss_rollout_temp"] = 0.0
                val_metrics["rmse_rollout_temp"] = 0.0
                val_metrics["loss_rollout_temp_teacher"] = 0.0
                val_metrics["rmse_rollout_temp_teacher"] = 0.0
                val_metrics["loss_rollout_temp_free"] = 0.0
                val_metrics["rmse_rollout_temp_free"] = 0.0
                val_metrics["loss_rollout_temp_mixed"] = 0.0
                val_metrics["rmse_rollout_temp_mixed"] = 0.0
            val_metrics["loss_total"] = float(val_loss)
            selection_metric_name = resolved_stage_c_selection_metric
            if selection_metric_name == "val_loss":
                selection_metric = float(val_loss)
            elif selection_metric_name == "val_rollout_rmse_free":
                selection_metric = float(val_metrics["rmse_rollout_temp_free"])
            elif selection_metric_name == "val_rollout_rmse_mixed":
                selection_metric = float(val_metrics["rmse_rollout_temp_mixed"])
            elif selection_metric_name == "val_rollout_loss_free":
                selection_metric = float(val_metrics["loss_rollout_temp_free"])
            elif selection_metric_name == "val_rollout_loss_mixed":
                selection_metric = float(val_metrics["loss_rollout_temp_mixed"])
            elif selection_metric_name == "val_power_mae_w":
                selection_metric = float(val_metrics["mae_power_w"])
            elif selection_metric_name == "val_power_rmse_w":
                selection_metric = float(val_metrics["rmse_power_w"])
            else:
                raise ValueError(f"Unsupported resolved Stage C selection metric: {selection_metric_name}")
        sched_c.step(selection_metric)

        stage_c_history.append(
            {
                "epoch": epoch,
                "train_loss": train_avg["loss_total"],
                "train_rmse_temp": train_avg["rmse_temp"],
                "val_loss": val_loss,
                "val_rmse_temp": float(val_metrics["rmse_temp"]),
                "loss_power_train": train_avg["loss_power"],
                "loss_power_val": float(val_metrics["loss_power"]),
                "loss_c_prior_train": train_avg["loss_c_prior"],
                "loss_c_prior_val": float(val_metrics["loss_c_prior"]),
                "loss_temp_reg_train": train_avg["loss_temp_reg"],
                "loss_temp_reg_val": float(val_metrics["loss_temp_reg"]),
                "power_mae_train_w": train_avg["mae_power_w"],
                "power_mae_val_w": float(val_metrics["mae_power_w"]),
                "power_rmse_train_w": train_avg["rmse_power_w"],
                "power_rmse_val_w": float(val_metrics["rmse_power_w"]),
                "power_bias_train_w": train_avg["bias_power_w"],
                "power_bias_val_w": float(val_metrics["bias_power_w"]),
                "loss_rollout_train": train_avg["loss_rollout_temp"],
                "loss_rollout_val": float(val_metrics["loss_rollout_temp"]),
                "loss_rollout_val_teacher": float(val_metrics["loss_rollout_temp_teacher"]),
                "loss_rollout_val_mixed": float(val_metrics["loss_rollout_temp_mixed"]),
                "rollout_rmse_train": train_avg["rmse_rollout_temp"],
                "rollout_rmse_val": float(val_metrics["rmse_rollout_temp"]),
                "rollout_rmse_val_teacher": float(val_metrics["rmse_rollout_temp_teacher"]),
                "rollout_rmse_val_mixed": float(val_metrics["rmse_rollout_temp_mixed"]),
                "loss_rollout_val_free": float(val_metrics["loss_rollout_temp_free"]),
                "rollout_rmse_val_free": float(val_metrics["rmse_rollout_temp_free"]),
                "rollout_free_run_ratio": float(rollout_free_run_ratio),
                "selection_metric": float(selection_metric),
                "selection_metric_name": selection_metric_name,
                "c_zon_j_per_k": float(model.surrogate.c_zon.detach().cpu().item()),
                "temp_scale": float(model.temp_head.scale_T.item()),
                "temp_bias_c": float(model.temp_head.bias_T.item()),
                "power_scale": float(model.power_head.scale_P.item()),
                "power_bias_w": float(model.power_head.bias_P.item()),
                "backbone_lr": float(next((g["lr"] for g in opt_c.param_groups if g.get("name") == "backbone"), 0.0)),
                "calib_lr": float(
                    next(
                        (g["lr"] for g in opt_c.param_groups if g.get("name") in ("calib", "heads", "temp_head", "power_head")),
                        0.0,
                    )
                ),
            }
        )
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[INV_BOPTEST_V35][C] epoch={epoch:4d} train_rmse={train_avg['rmse_temp']:.4f} "
                f"val_rmse={val_metrics['rmse_temp']:.4f} "
                f"power_mae_val={val_metrics['mae_power_w']:.2f} "
                f"free_ratio={rollout_free_run_ratio:.2f} "
                f"rollout_val_rmse={val_metrics['rmse_rollout_temp']:.4f} "
                f"rollout_val_free_rmse={val_metrics['rmse_rollout_temp_free']:.4f} "
                f"select={selection_metric_name}:{selection_metric:.4f} "
                f"c_zon={stage_c_history[-1]['c_zon_j_per_k']:.3e}"
            )

        if selection_metric < best_stage_c - 1e-6:
            best_stage_c = selection_metric
            best_stage_c_epoch = epoch
            best_stage_c_metric_name = selection_metric_name
            wait_c = 0
            torch.save(
                {
                    "stage": "c",
                    "surrogate_state": model.surrogate.state_dict(),
                    "temp_head_state": model.temp_head.state_dict(),
                    "power_head_state": model.power_head.state_dict(),
                    "artifact_spec": asdict(artifact_spec),
                    "temp_head_feature_set": model.temp_head.feature_set,
                    "best_stage_b_epoch": best_stage_b_epoch,
                    "best_stage_c_epoch": best_stage_c_epoch,
                    "best_stage_c_metric_name": best_stage_c_metric_name,
                    "best_stage_c_metric_value": best_stage_c,
                },
                ckpt_path,
            )
        else:
            wait_c += 1
            if wait_c >= stage_c_patience:
                print(f"[INV_BOPTEST_V35][C] Early stopping at epoch {epoch}")
                break

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.surrogate.load_state_dict(best_ckpt["surrogate_state"])
    model.temp_head.load_state_dict(best_ckpt["temp_head_state"])
    model.power_head.load_state_dict(best_ckpt["power_head_state"])
    model.eval()

    with torch.no_grad():
        t_surr, t_pred, p_surr, p_pred, q_net, c_zon = model(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )
    pred_df = preprocessed_df.copy()
    pred_df["t_pred_before"] = t_base.cpu().numpy()
    pred_df["p_pred_before"] = p_base.cpu().numpy()
    pred_df["t_pred_surrogate_v35"] = t_surr.cpu().numpy()
    pred_df["t_pred_after"] = t_pred.cpu().numpy()
    pred_df["p_pred_after"] = p_pred.cpu().numpy()
    pred_df["p_pred_surrogate_v35"] = p_surr.cpu().numpy()
    pred_df["q_net_after_w"] = q_net.cpu().numpy()
    pred_df["t_target_used"] = data["t_next_target"].cpu().numpy()
    pred_df["p_target_used"] = data["p_target"].cpu().numpy()
    preds_csv = out_dir / "calibration_predictions.csv"
    pred_df.to_csv(preds_csv, index=False)

    after_metrics = _metrics(
        pred_df["t_pred_after"].to_numpy(),
        pred_df["t_target_used"].to_numpy(),
        pred_df["p_pred_after"].to_numpy(),
        pred_df["p_target_used"].to_numpy(),
    )
    c_zon_final = float(c_zon.detach().cpu().item())
    czon_err_pct = (
        float(abs(c_zon_final - artifact_spec.c_zon_true_j_per_k) / artifact_spec.c_zon_true_j_per_k * 100.0)
        if inject_artifacts
        else None
    )

    pd.DataFrame(stage_b_history).to_csv(out_dir / "stage_b_history_v35.csv", index=False)
    pd.DataFrame(stage_c_history).to_csv(out_dir / "stage_c_history_v35.csv", index=False)
    (out_dir / "preprocess_summary.json").write_text(json.dumps(preprocess_summary, indent=2), encoding="utf-8")
    (out_dir / "artifact_spec.json").write_text(json.dumps(asdict(artifact_spec), indent=2), encoding="utf-8")
    excitation_summary["rollout_train"] = train_rollout_summary
    excitation_summary["rollout_val"] = val_rollout_summary
    (out_dir / "excitation_summary.json").write_text(json.dumps(excitation_summary, indent=2), encoding="utf-8")

    summary = {
        "data_path": data_path,
        "model_path": model_path,
        "target_mode": target_mode,
        "inject_artifacts": inject_artifacts,
        "policy_filter": policy,
        "season_filter": season,
        "rows": int(len(pred_df)),
        "best_stage_b_epoch": int(best_stage_b_epoch),
        "best_stage_c_epoch": int(best_stage_c_epoch),
        "best_stage_c_metric_name": best_stage_c_metric_name,
        "best_stage_c_metric_value": float(best_stage_c),
        "baseline_rmse_c": baseline_metrics["rmse_temp_c"],
        "baseline_mae_c": baseline_metrics["mae_temp_c"],
        "baseline_bias_c": baseline_metrics["bias_temp_c"],
        "baseline_power_mae_w": baseline_metrics["mae_power_w"],
        "calibrated_rmse_c": after_metrics["rmse_temp_c"],
        "calibrated_mae_c": after_metrics["mae_temp_c"],
        "calibrated_bias_c": after_metrics["bias_temp_c"],
        "calibrated_power_mae_w": after_metrics["mae_power_w"],
        "improvement_rmse_pct": float(
            (baseline_metrics["rmse_temp_c"] - after_metrics["rmse_temp_c"])
            / max(baseline_metrics["rmse_temp_c"], 1e-6)
            * 100.0
        ),
        "c_zon_prior_j_per_k": float(c_zon_prior),
        "c_zon_after_stage_b_j_per_k": c_zon_after_stage_b,
        "c_zon_final_j_per_k": c_zon_final,
        "czon_error_pct": czon_err_pct,
        "runtime_step_sec": int(runtime_step_sec),
        "legacy_checkpoint_step_sec": int(legacy_step_sec),
        "stage_c_mode": stage_c_mode,
        "temp_head_feature_set": model.temp_head.feature_set,
        "stage_c_selection_metric": resolved_stage_c_selection_metric,
        "init_summary_json": init_summary_json,
        "init_checkpoint": init_checkpoint_path,
        "temp_head_type": "nonlinear_residual_mlp",
        "temp_scale_final": float(model.temp_head.scale_T.item()),
        "temp_bias_final_c": float(model.temp_head.bias_T.item()),
        "power_scale_final": float(model.power_head.scale_P.item()),
        "power_bias_final_w": float(model.power_head.bias_P.item()),
        "stage_b_epochs_ran": int(len(stage_b_history)),
        "stage_c_epochs_ran": int(len(stage_c_history)),
        "czon_lr": float(czon_lr),
        "backbone_lr": float(backbone_lr),
        "calib_lr": float(calib_lr),
        "lambda_temp_reg": float(lambda_temp_reg),
        "lambda_rollout": float(lambda_rollout),
        "rollout_teacher_forced_epochs": int(rollout_teacher_forced_epochs),
        "rollout_free_run_final_ratio": float(rollout_free_run_final_ratio),
        "rollout_horizons": rollout_horizons_list,
        "rows_train_all": int(len(train_all_idx)),
        "train_rollout_window_count": int(len(train_rollout_starts)),
        "val_rollout_window_count": int(len(val_rollout_starts)),
        "train_rollout_segment_count": int(len(train_rollout_segments)),
        "val_rollout_segment_count": int(len(val_rollout_segments)),
        "preprocess_summary": preprocess_summary,
        "excitation_summary": excitation_summary,
    }
    summary_json = out_dir / "calibration_summary_boptest_v35.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("INVERSE CALIBRATION ON REAL BOPTEST TRACE (V3.5)")
    print("=" * 72)
    print(f"Baseline RMSE:      {baseline_metrics['rmse_temp_c']:.4f} C")
    print(f"Calibrated RMSE:    {after_metrics['rmse_temp_c']:.4f} C")
    print(f"Improvement:        {summary['improvement_rmse_pct']:.1f}%")
    print(f"Runtime step:       {runtime_step_sec} s")
    print(f"Legacy step:        {legacy_step_sec} s")
    print(f"C_zon after stageB: {c_zon_after_stage_b:.3e} J/K")
    print(f"C_zon final:        {c_zon_final:.3e} J/K")
    if czon_err_pct is not None:
        print(f"C_zon error:        {czon_err_pct:.2f}%")
    print(f"Observed log:       {observed_csv}")
    print(f"Preprocessed:       {preprocessed_csv}")
    print(f"Predictions:        {preds_csv}")
    print(f"Summary:            {summary_json}")
    return str(summary_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged inverse calibration for Surrogate v3.5 with explicit structural C_zon on real BOPTEST traces."
    )
    parser.add_argument(
        "--preset",
        choices=[
            "smoke",
            "full",
            "realclean",
            "block2_15min",
            "block2_15min_rollout",
            "block1_15min_episodeaware",
            "block1_15min_power_head_only",
            "block1_3_15min_closed_loop",
        ],
        default=None,
    )
    parser.add_argument("--data", default="data/surrogate_v2/boptest_v2_tsupply.csv")
    parser.add_argument("--model", default="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--output_dir", default="outputs/surrogate_v35_inverse_boptest")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--no-artifact-injection", action="store_true")
    parser.add_argument("--temp-bias-c", type=float, default=0.5)
    parser.add_argument("--temp-noise-std", type=float, default=0.08)
    parser.add_argument("--temp-latency-steps", type=int, default=2)
    parser.add_argument("--power-scale", type=float, default=1.04)
    parser.add_argument("--power-bias-w", type=float, default=35.0)
    parser.add_argument("--power-noise-rel", type=float, default=0.015)
    parser.add_argument("--c-zon-true", type=float, default=4.2e5)
    parser.add_argument("--surrogate-czon-ref", type=float, default=5.3e5)
    parser.add_argument("--max-latency-search", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--target-mode", default="clean", choices=["clean", "preprocessed"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--stage-b-epochs", type=int, default=120)
    parser.add_argument("--stage-b-patience", type=int, default=20)
    parser.add_argument("--stage-c-epochs", type=int, default=180)
    parser.add_argument("--stage-c-patience", type=int, default=30)
    parser.add_argument("--czon-lr", type=float, default=1e-2)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--calib-lr", type=float, default=1e-2)
    parser.add_argument("--lambda-c-prior-b", type=float, default=0.1)
    parser.add_argument("--lambda-c-prior-c", type=float, default=0.05)
    parser.add_argument("--lambda-power", type=float, default=0.08)
    parser.add_argument("--lambda-q-reg", type=float, default=0.001)
    parser.add_argument("--lambda-temp-reg", type=float, default=0.02)
    parser.add_argument("--lambda-rollout", type=float, default=0.0)
    parser.add_argument("--rollout-teacher-forced-epochs", type=int, default=10)
    parser.add_argument("--rollout-free-run-final-ratio", type=float, default=0.5)
    parser.add_argument("--excitation-quantile", type=float, default=0.8)
    parser.add_argument("--excitation-mix-ratio", type=float, default=0.6)
    parser.add_argument("--excitation-mode", choices=["hybrid", "dt_only"], default="hybrid")
    parser.add_argument("--c-zon-prior", type=float, default=5.3e5)
    parser.add_argument("--c-zon-min", type=float, default=5.0e4)
    parser.add_argument("--q-scale", type=float, default=3000.0)
    parser.add_argument("--temp-head-feature-set", choices=["v1", "block13_rich"], default="v1")
    parser.add_argument("--step-sec", type=int, default=None)
    parser.add_argument("--legacy-step-sec", type=int, default=int(RCNeuralODEv35.DT))
    parser.add_argument("--init-summary-json", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--stage-c-selection-metric",
        choices=[
            "auto",
            "val_loss",
            "val_rollout_rmse_free",
            "val_rollout_rmse_mixed",
            "val_rollout_loss_free",
            "val_rollout_loss_mixed",
            "val_power_mae_w",
            "val_power_rmse_w",
        ],
        default="auto",
    )
    parser.add_argument(
        "--stage-c-mode",
        choices=[
            "joint",
            "heads_only",
            "rollout_heads_only",
            "rollout_temp_head_only",
            "power_head_only",
        ],
        default="joint",
    )
    parser.add_argument("--rollout-horizons", default="4,8")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    args = _apply_preset(args)

    spec = ArtifactSpec(
        temp_bias_c=args.temp_bias_c,
        temp_noise_std=args.temp_noise_std,
        temp_latency_steps=args.temp_latency_steps,
        power_scale=args.power_scale,
        power_bias_w=args.power_bias_w,
        power_noise_rel=args.power_noise_rel,
        c_zon_true_j_per_k=args.c_zon_true,
        surrogate_czon_ref_j_per_k=args.surrogate_czon_ref,
    )

    calibrate_boptest_v35(
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output_dir,
        policy=args.policy,
        season=args.season,
        limit_rows=args.limit_rows,
        inject_artifacts=not args.no_artifact_injection,
        artifact_spec=spec,
        max_latency_search=args.max_latency_search,
        smooth_window=args.smooth_window,
        target_mode=args.target_mode,
        stage_b_epochs=args.stage_b_epochs,
        stage_b_patience=args.stage_b_patience,
        stage_c_epochs=args.stage_c_epochs,
        stage_c_patience=args.stage_c_patience,
        batch_size=args.batch_size,
        val_split=args.val_split,
        czon_lr=args.czon_lr,
        backbone_lr=args.backbone_lr,
        calib_lr=args.calib_lr,
        lambda_c_prior_b=args.lambda_c_prior_b,
        lambda_c_prior_c=args.lambda_c_prior_c,
        lambda_power=args.lambda_power,
        lambda_q_reg=args.lambda_q_reg,
        lambda_temp_reg=args.lambda_temp_reg,
        lambda_rollout=args.lambda_rollout,
        rollout_teacher_forced_epochs=args.rollout_teacher_forced_epochs,
        rollout_free_run_final_ratio=args.rollout_free_run_final_ratio,
        excitation_quantile=args.excitation_quantile,
        excitation_mix_ratio=args.excitation_mix_ratio,
        excitation_mode=args.excitation_mode,
        c_zon_prior=args.c_zon_prior,
        c_zon_min=args.c_zon_min,
        q_scale=args.q_scale,
        stage_c_mode=args.stage_c_mode,
        stage_c_selection_metric=args.stage_c_selection_metric,
        rollout_horizons=args.rollout_horizons,
        temp_head_feature_set=args.temp_head_feature_set,
        step_sec=args.step_sec,
        legacy_step_sec=args.legacy_step_sec,
        init_summary_json=args.init_summary_json,
        init_checkpoint=args.init_checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
