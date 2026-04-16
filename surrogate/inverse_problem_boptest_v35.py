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


class PowerCalibrationHeadV35(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_P = nn.Parameter(torch.tensor(1.0))
        self.bias_P = nn.Parameter(torch.tensor(0.0))

    def forward(self, p_surr: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.scale_P * p_surr + self.bias_P, min=0.0)

    def regularization_loss(self) -> torch.Tensor:
        return 0.2 * (self.scale_P - 1.0) ** 2 + (self.bias_P / 1000.0) ** 2


class TempCalibrationHeadV35(nn.Module):
    def __init__(
        self,
        t_min: float = 15.0,
        t_max: float = 35.0,
        t_zone_min: float = 15.0,
        t_zone_max: float = 35.0,
        t_amb_min: float = -10.0,
        t_amb_max: float = 40.0,
        hidden_dim: int = 32,
        residual_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.t_zone_min = float(t_zone_min)
        self.t_zone_max = float(t_zone_max)
        self.t_amb_min = float(t_amb_min)
        self.t_amb_max = float(t_amb_max)
        self.residual_scale = float(residual_scale)
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

    def forward(
        self,
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
        delta_t = torch.clamp((t_surr - t_zone) / 5.0, min=-3.0, max=3.0)
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
    def __init__(self, surrogate: RCNeuralODEv35):
        super().__init__()
        self.surrogate = surrogate
        self.temp_head = TempCalibrationHeadV35(
            t_min=surrogate.T_ZONE_MIN,
            t_max=surrogate.T_ZONE_MAX,
            t_zone_min=surrogate.T_ZONE_MIN,
            t_zone_max=surrogate.T_ZONE_MAX,
            t_amb_min=surrogate.T_AMB_MIN,
            t_amb_max=surrogate.T_AMB_MAX,
        )
        self.power_head = PowerCalibrationHeadV35()

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
        t_cal = self.temp_head(t_zone, t_amb, hour, day, a0, a1, t_surr)
        p_cal = self.power_head(p_surr)
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
        lambda_power: float = 0.08,
        lambda_c_prior: float = 0.05,
        lambda_q_reg: float = 0.001,
        lambda_power_reg: float = 0.02,
        lambda_temp_reg: float = 0.02,
    ) -> None:
        super().__init__()
        self.c_prior = float(c_prior)
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
        log_c = torch.log(torch.clamp(c_zon, min=1.0))
        log_prior = torch.log(torch.tensor(self.c_prior, device=c_zon.device, dtype=c_zon.dtype))
        l_prior = (log_c - log_prior) ** 2
        l_q = torch.mean((q_net / 3000.0) ** 2)
        l_temp_reg = temp_head.regularization_loss()
        l_power_reg = power_head.regularization_loss()
        total = (
            l_temp
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
        "stage_c_mode": "joint",
        "rollout_horizons": "4,8",
        "seed": None,
    }

    def _maybe_set(name: str, value) -> None:
        current = getattr(args, name)
        default = parser_defaults[name]
        if current is None or current == default:
            setattr(args, name, value)

    _maybe_set("policy", "mixed")
    _maybe_set("seed", 42)

    if args.preset == "smoke":
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


def _build_rollout_window_starts(start_idx: int, end_idx: int, max_horizon: int) -> np.ndarray:
    if max_horizon <= 0:
        return np.array([], dtype=int)
    last_start = end_idx - max_horizon
    if last_start < start_idx:
        return np.array([], dtype=int)
    return np.arange(start_idx, last_start + 1, dtype=int)


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
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n = len(df)
    n_train = max(1, int(n * (1.0 - val_split)))
    train_all = np.arange(0, n_train)
    val_idx = np.arange(n_train, n)
    if len(val_idx) == 0:
        val_idx = train_all.copy()

    scores = _compute_excitation_scores(df, mode=excitation_mode)
    threshold = float(np.quantile(scores[:n_train], excitation_quantile))
    exc_idx = train_all[scores[:n_train] >= threshold]
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
        "score_mean_train": float(np.mean(scores[:n_train])) if len(train_all) else 0.0,
        "score_mean_excitation": float(np.mean(scores[exc_idx])) if len(exc_idx) else 0.0,
    }
    return train_idx, val_idx, summary


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
    rollout_horizons: str | list[int] | tuple[int, ...],
    seed: int,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    clean_df = _load_clean_df(data_path, policy, season, limit_rows)
    warm_model = load_v35_from_v2_checkpoint(
        model_path,
        device=device,
        c_zon_init=c_zon_prior,
        c_zon_min=c_zon_min,
        q_scale=q_scale,
    )

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
    train_idx, val_idx, excitation_summary = _build_training_index_sets(
        preprocessed_df,
        val_split=val_split,
        excitation_quantile=excitation_quantile,
        excitation_mix_ratio=excitation_mix_ratio,
        excitation_mode=excitation_mode,
        seed=seed,
    )
    rollout_horizons_list = _parse_rollout_horizons(rollout_horizons)
    n_rows = len(preprocessed_df)
    n_train_all = max(1, int(n_rows * (1.0 - val_split)))
    val_start = n_train_all if n_train_all < n_rows else 0

    def _valid_horizons(segment_len: int) -> list[int]:
        return [h for h in rollout_horizons_list if h <= segment_len]

    train_rollout_horizons = _valid_horizons(n_train_all)
    val_rollout_horizons = _valid_horizons(n_rows - val_start)
    train_rollout_starts = _build_rollout_window_starts(
        start_idx=0,
        end_idx=n_train_all,
        max_horizon=max(train_rollout_horizons, default=0),
    )
    val_rollout_starts = _build_rollout_window_starts(
        start_idx=val_start,
        end_idx=n_rows,
        max_horizon=max(val_rollout_horizons, default=0),
    )

    if stage_c_mode == "rollout_heads_only" and len(train_rollout_starts) == 0:
        raise ValueError(
            "rollout_heads_only requires enough sequential training rows for the requested rollout horizons"
        )

    with torch.no_grad():
        t_base, p_base = warm_model(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )
        baseline_metrics = _metrics(
            t_base.cpu().numpy(),
            data["t_next_target"].cpu().numpy(),
            p_base.cpu().numpy(),
            data["p_target"].cpu().numpy(),
        )

    model = StagedCalibratedSurrogateV35(warm_model).to(device)
    stage_b_criterion = StageBLossV35(c_prior=c_zon_prior, lambda_c_prior=lambda_c_prior_b)
    stage_c_criterion = StageCLossV35(
        c_prior=c_zon_prior,
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
    best_stage_c_metric_name = "val_loss"

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
    model.temp_head.load_state_dict(stage_b_ckpt["temp_head_state"])
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
        if stage_c_mode == "rollout_heads_only" and lambda_rollout > 0.0:
            rollout_free_run_ratio = _scheduled_free_run_ratio(
                epoch=epoch,
                total_epochs=stage_c_epochs,
                teacher_forced_epochs=rollout_teacher_forced_epochs,
                free_run_final_ratio=rollout_free_run_final_ratio,
            )
        batch_rng = np.random.default_rng(seed + 20_000 + epoch)
        if stage_c_mode == "rollout_heads_only":
            epoch_batches = _subset_epoch_batches(train_rollout_starts, batch_size, seed + 10_000 + epoch)
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
            if stage_c_mode == "rollout_heads_only" and lambda_rollout > 0.0:
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
            if stage_c_mode == "rollout_heads_only" and lambda_rollout > 0.0 and len(val_rollout_starts) > 0:
                if rollout_free_run_ratio <= 0.0:
                    val_rollout_loss, val_rollout_metrics = _compute_rollout_loss(
                        model,
                        data,
                        start_idx_np=val_rollout_starts,
                        rollout_horizons=val_rollout_horizons,
                        free_run_mask_np=np.zeros(len(val_rollout_starts), dtype=bool),
                    )
                elif rollout_free_run_ratio >= 1.0:
                    val_rollout_loss, val_rollout_metrics = _compute_rollout_loss(
                        model,
                        data,
                        start_idx_np=val_rollout_starts,
                        rollout_horizons=val_rollout_horizons,
                        free_run_mask_np=np.ones(len(val_rollout_starts), dtype=bool),
                    )
                else:
                    val_rollout_teacher, _ = _compute_rollout_loss(
                        model,
                        data,
                        start_idx_np=val_rollout_starts,
                        rollout_horizons=val_rollout_horizons,
                        free_run_mask_np=np.zeros(len(val_rollout_starts), dtype=bool),
                    )
                    val_rollout_free, _ = _compute_rollout_loss(
                        model,
                        data,
                        start_idx_np=val_rollout_starts,
                        rollout_horizons=val_rollout_horizons,
                        free_run_mask_np=np.ones(len(val_rollout_starts), dtype=bool),
                    )
                    val_rollout_loss = (
                        (1.0 - rollout_free_run_ratio) * val_rollout_teacher
                        + rollout_free_run_ratio * val_rollout_free
                    )
                    val_rollout_metrics = {
                        "loss_rollout_temp": float(val_rollout_loss.item()),
                        "rmse_rollout_temp": float(torch.sqrt(val_rollout_loss).item()),
                    }
                val_loss = val_loss + lambda_rollout * float(val_rollout_loss.item())
                val_metrics["loss_total"] = float(val_loss)
                val_metrics.update(val_rollout_metrics)
            else:
                val_metrics["loss_rollout_temp"] = 0.0
                val_metrics["rmse_rollout_temp"] = 0.0
            if stage_c_mode == "rollout_heads_only" and len(val_rollout_starts) > 0:
                val_rollout_free_loss, val_rollout_free_metrics = _compute_rollout_loss(
                    model,
                    data,
                    start_idx_np=val_rollout_starts,
                    rollout_horizons=val_rollout_horizons,
                    free_run_mask_np=np.ones(len(val_rollout_starts), dtype=bool),
                )
                val_metrics["loss_rollout_temp_free"] = float(val_rollout_free_loss.item())
                val_metrics["rmse_rollout_temp_free"] = float(val_rollout_free_metrics["rmse_rollout_temp"])
                selection_metric = float(val_rollout_free_metrics["rmse_rollout_temp"])
                selection_metric_name = "val_rollout_rmse_free"
            else:
                val_metrics["loss_rollout_temp_free"] = float(val_metrics["loss_rollout_temp"])
                val_metrics["rmse_rollout_temp_free"] = float(val_metrics["rmse_rollout_temp"])
                selection_metric = float(val_loss)
                selection_metric_name = "val_loss"
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
                "loss_rollout_train": train_avg["loss_rollout_temp"],
                "loss_rollout_val": float(val_metrics["loss_rollout_temp"]),
                "rollout_rmse_train": train_avg["rmse_rollout_temp"],
                "rollout_rmse_val": float(val_metrics["rmse_rollout_temp"]),
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
                "calib_lr": float(next((g["lr"] for g in opt_c.param_groups if g.get("name") in ("calib", "heads")), 0.0)),
            }
        )
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[INV_BOPTEST_V35][C] epoch={epoch:4d} train_rmse={train_avg['rmse_temp']:.4f} "
                f"val_rmse={val_metrics['rmse_temp']:.4f} "
                f"free_ratio={rollout_free_run_ratio:.2f} "
                f"rollout_val_rmse={val_metrics['rmse_rollout_temp']:.4f} "
                f"rollout_val_free_rmse={val_metrics['rmse_rollout_temp_free']:.4f} "
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
        "stage_c_mode": stage_c_mode,
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
        "train_rollout_window_count": int(len(train_rollout_starts)),
        "val_rollout_window_count": int(len(val_rollout_starts)),
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
    parser.add_argument("--preset", choices=["smoke", "full", "realclean"], default=None)
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
    parser.add_argument(
        "--stage-c-mode",
        choices=[
            "joint",
            "heads_only",
            "rollout_heads_only",
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
        rollout_horizons=args.rollout_horizons,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
