from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
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

from surrogate.rc_node_v2 import RCNeuralODEv2


T_ZONE_MIN = 15.0
T_ZONE_MAX = 35.0
T_AMB_MIN = -10.0
T_AMB_MAX = 40.0
P_MAX = 5500.0


@dataclass
class ArtifactSpec:
    temp_bias_c: float = 0.5
    temp_noise_std: float = 0.08
    temp_latency_steps: int = 2
    power_scale: float = 1.04
    power_bias_w: float = 35.0
    power_noise_rel: float = 0.015
    c_zon_true_j_per_k: float = 4.2e5
    surrogate_czon_ref_j_per_k: float = 5.3e5

    @property
    def scale_dT_true(self) -> float:
        return float(self.surrogate_czon_ref_j_per_k / self.c_zon_true_j_per_k)


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = args.preset
    if preset is None:
        return args

    # Presets intentionally override defaults so the script can be run as a single-file entry point.
    args.policy = "mixed"
    args.seed = 42 if args.seed is None else args.seed

    if preset == "smoke":
        args.output_dir = "outputs/surrogate_v2_inverse_boptest_smoke"
        args.limit_rows = 600
        args.epochs = 5
        args.batch_size = 128
        args.patience = 5
        args.backbone_lr = 1e-4
        args.head_lr = 5e-3
        args.no_artifact_injection = False
        args.target_mode = "clean"
        args.temp_bias_c = 0.5
        args.temp_noise_std = 0.08
        args.temp_latency_steps = 2
        args.power_scale = 1.04
        args.power_bias_w = 35.0
        args.power_noise_rel = 0.015
        args.c_zon_true = 4.2e5
        args.surrogate_czon_ref = 5.3e5
        args.max_latency_search = 6
        args.smooth_window = 5
    elif preset == "full":
        args.output_dir = "outputs/surrogate_v2_inverse_boptest"
        args.limit_rows = None
        args.epochs = 200
        args.batch_size = 256
        args.patience = 30
        args.backbone_lr = 1e-4
        args.head_lr = 5e-3
        args.no_artifact_injection = False
        args.target_mode = "clean"
        args.temp_bias_c = 0.5
        args.temp_noise_std = 0.08
        args.temp_latency_steps = 2
        args.power_scale = 1.04
        args.power_bias_w = 35.0
        args.power_noise_rel = 0.015
        args.c_zon_true = 4.2e5
        args.surrogate_czon_ref = 5.3e5
        args.max_latency_search = 6
        args.smooth_window = 5
    elif preset == "realclean":
        args.output_dir = "outputs/surrogate_v2_inverse_boptest_realclean"
        args.limit_rows = None
        args.epochs = 200
        args.batch_size = 256
        args.patience = 30
        args.backbone_lr = 1e-4
        args.head_lr = 5e-3
        args.no_artifact_injection = True
        args.target_mode = "clean"
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return args


def _norm_t_zone(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * (t - T_ZONE_MIN) / (T_ZONE_MAX - T_ZONE_MIN) - 1.0


def _norm_t_amb(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * (t - T_AMB_MIN) / (T_AMB_MAX - T_AMB_MIN) - 1.0


def _encode_hour(hour: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rad = 2.0 * np.pi * hour / 24.0
    return torch.sin(rad), torch.cos(rad)


def _encode_day(day: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rad = 2.0 * np.pi * day / 365.0
    return torch.sin(rad), torch.cos(rad)


def _delay_array(arr: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return arr.copy()
    out = np.empty_like(arr)
    out[:lag] = arr[0]
    out[lag:] = arr[:-lag]
    return out


def _undo_delay_array(arr: np.ndarray, lag: int) -> np.ndarray:
    if arr.size == 0:
        return arr.copy()
    if lag <= 0:
        return arr.copy()
    out = np.empty_like(arr)
    out[:-lag] = arr[lag:]
    out[-lag:] = arr[-1]
    return out


def _rolling_denoise(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    med = series.rolling(window=window, center=True, min_periods=1).median()
    return med.ewm(span=window, adjust=False).mean()


def _load_clean_df(csv_path: str, policy: str | None, season: str | None, limit_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    available_policies = sorted(df["policy"].dropna().astype(str).unique().tolist()) if "policy" in df.columns else []
    available_seasons = sorted(df["season"].dropna().astype(str).unique().tolist()) if "season" in df.columns else []
    if policy:
        df = df[df["policy"] == policy]
    if season:
        df = df[df["season"] == season]
    if limit_rows:
        df = df.iloc[:limit_rows]
    df = df.reset_index(drop=True)

    required = ["t_zone", "t_amb", "hour", "day", "a0_raw", "a1_raw", "t_zone_next", "p_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in BOPTEST CSV: {missing}")
    if df.empty:
        raise ValueError(
            "Filtered BOPTEST dataframe is empty. "
            f"policy={policy!r}, season={season!r}, limit_rows={limit_rows!r}, csv={csv_path}. "
            f"Available policies={available_policies}, available seasons={available_seasons}."
        )
    return df


def _load_surrogate(model_path: str, device: torch.device) -> RCNeuralODEv2:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=int(ckpt.get("hidden_dim", 64))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _predict_surrogate(model: RCNeuralODEv2, df: pd.DataFrame, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        t_zone = torch.tensor(df["t_zone"].values, dtype=torch.float32, device=device)
        t_amb = torch.tensor(df["t_amb"].values, dtype=torch.float32, device=device)
        hour = torch.tensor(df["hour"].values, dtype=torch.float32, device=device)
        day = torch.tensor(df["day"].values, dtype=torch.float32, device=device)
        a0 = torch.tensor(df["a0_raw"].values, dtype=torch.float32, device=device)
        a1 = torch.tensor(df["a1_raw"].values, dtype=torch.float32, device=device)
        t_next, p_total = model(t_zone, t_amb, hour, day, a0, a1)
    return t_next.cpu().numpy(), p_total.cpu().numpy()


def inject_boptest_artifacts(
    clean_df: pd.DataFrame,
    model: RCNeuralODEv2,
    device: torch.device,
    spec: ArtifactSpec,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = clean_df.copy()
    t_surr, p_surr = _predict_surrogate(model, clean_df, device)

    t_zone_clean = clean_df["t_zone"].to_numpy(dtype=float)
    t_next_obs = t_zone_clean + spec.scale_dT_true * (t_surr - t_zone_clean) + spec.temp_bias_c
    t_curr_obs = t_zone_clean + spec.temp_bias_c

    if spec.temp_noise_std > 0:
        t_curr_obs = t_curr_obs + rng.normal(0.0, spec.temp_noise_std, size=len(t_curr_obs))
        t_next_obs = t_next_obs + rng.normal(0.0, spec.temp_noise_std, size=len(t_next_obs))

    t_curr_obs = _delay_array(t_curr_obs, spec.temp_latency_steps)
    t_next_obs = _delay_array(t_next_obs, spec.temp_latency_steps)

    p_obs = spec.power_scale * p_surr + spec.power_bias_w
    if spec.power_noise_rel > 0:
        p_obs = p_obs * (1.0 + rng.normal(0.0, spec.power_noise_rel, size=len(p_obs)))
    p_obs = np.clip(p_obs, 0.0, None)

    observed["t_zone_clean"] = clean_df["t_zone"]
    observed["t_zone_next_clean"] = clean_df["t_zone_next"]
    observed["p_total_clean"] = clean_df["p_total"]
    observed["t_zone"] = np.round(t_curr_obs, 4)
    observed["t_zone_next"] = np.round(t_next_obs, 4)
    observed["delta_t"] = np.round(observed["t_zone_next"] - observed["t_zone"], 4)
    observed["p_total"] = np.round(p_obs, 2)
    observed["t_zone_next_surrogate"] = np.round(t_surr, 4)
    observed["p_total_surrogate"] = np.round(p_surr, 2)
    observed["artifact_scale_dT_true"] = spec.scale_dT_true
    observed["artifact_bias_t_true_c"] = spec.temp_bias_c
    observed["artifact_latency_true_steps"] = spec.temp_latency_steps
    observed["artifact_power_scale_true"] = spec.power_scale
    observed["artifact_power_bias_true_w"] = spec.power_bias_w
    observed["artifact_c_zon_true_j_per_k"] = spec.c_zon_true_j_per_k
    return observed


def preprocess_artifacts(
    observed_df: pd.DataFrame,
    model: RCNeuralODEv2,
    device: torch.device,
    max_latency_search: int,
    smooth_window: int,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    work = observed_df.copy()
    pre_summary: Dict[str, float] = {}

    best_lag = 0
    best_rmse = float("inf")
    for lag in range(max_latency_search + 1):
        cand = work.copy()
        cand["t_zone"] = _undo_delay_array(cand["t_zone"].to_numpy(dtype=float), lag)
        cand["t_zone_next"] = _undo_delay_array(cand["t_zone_next"].to_numpy(dtype=float), lag)
        t_pred, _ = _predict_surrogate(model, cand, device)
        rmse = float(np.sqrt(np.mean((t_pred - cand["t_zone_next"].to_numpy(dtype=float)) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag

    work["t_zone"] = _undo_delay_array(work["t_zone"].to_numpy(dtype=float), best_lag)
    work["t_zone_next"] = _undo_delay_array(work["t_zone_next"].to_numpy(dtype=float), best_lag)
    pre_summary["latency_est_steps"] = float(best_lag)
    pre_summary["latency_search_rmse_c"] = float(best_rmse)

    t_pred_after_lag, p_pred_after_lag = _predict_surrogate(model, work, device)
    bias_est = float(np.median(work["t_zone_next"].to_numpy(dtype=float) - t_pred_after_lag))
    work["t_zone"] = work["t_zone"] - bias_est
    work["t_zone_next"] = work["t_zone_next"] - bias_est
    pre_summary["temp_bias_est_c"] = bias_est

    t_pred_after_bias, p_pred_after_bias = _predict_surrogate(model, work, device)
    p_pred = p_pred_after_bias.astype(float)
    p_obs = work["p_total"].to_numpy(dtype=float)
    A = np.column_stack([p_pred, np.ones_like(p_pred)])
    scale_est, bias_p_est = np.linalg.lstsq(A, p_obs, rcond=None)[0]
    scale_est = float(scale_est)
    bias_p_est = float(bias_p_est)
    if abs(scale_est) < 1e-6:
        scale_est = 1.0
    work["p_total"] = np.clip((work["p_total"] - bias_p_est) / scale_est, 0.0, None)
    pre_summary["power_scale_est"] = scale_est
    pre_summary["power_bias_est_w"] = bias_p_est

    for col in ["t_zone", "t_zone_next", "p_total"]:
        work[col] = _rolling_denoise(work[col], smooth_window)
    work["delta_t"] = work["t_zone_next"] - work["t_zone"]

    t_pred_final, p_pred_final = _predict_surrogate(model, work, device)
    pre_summary["postprocess_rmse_c"] = float(
        np.sqrt(np.mean((t_pred_final - work["t_zone_next"].to_numpy(dtype=float)) ** 2))
    )
    pre_summary["postprocess_power_mae_w"] = float(
        np.mean(np.abs(p_pred_final - work["p_total"].to_numpy(dtype=float)))
    )
    return work, pre_summary


class NonlinearCalibrationHeadV3(nn.Module):
    def __init__(self, hidden_dim: int = 32, p_max: float = P_MAX) -> None:
        super().__init__()
        self.p_max = p_max
        self.scale_dT = nn.Parameter(torch.tensor(1.0))
        self.bias_T = nn.Parameter(torch.tensor(0.0))
        self.scale_P = nn.Parameter(torch.tensor(1.0))
        self.bias_P = nn.Parameter(torch.tensor(0.0))
        self.temp_res_scale = 2.0
        self.power_res_scale = 300.0

        self.temp_head = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.power_head = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        t_curr: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        t_surr: torch.Tensor,
        p_surr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = t_surr - t_curr
        h_sin, h_cos = _encode_hour(hour)
        d_sin, d_cos = _encode_day(day)
        feats = torch.stack(
            [
                _norm_t_zone(t_curr),
                _norm_t_amb(t_amb),
                h_sin,
                h_cos,
                d_sin,
                d_cos,
                a0,
                a1,
                delta,
                p_surr / self.p_max,
            ],
            dim=-1,
        )
        temp_res = self.temp_head(feats).squeeze(-1) * self.temp_res_scale
        power_res = self.power_head(feats).squeeze(-1) * self.power_res_scale
        t_cal = torch.clamp(t_curr + self.scale_dT * delta + self.bias_T + temp_res, min=T_ZONE_MIN, max=T_ZONE_MAX)
        p_cal = torch.clamp(self.scale_P * p_surr + self.bias_P + power_res, min=0.0)
        return t_cal, p_cal, temp_res, power_res


class JointCalibratedSurrogateV3(nn.Module):
    def __init__(self, surrogate: RCNeuralODEv2, calibration: NonlinearCalibrationHeadV3) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.calibration = calibration

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_surr, p_surr = self.surrogate(t_zone, t_amb, hour, day, a0, a1)
        t_cal, p_cal, temp_res, power_res = self.calibration(
            t_zone, t_amb, hour, day, a0, a1, t_surr, p_surr
        )
        return t_surr, p_surr, t_cal, p_cal, temp_res, power_res

    def trainable_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        return [
            {"params": self.surrogate.parameters(), "lr": backbone_lr},
            {"params": self.calibration.parameters(), "lr": head_lr},
        ]


class JointCalibrationLossV3(nn.Module):
    def __init__(self, lambda_temp: float = 1.0, lambda_power: float = 0.08, lambda_reg: float = 0.02) -> None:
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_power = lambda_power
        self.lambda_reg = lambda_reg

    def forward(
        self,
        t_pred: torch.Tensor,
        t_true: torch.Tensor,
        p_pred: torch.Tensor,
        p_true: torch.Tensor,
        temp_res: torch.Tensor,
        power_res: torch.Tensor,
        calib: NonlinearCalibrationHeadV3,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        l_temp = torch.mean((t_pred - t_true) ** 2)
        p_scale = torch.clamp(p_true.max().detach(), min=1.0)
        l_power = torch.mean(((p_pred - p_true) / p_scale) ** 2)
        l_reg = (
            0.25 * (calib.scale_dT - 1.0) ** 2
            + 0.05 * calib.bias_T ** 2
            + 0.25 * (calib.scale_P - 1.0) ** 2
            + calib.bias_P ** 2 / 2000.0
            + 0.05 * torch.mean(temp_res ** 2)
            + 0.05 * torch.mean((power_res / p_scale) ** 2)
        )
        total = self.lambda_temp * l_temp + self.lambda_power * l_power + self.lambda_reg * l_reg
        return total, {
            "loss_total": float(total.item()),
            "loss_temp": float(l_temp.item()),
            "loss_power": float(l_power.item()),
            "loss_reg": float(l_reg.item()),
            "rmse_temp": float(torch.sqrt(l_temp).item()),
        }


def _split_indices(n: int, val_split: float) -> tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(n * (1.0 - val_split)))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n)
    if len(val_idx) == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


def _frame_to_tensors(df: pd.DataFrame, target_mode: str) -> Dict[str, torch.Tensor]:
    out = {
        "t_zone": torch.tensor(df["t_zone"].values, dtype=torch.float32),
        "t_amb": torch.tensor(df["t_amb"].values, dtype=torch.float32),
        "hour": torch.tensor(df["hour"].values, dtype=torch.float32),
        "day": torch.tensor(df["day"].values, dtype=torch.float32),
        "a0": torch.tensor(df["a0_raw"].values, dtype=torch.float32),
        "a1": torch.tensor(df["a1_raw"].values, dtype=torch.float32),
    }
    if target_mode == "clean":
        out["t_next_target"] = torch.tensor(df["t_zone_next_clean"].values, dtype=torch.float32)
        out["p_target"] = torch.tensor(df["p_total_clean"].values, dtype=torch.float32)
    else:
        out["t_next_target"] = torch.tensor(df["t_zone_next"].values, dtype=torch.float32)
        out["p_target"] = torch.tensor(df["p_total"].values, dtype=torch.float32)
    return out


def _metrics(t_pred: np.ndarray, t_true: np.ndarray, p_pred: np.ndarray, p_true: np.ndarray) -> Dict[str, float]:
    rmse_t = float(np.sqrt(np.mean((t_pred - t_true) ** 2)))
    mae_t = float(np.mean(np.abs(t_pred - t_true)))
    bias_t = float(np.mean(t_pred - t_true))
    mae_p = float(np.mean(np.abs(p_pred - p_true)))
    return {
        "rmse_temp_c": rmse_t,
        "mae_temp_c": mae_t,
        "bias_temp_c": bias_t,
        "mae_power_w": mae_p,
    }


def calibrate_boptest_v3(
    data_path: str,
    model_path: str,
    output_dir: str = "outputs/surrogate_v2_inverse_boptest",
    policy: str | None = None,
    season: str | None = None,
    limit_rows: int | None = None,
    inject_artifacts: bool = True,
    artifact_spec: ArtifactSpec | None = None,
    max_latency_search: int = 6,
    smooth_window: int = 5,
    target_mode: str = "clean",
    epochs: int = 200,
    batch_size: int = 256,
    patience: int = 30,
    val_split: float = 0.2,
    backbone_lr: float = 1e-4,
    head_lr: float = 5e-3,
    seed: int = 42,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    clean_df = _load_clean_df(data_path, policy, season, limit_rows)
    surrogate = _load_surrogate(model_path, device)

    if artifact_spec is None:
        artifact_spec = ArtifactSpec()

    if inject_artifacts:
        observed_df = inject_boptest_artifacts(clean_df, surrogate, device, artifact_spec, seed)
    else:
        observed_df = clean_df.copy()
        observed_df["t_zone_clean"] = clean_df["t_zone"]
        observed_df["t_zone_next_clean"] = clean_df["t_zone_next"]
        observed_df["p_total_clean"] = clean_df["p_total"]
        observed_df["t_zone_next_surrogate"] = np.nan
        observed_df["p_total_surrogate"] = np.nan

    preprocessed_df, preprocess_summary = preprocess_artifacts(
        observed_df,
        surrogate,
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
    n = len(preprocessed_df)
    train_idx, val_idx = _split_indices(n, val_split)

    with torch.no_grad():
        t_base, p_base = surrogate(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )
        base_np = {
            "t_pred": t_base.cpu().numpy(),
            "p_pred": p_base.cpu().numpy(),
            "t_true": data["t_next_target"].cpu().numpy(),
            "p_true": data["p_target"].cpu().numpy(),
        }
        baseline_metrics = _metrics(base_np["t_pred"], base_np["t_true"], base_np["p_pred"], base_np["p_true"])

    model = JointCalibratedSurrogateV3(
        surrogate=surrogate,
        calibration=NonlinearCalibrationHeadV3(),
    ).to(device)
    criterion = JointCalibrationLossV3()
    optimizer = optim.Adam(model.trainable_groups(backbone_lr=backbone_lr, head_lr=head_lr))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict] = []
    ckpt_path = out_dir / "rc_node_v3_boptest_joint_calibrated.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(train_idx)
        batch_rows = []
        for start in range(0, len(perm), batch_size):
            idx_np = perm[start:start + batch_size]
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            _, _, t_cal, p_cal, temp_res, power_res = model(
                data["t_zone"][idx],
                data["t_amb"][idx],
                data["hour"][idx],
                data["day"][idx],
                data["a0"][idx],
                data["a1"][idx],
            )
            loss, metrics = criterion(
                t_cal,
                data["t_next_target"][idx],
                p_cal,
                data["p_target"][idx],
                temp_res,
                power_res,
                model.calibration,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_rows.append(metrics)

        train_avg = {k: float(np.mean([r[k] for r in batch_rows])) for k in batch_rows[0]}

        model.eval()
        with torch.no_grad():
            idx_val = torch.tensor(val_idx, dtype=torch.long, device=device)
            _, _, t_val, p_val, temp_res_val, power_res_val = model(
                data["t_zone"][idx_val],
                data["t_amb"][idx_val],
                data["hour"][idx_val],
                data["day"][idx_val],
                data["a0"][idx_val],
                data["a1"][idx_val],
            )
            val_loss, val_metrics = criterion(
                t_val,
                data["t_next_target"][idx_val],
                p_val,
                data["p_target"][idx_val],
                temp_res_val,
                power_res_val,
                model.calibration,
            )

        scheduler.step(float(val_loss.item()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_avg["loss_total"],
                "train_rmse_temp": train_avg["rmse_temp"],
                "val_loss": float(val_loss.item()),
                "val_rmse_temp": float(val_metrics["rmse_temp"]),
                "scale_dT": float(model.calibration.scale_dT.item()),
                "bias_T": float(model.calibration.bias_T.item()),
                "scale_P": float(model.calibration.scale_P.item()),
                "bias_P": float(model.calibration.bias_P.item()),
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
            }
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[INV_BOPTEST_V3] epoch={epoch:4d} train_rmse={train_avg['rmse_temp']:.4f} "
                f"val_rmse={val_metrics['rmse_temp']:.4f} scale_dT={model.calibration.scale_dT.item():.4f}"
            )

        if float(val_loss.item()) < best_val - 1e-6:
            best_val = float(val_loss.item())
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "surrogate_state": model.surrogate.state_dict(),
                    "calibration_state": model.calibration.state_dict(),
                    "best_epoch": best_epoch,
                    "backbone_lr": backbone_lr,
                    "head_lr": head_lr,
                    "artifact_spec": asdict(artifact_spec),
                },
                ckpt_path,
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"[INV_BOPTEST_V3] Early stopping at epoch {epoch}")
                break

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.surrogate.load_state_dict(best_ckpt["surrogate_state"])
    model.calibration.load_state_dict(best_ckpt["calibration_state"])
    model.eval()

    with torch.no_grad():
        t_surr, p_surr, t_cal, p_cal, _, _ = model(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )

    pred_df = preprocessed_df.copy()
    pred_df["t_pred_before"] = t_surr.cpu().numpy()
    pred_df["p_pred_before"] = p_surr.cpu().numpy()
    pred_df["t_pred_after"] = t_cal.cpu().numpy()
    pred_df["p_pred_after"] = p_cal.cpu().numpy()
    pred_df["t_target_used"] = data["t_next_target"].cpu().numpy()
    pred_df["p_target_used"] = data["p_target"].cpu().numpy()

    preds_csv = out_dir / "calibration_predictions.csv"
    pred_df.to_csv(preds_csv, index=False)

    before_metrics = _metrics(
        pred_df["t_pred_before"].to_numpy(),
        pred_df["t_target_used"].to_numpy(),
        pred_df["p_pred_before"].to_numpy(),
        pred_df["p_target_used"].to_numpy(),
    )
    after_metrics = _metrics(
        pred_df["t_pred_after"].to_numpy(),
        pred_df["t_target_used"].to_numpy(),
        pred_df["p_pred_after"].to_numpy(),
        pred_df["p_target_used"].to_numpy(),
    )

    c = model.calibration
    scale_dt = float(c.scale_dT.item())
    scale_dt_abs = max(abs(scale_dt), 1e-6)
    czon_est = float(artifact_spec.surrogate_czon_ref_j_per_k / scale_dt_abs)
    czon_err_pct = float(
        abs(czon_est - artifact_spec.c_zon_true_j_per_k) / artifact_spec.c_zon_true_j_per_k * 100.0
    ) if inject_artifacts else None

    history_csv = out_dir / "calibration_history_boptest_v3.csv"
    summary_json = out_dir / "calibration_summary_boptest_v3.json"
    preprocess_json = out_dir / "preprocess_summary.json"
    artifact_json = out_dir / "artifact_spec.json"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    preprocess_json.write_text(json.dumps(preprocess_summary, indent=2), encoding="utf-8")
    artifact_json.write_text(json.dumps(asdict(artifact_spec), indent=2), encoding="utf-8")

    summary = {
        "data_path": data_path,
        "model_path": model_path,
        "target_mode": target_mode,
        "inject_artifacts": inject_artifacts,
        "policy_filter": policy,
        "season_filter": season,
        "rows": int(len(pred_df)),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "baseline_rmse_c": before_metrics["rmse_temp_c"],
        "baseline_mae_c": before_metrics["mae_temp_c"],
        "baseline_bias_c": before_metrics["bias_temp_c"],
        "baseline_power_mae_w": before_metrics["mae_power_w"],
        "calibrated_rmse_c": after_metrics["rmse_temp_c"],
        "calibrated_mae_c": after_metrics["mae_temp_c"],
        "calibrated_bias_c": after_metrics["bias_temp_c"],
        "calibrated_power_mae_w": after_metrics["mae_power_w"],
        "improvement_rmse_pct": float(
            (before_metrics["rmse_temp_c"] - after_metrics["rmse_temp_c"])
            / max(before_metrics["rmse_temp_c"], 1e-6)
            * 100.0
        ),
        "scale_dT": scale_dt,
        "scale_dT_abs": scale_dt_abs,
        "bias_T_c": float(c.bias_T.item()),
        "scale_P": float(c.scale_P.item()),
        "bias_P_w": float(c.bias_P.item()),
        "czon_est_j_per_k": czon_est,
        "czon_error_pct": czon_err_pct,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "preprocess_summary": preprocess_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("INVERSE CALIBRATION ON REAL BOPTEST TRACE (V3)")
    print("=" * 72)
    print(f"Baseline RMSE:   {before_metrics['rmse_temp_c']:.4f} C")
    print(f"Calibrated RMSE: {after_metrics['rmse_temp_c']:.4f} C")
    print(f"Improvement:     {summary['improvement_rmse_pct']:.1f}%")
    print(f"scale_dT:        {scale_dt:.4f}")
    print(f"bias_T:          {summary['bias_T_c']:+.4f} C")
    print(f"C_zon_est:       {czon_est:.3e} J/K")
    if czon_err_pct is not None:
        print(f"C_zon_error:     {czon_err_pct:.2f}%")
    print(f"Observed log:    {observed_csv}")
    print(f"Preprocessed:    {preprocessed_csv}")
    print(f"Predictions:     {preds_csv}")
    print(f"Summary:         {summary_json}")
    return str(summary_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverse calibration of updated surrogate on real BOPTEST traces with artifact preprocessing and joint nonlinear fine-tuning."
    )
    parser.add_argument("--preset", choices=["smoke", "full", "realclean"], default=None)
    parser.add_argument("--data", default="data/surrogate_v2/boptest_v2_tsupply.csv")
    parser.add_argument("--model", default="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--output_dir", default="outputs/surrogate_v2_inverse_boptest")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-3)
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

    calibrate_boptest_v3(
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        val_split=args.val_split,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
