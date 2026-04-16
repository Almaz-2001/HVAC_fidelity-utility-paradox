from __future__ import annotations

import argparse
import json
import os
import sys
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


class CalibrationLayerV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_dT = nn.Parameter(torch.tensor(1.0))
        self.bias_T = nn.Parameter(torch.tensor(0.0))
        self.scale_P = nn.Parameter(torch.tensor(1.0))
        self.bias_P = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        t_curr: torch.Tensor,
        t_surrogate: torch.Tensor,
        p_surrogate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_t = t_surrogate - t_curr
        t_cal = t_curr + self.scale_dT * d_t + self.bias_T
        p_cal = torch.clamp(self.scale_P * p_surrogate + self.bias_P, min=0.0)
        return t_cal, p_cal


class CalibratedSurrogateV2(nn.Module):
    def __init__(self, surrogate: RCNeuralODEv2, finetune: bool = False) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.calibration = CalibrationLayerV2()
        self.finetune = finetune

        for p in self.surrogate.parameters():
            p.requires_grad = finetune

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_surr, p_surr = self.surrogate(t_zone, t_amb, hour, day, a0, a1)
        return self.calibration(t_zone, t_surr, p_surr)

    def trainable_groups(self) -> list[dict]:
        if self.finetune:
            return [
                {"params": self.surrogate.parameters(), "lr": 1e-4},
                {"params": self.calibration.parameters(), "lr": 1e-2},
            ]
        return [{"params": self.calibration.parameters(), "lr": 1e-2}]


class CalibrationLossV2(nn.Module):
    def __init__(self, lambda_temp: float = 1.0, lambda_power: float = 0.05, lambda_reg: float = 0.05) -> None:
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
        calib: CalibrationLayerV2,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        l_temp = torch.mean((t_pred - t_true) ** 2)
        p_scale = torch.clamp(p_true.max().detach(), min=1.0)
        l_power = torch.mean(((p_pred - p_true) / p_scale) ** 2)
        l_reg = (
            (calib.scale_dT - 1.0) ** 2
            + calib.bias_T ** 2
            + (calib.scale_P - 1.0) ** 2
            + calib.bias_P ** 2 / 1000.0
        )
        total = self.lambda_temp * l_temp + self.lambda_power * l_power + self.lambda_reg * l_reg
        return total, {
            "loss_total": float(total.item()),
            "loss_temp": float(l_temp.item()),
            "loss_power": float(l_power.item()),
            "loss_reg": float(l_reg.item()),
            "rmse_temp": float(torch.sqrt(l_temp).item()),
        }


def _load_vectors(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    required = ["t_zone", "t_amb", "hour", "day", "a0_raw", "a1_raw", "t_zone_next", "p_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for inverse calibration: {missing}")
    return {
        "t_zone": torch.tensor(df["t_zone"].values, dtype=torch.float32),
        "t_amb": torch.tensor(df["t_amb"].values, dtype=torch.float32),
        "hour": torch.tensor(df["hour"].values, dtype=torch.float32),
        "day": torch.tensor(df["day"].values, dtype=torch.float32),
        "a0": torch.tensor(df["a0_raw"].values, dtype=torch.float32),
        "a1": torch.tensor(df["a1_raw"].values, dtype=torch.float32),
        "t_next": torch.tensor(df["t_zone_next"].values, dtype=torch.float32),
        "p_total": torch.tensor(df["p_total"].values, dtype=torch.float32),
    }


def _compute_metrics(t_pred: torch.Tensor, t_true: torch.Tensor) -> Dict[str, float]:
    rmse = float(torch.sqrt(torch.mean((t_pred - t_true) ** 2)).item())
    mae = float(torch.mean(torch.abs(t_pred - t_true)).item())
    bias = float(torch.mean(t_pred - t_true).item())
    return {"rmse_temp": rmse, "mae_temp": mae, "bias_temp": bias}


def calibrate_v2(
    data_path: str,
    model_path: str,
    output_dir: str = "outputs/surrogate_v2_inverse",
    epochs: int = 300,
    batch_size: int = 128,
    patience: int = 40,
    finetune: bool = False,
    surrogate_czon_ref: float = 5.3e5,
    synthetic_czon_true: float | None = 4.2e5,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    tensors = _load_vectors(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    surrogate = RCNeuralODEv2(hidden_dim=int(ckpt.get("hidden_dim", 64))).to(device)
    surrogate.load_state_dict(ckpt["model_state"])
    surrogate.eval()

    model = CalibratedSurrogateV2(surrogate, finetune=finetune).to(device)
    criterion = CalibrationLossV2()
    optimizer = optim.Adam(model.trainable_groups())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    data = {k: v.to(device) for k, v in tensors.items()}

    with torch.no_grad():
        t_base, p_base = surrogate(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )
        base_metrics = _compute_metrics(t_base, data["t_next"])
        base_power_mae = float(torch.mean(torch.abs(p_base - data["p_total"])).item())

    print(f"[INV_V2] Baseline RMSE: {base_metrics['rmse_temp']:.4f} C")
    print(f"[INV_V2] Baseline MAE:  {base_metrics['mae_temp']:.4f} C")
    print(f"[INV_V2] Baseline power MAE: {base_power_mae:.2f} W")

    n = len(df)
    best_loss = float("inf")
    patience_cnt = 0
    history: list[dict] = []
    save_path = os.path.join(output_dir, "rc_node_v3_tsupply_calibrated.pt")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        batch_metrics = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            t_pred, p_pred = model(
                data["t_zone"][idx],
                data["t_amb"][idx],
                data["hour"][idx],
                data["day"][idx],
                data["a0"][idx],
                data["a1"][idx],
            )
            loss, metrics = criterion(
                t_pred,
                data["t_next"][idx],
                p_pred,
                data["p_total"][idx],
                model.calibration,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_metrics.append(metrics)

        avg = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
        scheduler.step(avg["loss_total"])
        c = model.calibration
        history.append(
            {
                "epoch": epoch,
                "scale_dT": float(c.scale_dT.item()),
                "bias_T": float(c.bias_T.item()),
                "scale_P": float(c.scale_P.item()),
                "bias_P": float(c.bias_P.item()),
                **avg,
            }
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[INV_V2] epoch={epoch:4d} loss={avg['loss_total']:.4f} "
                f"rmse={avg['rmse_temp']:.4f} scale_dT={c.scale_dT.item():.4f} "
                f"bias_T={c.bias_T.item():+.4f}"
            )

        if avg["loss_total"] < best_loss - 1e-6:
            best_loss = avg["loss_total"]
            patience_cnt = 0
            best_epoch = epoch
            torch.save(
                {
                    "surrogate_state": model.surrogate.state_dict(),
                    "calibration_state": model.calibration.state_dict(),
                    "hidden_dim": int(ckpt.get("hidden_dim", 64)),
                    "epoch": epoch,
                    "finetune": finetune,
                    "rmse_before": base_metrics["rmse_temp"],
                    "surrogate_czon_ref": surrogate_czon_ref,
                    "synthetic_czon_true": synthetic_czon_true,
                },
                save_path,
            )
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[INV_V2] Early stopping at epoch {epoch}")
                break

    best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.surrogate.load_state_dict(best_ckpt["surrogate_state"])
    model.calibration.load_state_dict(best_ckpt["calibration_state"])
    model.eval()
    with torch.no_grad():
        t_cal, p_cal = model(
            data["t_zone"], data["t_amb"], data["hour"], data["day"], data["a0"], data["a1"]
        )
        cal_metrics = _compute_metrics(t_cal, data["t_next"])
        cal_power_mae = float(torch.mean(torch.abs(p_cal - data["p_total"])).item())

    c = model.calibration
    scale_dt = float(c.scale_dT.item())
    scale_dt_abs = max(abs(scale_dt), 1e-6)
    czon_est = float(surrogate_czon_ref / scale_dt_abs)
    czon_err_pct = None
    if synthetic_czon_true is not None:
        czon_err_pct = float(abs(czon_est - synthetic_czon_true) / synthetic_czon_true * 100.0)

    summary = {
        "mode": "finetune" if finetune else "linear",
        "data_path": data_path,
        "model_path": model_path,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "rmse_before_c": base_metrics["rmse_temp"],
        "mae_before_c": base_metrics["mae_temp"],
        "bias_before_c": base_metrics["bias_temp"],
        "power_mae_before_w": base_power_mae,
        "rmse_after_c": cal_metrics["rmse_temp"],
        "mae_after_c": cal_metrics["mae_temp"],
        "bias_after_c": cal_metrics["bias_temp"],
        "power_mae_after_w": cal_power_mae,
        "improvement_rmse_pct": float((base_metrics["rmse_temp"] - cal_metrics["rmse_temp"]) / max(base_metrics["rmse_temp"], 1e-6) * 100.0),
        "scale_dT": scale_dt,
        "scale_dT_abs": scale_dt_abs,
        "bias_T_c": float(c.bias_T.item()),
        "scale_P": float(c.scale_P.item()),
        "bias_P_w": float(c.bias_P.item()),
        "surrogate_czon_ref_j_per_k": surrogate_czon_ref,
        "czon_est_j_per_k": czon_est,
        "synthetic_czon_true_j_per_k": synthetic_czon_true,
        "czon_error_pct": czon_err_pct,
    }

    hist_path = os.path.join(output_dir, "calibration_history_v2.csv")
    summary_path = os.path.join(output_dir, "calibration_summary_v2.json")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("INVERSE CALIBRATION V2 RESULTS")
    print("=" * 60)
    print(f"RMSE before: {summary['rmse_before_c']:.4f} C")
    print(f"RMSE after:  {summary['rmse_after_c']:.4f} C")
    print(f"Improvement: {summary['improvement_rmse_pct']:.1f}%")
    print(f"scale_dT:    {summary['scale_dT']:.4f}")
    print(f"bias_T:      {summary['bias_T_c']:+.4f} C")
    print(f"C_zon_est:   {summary['czon_est_j_per_k']:.3e} J/K")
    if czon_err_pct is not None:
        print(f"C_zon error: {czon_err_pct:.2f}%")
    print(f"Saved:       {save_path}")
    print(f"Summary:     {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse-problem calibration for RCNeuralODEv2 / rc_node_v3_tsupply.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--output_dir", default="outputs/surrogate_v2_inverse")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--surrogate-czon-ref", type=float, default=5.3e5)
    parser.add_argument("--synthetic-czon-true", type=float, default=4.2e5)
    args = parser.parse_args()

    calibrate_v2(
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        finetune=args.finetune,
        surrogate_czon_ref=args.surrogate_czon_ref,
        synthetic_czon_true=args.synthetic_czon_true,
    )


if __name__ == "__main__":
    main()
