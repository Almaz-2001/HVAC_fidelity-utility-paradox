"""
surrogate/inverse_problem.py

Фаза 2: Inverse Problem — калибровка surrogate под реальное здание.

Два режима:
  1. linear  (default): обучаем только CalibrationLayer (4 параметра)
  2. finetune:          размораживаем нейросеть + CalibrationLayer

Задача:
    Дано:   измерения реального здания [T_real(t), P_real(t), a(t)]
    Найти:  параметры θ* такие что surrogate(θ*) минимизирует
            ||T_surrogate(θ) - T_real||²

Запуск:
  python surrogate/inverse_problem.py \
    --data /app/data/surrogate/synthetic_real_500_noise_bias.csv \
    --model /app/outputs/surrogate/rc_node_best.pt \
    --finetune

  python surrogate/inverse_problem.py \
    --data /app/data/surrogate/synthetic_real_500_noise_bias.csv \
    --model /app/outputs/surrogate/rc_node_best.pt
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple

from surrogate.rc_node import RCNeuralODE


# -----------------------------------------------------------------------
# CalibrationLayer — линейный адаптационный слой
# -----------------------------------------------------------------------

class CalibrationLayer(nn.Module):
    """
    Тонкий адаптационный слой поверх surrogate.

    T_calib = T_curr + scale_dT * (T_surr - T_curr) + bias_T
    P_calib = scale_P * P_surr + bias_P
    """
    def __init__(self):
        super().__init__()
        self.scale_dT = nn.Parameter(torch.tensor(1.0))
        self.bias_T   = nn.Parameter(torch.tensor(0.0))
        self.scale_P  = nn.Parameter(torch.tensor(1.0))
        self.bias_P   = nn.Parameter(torch.tensor(0.0))

    def forward(self, t_curr, t_surrogate, p_surrogate):
        dT      = t_surrogate - t_curr
        t_calib = t_curr + self.scale_dT * dT + self.bias_T
        p_calib = torch.clamp(self.scale_P * p_surrogate + self.bias_P, min=0.0)
        return t_calib, p_calib

    def summary(self):
        print(f"    scale_dT = {self.scale_dT.item():.4f}  (1.0=no change)")
        print(f"    bias_T   = {self.bias_T.item():.4f} C")
        print(f"    scale_P  = {self.scale_P.item():.4f}")
        print(f"    bias_P   = {self.bias_P.item():.2f} W")


# -----------------------------------------------------------------------
# CalibratedSurrogate
# -----------------------------------------------------------------------

class CalibratedSurrogate(nn.Module):
    def __init__(self, surrogate: RCNeuralODE, finetune: bool = False):
        super().__init__()
        self.surrogate   = surrogate
        self.calibration = CalibrationLayer()
        self.finetune    = finetune

        if finetune:
            # Размораживаем нейросеть surrogate
            for param in self.surrogate.parameters():
                param.requires_grad = True
            n_surr = sum(p.numel() for p in self.surrogate.parameters())
            print(f"[CALIB] Finetune mode: unfrozen {n_surr} surrogate params")
        else:
            # Замораживаем — обучаем только CalibrationLayer
            for param in self.surrogate.parameters():
                param.requires_grad = False
            print(f"[CALIB] Linear mode: surrogate frozen, 4 calib params only")

    def forward(self, t_zone, a0, a1):
        t_surr, p_surr = self.surrogate(t_zone, a0, a1)
        t_cal, p_cal   = self.calibration(t_zone, t_surr, p_surr)
        return t_cal, p_cal

    def trainable_params(self):
        """Возвращает две группы параметров с разными lr."""
        if self.finetune:
            return [
                # Нейросеть — маленький lr чтобы не забыть что выучили
                {"params": self.surrogate.parameters(), "lr": 1e-4},
                # Калибровочный слой — большой lr для быстрой адаптации
                {"params": self.calibration.parameters(), "lr": 1e-2},
            ]
        else:
            return [{"params": self.calibration.parameters(), "lr": 1e-2}]


# -----------------------------------------------------------------------
# Функция потерь
# -----------------------------------------------------------------------

class CalibrationLoss(nn.Module):
    """
    L = lambda_mse * MSE(T_pred, T_true)
      + lambda_reg * ||calib_params - identity||²

    Регуляризация удерживает калибровочный слой близко к
    единичному преобразованию — не даёт уйти в вырожденное решение.
    """
    def __init__(self, lambda_mse=1.0, lambda_reg=0.05):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_reg = lambda_reg

    def forward(self, t_pred, t_true, calib: CalibrationLayer):
        l_mse = torch.mean((t_pred - t_true) ** 2)
        l_reg = (
            (calib.scale_dT - 1.0) ** 2 +
             calib.bias_T ** 2 +
            (calib.scale_P - 1.0) ** 2 +
             calib.bias_P ** 2 / 1000.0
        )
        total = self.lambda_mse * l_mse + self.lambda_reg * l_reg
        return total, {
            "loss_total": total.item(),
            "loss_mse":   l_mse.item(),
            "loss_reg":   l_reg.item(),
            "rmse":       float(l_mse.item() ** 0.5),
        }


# -----------------------------------------------------------------------
# Основная функция калибровки
# -----------------------------------------------------------------------

def calibrate(
    data_path:  str,
    model_path: str,
    output_dir: str   = "/app/outputs/surrogate",
    epochs:     int   = 300,
    batch_size: int   = 64,
    finetune:   bool  = False,
    patience:   int   = 40,
) -> str:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mode_str = "FINETUNE" if finetune else "LINEAR"

    # Загружаем данные
    df = pd.read_csv(data_path)
    print(f"[CALIB] Mode: {mode_str}")
    print(f"[CALIB] Loaded {len(df)} samples from {data_path}")

    t_zone = torch.tensor(df["t_zone"].values,     dtype=torch.float32)
    a0     = torch.tensor(df["a0_raw"].values,      dtype=torch.float32)
    a1     = torch.tensor(df["a1_raw"].values,      dtype=torch.float32)
    t_next = torch.tensor(df["t_zone_next"].values, dtype=torch.float32)

    # Загружаем surrogate
    ckpt = torch.load(model_path, map_location="cpu")
    surrogate = RCNeuralODE(
        hidden_dim=ckpt.get("hidden_dim", 64),
        n_layers=ckpt.get("n_layers", 3),
    )
    surrogate.load_state_dict(ckpt["model_state"])
    surrogate.update_normalization(ckpt["t_mean"], ckpt["t_std"])

    # Baseline RMSE до калибровки
    surrogate.eval()
    with torch.no_grad():
        t_base, _ = surrogate(t_zone, a0, a1)
        rmse_before = float(torch.mean((t_base - t_next) ** 2) ** 0.5)
    print(f"[CALIB] RMSE before: {rmse_before:.4f} C")

    # Создаём модель
    model     = CalibratedSurrogate(surrogate, finetune=finetune)
    criterion = CalibrationLoss(lambda_mse=1.0, lambda_reg=0.05)
    optimizer = optim.Adam(model.trainable_params())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=False
    )

    N = len(t_zone)
    print(f"\n[CALIB] Training: epochs={epochs}, batch={batch_size}")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'RMSE':>8} | "
          f"{'scale_dT':>10} | {'bias_T':>8}")
    print("-" * 55)

    history      = []
    best_loss    = float("inf")
    patience_cnt = 0
    save_path    = os.path.join(output_dir, "rc_node_calibrated.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N)
        ep_metrics = []

        for i in range(0, N, batch_size):
            b = idx[i:i + batch_size]
            t_pred, _ = model(t_zone[b], a0[b], a1[b])
            loss, m   = criterion(t_pred, t_next[b], model.calibration)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for pg in model.trainable_params() for p in pg["params"]],
                max_norm=1.0
            )
            optimizer.step()
            ep_metrics.append(m)

        avg = {k: float(np.mean([m[k] for m in ep_metrics]))
               for k in ep_metrics[0]}
        scheduler.step(avg["loss_total"])

        c = model.calibration
        history.append({"epoch": epoch,
                         "scale_dT": c.scale_dT.item(),
                         "bias_T":   c.bias_T.item(),
                         **avg})

        if epoch % 30 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg['loss_total']:>10.4f} | "
                  f"{avg['rmse']:>8.4f} | "
                  f"{c.scale_dT.item():>10.4f} | "
                  f"{c.bias_T.item():>8.4f}")

        if avg["loss_total"] < best_loss - 1e-5:
            best_loss    = avg["loss_total"]
            patience_cnt = 0
            torch.save({
                "surrogate_state":   model.surrogate.state_dict(),
                "calibration_state": model.calibration.state_dict(),
                "t_mean":  ckpt["t_mean"],
                "t_std":   ckpt["t_std"],
                "hidden_dim": ckpt.get("hidden_dim", 64),
                "n_layers":   ckpt.get("n_layers", 3),
                "finetune": finetune,
                "epoch":    epoch,
                "rmse_before": rmse_before,
            }, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[CALIB] Early stopping at epoch {epoch}")
                break

    # Финальная оценка
    model.eval()
    with torch.no_grad():
        t_cal, _ = model(t_zone, a0, a1)
        rmse_after = float(torch.mean((t_cal - t_next) ** 2) ** 0.5)

    c = model.calibration
    print(f"\n{'='*55}")
    print(f"CALIBRATION RESULTS  [{mode_str}]")
    print(f"{'='*55}")
    print(f"  RMSE before: {rmse_before:.4f} C")
    print(f"  RMSE after:  {rmse_after:.4f} C")
    print(f"  Improvement: {(rmse_before-rmse_after)/rmse_before*100:.1f}%")
    print(f"\n  Calibration parameters:")
    c.summary()

    s = c.scale_dT.item()
    c_zon_est = 5.3e5 / max(s, 1e-6)
    print(f"\n  Physical interpretation:")
    print(f"    scale_dT = {s:.4f}")
    print(f"    Estimated C_zon_real ≈ {c_zon_est:.3e} J/K")
    print(f"    Ground truth C_zon   = 4.200e+05 J/K")
    print(f"    bias_T recovered     = {c.bias_T.item():.4f} C  "
          f"(true sensor bias = +0.500 C)")

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "calibration_history.csv"), index=False
    )
    print(f"\n  Saved: {save_path}")
    return save_path


# -----------------------------------------------------------------------
# Точка входа
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--model",      default="/app/outputs/surrogate/rc_node_best.pt")
    parser.add_argument("--output_dir", default="/app/outputs/surrogate")
    parser.add_argument("--epochs",     type=int,  default=300)
    parser.add_argument("--batch_size", type=int,  default=64)
    parser.add_argument("--patience",   type=int,  default=40)
    parser.add_argument("--finetune",   action="store_true",
                        help="Unfreeze surrogate neural net weights")
    args = parser.parse_args()

    calibrate(
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        finetune=args.finetune,
        patience=args.patience,
    )

if __name__ == "__main__":
    main()