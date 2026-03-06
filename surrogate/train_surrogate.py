"""
surrogate/train_surrogate.py

Обучение RC Neural ODE surrogate на данных из BOPTEST.

Запуск:
  python surrogate/train_surrogate.py --data /app/data/surrogate/boptest_mixed_10000.csv
  python surrogate/train_surrogate.py --data /app/data/surrogate/boptest_random_5000.csv --epochs 200
"""

from __future__ import annotations

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Dict

from surrogate.rc_node import RCNeuralODE, SurrogateLoss


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class BOPTESTDataset(Dataset):
    """
    Датасет одношаговых переходов:
        (t_zone, a0, a1) → (t_zone_next, p_total)
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self._validate(df)

        # Входы
        self.t_zone = torch.tensor(df["t_zone"].values,      dtype=torch.float32)
        self.a0     = torch.tensor(df["a0_raw"].values,      dtype=torch.float32)
        self.a1     = torch.tensor(df["a1_raw"].values,      dtype=torch.float32)

        # Цели
        self.t_next = torch.tensor(df["t_zone_next"].values, dtype=torch.float32)
        self.p_total = torch.tensor(df["p_total"].values,    dtype=torch.float32)

        # Статистика для нормализации модели
        self.t_mean = float(self.t_zone.mean())
        self.t_std  = float(self.t_zone.std())

        print(f"[DATASET] Loaded {len(self)} samples from {csv_path}")
        print(f"  T_zone: {self.t_zone.min():.1f} — {self.t_zone.max():.1f} °C "
              f"(mean={self.t_mean:.1f}, std={self.t_std:.1f})")
        print(f"  P_total: {self.p_total.min():.0f} — "
              f"{self.p_total.max():.0f} W")

    def _validate(self, df: pd.DataFrame) -> None:
        required = ["t_zone", "a0_raw", "a1_raw", "t_zone_next", "p_total"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.t_zone)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.t_zone[idx],
            self.a0[idx],
            self.a1[idx],
            self.t_next[idx],
            self.p_total[idx],
        )


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_metrics(
    t_pred: torch.Tensor,
    t_true: torch.Tensor,
    p_pred: torch.Tensor,
    p_true: torch.Tensor,
    p_max:  float = 5500.0,
) -> Dict[str, float]:
    with torch.no_grad():
        mse_t  = torch.mean((t_pred - t_true) ** 2).item()
        mae_t  = torch.mean((t_pred - t_true).abs()).item()
        rmse_t = mse_t ** 0.5

        mse_p  = torch.mean(((p_pred - p_true) / p_max) ** 2).item()
        mae_p  = torch.mean((p_pred - p_true).abs()).item()

        # R² для температуры
        ss_res = torch.sum((t_true - t_pred) ** 2).item()
        ss_tot = torch.sum((t_true - t_true.mean()) ** 2).item()
        r2_t   = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        "rmse_temp":  rmse_t,
        "mae_temp":   mae_t,
        "mse_power":  mse_p,
        "mae_power":  mae_p,
        "r2_temp":    r2_t,
    }


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train(
    data_path:   str,
    output_dir:  str   = "/app/outputs/surrogate",
    epochs:      int   = 300,
    batch_size:  int   = 256,
    lr:          float = 1e-3,
    hidden_dim:  int   = 64,
    n_layers:    int   = 3,
    val_split:   float = 0.2,
    patience:    int   = 30,   # early stopping
    device_str:  str   = "auto",
) -> str:
    """
    Обучает RC Neural ODE.

    Returns:
        Путь к сохранённой модели.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[TRAIN] Device: {device}")

    # Dataset
    dataset = BOPTESTDataset(data_path)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print(f"[TRAIN] Train: {n_train} | Val: {n_val} | Batch: {batch_size}")

    # Model
    model = RCNeuralODE(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.update_normalization(dataset.t_mean, dataset.t_std)
    model.summary()

    # Loss + Optimizer
    criterion = SurrogateLoss(
        lambda_temp=1.0, lambda_power=0.1, lambda_physics=0.01
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, verbose=False
    )

    # Training
    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    print(f"\n[TRAIN] Starting training: epochs={epochs}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | "
          f"{'RMSE T':>8} | {'MAE T':>7} | {'R²':>6} | {'C_zon':>12}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            t_zone, a0, a1, t_next, p_true = [b.to(device) for b in batch]

            optimizer.zero_grad()
            t_pred, p_pred = model(t_zone, a0, a1)
            loss, _ = criterion(t_pred, t_next, p_pred, p_true)
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        all_t_pred, all_t_true, all_p_pred, all_p_true = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                t_zone, a0, a1, t_next, p_true = [b.to(device) for b in batch]
                t_pred, p_pred = model(t_zone, a0, a1)
                loss, _ = criterion(t_pred, t_next, p_pred, p_true)
                val_losses.append(loss.item())
                all_t_pred.append(t_pred)
                all_t_true.append(t_next)
                all_p_pred.append(p_pred)
                all_p_true.append(p_true)

        val_loss  = np.mean(val_losses)
        metrics   = compute_metrics(
            torch.cat(all_t_pred), torch.cat(all_t_true),
            torch.cat(all_p_pred), torch.cat(all_p_true),
        )

        scheduler.step(val_loss)

        # Логируем
        row = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "c_zon":      model.c_zon.item(),
            **metrics,
        }
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>10.6f} | "
                  f"{metrics['rmse_temp']:>8.4f} | "
                  f"{metrics['mae_temp']:>7.4f} | "
                  f"{metrics['r2_temp']:>6.3f} | "
                  f"{model.c_zon.item():>12.3e}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_cnt  = 0
            # Сохраняем лучшую модель
            model_path = os.path.join(output_dir, "rc_node_best.pt")
            torch.save({
                "model_state":  model.state_dict(),
                "t_mean":       dataset.t_mean,
                "t_std":        dataset.t_std,
                "hidden_dim":   hidden_dim,
                "n_layers":     n_layers,
                "epoch":        epoch,
                "val_loss":     val_loss,
                "metrics":      metrics,
            }, model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Сохраняем историю
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(output_dir, "train_history.csv")
    hist_df.to_csv(hist_path, index=False)

    print(f"\n[TRAIN] Best val_loss = {best_val_loss:.6f}")
    print(f"[TRAIN] Model saved:   {model_path}")
    print(f"[TRAIN] History saved: {hist_path}")

    return model_path


# -----------------------------------------------------------------------
# Benchmark: скорость surrogate vs BOPTEST
# -----------------------------------------------------------------------

def benchmark_speed(model_path: str, n_steps: int = 10_000) -> None:
    """Сравнивает скорость surrogate и BOPTEST."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = RCNeuralODE(
        hidden_dim=checkpoint["hidden_dim"],
        n_layers=checkpoint["n_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.update_normalization(checkpoint["t_mean"], checkpoint["t_std"])
    model.eval()

    # Батч из n_steps случайных состояний
    t_zone = torch.FloatTensor(n_steps).uniform_(15, 35)
    a0     = torch.FloatTensor(n_steps).uniform_(-1, 1)
    a1     = torch.FloatTensor(n_steps).uniform_(-1, 1)

    # Прогрев
    with torch.no_grad():
        for _ in range(100):
            model(t_zone[:64], a0[:64], a1[:64])

    # Замер
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(t_zone, a0, a1)
    dt = time.perf_counter() - t0

    steps_per_sec = n_steps / dt
    print(f"\n[BENCH] Surrogate speed: {steps_per_sec:,.0f} steps/sec")
    print(f"[BENCH] BOPTEST speed:   ~1 step/sec (HTTP)")
    print(f"[BENCH] Speedup:         ~{steps_per_sec:,.0f}x")


# -----------------------------------------------------------------------
# Точка входа
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True, help="Path to CSV dataset")
    parser.add_argument("--output_dir", default="/app/outputs/surrogate")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int,   default=64)
    parser.add_argument("--n_layers",   type=int,   default=3)
    parser.add_argument("--patience",   type=int,   default=30)
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run speed benchmark after training")
    args = parser.parse_args()

    model_path = train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        patience=args.patience,
    )

    if args.benchmark:
        benchmark_speed(model_path)


if __name__ == "__main__":
    main()