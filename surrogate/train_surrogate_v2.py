"""

"""

from __future__ import annotations

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional

from surrogate.rc_node_v2 import RCNeuralODEv2, SurrogateLossV2




class BOPTESTDatasetV2(Dataset):
    def __init__(self, csv_path: str, max_horizon: int = 4):
        df = pd.read_csv(csv_path)
        self._validate(df)
        self.max_horizon = max_horizon

        self.t_zone = torch.tensor(df["t_zone"].values, dtype=torch.float32)
        self.t_amb  = torch.tensor(df["t_amb"].values,  dtype=torch.float32)
        self.hour   = torch.tensor(df["hour"].values,   dtype=torch.float32)
        self.day    = torch.tensor(df["day"].values,     dtype=torch.float32)
        self.a0     = torch.tensor(df["a0_raw"].values,  dtype=torch.float32)
        self.a1     = torch.tensor(df["a1_raw"].values,  dtype=torch.float32)
        self.t_next  = torch.tensor(df["t_zone_next"].values, dtype=torch.float32)
        self.p_total = torch.tensor(df["p_total"].values, dtype=torch.float32)

        N = len(df)
        self.valid_indices = list(range(N - max_horizon))

        print(f"[DATASET_V2] Loaded {N} rows from {csv_path}")
        print(f"  T_zone: [{self.t_zone.min():.1f}, {self.t_zone.max():.1f}] °C")
        print(f"  T_amb:  [{self.t_amb.min():.1f}, {self.t_amb.max():.1f}] °C")
        print(f"  P_total: [{self.p_total.min():.0f}, {self.p_total.max():.0f}] W")
        print(f"  Valid sequences (horizon={max_horizon}): {len(self.valid_indices)}")

    def _validate(self, df: pd.DataFrame) -> None:
        required = ["t_zone", "t_amb", "hour", "day",
                     "a0_raw", "a1_raw", "t_zone_next", "p_total"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.valid_indices[idx]
        h = self.max_horizon
        return {
            't_zone': self.t_zone[i], 't_amb': self.t_amb[i],
            'hour': self.hour[i], 'day': self.day[i],
            'a0': self.a0[i], 'a1': self.a1[i],
            't_next': self.t_next[i], 'p_total': self.p_total[i],
            'seq_t_zone': self.t_zone[i:i + h + 1],
            'seq_t_amb': self.t_amb[i:i + h],
            'seq_hour': self.hour[i:i + h],
            'seq_day': self.day[i:i + h],
            'seq_a0': self.a0[i:i + h],
            'seq_a1': self.a1[i:i + h],
        }




def compute_metrics(
    t_pred: torch.Tensor, t_true: torch.Tensor,
    p_pred: torch.Tensor, p_true: torch.Tensor,
    p_max: float = 5500.0,
) -> Dict[str, float]:
    with torch.no_grad():
        mse_t = torch.mean((t_pred - t_true) ** 2).item()
        rmse_t = mse_t ** 0.5
        mae_t = torch.mean((t_pred - t_true).abs()).item()
        mse_p = torch.mean(((p_pred - p_true) / p_max) ** 2).item()
        ss_res = torch.sum((t_true - t_pred) ** 2).item()
        ss_tot = torch.sum((t_true - t_true.mean()) ** 2).item()
        r2_t = 1.0 - ss_res / (ss_tot + 1e-8)
    return {"rmse_temp": rmse_t, "mae_temp": mae_t, "mse_power": mse_p, "r2_temp": r2_t}




def train_v2(
    data_path: str,
    output_dir: str = "/app/outputs/surrogate_v2",
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    val_split: float = 0.2,
    patience: int = 30,
    multi_horizons: list = None,
    device_str: str = "auto",
) -> str:
    if multi_horizons is None:
        multi_horizons = [2, 4]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    max_horizon = max(multi_horizons)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[TRAIN_V2] Device: {device}")

    full_dataset = BOPTESTDatasetV2(data_path, max_horizon=max_horizon)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset = torch.utils.data.Subset(full_dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, n_total))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[TRAIN_V2] Train: {n_train} | Val: {n_val}")
    print(f"[TRAIN_V2] Multi-step horizons: {multi_horizons}")

    model = RCNeuralODEv2(hidden_dim=hidden_dim).to(device)
    model.summary()

    criterion = SurrogateLossV2(
        lambda_temp=1.0, lambda_power=0.1, lambda_multi=0.5,
        lambda_physics=0.05, multi_horizons=multi_horizons,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_metrics = {}
    history = []
    model_path = os.path.join(output_dir, "rc_node_v2_best.pt")

    print(f"\n[TRAIN_V2] Starting training: epochs={epochs}, batch={batch_size}")
    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | "
          f"{'RMSE':>7} | {'R²':>6} | {'L_multi':>8} | {'LR':>10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            t_pred, p_pred = model(
                batch['t_zone'], batch['t_amb'], batch['hour'],
                batch['day'], batch['a0'], batch['a1']
            )
            multi_loss = criterion.compute_multi_step_loss(
                model, batch['seq_t_zone'], batch['seq_t_amb'],
                batch['seq_hour'], batch['seq_day'],
                batch['seq_a0'], batch['seq_a1'],
            )
            loss, loss_dict = criterion(
                t_pred, batch['t_next'], p_pred, batch['p_total'],
                multi_loss=multi_loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()
        train_loss = np.mean([l['loss_total'] for l in train_losses])

        model.eval()
        val_losses = []
        all_t_pred, all_t_true = [], []
        all_p_pred, all_p_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                t_pred, p_pred = model(
                    batch['t_zone'], batch['t_amb'], batch['hour'],
                    batch['day'], batch['a0'], batch['a1']
                )
                multi_loss = criterion.compute_multi_step_loss(
                    model, batch['seq_t_zone'], batch['seq_t_amb'],
                    batch['seq_hour'], batch['seq_day'],
                    batch['seq_a0'], batch['seq_a1'],
                )
                loss, loss_dict = criterion(
                    t_pred, batch['t_next'], p_pred, batch['p_total'],
                    multi_loss=multi_loss,
                )
                val_losses.append(loss_dict)
                all_t_pred.append(t_pred.cpu())
                all_t_true.append(batch['t_next'].cpu())
                all_p_pred.append(p_pred.cpu())
                all_p_true.append(batch['p_total'].cpu())

        val_loss = np.mean([l['loss_total'] for l in val_losses])
        val_multi = np.mean([l['loss_multi'] for l in val_losses])
        metrics = compute_metrics(
            torch.cat(all_t_pred), torch.cat(all_t_true),
            torch.cat(all_p_pred), torch.cat(all_p_true),
        )

        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": val_loss, "val_multi": val_multi, **metrics}
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} | {train_loss:>10.5f} | {val_loss:>10.5f} | "
                  f"{metrics['rmse_temp']:>7.4f} | {metrics['r2_temp']:>6.3f} | "
                  f"{val_multi:>8.4f} | {lr_now:>10.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_metrics = metrics.copy()
            patience_cnt = 0
            torch.save({
                "model_state": model.state_dict(),
                "hidden_dim": hidden_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "metrics": metrics,
                "multi_horizons": multi_horizons,
            }, model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[TRAIN_V2] Early stopping at epoch {epoch}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(output_dir, "train_history_v2.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val_loss:  {best_val_loss:.6f}")
    print(f"  RMSE (1-step):  {best_metrics['rmse_temp']:.4f}°C "
          f"(Phase 1: 1.128°C, Target: <0.5°C)")
    print(f"  R²:             {best_metrics['r2_temp']:.4f} "
          f"(Phase 1: 0.884, Target: >0.90)")
    print(f"  Model saved:    {model_path}")

    return model_path




def validate_safety(
    model_path: str,
    data_path: str,
    horizons: list = None,
    t_low: float = 21.0,
    t_high: float = 25.0,
) -> Dict[str, float]:
    if horizons is None:
        horizons = [1, 2, 4, 6]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = RCNeuralODEv2(hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(data_path)
    N = len(df)

    print(f"\n{'='*60}")
    print(f"SAFETY FILTER VALIDATION")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_path} ({N} rows)")
    print(f"  Comfort: [{t_low}, {t_high}] °C")

    results = {}

    for h in horizons:
        errors = []
        false_safe = 0
        false_unsafe = 0
        total = 0

        for i in range(0, N - h, h):
            t_curr = torch.tensor([df['t_zone'].iloc[i]], dtype=torch.float32)

            for step in range(h):
                idx = i + step
                if idx >= N:
                    break

                t_amb = torch.tensor([df['t_amb'].iloc[idx]], dtype=torch.float32)
                hour  = torch.tensor([df['hour'].iloc[idx]], dtype=torch.float32)
                day   = torch.tensor([df['day'].iloc[idx]], dtype=torch.float32)
                a0    = torch.tensor([df['a0_raw'].iloc[idx]], dtype=torch.float32)
                a1    = torch.tensor([df['a1_raw'].iloc[idx]], dtype=torch.float32)

                with torch.no_grad():
                    t_curr, _ = model(t_curr, t_amb, hour, day, a0, a1)

            t_pred_h = t_curr.item()
            t_true_h = df['t_zone'].iloc[min(i + h, N - 1)]

            errors.append(t_pred_h - t_true_h)

            pred_safe = t_low <= t_pred_h <= t_high
            true_safe = t_low <= t_true_h <= t_high

            if pred_safe and not true_safe:
                false_safe += 1
            if not pred_safe and true_safe:
                false_unsafe += 1
            total += 1

        errs = np.array(errors)
        rmse = np.sqrt(np.mean(errs ** 2))
        bias = np.mean(errs)
        fs_rate = false_safe / total * 100
        fu_rate = false_unsafe / total * 100
        margin_95 = np.percentile(np.abs(errs), 95)
        margin_99 = np.percentile(np.abs(errs), 99)

        results[f'h{h}_rmse'] = rmse
        results[f'h{h}_false_safe_pct'] = fs_rate
        results[f'h{h}_margin_95'] = margin_95

        status = "OK" if fs_rate < 2.0 else "NEED MARGIN"
        print(f"\n  Horizon = {h} steps ({h}h):")
        print(f"    Rollout RMSE:        {rmse:.3f} C")
        print(f"    Bias:                {bias:+.3f} C")
        print(f"    False-safe rate:     {fs_rate:.1f}%  {status}")
        print(f"    False-unsafe rate:   {fu_rate:.1f}%")
        print(f"    Safety margin (95%): {margin_95:.2f} C")
        print(f"    Safety margin (99%): {margin_99:.2f} C")

    h4_fs = results.get('h4_false_safe_pct', 999)
    h4_margin = results.get('h4_margin_95', 999)

    print(f"\n{'='*60}")
    if h4_fs < 2.0:
        print(f"VERDICT: Model SUITABLE for safety filter without extra margin")
    elif h4_fs < 5.0:
        print(f"VERDICT: Model suitable with safety margin = {h4_margin:.2f} C")
        print(f"  -> comfort band for filter: [{t_low + h4_margin:.1f}, "
              f"{t_high - h4_margin:.1f}] C")
    else:
        print(f"VERDICT: Model NOT SUITABLE. Need to improve RMSE.")
    print(f"{'='*60}")

    return results




def benchmark_speed_v2(model_path: str, n_steps: int = 100_000) -> None:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = RCNeuralODEv2(hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    t_zone = torch.FloatTensor(n_steps).uniform_(15, 35)
    t_amb = torch.FloatTensor(n_steps).uniform_(-5, 30)
    hour = torch.FloatTensor(n_steps).uniform_(0, 24)
    day = torch.FloatTensor(n_steps).uniform_(0, 365)
    a0 = torch.FloatTensor(n_steps).uniform_(-1, 1)
    a1 = torch.FloatTensor(n_steps).uniform_(-1, 1)

    with torch.no_grad():
        for _ in range(10):
            model(t_zone[:64], t_amb[:64], hour[:64], day[:64], a0[:64], a1[:64])

    t0 = time.perf_counter()
    with torch.no_grad():
        model(t_zone, t_amb, hour, day, a0, a1)
    dt = time.perf_counter() - t0

    sps = n_steps / dt
    print(f"\n[BENCH] SurrogateV2 speed: {sps:,.0f} steps/sec")
    print(f"[BENCH] Speedup vs BOPTEST (~1 step/s): ~{sps:,.0f}x")




def main():
    parser = argparse.ArgumentParser(description="Train RC Neural ODE v2")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="/app/outputs/surrogate_v2")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--multi_horizons", nargs='+', type=int, default=[2, 4])
    parser.add_argument("--validate_safety", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    model_path = train_v2(
        data_path=args.data, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, hidden_dim=args.hidden_dim,
        patience=args.patience, multi_horizons=args.multi_horizons,
    )

    if args.validate_safety:
        validate_safety(model_path, args.data)

    if args.benchmark:
        benchmark_speed_v2(model_path)


if __name__ == "__main__":
    main()