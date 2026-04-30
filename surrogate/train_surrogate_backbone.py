"""Train the surrogate backbone with optional rollout-aware checkpoint selection."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from surrogate.rc_node_v2 import RCNeuralODEv2, SurrogateLossV2




class BOPTESTBackboneDataset(Dataset):
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
        self.episode_id = df["episode_id"].astype(str).tolist() if "episode_id" in df.columns else None

        N = len(df)
        self.valid_indices = self._build_valid_indices(N)
        dropped = max(0, (N - max_horizon) - len(self.valid_indices))

        print(f"[DATASET_V2] Loaded {N} rows from {csv_path}")
        print(f"  T_zone: [{self.t_zone.min():.1f}, {self.t_zone.max():.1f}] °C")
        print(f"  T_amb:  [{self.t_amb.min():.1f}, {self.t_amb.max():.1f}] °C")
        print(f"  P_total: [{self.p_total.min():.0f}, {self.p_total.max():.0f}] W")
        print(f"  Valid sequences (horizon={max_horizon}): {len(self.valid_indices)}")
        if self.episode_id is not None:
            print(f"  Episode-aware dropped sequences: {dropped}")

    def _validate(self, df: pd.DataFrame) -> None:
        required = ["t_zone", "t_amb", "hour", "day",
                     "a0_raw", "a1_raw", "t_zone_next", "p_total"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _build_valid_indices(self, n_rows: int) -> list[int]:
        if n_rows <= self.max_horizon:
            return []
        if self.episode_id is None:
            return list(range(n_rows - self.max_horizon))

        valid = []
        for i in range(n_rows - self.max_horizon):
            if len(set(self.episode_id[i:i + self.max_horizon + 1])) == 1:
                valid.append(i)
        return valid

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


def _sample_subset_positions(positions: Sequence[int], max_windows: int) -> list[int]:
    if len(positions) <= max_windows:
        return list(positions)
    picks = np.linspace(0, len(positions) - 1, num=max_windows)
    sampled = sorted({int(round(float(idx))) for idx in picks})
    return [int(positions[idx]) for idx in sampled]


def _normalize_rollout_weights(horizons: Sequence[int], weights: Optional[Sequence[float]]) -> dict[int, float]:
    horizon_list = [int(h) for h in horizons]
    if not horizon_list:
        return {}
    if weights is None:
        uniform = 1.0 / float(len(horizon_list))
        return {h: uniform for h in horizon_list}
    if len(weights) != len(horizon_list):
        raise ValueError("rollout_selection_weights must match rollout_val_horizons length")

    clean_weights = [max(float(w), 0.0) for w in weights]
    total = float(sum(clean_weights))
    if total <= 0.0:
        raise ValueError("rollout_selection_weights must have positive sum")
    return {h: w / total for h, w in zip(horizon_list, clean_weights)}


def evaluate_rollout_rmse(
    model: RCNeuralODEv2,
    dataset: BOPTESTBackboneDataset,
    subset_positions: Sequence[int],
    horizons: Sequence[int],
    device: torch.device,
    max_windows: int = 256,
    selection_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons or not subset_positions:
        return {}
    horizon_weights = _normalize_rollout_weights(horizons, selection_weights)

    max_horizon = max(horizons)
    selected_positions = _sample_subset_positions(subset_positions, max_windows=max_windows)
    sq_errors: dict[int, list[float]] = {h: [] for h in horizons}

    model.eval()
    with torch.no_grad():
        for pos in selected_positions:
            sample = dataset[int(pos)]
            t_curr = sample["seq_t_zone"][0].view(1).to(device)

            for step in range(max_horizon):
                t_amb = sample["seq_t_amb"][step].view(1).to(device)
                hour = sample["seq_hour"][step].view(1).to(device)
                day = sample["seq_day"][step].view(1).to(device)
                a0 = sample["seq_a0"][step].view(1).to(device)
                a1 = sample["seq_a1"][step].view(1).to(device)

                t_curr, _ = model(t_curr, t_amb, hour, day, a0, a1)
                horizon = step + 1
                if horizon in sq_errors:
                    t_true = float(sample["seq_t_zone"][horizon].item())
                    sq_errors[horizon].append((float(t_curr.item()) - t_true) ** 2)

    metrics: Dict[str, float] = {"val_rollout_windows": float(len(selected_positions))}
    horizon_rmses: list[float] = []
    weighted_rollout_rmse = 0.0
    for horizon in horizons:
        errs = sq_errors[horizon]
        if not errs:
            continue
        rmse = float(np.sqrt(np.mean(errs)))
        metrics[f"val_rollout_h{horizon}_rmse"] = rmse
        horizon_rmses.append(rmse)
        weighted_rollout_rmse += horizon_weights.get(horizon, 0.0) * rmse

    if horizon_rmses:
        metrics["val_rollout_rmse"] = float(np.mean(horizon_rmses))
        metrics["val_rollout_weighted_rmse"] = float(weighted_rollout_rmse)
    for horizon in horizons:
        metrics[f"val_rollout_h{horizon}_weight"] = float(horizon_weights.get(horizon, 0.0))
    return metrics


def _save_checkpoint(checkpoint: dict, canonical_path: Path, compat_path: Path | None = None) -> None:
    torch.save(checkpoint, canonical_path)
    if compat_path is not None and compat_path != canonical_path:
        torch.save(checkpoint, compat_path)




def train_backbone(
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
    checkpoint_metric: str = "val_loss",
    rollout_val_horizons: Optional[Sequence[int]] = None,
    rollout_val_max_windows: int = 256,
    rollout_selection_weights: Optional[Sequence[float]] = None,
) -> str:
    if multi_horizons is None:
        multi_horizons = [2, 4]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    max_horizon = max(multi_horizons)
    rollout_val_horizons = list(rollout_val_horizons or multi_horizons)
    rollout_val_horizons = [int(h) for h in rollout_val_horizons if int(h) > 0 and int(h) <= max_horizon]
    if not rollout_val_horizons:
        rollout_val_horizons = [int(max_horizon)]
    rollout_selection_weights = (
        [float(w) for w in rollout_selection_weights]
        if rollout_selection_weights is not None
        else None
    )
    use_rollout_selection = checkpoint_metric == "rollout_rmse"
    if checkpoint_metric not in {"val_loss", "rollout_rmse"}:
        raise ValueError(f"Unsupported checkpoint_metric: {checkpoint_metric}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[TRAIN_BACKBONE] Device: {device}")

    full_dataset = BOPTESTBackboneDataset(data_path, max_horizon=max_horizon)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_positions = list(range(n_train))
    val_positions = list(range(n_train, n_total))
    train_dataset = torch.utils.data.Subset(full_dataset, train_positions)
    val_dataset = torch.utils.data.Subset(full_dataset, val_positions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[TRAIN_BACKBONE] Train: {n_train} | Val: {n_val}")
    print(f"[TRAIN_BACKBONE] Multi-step horizons: {multi_horizons}")
    print(f"[TRAIN_BACKBONE] Checkpoint metric: {checkpoint_metric}")
    if use_rollout_selection:
        print(f"[TRAIN_BACKBONE] Rollout val horizons: {rollout_val_horizons}")
        print(f"[TRAIN_BACKBONE] Rollout val max windows: {rollout_val_max_windows}")
        print(f"[TRAIN_BACKBONE] Rollout selection weights: {rollout_selection_weights}")

    model = RCNeuralODEv2(hidden_dim=hidden_dim).to(device)
    model.summary()

    criterion = SurrogateLossV2(
        lambda_temp=1.0, lambda_power=0.1, lambda_multi=0.5,
        lambda_physics=0.05, multi_horizons=multi_horizons,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_selection_value = float("inf")
    patience_cnt = 0
    best_metrics: Dict[str, float] = {}
    best_rollout_metrics: Dict[str, float] = {}
    history = []
    output_dir_path = Path(output_dir)
    model_path = output_dir_path / "rc_node_backbone_best.pt"
    compat_model_path = output_dir_path / "rc_node_v2_best.pt"

    print(f"\n[TRAIN_BACKBONE] Starting training: epochs={epochs}, batch={batch_size}")
    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | "
          f"{'RMSE':>7} | {'R2':>6} | {'L_multi':>8} | {'Rollout':>8} | {'LR':>10}")
    print("-" * 95)

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

        val_loss = np.mean([l["loss_total"] for l in val_losses])
        val_multi = np.mean([l["loss_multi"] for l in val_losses])
        metrics = compute_metrics(
            torch.cat(all_t_pred), torch.cat(all_t_true),
            torch.cat(all_p_pred), torch.cat(all_p_true),
        )
        rollout_metrics = evaluate_rollout_rmse(
            model=model,
            dataset=full_dataset,
            subset_positions=val_positions,
            horizons=rollout_val_horizons,
            device=device,
            max_windows=rollout_val_max_windows,
            selection_weights=rollout_selection_weights,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_multi": val_multi,
            **metrics,
            **rollout_metrics,
        }
        history.append(row)

        lr_now = scheduler.get_last_lr()[0]
        rollout_value = float(rollout_metrics.get("val_rollout_rmse", float("nan")))
        if epoch % 10 == 0 or epoch == 1:
            rollout_display = f"{rollout_value:>8.4f}" if np.isfinite(rollout_value) else f"{'-':>8}"
            print(
                f"{epoch:>6} | {train_loss:>10.5f} | {val_loss:>10.5f} | "
                f"{metrics['rmse_temp']:>7.4f} | {metrics['r2_temp']:>6.3f} | "
                f"{val_multi:>8.4f} | {rollout_display} | {lr_now:>10.6f}"
            )

        if use_rollout_selection:
            selection_value = float(
                rollout_metrics.get(
                    "val_rollout_weighted_rmse",
                    rollout_metrics.get("val_rollout_rmse", float("inf")),
                )
            )
        else:
            selection_value = float(val_loss)
        if selection_value < best_selection_value - 1e-6:
            best_selection_value = selection_value
            best_metrics = metrics.copy()
            best_rollout_metrics = rollout_metrics.copy()
            patience_cnt = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "hidden_dim": hidden_dim,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "metrics": metrics,
                "rollout_metrics": rollout_metrics,
                "multi_horizons": list(multi_horizons),
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_value": selection_value,
                "rollout_selection_weights": rollout_selection_weights,
            }
            _save_checkpoint(checkpoint, model_path, compat_model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[TRAIN_BACKBONE] Early stopping at epoch {epoch}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir_path / "train_history_backbone.csv", index=False)
    hist_df.to_csv(output_dir_path / "train_history_v2.csv", index=False)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best {checkpoint_metric}: {best_selection_value:.6f}")
    print(f"  RMSE (1-step):       {best_metrics['rmse_temp']:.4f} C")
    print(f"  R2:                  {best_metrics['r2_temp']:.4f}")
    if best_rollout_metrics:
        for horizon in rollout_val_horizons:
            key = f"val_rollout_h{int(horizon)}_rmse"
            if key in best_rollout_metrics:
                print(f"  Val rollout h{int(horizon)}:   {best_rollout_metrics[key]:.4f} C")
    print(f"  Model saved:         {model_path}")
    if compat_model_path != model_path:
        print(f"  Compat model saved:  {compat_model_path}")

    return str(model_path)




def validate_backbone_safety(
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
    if "step_sec" in df.columns and df["step_sec"].notna().any():
        step_sec = float(df["step_sec"].dropna().iloc[0])
    elif "sim_time_sec" in df.columns and len(df) >= 2:
        diffs = df["sim_time_sec"].diff().dropna().to_numpy(dtype=float)
        step_sec = float(diffs[0]) if len(diffs) else 3600.0
    else:
        step_sec = 3600.0

    def _format_horizon(h_steps: int) -> str:
        total_sec = float(h_steps) * step_sec
        if total_sec < 3600.0:
            return f"{int(round(total_sec / 60.0))} min"
        hours = total_sec / 3600.0
        if abs(hours - round(hours)) < 1e-9:
            return f"{int(round(hours))} h"
        return f"{hours:.2f} h"

    print(f"\n{'='*60}")
    print(f"SAFETY FILTER VALIDATION")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_path} ({N} rows)")
    print(f"  Step:    {step_sec:.0f} sec")
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
        print(f"\n  Horizon = {h} steps ({_format_horizon(h)}):")
        print(f"    Rollout RMSE:        {rmse:.3f} C")
        print(f"    Bias:                {bias:+.3f} C")
        print(f"    False-safe rate:     {fs_rate:.1f}%  {status}")
        print(f"    False-unsafe rate:   {fu_rate:.1f}%")
        print(f"    Safety margin (95%): {margin_95:.2f} C")
        print(f"    Safety margin (99%): {margin_99:.2f} C")

    h4_fs = results.get('h4_false_safe_pct', 999)
    h4_margin = results.get('h4_margin_95', 999)
    ref_horizon_label = _format_horizon(4)

    print(f"\n{'='*60}")
    if h4_fs < 2.0:
        print(f"VERDICT: Model SUITABLE for safety filter without extra margin at 4 steps ({ref_horizon_label})")
    elif h4_fs < 5.0:
        print(f"VERDICT: Model suitable with safety margin = {h4_margin:.2f} C at 4 steps ({ref_horizon_label})")
        print(f"  -> comfort band for filter: [{t_low + h4_margin:.1f}, "
              f"{t_high - h4_margin:.1f}] C")
    else:
        print(f"VERDICT: Model NOT SUITABLE at 4 steps ({ref_horizon_label}). Need to improve RMSE.")
    print(f"{'='*60}")

    return results




def benchmark_backbone_speed(model_path: str, n_steps: int = 100_000) -> None:
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
    print(f"\n[BENCH] Surrogate backbone speed: {sps:,.0f} steps/sec")
    print(f"[BENCH] Speedup vs BOPTEST (~1 step/s): ~{sps:,.0f}x")




BOPTESTDatasetV2 = BOPTESTBackboneDataset
train_v2 = train_backbone
validate_safety = validate_backbone_safety
benchmark_speed_v2 = benchmark_backbone_speed


def main():
    parser = argparse.ArgumentParser(description="Train surrogate backbone with optional rollout-aware checkpoint selection.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="/app/outputs/surrogate_v2")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--multi_horizons", nargs='+', type=int, default=[2, 4])
    parser.add_argument("--checkpoint_metric", choices=["val_loss", "rollout_rmse"], default="val_loss")
    parser.add_argument("--rollout_val_horizons", nargs='+', type=int, default=None)
    parser.add_argument("--rollout_val_max_windows", type=int, default=256)
    parser.add_argument("--validate_safety", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    model_path = train_backbone(
        data_path=args.data, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, hidden_dim=args.hidden_dim,
        patience=args.patience, multi_horizons=args.multi_horizons,
        checkpoint_metric=args.checkpoint_metric,
        rollout_val_horizons=args.rollout_val_horizons,
        rollout_val_max_windows=args.rollout_val_max_windows,
    )

    if args.validate_safety:
        validate_backbone_safety(model_path, args.data)

    if args.benchmark:
        benchmark_backbone_speed(model_path)


if __name__ == "__main__":
    main()
