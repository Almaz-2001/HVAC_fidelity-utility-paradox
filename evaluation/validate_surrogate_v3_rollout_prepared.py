"""Offline prepared-rollout validation for the direct-TSup v3 surrogate.

This creates the same artifact shape as the v3.5 prepared rollout validator:
all_full_rollouts.csv, window_errors.csv, horizon_metrics.csv, and episode_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.rc_node_v2 import RCNeuralODEv2


DEFAULT_MODEL = "outputs/surrogate_v2/rc_node_v3_tsupply.pt"
DEFAULT_DATA = "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv"
DEFAULT_OUT_DIR = "outputs/surrogate_v3_rollout_prepared_15min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v3 surrogate on prepared 15-minute traces.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon-hours", default="1,4,8,24")
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_model(path: Path, device: torch.device) -> RCNeuralODEv2:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=int(checkpoint.get("hidden_dim", 64)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def load_prepared_episodes(data_path: Path, max_episodes: int | None) -> list[pd.DataFrame]:
    df = pd.read_csv(data_path)
    required = {
        "episode_id",
        "step",
        "step_sec",
        "sim_time_sec",
        "t_zone",
        "t_zone_next",
        "t_amb",
        "hour",
        "day",
        "a0_raw",
        "a1_raw",
        "p_total",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in prepared dataset: {missing}")
    episodes = []
    for idx, (_, group) in enumerate(df.sort_values(["episode_id", "step"]).groupby("episode_id", sort=False), start=1):
        episodes.append(group.reset_index(drop=True).copy())
        if max_episodes is not None and idx >= max_episodes:
            break
    return episodes


def parse_horizon_steps(raw: str, step_sec: float) -> list[int]:
    steps = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            steps.append(int(round(float(token) * 3600.0 / step_sec)))
    return sorted({s for s in steps if s > 0})


def model_step(
    model: RCNeuralODEv2,
    device: torch.device,
    t_zone: float,
    row: pd.Series,
) -> tuple[float, float]:
    with torch.no_grad():
        t_next, p_next = model(
            torch.tensor([t_zone], dtype=torch.float32, device=device),
            torch.tensor([float(row["t_amb"])], dtype=torch.float32, device=device),
            torch.tensor([float(row["hour"])], dtype=torch.float32, device=device),
            torch.tensor([float(row["day"])], dtype=torch.float32, device=device),
            torch.tensor([float(row["a0_raw"])], dtype=torch.float32, device=device),
            torch.tensor([float(row["a1_raw"])], dtype=torch.float32, device=device),
        )
    return float(t_next[0].detach().cpu()), float(p_next[0].detach().cpu())


def build_full_rollout(model: RCNeuralODEv2, device: torch.device, episode: pd.DataFrame) -> pd.DataFrame:
    t_curr = float(episode.loc[0, "t_zone"])
    rows = []
    episode_id = str(episode.loc[0, "episode_id"])
    season = str(episode.loc[0, "season"]) if "season" in episode.columns else "unknown"
    policy = str(episode.loc[0, "policy"]) if "policy" in episode.columns else "unknown"
    for _, row in episode.iterrows():
        pred_t, pred_p = model_step(model, device, t_curr, row)
        rows.append(
            {
                "episode_id": episode_id,
                "season": season,
                "policy": policy,
                "step": int(row["step"]),
                "sim_time_sec": float(row["sim_time_sec"]),
                "hour": float(row["hour"]),
                "day": float(row["day"]),
                "t_amb": float(row["t_amb"]),
                "actual_t_zone": float(row["t_zone_next"]),
                "actual_p_total_w": float(row["p_total"]),
                "t_supply_cmd_c": float(row.get("t_supply_cmd_c", np.nan)),
                "fan_u": float(row.get("fan_cmd_u", np.nan)),
                "pred_t_zone": pred_t,
                "temp_error_c": pred_t - float(row["t_zone_next"]),
                "pred_p_total_w": pred_p,
                "power_error_w": pred_p - float(row["p_total"]),
                "variant": "v3",
            }
        )
        t_curr = pred_t
    return pd.DataFrame(rows)


def build_window_errors(rollout: pd.DataFrame, horizon_steps: list[int], step_stride: int, step_sec: float) -> pd.DataFrame:
    rows = []
    for episode_id, ep in rollout.groupby("episode_id", sort=False):
        ep = ep.sort_values("step").reset_index(drop=True)
        for horizon in horizon_steps:
            for start_idx in range(0, len(ep) - horizon + 1, step_stride):
                win = ep.iloc[start_idx : start_idx + horizon]
                temp_errors = win["temp_error_c"].to_numpy(dtype=float)
                power_errors = win["power_error_w"].to_numpy(dtype=float)
                final = win.iloc[-1]
                rows.append(
                    {
                        "episode_id": episode_id,
                        "season": final["season"],
                        "policy": final["policy"],
                        "start_step": int(win.iloc[0]["step"]),
                        "horizon_h": float(horizon * step_sec / 3600.0),
                        "horizon_steps": int(horizon),
                        "pred_t_end_c": float(final["pred_t_zone"]),
                        "actual_t_end_c": float(final["actual_t_zone"]),
                        "temp_error_c": float(final["temp_error_c"]),
                        "energy_target_error_kwh": float(win["power_error_w"].sum() * step_sec / 3600.0 / 1000.0),
                        "temp_window_rmse_c": float(np.sqrt(np.mean(temp_errors**2))),
                        "power_window_rmse_w": float(np.sqrt(np.mean(power_errors**2))),
                    }
                )
    return pd.DataFrame(rows)


def summarize_horizons(window_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in window_df.groupby("horizon_h", sort=True):
        temp = group["temp_error_c"].to_numpy(dtype=float)
        power = group["energy_target_error_kwh"].to_numpy(dtype=float)
        rows.append(
            {
                "horizon_h": float(horizon),
                "horizon_steps": int(group["horizon_steps"].iloc[0]),
                "n_windows": int(len(group)),
                "temp_rmse_c": float(np.sqrt(np.mean(temp**2))),
                "temp_mae_c": float(np.mean(np.abs(temp))),
                "temp_bias_c": float(np.mean(temp)),
                "temp_p95_abs_error_c": float(np.quantile(np.abs(temp), 0.95)),
                "energy_error_rmse_kwh": float(np.sqrt(np.mean(power**2))),
                "energy_error_mae_kwh": float(np.mean(np.abs(power))),
            }
        )
    return pd.DataFrame(rows)


def summarize_episodes(rollout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode_id, group in rollout.groupby("episode_id", sort=False):
        temp = group["temp_error_c"].to_numpy(dtype=float)
        power = group["power_error_w"].to_numpy(dtype=float)
        rows.append(
            {
                "episode_id": episode_id,
                "season": group["season"].iloc[0],
                "policy": group["policy"].iloc[0],
                "n_steps": int(len(group)),
                "temp_rmse_c": float(np.sqrt(np.mean(temp**2))),
                "temp_mae_c": float(np.mean(np.abs(temp))),
                "temp_bias_c": float(np.mean(temp)),
                "power_rmse_w": float(np.sqrt(np.mean(power**2))),
                "power_mae_w": float(np.mean(np.abs(power))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v3_dir = out_dir / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(model_path, device)
    episodes = load_prepared_episodes(data_path, args.max_episodes)
    if not episodes:
        raise RuntimeError("No prepared episodes found.")
    step_sec = float(episodes[0]["step_sec"].iloc[0])
    horizon_steps = parse_horizon_steps(args.horizon_hours, step_sec)

    rollouts = []
    for idx, episode in enumerate(episodes, start=1):
        episode_id = str(episode.loc[0, "episode_id"])
        rollout = build_full_rollout(model, device, episode)
        rmse = float(np.sqrt(np.mean(rollout["temp_error_c"].to_numpy(dtype=float) ** 2)))
        print(f"[{idx}/{len(episodes)}] {episode_id}: RMSE={rmse:.3f} C")
        rollouts.append(rollout)

    all_rollouts = pd.concat(rollouts, ignore_index=True)
    windows = build_window_errors(all_rollouts, horizon_steps, args.step_stride, step_sec)
    horizon_summary = summarize_horizons(windows)
    episode_summary = summarize_episodes(all_rollouts)

    all_rollouts.to_csv(v3_dir / "all_full_rollouts.csv", index=False)
    windows.to_csv(v3_dir / "window_errors.csv", index=False)
    horizon_summary.to_csv(v3_dir / "horizon_metrics.csv", index=False)
    episode_summary.to_csv(v3_dir / "episode_summary.csv", index=False)
    print(f"Saved v3 prepared rollout artifacts to {v3_dir}")


if __name__ == "__main__":
    main()
