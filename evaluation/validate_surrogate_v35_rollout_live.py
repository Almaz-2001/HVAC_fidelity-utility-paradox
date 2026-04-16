"""
evaluation/validate_surrogate_v35_rollout_live.py

Live rollout validation of the calibrated Surrogate v3.5 against BOPTEST.

This script:
  1. Replays the recorded direct-TSup action sequences from boptest_v2_tsupply.csv
     on the live BESTEST Air BOPTEST testcase.
  2. Runs two recursive v3.5 rollouts on the same action/weather sequence:
       - raw structural surrogate output
       - calibrated twin output (temp/power heads applied)
  3. Computes horizon-wise rollout RMSE/bias for both variants.
  4. Saves separate CSV/plots for raw and calibrated rollouts plus a compact comparison summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.validate_surrogate_rollout_live import (
    load_action_episodes,
    replay_episode_live,
    summarize_episodes,
    summarize_horizons,
    plot_energy_metrics,
    plot_full_trajectory_grid,
    plot_horizon_metrics,
    write_summary_text,
)
from surrogate.inverse_problem_boptest_v35 import StagedCalibratedSurrogateV35
from surrogate.rc_node_v35 import load_v35_from_v2_checkpoint


DEFAULT_SUMMARY = "/app/outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json"
DEFAULT_DATA = "/app/data/surrogate_v2/boptest_v2_tsupply.csv"
DEFAULT_OUT_DIR = "/app/outputs/surrogate_v35_rollout_live"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live rollout validation of calibrated Surrogate v3.5 against BOPTEST."
    )
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 6, 12, 24])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--c-zon-min", type=float, default=5.0e4)
    parser.add_argument("--q-scale", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_v35_calibrated_model(
    summary_json: str,
    checkpoint_path: str | None,
    base_model_path: str | None,
    device: str,
    c_zon_min: float,
    q_scale: float,
) -> tuple[StagedCalibratedSurrogateV35, torch.device, dict]:
    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if checkpoint_path is None:
        checkpoint_path = str(summary_path.with_name("rc_node_v35_boptest_staged_calibrated.pt"))
    if base_model_path is None:
        base_model_path = str(summary["model_path"])

    torch_device = torch.device(device)
    warm_surrogate = load_v35_from_v2_checkpoint(
        base_model_path,
        device=torch_device,
        c_zon_init=float(summary.get("c_zon_prior_j_per_k", 5.3e5)),
        c_zon_min=c_zon_min,
        q_scale=q_scale,
    )
    model = StagedCalibratedSurrogateV35(warm_surrogate).to(torch_device)

    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.surrogate.load_state_dict(checkpoint["surrogate_state"])
    if "temp_head_state" in checkpoint:
        model.temp_head.load_state_dict(checkpoint["temp_head_state"], strict=False)
    model.power_head.load_state_dict(checkpoint["power_head_state"], strict=False)
    model.eval()
    return model, torch_device, summary


def build_episode_rollout_dual(
    model: StagedCalibratedSurrogateV35,
    device: torch.device,
    episode_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episode_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    raw_t_curr = float(episode_df.loc[0, "t_zone"])
    cal_t_curr = float(episode_df.loc[0, "t_zone"])
    rows_raw: list[dict] = []
    rows_cal: list[dict] = []

    for row in episode_df.itertuples(index=False):
        with torch.no_grad():
            raw_t_surr, raw_t_cal, raw_p_surr, raw_p_cal, _, _ = model(
                torch.tensor([raw_t_curr], dtype=torch.float32, device=device),
                torch.tensor([float(row.t_amb)], dtype=torch.float32, device=device),
                torch.tensor([float(row.hour)], dtype=torch.float32, device=device),
                torch.tensor([float(row.day)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a0_raw)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a1_raw)], dtype=torch.float32, device=device),
            )
            cal_t_surr, cal_t_cal, cal_p_surr, cal_p_cal, _, _ = model(
                torch.tensor([cal_t_curr], dtype=torch.float32, device=device),
                torch.tensor([float(row.t_amb)], dtype=torch.float32, device=device),
                torch.tensor([float(row.hour)], dtype=torch.float32, device=device),
                torch.tensor([float(row.day)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a0_raw)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a1_raw)], dtype=torch.float32, device=device),
            )

        raw_t_next = float(raw_t_surr[0].detach().cpu())
        raw_p_next = float(raw_p_surr[0].detach().cpu())
        cal_t_next = float(cal_t_cal[0].detach().cpu())
        cal_p_next = float(cal_p_cal[0].detach().cpu())

        common = {
            "episode_id": row.episode_id,
            "season": row.season,
            "policy": row.policy,
            "step": int(row.step),
            "actual_t_zone": float(row.next_t_zone),
            "actual_p_target_w": float(row.next_p_target_w),
            "actual_p_total_w": float(row.next_p_total_w),
            "t_supply_cmd_c": float(row.t_supply_cmd_c),
            "fan_u": float(row.fan_u),
            "t_amb": float(row.t_amb),
        }
        rows_raw.append(
            {
                **common,
                "pred_t_zone": raw_t_next,
                "temp_error_c": raw_t_next - float(row.next_t_zone),
                "pred_p_target_w": raw_p_next,
                "power_error_w": raw_p_next - float(row.next_p_target_w),
                "variant": "raw_v35",
            }
        )
        rows_cal.append(
            {
                **common,
                "pred_t_zone": cal_t_next,
                "temp_error_c": cal_t_next - float(row.next_t_zone),
                "pred_p_target_w": cal_p_next,
                "power_error_w": cal_p_next - float(row.next_p_target_w),
                "variant": "calibrated_v35",
            }
        )

        raw_t_curr = raw_t_next
        cal_t_curr = cal_t_next

    return pd.DataFrame(rows_raw), pd.DataFrame(rows_cal)


def build_horizon_windows_v35(
    model: StagedCalibratedSurrogateV35,
    device: torch.device,
    episode_df: pd.DataFrame,
    horizons: list[int],
    step_stride: int,
    mode: str,
) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    if mode not in {"raw", "calibrated"}:
        raise ValueError(f"Unsupported rollout mode: {mode}")

    windows: list[dict] = []
    n = len(episode_df)

    for horizon in horizons:
        for start_idx in range(0, n - horizon + 1, step_stride):
            t_curr = float(episode_df.loc[start_idx, "t_zone"])
            pred_energy_target_wh = 0.0

            for offset in range(horizon):
                row = episode_df.iloc[start_idx + offset]
                with torch.no_grad():
                    t_surr, t_cal, p_surr, p_cal, _, _ = model(
                        torch.tensor([t_curr], dtype=torch.float32, device=device),
                        torch.tensor([float(row["t_amb"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["hour"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["day"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["a0_raw"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["a1_raw"])], dtype=torch.float32, device=device),
                    )
                if mode == "raw":
                    t_curr = float(t_surr[0].detach().cpu())
                    pred_energy_target_wh += float(p_surr[0].detach().cpu())
                else:
                    t_curr = float(t_cal[0].detach().cpu())
                    pred_energy_target_wh += float(p_cal[0].detach().cpu())

            window_df = episode_df.iloc[start_idx : start_idx + horizon]
            final_row = window_df.iloc[-1]
            actual_t_end = float(final_row["next_t_zone"])
            actual_energy_target_wh = float(window_df["next_p_target_w"].sum())
            actual_energy_total_wh = float(window_df["next_p_total_w"].sum())

            windows.append(
                {
                    "episode_id": final_row["episode_id"],
                    "season": final_row["season"],
                    "policy": final_row["policy"],
                    "start_step": start_idx,
                    "horizon_h": horizon,
                    "pred_t_end_c": t_curr,
                    "actual_t_end_c": actual_t_end,
                    "temp_error_c": t_curr - actual_t_end,
                    "pred_energy_target_kwh": pred_energy_target_wh / 1000.0,
                    "actual_energy_target_kwh": actual_energy_target_wh / 1000.0,
                    "energy_target_error_kwh": (pred_energy_target_wh - actual_energy_target_wh) / 1000.0,
                    "actual_energy_total_kwh": actual_energy_total_wh / 1000.0,
                    "variant": mode,
                }
            )

    return pd.DataFrame(windows)


def compare_summary(
    raw_h: pd.DataFrame,
    cal_h: pd.DataFrame,
    raw_e: pd.DataFrame,
    cal_e: pd.DataFrame,
    summary_meta: dict,
) -> pd.DataFrame:
    rows = []
    for name, h_df, e_df in [
        ("raw_v35", raw_h, raw_e),
        ("calibrated_v35", cal_h, cal_e),
    ]:
        if h_df.empty or e_df.empty:
            continue
        one_h = h_df[h_df["horizon_h"] == 1]
        row = {
            "variant": name,
            "one_step_rmse_c": float(one_h["temp_rmse_c"].iloc[0]) if not one_h.empty else float("nan"),
            "one_step_bias_c": float(one_h["temp_bias_c"].iloc[0]) if not one_h.empty else float("nan"),
            "best_horizon_rmse_c": float(h_df["temp_rmse_c"].min()),
            "best_horizon_h": int(h_df.loc[h_df["temp_rmse_c"].idxmin(), "horizon_h"]),
            "longest_horizon_rmse_c": float(h_df.iloc[-1]["temp_rmse_c"]),
            "mean_episode_rmse_c": float(e_df["temp_rmse_c"].mean()),
            "mean_episode_bias_c": float(e_df["temp_bias_c"].mean()),
            "mean_episode_power_rmse_w": float(e_df["power_rmse_w"].mean()),
            "c_zon_final_j_per_k": float(summary_meta["c_zon_final_j_per_k"]),
            "czon_error_pct": float(summary_meta["czon_error_pct"]) if summary_meta.get("czon_error_pct") is not None else float("nan"),
            "stage_c_mode": summary_meta.get("stage_c_mode", "unknown"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_v35"
    cal_dir = out_dir / "calibrated_v35"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    model, device, calib_summary = load_v35_calibrated_model(
        summary_json=args.summary_json,
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        device=args.device,
        c_zon_min=args.c_zon_min,
        q_scale=args.q_scale,
    )
    episodes = load_action_episodes(args.data, max_steps_per_episode=args.max_steps_per_episode)

    print("=" * 88)
    print("LIVE V3.5 ROLLOUT VALIDATION AGAINST BOPTEST")
    print("=" * 88)
    print(f"Summary JSON: {args.summary_json}")
    print(f"Data:         {args.data}")
    print(f"Episodes:     {len(episodes)}")
    print(f"Horizons:     {args.horizons}")
    print(f"Output dir:   {out_dir}")
    print(f"Device:       {device}")
    print(f"Stage C mode: {calib_summary.get('stage_c_mode', 'unknown')}")
    print(f"C_zon final:  {calib_summary['c_zon_final_j_per_k']:.3e} J/K")

    replay_frames = []
    raw_rollout_frames = []
    cal_rollout_frames = []
    raw_window_frames = []
    cal_window_frames = []

    import requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for idx, episode in enumerate(episodes, start=1):
        print(f"\n[{idx}/{len(episodes)}] Replaying {episode['episode_id']} ...")
        replay_df = replay_episode_live(session, episode, out_dir)
        raw_df, cal_df = build_episode_rollout_dual(model, device, replay_df)
        raw_win = build_horizon_windows_v35(model, device, replay_df, args.horizons, args.step_stride, mode="raw")
        cal_win = build_horizon_windows_v35(model, device, replay_df, args.horizons, args.step_stride, mode="calibrated")

        replay_frames.append(replay_df)
        raw_rollout_frames.append(raw_df)
        cal_rollout_frames.append(cal_df)
        raw_window_frames.append(raw_win)
        cal_window_frames.append(cal_win)

        raw_rmse = float(np.sqrt(np.mean(raw_df["temp_error_c"].to_numpy(dtype=np.float64) ** 2)))
        cal_rmse = float(np.sqrt(np.mean(cal_df["temp_error_c"].to_numpy(dtype=np.float64) ** 2)))
        print(
            f"  steps={len(replay_df):4d} | raw RMSE={raw_rmse:.3f} C | "
            f"calibrated RMSE={cal_rmse:.3f} C"
        )

    replay_all = pd.concat(replay_frames, ignore_index=True) if replay_frames else pd.DataFrame()
    raw_all = pd.concat(raw_rollout_frames, ignore_index=True) if raw_rollout_frames else pd.DataFrame()
    cal_all = pd.concat(cal_rollout_frames, ignore_index=True) if cal_rollout_frames else pd.DataFrame()
    raw_win_all = pd.concat(raw_window_frames, ignore_index=True) if raw_window_frames else pd.DataFrame()
    cal_win_all = pd.concat(cal_window_frames, ignore_index=True) if cal_window_frames else pd.DataFrame()

    replay_all.to_csv(out_dir / "all_replays.csv", index=False)
    raw_all.to_csv(raw_dir / "all_full_rollouts.csv", index=False)
    cal_all.to_csv(cal_dir / "all_full_rollouts.csv", index=False)
    raw_win_all.to_csv(raw_dir / "window_errors.csv", index=False)
    cal_win_all.to_csv(cal_dir / "window_errors.csv", index=False)

    raw_h = summarize_horizons(raw_win_all, args.horizons, args.bootstrap, args.ci_level, args.seed)
    cal_h = summarize_horizons(cal_win_all, args.horizons, args.bootstrap, args.ci_level, args.seed + 1000)
    raw_e = summarize_episodes(raw_all)
    cal_e = summarize_episodes(cal_all)

    raw_h.to_csv(raw_dir / "horizon_metrics.csv", index=False)
    cal_h.to_csv(cal_dir / "horizon_metrics.csv", index=False)
    raw_e.to_csv(raw_dir / "episode_summary.csv", index=False)
    cal_e.to_csv(cal_dir / "episode_summary.csv", index=False)

    plot_horizon_metrics(raw_h, raw_dir)
    plot_energy_metrics(raw_h, raw_dir)
    plot_full_trajectory_grid(raw_all, raw_dir)
    write_summary_text(raw_h, raw_e, raw_dir)

    plot_horizon_metrics(cal_h, cal_dir)
    plot_energy_metrics(cal_h, cal_dir)
    plot_full_trajectory_grid(cal_all, cal_dir)
    write_summary_text(cal_h, cal_e, cal_dir)

    compare_df = compare_summary(raw_h, cal_h, raw_e, cal_e, calib_summary)
    compare_df.to_csv(out_dir / "v35_compare_summary.csv", index=False)
    (out_dir / "calibration_summary_snapshot.json").write_text(
        json.dumps(calib_summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 88}")
    print("LIVE V3.5 ROLLOUT VALIDATION COMPLETE")
    print(f"{'=' * 88}")
    if not compare_df.empty:
        print(compare_df.to_string(index=False, justify="center"))
    print(f"\nSaved:")
    print(f"  {out_dir / 'v35_compare_summary.csv'}")
    print(f"  {raw_dir / 'horizon_metrics.csv'}")
    print(f"  {cal_dir / 'horizon_metrics.csv'}")
    print(f"  {raw_dir / 'live_rollout_summary.txt'}")
    print(f"  {cal_dir / 'live_rollout_summary.txt'}")


if __name__ == "__main__":
    main()
