"""
evaluation/validate_surrogate_rollout_live.py

Live rollout validation of the current direct-TSup surrogate against BOPTEST.

This script:
  1. Replays the recorded direct-TSup action sequences from boptest_v2_tsupply.csv
     on the live BESTEST Air BOPTEST testcase.
  2. Runs recursive surrogate rollouts on the same action/weather sequence.
  3. Computes horizon-wise rollout RMSE and bias with bootstrap 95% confidence intervals.
  4. Saves full-trajectory CSVs, window-level CSVs, summary CSVs, and plots.

Important:
  - Temperature validation is fully live against BOPTEST.
  - Power validation is reported against the surrogate training target
    (P_cool + P_fan), because that is what the current tsupply dataset stores.
    Full physical power (P_cool + P_fan + P_heat) is also saved for reference.

Usage:
    PYTHONPATH=/app python3 evaluation/validate_surrogate_rollout_live.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import action_to_fan, action_to_t_supply
from surrogate.rc_node_v2 import RCNeuralODEv2


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
STEP_SEC = 3600
SELECT_TIMEOUT = 300
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0

DEFAULT_MODEL = "/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt"
DEFAULT_DATA = "/app/data/surrogate_v2/boptest_v2_tsupply.csv"
DEFAULT_OUT_DIR = "/app/outputs/surrogate_rollout_live"

SEASON_START_TIMES = {
    "winter": 0,
    "spring": 7776000,
    "summer": 15552000,
    "autumn": 23328000,
}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
POLICY_ORDER = ["random", "heat", "cool", "mixed"]


def boptest_request(
    session: requests.Session,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> dict:
    url = f"{BOPTEST_URL}{path}"
    for attempt in range(retries):
        try:
            if method == "POST":
                response = session.post(url, json=payload or {}, timeout=timeout)
            elif method == "PUT":
                response = session.put(url, json=payload or {}, timeout=timeout)
            elif method == "GET":
                response = session.get(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            if response.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed request: {url}")


def gv(payload: dict, key: str) -> float:
    value = payload.get(key, 0.0)
    return float(value.get("value", value) if isinstance(value, dict) else value)


def extract_state(payload: dict) -> dict:
    sim_time = gv(payload, "time")
    p_cool = gv(payload, "fcu_reaPCoo_y")
    p_fan = gv(payload, "fcu_reaPFan_y")
    p_heat = gv(payload, "fcu_reaPHea_y")
    t_amb_raw = gv(payload, "zon_weaSta_reaWeaTDryBul_y")
    return {
        "time": sim_time,
        "hour": (sim_time / 3600.0) % 24.0,
        "day": (sim_time / 86400.0) % 365.0,
        "t_zone": gv(payload, "zon_reaTRooAir_y") - 273.15,
        "co2": gv(payload, "zon_reaCO2RooAir_y"),
        "t_amb": t_amb_raw - 273.15 if t_amb_raw > 200.0 else t_amb_raw,
        "p_cool": p_cool,
        "p_fan": p_fan,
        "p_heat": p_heat,
        "p_target": p_cool + p_fan,
        "p_total": p_cool + p_fan + p_heat,
    }


def action_to_boptest(a0: float, a1: float) -> dict:
    t_supply = action_to_t_supply(a0)
    fan_u = action_to_fan(a1)
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": t_supply + 273.15,
    }


def load_surrogate(model_path: str, device: str = "cpu") -> tuple[RCNeuralODEv2, torch.device]:
    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=checkpoint.get("hidden_dim", 64))
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch_device)
    model.eval()
    return model, torch_device


def load_action_episodes(data_path: str, max_steps_per_episode: int | None = None) -> list[dict]:
    df = pd.read_csv(data_path)
    required = {"season", "policy", "step", "a0_raw", "a1_raw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {sorted(missing)}")

    episodes: list[dict] = []
    for season in SEASON_ORDER:
        for policy in POLICY_ORDER:
            group = df[(df["season"] == season) & (df["policy"] == policy)].sort_values("step").reset_index(drop=True)
            if group.empty:
                continue
            if max_steps_per_episode is not None:
                group = group.iloc[:max_steps_per_episode].copy()
            episode_id = f"{season}_{policy}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "season": season,
                    "policy": policy,
                    "start_time": SEASON_START_TIMES[season],
                    "actions": group,
                }
            )
    if not episodes:
        raise RuntimeError(f"No season/policy episodes found in {data_path}")
    return episodes


def replay_episode_live(session: requests.Session, episode: dict, out_dir: Path) -> pd.DataFrame:
    data = boptest_request(session, "POST", f"/testcases/{TESTCASE}/select", timeout=SELECT_TIMEOUT)
    testid = data["testid"]
    boptest_request(session, "PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request(
        session,
        "PUT",
        f"/initialize/{testid}",
        {"start_time": int(episode["start_time"]), "warmup_period": 0},
        timeout=SELECT_TIMEOUT,
    )

    payload = boptest_request(session, "POST", f"/advance/{testid}", {})
    payload = payload.get("payload", payload)
    current_state = extract_state(payload)

    rows: list[dict] = []
    try:
        for action_row in episode["actions"].itertuples(index=False):
            a0 = float(action_row.a0_raw)
            a1 = float(action_row.a1_raw)
            response = boptest_request(session, "POST", f"/advance/{testid}", action_to_boptest(a0, a1))
            next_payload = response.get("payload", response)
            next_state = extract_state(next_payload)
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "season": episode["season"],
                    "policy": episode["policy"],
                    "step": int(action_row.step),
                    "time": current_state["time"],
                    "hour": current_state["hour"],
                    "day": current_state["day"],
                    "t_zone": current_state["t_zone"],
                    "t_amb": current_state["t_amb"],
                    "co2": current_state["co2"],
                    "a0_raw": a0,
                    "a1_raw": a1,
                    "t_supply_cmd_c": action_to_t_supply(a0),
                    "fan_u": action_to_fan(a1),
                    "next_t_zone": next_state["t_zone"],
                    "next_t_amb": next_state["t_amb"],
                    "next_p_target_w": next_state["p_target"],
                    "next_p_total_w": next_state["p_total"],
                    "next_p_heat_w": next_state["p_heat"],
                }
            )
            current_state = next_state
    finally:
        try:
            boptest_request(session, "PUT", f"/stop/{testid}", timeout=10)
        except Exception:
            pass

    episode_df = pd.DataFrame(rows)
    episode_df.to_csv(out_dir / f"replay_{episode['episode_id']}.csv", index=False)
    return episode_df


def build_episode_rollout(
    model: RCNeuralODEv2,
    device: torch.device,
    episode_df: pd.DataFrame,
) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    t_curr = float(episode_df.loc[0, "t_zone"])
    rows: list[dict] = []

    for row in episode_df.itertuples(index=False):
        with torch.no_grad():
            t_next, p_pred = model(
                torch.tensor([t_curr], dtype=torch.float32, device=device),
                torch.tensor([float(row.t_amb)], dtype=torch.float32, device=device),
                torch.tensor([float(row.hour)], dtype=torch.float32, device=device),
                torch.tensor([float(row.day)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a0_raw)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a1_raw)], dtype=torch.float32, device=device),
            )
        t_pred_next = float(t_next[0].detach().cpu())
        p_pred_target_w = float(p_pred[0].detach().cpu())
        rows.append(
            {
                "episode_id": row.episode_id,
                "season": row.season,
                "policy": row.policy,
                "step": int(row.step),
                "actual_t_zone": float(row.next_t_zone),
                "pred_t_zone": t_pred_next,
                "temp_error_c": t_pred_next - float(row.next_t_zone),
                "actual_p_target_w": float(row.next_p_target_w),
                "pred_p_target_w": p_pred_target_w,
                "power_error_w": p_pred_target_w - float(row.next_p_target_w),
                "actual_p_total_w": float(row.next_p_total_w),
                "t_supply_cmd_c": float(row.t_supply_cmd_c),
                "fan_u": float(row.fan_u),
                "t_amb": float(row.t_amb),
            }
        )
        t_curr = t_pred_next

    return pd.DataFrame(rows)


def build_horizon_windows(
    model: RCNeuralODEv2,
    device: torch.device,
    episode_df: pd.DataFrame,
    horizons: list[int],
    step_stride: int,
) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    windows: list[dict] = []
    n = len(episode_df)

    for horizon in horizons:
        for start_idx in range(0, n - horizon + 1, step_stride):
            t_curr = float(episode_df.loc[start_idx, "t_zone"])
            pred_energy_target_wh = 0.0

            for offset in range(horizon):
                row = episode_df.iloc[start_idx + offset]
                with torch.no_grad():
                    t_next, p_pred = model(
                        torch.tensor([t_curr], dtype=torch.float32, device=device),
                        torch.tensor([float(row["t_amb"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["hour"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["day"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["a0_raw"])], dtype=torch.float32, device=device),
                        torch.tensor([float(row["a1_raw"])], dtype=torch.float32, device=device),
                    )
                t_curr = float(t_next[0].detach().cpu())
                pred_energy_target_wh += float(p_pred[0].detach().cpu())

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
                }
            )

    return pd.DataFrame(windows)


def block_bootstrap_ci(
    per_episode_arrays: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], float],
    n_boot: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    episode_ids = list(per_episode_arrays.keys())
    if not episode_ids:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sampled_arrays = [per_episode_arrays[eid] for eid in sampled_ids if per_episode_arrays[eid].size > 0]
        if not sampled_arrays:
            continue
        concatenated = np.concatenate(sampled_arrays)
        metrics.append(metric_fn(concatenated))

    if not metrics:
        return float("nan"), float("nan")

    alpha = 1.0 - ci_level
    return (
        float(np.quantile(metrics, alpha / 2.0)),
        float(np.quantile(metrics, 1.0 - alpha / 2.0)),
    )


def summarize_horizons(
    window_df: pd.DataFrame,
    horizons: list[int],
    n_boot: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        subset = window_df[window_df["horizon_h"] == horizon].copy()
        if subset.empty:
            continue

        temp_episode_arrays = {
            eid: grp["temp_error_c"].to_numpy(dtype=np.float64)
            for eid, grp in subset.groupby("episode_id")
        }
        energy_episode_arrays = {
            eid: grp["energy_target_error_kwh"].to_numpy(dtype=np.float64)
            for eid, grp in subset.groupby("episode_id")
        }

        temp_errors = subset["temp_error_c"].to_numpy(dtype=np.float64)
        energy_errors = subset["energy_target_error_kwh"].to_numpy(dtype=np.float64)

        temp_rmse = float(np.sqrt(np.mean(temp_errors ** 2)))
        temp_bias = float(np.mean(temp_errors))
        temp_mae = float(np.mean(np.abs(temp_errors)))
        energy_rmse = float(np.sqrt(np.mean(energy_errors ** 2)))
        energy_bias = float(np.mean(energy_errors))

        temp_rmse_ci = block_bootstrap_ci(
            temp_episode_arrays,
            lambda x: float(np.sqrt(np.mean(x ** 2))),
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed + horizon,
        )
        temp_bias_ci = block_bootstrap_ci(
            temp_episode_arrays,
            lambda x: float(np.mean(x)),
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed + 100 + horizon,
        )
        energy_rmse_ci = block_bootstrap_ci(
            energy_episode_arrays,
            lambda x: float(np.sqrt(np.mean(x ** 2))),
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed + 200 + horizon,
        )
        energy_bias_ci = block_bootstrap_ci(
            energy_episode_arrays,
            lambda x: float(np.mean(x)),
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed + 300 + horizon,
        )

        rows.append(
            {
                "horizon_h": horizon,
                "n_windows": int(len(subset)),
                "n_episodes": int(subset["episode_id"].nunique()),
                "temp_rmse_c": temp_rmse,
                "temp_rmse_ci_low": temp_rmse_ci[0],
                "temp_rmse_ci_high": temp_rmse_ci[1],
                "temp_bias_c": temp_bias,
                "temp_bias_ci_low": temp_bias_ci[0],
                "temp_bias_ci_high": temp_bias_ci[1],
                "temp_mae_c": temp_mae,
                "energy_target_rmse_kwh": energy_rmse,
                "energy_target_rmse_ci_low": energy_rmse_ci[0],
                "energy_target_rmse_ci_high": energy_rmse_ci[1],
                "energy_target_bias_kwh": energy_bias,
                "energy_target_bias_ci_low": energy_bias_ci[0],
                "energy_target_bias_ci_high": energy_bias_ci[1],
                "temp_p95_abs_error_c": float(np.quantile(np.abs(temp_errors), 0.95)),
                "energy_p95_abs_error_kwh": float(np.quantile(np.abs(energy_errors), 0.95)),
            }
        )

    return pd.DataFrame(rows)


def summarize_episodes(full_rollout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode_id, group in full_rollout_df.groupby("episode_id"):
        temp_errors = group["temp_error_c"].to_numpy(dtype=np.float64)
        power_errors = group["power_error_w"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "episode_id": episode_id,
                "season": group["season"].iloc[0],
                "policy": group["policy"].iloc[0],
                "steps": int(len(group)),
                "temp_rmse_c": float(np.sqrt(np.mean(temp_errors ** 2))),
                "temp_bias_c": float(np.mean(temp_errors)),
                "temp_mae_c": float(np.mean(np.abs(temp_errors))),
                "power_rmse_w": float(np.sqrt(np.mean(power_errors ** 2))),
                "power_bias_w": float(np.mean(power_errors)),
            }
        )
    return pd.DataFrame(rows)


def plot_horizon_metrics(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    x = summary_df["horizon_h"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(x, summary_df["temp_rmse_c"], "o-", color="#3b5bdb", linewidth=2)
    ax.fill_between(
        x,
        summary_df["temp_rmse_ci_low"],
        summary_df["temp_rmse_ci_high"],
        color="#3b5bdb",
        alpha=0.18,
        label="95% bootstrap CI",
    )
    ax.set_xlabel("Horizon [h]")
    ax.set_ylabel("Temperature RMSE [C]")
    ax.set_title("Live Surrogate Rollout RMSE vs Horizon")
    ax.legend()

    ax = axes[1]
    ax.plot(x, summary_df["temp_bias_c"], "o-", color="#d9480f", linewidth=2)
    ax.fill_between(
        x,
        summary_df["temp_bias_ci_low"],
        summary_df["temp_bias_ci_high"],
        color="#d9480f",
        alpha=0.18,
        label="95% bootstrap CI",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon [h]")
    ax.set_ylabel("Temperature bias [C]")
    ax.set_title("Live Surrogate Rollout Bias vs Horizon")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "live_rollout_temp_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_energy_metrics(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    x = summary_df["horizon_h"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(x, summary_df["energy_target_rmse_kwh"], "o-", color="#2b8a3e", linewidth=2)
    ax.fill_between(
        x,
        summary_df["energy_target_rmse_ci_low"],
        summary_df["energy_target_rmse_ci_high"],
        color="#2b8a3e",
        alpha=0.18,
        label="95% bootstrap CI",
    )
    ax.set_xlabel("Horizon [h]")
    ax.set_ylabel("Target-energy RMSE [kWh]")
    ax.set_title("Rollout Energy Error vs Horizon")
    ax.legend()

    ax = axes[1]
    ax.plot(x, summary_df["energy_target_bias_kwh"], "o-", color="#9c36b5", linewidth=2)
    ax.fill_between(
        x,
        summary_df["energy_target_bias_ci_low"],
        summary_df["energy_target_bias_ci_high"],
        color="#9c36b5",
        alpha=0.18,
        label="95% bootstrap CI",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon [h]")
    ax.set_ylabel("Target-energy bias [kWh]")
    ax.set_title("Rollout Energy Bias vs Horizon")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "live_rollout_energy_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_full_trajectory_grid(full_rollout_df: pd.DataFrame, out_dir: Path) -> None:
    if full_rollout_df.empty:
        return

    fig, axes = plt.subplots(len(SEASON_ORDER), len(POLICY_ORDER), figsize=(16, 10), sharex=False, sharey=False)

    for r, season in enumerate(SEASON_ORDER):
        for c, policy in enumerate(POLICY_ORDER):
            ax = axes[r, c]
            subset = full_rollout_df[
                (full_rollout_df["season"] == season) & (full_rollout_df["policy"] == policy)
            ]
            if subset.empty:
                ax.axis("off")
                continue
            ax.plot(subset["step"], subset["actual_t_zone"], color="#1c7ed6", linewidth=1.6, label="BOPTEST")
            ax.plot(subset["step"], subset["pred_t_zone"], color="#e8590c", linewidth=1.2, label="Surrogate")
            if r == 0:
                ax.set_title(policy)
            if c == 0:
                ax.set_ylabel(f"{season}\nT_zone [C]")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Full Live Rollout: BOPTEST vs Surrogate (all 16 season/policy episodes)", y=0.98, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "live_rollout_trajectory_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(summary_df: pd.DataFrame, episode_df: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    lines.append("LIVE SURROGATE ROLLOUT VALIDATION")
    lines.append("=" * 72)
    if not summary_df.empty:
        one_h = summary_df[summary_df["horizon_h"] == 1]
        best_row = summary_df.loc[summary_df["temp_rmse_c"].idxmin()]
        if not one_h.empty:
            row = one_h.iloc[0]
            lines.append(
                f"1h rollout: RMSE={row['temp_rmse_c']:.3f} C "
                f"[{row['temp_rmse_ci_low']:.3f}, {row['temp_rmse_ci_high']:.3f}], "
                f"bias={row['temp_bias_c']:+.3f} C"
            )
        lines.append(
            f"Best horizon by temp RMSE: {int(best_row['horizon_h'])}h "
            f"(RMSE={best_row['temp_rmse_c']:.3f} C, bias={best_row['temp_bias_c']:+.3f} C)"
        )
        last_row = summary_df.iloc[-1]
        lines.append(
            f"Longest tested horizon {int(last_row['horizon_h'])}h: "
            f"RMSE={last_row['temp_rmse_c']:.3f} C, "
            f"95%% abs-error={last_row['temp_p95_abs_error_c']:.3f} C"
        )
    if not episode_df.empty:
        worst = episode_df.loc[episode_df["temp_rmse_c"].idxmax()]
        best = episode_df.loc[episode_df["temp_rmse_c"].idxmin()]
        lines.append(
            f"Best full episode: {best['episode_id']} (RMSE={best['temp_rmse_c']:.3f} C, bias={best['temp_bias_c']:+.3f} C)"
        )
        lines.append(
            f"Worst full episode: {worst['episode_id']} (RMSE={worst['temp_rmse_c']:.3f} C, bias={worst['temp_bias_c']:+.3f} C)"
        )
    lines.append("")
    lines.append("Note: energy validation follows the surrogate training target (P_cool + P_fan).")
    (out_dir / "live_rollout_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live rollout validation of rc_node_v3_tsupply.pt against BOPTEST with bootstrap confidence intervals.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 6, 12, 24])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ci_level", type=float, default=0.95)
    parser.add_argument("--step_stride", type=int, default=1)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    model, device = load_surrogate(args.model, device=args.device)
    episodes = load_action_episodes(args.data, max_steps_per_episode=args.max_steps_per_episode)

    print("=" * 88)
    print("LIVE SURROGATE ROLLOUT VALIDATION AGAINST BOPTEST")
    print("=" * 88)
    print(f"Model:       {args.model}")
    print(f"Data:        {args.data}")
    print(f"Episodes:    {len(episodes)}")
    print(f"Horizons:    {args.horizons}")
    print(f"Bootstrap:   {args.bootstrap}")
    print(f"Output dir:  {out_dir}")
    print(f"Device:      {device}")

    try:
        version = boptest_request(session, "GET", "/version", timeout=10)
        print(f"BOPTEST:     {version['payload']['version']}")
    except Exception as exc:
        raise RuntimeError(f"BOPTEST not available: {exc}") from exc

    replay_frames = []
    full_rollout_frames = []
    window_frames = []

    t_start = time.time()
    for idx, episode in enumerate(episodes, start=1):
        print(f"\n[{idx}/{len(episodes)}] Replaying {episode['episode_id']} ...")
        replay_df = replay_episode_live(session, episode, out_dir)
        full_df = build_episode_rollout(model, device, replay_df)
        full_df.to_csv(out_dir / f"trajectory_{episode['episode_id']}.csv", index=False)
        window_df = build_horizon_windows(model, device, replay_df, args.horizons, args.step_stride)

        replay_frames.append(replay_df)
        full_rollout_frames.append(full_df)
        window_frames.append(window_df)

        temp_rmse = float(np.sqrt(np.mean(full_df["temp_error_c"].to_numpy(dtype=np.float64) ** 2)))
        print(
            f"  steps={len(replay_df):4d} | full-rollout RMSE={temp_rmse:.3f} C | "
            f"T=[{full_df['actual_t_zone'].min():.1f},{full_df['actual_t_zone'].max():.1f}]"
        )

    replay_all = pd.concat(replay_frames, ignore_index=True) if replay_frames else pd.DataFrame()
    full_rollout_all = pd.concat(full_rollout_frames, ignore_index=True) if full_rollout_frames else pd.DataFrame()
    window_all = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()

    replay_all.to_csv(out_dir / "all_replays.csv", index=False)
    full_rollout_all.to_csv(out_dir / "all_full_rollouts.csv", index=False)
    window_all.to_csv(out_dir / "window_errors.csv", index=False)

    horizon_summary = summarize_horizons(
        window_all,
        horizons=args.horizons,
        n_boot=args.bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
    )
    episode_summary = summarize_episodes(full_rollout_all)

    horizon_summary.to_csv(out_dir / "horizon_metrics.csv", index=False)
    episode_summary.to_csv(out_dir / "episode_summary.csv", index=False)

    plot_horizon_metrics(horizon_summary, out_dir)
    plot_energy_metrics(horizon_summary, out_dir)
    plot_full_trajectory_grid(full_rollout_all, out_dir)
    write_summary_text(horizon_summary, episode_summary, out_dir)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n{'=' * 88}")
    print(f"LIVE ROLLOUT VALIDATION COMPLETE ({elapsed_min:.1f} min)")
    print(f"{'=' * 88}")
    if not horizon_summary.empty:
        print(horizon_summary.to_string(index=False, justify='center', max_colwidth=18))
    print(f"\nSaved:")
    print(f"  {out_dir / 'horizon_metrics.csv'}")
    print(f"  {out_dir / 'episode_summary.csv'}")
    print(f"  {out_dir / 'live_rollout_temp_metrics.png'}")
    print(f"  {out_dir / 'live_rollout_energy_metrics.png'}")
    print(f"  {out_dir / 'live_rollout_trajectory_grid.png'}")
    print(f"  {out_dir / 'live_rollout_summary.txt'}")


if __name__ == "__main__":
    main()
