"""
evaluation/validate_surrogate_v35_rollout_prepared.py

Offline rollout validation of calibrated Surrogate v3.5 on prepared 15-minute
BOPTEST traces. This avoids live replay and evaluates the current canonical
`v3.5 15min` twin directly on the prepared Block 1.2 dataset.

Outputs:
  - horizon RMSE/bias summaries for raw_v35 and calibrated_v35
  - per-episode rollout CSVs
  - comfort-trace plots with the 21-24 C comfort band against BOPTEST
  - compact comparison summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.validate_surrogate_rollout_live import (
    plot_energy_metrics,
    plot_horizon_metrics,
    summarize_episodes,
    summarize_horizons,
    write_summary_text,
)
from surrogate.inverse_problem_boptest_v35 import StagedCalibratedSurrogateV35
from surrogate.rc_node_v35 import load_v35_from_v2_checkpoint


DEFAULT_SUMMARY = "outputs/surrogate_v35_inverse_boptest_15min/calibration_summary_boptest_v35.json"
DEFAULT_DATA = "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv"
DEFAULT_OUT_DIR = "outputs/surrogate_v35_rollout_prepared_15min"
DEFAULT_T_LOW = 21.0
DEFAULT_T_HIGH = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline rollout validation of calibrated Surrogate v3.5 on prepared 15-minute traces."
    )
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon-hours", default="1,4,8,24")
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--c-zon-min", type=float, default=5.0e4)
    parser.add_argument("--q-scale", type=float, default=3000.0)
    parser.add_argument("--comfort-low", type=float, default=DEFAULT_T_LOW)
    parser.add_argument("--comfort-high", type=float, default=DEFAULT_T_HIGH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_horizon_hours(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(float(token)))
    if not values:
        raise ValueError("At least one horizon hour is required.")
    return sorted({int(v) for v in values if int(v) > 0})


def _infer_step_seconds(df: pd.DataFrame, fallback: int = 900) -> int:
    if "step_sec" in df.columns:
        values = pd.to_numeric(df["step_sec"], errors="coerce").dropna()
        if not values.empty:
            return int(round(float(values.iloc[0])))
    if "sim_time_sec" in df.columns and len(df) >= 2:
        diffs = pd.to_numeric(df["sim_time_sec"], errors="coerce").diff().dropna()
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if not diffs.empty:
            return int(round(float(diffs.median())))
    return int(fallback)


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
    anchors = [summary_path.parent, ROOT]

    def _resolve_path(raw: str | None, default_name: str | None = None) -> str:
        candidates: list[Path] = []
        if raw is not None:
            raw_path = Path(raw)
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                candidates.append(raw_path)
                for anchor in anchors:
                    candidates.append(anchor / raw_path)
        if default_name is not None:
            default_path = summary_path.with_name(default_name)
            candidates.append(default_path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str(candidates[-1])

    checkpoint_path = _resolve_path(checkpoint_path, default_name="rc_node_v35_boptest_staged_calibrated.pt")
    base_model_path = _resolve_path(base_model_path or str(summary["model_path"]))

    torch_device = torch.device(device)
    warm_surrogate = load_v35_from_v2_checkpoint(
        base_model_path,
        device=torch_device,
        c_zon_init=float(summary.get("c_zon_prior_j_per_k", 5.3e5)),
        c_zon_min=c_zon_min,
        q_scale=q_scale,
        dt_seconds=float(summary.get("runtime_step_sec", 900)),
        legacy_step_seconds=float(summary.get("legacy_checkpoint_step_sec", 3600)),
    )
    temp_head_feature_set = str(summary.get("temp_head_feature_set", "v1"))
    model = StagedCalibratedSurrogateV35(warm_surrogate, temp_head_feature_set=temp_head_feature_set).to(torch_device)

    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.surrogate.load_state_dict(checkpoint["surrogate_state"])
    if "temp_head_state" in checkpoint:
        model.temp_head.load_state_dict(checkpoint["temp_head_state"], strict=False)
    if "power_head_state" in checkpoint:
        model.power_head.load_state_dict(checkpoint["power_head_state"], strict=False)
    model.eval()
    return model, torch_device, summary


def load_prepared_episodes(data_path: str, max_episodes: int | None = None) -> list[pd.DataFrame]:
    df = pd.read_csv(data_path)
    required = {
        "episode_id",
        "step",
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

    episodes: list[pd.DataFrame] = []
    grouped = df.sort_values(["episode_id", "step"]).groupby("episode_id", sort=False)
    for idx, (_, group) in enumerate(grouped, start=1):
        episodes.append(group.reset_index(drop=True).copy())
        if max_episodes is not None and idx >= max_episodes:
            break
    if not episodes:
        raise RuntimeError(f"No episodes found in {data_path}")
    return episodes


def build_episode_rollout_dual(
    model: StagedCalibratedSurrogateV35,
    device: torch.device,
    episode_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_t_curr = float(episode_df.loc[0, "t_zone"])
    cal_t_curr = float(episode_df.loc[0, "t_zone"])
    rows_raw: list[dict] = []
    rows_cal: list[dict] = []

    episode_id = str(episode_df.loc[0, "episode_id"])
    season = str(episode_df.loc[0, "season"]) if "season" in episode_df.columns else "unknown"
    policy = str(episode_df.loc[0, "policy"]) if "policy" in episode_df.columns else "unknown"

    for row in episode_df.itertuples(index=False):
        with torch.no_grad():
            raw_t_surr, _, raw_p_surr, _, _, _ = model(
                torch.tensor([raw_t_curr], dtype=torch.float32, device=device),
                torch.tensor([float(row.t_amb)], dtype=torch.float32, device=device),
                torch.tensor([float(row.hour)], dtype=torch.float32, device=device),
                torch.tensor([float(row.day)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a0_raw)], dtype=torch.float32, device=device),
                torch.tensor([float(row.a1_raw)], dtype=torch.float32, device=device),
            )
            _, cal_t_cal, _, cal_p_cal, _, _ = model(
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
            "episode_id": episode_id,
            "season": season,
            "policy": policy,
            "step": int(row.step),
            "sim_time_sec": float(getattr(row, "sim_time_sec", np.nan)),
            "hour": float(row.hour),
            "day": float(row.day),
            "t_amb": float(row.t_amb),
            "actual_t_zone": float(row.t_zone_next),
            "actual_p_total_w": float(row.p_total),
            "t_supply_cmd_c": float(getattr(row, "t_supply_cmd_c", np.nan)),
            "fan_u": float(getattr(row, "fan_cmd_u", np.nan)),
        }
        rows_raw.append(
            {
                **common,
                "pred_t_zone": raw_t_next,
                "temp_error_c": raw_t_next - float(row.t_zone_next),
                "pred_p_total_w": raw_p_next,
                "power_error_w": raw_p_next - float(row.p_total),
                "variant": "raw_v35",
            }
        )
        rows_cal.append(
            {
                **common,
                "pred_t_zone": cal_t_next,
                "temp_error_c": cal_t_next - float(row.t_zone_next),
                "pred_p_total_w": cal_p_next,
                "power_error_w": cal_p_next - float(row.p_total),
                "variant": "calibrated_v35",
            }
        )
        raw_t_curr = raw_t_next
        cal_t_curr = cal_t_next

    return pd.DataFrame(rows_raw), pd.DataFrame(rows_cal)


def build_horizon_windows(
    rollout_df: pd.DataFrame,
    horizon_steps: list[int],
    step_stride: int,
    step_sec: int,
) -> pd.DataFrame:
    windows: list[dict] = []
    for episode_id, ep_df in rollout_df.groupby("episode_id", sort=False):
        ep_df = ep_df.sort_values("step").reset_index(drop=True)
        n = len(ep_df)
        for horizon_steps_i in horizon_steps:
            for start_idx in range(0, n - horizon_steps_i + 1, step_stride):
                window_df = ep_df.iloc[start_idx : start_idx + horizon_steps_i]
                final_row = window_df.iloc[-1]
                temp_errors = window_df["temp_error_c"].to_numpy(dtype=np.float64)
                power_errors = window_df["power_error_w"].to_numpy(dtype=np.float64)
                windows.append(
                    {
                        "episode_id": episode_id,
                        "season": final_row["season"],
                        "policy": final_row["policy"],
                        "start_step": int(window_df.iloc[0]["step"]),
                        "horizon_h": float(horizon_steps_i * step_sec / 3600.0),
                        "horizon_steps": int(horizon_steps_i),
                        "pred_t_end_c": float(final_row["pred_t_zone"]),
                        "actual_t_end_c": float(final_row["actual_t_zone"]),
                        "temp_error_c": float(final_row["temp_error_c"]),
                        "energy_target_error_kwh": float(window_df["power_error_w"].sum() * step_sec / 3600.0 / 1000.0),
                        "temp_window_rmse_c": float(np.sqrt(np.mean(temp_errors ** 2))),
                        "power_window_rmse_w": float(np.sqrt(np.mean(power_errors ** 2))),
                    }
                )
    return pd.DataFrame(windows)


def summarize_horizons_prepared(
    window_df: pd.DataFrame,
    horizon_hours: list[int],
    n_boot: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    summary_input = window_df.rename(columns={"horizon_steps": "horizon_steps_unused"})
    return summarize_horizons(
        summary_input,
        horizons=[int(h) for h in horizon_hours],
        n_boot=n_boot,
        ci_level=ci_level,
        seed=int(seed),
    )


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
        one_h = h_df[np.isclose(h_df["horizon_h"], 1.0)]
        row = {
            "variant": name,
            "one_step_rmse_c": float(one_h["temp_rmse_c"].iloc[0]) if not one_h.empty else float("nan"),
            "one_step_bias_c": float(one_h["temp_bias_c"].iloc[0]) if not one_h.empty else float("nan"),
            "best_horizon_rmse_c": float(h_df["temp_rmse_c"].min()),
            "best_horizon_h": float(h_df.loc[h_df["temp_rmse_c"].idxmin(), "horizon_h"]),
            "longest_horizon_rmse_c": float(h_df.iloc[-1]["temp_rmse_c"]),
            "mean_episode_rmse_c": float(e_df["temp_rmse_c"].mean()),
            "mean_episode_bias_c": float(e_df["temp_bias_c"].mean()),
            "mean_episode_power_rmse_w": float(e_df["power_rmse_w"].mean()),
            "c_zon_final_j_per_k": float(summary_meta["c_zon_final_j_per_k"]),
            "stage_c_mode": summary_meta.get("stage_c_mode", "unknown"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _band_violation(temp: pd.Series, t_low: float, t_high: float) -> pd.Series:
    below = (t_low - temp).clip(lower=0.0)
    above = (temp - t_high).clip(lower=0.0)
    return below + above


def plot_comfort_trace_grid(
    actual_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    out_path: Path,
    t_low: float,
    t_high: float,
    max_panels: int = 8,
) -> None:
    episode_ids = list(actual_df["episode_id"].drop_duplicates())[:max_panels]
    if not episode_ids:
        return

    n = len(episode_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, episode_id in enumerate(episode_ids):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        a_df = actual_df[actual_df["episode_id"] == episode_id].sort_values("step")
        r_df = raw_df[raw_df["episode_id"] == episode_id].sort_values("step")
        c_df = cal_df[cal_df["episode_id"] == episode_id].sort_values("step")

        x = a_df["step"].to_numpy(dtype=float)
        ax.axhspan(t_low, t_high, color="#ccebc5", alpha=0.45, label="Comfort band")
        ax.plot(x, a_df["actual_t_zone"], color="#1f77b4", linewidth=1.8, label="BOPTEST")
        ax.plot(x, r_df["pred_t_zone"], color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Raw v3.5")
        ax.plot(x, c_df["pred_t_zone"], color="#2ca02c", linewidth=1.4, alpha=0.9, label="Calibrated v3.5")
        season = str(a_df["season"].iloc[0]) if "season" in a_df.columns else "unknown"
        policy = str(a_df["policy"].iloc[0]) if "policy" in a_df.columns else "unknown"
        ax.set_title(f"{season} | {policy}")
        ax.set_xlabel("Step (15 min)")
        ax.set_ylabel("Zone temp [C]")
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle("Comfort Trace: BOPTEST vs Raw/Calibrated v3.5 (15-minute prepared episodes)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_comfort_violation_comparison(
    rollout_df: pd.DataFrame,
    out_path: Path,
    t_low: float,
    t_high: float,
) -> None:
    if rollout_df.empty:
        return
    rows = []
    for variant, group in rollout_df.groupby("variant", sort=False):
        actual_violation = _band_violation(group["actual_t_zone"], t_low, t_high)
        pred_violation = _band_violation(group["pred_t_zone"], t_low, t_high)
        rows.append(
            {
                "variant": variant,
                "actual_in_band_pct": float((actual_violation <= 0).mean() * 100.0),
                "pred_in_band_pct": float((pred_violation <= 0).mean() * 100.0),
                "pred_mean_violation_c": float(pred_violation.mean()),
                "pred_p95_violation_c": float(np.quantile(pred_violation, 0.95)),
            }
        )
    summary_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    x = np.arange(len(summary_df))
    width = 0.35
    axes[0].bar(x - width / 2, summary_df["actual_in_band_pct"], width, label="BOPTEST", color="#4c78a8")
    axes[0].bar(x + width / 2, summary_df["pred_in_band_pct"], width, label="Surrogate", color="#54a24b")
    axes[0].set_xticks(x, summary_df["variant"])
    axes[0].set_ylabel("In-band [%]")
    axes[0].set_title("Comfort-band occupancy")
    axes[0].legend()

    axes[1].bar(x - width / 2, summary_df["pred_mean_violation_c"], width, label="Mean violation", color="#f58518")
    axes[1].bar(x + width / 2, summary_df["pred_p95_violation_c"], width, label="P95 violation", color="#e45756")
    axes[1].set_xticks(x, summary_df["variant"])
    axes[1].set_ylabel("Violation [C]")
    axes[1].set_title("Predicted comfort violation")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_power_trace_grid(
    raw_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    out_path: Path,
    max_panels: int = 8,
) -> None:
    episode_ids = list(raw_df["episode_id"].drop_duplicates())[:max_panels]
    if not episode_ids:
        return

    n = len(episode_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, episode_id in enumerate(episode_ids):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        r_df = raw_df[raw_df["episode_id"] == episode_id].sort_values("step")
        c_df = cal_df[cal_df["episode_id"] == episode_id].sort_values("step")
        x = r_df["step"].to_numpy(dtype=float)
        ax.plot(x, r_df["actual_p_total_w"], color="#1f77b4", linewidth=1.8, label="BOPTEST")
        ax.plot(x, r_df["pred_p_total_w"], color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Raw v3.5")
        ax.plot(x, c_df["pred_p_total_w"], color="#2ca02c", linewidth=1.4, alpha=0.9, label="Calibrated v3.5")
        season = str(r_df["season"].iloc[0]) if "season" in r_df.columns else "unknown"
        policy = str(r_df["policy"].iloc[0]) if "policy" in r_df.columns else "unknown"
        ax.set_title(f"{season} | {policy}")
        ax.set_xlabel("Step (15 min)")
        ax.set_ylabel("HVAC power [W]")
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle("HVAC Power Trace: BOPTEST vs Raw/Calibrated v3.5", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_energy_trace_grid(
    raw_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    out_path: Path,
    step_sec: int,
    max_panels: int = 8,
) -> None:
    episode_ids = list(raw_df["episode_id"].drop_duplicates())[:max_panels]
    if not episode_ids:
        return

    n = len(episode_ids)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    scale_kwh = float(step_sec) / 3600.0 / 1000.0

    for idx, episode_id in enumerate(episode_ids):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        r_df = raw_df[raw_df["episode_id"] == episode_id].sort_values("step")
        c_df = cal_df[cal_df["episode_id"] == episode_id].sort_values("step")
        x = r_df["step"].to_numpy(dtype=float)
        actual_energy = np.cumsum(r_df["actual_p_total_w"].to_numpy(dtype=float) * scale_kwh)
        raw_energy = np.cumsum(r_df["pred_p_total_w"].to_numpy(dtype=float) * scale_kwh)
        cal_energy = np.cumsum(c_df["pred_p_total_w"].to_numpy(dtype=float) * scale_kwh)
        ax.plot(x, actual_energy, color="#1f77b4", linewidth=1.8, label="BOPTEST")
        ax.plot(x, raw_energy, color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Raw v3.5")
        ax.plot(x, cal_energy, color="#2ca02c", linewidth=1.4, alpha=0.9, label="Calibrated v3.5")
        season = str(r_df["season"].iloc[0]) if "season" in r_df.columns else "unknown"
        policy = str(r_df["policy"].iloc[0]) if "policy" in r_df.columns else "unknown"
        ax.set_title(f"{season} | {policy}")
        ax.set_xlabel("Step (15 min)")
        ax.set_ylabel("Cumulative HVAC energy [kWh]")
        ax.grid(True, alpha=0.25)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle("Cumulative HVAC Energy: BOPTEST vs Raw/Calibrated v3.5", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_prepared_summary_text(
    summary_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    out_dir: Path,
    t_low: float,
    t_high: float,
) -> None:
    lines = []
    lines.append("OFFLINE V3.5 ROLLOUT VALIDATION ON PREPARED 15-MINUTE TRACES")
    lines.append("=" * 72)
    lines.append(f"Comfort band: [{t_low:.1f}, {t_high:.1f}] C")
    if not summary_df.empty:
        one_h = summary_df[np.isclose(summary_df["horizon_h"], 1.0)]
        if not one_h.empty:
            row = one_h.iloc[0]
            lines.append(
                f"1h rollout: RMSE={row['temp_rmse_c']:.3f} C "
                f"[{row['temp_rmse_ci_low']:.3f}, {row['temp_rmse_ci_high']:.3f}], "
                f"bias={row['temp_bias_c']:+.3f} C"
            )
        last = summary_df.iloc[-1]
        lines.append(
            f"Longest tested horizon {last['horizon_h']:.1f}h: "
            f"RMSE={last['temp_rmse_c']:.3f} C, P95 abs-error={last['temp_p95_abs_error_c']:.3f} C"
        )
    if not episode_df.empty:
        best = episode_df.loc[episode_df["temp_rmse_c"].idxmin()]
        worst = episode_df.loc[episode_df["temp_rmse_c"].idxmax()]
        lines.append(
            f"Best episode: {best['episode_id']} (RMSE={best['temp_rmse_c']:.3f} C, bias={best['temp_bias_c']:+.3f} C)"
        )
        lines.append(
            f"Worst episode: {worst['episode_id']} (RMSE={worst['temp_rmse_c']:.3f} C, bias={worst['temp_bias_c']:+.3f} C)"
        )
        lines.append(
            f"Mean episode power RMSE: {episode_df['power_rmse_w'].mean():.2f} W"
        )
    (out_dir / "prepared_rollout_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_v35"
    cal_dir = out_dir / "calibrated_v35"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)

    horizon_hours = _parse_horizon_hours(args.horizon_hours)
    model, device, calib_summary = load_v35_calibrated_model(
        summary_json=args.summary_json,
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        device=args.device,
        c_zon_min=args.c_zon_min,
        q_scale=args.q_scale,
    )
    episodes = load_prepared_episodes(args.data, max_episodes=args.max_episodes)
    first_df = episodes[0]
    step_sec = _infer_step_seconds(first_df, fallback=int(calib_summary.get("runtime_step_sec", 900)))
    horizon_steps = [int(round(h * 3600 / step_sec)) for h in horizon_hours]

    print("=" * 88)
    print("OFFLINE V3.5 ROLLOUT VALIDATION ON PREPARED 15-MINUTE DATASET")
    print("=" * 88)
    print(f"Summary JSON: {args.summary_json}")
    print(f"Data:         {args.data}")
    print(f"Episodes:     {len(episodes)}")
    print(f"Step:         {step_sec} s")
    print(f"Horizons [h]: {horizon_hours}")
    print(f"Horizons [-]: {horizon_steps} steps")
    print(f"Output dir:   {out_dir}")
    print(f"Device:       {device}")
    print(f"Stage C mode: {calib_summary.get('stage_c_mode', 'unknown')}")
    print(f"C_zon final:  {calib_summary['c_zon_final_j_per_k']:.3e} J/K")

    raw_frames = []
    cal_frames = []
    raw_windows = []
    cal_windows = []

    for idx, episode_df in enumerate(episodes, start=1):
        episode_id = str(episode_df.loc[0, "episode_id"])
        print(f"\n[{idx}/{len(episodes)}] Evaluating {episode_id} ...")
        raw_df, cal_df = build_episode_rollout_dual(model, device, episode_df)
        raw_frames.append(raw_df)
        cal_frames.append(cal_df)
        raw_windows.append(build_horizon_windows(raw_df, horizon_steps, args.step_stride, step_sec))
        cal_windows.append(build_horizon_windows(cal_df, horizon_steps, args.step_stride, step_sec))

        raw_rmse = float(np.sqrt(np.mean(raw_df["temp_error_c"].to_numpy(dtype=np.float64) ** 2)))
        cal_rmse = float(np.sqrt(np.mean(cal_df["temp_error_c"].to_numpy(dtype=np.float64) ** 2)))
        print(f"  steps={len(episode_df):4d} | raw RMSE={raw_rmse:.3f} C | calibrated RMSE={cal_rmse:.3f} C")

    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    cal_all = pd.concat(cal_frames, ignore_index=True) if cal_frames else pd.DataFrame()
    raw_win_all = pd.concat(raw_windows, ignore_index=True) if raw_windows else pd.DataFrame()
    cal_win_all = pd.concat(cal_windows, ignore_index=True) if cal_windows else pd.DataFrame()

    raw_all.to_csv(raw_dir / "all_full_rollouts.csv", index=False)
    cal_all.to_csv(cal_dir / "all_full_rollouts.csv", index=False)
    raw_win_all.to_csv(raw_dir / "window_errors.csv", index=False)
    cal_win_all.to_csv(cal_dir / "window_errors.csv", index=False)

    raw_h = summarize_horizons_prepared(raw_win_all, horizon_hours, args.bootstrap, args.ci_level, args.seed)
    cal_h = summarize_horizons_prepared(cal_win_all, horizon_hours, args.bootstrap, args.ci_level, args.seed + 1000)
    raw_e = summarize_episodes(raw_all)
    cal_e = summarize_episodes(cal_all)

    raw_h.to_csv(raw_dir / "horizon_metrics.csv", index=False)
    cal_h.to_csv(cal_dir / "horizon_metrics.csv", index=False)
    raw_e.to_csv(raw_dir / "episode_summary.csv", index=False)
    cal_e.to_csv(cal_dir / "episode_summary.csv", index=False)

    plot_horizon_metrics(raw_h, raw_dir)
    plot_energy_metrics(raw_h, raw_dir)
    write_summary_text(raw_h, raw_e, raw_dir)

    plot_horizon_metrics(cal_h, cal_dir)
    plot_energy_metrics(cal_h, cal_dir)
    write_summary_text(cal_h, cal_e, cal_dir)

    actual_df = raw_all[["episode_id", "season", "policy", "step", "actual_t_zone"]].copy()
    actual_df = actual_df.drop_duplicates(subset=["episode_id", "step"]).reset_index(drop=True)
    plot_comfort_trace_grid(
        actual_df=actual_df,
        raw_df=raw_all,
        cal_df=cal_all,
        out_path=out_dir / "comfort_trace_grid.png",
        t_low=float(args.comfort_low),
        t_high=float(args.comfort_high),
    )
    plot_comfort_trace_grid(
        actual_df=actual_df,
        raw_df=raw_all,
        cal_df=cal_all,
        out_path=out_dir / "comfort_trace_21_24_vs_boptest.png",
        t_low=float(args.comfort_low),
        t_high=float(args.comfort_high),
    )
    plot_power_trace_grid(
        raw_df=raw_all,
        cal_df=cal_all,
        out_path=out_dir / "hvac_power_trace_vs_boptest.png",
    )
    plot_cumulative_energy_trace_grid(
        raw_df=raw_all,
        cal_df=cal_all,
        out_path=out_dir / "cumulative_energy_trace_vs_boptest.png",
        step_sec=step_sec,
    )
    combined_rollout = pd.concat([raw_all, cal_all], ignore_index=True)
    plot_comfort_violation_comparison(
        rollout_df=combined_rollout,
        out_path=out_dir / "comfort_violation_comparison.png",
        t_low=float(args.comfort_low),
        t_high=float(args.comfort_high),
    )

    compare_df = compare_summary(raw_h, cal_h, raw_e, cal_e, calib_summary)
    compare_df.to_csv(out_dir / "v35_prepared_compare_summary.csv", index=False)
    (out_dir / "calibration_summary_snapshot.json").write_text(json.dumps(calib_summary, indent=2), encoding="utf-8")
    write_prepared_summary_text(cal_h, cal_e, out_dir, float(args.comfort_low), float(args.comfort_high))

    print(f"\n{'=' * 88}")
    print("OFFLINE V3.5 PREPARED ROLLOUT VALIDATION COMPLETE")
    print(f"{'=' * 88}")
    if not compare_df.empty:
        print(compare_df.to_string(index=False, justify="center"))
    print("\nSaved:")
    print(f"  {out_dir / 'v35_prepared_compare_summary.csv'}")
    print(f"  {cal_dir / 'horizon_metrics.csv'}")
    print(f"  {out_dir / 'comfort_trace_grid.png'}")
    print(f"  {out_dir / 'comfort_trace_21_24_vs_boptest.png'}")
    print(f"  {out_dir / 'hvac_power_trace_vs_boptest.png'}")
    print(f"  {out_dir / 'cumulative_energy_trace_vs_boptest.png'}")
    print(f"  {out_dir / 'comfort_violation_comparison.png'}")
    print(f"  {out_dir / 'prepared_rollout_summary.txt'}")


if __name__ == "__main__":
    main()
