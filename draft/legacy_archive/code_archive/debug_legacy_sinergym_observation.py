from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory


DEFAULT_CONFIG_DIR = "configs/legacy_sinergym"
DEFAULT_OUT_DIR = "outputs/legacy_sinergym/debug_obs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect legacy Sinergym observations and info keys to identify real temperature/power channels."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--head", type=int, default=40, help="How many obs indices to print in console")
    parser.add_argument(
        "--profiles",
        default="zeros,ones,random",
        help="Comma-separated action probes: zeros, ones, neg_ones, random",
    )
    return parser.parse_args()


def _flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    return arr


def _to_scalar(value: Any) -> float | None:
    try:
        scalar = float(value)
    except Exception:
        return None
    return scalar if math.isfinite(scalar) else None


def _extract_numeric_info(info: dict[str, Any] | None) -> dict[str, float]:
    info = dict(info or {})
    numeric: dict[str, float] = {}
    for key, value in info.items():
        scalar = _to_scalar(value)
        if scalar is not None:
            numeric[key] = scalar
            continue

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == 1 and np.isfinite(arr[0]):
                numeric[key] = float(arr[0])
    return numeric


def _action_from_profile(profile: str, env, rng: np.random.Generator) -> np.ndarray:
    shape = tuple(env.action_space.shape)
    if profile == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if profile == "ones":
        return np.ones(shape, dtype=np.float32)
    if profile == "neg_ones":
        return -np.ones(shape, dtype=np.float32)
    if profile == "random":
        try:
            env.action_space.seed(int(rng.integers(0, 2**31 - 1)))
        except Exception:
            pass
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    raise ValueError(f"Unsupported profile: {profile}")


def _obs_summary(df: pd.DataFrame) -> pd.DataFrame:
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    rows = []
    for col in obs_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        idx = int(col.split("_", 1)[1])
        rows.append(
            {
                "obs_index": idx,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "range": float(np.nanmax(values) - np.nanmin(values)),
                "temp_like_10_40": bool(np.nanmin(values) >= 10.0 and np.nanmax(values) <= 40.0),
                "power_like_0_5000": bool(np.nanmin(values) >= 0.0 and np.nanmax(values) <= 5000.0),
            }
        )
    return pd.DataFrame(rows).sort_values("obs_index").reset_index(drop=True)


def _info_summary(df: pd.DataFrame) -> pd.DataFrame:
    info_cols = [c for c in df.columns if c.startswith("info_")]
    rows = []
    for col in info_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "info_key": col.replace("info_", "", 1),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "range": float(np.nanmax(values) - np.nanmin(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("info_key").reset_index(drop=True)


def _print_head(obs: np.ndarray, limit: int) -> None:
    print(f"Observation shape: {obs.shape}")
    print("First observation values:")
    for idx, value in enumerate(obs[:limit]):
        print(f"  obs[{idx:02d}] = {value:.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_all_configs(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvFactory.create(dict(cfg["env"]))
    rng = np.random.default_rng(args.seed)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    try:
        obs, info = env.reset(seed=args.seed)
        flat_obs = _flatten_obs(obs)
        _print_head(flat_obs, args.head)
        numeric_info = _extract_numeric_info(info)
        print("\nNumeric info keys on reset:")
        if numeric_info:
            for key, value in sorted(numeric_info.items()):
                print(f"  {key} = {value:.6f}")
        else:
            print("  <none>")

        report_lines = [
            "LEGACY SINERGYM OBSERVATION DEBUG",
            "=================================",
            f"Config dir: {args.config_dir}",
            f"Seed: {args.seed}",
            f"Profiles: {', '.join(profiles)}",
            f"Observation shape: {flat_obs.shape}",
            "",
        ]

        for profile in profiles:
            print(f"\n=== PROFILE: {profile} ===")
            obs, info = env.reset(seed=args.seed)
            rows: list[dict[str, Any]] = []

            for step in range(args.steps):
                action = _action_from_profile(profile, env, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                obs_arr = _flatten_obs(obs)
                numeric_info = _extract_numeric_info(info)

                row: dict[str, Any] = {
                    "step": step,
                    "reward_scalar": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                for idx, value in enumerate(obs_arr):
                    row[f"obs_{idx}"] = float(value)
                for a_idx, value in enumerate(np.asarray(action, dtype=float).reshape(-1)):
                    row[f"action_{a_idx}"] = float(value)
                for key, value in numeric_info.items():
                    row[f"info_{key}"] = float(value)

                rows.append(row)

                if step == 0:
                    print("Step 0 numeric info keys:")
                    if numeric_info:
                        for key, value in sorted(numeric_info.items()):
                            print(f"  {key} = {value:.6f}")
                    else:
                        print("  <none>")

                if terminated or truncated:
                    obs, info = env.reset(seed=args.seed)

            steps_df = pd.DataFrame(rows)
            steps_path = out_dir / f"{profile}_steps.csv"
            steps_df.to_csv(steps_path, index=False)

            obs_summary = _obs_summary(steps_df)
            obs_summary_path = out_dir / f"{profile}_obs_summary.csv"
            obs_summary.to_csv(obs_summary_path, index=False)

            info_summary = _info_summary(steps_df)
            info_summary_path = out_dir / f"{profile}_info_summary.csv"
            info_summary.to_csv(info_summary_path, index=False)

            temp_candidates = obs_summary[
                (obs_summary["temp_like_10_40"]) & (obs_summary["std"] > 1e-4)
            ].sort_values(["range", "std"], ascending=False).head(10)
            power_candidates = obs_summary[
                (obs_summary["power_like_0_5000"]) & (obs_summary["std"] > 1.0)
            ].sort_values(["range", "std"], ascending=False).head(10)

            print("Top temp-like obs candidates:")
            if temp_candidates.empty:
                print("  <none>")
            else:
                for row in temp_candidates.itertuples(index=False):
                    print(
                        f"  obs[{row.obs_index}] mean={row.mean:.3f} std={row.std:.3f} "
                        f"min={row.min:.3f} max={row.max:.3f}"
                    )

            print("Top power-like obs candidates:")
            if power_candidates.empty:
                print("  <none>")
            else:
                for row in power_candidates.itertuples(index=False):
                    print(
                        f"  obs[{row.obs_index}] mean={row.mean:.3f} std={row.std:.3f} "
                        f"min={row.min:.3f} max={row.max:.3f}"
                    )

            report_lines.extend(
                [
                    f"[{profile}]",
                    f"steps_csv: {steps_path}",
                    f"obs_summary_csv: {obs_summary_path}",
                    f"info_summary_csv: {info_summary_path}",
                    "top_temp_like_candidates:",
                ]
            )
            if temp_candidates.empty:
                report_lines.append("  - <none>")
            else:
                for row in temp_candidates.itertuples(index=False):
                    report_lines.append(
                        f"  - obs[{row.obs_index}] mean={row.mean:.3f} std={row.std:.3f} "
                        f"min={row.min:.3f} max={row.max:.3f}"
                    )
            report_lines.append("top_power_like_candidates:")
            if power_candidates.empty:
                report_lines.append("  - <none>")
            else:
                for row in power_candidates.itertuples(index=False):
                    report_lines.append(
                        f"  - obs[{row.obs_index}] mean={row.mean:.3f} std={row.std:.3f} "
                        f"min={row.min:.3f} max={row.max:.3f}"
                    )
            report_lines.append("")

        report_path = out_dir / "debug_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\nSaved debug report: {report_path}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
