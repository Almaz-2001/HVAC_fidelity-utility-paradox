from __future__ import annotations

import argparse
import json
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
DEFAULT_OUT_DIR = "outputs/legacy_sinergym/debug_action_space"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect legacy Sinergym action mapping from raw agent action to final physical setpoints."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-steps", type=int, default=10)
    return parser.parse_args()


def _flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _as_array(value: Any, expected_dim: int) -> np.ndarray:
    if value is None:
        return np.full(expected_dim, np.nan, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < expected_dim:
        pad = np.full(expected_dim - arr.size, np.nan, dtype=np.float32)
        arr = np.concatenate([arr, pad])
    return arr[:expected_dim]


def _obs_name(env: Any, idx: int, fallback: str) -> str:
    names = getattr(env, "observation_variables", None) or []
    if 0 <= idx < len(names):
        return str(names[idx])
    return fallback


def _row_from_step(
    *,
    probe_name: str,
    step_idx: int,
    reward: float,
    obs: np.ndarray,
    info: dict[str, Any],
    action_dim: int,
    heat_sp_idx: int,
    cool_sp_idx: int,
    zone_idx: int,
    power_idx: int,
) -> dict[str, Any]:
    raw = _as_array(info.get("action_raw"), action_dim)
    limited = _as_array(info.get("action_rate_limited"), action_dim)
    physical_pre = _as_array(info.get("action_physical_pre_safety"), action_dim)
    physical_final = _as_array(info.get("action_physical_final"), action_dim)

    row: dict[str, Any] = {
        "probe": probe_name,
        "step": step_idx,
        "reward_scalar": float(reward),
        "heating_setpoint_obs": float(obs[heat_sp_idx]),
        "cooling_setpoint_obs": float(obs[cool_sp_idx]),
        "zone_temp_obs": float(obs[zone_idx]),
        "hvac_power_obs": float(obs[power_idx]),
        "zone_temp_source": info.get("zone_temp_source"),
        "hvac_power_source": info.get("hvac_power_source"),
    }

    for prefix, arr in (
        ("raw", raw),
        ("limited", limited),
        ("physical_pre_safety", physical_pre),
        ("physical_final", physical_final),
    ):
        row[f"{prefix}_a0"] = float(arr[0]) if action_dim > 0 else np.nan
        row[f"{prefix}_a1"] = float(arr[1]) if action_dim > 1 else np.nan

    return row


def _single_step_probe(env: Any, action: np.ndarray, seed: int, probe_name: str, indices: dict[str, int]) -> dict[str, Any]:
    env.reset(seed=seed)
    obs, reward, terminated, truncated, info = env.step(action)
    row = _row_from_step(
        probe_name=probe_name,
        step_idx=0,
        reward=reward,
        obs=_flatten_obs(obs),
        info=info,
        action_dim=int(np.prod(env.action_space.shape)),
        heat_sp_idx=indices["heat_sp"],
        cool_sp_idx=indices["cool_sp"],
        zone_idx=indices["zone"],
        power_idx=indices["power"],
    )
    row["terminated"] = bool(terminated)
    row["truncated"] = bool(truncated)
    return row


def _sequence_probe(
    env: Any,
    actions: list[np.ndarray],
    seed: int,
    probe_name: str,
    indices: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env.reset(seed=seed)
    for step_idx, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        row = _row_from_step(
            probe_name=probe_name,
            step_idx=step_idx,
            reward=reward,
            obs=_flatten_obs(obs),
            info=info,
            action_dim=int(np.prod(env.action_space.shape)),
            heat_sp_idx=indices["heat_sp"],
            cool_sp_idx=indices["cool_sp"],
            zone_idx=indices["zone"],
            power_idx=indices["power"],
        )
        row["terminated"] = bool(terminated)
        row["truncated"] = bool(truncated)
        rows.append(row)
        if terminated or truncated:
            break
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_all_configs(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvFactory.create(dict(cfg["env"]))
    action_dim = int(np.prod(env.action_space.shape))

    indices = {
        "heat_sp": 10,
        "cool_sp": 11,
        "zone": getattr(env, "temp_index", 12),
        "power": getattr(env, "energy_index", 16),
    }

    one_step_actions: dict[str, np.ndarray] = {
        "raw_neg_neg": np.array([-1.0, -1.0], dtype=np.float32),
        "raw_zero_zero": np.array([0.0, 0.0], dtype=np.float32),
        "raw_pos_pos": np.array([1.0, 1.0], dtype=np.float32),
        "heat_like": np.array([0.5, 1.0], dtype=np.float32),
        "cool_like": np.array([-1.0, -0.5], dtype=np.float32),
        "cross_low_high": np.array([-1.0, 1.0], dtype=np.float32),
        "cross_high_low": np.array([1.0, -1.0], dtype=np.float32),
    }

    sequence_len = max(int(args.sequence_steps), 2)
    sequence_actions: dict[str, list[np.ndarray]] = {
        "hold_heat_like": [np.array([0.5, 1.0], dtype=np.float32) for _ in range(sequence_len)],
        "hold_cool_like": [np.array([-1.0, -0.5], dtype=np.float32) for _ in range(sequence_len)],
        "alternate_heat_cool": [
            np.array([0.5, 1.0], dtype=np.float32) if i % 2 == 0 else np.array([-1.0, -0.5], dtype=np.float32)
            for i in range(sequence_len)
        ],
        "sweep_to_cool": [
            np.array([0.5, 1.0], dtype=np.float32),
            np.array([0.2, 0.7], dtype=np.float32),
            np.array([0.0, 0.4], dtype=np.float32),
            np.array([-0.3, 0.1], dtype=np.float32),
            np.array([-0.6, -0.2], dtype=np.float32),
            np.array([-1.0, -0.5], dtype=np.float32),
        ],
    }

    try:
        single_rows = [
            _single_step_probe(env, action=action, seed=args.seed, probe_name=name, indices=indices)
            for name, action in one_step_actions.items()
        ]
        single_df = pd.DataFrame(single_rows)
        single_path = out_dir / "single_step_action_mapping.csv"
        single_df.to_csv(single_path, index=False)

        seq_rows: list[dict[str, Any]] = []
        for name, actions in sequence_actions.items():
            seq_rows.extend(_sequence_probe(env, actions, args.seed, name, indices))
        seq_df = pd.DataFrame(seq_rows)
        seq_path = out_dir / "sequence_action_mapping.csv"
        seq_df.to_csv(seq_path, index=False)

        report = {
            "config_dir": args.config_dir,
            "seed": args.seed,
            "action_space_low": np.asarray(env.action_space.low, dtype=float).reshape(-1).tolist(),
            "action_space_high": np.asarray(env.action_space.high, dtype=float).reshape(-1).tolist(),
            "indices": indices,
            "observation_names": {
                "heating_setpoint_obs": _obs_name(env, indices["heat_sp"], "obs[10]"),
                "cooling_setpoint_obs": _obs_name(env, indices["cool_sp"], "obs[11]"),
                "zone_temp_obs": _obs_name(env, indices["zone"], f"obs[{indices['zone']}]"),
                "hvac_power_obs": _obs_name(env, indices["power"], f"obs[{indices['power']}]"),
            },
            "single_step_csv": str(single_path),
            "sequence_csv": str(seq_path),
        }
        report_path = out_dir / "action_space_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        txt_lines = [
            "LEGACY SINERGYM ACTION SPACE DEBUG",
            "==================================",
            f"Config dir: {args.config_dir}",
            f"Seed: {args.seed}",
            f"Action space low: {report['action_space_low']}",
            f"Action space high: {report['action_space_high']}",
            "",
            "Observation channels used for interpretation:",
            f"  heating_setpoint_obs: {report['observation_names']['heating_setpoint_obs']}",
            f"  cooling_setpoint_obs: {report['observation_names']['cooling_setpoint_obs']}",
            f"  zone_temp_obs: {report['observation_names']['zone_temp_obs']}",
            f"  hvac_power_obs: {report['observation_names']['hvac_power_obs']}",
            "",
            f"Saved single-step mapping: {single_path}",
            f"Saved sequence mapping: {seq_path}",
            f"Saved JSON report: {report_path}",
        ]
        txt_path = out_dir / "action_space_report.txt"
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        print("\n".join(txt_lines))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
