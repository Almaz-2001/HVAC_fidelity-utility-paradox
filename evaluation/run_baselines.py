from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _unwrap_obs(reset_result):
    # gymnasium reset -> (obs, info)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0], reset_result[1]
    return reset_result, {}


def _unwrap_step(step_result):
    # gymnasium step -> (obs, reward, terminated, truncated, info)
    obs, rew, terminated, truncated, info = step_result
    return obs, float(rew), bool(terminated), bool(truncated), (info or {})


def run_baseline(
    env,
    name: str,
    n_steps: int,
    out_dir: str,
    fixed_action: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> str:
    """
    
    """
    _ensure_dir(out_dir)

    # reset
    if seed is not None:
        obs, info = _unwrap_obs(env.reset(seed=seed))
    else:
        obs, info = _unwrap_obs(env.reset())

    rows: List[Dict[str, Any]] = []

    for t in range(n_steps):
        if fixed_action is None:
            action = env.action_space.sample()
        else:
            action = fixed_action

        obs, rew, terminated, truncated, info = _unwrap_step(env.step(action))

        rv = info.get("reward_vector", {})
        
        a_final = info.get("action_final", action)

        rows.append(
            {
                "step": t,
                "reward_scalar": rew,
                "comfort": rv.get("comfort"),
                "energy": rv.get("energy"),
                "zone_temp": rv.get("zone_temp", info.get("zone_temp")),
                "hvac_power": rv.get("hvac_power", info.get("hvac_power")),
                "a0": float(a_final[0]) if hasattr(a_final, "__len__") else float(a_final),
                "a1": float(a_final[1]) if hasattr(a_final, "__len__") and len(a_final) > 1 else np.nan,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    out_path = str(Path(out_dir) / f"{name}.csv")
    df.to_csv(out_path, index=False)
    return out_path
