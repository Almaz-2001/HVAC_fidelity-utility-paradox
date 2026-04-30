from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from configs.loader import load_all_configs
from envs.factory import EnvFactory


def eval_model(model_path: str, out_csv: str, n_steps: int = 2000, seed: int | None = None):
    cfg = load_all_configs("configs")
    env = EnvFactory.create(cfg["env"])

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    model = PPO.load(model_path)

    obs, _ = env.reset()
    rows = []

    for t in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        rv = (info or {}).get("reward_vector", {})
        rows.append({
            "step": t,
            "reward_scalar": float(reward),
            "comfort": rv.get("comfort"),
            "energy": rv.get("energy"),
            "zone_temp": rv.get("zone_temp"),
            "hvac_power": rv.get("hvac_power"),
            "a0": float(action[0]) if np.ndim(action) > 0 else float(action),
            "a1": float(action[1]) if np.ndim(action) > 0 and len(action) > 1 else np.nan,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

        if terminated or truncated:
            obs, _ = env.reset()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    try:
        env.close()
    except Exception:
        pass

    print("Saved eval:", out_csv)


if __name__ == "__main__":
    
    
    import sys
    mp = sys.argv[1]
    out = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    eval_model(mp, out, n_steps=steps)
