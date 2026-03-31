from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from training.train_ppo import build_ppo, maybe_save_model, train_ppo


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MORL PPO on the direct-TSup surrogate backend.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    cfg = load_all_configs("configs")
    env_cfg = dict(cfg["env"])
    agent_cfg = dict(cfg["agent"])
    train_cfg = dict(cfg["train"])

    env_cfg["backend"] = "surrogate"
    env_cfg["control_mode"] = "tsup_direct"

    out_dir = Path(args.out_dir or f"outputs/morl_surrogate_seed{args.seed}")
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_cfg["seed"] = args.seed
    train_cfg["total_timesteps"] = args.steps
    train_cfg["output_dir"] = str(out_dir)
    train_cfg["morl_csv_name"] = "morl_surrogate_log.csv"
    train_cfg["save_model"] = True
    train_cfg["save_path"] = str(models_dir)

    snapshot = {
        "env": env_cfg,
        "agent": agent_cfg,
        "train": train_cfg,
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("=" * 72)
    print("MORL SURROGATE PRETRAIN")
    print("=" * 72)
    print(f"Seed:       {args.seed}")
    print(f"Steps:      {args.steps:,}")
    print(f"Backend:    {env_cfg['backend']}")
    print(f"Control:    {env_cfg['control_mode']}")
    print(f"Surrogate:  {env_cfg.get('surrogate_path')}")
    print(f"Output dir: {out_dir}")

    env = EnvFactory.create(env_cfg)
    print(f"Obs space:  {env.observation_space}")
    print(f"Act space:  {env.action_space}")

    model = build_ppo(env, agent_cfg)
    train_ppo(model, train_cfg, reset_num_timesteps=True)

    saved = maybe_save_model(model, train_cfg)
    if saved:
        print(f"Saved model: {saved}")

    try:
        env.close()
    except Exception:
        pass

    print("=" * 72)
    print("SURROGATE PRETRAIN COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
