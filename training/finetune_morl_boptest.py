from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from training.train_ppo import MORLLogCallback, maybe_save_model


YEARLY_STARTS = [
    0,
    2678400,
    5097600,
    7776000,
    10368000,
    13132800,
    15552000,
    18316800,
    20995200,
    23587200,
    26265600,
    28857600,
]


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def seed_env_spaces(env, seed: int) -> None:
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)


def default_pretrained_path(seed: int) -> str:
    return f"outputs/morl_surrogate_seed{seed}/models/ppo_model.zip"


def load_weights(weights_path: str | None) -> dict[str, float]:
    if not weights_path:
        return {}
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Objective weight file not found: {weights_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "w_comfort": float(data.get("w_comfort", 0.0)),
        "w_energy": float(data.get("w_energy", 0.0)),
        "w_safety": float(data.get("w_safety", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MORL PPO on BOPTEST after surrogate pretraining.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing env.yaml, agent.yaml, train.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--jitter_days", type=float, default=3.0)
    parser.add_argument("--model", default=None, help="Path to surrogate-pretrained PPO model.")
    parser.add_argument("--weights_json", default=None, help="Optional JSON with w_comfort/w_energy/w_safety from ERAM pretraining.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    model_path = args.model or default_pretrained_path(args.seed)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {model_path}. "
            "Run training/train_morl_surrogate.py first or pass --model."
        )

    cfg = load_all_configs(args.config_dir)
    env_cfg = dict(cfg["env"])
    agent_cfg = dict(cfg["agent"])
    train_cfg = dict(cfg["train"])
    eram_weights = load_weights(args.weights_json)

    env_cfg["backend"] = "boptest"
    env_cfg["control_mode"] = "tsup_direct"
    env_cfg["boptest_start_time_choices"] = YEARLY_STARTS
    env_cfg["boptest_start_jitter_sec"] = float(args.jitter_days) * 86400.0
    env_cfg["boptest_warmup_sec"] = 0.0
    if eram_weights:
        env_cfg["morl"] = dict(env_cfg.get("morl", {}))
        env_cfg["morl"].update(eram_weights)

    out_dir = Path(args.out_dir or f"outputs/morl_boptest_finetune_seed{args.seed}")
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_cfg["seed"] = args.seed
    train_cfg["total_timesteps"] = args.steps
    train_cfg["output_dir"] = str(out_dir)
    train_cfg["morl_csv_name"] = "morl_boptest_finetune_log.csv"
    train_cfg["save_model"] = True
    train_cfg["save_path"] = str(models_dir)

    snapshot = {
        "env": env_cfg,
        "agent": agent_cfg,
        "train": train_cfg,
        "pretrained_model": model_path,
        "learning_rate": args.learning_rate,
        "objective_weights": eram_weights,
        "config_dir": str(Path(args.config_dir).resolve()),
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("=" * 72)
    print("MORL BOPTEST FINE-TUNE")
    print("=" * 72)
    print(f"Seed:            {args.seed}")
    print(f"Pretrained:      {model_path}")
    print(f"Fine-tune steps: {args.steps:,}")
    print(f"Learning rate:   {args.learning_rate:g}")
    print(f"Start anchors:   {len(YEARLY_STARTS)} yearly windows")
    print(f"Jitter days:     {args.jitter_days}")
    if eram_weights:
        print(
            "Weights:         "
            f"comfort={eram_weights['w_comfort']:.4f}, "
            f"energy={eram_weights['w_energy']:.4f}, "
            f"safety={eram_weights['w_safety']:.4f}"
        )
    print(f"Output dir:      {out_dir}")

    raw_env = EnvFactory.create(env_cfg)
    seed_env_spaces(raw_env, args.seed)
    env = Monitor(raw_env)
    print(f"Obs space:       {env.observation_space}")
    print(f"Act space:       {env.action_space}")

    clip_range = float(agent_cfg.get("ppo", {}).get("clip_range", 0.2))
    model = PPO.load(
        model_path,
        env=env,
        device=agent_cfg.get("device", "cpu"),
        custom_objects={
            "clip_range": lambda _: clip_range,
            "lr_schedule": lambda _: args.learning_rate,
        },
    )

    if model.observation_space.shape != env.observation_space.shape:
        raise RuntimeError(
            f"Model obs shape {model.observation_space.shape} does not match env "
            f"obs shape {env.observation_space.shape}. Pretrain MORL on the direct-TSup surrogate first."
        )

    model.learning_rate = args.learning_rate
    model.lr_schedule = lambda _: args.learning_rate
    model.set_random_seed(args.seed)

    callback = MORLLogCallback(str(out_dir), str(train_cfg["morl_csv_name"]))
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        reset_num_timesteps=False,
    )

    saved = maybe_save_model(model, train_cfg)
    if saved:
        print(f"Saved model: {saved}")

    try:
        env.close()
    except Exception:
        pass

    print("=" * 72)
    print("BOPTEST FINE-TUNE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
