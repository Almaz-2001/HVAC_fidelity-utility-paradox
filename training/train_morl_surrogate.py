from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from envs.tsup_features import SUPPORTED_POWER_FEATURE_MODES, SUPPORTED_T_ZONE_FEATURE_MODES
from training.train_ppo import build_ppo, maybe_save_model, train_ppo


def parse_morl_weights(text: str | None) -> tuple[float, float, float] | None:
    if not text:
        return None
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("MORL weights must have three comma-separated values: comfort,energy,safety")
    arr = np.asarray(parts, dtype=np.float64)
    if np.any(arr < 0.0) or arr.sum() <= 0.0:
        raise ValueError("MORL weights must be non-negative and not all zero.")
    arr = arr / arr.sum()
    return float(arr[0]), float(arr[1]), float(arr[2])


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
    try:
        env.reset(seed=seed)
    except Exception:
        pass
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MORL PPO on the direct-TSup surrogate backend.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing env.yaml, agent.yaml, train.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--surrogate-kind",
        choices=["legacy_v3", "v35_raw", "v35_calibrated", "hybrid_v3_v35"],
        default=None,
        help="Select which direct-TSup surrogate implementation to use.",
    )
    parser.add_argument("--surrogate-path", default=None, help="Legacy v3 base model path.")
    parser.add_argument("--surrogate-summary-json", default=None, help="Calibration summary JSON for v3.5 adapters.")
    parser.add_argument("--surrogate-checkpoint", default=None, help="Explicit v3.5 staged checkpoint path.")
    parser.add_argument("--surrogate-base-model", default=None, help="Explicit base v3 TSup checkpoint for v3.5 adapters.")
    parser.add_argument("--lambda-temp-disagree", type=float, default=None)
    parser.add_argument("--lambda-power-disagree", type=float, default=None)
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default=None)
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default=None)
    parser.add_argument("--morl-weights", default=None, help="Fixed MORL weights as comfort,energy,safety.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    cfg = load_all_configs(args.config_dir)
    env_cfg = dict(cfg["env"])
    agent_cfg = dict(cfg["agent"])
    train_cfg = dict(cfg["train"])

    env_cfg["backend"] = "surrogate"
    env_cfg["control_mode"] = "tsup_direct"
    if args.surrogate_kind is not None:
        env_cfg["surrogate_kind"] = args.surrogate_kind
    if args.surrogate_path is not None:
        env_cfg["surrogate_path"] = args.surrogate_path
    if args.surrogate_summary_json is not None:
        env_cfg["surrogate_summary_json"] = args.surrogate_summary_json
    if args.surrogate_checkpoint is not None:
        env_cfg["surrogate_checkpoint"] = args.surrogate_checkpoint
    if args.surrogate_base_model is not None:
        env_cfg["surrogate_base_model"] = args.surrogate_base_model
    if args.lambda_temp_disagree is not None:
        env_cfg["lambda_temp_disagree"] = float(args.lambda_temp_disagree)
    if args.lambda_power_disagree is not None:
        env_cfg["lambda_power_disagree"] = float(args.lambda_power_disagree)
    if args.power_feature_mode is not None:
        env_cfg["power_feature_mode"] = args.power_feature_mode
    if args.t_zone_feature_mode is not None:
        env_cfg["t_zone_feature_mode"] = args.t_zone_feature_mode
    morl_weights = parse_morl_weights(args.morl_weights)
    if morl_weights is not None:
        env_cfg["morl"] = dict(env_cfg.get("morl", {}))
        env_cfg["morl"].update(
            {
                "w_comfort": morl_weights[0],
                "w_energy": morl_weights[1],
                "w_safety": morl_weights[2],
            }
        )

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
        "config_dir": str(Path(args.config_dir).resolve()),
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("=" * 72)
    print("MORL SURROGATE PRETRAIN")
    print("=" * 72)
    print(f"Seed:       {args.seed}")
    print(f"Steps:      {args.steps:,}")
    print(f"Backend:    {env_cfg['backend']}")
    print(f"Control:    {env_cfg['control_mode']}")
    print(f"Surrogate:  {env_cfg.get('surrogate_kind', 'legacy_v3')}")
    if env_cfg.get("surrogate_path"):
        print(f"V3 path:    {env_cfg.get('surrogate_path')}")
    if env_cfg.get("surrogate_summary_json"):
        print(f"V3.5 sum:   {env_cfg.get('surrogate_summary_json')}")
    if env_cfg.get("surrogate_kind") == "hybrid_v3_v35":
        print(
            "Hybrid:     "
            f"lambda_temp={float(env_cfg.get('lambda_temp_disagree', 0.0)):.3f}, "
            f"lambda_power={float(env_cfg.get('lambda_power_disagree', 0.0)):.1e}, "
            f"power_mode={env_cfg.get('power_feature_mode', 'raw')}, "
            f"t_zone_mode={env_cfg.get('t_zone_feature_mode', 'raw')}"
        )
    morl_cfg = env_cfg.get("morl", {})
    print(
        "MORL w:    "
        f"comfort={float(morl_cfg.get('w_comfort', 0.0)):.3f}, "
        f"energy={float(morl_cfg.get('w_energy', 0.0)):.3f}, "
        f"safety={float(morl_cfg.get('w_safety', 0.0)):.3f}"
    )
    print(f"Output dir: {out_dir}")

    env = EnvFactory.create(env_cfg)
    seed_env_spaces(env, args.seed)
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
