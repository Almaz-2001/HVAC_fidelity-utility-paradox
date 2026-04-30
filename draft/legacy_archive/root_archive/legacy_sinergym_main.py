from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from evaluation.run_baselines import RuleBasedSinergymPolicy, run_baseline
from pareto_sweep import run_pareto_sweep
from draft.legacy_archive.code_archive.bc_warmstart import maybe_run_rbc_warmstart
from training.train_ppo import build_ppo, maybe_save_model, train_ppo


DEFAULT_CONFIG_DIR = "configs/legacy_sinergym"


def _parse_seeds(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in raw]


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _base_output_dir(cfg: dict) -> str:
    return str(cfg["env"].get("output_dir", "/app/outputs/legacy_sinergym"))


def _find_model_path(base_output: str, seed: int) -> str:
    model_path = Path(base_output) / f"seed{seed}" / "models" / "ppo_model.zip"
    if model_path.exists():
        return str(model_path)
    raise FileNotFoundError(f"Legacy Sinergym model not found: {model_path}")


def run_one(seed: int, cfg: dict, baseline_steps: int = 2000, run_baselines_flag: bool = True) -> str:
    _set_all_seeds(seed)

    env_cfg = dict(cfg["env"])
    agent_cfg = dict(cfg["agent"])
    train_cfg = dict(cfg["train"])

    base_output = _base_output_dir(cfg)
    out_dir = Path(base_output) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg["seed"] = seed
    train_cfg["output_dir"] = str(out_dir)
    train_cfg["morl_csv_name"] = "morl_log.csv"
    train_cfg["save_path"] = str(out_dir / "models")
    Path(train_cfg["save_path"]).mkdir(parents=True, exist_ok=True)

    print(f"\n=== LEGACY SINERGYM RUN seed={seed} ===")
    print(f"Config dir: {cfg['_config_dir']}")
    print(f"Output dir: {out_dir}")

    if run_baselines_flag:
        run_baselines_only(seed=seed, cfg=cfg, baseline_steps=baseline_steps)

    env = EnvFactory.create(env_cfg)
    model = build_ppo(env, agent_cfg)
    bc_summary = maybe_run_rbc_warmstart(
        model,
        env_factory=EnvFactory,
        env_cfg=env_cfg,
        warm_cfg=train_cfg.get("bc_warmstart", {}) or {},
        seed=seed,
        out_dir=out_dir,
    )
    if bc_summary is not None:
        print("BC warm-start complete:")
        print(f"  dataset: {bc_summary.get('dataset_csv')}")
        print(f"  final_bc_loss: {bc_summary.get('final_bc_loss')}")
    train_ppo(model, train_cfg)
    saved = maybe_save_model(model, train_cfg)
    env.close()

    if saved:
        print("Model saved:", saved)
    return saved or str(Path(train_cfg["save_path"]) / "ppo_model.zip")


def run_eval(seed: int, cfg: dict, eval_steps: int = 2000) -> str:
    base_output = _base_output_dir(cfg)
    model_path = _find_model_path(base_output, seed)
    out_dir = Path(base_output) / f"seed{seed}" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ppo_eval.csv"

    print("\n=== LEGACY SINERGYM EVAL ===")
    print("Seed:", seed)
    print("Model:", model_path)
    print("Eval steps:", eval_steps)

    env = EnvFactory.create(dict(cfg["env"]))
    model = PPO.load(model_path, device="cpu")
    obs, info = env.reset(seed=seed)

    def _action_component(container, idx: int) -> float:
        if container is None:
            return np.nan
        try:
            arr = np.asarray(container, dtype=np.float32).reshape(-1)
            return float(arr[idx]) if idx < arr.size else np.nan
        except Exception:
            return np.nan

    rows = []
    for step in range(int(eval_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_scalar, terminated, truncated, info = env.step(action)
        info = dict(info or {})
        rv = info.get("reward_vector") if isinstance(info, dict) else None

        rows.append(
            {
                "step": step,
                "reward_scalar": float(reward_scalar),
                "comfort": rv.get("comfort") if isinstance(rv, dict) else None,
                "energy": rv.get("energy") if isinstance(rv, dict) else None,
                "zone_temp": rv.get("zone_temp") if isinstance(rv, dict) else None,
                "hvac_power": rv.get("hvac_power") if isinstance(rv, dict) else None,
                "zone_temp_from_obs": info.get("zone_temp_from_obs"),
                "zone_temp_from_info": info.get("zone_temp_from_info"),
                "zone_temp_source": info.get("zone_temp_source"),
                "hvac_power_from_obs": info.get("hvac_power_from_obs"),
                "hvac_power_from_info": info.get("hvac_power_from_info"),
                "hvac_power_source": info.get("hvac_power_source"),
                "raw_a0": _action_component(info.get("action_raw"), 0),
                "raw_a1": _action_component(info.get("action_raw"), 1),
                "limited_a0": _action_component(info.get("action_rate_limited"), 0),
                "limited_a1": _action_component(info.get("action_rate_limited"), 1),
                "physical_pre_safety_a0": _action_component(info.get("action_physical_pre_safety"), 0),
                "physical_pre_safety_a1": _action_component(info.get("action_physical_pre_safety"), 1),
                "physical_final_a0": _action_component(info.get("action_physical_final"), 0),
                "physical_final_a1": _action_component(info.get("action_physical_final"), 1),
                "w_comfort": rv.get("w_comfort") if isinstance(rv, dict) else None,
                "w_energy": rv.get("w_energy") if isinstance(rv, dict) else None,
                "a0": float(action[0]) if hasattr(action, "__len__") else float(action),
                "a1": float(action[1]) if hasattr(action, "__len__") and len(action) > 1 else np.nan,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

    env.close()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved eval:", out_csv)
    return str(out_csv)


def run_multiseed(
    seeds: list[int],
    cfg: dict,
    baseline_steps: int = 2000,
    eval_steps: int = 2000,
    do_eval: bool = True,
    run_baselines_flag: bool = True,
) -> None:
    for seed in seeds:
        run_one(seed, cfg, baseline_steps=baseline_steps, run_baselines_flag=run_baselines_flag)

    if do_eval:
        for seed in seeds:
            try:
                run_eval(seed, cfg, eval_steps=eval_steps)
            except Exception as exc:
                print(f"[WARN] Eval failed for seed={seed}: {exc}")


def run_baselines_only(seed: int, cfg: dict, baseline_steps: int = 2000) -> None:
    env_cfg = dict(cfg["env"])
    base_output = _base_output_dir(cfg)
    out_dir = Path(base_output) / f"seed{seed}" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LEGACY SINERGYM BASELINES seed={seed} ===")
    print(f"Output dir: {out_dir}")

    env_b = EnvFactory.create(env_cfg)

    p_random = run_baseline(env_b, name="random", n_steps=baseline_steps, out_dir=out_dir, fixed_action=None, seed=seed)
    print("Saved baseline:", p_random)

    rule_policy = RuleBasedSinergymPolicy(env_b)
    p_rule = run_baseline(
        env_b,
        name="rule_based",
        n_steps=baseline_steps,
        out_dir=out_dir,
        action_fn=rule_policy,
        seed=seed,
    )
    print("Saved baseline:", p_rule)

    zero = np.zeros(env_b.action_space.shape, dtype=np.float32)
    p_zero = run_baseline(env_b, name="zero_hold", n_steps=baseline_steps, out_dir=out_dir, fixed_action=zero, seed=seed)
    print("Saved baseline:", p_zero)
    env_b.close()


def run_pareto(cfg: dict, seeds: list[int], total_timesteps: int, eval_steps: int) -> None:
    run_pareto_sweep(cfg, seeds=seeds, total_timesteps=total_timesteps, eval_steps=eval_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy Sinergym MORL-PPO launcher for ArticleRus reproduction")
    parser.add_argument("--config-dir", default=os.environ.get("CONFIG_DIR", DEFAULT_CONFIG_DIR))
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "multiseed", "pareto", "baselines"],
        default=os.environ.get("MODE", "train"),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", "42,43,44"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "500000")))
    parser.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "2000")))
    parser.add_argument("--baseline-steps", type=int, default=int(os.environ.get("BASELINE_STEPS", "2000")))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--temp-index", type=int, default=None)
    parser.add_argument("--energy-index", type=int, default=None)
    parser.add_argument("--warmstart-rbc", action="store_true")
    parser.add_argument("--warmstart-steps", type=int, default=None)
    parser.add_argument("--warmstart-epochs", type=int, default=None)
    parser.add_argument("--warmstart-batch-size", type=int, default=None)
    parser.add_argument("--warmstart-lr", type=float, default=None)
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    env_cfg = cfg.get("env", {})
    if args.output_dir:
        env_cfg["output_dir"] = args.output_dir
    morl_cfg = env_cfg.get("morl", {})
    if args.temp_index is not None:
        morl_cfg["temp_index"] = int(args.temp_index)
    if args.energy_index is not None:
        morl_cfg["energy_index"] = int(args.energy_index)
    env_cfg["morl"] = morl_cfg
    cfg["env"] = env_cfg

    train_cfg = cfg.get("train", {})
    warm_cfg = train_cfg.get("bc_warmstart", {}) or {}
    if args.warmstart_rbc:
        warm_cfg["enabled"] = True
    if args.warmstart_steps is not None:
        warm_cfg["steps"] = int(args.warmstart_steps)
        warm_cfg["enabled"] = True
    if args.warmstart_epochs is not None:
        warm_cfg["epochs"] = int(args.warmstart_epochs)
        warm_cfg["enabled"] = True
    if args.warmstart_batch_size is not None:
        warm_cfg["batch_size"] = int(args.warmstart_batch_size)
        warm_cfg["enabled"] = True
    if args.warmstart_lr is not None:
        warm_cfg["learning_rate"] = float(args.warmstart_lr)
        warm_cfg["enabled"] = True
    train_cfg["bc_warmstart"] = warm_cfg
    cfg["train"] = train_cfg

    agent_cfg = cfg.get("agent", {})
    ppo_cfg = agent_cfg.get("ppo", {}) or {}
    if args.ppo_learning_rate is not None:
        ppo_cfg["learning_rate"] = float(args.ppo_learning_rate)
    agent_cfg["ppo"] = ppo_cfg
    cfg["agent"] = agent_cfg
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_all_configs(args.config_dir)
    cfg["_config_dir"] = args.config_dir
    cfg = apply_overrides(cfg, args)

    seeds = _parse_seeds(args.seeds)

    if args.mode == "train":
        run_one(
            seed=args.seed,
            cfg=cfg,
            baseline_steps=args.baseline_steps,
            run_baselines_flag=not args.skip_baselines,
        )
        return

    if args.mode == "eval":
        run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
        return

    if args.mode == "multiseed":
        run_multiseed(
            seeds=seeds,
            cfg=cfg,
            baseline_steps=args.baseline_steps,
            eval_steps=args.eval_steps,
            do_eval=not args.skip_eval,
            run_baselines_flag=not args.skip_baselines,
        )
        return

    if args.mode == "baselines":
        for seed in seeds:
            run_baselines_only(seed=seed, cfg=cfg, baseline_steps=args.baseline_steps)
        return

    if args.mode == "pareto":
        run_pareto(cfg=cfg, seeds=seeds, total_timesteps=args.steps, eval_steps=args.eval_steps)
        return


if __name__ == "__main__":
    main()
