from __future__ import annotations

import os
import random
from typing import List
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO

import numpy as np

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from training.train_ppo import build_ppo, train_ppo, maybe_save_model
from evaluation.run_baselines import run_baseline


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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


def run_one(seed: int, cfg: dict) -> None:
    _set_all_seeds(seed)

    env_cfg   = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = dict(cfg["train"])

    out_dir = f"/app/outputs/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    train_cfg["seed"]          = seed
    train_cfg["output_dir"]    = out_dir
    train_cfg["morl_csv_name"] = "morl_log.csv"
    train_cfg["save_path"]     = f"{out_dir}/models"
    os.makedirs(train_cfg["save_path"], exist_ok=True)

    if os.environ.get("STEPS"):
        train_cfg["total_timesteps"] = int(os.environ["STEPS"])

    print(f"\n=== RUN seed={seed} ===")

    run_baselines_flag = os.environ.get("RUN_BASELINES", "1").strip()
    run_baselines_flag = run_baselines_flag not in ("0", "false", "False", "no", "NO")

    if run_baselines_flag:
        env_b = EnvFactory.create(env_cfg)
        print("Action space:", env_b.action_space)
        print("Observation space:", env_b.observation_space)

        baseline_steps = int(os.environ.get("BASELINE_STEPS", "2000"))
        baseline_dir   = f"{out_dir}/baselines"
        os.makedirs(baseline_dir, exist_ok=True)

        p1 = run_baseline(env_b, name="random", n_steps=baseline_steps,
                          out_dir=baseline_dir, fixed_action=None, seed=seed)
        print("Saved baseline:", p1)

        zero = np.zeros(env_b.action_space.shape, dtype=np.float32)
        p2   = run_baseline(env_b, name="zero_hold", n_steps=baseline_steps,
                            out_dir=baseline_dir, fixed_action=zero, seed=seed)
        print("Saved baseline:", p2)

        try:
            env_b.close()
        except Exception:
            pass

    env   = EnvFactory.create(env_cfg)
    model = build_ppo(env, agent_cfg)
    train_ppo(model, train_cfg)

    saved = maybe_save_model(model, train_cfg)
    if saved:
        print("Model saved:", saved)

    try:
        env.close()
    except Exception:
        pass

    print(f"=== DONE seed={seed} ===")


def _find_model_path(seed: int) -> str:
    p = f"/app/outputs/seed{seed}/models/ppo_model.zip"
    if os.path.exists(p):
        return p
    p2 = "/app/outputs/models/ppo_model.zip"
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError("ppo_model.zip not found in /app/outputs")


def run_eval(seed: int, cfg: dict) -> None:
    env_cfg    = cfg["env"]
    model_path = _find_model_path(seed)
    n_steps    = int(os.environ.get("EVAL_STEPS", "2000"))

    out_dir = f"/app/outputs/seed{seed}/eval"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = f"{out_dir}/ppo_eval.csv"

    print("=== EVAL MODE ===")
    print("Seed:", seed)
    print("Using model:", model_path)
    print("Eval steps:", n_steps)

    env   = EnvFactory.create(env_cfg)
    model = PPO.load(model_path, device="cpu")

    obs, info = env.reset(seed=seed)
    rows = []

    for t in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_scalar, terminated, truncated, info = env.step(action)

        if (t + 1) % 25 == 0:
            print(f"[EVAL seed={seed}] {t+1}/{n_steps} reward={float(reward_scalar):.3f}")

        rv = info.get("reward_vector") if isinstance(info, dict) else None

        # Добавляем safety metric если доступна
        safety = info.get("safety", {}) if isinstance(info, dict) else {}

        rows.append({
            "step":          t,
            "reward_scalar": float(reward_scalar),
            "comfort":       rv.get("comfort")    if isinstance(rv, dict) else None,
            "energy":        rv.get("energy")     if isinstance(rv, dict) else None,
            "zone_temp":     rv.get("zone_temp")  if isinstance(rv, dict) else None,
            "hvac_power":    rv.get("hvac_power") if isinstance(rv, dict) else None,
            "w_comfort":     rv.get("w_comfort")  if isinstance(rv, dict) else None,
            "w_energy":      rv.get("w_energy")   if isinstance(rv, dict) else None,
            "a0":            float(action[0]) if hasattr(action, "__len__") else float(action),
            "a1":            float(action[1]) if hasattr(action, "__len__") and len(action) > 1 else 0.0,
            # Safety metric (Wang et al., 2024)
            "r_time":        safety.get("r_time", None),
            "r_sev":         safety.get("r_sev",  None),
            "m_s":           safety.get("m_s",    None),
            "terminated":    bool(terminated),
            "truncated":     bool(truncated),
        })

        if terminated or truncated:
            obs, info = env.reset(seed=seed)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved eval:", out_csv)

    # Финальный safety metric за весь eval
    try:
        safety_final = env.get_safety_metric()
        print(f"\n[EVAL seed={seed}] Safety metric:")
        print(f"  r_time = {safety_final['r_time']:.4f}  (доля времени нарушений)")
        print(f"  r_sev  = {safety_final['r_sev']:.4f}  (макс. отклонение)")
        print(f"  m_s    = {safety_final['m_s']:.4f}  (итоговая метрика)")
        print(f"  (для сравнения: PI=0.096, MPC=0.016, Safe DRL=0.000)")
    except AttributeError:
        pass

    try:
        env.close()
    except Exception:
        pass


# -----------------------------------------------------------------------
# PARETO MODE — новый режим Фазы 0
# -----------------------------------------------------------------------

def run_pareto(cfg: dict) -> None:
    """
    Запускает Pareto sweep: 5 конфигураций весов × N seeds.
    Результат: Pareto-фронт комфорт vs энергия + safety metric.

    Переменные окружения:
        SEEDS       — через запятую, default "42,43,44"
        STEPS       — шагов обучения, default 500000
        EVAL_STEPS  — шагов оценки, default 2000
    """
    from pareto_sweep import run_pareto_sweep
    seeds      = _parse_seeds(os.environ.get("SEEDS", "42,43,44"))
    steps      = int(os.environ.get("STEPS", "500000"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "2000"))
    run_pareto_sweep(cfg, seeds=seeds, total_timesteps=steps, eval_steps=eval_steps)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("--- Запуск HVAC_DRL_MORL ---")
    cfg  = load_all_configs("configs")
    mode = os.environ.get("MODE", "").lower().strip()

    # ===== EVAL =====
    if mode == "eval":
        seed = int(os.environ.get("SEED", "42"))
        run_eval(seed, cfg)
        return

    # ===== PARETO (новый режим Фазы 0) =====
    if mode == "pareto":
        print("=== PARETO SWEEP MODE (Фаза 0) ===")
        run_pareto(cfg)
        return

    # ===== MULTI-SEED =====
    if mode == "multiseed":
        seeds = _parse_seeds(os.environ.get("SEEDS", "42,43,44,45,46"))
        print("=== MULTI-SEED MODE ===")
        print("Seeds:", seeds)

        for sd in seeds:
            run_one(sd, cfg)

        do_eval = os.environ.get("DO_EVAL", "1").strip().lower() not in ("0", "false", "no")
        if do_eval:
            print("\n=== EVAL ALL SEEDS ===")
            for sd in seeds:
                try:
                    run_eval(sd, cfg)
                except Exception as e:
                    print(f"[WARN] Eval failed for seed={sd}: {e}")

        print("\nAll seeds finished. Outputs are in /app/outputs/seedXX/")
        return

    # ===== SINGLE SEED =====
    seed_env = os.environ.get("SEED")
    if seed_env is not None:
        seed = int(seed_env)
        run_one(seed, cfg)
        return

    # ===== DEFAULT =====
    env_cfg   = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    print("Шаг 1: Инициализация среды...")
    env = EnvFactory.create(env_cfg)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    print("Шаг 2: Инициализация PPO...")
    model = build_ppo(env, agent_cfg)

    print("Шаг 3: Обучение...")
    train_ppo(model, train_cfg)

    saved = maybe_save_model(model, train_cfg)
    if saved:
        print(f"Модель сохранена: {saved}")

    try:
        env.close()
    except Exception:
        pass

    print("Готово.")


if __name__ == "__main__":
    main()