

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from configs.loader import load_all_configs
from envs.factory import EnvFactory




def _load_ppo(model_path: str):
    """Загружает PPO модель если путь существует."""
    from stable_baselines3 import PPO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    print(f"[DATA] Loading PPO model: {model_path}")
    return PPO.load(model_path, device="cpu")


def _best_ppo_path() -> str:
    """Ищет лучшую доступную PPO модель из pareto sweep."""
    candidates = [
        "/app/outputs/pareto/comfort_dominant/seed42/models/ppo_model.zip",
        "/app/outputs/pareto/balanced/seed42/models/ppo_model.zip",
        "/app/outputs/pareto/comfort_only/seed42/models/ppo_model.zip",
        "/app/outputs/seed42/models/ppo_model.zip",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No PPO model found. Run MODE=pareto first or specify --model_path"
    )




def collect(
    policy:     str   = "mixed",
    n_steps:    int   = 10_000,
    seed:       int   = 0,
    model_path: Optional[str] = None,
    output_dir: str   = "/app/data/surrogate",
) -> str:
    """
    
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    cfg = load_all_configs("configs")
    env_cfg = cfg["env"]
    env_cfg["control_mode"] = "thermostat"

    # Для сбора данных отключаем логи
    env_cfg["output_dir"] = output_dir

    env = EnvFactory.create(env_cfg)

    # Загружаем PPO если нужно
    ppo_model = None
    if policy in ("ppo", "mixed"):
        path = model_path or _best_ppo_path()
        ppo_model = _load_ppo(path)

    print(f"[DATA] Starting collection: policy={policy}, steps={n_steps}")

    obs, info = env.reset(seed=seed)
    rows = []

    for step in range(n_steps):
        # Выбираем действие согласно политике
        if policy == "random":
            action = env.action_space.sample()

        elif policy == "ppo":
            action, _ = ppo_model.predict(obs, deterministic=False)

        elif policy == "mixed":
            # 50/50: случайно выбираем политику на каждом шаге
            if np.random.rand() < 0.5:
                action = env.action_space.sample()
            else:
                action, _ = ppo_model.predict(obs, deterministic=False)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        
        if policy != "random":
            noise = np.random.normal(0, 0.05, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        obs_prev = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)

        rv = info.get("reward_vector", {}) if isinstance(info, dict) else {}

        
        OBS_LOW  = np.array([15.0,  400.0,    0.0,   0.0])
        OBS_HIGH = np.array([35.0, 2000.0, 5000.0, 500.0])
        raw_prev = (obs_prev + 1.0) / 2.0 * (OBS_HIGH - OBS_LOW) + OBS_LOW
        raw_next = (obs      + 1.0) / 2.0 * (OBS_HIGH - OBS_LOW) + OBS_LOW

        t_zone_prev = float(raw_prev[0])
        co2_prev    = float(raw_prev[1])
        p_cool_prev = float(raw_prev[2])
        p_fan_prev  = float(raw_prev[3])
        t_zone_next = float(raw_next[0])

        p_total = float(rv.get("hvac_power", p_cool_prev + p_fan_prev))

        
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0
        t_setpoint = 21.0 + (a0 + 1.0) / 2.0 * 4.0    # [21, 25] °C
        fan_signal = float(np.clip((a1 + 1.0) / 2.0, 0.0, 1.0))

        rows.append({
            "step":        step,
            
            "t_zone":      round(t_zone_prev, 4),
            "co2":         round(co2_prev,    2),
            "p_cool":      round(p_cool_prev, 2),
            "p_fan":       round(p_fan_prev,  2),
            
            "a0_raw":      round(a0, 5),
            "a1_raw":      round(a1, 5),
            "t_setpoint":  round(t_setpoint, 3),
            "fan_signal":  round(fan_signal,  4),
            
            "t_zone_next": round(t_zone_next, 4),
            
            "p_total":     round(p_total,     2),
            
            "reward":      round(float(reward), 5),
        })

        if step % 500 == 0:
            print(f"[DATA] step={step}/{n_steps} "
                  f"T={t_zone_prev:.1f}°C "
                  f"P={p_total:.0f}W "
                  f"r={reward:.3f}")

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step)

    try:
        env.close()
    except Exception:
        pass

    
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"boptest_{policy}_{n_steps}.csv")
    df.to_csv(out_path, index=False)

    print(f"\n[DATA] Saved {len(df)} rows → {out_path}")
    print(f"[DATA] T_zone range: {df['t_zone'].min():.1f} — "
          f"{df['t_zone'].max():.1f} °C")
    print(f"[DATA] P_total range: {df['p_total'].min():.0f} — "
          f"{df['p_total'].max():.0f} W")
    print(f"[DATA] dT per step: "
          f"{(df['t_zone_next'] - df['t_zone']).abs().mean():.3f} °C/step")

    return out_path




def describe_dataset(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n{'='*55}")
    print(f"DATASET: {csv_path}")
    print(f"{'='*55}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Physical ranges:")
    for col in ["t_zone", "t_zone_next", "p_cool", "p_fan", "p_total"]:
        if col in df.columns:
            print(f"    {col:15s}: [{df[col].min():.2f}, "
                  f"{df[col].max():.2f}]  "
                  f"mean={df[col].mean():.2f}")
    print(f"\n  Action ranges:")
    for col in ["t_setpoint", "fan_signal"]:
        if col in df.columns:
            print(f"    {col:15s}: [{df[col].min():.3f}, "
                  f"{df[col].max():.3f}]  "
                  f"mean={df[col].mean():.3f}")

    
    dT = df["t_zone_next"] - df["t_zone"]
    print(f"\n  Temperature dynamics (dT = T_next - T_curr):")
    print(f"    mean dT = {dT.mean():.4f} °C/step")
    print(f"    std  dT = {dT.std():.4f} °C/step")
    print(f"    max |dT| = {dT.abs().max():.4f} °C/step")




def main():
    parser = argparse.ArgumentParser(
        description="Collect BOPTEST data for surrogate training"
    )
    parser.add_argument(
        "--policy",
        default="mixed",
        choices=["random", "ppo", "mixed"],
        help="Data collection policy"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to PPO model .zip (auto-detect if not set)"
    )
    parser.add_argument(
        "--output_dir",
        default="/app/data/surrogate",
        help="Output directory for CSV"
    )
    parser.add_argument(
        "--describe",
        default=None,
        help="Describe existing CSV dataset (skip collection)"
    )
    args = parser.parse_args()

    if args.describe:
        describe_dataset(args.describe)
        return

    out = collect(
        policy=args.policy,
        n_steps=args.steps,
        seed=args.seed,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    describe_dataset(out)


if __name__ == "__main__":
    main()
