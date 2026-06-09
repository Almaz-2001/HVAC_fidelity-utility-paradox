

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, List

from configs.loader import load_all_configs
from envs.factory import EnvFactory




def random_policy(obs: np.ndarray, info: dict) -> np.ndarray:
    """Случайные действия — максимальное покрытие."""
    return np.random.uniform(-1, 1, size=2).astype(np.float32)


def comfort_greedy_policy(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    
    """
    # obs[0] — t_zone_norm в [-1, 1], где -1=15°C, +1=35°C
    t_zone_norm = obs[0] if len(obs) > 0 else 0.0
    t_zone = 15.0 + (t_zone_norm + 1.0) / 2.0 * 20.0  # denorm

    # Целевая: если холодно → греть сильнее, если жарко → меньше
    error = 22.0 - t_zone
    t_target = np.clip(22.0 + 0.5 * error, 21.0, 25.0)
    a0 = 2.0 * (t_target - 21.0) / 4.0 - 1.0

    # Умеренный вентилятор
    a1 = np.random.uniform(-0.3, 0.3)
    return np.array([np.clip(a0, -1, 1), np.clip(a1, -1, 1)], dtype=np.float32)


def energy_saving_policy(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    
    """
    return np.array([-1.0, -1.0], dtype=np.float32)


def max_heating_policy(obs: np.ndarray, info: dict) -> np.ndarray:
    """
    
    """
    return np.array([1.0, 1.0], dtype=np.float32)


POLICY_REGISTRY = {
    "random":          random_policy,
    "comfort_greedy":  comfort_greedy_policy,
    "energy_saving":   energy_saving_policy,
    "max_heating":     max_heating_policy,
}


# -----------------------------------------------------------------------
# Извлечение T_amb и временных признаков
# -----------------------------------------------------------------------

def extract_t_amb(info: dict, default: float = 10.0) -> float:
    """
    
    """
    # Прямой доступ через info
    for key in ['TDryBul', 'T_amb', 'weaBus.TDryBul', 'TOut']:
        if key in info:
            val = info[key]
            if isinstance(val, dict):
                val = val.get('value', val.get('y', default))
            # BOPTEST может возвращать в Kelvin
            val = float(val)
            if val > 200:  # вероятно Kelvin
                val -= 273.15
            return val

    # Через вложенные measurements
    meas = info.get('measurements', {})
    for key in ['TDryBul_y', 'weaTDryBul_y', 'TOut_y']:
        if key in meas:
            val = float(meas[key])
            return val - 273.15 if val > 200 else val

    return default


def compute_time_features(step: int, dt: float = 3600.0,
                          start_time: float = 0.0) -> dict:
    """
    
    """
    total_seconds = start_time + step * dt
    hour = (total_seconds / 3600.0) % 24.0
    day  = (total_seconds / 86400.0) % 365.0
    return {'hour': round(hour, 2), 'day': round(day, 2)}






SEASON_START_TIMES = {
    'winter':  0,              # 1 января
    'spring':  90 * 86400,     # ~1 апреля
    'summer':  180 * 86400,    # ~1 июля
    'autumn':  270 * 86400,    # ~1 октября
}






def collect_v2(
    policy:             str   = "random",
    n_steps:            int   = 3200,
    seed:               int   = 0,
    model_path:         Optional[str] = None,
    output_dir:         str   = "/app/data/surrogate_v2",
    start_time:         float = 0.0,
    season:             str   = "winter",
) -> pd.DataFrame:
    """
    Собирает один эпизод данных v2.

    Returns:
        DataFrame с собранными данными
    """
    np.random.seed(seed)

    cfg = load_all_configs("configs")
    env_cfg = cfg["env"]
    env_cfg["control_mode"] = "thermostat"
    env_cfg["output_dir"] = output_dir

    # Устанавливаем начальное время если backend поддерживает
    env_cfg.setdefault("start_time", start_time)

    env = EnvFactory.create(env_cfg)

    # Выбираем политику
    if policy in POLICY_REGISTRY:
        policy_fn = POLICY_REGISTRY[policy]
    elif policy == "mixed":
        # 50% random + 50% PPO (как в v1)
        from stable_baselines3 import PPO
        path = model_path or _find_ppo_model()
        ppo_model = PPO.load(path, device="cpu")
        def policy_fn(obs, info):
            if np.random.rand() < 0.5:
                return env.action_space.sample()
            else:
                action, _ = ppo_model.predict(obs, deterministic=False)
                return action
    else:
        raise ValueError(f"Unknown policy: {policy}. "
                         f"Available: {list(POLICY_REGISTRY.keys()) + ['mixed']}")

    print(f"[DATA_V2] Collecting: policy={policy}, season={season}, "
          f"steps={n_steps}, start_time={start_time:.0f}s")

    obs, info = env.reset(seed=seed)
    rows = []

    
    OBS_LOW  = np.array([15.0,  400.0,    0.0,   0.0])
    OBS_HIGH = np.array([35.0, 2000.0, 5000.0, 500.0])

    for step in range(n_steps):
        # Действие
        action = policy_fn(obs, info)
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        # Текущее состояние
        raw = (obs + 1.0) / 2.0 * (OBS_HIGH - OBS_LOW) + OBS_LOW
        t_zone_prev = float(raw[0])

        # T_amb из info
        t_amb = info.get('t_amb', 10.0) if isinstance(info, dict) else 10.0

        # Временные признаки
        time_feat = compute_time_features(step, dt=3600.0,
                                          start_time=start_time)

        # Шаг
        obs_prev = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)

        # Следующее состояние
        raw_next = (obs + 1.0) / 2.0 * (OBS_HIGH - OBS_LOW) + OBS_LOW
        t_zone_next = float(raw_next[0])

        # Мощность
        rv = info.get("reward_vector", {}) if isinstance(info, dict) else {}
        p_total = float(rv.get("hvac_power", float(raw[2]) + float(raw[3])))

        # Действия
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        rows.append({
            "step":        step,
            # Состояние (расширенное)
            "t_zone":      round(t_zone_prev, 4),
            "t_amb":       round(t_amb, 2),           # НОВОЕ
            "hour":        time_feat['hour'],          # НОВОЕ
            "day":         time_feat['day'],            # НОВОЕ
            # Действия
            "a0_raw":      round(a0, 5),
            "a1_raw":      round(a1, 5),
            # Цели
            "t_zone_next": round(t_zone_next, 4),
            "delta_t":     round(t_zone_next - t_zone_prev, 4),  # НОВОЕ
            "p_total":     round(p_total, 2),
            # Мета-информация
            "policy":      policy,
            "season":      season,
            "reward":      round(float(reward), 5),
        })

        if step % 500 == 0:
            print(f"  step={step}/{n_steps} "
                  f"T_zone={t_zone_prev:.1f}°C "
                  f"T_amb={t_amb:.1f}°C "
                  f"P={p_total:.0f}W "
                  f"ΔT={t_zone_next - t_zone_prev:+.2f}°C")

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step)

    try:
        env.close()
    except Exception:
        pass

    return pd.DataFrame(rows)






def collect_all(
    steps_per_episode: int = 3200,
    output_dir:        str = "/app/data/surrogate_v2",
    base_seed:         int = 42,
    policies:          Optional[List[str]] = None,
    seasons:           Optional[List[str]] = None,
) -> str:
    """
    
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if policies is None:
        policies = list(POLICY_REGISTRY.keys())
    if seasons is None:
        seasons = list(SEASON_START_TIMES.keys())

    all_dfs = []
    episode_idx = 0

    print(f"\n{'='*60}")
    print(f"SURROGATE V2 DATA COLLECTION")
    print(f"{'='*60}")
    print(f"  Policies:          {policies}")
    print(f"  Seasons:           {seasons}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Total episodes:    {len(policies) * len(seasons)}")
    print(f"  Expected total:    {len(policies) * len(seasons) * steps_per_episode:,} steps")

    for policy in policies:
        for season in seasons:
            start_time = SEASON_START_TIMES[season]
            seed = base_seed + episode_idx

            df = collect_v2(
                policy=policy,
                n_steps=steps_per_episode,
                seed=seed,
                output_dir=output_dir,
                start_time=start_time,
                season=season,
            )
            all_dfs.append(df)
            episode_idx += 1

    # Объединяем
    combined = pd.concat(all_dfs, ignore_index=True)

    # Сохраняем
    out_path = os.path.join(output_dir, "boptest_v2_all.csv")
    combined.to_csv(out_path, index=False)

    # Статистика
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Saved: {out_path}")
    print(f"\n  Ranges:")
    for col in ['t_zone', 't_amb', 'delta_t', 'p_total']:
        if col in combined.columns:
            print(f"    {col:12s}: [{combined[col].min():.2f}, "
                  f"{combined[col].max():.2f}]  "
                  f"mean={combined[col].mean():.2f}")
    print(f"\n  By policy:")
    for p in policies:
        subset = combined[combined['policy'] == p]
        print(f"    {p:18s}: {len(subset):,} rows, "
              f"mean |ΔT|={subset['delta_t'].abs().mean():.3f}°C")
    print(f"\n  By season:")
    for s in seasons:
        subset = combined[combined['season'] == s]
        print(f"    {s:10s}: {len(subset):,} rows, "
              f"T_amb mean={subset['t_amb'].mean():.1f}°C")

    return out_path




def _find_ppo_model() -> str:
    candidates = [
        "/app/outputs/pareto/comfort_dominant/seed42/models/ppo_model.zip",
        "/app/outputs/pareto/balanced/seed42/models/ppo_model.zip",
        "/app/outputs/seed42/models/ppo_model.zip",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No PPO model found. Specify --model_path")




def main():
    parser = argparse.ArgumentParser(
        description="Collect BOPTEST data v2 (with T_amb and time features)"
    )
    parser.add_argument(
        "--policy", default="all",
        help="Policy: random|comfort_greedy|energy_saving|max_heating|mixed|all"
    )
    parser.add_argument("--steps_per_episode", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/app/data/surrogate_v2")
    parser.add_argument("--model_path", default=None,
                        help="Path to PPO model for 'mixed' policy")
    args = parser.parse_args()

    if args.policy == "all":
        collect_all(
            steps_per_episode=args.steps_per_episode,
            output_dir=args.output_dir,
            base_seed=args.seed,
        )
    else:
        df = collect_v2(
            policy=args.policy,
            n_steps=args.steps_per_episode,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        out_path = os.path.join(args.output_dir,
                                f"boptest_v2_{args.policy}.csv")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
