"""
training/train_hdrl.py

Hierarchical DRL: two specialized PPO agents + meta-controller.
Both agents use direct TSup control on the tsupply-trained surrogate.

Usage:
    python training/train_hdrl.py
"""

import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def _resolve_torch_device(env_var_name: str, default: str) -> str:
    raw = os.environ.get(env_var_name, default).strip().lower()
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raw = "cpu"
    return raw


def _resolve_weather_csv(project_root: str) -> str:
    tsup = os.path.join(project_root, "data", "surrogate_v2", "boptest_v2_tsupply.csv")
    if os.path.exists(tsup):
        return tsup
    return os.path.join(project_root, "data", "surrogate_v2", "boptest_v2_all.csv")


def _seasonal_day_ranges(season: str):
    if season == "winter":
        return [[0.0, 59.0], [334.0, 365.0]]
    return [[120.0, 273.0]]


def _seasonal_shaping(season: str):
    if season == "winter":
        return {
            "deadband_c": 0.5,
            "band_bonus": 0.04,
            "undershoot_weight": 1.2,
            "overshoot_weight": 1.05,
            "cold_amb_threshold_c": 8.0,
            "hot_amb_threshold_c": 24.0,
            "cold_undershoot_weight": 1.85,
            "hot_overshoot_weight": 1.25,
            "heating_action_bonus": 0.08,
            "cooling_action_bonus": 0.02,
            "heating_t_supply_c": 29.5,
            "cooling_t_supply_c": 21.0,
            "action_fan_threshold": 0.55,
        }
    return {
        "deadband_c": 0.5,
        "band_bonus": 0.04,
        "undershoot_weight": 1.05,
        "overshoot_weight": 1.25,
        "cold_amb_threshold_c": 8.0,
        "hot_amb_threshold_c": 24.0,
        "cold_undershoot_weight": 1.2,
        "hot_overshoot_weight": 2.0,
        "heating_action_bonus": 0.02,
        "cooling_action_bonus": 0.10,
        "heating_t_supply_c": 29.0,
        "cooling_t_supply_c": 20.5,
        "action_fan_threshold": 0.6,
    }


def make_seasonal_env(env_id, rank, seed=0, season="winter"):
    def _init():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_surrogate_path = os.path.join(project_root, "outputs", "surrogate_v2", "rc_node_v3_tsupply.pt")
        surrogate_device = _resolve_torch_device("HDRL_SURROGATE_DEVICE", "cpu")
        surrogate_kind = os.environ.get("HDRL_SURROGATE_KIND", "legacy_v3")
        surrogate_summary_json = os.environ.get("HDRL_SURROGATE_SUMMARY_JSON")
        surrogate_checkpoint = os.environ.get("HDRL_SURROGATE_CHECKPOINT")
        surrogate_base_model = os.environ.get("HDRL_SURROGATE_BASE_MODEL")
        surrogate_path = os.environ.get("HDRL_SURROGATE_PATH", local_surrogate_path)

        if season == "winter":
            config = {
                "backend": "surrogate",
                "control_mode": "tsup_direct",
                "obs_mode": "extended",
                "surrogate_kind": surrogate_kind,
                "surrogate_path": surrogate_path,
                "surrogate_summary_json": surrogate_summary_json,
                "surrogate_checkpoint": surrogate_checkpoint,
                "surrogate_base_model": surrogate_base_model,
                "surrogate_device": surrogate_device,
                "weather_csv": _resolve_weather_csv(project_root),
                "domain_randomization": {
                    "enabled": True,
                    "t_init_low": 5.0,
                    "t_init_high": 20.0,
                    "weather_noise_std": 1.5,
                    "start_day_ranges": _seasonal_day_ranges("winter"),
                },
                "comfort_shaping": _seasonal_shaping("winter"),
            }
        else:
            config = {
                "backend": "surrogate",
                "control_mode": "tsup_direct",
                "obs_mode": "extended",
                "surrogate_kind": surrogate_kind,
                "surrogate_path": surrogate_path,
                "surrogate_summary_json": surrogate_summary_json,
                "surrogate_checkpoint": surrogate_checkpoint,
                "surrogate_base_model": surrogate_base_model,
                "surrogate_device": surrogate_device,
                "weather_csv": _resolve_weather_csv(project_root),
                "domain_randomization": {
                    "enabled": True,
                    "t_init_low": 20.0,
                    "t_init_high": 32.0,
                    "weather_noise_std": 1.25,
                    "start_day_ranges": _seasonal_day_ranges("summer"),
                },
                "comfort_shaping": _seasonal_shaping("summer"),
            }

        from envs.factory import EnvFactory
        from gymnasium.wrappers import TimeLimit

        env = EnvFactory.create(config)
        env = TimeLimit(env, max_episode_steps=336)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    return _init


def train_agent(season, total_timesteps=5_000_000, num_envs=16):
    policy_device = _resolve_torch_device("HDRL_POLICY_DEVICE", "auto")
    print(f"\n{'=' * 60}")
    print(f"TRAINING {season.upper()} AGENT")
    print(f"{'=' * 60}")
    print(f"  Policy device: {policy_device}")
    print(f"  Surrogate device: {_resolve_torch_device('HDRL_SURROGATE_DEVICE', 'cpu')}")

    vec_env = SubprocVecEnv([make_seasonal_env("HVAC", i, seed=42, season=season) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        device=policy_device,
        verbose=1,
        tensorboard_log=f"./logs/hdrl_{season}_tsup/",
    )

    save_path = f"./models/ppo_{season}_final"
    os.makedirs("./models/", exist_ok=True)

    t_start = time.time()
    print(f"  Steps: {total_timesteps:,}, Envs: {num_envs}")

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    vec_env.close()

    elapsed = (time.time() - t_start) / 60
    print(f"  {season.upper()} trained in {elapsed:.1f} min -> {save_path}.zip")
    return save_path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Train HDRL winter/summer agents on the direct-TSup surrogate.")
    parser.add_argument("--winter-steps", type=int, default=int(os.environ.get("WINTER_TIMESTEPS", "5000000")))
    parser.add_argument("--summer-steps", type=int, default=int(os.environ.get("SUMMER_TIMESTEPS", "7000000")))
    parser.add_argument("--num-envs", type=int, default=int(os.environ.get("HDRL_ENVS", "16")))
    parser.add_argument("--surrogate-kind", choices=["legacy_v3", "v35_raw", "v35_calibrated"], default=os.environ.get("HDRL_SURROGATE_KIND", "legacy_v3"))
    parser.add_argument("--surrogate-path", default=os.environ.get("HDRL_SURROGATE_PATH"))
    parser.add_argument("--surrogate-summary-json", default=os.environ.get("HDRL_SURROGATE_SUMMARY_JSON"))
    parser.add_argument("--surrogate-checkpoint", default=os.environ.get("HDRL_SURROGATE_CHECKPOINT"))
    parser.add_argument("--surrogate-base-model", default=os.environ.get("HDRL_SURROGATE_BASE_MODEL"))
    args = parser.parse_args()

    os.environ["HDRL_SURROGATE_KIND"] = args.surrogate_kind
    if args.surrogate_path:
        os.environ["HDRL_SURROGATE_PATH"] = args.surrogate_path
    if args.surrogate_summary_json:
        os.environ["HDRL_SURROGATE_SUMMARY_JSON"] = args.surrogate_summary_json
    if args.surrogate_checkpoint:
        os.environ["HDRL_SURROGATE_CHECKPOINT"] = args.surrogate_checkpoint
    if args.surrogate_base_model:
        os.environ["HDRL_SURROGATE_BASE_MODEL"] = args.surrogate_base_model

    print("HIERARCHICAL DRL TRAINING (direct TSup)")
    t_total = time.time()

    winter_path = train_agent("winter", total_timesteps=int(args.winter_steps), num_envs=int(args.num_envs))
    summer_path = train_agent("summer", total_timesteps=int(args.summer_steps), num_envs=int(args.num_envs))

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'=' * 60}")
    print(f"HDRL TRAINING COMPLETE ({elapsed:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Surrogate kind: {args.surrogate_kind}")
    if args.surrogate_summary_json:
        print(f"  V3.5 summary: {args.surrogate_summary_json}")
    print(f"  Winter: {winter_path}")
    print(f"  Summer: {summer_path}")
    print("  Observation: 17D extended TSup features (time + forecast + action history)")
    print("  Meta-controller eval path: hysteresis gate + emergency heating")


if __name__ == "__main__":
    main()
