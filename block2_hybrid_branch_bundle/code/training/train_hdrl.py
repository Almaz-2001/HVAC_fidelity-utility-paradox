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

from envs.tsup_features import (
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    SUPPORTED_T_ZONE_FEATURE_MODES,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

DEFAULT_STEP_SEC = 900
DEFAULT_EPISODE_DAYS = 14.0
DEFAULT_TEMP_LOW_C = 21.0
DEFAULT_TEMP_HIGH_C = 24.0
DEFAULT_SURROGATE_KIND = "legacy_v3"
DEFAULT_OBS_ABLATION = "none"
DEFAULT_DELTA_FEATURE_MODE = "raw"
DEFAULT_POWER_FEATURE_MODE = "raw"
DEFAULT_T_ZONE_FEATURE_MODE = "raw"
DEFAULT_LAMBDA_TEMP_DISAGREE = 0.10
DEFAULT_LAMBDA_POWER_DISAGREE = 5.0e-5


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


def make_seasonal_env(
    env_id,
    rank,
    seed=0,
    season="winter",
    step_sec: int = DEFAULT_STEP_SEC,
    episode_days: float = DEFAULT_EPISODE_DAYS,
    temp_low_c: float = DEFAULT_TEMP_LOW_C,
    temp_high_c: float = DEFAULT_TEMP_HIGH_C,
):
    def _init():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_surrogate_path = os.path.join(project_root, "outputs", "surrogate_v2", "rc_node_v3_tsupply.pt")
        surrogate_device = _resolve_torch_device("HDRL_SURROGATE_DEVICE", "cpu")
        surrogate_kind = os.environ.get("HDRL_SURROGATE_KIND", "legacy_v3")
        surrogate_summary_json = os.environ.get("HDRL_SURROGATE_SUMMARY_JSON")
        surrogate_checkpoint = os.environ.get("HDRL_SURROGATE_CHECKPOINT")
        surrogate_base_model = os.environ.get("HDRL_SURROGATE_BASE_MODEL")
        surrogate_path = os.environ.get("HDRL_SURROGATE_PATH", local_surrogate_path)
        obs_ablation = os.environ.get("HDRL_OBS_ABLATION", DEFAULT_OBS_ABLATION)
        delta_feature_mode = os.environ.get("HDRL_DELTA_FEATURE_MODE", DEFAULT_DELTA_FEATURE_MODE)
        power_feature_mode = os.environ.get("HDRL_POWER_FEATURE_MODE", DEFAULT_POWER_FEATURE_MODE)
        t_zone_feature_mode = os.environ.get("HDRL_T_ZONE_FEATURE_MODE", DEFAULT_T_ZONE_FEATURE_MODE)
        lambda_temp_disagree = float(
            os.environ.get("HDRL_LAMBDA_TEMP_DISAGREE", str(DEFAULT_LAMBDA_TEMP_DISAGREE))
        )
        lambda_power_disagree = float(
            os.environ.get("HDRL_LAMBDA_POWER_DISAGREE", str(DEFAULT_LAMBDA_POWER_DISAGREE))
        )

        if season == "winter":
            config = {
                "backend": "surrogate",
                "control_mode": "tsup_direct",
                "obs_mode": "extended",
                "step_sec": int(step_sec),
                "max_episode_steps": int(round(float(episode_days) * 86400.0 / float(step_sec))),
                "surrogate_kind": surrogate_kind,
                "surrogate_path": surrogate_path,
                "surrogate_summary_json": surrogate_summary_json,
                "surrogate_checkpoint": surrogate_checkpoint,
                "surrogate_base_model": surrogate_base_model,
                "surrogate_device": surrogate_device,
                "weather_csv": _resolve_weather_csv(project_root),
                "obs_ablation": obs_ablation,
                "delta_feature_mode": delta_feature_mode,
                "power_feature_mode": power_feature_mode,
                "t_zone_feature_mode": t_zone_feature_mode,
                "lambda_temp_disagree": lambda_temp_disagree,
                "lambda_power_disagree": lambda_power_disagree,
                "morl": {
                    "temp_low": float(temp_low_c),
                    "temp_high": float(temp_high_c),
                },
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
                "step_sec": int(step_sec),
                "max_episode_steps": int(round(float(episode_days) * 86400.0 / float(step_sec))),
                "surrogate_kind": surrogate_kind,
                "surrogate_path": surrogate_path,
                "surrogate_summary_json": surrogate_summary_json,
                "surrogate_checkpoint": surrogate_checkpoint,
                "surrogate_base_model": surrogate_base_model,
                "surrogate_device": surrogate_device,
                "weather_csv": _resolve_weather_csv(project_root),
                "obs_ablation": obs_ablation,
                "delta_feature_mode": delta_feature_mode,
                "power_feature_mode": power_feature_mode,
                "t_zone_feature_mode": t_zone_feature_mode,
                "lambda_temp_disagree": lambda_temp_disagree,
                "lambda_power_disagree": lambda_power_disagree,
                "morl": {
                    "temp_low": float(temp_low_c),
                    "temp_high": float(temp_high_c),
                },
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
        env = TimeLimit(env, max_episode_steps=int(round(float(episode_days) * 86400.0 / float(step_sec))))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    return _init


def train_agent(
    season,
    total_timesteps=5_000_000,
    num_envs=16,
    step_sec: int = DEFAULT_STEP_SEC,
    episode_days: float = DEFAULT_EPISODE_DAYS,
    temp_low_c: float = DEFAULT_TEMP_LOW_C,
    temp_high_c: float = DEFAULT_TEMP_HIGH_C,
    save_prefix: str = "ppo",
):
    policy_device = _resolve_torch_device("HDRL_POLICY_DEVICE", "auto")
    print(f"\n{'=' * 60}")
    print(f"TRAINING {season.upper()} AGENT")
    print(f"{'=' * 60}")
    print(f"  Policy device: {policy_device}")
    print(f"  Surrogate device: {_resolve_torch_device('HDRL_SURROGATE_DEVICE', 'cpu')}")
    print(f"  Step: {int(step_sec)} s | Episode: {float(episode_days):.1f} days")
    print(f"  Comfort band: [{float(temp_low_c):.1f}, {float(temp_high_c):.1f}] C")
    print(f"  Surrogate kind: {os.environ.get('HDRL_SURROGATE_KIND', DEFAULT_SURROGATE_KIND)}")

    vec_env = SubprocVecEnv(
        [
            make_seasonal_env(
                "HVAC",
                i,
                seed=42,
                season=season,
                step_sec=step_sec,
                episode_days=episode_days,
                temp_low_c=temp_low_c,
                temp_high_c=temp_high_c,
            )
            for i in range(num_envs)
        ]
    )
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

    save_path = f"./models/{save_prefix}_{season}_final"
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
    parser.add_argument("--step-sec", type=int, default=int(os.environ.get("HDRL_STEP_SEC", str(DEFAULT_STEP_SEC))))
    parser.add_argument("--episode-days", type=float, default=float(os.environ.get("HDRL_EPISODE_DAYS", str(DEFAULT_EPISODE_DAYS))))
    parser.add_argument("--temp-low", type=float, default=float(os.environ.get("HDRL_TEMP_LOW", str(DEFAULT_TEMP_LOW_C))))
    parser.add_argument("--temp-high", type=float, default=float(os.environ.get("HDRL_TEMP_HIGH", str(DEFAULT_TEMP_HIGH_C))))
    parser.add_argument(
        "--surrogate-kind",
        choices=["legacy_v3", "v35_raw", "v35_calibrated", "hybrid_v3_v35"],
        default=os.environ.get("HDRL_SURROGATE_KIND", DEFAULT_SURROGATE_KIND),
    )
    parser.add_argument("--surrogate-path", default=os.environ.get("HDRL_SURROGATE_PATH"))
    parser.add_argument("--surrogate-summary-json", default=os.environ.get("HDRL_SURROGATE_SUMMARY_JSON"))
    parser.add_argument("--surrogate-checkpoint", default=os.environ.get("HDRL_SURROGATE_CHECKPOINT"))
    parser.add_argument("--surrogate-base-model", default=os.environ.get("HDRL_SURROGATE_BASE_MODEL"))
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default=os.environ.get("HDRL_OBS_ABLATION", DEFAULT_OBS_ABLATION))
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default=os.environ.get("HDRL_DELTA_FEATURE_MODE", DEFAULT_DELTA_FEATURE_MODE))
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default=os.environ.get("HDRL_POWER_FEATURE_MODE", DEFAULT_POWER_FEATURE_MODE))
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default=os.environ.get("HDRL_T_ZONE_FEATURE_MODE", DEFAULT_T_ZONE_FEATURE_MODE))
    parser.add_argument("--lambda-temp-disagree", type=float, default=float(os.environ.get("HDRL_LAMBDA_TEMP_DISAGREE", str(DEFAULT_LAMBDA_TEMP_DISAGREE))))
    parser.add_argument("--lambda-power-disagree", type=float, default=float(os.environ.get("HDRL_LAMBDA_POWER_DISAGREE", str(DEFAULT_LAMBDA_POWER_DISAGREE))))
    parser.add_argument("--save-prefix", default="ppo", help="Prefix for saved winter/summer PPO zip files.")
    args = parser.parse_args()

    os.environ["HDRL_SURROGATE_KIND"] = args.surrogate_kind
    os.environ["HDRL_OBS_ABLATION"] = args.obs_ablation
    os.environ["HDRL_DELTA_FEATURE_MODE"] = args.delta_feature_mode
    os.environ["HDRL_POWER_FEATURE_MODE"] = args.power_feature_mode
    os.environ["HDRL_T_ZONE_FEATURE_MODE"] = args.t_zone_feature_mode
    os.environ["HDRL_LAMBDA_TEMP_DISAGREE"] = str(args.lambda_temp_disagree)
    os.environ["HDRL_LAMBDA_POWER_DISAGREE"] = str(args.lambda_power_disagree)
    if args.surrogate_path:
        os.environ["HDRL_SURROGATE_PATH"] = args.surrogate_path
    if args.surrogate_summary_json:
        os.environ["HDRL_SURROGATE_SUMMARY_JSON"] = args.surrogate_summary_json
    if args.surrogate_checkpoint:
        os.environ["HDRL_SURROGATE_CHECKPOINT"] = args.surrogate_checkpoint
    if args.surrogate_base_model:
        os.environ["HDRL_SURROGATE_BASE_MODEL"] = args.surrogate_base_model

    print("HIERARCHICAL DRL TRAINING (direct TSup)")
    if args.surrogate_kind == "hybrid_v3_v35":
        print(
            "Hybrid settings: "
            f"obs_ablation={args.obs_ablation}, "
            f"delta={args.delta_feature_mode}, "
            f"power={args.power_feature_mode}, "
            f"t_zone={args.t_zone_feature_mode}, "
            f"lambda_temp={args.lambda_temp_disagree:.3f}, "
            f"lambda_power={args.lambda_power_disagree:.1e}"
        )
    t_total = time.time()

    winter_path = train_agent(
        "winter",
        total_timesteps=int(args.winter_steps),
        num_envs=int(args.num_envs),
        step_sec=int(args.step_sec),
        episode_days=float(args.episode_days),
        temp_low_c=float(args.temp_low),
        temp_high_c=float(args.temp_high),
        save_prefix=str(args.save_prefix),
    )
    summer_path = train_agent(
        "summer",
        total_timesteps=int(args.summer_steps),
        num_envs=int(args.num_envs),
        step_sec=int(args.step_sec),
        episode_days=float(args.episode_days),
        temp_low_c=float(args.temp_low),
        temp_high_c=float(args.temp_high),
        save_prefix=str(args.save_prefix),
    )

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'=' * 60}")
    print(f"HDRL TRAINING COMPLETE ({elapsed:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Surrogate kind: {args.surrogate_kind}")
    print(f"  Step: {int(args.step_sec)} s | Comfort band: [{float(args.temp_low):.1f}, {float(args.temp_high):.1f}] C")
    if args.surrogate_summary_json:
        print(f"  V3.5 summary: {args.surrogate_summary_json}")
    if args.surrogate_kind == "hybrid_v3_v35":
        print(
            "  Hybrid settings: "
            f"obs_ablation={args.obs_ablation}, "
            f"delta={args.delta_feature_mode}, "
            f"power={args.power_feature_mode}, "
            f"t_zone={args.t_zone_feature_mode}, "
            f"lambda_temp={args.lambda_temp_disagree:.3f}, "
            f"lambda_power={args.lambda_power_disagree:.1e}"
        )
    print(f"  Winter: {winter_path}")
    print(f"  Summer: {summer_path}")
    print("  Observation: 17D extended TSup features (time + forecast + action history)")
    print("  Meta-controller eval path: hysteresis gate + emergency heating")


if __name__ == "__main__":
    main()
