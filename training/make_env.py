import os

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed

from envs.factory import EnvFactory


def _resolve_weather_csv(project_root: str) -> str:
    tsup = os.path.join(project_root, "data", "surrogate_v2", "boptest_v2_tsupply.csv")
    if os.path.exists(tsup):
        return tsup
    return os.path.join(project_root, "data", "surrogate_v2", "boptest_v2_all.csv")


def make_surrogate_env(env_id, rank, seed=0):
    def _init():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_surrogate_path = os.path.join(project_root, "outputs", "surrogate_v2", "rc_node_v3_tsupply.pt")

        config = {
            "backend": "surrogate",
            "control_mode": "tsup_direct",
            "surrogate_path": local_surrogate_path,
            "weather_csv": _resolve_weather_csv(project_root),
            "comfort_shaping": {
                "deadband_c": 0.5,
                "band_bonus": 0.05,
                "undershoot_weight": 1.15,
                "overshoot_weight": 1.15,
                "cold_amb_threshold_c": 8.0,
                "hot_amb_threshold_c": 24.0,
                "cold_undershoot_weight": 1.55,
                "hot_overshoot_weight": 1.8,
                "heating_action_bonus": 0.04,
                "cooling_action_bonus": 0.06,
                "heating_t_supply_c": 29.0,
                "cooling_t_supply_c": 21.0,
                "action_fan_threshold": 0.55,
            },
        }

        env = EnvFactory.create(config)
        env = TimeLimit(env, max_episode_steps=336)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init
