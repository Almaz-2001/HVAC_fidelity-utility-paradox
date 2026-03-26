"""
training/train_hdrl.py

Hierarchical DRL: two specialized PPO agents + meta-controller.

Winter agent: trained on cold weather (T_amb_base: -10..5°C)
Summer agent: trained on warm weather (T_amb_base: 10..25°C)
Meta-controller: if T_amb < 12°C → winter agent, else → summer agent

Based on: Liao et al. - Hierarchical DRL for HVAC control

Usage:
    python training/train_hdrl.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from training.make_env import make_surrogate_env


def make_seasonal_env(env_id, rank, seed=0, season="winter"):
    """Create surrogate env with seasonal DR bias."""
    def _init():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_surrogate_path = os.path.join(project_root, "outputs", "surrogate_v2", "rc_node_v2_best.pt")

        if season == "winter":
            config = {
                "backend": "surrogate",
                "surrogate_path": local_surrogate_path,
                "domain_randomization": {
                    "enabled": True,
                    "t_init_low": 5.0,       # cold starts
                    "t_init_high": 20.0,
                    "t_amb_base_low": -15.0,  # winter: -15 to 5°C base
                    "t_amb_base_high": 5.0,
                    "t_amb_amp_low": 3.0,
                    "t_amb_amp_high": 8.0,
                    "diurnal_low": 2.0,
                    "diurnal_high": 5.0,
                    "noise_low": 0.5,
                    "noise_high": 2.0,
                }
            }
        else:  # summer
            config = {
                "backend": "surrogate",
                "surrogate_path": local_surrogate_path,
                "domain_randomization": {
                    "enabled": True,
                    "t_init_low": 18.0,       # warm starts
                    "t_init_high": 30.0,
                    "t_amb_base_low": 10.0,   # summer: 10 to 30°C base
                    "t_amb_base_high": 30.0,
                    "t_amb_amp_low": 5.0,
                    "t_amb_amp_high": 12.0,
                    "diurnal_low": 3.0,
                    "diurnal_high": 8.0,
                    "noise_low": 0.5,
                    "noise_high": 1.5,
                }
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
    """Train one seasonal agent."""
    print(f"\n{'='*60}")
    print(f"TRAINING {season.upper()} AGENT")
    print(f"{'='*60}")

    vec_env = SubprocVecEnv([
        make_seasonal_env("HVAC", i, seed=42, season=season) for i in range(num_envs)
    ])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"./logs/hdrl_{season}/"
    )

    save_path = f"./models/ppo_{season}_final"
    os.makedirs('./models/', exist_ok=True)

    t_start = time.time()
    print(f"  Steps: {total_timesteps:,}, Envs: {num_envs}")
    print(f"  DR range: {'T_amb -15..5°C' if season == 'winter' else 'T_amb 10..30°C'}")

    model.learn(total_timesteps=total_timesteps)

    model.save(save_path)
    vec_env.close()

    elapsed = (time.time() - t_start) / 60
    print(f"  {season.upper()} agent trained in {elapsed:.1f} min")
    print(f"  Saved: {save_path}.zip")

    return save_path + ".zip"


def main():
    print("HIERARCHICAL DRL TRAINING")
    print("Two specialized agents + meta-controller")
    print()

    t_total = time.time()

    # Train both agents
    winter_path = train_agent("winter", total_timesteps=5_000_000, num_envs=16)
    summer_path = train_agent("summer", total_timesteps=5_000_000, num_envs=16)

    elapsed = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"HDRL TRAINING COMPLETE ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"  Winter agent: {winter_path}")
    print(f"  Summer agent: {summer_path}")
    print(f"  Meta-controller: T_amb < 12°C → winter, else → summer")
    print(f"\nRun validation:")
    print(f"  PYTHONPATH=/app python3 evaluation/yearly_validation.py")


if __name__ == "__main__":
    main()