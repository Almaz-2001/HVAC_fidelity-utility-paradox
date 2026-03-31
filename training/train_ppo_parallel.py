"""
training/train_ppo_parallel.py

Standard PPO training on 32 parallel surrogate envs.

Usage:
    python training/train_ppo_parallel.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from training.make_env import make_surrogate_env


def train_parallel_ppo():
    num_envs = 32
    total_timesteps = 10_000_000

    vec_env = SubprocVecEnv([
        make_surrogate_env("HVAC-Surrogate-v0", i, seed=42) for i in range(num_envs)
    ])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=1024,
        batch_size=4096, n_epochs=10, gamma=0.99,
        verbose=1, tensorboard_log="./logs/ppo_surrogate_parallel/"
    )

    os.makedirs('./models/ppo_parallel/', exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=31250, save_path='./models/ppo_parallel/',
        name_prefix='ppo_surrogate'
    )

    print(f"Training PPO on {num_envs} parallel envs...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_surrogate_final")
    vec_env.close()
    print("Training complete.")


if __name__ == "__main__":
    train_parallel_ppo()