import os
import sys

# Добавляем корневую папку в sys.path, чтобы импорты работали корректно из любой директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from training.make_env import make_surrogate_env

def train_parallel_ppo():
    num_envs = 32
    total_timesteps = 10_000_000
    env_id = "HVAC-Surrogate-v0" # Замените на реальное название вашей среды
    
    # 1. Создаем 32 параллельных процесса
    vec_env = SubprocVecEnv([
        make_surrogate_env(env_id, i, seed=42) for i in range(num_envs)
    ])
    vec_env = VecMonitor(vec_env)
    
    # 2. Инициализируем PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/ppo_surrogate_parallel/"
    )
    
    # Создаем папку для сохранения моделей, если ее нет
    os.makedirs('./models/ppo_parallel/', exist_ok=True)
    
    # 3. Коллбэк для сохранения (каждые 1 млн шагов: 31250 * 32 = 1 000 000)
    checkpoint_callback = CheckpointCallback(
        save_freq=31250, 
        save_path='./models/ppo_parallel/',
        name_prefix='ppo_surrogate'
    )
    
    # 4. Запуск
    print(f"Запуск обучения PPO на {num_envs} параллельных средах...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
    )
    
    model.save("./models/ppo_surrogate_final")
    vec_env.close()

# НА WINDOWS ЭТОТ БЛОК ОБЯЗАТЕЛЕН!
if __name__ == "__main__":
    train_parallel_ppo()