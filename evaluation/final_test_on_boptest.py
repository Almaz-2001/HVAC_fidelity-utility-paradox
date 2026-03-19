import sys
import numpy as np
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric

# 2. Патч генератора случайных чисел (Перехватчик)
import numpy.random._pickle as npr_pickle
_original_ctor = npr_pickle.__bit_generator_ctor

def _patched_ctor(bit_generator_name="PCG64"):
    # Если получаем объект класса из нового numpy, принудительно превращаем его в строку для старого
    if not isinstance(bit_generator_name, str) or "<class" in str(bit_generator_name):
        bit_generator_name = "PCG64"
    return _original_ctor(bit_generator_name)

npr_pickle.__bit_generator_ctor = _patched_ctor
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from envs.factory import EnvFactory

def run_evaluation(seed, start_time, scenario_name):
    model_path = "models/ppo_surrogate_final.zip"
    config_path = "configs/env.yaml"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка конфига
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    config["backend"] = "boptest"
    
    # ПЕРЕОПРЕДЕЛЯЕМ время старта, если оно передано
    if start_time is not None:
        config["boptest_start_time"] = start_time
        print(f"[{scenario_name}] Установлено время старта: {start_time} сек.")

    # 2. Воспроизводимость
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 3. Загрузка модели
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель {model_path} не найдена")
        return

    env = EnvFactory.create(config)
    from gymnasium import spaces
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
    
    history = {"temp": [], "power": [], "m_s": []}
    obs, _ = env.reset(seed=seed)
    
    steps = 336 # 14 дней
    print(f"\n--- Старт сценария: {scenario_name} ---")
    
    try:
        for step in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            r_vec = info.get("reward_vector", {})
            s_vec = info.get("safety", {})
            
            history["temp"].append(r_vec.get("zone_temp", 0.0))
            history["power"].append(r_vec.get("hvac_power", 0.0))
            history["m_s"].append(s_vec.get("m_s", 0.0))
            
            if step % 48 == 0:
                print(f"{scenario_name} | Шаг {step:3d} | T: {history['temp'][-1]:.2f}°C | Power: {history['power'][-1]:.1f}W")

        # Сохранение с именем сценария
        df = pd.DataFrame(history)
        csv_path = os.path.join(output_dir, f"metrics_scenario_{scenario_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Результаты сохранены в {csv_path}")
        
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_time", type=float, default=None)
    parser.add_argument("--scenario_name", type=str, default="default")
    args = parser.parse_args()
    
    run_evaluation(args.seed, args.start_time, args.scenario_name)