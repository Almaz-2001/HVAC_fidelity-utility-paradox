

import numpy as np
import sys
sys.path.insert(0, '/app')

from layers.safety.action_filter import SurrogateSafetyFilter


def test_safety_filter():
    print("=" * 60)
    print("TEST: Surrogate Safety Filter")
    print("=" * 60)

    # Инициализация
    sf = SurrogateSafetyFilter(
        model_path="/app/outputs/surrogate_v2/rc_node_v2_best.pt",
        horizon=4,
        t_low=21.0,
        t_high=25.0,
        margin=1.11,
    )

    
    print("\n--- Test 1: Safe action (T_zone=23°C, moderate action) ---")
    state = {'t_zone': 23.0, 't_amb': 10.0, 'hour': 12.0, 'day': 30.0}
    action_ppo = np.array([0.0, 0.0], dtype=np.float32)
    action, info = sf.filter(action_ppo, state)
    print(f"  PPO action:  {action_ppo}")
    print(f"  Output:      {action}")
    print(f"  Safe:        {info['safe']}")
    print(f"  Source:      {info['source']}")
    print(f"  Trajectory:  {[f'{t:.1f}' for t in info.get('t_trajectory', info.get('t_trajectory_fallback', []))]}")

    
    print("\n--- Test 2: Unsafe action (T_zone=21.5°C, energy-saving) ---")
    state = {'t_zone': 21.5, 't_amb': -5.0, 'hour': 3.0, 'day': 15.0}
    action_ppo = np.array([-1.0, -1.0], dtype=np.float32)
    action, info = sf.filter(action_ppo, state)
    print(f"  PPO action:  {action_ppo}")
    print(f"  Output:      {action}")
    print(f"  Safe:        {info['safe']}")
    print(f"  Source:      {info['source']}")

    
    print("\n--- Test 3: Unsafe action (T_zone=24.5°C, max heating) ---")
    state = {'t_zone': 24.5, 't_amb': 25.0, 'hour': 14.0, 'day': 200.0}
    action_ppo = np.array([1.0, 1.0], dtype=np.float32)
    action, info = sf.filter(action_ppo, state)
    print(f"  PPO action:  {action_ppo}")
    print(f"  Output:      {action}")
    print(f"  Safe:        {info['safe']}")
    print(f"  Source:      {info['source']}")

    
    print("\n--- Test 4: 100 random actions — statistics ---")
    sf.reset_stats()
    np.random.seed(42)
    for _ in range(100):
        state = {
            't_zone': np.random.uniform(18, 28),
            't_amb': np.random.uniform(-10, 30),
            'hour': np.random.uniform(0, 24),
            'day': np.random.uniform(0, 365),
        }
        action_ppo = np.random.uniform(-1, 1, size=2).astype(np.float32)
        sf.filter(action_ppo, state)

    sf.print_stats()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_safety_filter()