


import os
import sys
import time
import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, '/app')




class BOPTESTDirectEnv(gym.Env):
    """
    Gymnasium env that talks to BOPTEST via raw HTTP.
    Same approach as yearly_validation.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, url="http://web:8000", testcase="bestest_air",
                 start_time=0, step_sec=3600, max_steps=336):
        super().__init__()

        self.url = url
        self.testcase = testcase
        self.start_time = start_time
        self.step_sec = step_sec
        self.max_steps = max_steps
        self.current_step = 0
        self.testid = None

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        self.STATE_LOW = np.array([5.0, 400.0, 0.0, 0.0], dtype=np.float32)
        self.STATE_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0], dtype=np.float32)

    def _request(self, method, path, payload=None, timeout=120):
        url = f"{self.url}{path}"
        for attempt in range(3):
            try:
                if method == "POST":
                    r = self.session.post(url, json=payload or {}, timeout=timeout)
                elif method == "PUT":
                    r = self.session.put(url, json=payload or {}, timeout=timeout)
                else:
                    r = self.session.get(url, timeout=timeout)
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _get_val(self, payload, key):
        v = payload.get(key, 0.0)
        return float(v.get("value", v) if isinstance(v, dict) else v)

    def _make_obs(self, payload):
        t_c = self._get_val(payload, "zon_reaTRooAir_y") - 273.15
        co2 = self._get_val(payload, "zon_reaCO2RooAir_y")
        p_cool = self._get_val(payload, "fcu_reaPCoo_y")
        p_fan = self._get_val(payload, "fcu_reaPFan_y")
        raw = np.array([t_c, co2, p_cool, p_fan], dtype=np.float32)
        obs = 2.0 * (raw - self.STATE_LOW) / (self.STATE_HIGH - self.STATE_LOW) - 1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32), t_c, p_cool + p_fan

    def _compute_reward(self, t_c, p_total):
        if t_c < 21.0:
            r_comfort = -(21.0 - t_c)
        elif t_c > 25.0:
            r_comfort = -(t_c - 25.0)
        else:
            r_comfort = 0.0
        r_energy = -2e-4 * p_total
        return 0.8 * r_comfort + 0.2 * r_energy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Stop previous session
        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10)
            except Exception:
                pass

        # Select new testcase
        data = self._request("POST", f"/testcases/{self.testcase}/select", timeout=300)
        self.testid = data.get("testid")

        # Set step and initialize
        self._request("PUT", f"/step/{self.testid}", {"step": self.step_sec}, timeout=30)

        # Randomize start time slightly for diversity
        if seed is not None:
            rng = np.random.RandomState(seed)
            offset = rng.randint(-7 * 24 * 3600, 7 * 24 * 3600)
            st = max(0, self.start_time + offset)
        else:
            st = self.start_time

        self._request("PUT", f"/initialize/{self.testid}",
                      {"start_time": st, "warmup_period": 0}, timeout=300)

        payload = self._request("POST", f"/advance/{self.testid}", {})
        payload = payload.get("payload", payload)

        obs, _, _ = self._make_obs(payload)
        self.current_step = 0

        return obs, {}

    def step(self, action):
        a0, a1 = float(action[0]), float(action[1])
        t_target = 21.0 + (a0 + 1.0) * 0.5 * (25.0 - 21.0)
        fan_u = float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))

        u = {
            "con_oveTSetCoo_activate": 1, "con_oveTSetCoo_u": t_target + 0.5 + 273.15,
            "con_oveTSetHea_activate": 1, "con_oveTSetHea_u": t_target - 0.5 + 273.15,
            "fcu_oveFan_activate": 1, "fcu_oveFan_u": fan_u,
            "fcu_oveTSup_activate": 1, "fcu_oveTSup_u": 18.0 + 273.15,
        }

        payload = self._request("POST", f"/advance/{self.testid}", u)
        payload = payload.get("payload", payload)

        obs, t_c, p_total = self._make_obs(payload)
        reward = self._compute_reward(t_c, p_total)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False

        return obs, reward, terminated, truncated, {"zone_temp": t_c, "power": p_total}

    def close(self):
        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10)
            except Exception:
                pass
        self.session.close()




def finetune_agent(model_path, save_path, start_time, agent_name, steps=10000):
    print(f"\n{'='*60}")
    print(f"FINE-TUNING {agent_name.upper()} ON BOPTEST")
    print(f"{'='*60}")

    env = BOPTESTDirectEnv(
        url="http://web:8000",
        testcase="bestest_air",
        start_time=start_time,
        step_sec=3600,
        max_steps=336,
    )
    env = Monitor(env)

    print(f"  Testing BOPTEST connection...")
    obs, _ = env.reset()
    print(f"  Connected! Initial obs: {obs}")

    custom_objects = {
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 1e-4,
    }
    model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4
    model.n_epochs = 5

    print(f"  Model: {model_path}")
    print(f"  Start time: {start_time}s ({agent_name})")
    print(f"  Steps: {steps:,}")
    print(f"  Episodes: ~{steps // 336}")
    print(f"  Estimated: {steps / 37.6 / 60:.0f} min")
    print()

    t0 = time.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=1)
    elapsed = time.time() - t0

    model.save(save_path)
    env.close()

    print(f"\n  {agent_name.upper()} done: {elapsed/60:.1f} min")
    print(f"  Saved: {save_path}.zip")


def main():
    steps = 50000  

    finetune_agent(
        model_path="models/ppo_winter_final.zip",
        save_path="models/ppo_winter_finetuned",
        start_time=0,
        agent_name="winter",
        steps=steps,
    )

    finetune_agent(
        model_path="models/ppo_summer_final.zip",
        save_path="models/ppo_summer_finetuned",
        start_time=15552000,
        agent_name="summer",
        steps=steps,
    )

    print(f"\n{'='*60}")
    print(f"BOTH AGENTS FINE-TUNED ON BOPTEST")
    print(f"{'='*60}")
    print(f"  Winter: models/ppo_winter_finetuned.zip")
    print(f"  Summer: models/ppo_summer_finetuned.zip")
    print(f"\nUpdate yearly_validation.py and run:")
    print(f"  PYTHONPATH=/app python3 evaluation/yearly_validation.py")


if __name__ == "__main__":
    main()