import json
import os
import sys
import time
import zipfile

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, "/app")

from envs.tsup_features import (
    EXTENDED_TSUP_OBS_DIM,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    build_extended_tsup_obs,
)


T_LOW = 21.0
T_HIGH = 25.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0
class BOPTESTDirectEnv(gym.Env):
    """
    Gymnasium env that talks to BOPTEST via raw HTTP using direct TSup control.
    """

    metadata = {"render_modes": []}

    def __init__(self, url="http://web:8000", testcase="bestest_air", start_time=0, step_sec=3600, max_steps=336):
        super().__init__()

        self.url = url
        self.testcase = testcase
        self.start_time = start_time
        self.step_sec = step_sec
        self.max_steps = max_steps
        self.current_step = 0
        self.testid = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_t_zone = None
        self.weather = WeatherLookup()

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(EXTENDED_TSUP_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method, path, payload=None, timeout=120):
        url = f"{self.url}{path}"
        for attempt in range(3):
            try:
                if method == "POST":
                    response = self.session.post(url, json=payload or {}, timeout=timeout)
                elif method == "PUT":
                    response = self.session.put(url, json=payload or {}, timeout=timeout)
                else:
                    response = self.session.get(url, timeout=timeout)
                if response.status_code in (500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _get_val(self, payload, key):
        value = payload.get(key, 0.0)
        return float(value.get("value", value) if isinstance(value, dict) else value)

    def _make_obs(self, payload, prev_action):
        t_c = self._get_val(payload, "zon_reaTRooAir_y") - 273.15
        co2 = self._get_val(payload, "zon_reaCO2RooAir_y")
        p_cool = self._get_val(payload, "fcu_reaPCoo_y")
        p_fan = self._get_val(payload, "fcu_reaPFan_y")
        p_heat = self._get_val(payload, "fcu_reaPHea_y")
        t_amb_k = self._get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
        t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k
        sim_time = self._get_val(payload, "time")
        hour = (sim_time / 3600.0) % 24.0
        day = (sim_time / 86400.0) % 365.0

        p_total = p_cool + p_fan + p_heat
        prev_t_supply = action_to_t_supply(prev_action[0]) if prev_action is not None else 0.5 * (
            T_SUPPLY_LOW + T_SUPPLY_HIGH
        )
        delta_t_zone = 0.0 if self.prev_t_zone is None else (t_c - self.prev_t_zone)
        obs = build_extended_tsup_obs(
            t_c,
            co2,
            p_total,
            prev_t_supply,
            t_amb,
            hour,
            day,
            prev_action if prev_action is not None else np.zeros(2, dtype=np.float32),
            delta_t_zone,
            self.weather,
        )
        self.prev_t_zone = t_c
        return obs, t_c, p_total

    def _compute_reward(self, t_c, p_total):
        if t_c < T_LOW:
            r_comfort = -(T_LOW - t_c)
        elif t_c > T_HIGH:
            r_comfort = -(t_c - T_HIGH)
        else:
            r_comfort = 0.0
        r_energy = -2e-4 * p_total
        return 0.8 * r_comfort + 0.2 * r_energy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10)
            except Exception:
                pass

        data = self._request("POST", f"/testcases/{self.testcase}/select", timeout=300)
        self.testid = data.get("testid")

        self._request("PUT", f"/step/{self.testid}", {"step": self.step_sec}, timeout=30)

        if seed is not None:
            rng = np.random.RandomState(seed)
            offset = rng.randint(-7 * 24 * 3600, 7 * 24 * 3600)
            start_time = max(0, self.start_time + offset)
        else:
            start_time = self.start_time

        self._request(
            "PUT",
            f"/initialize/{self.testid}",
            {"start_time": start_time, "warmup_period": 0},
            timeout=300,
        )

        payload = self._request("POST", f"/advance/{self.testid}", {})
        payload = payload.get("payload", payload)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_t_zone = None
        obs, _, _ = self._make_obs(payload, self.prev_action)
        self.current_step = 0
        return obs, {}

    def step(self, action):
        t_supply = action_to_t_supply(action[0])
        fan_u = action_to_fan(action[1])

        command = {
            "con_oveTSetCoo_activate": 1,
            "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
            "con_oveTSetHea_activate": 1,
            "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": fan_u,
            "fcu_oveTSup_activate": 1,
            "fcu_oveTSup_u": t_supply + 273.15,
        }

        payload = self._request("POST", f"/advance/{self.testid}", command)
        payload = payload.get("payload", payload)

        obs, t_c, p_total = self._make_obs(payload, action)
        reward = self._compute_reward(t_c, p_total)

        self.prev_action = np.asarray(action, dtype=np.float32)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False

        return obs, reward, terminated, truncated, {
            "zone_temp": t_c,
            "power": p_total,
            "t_supply_cmd": t_supply,
        }

    def close(self):
        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10)
            except Exception:
                pass
        self.session.close()


def validate_model_obs_dim(model_path, expected_obs_dim=EXTENDED_TSUP_OBS_DIM):
    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))
    obs_shape = data.get("observation_space", {}).get("_shape", [expected_obs_dim])
    obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
    if obs_dim != expected_obs_dim:
        raise RuntimeError(
            f"Model {model_path} has obs dim {obs_dim}, but BOPTEST fine-tuning expects {expected_obs_dim}. "
            "Retrain the seasonal HDRL agents on the direct-TSup surrogate first."
        )


def finetune_agent(model_path, save_path, start_time, agent_name, steps=10000):
    print(f"\n{'=' * 60}")
    print(f"FINE-TUNING {agent_name.upper()} ON BOPTEST")
    print(f"{'=' * 60}")

    validate_model_obs_dim(model_path)

    env = BOPTESTDirectEnv(
        url="http://web:8000",
        testcase="bestest_air",
        start_time=start_time,
        step_sec=3600,
        max_steps=336,
    )
    env = Monitor(env)

    print("  Testing BOPTEST connection...")
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
    print(f"  Control: direct TSup in [{T_SUPPLY_LOW}, {T_SUPPLY_HIGH}]C")
    print(f"  Steps: {steps:,}")
    print(f"  Episodes: ~{steps // 336}")
    print(f"  Estimated: {steps / 37.6 / 60:.0f} min")
    print()

    start = time.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=1)
    elapsed = time.time() - start

    model.save(save_path)
    env.close()

    print(f"\n  {agent_name.upper()} done: {elapsed / 60:.1f} min")
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

    print(f"\n{'=' * 60}")
    print("BOTH AGENTS FINE-TUNED ON BOPTEST")
    print(f"{'=' * 60}")
    print("  Winter: models/ppo_winter_finetuned.zip")
    print("  Summer: models/ppo_summer_finetuned.zip")
    print("\nRun yearly validation:")
    print("  PYTHONPATH=/app python3 evaluation/yearly_validation_hdrl.py")


if __name__ == "__main__":
    main()
