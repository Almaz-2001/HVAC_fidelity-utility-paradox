from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple, Optional

import numpy as np
import requests
from gymnasium import spaces

from envs.base_env import HVACBaseEnv


def c_to_k(c: float) -> float:
    return float(c) + 273.15


# Physical borders for normalization observation in [-1,1]
#We use  it only in the _make_obs (dont changing thec reward logic)
_OBS_LOW = np.array([15.0, 400.0, 0.0, 0.0], dtype=np.float32)
_OBS_HIGH = np.array([35.0, 2000.0, 5000.0, 500.0], dtype=np.float32)


class BOPTESTBackend(HVACBaseEnv):
    # Measurement keys (as in your setup)
    KEY_T_ROOM = "zon_reaTRooAir_y"
    KEY_CO2 = "zon_reaCO2RooAir_y"
    KEY_P_COO = "fcu_reaPCoo_y"
    KEY_P_FAN = "fcu_reaPFan_y"

    def __init__(self, config: Dict[str, Any]):
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        super().__init__(config)



        # --- BOPTEST connection/config ---
        self.base_url = config.get("boptest_url", "http://web:8000").rstrip("/")
        self.testcase_id = config.get("testcase_id", "bestest_air")
        self.step_sec = int(config.get("step_sec", 3600))

        # timeouts/retries (support both boptest_* and http_* keys)
        self.timeout = float(config.get("boptest_timeout", config.get("http_timeout", 120.0)))
        self.max_retries = int(config.get("boptest_retries", config.get("http_retries", 5)))
        self.backoff_base = float(config.get("boptest_backoff", config.get("http_backoff_base", 0.5)))
        self.recover_on_fail = bool(config.get("recover_on_fail", True))

        # select can be slow
        self.select_timeout = float(
            config.get(
                "boptest_select_timeout",
                config.get("select_timeout", max(600.0, self.timeout)),
            )
        )

        # --- MORL/reward params ---
        morl = config.get("morl", {})
        self.temp_low = float(morl.get("temp_low", 20.0))
        self.temp_high = float(morl.get("temp_high", 26.0))


        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.w_comfort = float(morl.get("w_comfort", 0.8))
        self.w_energy = float(morl.get("w_energy", 0.2))

        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0

        # --- runtime state ---
        self.testid: Optional[str] = None
        self.t = 0.0

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    # -------------------------
    # HTTP helpers
    # -------------------------
    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** attempt)
        time.sleep(min(delay, 8.0))

    def _request_json(self, method: str, url: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        """
        Generic request with retry/backoff on transient errors.
        """
        timeout = float(timeout if timeout is not None else self.timeout)
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    res = self.session.get(url, timeout=timeout)
                elif method.upper() == "POST":
                    res = self.session.post(url, json=(payload or {}), timeout=timeout)
                elif method.upper() == "PUT":
                    res = self.session.put(url, json=(payload or {}), timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # retry on transient server errors
                if res.status_code in (500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue

                res.raise_for_status()
                return res.json()

            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
                last_err = e
                self._sleep_backoff(attempt)

        if last_err:
            raise last_err
        raise RuntimeError("HTTP request failed without exception (unexpected)")

    # -------------------------
    # BOPTEST session lifecycle
    # -------------------------
    def _select_and_warmup(self) -> None:
        """
        Creates/selects a testcase and stores testid.
        Requires BOPTEST endpoint: POST /testcases/<testcaseID>/select
        """
        url_sel = f"{self.base_url}/testcases/{self.testcase_id}/select"
        data = self._request_json("POST", url_sel, payload={}, timeout=self.select_timeout)

        self.testid = data.get("testid")
        if not self.testid:
            raise RuntimeError(f"Failed to obtain testid from select response: {data}")

        # Set control step if endpoint exists (your routes show PUT /step/:testid)
        try:
            self._request_json("PUT", f"{self.base_url}/step/{self.testid}", payload={"step": self.step_sec})
        except Exception:
            # Not fatal; proceed
            pass

        self.t = 0.0
        print(f"[BOPTEST] Selected testid: {self.testid}")

    def _stop_current(self) -> None:
        """
        Stops current test session if possible.
        Your web routes show PUT /stop/:testid.
        """
        if not self.testid:
            return
        try:
            self._request_json("PUT", f"{self.base_url}/stop/{self.testid}", payload={})
        except Exception:
            pass

    def _recover(self) -> None:
        if not self.recover_on_fail:
            return
        self._stop_current()
        self.testid = None
        self._select_and_warmup()

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self._total_steps = 0
        self._violation_steps = 0
        self._max_overshoot = 0.0
        self._max_undershoot = 0.0

        try:
            self._select_and_warmup()
            payload = self.advance({})
        except Exception:
            self._recover()
            payload = self.advance({})

        obs = self._make_obs(payload)
        return obs, {"testid": self.testid, "time": self.t}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # map normalized action to setpoint + fan
        t_target = self.temp_low + (float(action[0]) + 1.0) * 0.5 * (self.temp_high - self.temp_low)
        fan_u = float(np.clip((float(action[1]) + 1.0) * 0.5, 0.0, 1.0))

        u = {
            "con_oveTSetCoo_activate": 1,
            "con_oveTSetCoo_u": c_to_k(t_target + 0.5),
            "con_oveTSetHea_activate": 1,
            "con_oveTSetHea_u": c_to_k(t_target - 0.5),
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": fan_u,
            "fcu_oveTSup_activate": 1,
            "fcu_oveTSup_u": c_to_k(18.0),
        }

        payload = self.advance(u)
        self.t = self._get_val(payload, "time")

        obs = self._make_obs(payload)
        rv = self._make_reward_vector(payload)
        reward = (self.w_comfort * rv["comfort"]) + (self.w_energy * rv["energy"])

        self._update_safety(rv["zone_temp"])

        info = {
            "reward_vector": rv,
            "time": self.t,
            "safety": self.get_safety_metric(),
        }

        return obs, float(reward), False, False, {"reward_vector": rv, "time": self.t}

    # -------------------------
    # BOPTEST calls
    # -------------------------
    def advance(self, actions: Dict[str, Any]) -> dict:
        if not self.testid:
            self._recover()

        url = f"{self.base_url}/advance/{self.testid}"
        try:
            data = self._request_json("POST", url, payload=actions)
        except Exception:
            self._recover()
            url = f"{self.base_url}/advance/{self.testid}"
            data = self._request_json("POST", url, payload=actions)

        return data.get("payload", data)

    # -------------------------
    # Observation 
    # -------------------------
    def _get_val(self, values: dict, key: str) -> float:
        v = values.get(key, 0.0)
        return float(v.get("value", v) if isinstance(v, dict) else v)

    def _make_obs_raw(self, values: dict) -> np.ndarray:
        t_c = self._get_val(values, self.KEY_T_ROOM) - 273.15
        co2 = self._get_val(values, self.KEY_CO2)
        p_cool = self._get_val(values, self.KEY_P_COO)
        p_fan = self._get_val(values, self.KEY_P_FAN)
        return np.array([t_c, co2, p_cool, p_fan], dtype=np.float32)

    def _make_obs(self, values: dict) -> np.ndarray:
        raw = self._make_obs_raw(values)
        obs = 2.0 * (raw - _OBS_LOW) / (_OBS_HIGH - _OBS_LOW) - 1.0
        return np.clip(obs, -1.0, 1.0)

    def _make_reward_vector(self, values: dict) -> dict:
        raw   = self._make_obs_raw(values)
        t_c     = float(raw[0])
        p_total = float(raw[2] + raw[3])

        comfort = 0.0
        if t_c < self.temp_low:
            comfort = -(self.temp_low - t_c)
        elif t_c > self.temp_high:
            comfort = -(t_c - self.temp_high)

        energy = -self.energy_scale * p_total

        return {
            "comfort": float(comfort),
            "energy" : float(energy),
            "zone_temp": float(t_c),
            "hvac_power": float(p_total),
            "w_comfort": float(self.w_comfort),
            "w_energy": float(self.w_energy),
        }
    
    def _update_safety(self, t_c: float) -> None:
        self._total_steps += 1

        if t_c > self.temp_high:
            self._violation_steps += 1
            overshoot = (t_c - self.temp_high ) / self.temp_high
            self._max_overshoot = max(self._max_overshoot, overshoot)
        elif t_c < self.temp_low:
            self._violation_steps += 1
            undershoot = (self.temp_low - t_c) / self.temp_low
            self._max_undershoot = max(self._max_undershoot,undershoot)

    def get_safety_metric(self) -> dict:
        if self._total_steps == 0:
            return {"r_time": 0.0, "r_sev": 0.0, "m_s":0.0,
                    "violation_steps": 0, "total_steps": 0}
        r_time = self._violation_steps / self._total_steps
        r_sev = max(self._max_overshoot, self._max_undershoot)
        m_s = r_time + r_sev

        return {
            "r_time": r_time,
            "r_sev": r_sev,
            "m_s":   m_s,
            "violation_steps": self._violation_steps,
            "total_steps":     self._total_steps,
        }

    
        

    def close(self):
        self._stop_current()
        self.session.close()