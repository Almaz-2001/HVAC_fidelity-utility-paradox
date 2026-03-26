

import os
import sys
import time
import numpy as np
import pandas as pd
import requests
import zipfile
import json
from gymnasium import spaces
from stable_baselines3 import PPO

sys.path.insert(0, '/app')

BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
WINTER_MODEL = "models/ppo_winter_finetuned.zip"
SUMMER_MODEL = "models/ppo_summer_finetuned.zip"
OUTPUT_DIR = "outputs"
STEP_SEC = 3600
STEPS_PER_SCENARIO = 336
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60
T_AMB_THRESHOLD = 12.0

# Emergency Heating Mode thresholds
EMERGENCY_T_AMB = 5.0      # T_amb below this triggers check
EMERGENCY_T_ZONE = 20.0    # T_zone below this triggers emergency
EMERGENCY_ACTION = np.array([1.0, 1.0], dtype=np.float32)  # max heat, max fan

SCENARIOS = {
    "Jan_Winter": 0,
    "Feb_Winter": 2678400,
    "Mar_Spring": 5097600,
    "Apr_Spring": 7776000,
    "May_Spring": 10368000,
    "Jun_Summer": 13132800,
    "Jul_Summer": 15552000,
    "Aug_Summer": 18144000,
    "Sep_Autumn": 20736000,
    "Oct_Autumn": 23328000,
    "Nov_Autumn": 25920000,
    "Dec_Winter": 28512000,
}

STATE_LOW  = np.array([5.0,  400.0,    0.0,   0.0], dtype=np.float32)
STATE_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0], dtype=np.float32)
T_LOW = 21.0
T_HIGH = 25.0

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def boptest_request(method, path, payload=None, timeout=60, retries=3):
    url = f"{BOPTEST_URL}{path}"
    for attempt in range(retries):
        try:
            if method == "POST":
                r = session.post(url, json=payload or {}, timeout=timeout)
            elif method == "PUT":
                r = session.put(url, json=payload or {}, timeout=timeout)
            elif method == "GET":
                r = session.get(url, timeout=timeout)
            else:
                raise ValueError(f"Unknown: {method}")
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed: {url}")


def select_testcase():
    print("  Selecting testcase...")
    data = boptest_request("POST", f"/testcases/{TESTCASE}/select",
                           timeout=SELECT_TIMEOUT, retries=2)
    testid = data.get("testid")
    if not testid:
        raise RuntimeError(f"No testid: {data}")
    return testid


def initialize(testid, start_time):
    boptest_request("PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request("PUT", f"/initialize/{testid}",
                    {"start_time": start_time, "warmup_period": 0},
                    timeout=SELECT_TIMEOUT)


def advance(testid, actions):
    data = boptest_request("POST", f"/advance/{testid}", actions, timeout=ADVANCE_TIMEOUT)
    return data.get("payload", data)


def stop(testid):
    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass


def get_val(payload, key):
    v = payload.get(key, 0.0)
    return float(v.get("value", v) if isinstance(v, dict) else v)


def make_obs(payload):
    t_c = get_val(payload, "zon_reaTRooAir_y") - 273.15
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
    t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k
    raw = np.array([t_c, co2, p_cool, p_fan], dtype=np.float32)
    obs = 2.0 * (raw - STATE_LOW) / (STATE_HIGH - STATE_LOW) - 1.0
    return np.clip(obs, -1.0, 1.0), t_c, p_cool + p_fan, t_amb


def action_to_boptest(action):
    a0, a1 = float(action[0]), float(action[1])
    t_target = T_LOW + (a0 + 1.0) * 0.5 * (T_HIGH - T_LOW)
    fan_u = float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))
    return {
        "con_oveTSetCoo_activate": 1, "con_oveTSetCoo_u": t_target + 0.5 + 273.15,
        "con_oveTSetHea_activate": 1, "con_oveTSetHea_u": t_target - 0.5 + 273.15,
        "fcu_oveFan_activate": 1, "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1, "fcu_oveTSup_u": 18.0 + 273.15,
    }


def run_scenario(winter_model, summer_model, name, start_time):
    print(f"\n{'='*55}")
    print(f"SCENARIO: {name} (start={start_time}s)")
    print(f"{'='*55}")

    testid = select_testcase()
    print(f"  testid: {testid}")
    initialize(testid, start_time)

    payload = advance(testid, {})
    obs, t_zone, p_total, t_amb = make_obs(payload)

    history = {"temp": [], "power": [], "t_amb": [], "agent": []}
    winter_count = 0
    summer_count = 0
    emergency_count = 0

    for step in range(STEPS_PER_SCENARIO):
        # Emergency Heating Mode: extreme cold + building cooling down
        if t_amb < EMERGENCY_T_AMB and t_zone < EMERGENCY_T_ZONE:
            action = EMERGENCY_ACTION.copy()
            agent_name = "E"  # Emergency
            emergency_count += 1
        elif t_amb < T_AMB_THRESHOLD:
            action, _ = winter_model.predict(obs, deterministic=True)
            agent_name = "W"
            winter_count += 1
        else:
            action, _ = summer_model.predict(obs, deterministic=True)
            agent_name = "S"
            summer_count += 1

        u = action_to_boptest(action)
        payload = advance(testid, u)
        obs, t_zone, p_total, t_amb = make_obs(payload)

        history["temp"].append(t_zone)
        history["power"].append(p_total)
        history["t_amb"].append(t_amb)
        history["agent"].append(agent_name)

        if step % 48 == 0:
            print(f"  Step {step:3d} | T={t_zone:.1f}C | P={p_total:.0f}W "
                  f"| T_amb={t_amb:.1f}C | {agent_name} "
                  f"| a=[{action[0]:.2f},{action[1]:.2f}]")

    stop(testid)

    df = pd.DataFrame(history)
    csv_path = os.path.join(OUTPUT_DIR, f"metrics_scenario_{name}.csv")
    df.to_csv(csv_path, index=False)

    temps = np.array(history["temp"])
    viol = ((temps < T_LOW) | (temps > T_HIGH)).mean() * 100
    energy = np.sum(history["power"]) / 1000
    r_time = ((temps < T_LOW) | (temps > T_HIGH)).mean()
    over = ((temps - T_HIGH) / T_HIGH).clip(min=0).max()
    under = ((T_LOW - temps) / T_LOW).clip(min=0).max()
    ms = r_time + max(over, under)

    total = winter_count + summer_count + emergency_count
    print(f"  Agents: W={winter_count}({winter_count/total*100:.0f}%) "
          f"S={summer_count}({summer_count/total*100:.0f}%) "
          f"E={emergency_count}({emergency_count/total*100:.0f}%)")
    print(f"  RESULT: viol={viol:.1f}%, E={energy:.1f}kWh, m_s={ms:.3f}, "
          f"T_min={temps.min():.1f}")

    return {"name": name, "viol": viol, "energy": energy, "ms": ms,
            "t_min": temps.min(), "t_max": temps.max(), "t_mean": temps.mean(),
            "winter_pct": winter_count/total*100,
            "emergency_pct": emergency_count/total*100}


def load_model(path):
    z = zipfile.ZipFile(path)
    data = json.loads(z.read('data'))
    obs_shape = data.get('observation_space', {}).get('_shape', [4])
    obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    return PPO.load(path, device="cpu", custom_objects=custom_objects)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Checking BOPTEST...")
    try:
        r = boptest_request("GET", "/version", timeout=10)
        print(f"BOPTEST version: {r['payload']['version']}")
    except Exception as e:
        print(f"BOPTEST not available: {e}")
        return

    print(f"\nLoading HDRL agents:")
    winter_model = load_model(WINTER_MODEL)
    print(f"  Winter: loaded")
    summer_model = load_model(SUMMER_MODEL)
    print(f"  Summer: loaded")
    print(f"  Meta: T_amb < {T_AMB_THRESHOLD}°C → winter")
    print(f"  Emergency: T_amb < {EMERGENCY_T_AMB}°C AND T_zone < {EMERGENCY_T_ZONE}°C → max heat")

    results = []
    start_all = time.time()

    for name, t_start in SCENARIOS.items():
        try:
            r = run_scenario(winter_model, summer_model, name, t_start)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": name, "viol": None, "energy": None, "ms": None,
                            "t_min": None, "t_max": None, "t_mean": None,
                            "winter_pct": None, "emergency_pct": None})

    elapsed = (time.time() - start_all) / 60

    print(f"\n{'='*75}")
    print(f"HDRL + EMERGENCY HEATING ({elapsed:.1f} min)")
    print(f"{'='*75}")
    print(f"{'Scenario':25s} {'Viol%':>7s} {'T_min':>7s} {'T_max':>7s} "
          f"{'E_kWh':>7s} {'m_s':>7s} {'W%':>4s} {'E%':>4s}")
    print("-" * 75)

    valid = [r for r in results if r["viol"] is not None]
    for r in results:
        if r["viol"] is not None:
            print(f"{r['name']:25s} {r['viol']:7.1f} {r['t_min']:7.1f} "
                  f"{r['t_max']:7.1f} {r['energy']:7.1f} {r['ms']:7.3f} "
                  f"{r['winter_pct']:4.0f} {r['emergency_pct']:4.0f}")
        else:
            print(f"{r['name']:25s}  FAILED")

    if valid:
        viols = [r["viol"] for r in valid]
        energies = [r["energy"] for r in valid]
        ms_vals = [r["ms"] for r in valid]
        print("-" * 75)
        print(f"{'MEAN':25s} {np.mean(viols):7.1f} {'':7s} {'':7s} "
              f"{np.mean(energies):7.1f} {np.mean(ms_vals):7.3f}")
        print(f"{'STD':25s} {np.std(viols):7.1f} {'':7s} {'':7s} "
              f"{np.std(energies):7.1f} {np.std(ms_vals):7.3f}")

    print(f"{'='*75}")


if __name__ == "__main__":
    main()