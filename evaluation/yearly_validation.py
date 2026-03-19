

import os
import time
import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3 import PPO


# -----------------------------------------------------------------------
# Конфигурация
# -----------------------------------------------------------------------

BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
MODEL_PATH = "models/ppo_surrogate_final.zip"
OUTPUT_DIR = "outputs"
STEP_SEC = 3600
STEPS_PER_SCENARIO = 336  # 14 дней
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60

SCENARIOS = {
    "Jan_Winter": 0,
    "Feb_Winter": 2678400,
    "Mar_Spring": 5097600,
    "Apr_Spring": 7776000,
    "May_Spring": 10368000,
    "Jun_Summer": 13132800,
    "Jul_Summer": 15552000,
    "Aug_Summer": 18144000,
    "Oct_Autumn": 23328000,
    "Nov_Autumn": 25920000,
}

# Observation normalization (4 features)
OBS_LOW  = np.array([5.0,  400.0,    0.0,   0.0], dtype=np.float32)
OBS_HIGH = np.array([40.0, 2000.0, 5000.0, 500.0], dtype=np.float32)

# Comfort bounds
T_LOW = 21.0
T_HIGH = 25.0


# -----------------------------------------------------------------------
# BOPTEST HTTP helpers
# -----------------------------------------------------------------------



session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def boptest_request(method, path, payload=None, timeout=60, retries=3):
    """Robust HTTP request with retries."""
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
                raise ValueError(f"Unknown method: {method}")

            if r.status_code in (500, 502, 503, 504):
                print(f"  Server error {r.status_code}, retry {attempt+1}/{retries}...")
                time.sleep(2 ** attempt)
                continue

            r.raise_for_status()
            return r.json()

        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  Connection error, retry {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed after {retries} retries: {url}")


def select_testcase():
    """Select testcase and return testid."""
    print("  Selecting testcase (may take 1-3 min)...")
    data = boptest_request("POST", f"/testcases/{TESTCASE}/select",
                           timeout=SELECT_TIMEOUT, retries=2)
    testid = data.get("testid")
    if not testid:
        raise RuntimeError(f"No testid in response: {data}")
    return testid


def initialize(testid, start_time):
    """Set step size and initialize simulation time."""
    boptest_request("PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request("PUT", f"/initialize/{testid}",
                    {"start_time": start_time, "warmup_period": 0},
                    timeout=SELECT_TIMEOUT)


def advance(testid, actions):
    """Advance one step."""
    data = boptest_request("POST", f"/advance/{testid}", actions, timeout=ADVANCE_TIMEOUT)
    return data.get("payload", data)


def stop(testid):
    """Stop test session."""
    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass


def get_val(payload, key):
    """Extract value from BOPTEST payload."""
    v = payload.get(key, 0.0)
    return float(v.get("value", v) if isinstance(v, dict) else v)




def make_obs(payload):
    """Build normalized 4-feature observation from BOPTEST payload."""
    t_c = get_val(payload, "zon_reaTRooAir_y") - 273.15
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    raw = np.array([t_c, co2, p_cool, p_fan], dtype=np.float32)
    obs = 2.0 * (raw - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1.0
    return np.clip(obs, -1.0, 1.0), t_c, p_cool + p_fan


def action_to_boptest(action):
    """Convert normalized action [-1,1] to BOPTEST control signals."""
    a0, a1 = float(action[0]), float(action[1])
    t_target = T_LOW + (a0 + 1.0) * 0.5 * (T_HIGH - T_LOW)
    fan_u = float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": t_target + 0.5 + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": t_target - 0.5 + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": 18.0 + 273.15,
    }




def run_scenario(model, name, start_time):
    """Run one 14-day scenario on BOPTEST."""
    print(f"\n{'='*50}")
    print(f"SCENARIO: {name} (start_time={start_time}s)")
    print(f"{'='*50}")

    testid = select_testcase()
    print(f"  testid: {testid}")

    initialize(testid, start_time)
    print(f"  Initialized at t={start_time}s")

    # First step (empty) to get initial state
    payload = advance(testid, {})
    obs, t_zone, p_total = make_obs(payload)

    t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
    t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

    history = {"temp": [], "power": [], "t_amb": []}

    for step in range(STEPS_PER_SCENARIO):
        action, _ = model.predict(obs, deterministic=True)
        u = action_to_boptest(action)
        payload = advance(testid, u)

        obs, t_zone, p_total = make_obs(payload)
        t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
        t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

        history["temp"].append(t_zone)
        history["power"].append(p_total)
        history["t_amb"].append(t_amb)

        if step % 48 == 0:
            print(f"  Step {step:3d} | T={t_zone:.1f}C | P={p_total:.0f}W "
                  f"| T_amb={t_amb:.1f}C | a=[{action[0]:.2f},{action[1]:.2f}]")

    stop(testid)

    # Save
    df = pd.DataFrame(history)
    csv_path = os.path.join(OUTPUT_DIR, f"metrics_scenario_{name}.csv")
    df.to_csv(csv_path, index=False)

    # Stats
    temps = np.array(history["temp"])
    viol = ((temps < T_LOW) | (temps > T_HIGH)).mean() * 100
    energy = np.sum(history["power"]) / 1000

    print(f"  RESULT: viol={viol:.1f}%, E={energy:.1f}kWh, "
          f"T=[{temps.min():.1f}, {temps.max():.1f}]")

    return {"name": name, "viol": viol, "energy": energy,
            "t_min": temps.min(), "t_max": temps.max(), "t_mean": temps.mean()}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check BOPTEST
    print("Checking BOPTEST...")
    try:
        r = boptest_request("GET", "/version", timeout=10)
        print(f"BOPTEST version: {r['payload']['version']}")
    except Exception as e:
        print(f"BOPTEST not available: {e}")
        return

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    model = PPO.load(MODEL_PATH, device="cpu", custom_objects=custom_objects)
    print("Model loaded OK")

    # Run all scenarios
    results = []
    start_all = time.time()

    for name, t_start in SCENARIOS.items():
        try:
            r = run_scenario(model, name, t_start)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "viol": None, "energy": None,
                            "t_min": None, "t_max": None, "t_mean": None})

    elapsed = (time.time() - start_all) / 60

    # Summary
    print(f"\n{'='*70}")
    print(f"YEARLY VALIDATION COMPLETE ({elapsed:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Scenario':25s} {'Viol%':>7s} {'T_min':>7s} {'T_max':>7s} {'T_mean':>7s} {'E_kWh':>7s}")
    print("-" * 70)

    valid = [r for r in results if r["viol"] is not None]
    for r in results:
        if r["viol"] is not None:
            print(f"{r['name']:25s} {r['viol']:7.1f} {r['t_min']:7.1f} "
                  f"{r['t_max']:7.1f} {r['t_mean']:7.1f} {r['energy']:7.1f}")
        else:
            print(f"{r['name']:25s}  FAILED")

    if valid:
        viols = [r["viol"] for r in valid]
        energies = [r["energy"] for r in valid]
        print("-" * 70)
        print(f"{'MEAN':25s} {np.mean(viols):7.1f} {'':7s} {'':7s} {'':7s} {np.mean(energies):7.1f}")
        print(f"{'STD':25s} {np.std(viols):7.1f} {'':7s} {'':7s} {'':7s} {np.std(energies):7.1f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()