"""
evaluation/yearly_validation_hdrl.py

Yearly BOPTEST validation for the seasonal HDRL setup under direct TSup control.
"""

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3 import PPO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.tsup_features import (
    EXTENDED_TSUP_OBS_DIM,
    SUPPORTED_DELTA_FEATURE_MODES,
    SUPPORTED_POWER_FEATURE_MODES,
    SUPPORTED_TSUP_OBS_ABLATIONS,
    SUPPORTED_T_ZONE_FEATURE_MODES,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    build_extended_tsup_obs,
)


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
WINTER_MODEL = "models/ppo_winter_final.zip"
SUMMER_MODEL = "models/ppo_summer_final.zip"
OUTPUT_DIR = "outputs"
STEP_SEC = 3600
SCENARIO_DAYS = 14
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60
WINTER_ENTER_T_AMB = 10.0
WINTER_EXIT_T_AMB = 15.0
N_OBS = EXTENDED_TSUP_OBS_DIM

EMERGENCY_T_AMB = 5.0
EMERGENCY_T_ZONE = 20.0
EMERGENCY_ACTION = np.array([1.0, 1.0], dtype=np.float32)

T_LOW = 21.0
T_HIGH = 24.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0
OBS_ABLATION = "none"
DELTA_FEATURE_MODE = "raw"
POWER_FEATURE_MODE = "raw"
T_ZONE_FEATURE_MODE = "raw"

SCENARIOS = {
    "Jan_Winter": 0,
    "Feb_Winter": 2678400,
    "Mar_Spring": 5097600,
    "Apr_Spring": 7776000,
    "May_Spring": 10368000,
    "Jun_Summer": 13132800,
    "Jul_Summer": 15552000,
    "Aug_Summer": 18316800,
    "Sep_Autumn": 20995200,
    "Oct_Autumn": 23587200,
    "Nov_Autumn": 26265600,
    "Dec_Winter": 28857600,
}

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def boptest_request(method, path, payload=None, timeout=60, retries=3):
    url = f"{BOPTEST_URL}{path}"
    for attempt in range(retries):
        try:
            if method == "POST":
                response = session.post(url, json=payload or {}, timeout=timeout)
            elif method == "PUT":
                response = session.put(url, json=payload or {}, timeout=timeout)
            elif method == "GET":
                response = session.get(url, timeout=timeout)
            else:
                raise ValueError(f"Unknown HTTP method: {method}")
            if response.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            print(f"  Retry {attempt + 1}/{retries}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed: {url}")


def select_testcase():
    print("  Selecting testcase...")
    data = boptest_request("POST", f"/testcases/{TESTCASE}/select", timeout=SELECT_TIMEOUT, retries=2)
    testid = data.get("testid")
    if not testid:
        raise RuntimeError(f"No testid in response: {data}")
    return testid


def initialize(testid, start_time):
    boptest_request("PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request(
        "PUT",
        f"/initialize/{testid}",
        {"start_time": start_time, "warmup_period": 0},
        timeout=SELECT_TIMEOUT,
    )


def advance(testid, actions):
    data = boptest_request("POST", f"/advance/{testid}", actions, timeout=ADVANCE_TIMEOUT)
    return data.get("payload", data)


def stop(testid):
    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass


def get_val(payload, key):
    value = payload.get(key, 0.0)
    return float(value.get("value", value) if isinstance(value, dict) else value)

def make_obs(payload, prev_action, prev_t_zone, weather):
    t_c = get_val(payload, "zon_reaTRooAir_y") - 273.15
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    p_heat = get_val(payload, "fcu_reaPHea_y")
    t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
    t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

    sim_time = get_val(payload, "time")
    hour = (sim_time / 3600.0) % 24.0
    day = (sim_time / 86400.0) % 365.0

    p_total = p_cool + p_fan + p_heat
    prev_t_supply = action_to_t_supply(prev_action[0]) if prev_action is not None else 0.5 * (
        T_SUPPLY_LOW + T_SUPPLY_HIGH
    )
    delta_t_zone = 0.0 if prev_t_zone is None else (t_c - prev_t_zone)
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
        weather,
        obs_ablation=OBS_ABLATION,
        delta_feature_mode=DELTA_FEATURE_MODE,
        t_zone_feature_mode=T_ZONE_FEATURE_MODE,
        power_feature_mode=POWER_FEATURE_MODE,
    )
    return obs, t_c, p_total, t_amb, sim_time


def action_to_boptest(action):
    t_supply = action_to_t_supply(action[0])
    fan_u = action_to_fan(action[1])
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": t_supply + 273.15,
    }


def load_model(path):
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    obs_shape = data.get("observation_space", {}).get("_shape", [N_OBS])
    obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
    if obs_dim != N_OBS:
        raise RuntimeError(
            f"Model {path} has obs dim {obs_dim}, but HDRL direct-TSup validation expects {N_OBS}."
        )
    custom_objects = {
        "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 3e-4,
    }
    return PPO.load(path, device="cpu", custom_objects=custom_objects)


def update_gate_mode(current_mode, t_amb):
    if current_mode == "winter":
        return "summer" if t_amb > WINTER_EXIT_T_AMB else "winter"
    return "winter" if t_amb < WINTER_ENTER_T_AMB else "summer"


def run_scenario(winter_model, summer_model, name, start_time):
    print(f"\n{'=' * 55}")
    print(f"SCENARIO: {name} (start={start_time}s)")
    print(f"{'=' * 55}")

    testid = select_testcase()
    print(f"  testid: {testid}")
    initialize(testid, start_time)

    payload = advance(testid, {})
    weather = WeatherLookup()
    prev_action = np.zeros(2, dtype=np.float32)
    obs, t_zone, p_total, t_amb, sim_time = make_obs(payload, prev_action, None, weather)
    gate_mode = "winter" if t_amb < WINTER_ENTER_T_AMB else "summer"

    history_rows = []
    winter_count = 0
    summer_count = 0
    emergency_count = 0
    gate_switches = 0
    steps_per_scenario = int(round(SCENARIO_DAYS * 86400 / STEP_SEC))
    log_interval = max(1, int(round(86400 / STEP_SEC)))

    for step in range(steps_per_scenario):
        if t_amb < EMERGENCY_T_AMB and t_zone < EMERGENCY_T_ZONE:
            action = EMERGENCY_ACTION.copy()
            agent_name = "E"
            emergency_count += 1
        else:
            next_mode = update_gate_mode(gate_mode, t_amb)
            if next_mode != gate_mode:
                gate_switches += 1
                gate_mode = next_mode
            if gate_mode == "winter":
                action, _ = winter_model.predict(obs, deterministic=True)
                agent_name = "W"
                winter_count += 1
            else:
                action, _ = summer_model.predict(obs, deterministic=True)
                agent_name = "S"
                summer_count += 1

        prev_t_zone = t_zone
        prev_t_amb = t_amb
        prev_power = p_total
        prev_sim_time = sim_time
        command = action_to_boptest(action)
        payload = advance(testid, command)
        obs, t_zone, p_total, t_amb, sim_time = make_obs(payload, action, prev_t_zone, weather)

        a0 = float(action[0])
        a1 = float(action[1])
        t_supply = action_to_t_supply(a0)
        fan_u = action_to_fan(a1)
        history_rows.append(
            {
                "step": step,
                "prev_time": prev_sim_time,
                "time": sim_time,
                "prev_t_zone": prev_t_zone,
                "prev_t_amb": prev_t_amb,
                "prev_power": prev_power,
                "temp": t_zone,
                "power": p_total,
                "t_amb": t_amb,
                "agent": agent_name,
                "mode": "winter" if gate_mode == "winter" else "summer",
                "a0": a0,
                "a1": a1,
                "fan_u": fan_u,
                "t_supply": t_supply,
            }
        )

        if step % log_interval == 0:
            print(
                f"  Step {step:3d} | T={t_zone:.1f}C | P={p_total:.0f}W "
                f"| T_amb={t_amb:.1f}C | TSup={t_supply:.1f}C "
                f"| {agent_name} | gate={gate_mode[0].upper()} | a=[{action[0]:.2f},{action[1]:.2f}]"
            )

        prev_action = np.asarray(action, dtype=np.float32)

    stop(testid)

    df = pd.DataFrame(history_rows)
    csv_path = os.path.join(OUTPUT_DIR, f"hdrl_scenario_{name}.csv")
    df.to_csv(csv_path, index=False)

    temps = df["temp"].to_numpy(dtype=float)
    viol = ((temps < T_LOW) | (temps > T_HIGH)).mean() * 100
    energy = float(df["power"].sum() * (STEP_SEC / 3600.0) / 1000.0)
    r_time = ((temps < T_LOW) | (temps > T_HIGH)).mean()
    over = ((temps - T_HIGH) / T_HIGH).clip(min=0).max()
    under = ((T_LOW - temps) / T_LOW).clip(min=0).max()
    ms = r_time + max(over, under)

    total = winter_count + summer_count + emergency_count
    print(
        f"  Agents: W={winter_count}({winter_count / total * 100:.0f}%) "
        f"S={summer_count}({summer_count / total * 100:.0f}%) "
        f"E={emergency_count}({emergency_count / total * 100:.0f}%) "
        f"| switches={gate_switches}"
    )
    print(f"  RESULT: viol={viol:.1f}%, E={energy:.1f}kWh, m_s={ms:.3f}")

    return {
        "name": name,
        "viol": viol,
        "energy": energy,
        "ms": ms,
        "t_min": temps.min(),
        "t_max": temps.max(),
        "t_mean": temps.mean(),
        "winter_pct": winter_count / total * 100,
        "emergency_pct": emergency_count / total * 100,
        "gate_switches": gate_switches,
    }


def main():
    global WINTER_MODEL, SUMMER_MODEL, OUTPUT_DIR, STEP_SEC, SCENARIO_DAYS, BOPTEST_URL, TESTCASE
    global T_LOW, T_HIGH, OBS_ABLATION, DELTA_FEATURE_MODE, POWER_FEATURE_MODE, T_ZONE_FEATURE_MODE

    parser = argparse.ArgumentParser(description="Yearly BOPTEST validation for HDRL direct TSup control.")
    parser.add_argument("--winter-model", default=WINTER_MODEL)
    parser.add_argument("--summer-model", default=SUMMER_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--step-sec", type=int, default=STEP_SEC)
    parser.add_argument("--scenario-days", type=int, default=SCENARIO_DAYS)
    parser.add_argument("--boptest-url", default=BOPTEST_URL)
    parser.add_argument("--testcase", default=TESTCASE)
    parser.add_argument("--temp-low", type=float, default=T_LOW)
    parser.add_argument("--temp-high", type=float, default=T_HIGH)
    parser.add_argument("--obs-ablation", choices=sorted(SUPPORTED_TSUP_OBS_ABLATIONS), default=OBS_ABLATION)
    parser.add_argument("--delta-feature-mode", choices=sorted(SUPPORTED_DELTA_FEATURE_MODES), default=DELTA_FEATURE_MODE)
    parser.add_argument("--power-feature-mode", choices=sorted(SUPPORTED_POWER_FEATURE_MODES), default=POWER_FEATURE_MODE)
    parser.add_argument("--t-zone-feature-mode", choices=sorted(SUPPORTED_T_ZONE_FEATURE_MODES), default=T_ZONE_FEATURE_MODE)
    args = parser.parse_args()

    WINTER_MODEL = args.winter_model
    SUMMER_MODEL = args.summer_model
    OUTPUT_DIR = args.output_dir
    STEP_SEC = int(args.step_sec)
    SCENARIO_DAYS = int(args.scenario_days)
    BOPTEST_URL = args.boptest_url
    TESTCASE = args.testcase
    T_LOW = float(args.temp_low)
    T_HIGH = float(args.temp_high)
    OBS_ABLATION = args.obs_ablation
    DELTA_FEATURE_MODE = args.delta_feature_mode
    POWER_FEATURE_MODE = args.power_feature_mode
    T_ZONE_FEATURE_MODE = args.t_zone_feature_mode

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Checking BOPTEST...")
    try:
        response = boptest_request("GET", "/version", timeout=10)
        print(f"BOPTEST version: {response['payload']['version']}")
    except Exception as exc:
        print(f"BOPTEST not available: {exc}")
        return

    print("\nHDRL + Emergency Heating Mode (direct TSup, extended obs)")
    print(f"Emergency: T_amb < {EMERGENCY_T_AMB}C AND T_zone < {EMERGENCY_T_ZONE}C")
    print(f"Hysteresis gate: enter winter < {WINTER_ENTER_T_AMB}C, exit winter > {WINTER_EXIT_T_AMB}C")
    print()

    print("Loading HDRL agents:")
    winter_model = load_model(WINTER_MODEL)
    print("  Winter: loaded")
    summer_model = load_model(SUMMER_MODEL)
    print("  Summer: loaded")

    results = []
    start_all = time.time()

    for name, t_start in SCENARIOS.items():
        try:
            results.append(run_scenario(winter_model, summer_model, name, t_start))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append(
                {
                    "name": name,
                    "viol": None,
                    "energy": None,
                    "ms": None,
                    "t_min": None,
                    "t_max": None,
                    "t_mean": None,
                    "winter_pct": None,
                    "emergency_pct": None,
                    "gate_switches": None,
                }
            )

    elapsed = (time.time() - start_all) / 60.0

    print(f"\n{'=' * 75}")
    print(f"HDRL + EMERGENCY HEATING ({elapsed:.1f} min)")
    print(f"{'=' * 75}")
    print(
        f"{'Scenario':25s} {'Viol%':>7s} {'T_min':>7s} {'T_max':>7s} "
        f"{'E_kWh':>7s} {'m_s':>7s} {'W%':>4s} {'E%':>4s} {'GS':>4s}"
    )
    print("-" * 75)

    valid = [row for row in results if row["viol"] is not None]
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "hdrl_yearly_summary.csv"), index=False)
    for row in results:
        if row["viol"] is not None:
            print(
                f"{row['name']:25s} {row['viol']:7.1f} {row['t_min']:7.1f} "
                f"{row['t_max']:7.1f} {row['energy']:7.1f} {row['ms']:7.3f} "
                f"{row['winter_pct']:4.0f} {row['emergency_pct']:4.0f} {row['gate_switches']:4.0f}"
            )
        else:
            print(f"{row['name']:25s}  FAILED")

    if valid:
        viols = [row["viol"] for row in valid]
        energies = [row["energy"] for row in valid]
        ms_vals = [row["ms"] for row in valid]
        print("-" * 75)
        print(
            f"{'MEAN':25s} {np.mean(viols):7.1f} {'':7s} {'':7s} "
            f"{np.mean(energies):7.1f} {np.mean(ms_vals):7.3f}"
        )
        print(
            f"{'STD':25s} {np.std(viols):7.1f} {'':7s} {'':7s} "
            f"{np.std(energies):7.1f} {np.std(ms_vals):7.3f}"
        )

    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
