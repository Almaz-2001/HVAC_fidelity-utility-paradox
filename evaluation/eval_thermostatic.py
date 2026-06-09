"""
evaluation/eval_thermostatic.py

BOPTEST validation for direct-TSup thermostatic control.

Usage:
    PYTHONPATH=/app python3 evaluation/eval_thermostatic.py
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

sys.path.insert(0, "/app")

BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
MODEL_PATH = "models/ppo_thermostatic.zip"
OUTPUT_DIR = "outputs"
T_TARGET = 22.0
STEP_SEC = 3600
SCENARIO_DAYS = 14
SELECT_TIMEOUT = 300
FORECAST_HORIZONS = [1, 3, 6, 12, 24]
N_PHYS = 5
N_TIME = 4
N_FORECAST = len(FORECAST_HORIZONS)
N_HISTORY = 3
N_OBS_M1 = N_PHYS + N_TIME + N_HISTORY
N_OBS_M23 = N_PHYS + N_TIME + N_FORECAST + N_HISTORY

PHYS_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
PHYS_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)
T_AMB_LOW, T_AMB_HIGH = -30.0, 45.0
DELTA_T_MAX = 5.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0
WEATHER_CSV_PRIMARY = "/app/data/surrogate_v2/boptest_v2_tsupply.csv"
WEATHER_CSV_FALLBACK = "/app/data/surrogate_v2/boptest_v2_all.csv"

SCENARIOS = {
    "Jan_Winter": 0,           # 0
    "Feb_Winter": 2678400,     # 31
    "Mar_Spring": 5097600,     # 59
    "Apr_Spring": 7776000,     # 90
    "May_Spring": 10368000,    # 120
    "Jun_Summer": 13132800,    # 151
    "Jul_Summer": 15552000,    # 181
    "Aug_Summer": 18316800,    # 212
    "Sep_Autumn": 20995200,    # 243
    "Oct_Autumn": 23587200,    # 273
    "Nov_Autumn": 26265600,    # 304
    "Dec_Winter": 28857600,    # 334
}

weather_grid = np.zeros((366, 24), dtype=np.float32)
weather_count = np.zeros((366, 24), dtype=np.int32)
_csv = WEATHER_CSV_PRIMARY if os.path.exists(WEATHER_CSV_PRIMARY) else WEATHER_CSV_FALLBACK
if os.path.exists(_csv):
    _df = pd.read_csv(_csv)
    for _, row in _df.iterrows():
        d = int(row["day"]) % 366
        h = int(row["hour"]) % 24
        t = float(row["t_amb"])
        if -30.0 < t < 50.0:
            weather_grid[d, h] += t
            weather_count[d, h] += 1
    mask = weather_count > 0
    if np.any(mask):
        weather_grid[mask] /= weather_count[mask]
        for d in range(366):
            for h in range(24):
                if weather_count[d, h] == 0:
                    for off in range(1, 30):
                        dp = (d - off) % 366
                        dn = (d + off) % 366
                        if weather_count[dp, h] > 0:
                            weather_grid[d, h] = weather_grid[dp, h]
                            break
                        if weather_count[dn, h] > 0:
                            weather_grid[d, h] = weather_grid[dn, h]
                            break
    else:
        weather_grid.fill(10.0)
else:
    weather_grid.fill(10.0)


def weather_forecast(hour, day, horizon):
    fh = (hour + horizon) % 24
    fd = day + horizon / 24.0
    return float(weather_grid[int(fd) % 366, int(fh) % 24])


def norm_t(t):
    return np.clip(2.0 * (t - T_AMB_LOW) / (T_AMB_HIGH - T_AMB_LOW) - 1.0, -1.0, 1.0)


def norm_delta_t(delta_t):
    return np.clip(delta_t / DELTA_T_MAX, -1.0, 1.0)


def action_to_t_supply(a0):
    return T_SUPPLY_LOW + (a0 + 1.0) * 0.5 * (T_SUPPLY_HIGH - T_SUPPLY_LOW)


def action_to_fan(a1):
    return float(np.clip((a1 + 1.0) * 0.5, 0.0, 1.0))


def encode_hour_cyc(hour):
    rad = 2.0 * np.pi * (hour / 24.0)
    return np.sin(rad), np.cos(rad)


def encode_day_cyc(day):
    rad = 2.0 * np.pi * (day / 365.0)
    return np.sin(rad), np.cos(rad)


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
            print(f"  Retry {attempt + 1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed: {url}")


def get_val(payload, key):
    v = payload.get(key, 0.0)
    return float(v.get("value", v) if isinstance(v, dict) else v)


def make_obs(payload, sim_time, prev_action, prev_t_zone=None, obs_dim=N_OBS_M23):
    t_c = get_val(payload, "zon_reaTRooAir_y") - 273.15
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    p_heat = get_val(payload, "fcu_reaPHea_y")
    t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
    t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

    hour = (sim_time / 3600.0) % 24.0
    day = (sim_time / 86400.0) % 365.0

    p_total = p_cool + p_fan + p_heat
    prev_t_supply = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 0.5 * (T_SUPPLY_LOW + T_SUPPLY_HIGH)
    phys = np.array([t_c, co2, p_total, prev_t_supply, t_amb], dtype=np.float32)
    phys_norm = np.clip(2.0 * (phys - PHYS_LOW) / (PHYS_HIGH - PHYS_LOW) - 1.0, -1.0, 1.0)

    hour_sin, hour_cos = encode_hour_cyc(hour)
    day_sin, day_cos = encode_day_cyc(day)
    include_forecast = obs_dim == N_OBS_M23
    forecasts = [norm_t(weather_forecast(hour, day, h)) for h in FORECAST_HORIZONS] if include_forecast else []
    delta_t = 0.0 if prev_t_zone is None else (t_c - prev_t_zone)

    blocks = [
        phys_norm,
        [hour_sin, hour_cos, day_sin, day_cos],
    ]
    if include_forecast:
        blocks.append(forecasts)
    blocks.extend(
        [
            np.array(prev_action, dtype=np.float32),
            [norm_delta_t(delta_t)],
        ]
    )
    obs = np.concatenate(blocks).astype(np.float32)
    return obs, t_c, t_amb, p_total


def action_to_boptest(action):
    a0, a1 = float(action[0]), float(action[1])
    t_supply = action_to_t_supply(a0)
    fan_u = action_to_fan(a1)
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


def run_scenario(model, name, start_time, obs_dim):
    print(f"\n  {name}:", end=" ")
    data = boptest_request("POST", f"/testcases/{TESTCASE}/select", timeout=SELECT_TIMEOUT)
    testid = data["testid"]
    boptest_request("PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request(
        "PUT",
        f"/initialize/{testid}",
        {"start_time": start_time, "warmup_period": 0},
        timeout=SELECT_TIMEOUT,
    )

    payload = boptest_request("POST", f"/advance/{testid}", {})
    payload = payload.get("payload", payload)
    sim_time = start_time + STEP_SEC
    obs, t_zone, t_amb, p_total = make_obs(payload, sim_time, np.zeros(2, dtype=np.float32), obs_dim=obs_dim)

    errors, temps, powers, ambs, tsups = [], [], [], [], []
    trace_rows = []

    steps_per_scenario = int(round(SCENARIO_DAYS * 86400 / STEP_SEC))

    for _ in range(steps_per_scenario):
        action, _ = model.predict(obs, deterministic=True)
        u = action_to_boptest(action)
        prev_sim_time = sim_time
        prev_t_zone = t_zone
        prev_t_amb = t_amb
        prev_p_total = p_total
        payload = boptest_request("POST", f"/advance/{testid}", u)
        payload = payload.get("payload", payload)
        sim_time += STEP_SEC
        obs, t_zone, t_amb, p_total = make_obs(payload, sim_time, action, prev_t_zone, obs_dim=obs_dim)
        a0 = float(action[0])
        a1 = float(action[1])
        t_supply = action_to_t_supply(a0)
        fan_u = action_to_fan(a1)

        errors.append(abs(t_zone - T_TARGET))
        temps.append(t_zone)
        powers.append(p_total)
        ambs.append(t_amb)
        tsups.append(t_supply)
        trace_rows.append(
            {
                "step": len(errors) - 1,
                "prev_time": prev_sim_time,
                "time": sim_time,
                "prev_t_zone": prev_t_zone,
                "prev_t_amb": prev_t_amb,
                "prev_p_total": prev_p_total,
                "a0": a0,
                "a1": a1,
                "fan_u": fan_u,
                "t_supply": t_supply,
                "t_zone": t_zone,
                "abs_error": abs(t_zone - T_TARGET),
                "p_total": p_total,
                "t_amb": t_amb,
            }
        )

    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass

    errors = np.array(errors)
    temps = np.array(temps)
    detail_df = pd.DataFrame(trace_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detail_df.to_csv(os.path.join(OUTPUT_DIR, f"thermostatic_scenario_{name}.csv"), index=False)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(errors)
    within_1 = (errors < 1.0).mean() * 100
    within_05 = (errors < 0.5).mean() * 100
    energy = np.sum(powers) * (STEP_SEC / 3600.0) / 1000

    print(
        f"RMSE={rmse:.2f}C | +-1C={within_1:.0f}% | +-0.5C={within_05:.0f}% | "
        f"E={energy:.0f}kWh | T=[{temps.min():.1f},{temps.max():.1f}]"
    )

    return {
        "name": name,
        "rmse": rmse,
        "mae": mae,
        "within_1": within_1,
        "within_05": within_05,
        "t_min": temps.min(),
        "t_max": temps.max(),
        "energy": energy,
    }


def main():
    global MODEL_PATH, OUTPUT_DIR, STEP_SEC, SCENARIO_DAYS, BOPTEST_URL, TESTCASE

    parser = argparse.ArgumentParser(description="BOPTEST validation for direct-TSup thermostatic control.")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--step-sec", type=int, default=STEP_SEC)
    parser.add_argument("--scenario-days", type=int, default=SCENARIO_DAYS)
    parser.add_argument("--boptest-url", default=BOPTEST_URL)
    parser.add_argument("--testcase", default=TESTCASE)
    args = parser.parse_args()

    MODEL_PATH = args.model
    OUTPUT_DIR = args.output_dir
    STEP_SEC = int(args.step_sec)
    SCENARIO_DAYS = int(args.scenario_days)
    BOPTEST_URL = args.boptest_url
    TESTCASE = args.testcase

    print("=" * 70)
    print(f"THERMOSTATIC v4 BOPTEST VALIDATION (target={T_TARGET}C)")
    print(f"Control: direct TSup in [{T_SUPPLY_LOW}, {T_SUPPLY_HIGH}]C")
    print("=" * 70)

    try:
        r = boptest_request("GET", "/version", timeout=10)
        print(f"BOPTEST: {r['payload']['version']}")
    except Exception as e:
        print(f"BOPTEST not available: {e}")
        return

    z = zipfile.ZipFile(MODEL_PATH)
    data = json.loads(z.read("data"))
    obs_dim = data.get("observation_space", {}).get("_shape", [N_OBS_M23])[0]
    if obs_dim not in (N_OBS_M1, N_OBS_M23):
        raise RuntimeError(
            f"Model observation dim is {obs_dim}, but evaluator expects one of ({N_OBS_M1}, {N_OBS_M23}). "
            "Retrain the thermostatic policy with an Article 22 compatible observation layout."
        )
    include_forecast = obs_dim == N_OBS_M23
    variant = "M.2/M.3-like" if include_forecast else "M.1-like"
    print(f"Obs: {obs_dim}F | Forecast enabled: {include_forecast} | Variant: {variant}")

    model = PPO.load(
        MODEL_PATH,
        device="cpu",
        custom_objects={
            "action_space": spaces.Box(-1, 1, (2,), np.float32),
            "observation_space": spaces.Box(-1, 1, (obs_dim,), np.float32),
            "clip_range": lambda _: 0.2,
            "lr_schedule": lambda _: 3e-4,
        },
    )
    print(f"Model loaded (obs={obs_dim})")

    results = []
    t0 = time.time()

    for name, st in SCENARIOS.items():
        try:
            results.append(run_scenario(model, name, st, obs_dim))
        except Exception as e:
            print(f"FAILED: {e}")

    elapsed = (time.time() - t0) / 60

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({elapsed:.1f} min)")
    print(f"{'=' * 70}")
    print(f"{'Scenario':20s} {'RMSE':>6s} {'MAE':>6s} {'+-1C':>5s} {'+-0.5C':>6s} {'E_kWh':>6s}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:20s} {r['rmse']:6.2f} {r['mae']:6.2f} "
            f"{r['within_1']:5.0f}% {r['within_05']:5.0f}% {r['energy']:6.0f}"
        )

    if results:
        summary_df = pd.DataFrame(results)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "thermostatic_yearly_summary.csv"), index=False)
        summer = [r for r in results if r["name"] in ("Jun_Summer", "Jul_Summer", "Sep_Autumn")]
        winter = [r for r in results if "Winter" in r["name"]]

        print("-" * 60)
        print(
            f"{'MEAN (all)':20s} {np.mean([r['rmse'] for r in results]):6.2f} "
            f"{np.mean([r['mae'] for r in results]):6.2f} "
            f"{np.mean([r['within_1'] for r in results]):5.0f}% "
            f"{np.mean([r['within_05'] for r in results]):5.0f}%"
        )
        if summer:
            print(
                f"{'MEAN (summer)':20s} {np.mean([r['rmse'] for r in summer]):6.2f} "
                f"{np.mean([r['mae'] for r in summer]):6.2f} "
                f"{np.mean([r['within_1'] for r in summer]):5.0f}% "
                f"{np.mean([r['within_05'] for r in summer]):5.0f}%"
            )
        if winter:
            print(
                f"{'MEAN (winter)':20s} {np.mean([r['rmse'] for r in winter]):6.2f} "
                f"{np.mean([r['mae'] for r in winter]):6.2f} "
                f"{np.mean([r['within_1'] for r in winter]):5.0f}% "
                f"{np.mean([r['within_05'] for r in winter]):5.0f}%"
            )

        print(f"\n{'=' * 70}")
        print("COMPARISON WITH GAO ET AL.")
        print(f"{'=' * 70}")
        summer_rmse = np.mean([r["rmse"] for r in summer]) if summer else 0.0
        print("  Gao M.3 (PPO+GRU):    RMSE ~= 0.5C (office building)")
        print(f"  Ours (summer mean):   RMSE = {summer_rmse:.2f}C")
        print(f"  Ours (best):          RMSE = {min([r['rmse'] for r in results]):.2f}C")
        print("  Forecast horizons:    1, 3, 6, 12, 24h")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
