from __future__ import annotations

import argparse
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3 import PPO


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
STEP_SEC = 3600
SCENARIO_DAYS = 14
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60
N_OBS = 5

STATE_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
STATE_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)

T_TARGET = 22.0
T_LOW = 21.0
T_HIGH = 25.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0

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


def action_to_t_supply(a0):
    return T_SUPPLY_LOW + (float(a0) + 1.0) * 0.5 * (T_SUPPLY_HIGH - T_SUPPLY_LOW)


def action_to_fan(a1):
    return float(np.clip((float(a1) + 1.0) * 0.5, 0.0, 1.0))


def make_obs(payload, prev_action):
    t_c = get_val(payload, "zon_reaTRooAir_y") - 273.15
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    p_heat = get_val(payload, "fcu_reaPHea_y")
    t_amb_k = get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
    t_amb = t_amb_k - 273.15 if t_amb_k > 200 else t_amb_k

    p_total = p_cool + p_fan + p_heat
    prev_t_supply = action_to_t_supply(prev_action[0]) if prev_action is not None else 0.5 * (
        T_SUPPLY_LOW + T_SUPPLY_HIGH
    )
    raw = np.array([t_c, co2, p_total, prev_t_supply, t_amb], dtype=np.float32)
    obs = 2.0 * (raw - STATE_LOW) / (STATE_HIGH - STATE_LOW) - 1.0
    return np.clip(obs, -1.0, 1.0), t_c, p_total, t_amb


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


def compute_metrics(temps: np.ndarray, powers: np.ndarray) -> dict[str, float]:
    errors = np.abs(temps - T_TARGET)
    rmse = float(np.sqrt(np.mean((temps - T_TARGET) ** 2)))
    mae = float(np.mean(errors))
    within_1 = float(np.mean(errors < 1.0) * 100.0)
    within_05 = float(np.mean(errors < 0.5) * 100.0)
    r_time = float(np.mean((temps < T_LOW) | (temps > T_HIGH)))
    over = float(np.maximum((temps - T_HIGH) / T_HIGH, 0.0).max())
    under = float(np.maximum((T_LOW - temps) / T_LOW, 0.0).max())
    return {
        "rmse": rmse,
        "mae": mae,
        "within_1c_pct": within_1,
        "within_05c_pct": within_05,
        "viol_pct": r_time * 100.0,
        "energy_kwh": float(powers.sum() * (STEP_SEC / 3600.0) / 1000.0),
        "ms": float(r_time + max(over, under)),
        "t_min": float(temps.min()),
        "t_max": float(temps.max()),
        "t_mean": float(temps.mean()),
    }


def run_scenario(model, name, start_time, output_dir):
    print(f"\n{'=' * 72}")
    print(f"MORL SCENARIO: {name} (start={start_time}s)")
    print(f"{'=' * 72}")

    testid = select_testcase()
    initialize(testid, start_time)
    payload = advance(testid, {})

    prev_action = np.zeros(2, dtype=np.float32)
    obs, t_zone, p_total, t_amb = make_obs(payload, prev_action)

    history = {"t_zone": [], "p_total": [], "t_amb": [], "t_supply": [], "a0": [], "a1": []}
    steps_per_scenario = int(round(SCENARIO_DAYS * 86400 / STEP_SEC))
    log_interval = max(1, int(round(86400 / STEP_SEC)))

    for step in range(steps_per_scenario):
        action, _ = model.predict(obs, deterministic=True)
        payload = advance(testid, action_to_boptest(action))
        obs, t_zone, p_total, t_amb = make_obs(payload, action)

        history["t_zone"].append(t_zone)
        history["p_total"].append(p_total)
        history["t_amb"].append(t_amb)
        history["t_supply"].append(action_to_t_supply(action[0]))
        history["a0"].append(float(action[0]))
        history["a1"].append(float(action[1]))

        if step % log_interval == 0:
            print(
                f"  Step {step:3d} | T={t_zone:.1f}C | P={p_total:.0f}W | "
                f"T_amb={t_amb:.1f}C | TSup={history['t_supply'][-1]:.1f}C | "
                f"a=[{action[0]:.2f},{action[1]:.2f}]"
            )

        prev_action = np.asarray(action, dtype=np.float32)

    stop(testid)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, f"morl_scenario_{name}.csv"), index=False)

    metrics = compute_metrics(df["t_zone"].to_numpy(dtype=float), df["p_total"].to_numpy(dtype=float))
    print(
        f"  RESULT: RMSE={metrics['rmse']:.2f}C | MAE={metrics['mae']:.2f}C | "
        f"Viol={metrics['viol_pct']:.1f}% | E={metrics['energy_kwh']:.1f}kWh | m_s={metrics['ms']:.3f}"
    )
    return {"name": name, **metrics}


def main() -> None:
    global STEP_SEC, SCENARIO_DAYS, BOPTEST_URL, TESTCASE

    parser = argparse.ArgumentParser(description="Yearly BOPTEST validation for the MORL PPO controller.")
    parser.add_argument("--model", default="outputs/morl_boptest_finetune_seed42/models/ppo_model.zip")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--step-sec", type=int, default=STEP_SEC)
    parser.add_argument("--scenario-days", type=int, default=SCENARIO_DAYS)
    parser.add_argument("--boptest-url", default=BOPTEST_URL)
    parser.add_argument("--testcase", default=TESTCASE)
    args = parser.parse_args()

    STEP_SEC = int(args.step_sec)
    SCENARIO_DAYS = int(args.scenario_days)
    BOPTEST_URL = args.boptest_url
    TESTCASE = args.testcase

    os.makedirs(args.output_dir, exist_ok=True)

    print("Checking BOPTEST...")
    version = boptest_request("GET", "/version", timeout=10)
    print(f"BOPTEST version: {version['payload']['version']}")

    print(f"Loading MORL model: {args.model}")
    with zipfile.ZipFile(args.model) as archive:
        data = json.loads(archive.read("data"))
    obs_shape = data.get("observation_space", {}).get("_shape", [N_OBS])
    obs_dim = obs_shape[0] if isinstance(obs_shape, list) else obs_shape
    if obs_dim != N_OBS:
        raise RuntimeError(
            f"Model observation dim is {obs_dim}, but MORL yearly validation expects {N_OBS}. "
            "Pretrain and fine-tune MORL on the direct-TSup pipeline first."
        )

    model = PPO.load(
        args.model,
        device="cpu",
        custom_objects={
            "action_space": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "observation_space": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "clip_range": lambda _: 0.2,
            "lr_schedule": lambda _: 3e-4,
        },
    )

    results = []
    start_all = time.time()
    for name, start_time in SCENARIOS.items():
        results.append(run_scenario(model, name, start_time, args.output_dir))

    elapsed = (time.time() - start_all) / 60.0
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(args.output_dir, "morl_yearly_summary.csv"), index=False)

    print(f"\n{'=' * 86}")
    print(f"MORL YEARLY VALIDATION COMPLETE ({elapsed:.1f} min)")
    print(f"{'=' * 86}")
    print(f"{'Scenario':15s} {'RMSE':>6s} {'MAE':>6s} {'±1C':>6s} {'±0.5C':>7s} {'Viol%':>7s} {'E_kWh':>8s} {'m_s':>7s}")
    print("-" * 86)

    for row in results:
        print(
            f"{row['name']:15s} {row['rmse']:6.2f} {row['mae']:6.2f} "
            f"{row['within_1c_pct']:6.0f} {row['within_05c_pct']:7.0f} "
            f"{row['viol_pct']:7.1f} {row['energy_kwh']:8.1f} {row['ms']:7.3f}"
        )

    print("-" * 86)
    print(
        f"{'MEAN':15s} {summary['rmse'].mean():6.2f} {summary['mae'].mean():6.2f} "
        f"{summary['within_1c_pct'].mean():6.0f} {summary['within_05c_pct'].mean():7.0f} "
        f"{summary['viol_pct'].mean():7.1f} {summary['energy_kwh'].mean():8.1f} {summary['ms'].mean():7.3f}"
    )
    print("=" * 86)
    print(f"Saved summary: {os.path.join(args.output_dir, 'morl_yearly_summary.csv')}")


if __name__ == "__main__":
    main()
