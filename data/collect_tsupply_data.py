"""
data/collect_tsupply_data.py

Collect BOPTEST data with direct T_supply control.
4 seasons × 4 policies × 3200 steps = 51,200 samples.

Usage:
    PYTHONPATH=/app python3 data/collect_tsupply_data.py
"""

import os
import time
import random
import requests
import numpy as np
import pandas as pd


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
STEP_SEC = 3600
STEPS_PER_EPISODE = 3200
OUTPUT_PATH = "data/surrogate_v2/boptest_v2_tsupply.csv"

# T_supply range for data collection
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0

SEASONS = {
    "winter": 0,
    "spring": 7776000,
    "summer": 15552000,
    "autumn": 23328000,
}

POLICIES = ["random", "heat", "cool", "mixed"]

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def request(method, path, payload=None, timeout=120):
    url = f"{BOPTEST_URL}{path}"
    for attempt in range(3):
        try:
            if method == "POST":
                r = session.post(url, json=payload or {}, timeout=timeout)
            elif method == "PUT":
                r = session.put(url, json=payload or {}, timeout=timeout)
            else:
                r = session.get(url, timeout=timeout)
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def gv(payload, key):
    v = payload.get(key, 0.0)
    return float(v.get("value", v) if isinstance(v, dict) else v)


def generate_action(policy):
    if policy == "random":
        a0 = random.uniform(-1, 1)
        a1 = random.uniform(-1, 1)
    elif policy == "heat":
        a0 = random.uniform(0.3, 1.0)
        a1 = random.uniform(0.5, 1.0)
    elif policy == "cool":
        a0 = random.uniform(-1.0, -0.3)
        a1 = random.uniform(0.5, 1.0)
    else:  # mixed
        a0 = random.uniform(-0.5, 0.5)
        a1 = random.uniform(0.2, 0.8)
    return a0, a1


def collect_episode(season_name, start_time, policy):
    print(f"\n  {season_name}/{policy}: selecting testcase...")

    data = request("POST", f"/testcases/{TESTCASE}/select", timeout=300)
    tid = data["testid"]

    request("PUT", f"/step/{tid}", {"step": STEP_SEC}, timeout=30)
    request("PUT", f"/initialize/{tid}",
            {"start_time": start_time, "warmup_period": 0}, timeout=300)

    # Initial advance
    r = request("POST", f"/advance/{tid}", {})
    payload = r.get("payload", r)

    rows = []
    t0 = time.time()

    for step in range(STEPS_PER_EPISODE):
        t_zone = gv(payload, "zon_reaTRooAir_y") - 273.15
        t_amb = gv(payload, "zon_weaSta_reaWeaTDryBul_y") - 273.15
        sim_time = gv(payload, "time")
        hour = (sim_time / 3600) % 24
        day = (sim_time / 86400) % 365

        a0, a1 = generate_action(policy)

        # T_supply mapping
        t_supply = T_SUPPLY_LOW + (a0 + 1.0) * 0.5 * (T_SUPPLY_HIGH - T_SUPPLY_LOW)
        fan_u = max(0.0, min(1.0, (a1 + 1.0) * 0.5))

        u = {
            "con_oveTSetCoo_activate": 1, "con_oveTSetCoo_u": 313.15,
            "con_oveTSetHea_activate": 1, "con_oveTSetHea_u": 288.15,
            "fcu_oveFan_activate": 1, "fcu_oveFan_u": fan_u,
            "fcu_oveTSup_activate": 1, "fcu_oveTSup_u": t_supply + 273.15,
        }

        r = request("POST", f"/advance/{tid}", u)
        payload = r.get("payload", r)

        t_next = gv(payload, "zon_reaTRooAir_y") - 273.15
        p_cool = gv(payload, "fcu_reaPCoo_y")
        p_fan = gv(payload, "fcu_reaPFan_y")

        rows.append({
            "step": step,
            "t_zone": round(t_zone, 4),
            "t_amb": round(t_amb, 1),
            "hour": round(hour, 2),
            "day": round(day, 2),
            "a0_raw": round(a0, 5),
            "a1_raw": round(a1, 5),
            "t_zone_next": round(t_next, 4),
            "delta_t": round(t_next - t_zone, 4),
            "p_total": round(p_cool + p_fan, 2),
            "policy": policy,
            "season": season_name,
        })

        if step % 200 == 0:
            elapsed = time.time() - t0
            fps = (step + 1) / max(elapsed, 0.1)
            print(f"    step {step:4d} | T={t_zone:.1f}C | Tsup={t_supply:.0f}C "
                  f"| Tnext={t_next:.1f}C | T_amb={t_amb:.1f}C | {fps:.1f} fps")

    # Stop session
    try:
        request("PUT", f"/stop/{tid}", timeout=10)
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"    Done: {len(rows)} steps in {elapsed:.0f}s ({len(rows)/elapsed:.1f} fps)")
    return rows


def main():
    print("=" * 60)
    print("COLLECTING T_SUPPLY CONTROL DATA FROM BOPTEST")
    print(f"Seasons: {list(SEASONS.keys())}")
    print(f"Policies: {POLICIES}")
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    print(f"Total: {len(SEASONS) * len(POLICIES) * STEPS_PER_EPISODE:,} steps")
    print(f"T_supply range: [{T_SUPPLY_LOW}, {T_SUPPLY_HIGH}]°C")
    print("=" * 60)

    # Check BOPTEST
    try:
        r = request("GET", "/version", timeout=10)
        print(f"BOPTEST version: {r['payload']['version']}")
    except Exception as e:
        print(f"BOPTEST not available: {e}")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_data = []
    t_total = time.time()

    for season_name, start_time in SEASONS.items():
        for policy in POLICIES:
            try:
                rows = collect_episode(season_name, start_time, policy)
                all_data.extend(rows)

                # Save intermediate
                df = pd.DataFrame(all_data)
                df.to_csv(OUTPUT_PATH, index=False)
                print(f"  Saved {len(all_data)} rows (intermediate)")

            except Exception as e:
                print(f"  FAILED {season_name}/{policy}: {e}")
                import traceback; traceback.print_exc()

    elapsed = (time.time() - t_total) / 60
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Time: {elapsed:.1f} min")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"T_zone range: [{df['t_zone'].min():.1f}, {df['t_zone'].max():.1f}]°C")
    print(f"T_amb range:  [{df['t_amb'].min():.1f}, {df['t_amb'].max():.1f}]°C")
    print(f"P_total range: [{df['p_total'].min():.0f}, {df['p_total'].max():.0f}]W")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()