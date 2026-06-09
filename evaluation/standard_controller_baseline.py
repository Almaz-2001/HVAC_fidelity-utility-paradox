"""
evaluation/standard_controller_baseline.py

Yearly validation for the built-in BOPTEST controller without any overrides.

Reports two views:
1. Fixed-target metrics against 22 C, for direct comparison with the PPO baselines.
2. Schedule-aware comfort metrics against the known occupied/unoccupied thermostat schedule.

Usage:
    PYTHONPATH=/app python3 evaluation/standard_controller_baseline.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "/app")


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
OUTPUT_DIR = "outputs"
STEP_SEC = 3600
STEPS_PER_SCENARIO = 336
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60

BUILDING_AREA_M2 = 48.0
FIXED_TARGET_C = 22.0
FIXED_LOW_C = 21.0
FIXED_HIGH_C = 25.0

SCHEDULE_CSV = os.path.join(
    "boptest_rte",
    "testcases",
    TESTCASE,
    "models",
    "Resources",
    "internal_setpoints_occupancy.csv",
)

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
_SCHEDULE_DF = None


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


def advance(testid, actions=None):
    data = boptest_request("POST", f"/advance/{testid}", actions or {}, timeout=ADVANCE_TIMEOUT)
    return data.get("payload", data)


def stop(testid):
    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass


def get_val(payload, key):
    value = payload.get(key, 0.0)
    return float(value.get("value", value) if isinstance(value, dict) else value)


def k_to_c(value):
    return value - 273.15 if value > 200 else value


def load_expected_schedule():
    global _SCHEDULE_DF
    if _SCHEDULE_DF is not None:
        return _SCHEDULE_DF

    schedule_df = pd.read_csv(SCHEDULE_CSV, skiprows=8)
    needed = ["time", "LowerSetp[1]", "UpperSetp[1]", "Occupancy[1]"]
    if not set(needed).issubset(set(schedule_df.columns)):
        raise RuntimeError(f"Schedule CSV missing required columns: {SCHEDULE_CSV}")

    schedule_df = schedule_df[needed].copy()
    for column in needed:
        schedule_df[column] = pd.to_numeric(schedule_df[column], errors="coerce")
    schedule_df = schedule_df.dropna().reset_index(drop=True)
    schedule_df["LowerSetp[1]"] = schedule_df["LowerSetp[1]"] - 273.15
    schedule_df["UpperSetp[1]"] = schedule_df["UpperSetp[1]"] - 273.15
    _SCHEDULE_DF = schedule_df
    return _SCHEDULE_DF


def expected_schedule_setpoints(sim_time_seconds):
    schedule_df = load_expected_schedule()
    idx = int(np.searchsorted(schedule_df["time"].to_numpy(), sim_time_seconds, side="right") - 1)
    idx = int(np.clip(idx, 0, len(schedule_df) - 1))
    row = schedule_df.iloc[idx]
    return float(row["LowerSetp[1]"]), float(row["UpperSetp[1]"]), bool(row["Occupancy[1]"] > 0.0)


def parse_payload(payload):
    sim_time = get_val(payload, "time")
    t_zone = k_to_c(get_val(payload, "zon_reaTRooAir_y"))
    co2 = get_val(payload, "zon_reaCO2RooAir_y")
    p_cool = get_val(payload, "fcu_reaPCoo_y")
    p_fan = get_val(payload, "fcu_reaPFan_y")
    p_heat = get_val(payload, "fcu_reaPHea_y")
    t_amb = k_to_c(get_val(payload, "zon_weaSta_reaWeaTDryBul_y"))
    t_set_cool_raw = k_to_c(get_val(payload, "con_oveTSetCoo_y"))
    t_set_heat_raw = k_to_c(get_val(payload, "con_oveTSetHea_y"))
    t_set_heat_sched, t_set_cool_sched, occupied = expected_schedule_setpoints(sim_time)
    raw_valid = (
        5.0 <= t_set_heat_raw <= 35.0
        and 5.0 <= t_set_cool_raw <= 35.0
        and t_set_cool_raw > t_set_heat_raw
    )
    p_total = p_cool + p_fan + p_heat
    return {
        "time": sim_time,
        "hour": (sim_time / 3600.0) % 24.0,
        "occupied": occupied,
        "t_zone": t_zone,
        "co2": co2,
        "t_amb": t_amb,
        "p_cool": p_cool,
        "p_fan": p_fan,
        "p_heat": p_heat,
        "p_total": p_total,
        "t_set_cool_raw": t_set_cool_raw,
        "t_set_heat_raw": t_set_heat_raw,
        "t_set_cool_sched": t_set_cool_sched,
        "t_set_heat_sched": t_set_heat_sched,
        "raw_setpoints_valid": raw_valid,
    }


def compute_fixed_metrics(df):
    errors = np.abs(df["t_zone"].to_numpy() - FIXED_TARGET_C)
    temps = df["t_zone"].to_numpy()
    return {
        "rmse_22c": float(np.sqrt(np.mean(errors ** 2))),
        "mae_22c": float(np.mean(errors)),
        "within_1c_22c": float((errors < 1.0).mean() * 100.0),
        "within_05c_22c": float((errors < 0.5).mean() * 100.0),
        "viol_21_25_pct": float(((temps < FIXED_LOW_C) | (temps > FIXED_HIGH_C)).mean() * 100.0),
    }


def compute_schedule_metrics(df):
    temps = df["t_zone"].to_numpy()
    t_low = df["t_set_heat_sched"].to_numpy()
    t_high = df["t_set_cool_sched"].to_numpy()
    below = temps < t_low
    above = temps > t_high
    violation = below | above
    r_time = float(violation.mean())
    under = np.where(below, (t_low - temps) / np.maximum(t_low, 1e-6), 0.0)
    over = np.where(above, (temps - t_high) / np.maximum(t_high, 1e-6), 0.0)
    r_sev = float(max(np.max(under), np.max(over)))
    return {
        "schedule_viol_pct": float(r_time * 100.0),
        "schedule_r_time": r_time,
        "schedule_r_sev": r_sev,
        "schedule_m_s": float(r_time + r_sev),
    }


def compare_to_baseline(reference_df, candidate_path, candidate_name):
    if not os.path.exists(candidate_path):
        return None
    candidate_df = pd.read_csv(candidate_path)
    needed = {"name", "energy"}
    if not needed.issubset(set(candidate_df.columns)):
        return None

    merged = reference_df[["name", "energy_kwh", "energy_kwh_m2"]].merge(
        candidate_df[["name", "energy"]], on="name", how="inner"
    )
    if merged.empty:
        return None

    merged["candidate"] = candidate_name
    merged["candidate_energy_kwh"] = merged["energy"]
    merged["candidate_energy_kwh_m2"] = merged["candidate_energy_kwh"] / BUILDING_AREA_M2
    merged["savings_kwh"] = merged["energy_kwh"] - merged["candidate_energy_kwh"]
    merged["savings_pct"] = 100.0 * merged["savings_kwh"] / np.maximum(merged["energy_kwh"], 1e-6)
    return merged[
        [
            "candidate",
            "name",
            "energy_kwh",
            "energy_kwh_m2",
            "candidate_energy_kwh",
            "candidate_energy_kwh_m2",
            "savings_kwh",
            "savings_pct",
        ]
    ]


def run_scenario(name, start_time):
    print(f"\n{'=' * 60}")
    print(f"STANDARD CONTROLLER: {name} (start={start_time}s)")
    print(f"{'=' * 60}")

    testid = select_testcase()
    print(f"  testid: {testid}")
    initialize(testid, start_time)

    payload = advance(testid, {})
    rows = []

    for step in range(STEPS_PER_SCENARIO):
        parsed = parse_payload(payload)
        parsed["step"] = step
        rows.append(parsed)

        if step % 48 == 0:
            print(
                f"  Step {step:3d} | T={parsed['t_zone']:.1f}C | "
                f"Sched=[{parsed['t_set_heat_sched']:.1f},{parsed['t_set_cool_sched']:.1f}]C | "
                f"P={parsed['p_total']:.0f}W | T_amb={parsed['t_amb']:.1f}C"
            )

        payload = advance(testid, {})

    stop(testid)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, f"standard_controller_scenario_{name}.csv"), index=False)

    fixed_metrics = compute_fixed_metrics(df)
    schedule_metrics = compute_schedule_metrics(df)
    energy = float(df["p_total"].sum() / 1000.0)
    energy_kwh_m2 = energy / BUILDING_AREA_M2

    print(
        f"  RESULT fixed22: RMSE={fixed_metrics['rmse_22c']:.2f}C | "
        f"+-1C={fixed_metrics['within_1c_22c']:.0f}% | E={energy:.0f}kWh"
    )
    print(
        f"  RESULT schedule: viol={schedule_metrics['schedule_viol_pct']:.1f}% | "
        f"m_s={schedule_metrics['schedule_m_s']:.3f}"
    )

    return {
        "name": name,
        "energy_kwh": energy,
        "energy_kwh_m2": energy_kwh_m2,
        "t_min": float(df["t_zone"].min()),
        "t_max": float(df["t_zone"].max()),
        "t_mean": float(df["t_zone"].mean()),
        **fixed_metrics,
        **schedule_metrics,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 104)
    print("STANDARD BOPTEST CONTROLLER YEARLY VALIDATION")
    print("No override signals are sent. The built-in testcase thermostat/PI controller is used.")
    print(f"Area normalization: {BUILDING_AREA_M2:.1f} m^2")
    print("=" * 104)

    try:
        response = boptest_request("GET", "/version", timeout=10)
        print(f"BOPTEST version: {response['payload']['version']}")
    except Exception as exc:
        print(f"BOPTEST not available: {exc}")
        return

    results = []
    start_all = time.time()

    for name, start_time in SCENARIOS.items():
        try:
            results.append(run_scenario(name, start_time))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append(
                {
                    "name": name,
                    "energy_kwh": None,
                    "energy_kwh_m2": None,
                    "t_min": None,
                    "t_max": None,
                    "t_mean": None,
                    "rmse_22c": None,
                    "mae_22c": None,
                    "within_1c_22c": None,
                    "within_05c_22c": None,
                    "viol_21_25_pct": None,
                    "schedule_viol_pct": None,
                    "schedule_r_time": None,
                    "schedule_r_sev": None,
                    "schedule_m_s": None,
                }
            )

    elapsed = (time.time() - start_all) / 60.0
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(OUTPUT_DIR, "standard_controller_yearly_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    valid = summary_df.dropna(subset=["energy_kwh"])

    print(f"\n{'=' * 104}")
    print(f"STANDARD CONTROLLER RESULTS ({elapsed:.1f} min)")
    print(f"{'=' * 104}")
    print(
        f"{'Scenario':20s} {'RMSE22':>7s} {'Viol21-25':>10s} {'SchedViol':>10s} "
        f"{'Sched_m_s':>9s} {'E_kWh':>8s} {'kWh/m2':>8s}"
    )
    print("-" * 104)
    for _, row in summary_df.iterrows():
        if pd.notna(row["energy_kwh"]):
            print(
                f"{row['name']:20s} {row['rmse_22c']:7.2f} {row['viol_21_25_pct']:10.1f} "
                f"{row['schedule_viol_pct']:10.1f} {row['schedule_m_s']:9.3f} "
                f"{row['energy_kwh']:8.1f} {row['energy_kwh_m2']:8.3f}"
            )
        else:
            print(f"{row['name']:20s} FAILED")

    if not valid.empty:
        print("-" * 104)
        print(
            f"{'MEAN':20s} {valid['rmse_22c'].mean():7.2f} {valid['viol_21_25_pct'].mean():10.1f} "
            f"{valid['schedule_viol_pct'].mean():10.1f} {valid['schedule_m_s'].mean():9.3f} "
            f"{valid['energy_kwh'].mean():8.1f} {valid['energy_kwh_m2'].mean():8.3f}"
        )
        print(
            f"{'STD':20s} {valid['rmse_22c'].std():7.2f} {valid['viol_21_25_pct'].std():10.1f} "
            f"{valid['schedule_viol_pct'].std():10.1f} {valid['schedule_m_s'].std():9.3f} "
            f"{valid['energy_kwh'].std():8.1f} {valid['energy_kwh_m2'].std():8.3f}"
        )
        print("\nInterpretation:")
        print("  RMSE22 / Viol21-25 are for direct comparison with the constant-22C PPO baselines.")
        print("  SchedViol / Sched_m_s use the testcase schedule from Resources/internal_setpoints_occupancy.csv.")

        comparisons = []
        thermo_cmp = compare_to_baseline(summary_df, os.path.join(OUTPUT_DIR, "thermostatic_yearly_summary.csv"), "thermostatic")
        if thermo_cmp is not None:
            comparisons.append(thermo_cmp)
        hdrl_cmp = compare_to_baseline(summary_df, os.path.join(OUTPUT_DIR, "hdrl_yearly_summary.csv"), "hdrl")
        if hdrl_cmp is not None:
            comparisons.append(hdrl_cmp)

        if comparisons:
            compare_df = pd.concat(comparisons, ignore_index=True)
            compare_path = os.path.join(OUTPUT_DIR, "standard_controller_energy_comparison.csv")
            compare_df.to_csv(compare_path, index=False)
            print("\nEnergy comparison versus standard PI baseline:")
            for candidate in compare_df["candidate"].unique():
                subset = compare_df[compare_df["candidate"] == candidate]
                baseline_total = subset["energy_kwh"].sum()
                candidate_total = subset["candidate_energy_kwh"].sum()
                baseline_total_m2 = subset["energy_kwh_m2"].sum()
                candidate_total_m2 = subset["candidate_energy_kwh_m2"].sum()
                delta_pct = 100.0 * (baseline_total - candidate_total) / max(baseline_total, 1e-6)
                print(
                    f"  {candidate:12s}: PI={baseline_total:.1f} kWh ({baseline_total_m2:.3f} kWh/m2), "
                    f"candidate={candidate_total:.1f} kWh ({candidate_total_m2:.3f} kWh/m2), "
                    f"savings={delta_pct:+.1f}%"
                )
            print(f"  Saved comparison CSV: {compare_path}")

    print(f"{'=' * 104}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
