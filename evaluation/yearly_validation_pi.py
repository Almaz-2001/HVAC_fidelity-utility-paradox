from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
STEP_SEC = 900
SCENARIO_DAYS = 14
SELECT_TIMEOUT = 300
ADVANCE_TIMEOUT = 60
HTTP_RETRIES = 3

T_TARGET = 22.0
T_LOW = 21.0
T_HIGH = 24.0

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


def boptest_request(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 60, retries: int = 3):
    url = f"{BOPTEST_URL}{path}"
    last_error: Exception | None = None
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
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            print(f"  Retry {attempt + 1}/{retries}: {exc}")
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed: {url}") from last_error


def select_testcase() -> str:
    data = boptest_request("POST", f"/testcases/{TESTCASE}/select", timeout=SELECT_TIMEOUT, retries=HTTP_RETRIES)
    testid = data.get("testid")
    if not testid:
        raise RuntimeError(f"No testid in response: {data}")
    return str(testid)


def initialize(testid: str, start_time: float) -> None:
    boptest_request("PUT", f"/step/{testid}", {"step": STEP_SEC}, timeout=30)
    boptest_request(
        "PUT",
        f"/initialize/{testid}",
        {"start_time": float(start_time), "warmup_period": 0},
        timeout=SELECT_TIMEOUT,
        retries=HTTP_RETRIES,
    )


def advance(testid: str, actions: dict[str, Any] | None = None) -> dict[str, Any]:
    data = boptest_request("POST", f"/advance/{testid}", actions or {}, timeout=ADVANCE_TIMEOUT, retries=HTTP_RETRIES)
    return data.get("payload", data)


def stop(testid: str) -> None:
    try:
        boptest_request("PUT", f"/stop/{testid}", timeout=10)
    except Exception:
        pass


def get_val(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    return float(value.get("value", value) if isinstance(value, dict) else value)


def get_first_val(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in payload:
            return get_val(payload, key)
    raise KeyError(f"None of the expected BOPTEST keys were found: {keys}")


def sum_existing_vals(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    found = [key for key in keys if key in payload]
    if not found:
        raise KeyError(f"None of the expected BOPTEST power keys were found: {keys}")
    return float(sum(get_val(payload, key) for key in found))


def k_to_c(value: float) -> float:
    value = float(value)
    return value - 273.15 if value > 200.0 else value


def parse_state(payload: dict[str, Any]) -> tuple[float, float, float]:
    t_zone = k_to_c(
        get_first_val(
            payload,
            (
                "zon_reaTRooAir_y",  # bestest_air
                "reaTZon_y",  # hydronic testcases
                "reaTRoo_y",  # bestest_hydronic
                "reaTRooAir_y",
            ),
        )
    )
    p_total = sum_existing_vals(
        payload,
        (
            "fcu_reaPCoo_y",  # bestest_air
            "fcu_reaPFan_y",
            "fcu_reaPHea_y",
            "reaPHeaPum_y",  # hydronic heat pump
            "reaPFan_y",
            "reaPPumEmi_y",
            "ahu_reaPFanExt_y",
            "ahu_reaPFanSup_y",
            "reaPHea_y",
            "reaQHea_y",
            "reaPCoo_y",
            "reaPPum_y",
        ),
    )
    t_amb = k_to_c(
        get_first_val(
            payload,
            (
                "zon_weaSta_reaWeaTDryBul_y",  # bestest_air
                "weaSta_reaWeaTDryBul_y",  # hydronic testcases
                "reaWeaTDryBul_y",
            ),
        )
    )
    return t_zone, p_total, t_amb


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


def run_scenario(name: str, start_time: float, output_dir: str) -> dict[str, float | str]:
    print(f"\n{'=' * 72}")
    print(f"PI SCENARIO: {name} (start={start_time}s)")
    print(f"{'=' * 72}")

    print("  Selecting testcase...")
    testid = select_testcase()
    print(f"  Selected testid={testid}")
    print("  Initializing testcase...")
    initialize(testid, start_time)
    print("  Initialization complete")

    history = {"t_zone": [], "p_total": [], "t_amb": []}
    steps_per_scenario = int(round(SCENARIO_DAYS * 86400 / STEP_SEC))
    log_interval = max(1, int(round(86400 / STEP_SEC)))

    try:
        for step in range(steps_per_scenario):
            payload = advance(testid, {})
            t_zone, p_total, t_amb = parse_state(payload)
            history["t_zone"].append(t_zone)
            history["p_total"].append(p_total)
            history["t_amb"].append(t_amb)

            if step % log_interval == 0:
                print(f"  Step {step:3d} | T={t_zone:.1f}C | P={p_total:.0f}W | T_amb={t_amb:.1f}C")
    finally:
        stop(testid)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, f"pi_scenario_{name}.csv"), index=False)
    metrics = compute_metrics(df["t_zone"].to_numpy(dtype=float), df["p_total"].to_numpy(dtype=float))
    print(
        f"  RESULT: RMSE={metrics['rmse']:.2f}C | MAE={metrics['mae']:.2f}C | "
        f"Viol={metrics['viol_pct']:.1f}% | E={metrics['energy_kwh']:.1f}kWh | m_s={metrics['ms']:.3f}"
    )
    return {"name": name, **metrics}


def load_existing_scenario(name: str, output_dir: str) -> dict[str, float | str] | None:
    path = os.path.join(output_dir, f"pi_scenario_{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"t_zone", "p_total"}
    if not required.issubset(df.columns):
        print(f"  Existing {path} is missing required columns; rerunning.")
        return None
    steps_per_scenario = int(round(SCENARIO_DAYS * 86400 / STEP_SEC))
    if len(df) < steps_per_scenario:
        print(f"  Existing {path} has {len(df)}/{steps_per_scenario} rows; rerunning.")
        return None
    metrics = compute_metrics(df["t_zone"].to_numpy(dtype=float), df["p_total"].to_numpy(dtype=float))
    print(f"  Reusing existing scenario {name}: {path}")
    return {"name": name, **metrics}


def main() -> None:
    global STEP_SEC, SCENARIO_DAYS, BOPTEST_URL, TESTCASE, T_LOW, T_HIGH, SELECT_TIMEOUT, ADVANCE_TIMEOUT, HTTP_RETRIES

    parser = argparse.ArgumentParser(description="Yearly BOPTEST validation for the built-in PI baseline.")
    parser.add_argument("--output_dir", default="outputs/pi_baseline_15min_yearly")
    parser.add_argument("--step-sec", type=int, default=STEP_SEC)
    parser.add_argument("--scenario-days", type=int, default=SCENARIO_DAYS)
    parser.add_argument("--temp-low", type=float, default=T_LOW)
    parser.add_argument("--temp-high", type=float, default=T_HIGH)
    parser.add_argument("--select-timeout", type=float, default=SELECT_TIMEOUT)
    parser.add_argument("--advance-timeout", type=float, default=ADVANCE_TIMEOUT)
    parser.add_argument("--http-retries", type=int, default=HTTP_RETRIES)
    parser.add_argument("--boptest-url", default=BOPTEST_URL)
    parser.add_argument("--testcase", "--testcase-id", dest="testcase", default=TESTCASE)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse complete per-scenario CSVs already in output_dir.")
    args = parser.parse_args()

    STEP_SEC = int(args.step_sec)
    SCENARIO_DAYS = int(args.scenario_days)
    T_LOW = float(args.temp_low)
    T_HIGH = float(args.temp_high)
    SELECT_TIMEOUT = float(args.select_timeout)
    ADVANCE_TIMEOUT = float(args.advance_timeout)
    HTTP_RETRIES = int(args.http_retries)
    BOPTEST_URL = args.boptest_url
    TESTCASE = args.testcase

    os.makedirs(args.output_dir, exist_ok=True)

    print("Checking BOPTEST...")
    version = boptest_request("GET", "/version", timeout=10)
    print(f"BOPTEST version: {version['payload']['version']}")

    results = []
    start_all = time.time()
    for name, start_time in SCENARIOS.items():
        if args.skip_existing:
            existing = load_existing_scenario(name, args.output_dir)
            if existing is not None:
                results.append(existing)
                continue
        results.append(run_scenario(name, start_time, args.output_dir))

    elapsed = (time.time() - start_all) / 60.0
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(args.output_dir, "pi_yearly_summary.csv"), index=False)

    print(f"\n{'=' * 86}")
    print(f"PI YEARLY VALIDATION COMPLETE ({elapsed:.1f} min)")
    print(f"{'=' * 86}")
    print(f"{'Scenario':15s} {'RMSE':>6s} {'MAE':>6s} {'+/-1C':>6s} {'+/-0.5C':>7s} {'Viol%':>7s} {'E_kWh':>8s} {'m_s':>7s}")
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
    print(f"Saved summary: {os.path.join(args.output_dir, 'pi_yearly_summary.csv')}")


if __name__ == "__main__":
    main()
