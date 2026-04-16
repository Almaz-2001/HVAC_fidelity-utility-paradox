from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return data


def k_to_c(value: float) -> float:
    return float(value) - 273.15 if float(value) > 200.0 else float(value)


def flatten_boptest_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return float(value)


class BOPTESTClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["url"]).rstrip("/")
        self.testcase_id = str(config["testcase_id"])
        self.step_sec = int(config["step_sec"])
        self.warmup_period_sec = float(config.get("warmup_period_sec", 0.0))
        self.timeout_sec = float(config.get("timeout_sec", 60.0))
        self.select_timeout_sec = float(config.get("select_timeout_sec", max(300.0, self.timeout_sec)))
        self.retries = int(config.get("retries", 3))
        self.backoff_base_sec = float(config.get("backoff_base_sec", 1.0))
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def check_connectivity(self) -> dict[str, Any]:
        try:
            data = self._request_json("GET", "/version", timeout=min(self.timeout_sec, 10.0))
            return data.get("payload", data)
        except requests.ConnectionError as exc:
            host_hint = self.base_url
            raise RuntimeError(
                "Cannot reach BOPTEST. "
                f"Configured URL is '{host_hint}'. "
                "If you are not running inside the docker-compose network, "
                "the hostname 'web' will not resolve. "
                "Pass --boptest-url with the real endpoint, for example "
                "'http://127.0.0.1:5000' or the host:port where BOPTEST is exposed."
            ) from exc

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        timeout = float(timeout if timeout is not None else self.timeout_sec)
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                if method == "GET":
                    response = self.session.get(url, timeout=timeout)
                elif method == "POST":
                    response = self.session.post(url, json=payload or {}, timeout=timeout)
                elif method == "PUT":
                    response = self.session.put(url, json=payload or {}, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code in (500, 502, 503, 504):
                    time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))
                    continue

                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = exc
                time.sleep(min(self.backoff_base_sec * (2**attempt), 8.0))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"BOPTEST request failed without explicit exception: {url}")

    def select_testcase(self) -> str:
        data = self._request_json(
            "POST",
            f"/testcases/{self.testcase_id}/select",
            payload={},
            timeout=self.select_timeout_sec,
        )
        testid = data.get("testid")
        if not testid:
            raise RuntimeError(f"Could not obtain testid from BOPTEST response: {data}")
        return str(testid)

    def initialize(self, testid: str, start_time_sec: float) -> None:
        self._request_json("PUT", f"/step/{testid}", payload={"step": self.step_sec}, timeout=30.0)
        self._request_json(
            "PUT",
            f"/initialize/{testid}",
            payload={"start_time": float(start_time_sec), "warmup_period": self.warmup_period_sec},
            timeout=self.select_timeout_sec,
        )

    def advance(self, testid: str, actions: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._request_json("POST", f"/advance/{testid}", payload=actions or {})
        return data.get("payload", data)

    def get_kpis(self, testid: str) -> dict[str, Any]:
        data = self._request_json("GET", f"/kpi/{testid}")
        return data.get("payload", data)

    def stop(self, testid: str) -> None:
        try:
            self._request_json("PUT", f"/stop/{testid}", payload={}, timeout=10.0)
        except Exception:
            pass


def load_article7_schedule(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"time", "Occupancy[1]", "LowerSetp[1]", "UpperSetp[1]"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing schedule columns in {csv_path}: {sorted(missing)}")

    df = df[["time", "Occupancy[1]", "LowerSetp[1]", "UpperSetp[1]"]].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["Occupancy[1]"] = pd.to_numeric(df["Occupancy[1]"], errors="coerce")
    df["LowerSetp[1]"] = pd.to_numeric(df["LowerSetp[1]"], errors="coerce").map(k_to_c)
    df["UpperSetp[1]"] = pd.to_numeric(df["UpperSetp[1]"], errors="coerce").map(k_to_c)
    df = df.dropna().reset_index(drop=True)
    return df


def load_local_days_json(days_json_path: Path) -> dict[str, int]:
    with days_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected mapping in {days_json_path}, got: {type(payload).__name__}")

    result: dict[str, int] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid day-of-year for '{key}' in {days_json_path}: {value}") from exc
    return result


def resolve_scenario_start_day(
    config: dict[str, Any],
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    local_days: dict[str, int] | None,
) -> tuple[int, str]:
    source = str(config.get("scenario_day_source", "config")).strip().lower()
    fallback = int(scenario_cfg["start_day_of_year"])

    if source == "local_days_json":
        if local_days is None:
            raise RuntimeError(
                "scenario_day_source is 'local_days_json' but no local days mapping was loaded."
            )
        if scenario_name not in local_days:
            raise RuntimeError(
                f"Scenario '{scenario_name}' is missing in local days mapping. "
                f"Available keys: {sorted(local_days)}"
            )
        return int(local_days[scenario_name]), "local_days_json"

    if source != "config":
        raise RuntimeError(
            f"Unsupported scenario_day_source: {source}. Expected 'config' or 'local_days_json'."
        )

    return fallback, "config"


def expected_bounds(schedule_df: pd.DataFrame, sim_time_sec: float) -> tuple[float, float, float]:
    times = schedule_df["time"].to_numpy(dtype=float)
    idx = int(np.searchsorted(times, sim_time_sec, side="right") - 1)
    idx = max(0, min(idx, len(schedule_df) - 1))
    row = schedule_df.iloc[idx]
    return (
        float(row["LowerSetp[1]"]),
        float(row["UpperSetp[1]"]),
        float(row["Occupancy[1]"]),
    )


def scenario_start_time_sec(start_day_of_year: int) -> float:
    return float(max(start_day_of_year - 1, 0) * 24 * 3600)


def compute_safety_metrics(trace_df: pd.DataFrame) -> dict[str, float]:
    temps = trace_df["t_zone_c"].to_numpy(dtype=float)
    lower = trace_df["t_low_c"].to_numpy(dtype=float)
    upper = trace_df["t_high_c"].to_numpy(dtype=float)

    below = temps < lower
    above = temps > upper
    violation = below | above

    r_time = float(np.mean(violation))
    under = np.where(below, (lower - temps) / np.maximum(lower, 1e-6), 0.0)
    over = np.where(above, (temps - upper) / np.maximum(upper, 1e-6), 0.0)
    r_sev = float(max(np.max(under), np.max(over)))
    m_s = float(r_time + r_sev)

    return {
        "r_time": r_time,
        "r_sev": r_sev,
        "m_s": m_s,
        "violation_pct": float(r_time * 100.0),
        "t_min_c": float(np.min(temps)),
        "t_max_c": float(np.max(temps)),
        "t_mean_c": float(np.mean(temps)),
        "energy_kwh": float(np.sum(trace_df["p_total_w"].to_numpy(dtype=float)) * 0.25 / 1000.0),
    }


def plot_trace(trace_df: pd.DataFrame, scenario_name: str, article_period: str, out_path: Path) -> None:
    x_days = trace_df["sim_time_sec"].to_numpy(dtype=float) / 86400.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.fill_between(x_days, trace_df["t_low_c"], trace_df["t_high_c"], color="#dff3e4", alpha=0.8, label="Safety band")
    ax1.plot(x_days, trace_df["t_zone_c"], color="#c0392b", linewidth=2.0, label="Zone temperature")
    ax1.set_ylabel("Temperature, C")
    ax1.set_title(f"{scenario_name} | {article_period}")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    ax2.plot(x_days, trace_df["p_total_w"], color="#2563eb", linewidth=1.8, label="Electric power")
    ax2.set_ylabel("Power, W")
    ax2.set_xlabel("Simulation time, days")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, ref_cfg: dict[str, Any], out_path: Path) -> None:
    scenarios = summary_df["scenario"].tolist()
    x = np.arange(len(scenarios))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, summary_df["m_s"], width=width, color="#2563eb", label="This run")

    pi_refs = [ref_cfg.get("pi", {}).get(s, np.nan) for s in scenarios]
    mpc_refs = [ref_cfg.get("mpc", {}).get(s, np.nan) for s in scenarios]
    ax.bar(x, pi_refs, width=width, color="#7f8c8d", label="Article 7 PI")
    ax.bar(x + width, mpc_refs, width=width, color="#16a085", label="Article 7 MPC")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("m_s safety metric")
    ax.set_title("Article 7 safety reproduction summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pi_reproduction(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = REPO_ROOT / config["paths"]["comfort_schedule_csv"]
    schedule_df = load_article7_schedule(schedule_path)
    local_days_path_cfg = config.get("paths", {}).get("local_days_json")
    local_days = None
    if local_days_path_cfg:
        local_days_path = REPO_ROOT / str(local_days_path_cfg)
        local_days = load_local_days_json(local_days_path)
    client = BOPTESTClient(config["boptest"])
    version_payload = client.check_connectivity()
    print(f"[BOPTEST] Connected: {version_payload}")

    summary_rows: list[dict[str, Any]] = []

    for scenario_name, scenario_cfg in config["scenarios"].items():
        label = str(scenario_cfg["label"])
        article_period = str(scenario_cfg["article_period"])
        start_day, start_day_source = resolve_scenario_start_day(
            config,
            scenario_name,
            scenario_cfg,
            local_days,
        )
        duration_days = int(scenario_cfg["duration_days"])
        start_time_sec = scenario_start_time_sec(start_day)
        total_steps = int(duration_days * 24 * 3600 / client.step_sec)

        print(f"\n{'=' * 72}")
        print(f"ARTICLE 7 SCENARIO: {scenario_name} | {label}")
        print(f"Period: {article_period}")
        print(
            f"Start day-of-year: {start_day} | Source: {start_day_source} | Steps: {total_steps}"
        )
        print(f"{'=' * 72}")

        testid = client.select_testcase()
        client.initialize(testid, start_time_sec)
        payload = client.advance(testid, {})

        rows: list[dict[str, Any]] = []

        for step in range(total_steps):
            sim_time_sec = flatten_boptest_value(payload, "time", start_time_sec + step * client.step_sec)
            t_low_c, t_high_c, occupancy = expected_bounds(schedule_df, sim_time_sec)
            t_zone_c = k_to_c(flatten_boptest_value(payload, "reaTZon_y"))
            co2_ppm = flatten_boptest_value(payload, "reaCO2RooAir_y")
            p_fan_w = flatten_boptest_value(payload, "reaPFan_y")
            p_heat_pump_w = flatten_boptest_value(payload, "reaPHeaPum_y")
            p_pump_w = flatten_boptest_value(payload, "reaPPumEmi_y")
            t_sup_c = k_to_c(flatten_boptest_value(payload, "reaTSup_y"))
            p_total_w = p_fan_w + p_heat_pump_w + p_pump_w

            below = max(t_low_c - t_zone_c, 0.0)
            above = max(t_zone_c - t_high_c, 0.0)

            rows.append(
                {
                    "step": step,
                    "sim_time_sec": sim_time_sec,
                    "t_zone_c": t_zone_c,
                    "t_low_c": t_low_c,
                    "t_high_c": t_high_c,
                    "occupancy_signal": occupancy,
                    "co2_ppm": co2_ppm,
                    "p_fan_w": p_fan_w,
                    "p_heat_pump_w": p_heat_pump_w,
                    "p_pump_w": p_pump_w,
                    "p_total_w": p_total_w,
                    "t_supply_c": t_sup_c,
                    "undershoot_c": below,
                    "overshoot_c": above,
                    "violates": bool((t_zone_c < t_low_c) or (t_zone_c > t_high_c)),
                }
            )

            if (step + 1) % 192 == 0:
                print(f"  Step {step + 1:4d}/{total_steps} | Tz={t_zone_c:5.2f} C | band=[{t_low_c:4.1f},{t_high_c:4.1f}] | P={p_total_w:6.1f} W")

            if step + 1 < total_steps:
                payload = client.advance(testid, {})

        kpis = client.get_kpis(testid)
        client.stop(testid)

        trace_df = pd.DataFrame(rows)
        metrics = compute_safety_metrics(trace_df)
        article_refs = config.get("references", {}).get("article7_table5", {})
        summary_row = {
            "controller": "boptest_pi",
            "scenario": scenario_name,
            "label": label,
            "article_period": article_period,
            "start_day_of_year": start_day,
            "start_day_source": start_day_source,
            "start_time_sec": start_time_sec,
            "duration_days": duration_days,
            "step_sec": client.step_sec,
            "total_steps": total_steps,
            "article7_pi_m_s": article_refs.get("pi", {}).get(scenario_name),
            "article7_mpc_m_s": article_refs.get("mpc", {}).get(scenario_name),
            **metrics,
        }
        for key, value in (kpis or {}).items():
            if isinstance(value, dict):
                summary_row[f"kpi_{key}"] = value.get("value")
            else:
                summary_row[f"kpi_{key}"] = value
        summary_rows.append(summary_row)

        trace_path = output_dir / f"{scenario_name}_trace.csv"
        trace_df.to_csv(trace_path, index=False)
        plot_path = output_dir / f"{scenario_name}_trace.png"
        plot_trace(trace_df, label, article_period, plot_path)
        print(
            f"  RESULT | m_s={metrics['m_s']:.4f} | r_time={metrics['r_time']:.4f} | "
            f"r_sev={metrics['r_sev']:.4f} | viol={metrics['violation_pct']:.1f}% | "
            f"E={metrics['energy_kwh']:.2f} kWh"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_summary(
        summary_df,
        config.get("references", {}).get("article7_table5", {}),
        output_dir / "summary_ms_comparison.png",
    )
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Article 7 hydronic safety benchmark on BOPTEST.")
    parser.add_argument(
        "--config",
        default="configs/article7_hydronic.yaml",
        help="Path to Article 7 hydronic benchmark config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/article7_hydronic_pi",
        help="Directory for traces, summaries and plots.",
    )
    parser.add_argument(
        "--boptest-url",
        default=None,
        help="Override BOPTEST base URL, for example http://127.0.0.1:5000.",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    output_dir = REPO_ROOT / args.output_dir
    config = load_yaml(config_path)
    env_boptest_url = os.environ.get("BOPTEST_URL")
    if args.boptest_url or env_boptest_url:
        config.setdefault("boptest", {})
        config["boptest"]["url"] = args.boptest_url or env_boptest_url

    summary_df = run_pi_reproduction(config, output_dir)
    print(f"\nSaved summary: {output_dir / 'summary.csv'}")
    print(summary_df[["scenario", "m_s", "r_time", "r_sev", "violation_pct", "energy_kwh"]].to_string(index=False))


if __name__ == "__main__":
    main()
