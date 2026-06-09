from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.tsup_features import action_to_fan, action_to_t_supply
from surrogate.direct_tsup_adapter import load_direct_tsup_adapter


COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0

DEFAULT_DATA = REPO_ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_prepared.csv"
DEFAULT_V3_MODEL = REPO_ROOT / "outputs" / "surrogate_v2" / "rc_node_v3_tsupply.pt"
DEFAULT_V35_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "surrogate_v35_inverse_boptest_15min_power_head_only"
    / "calibration_summary_boptest_v35.json"
)
DEFAULT_OUTPUT_TABLE = REPO_ROOT / "reports" / "speed_benchmark_table.csv"


@dataclass(frozen=True)
class StepSchedule:
    frame: pd.DataFrame

    def row(self, index: int) -> pd.Series:
        return self.frame.iloc[index % len(self.frame)]


class BOPTESTClient:
    def __init__(
        self,
        base_url: str,
        testcase_id: str,
        step_sec: int,
        timeout_sec: float,
        select_timeout_sec: float,
        retries: int,
        backoff_base_sec: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.testcase_id = testcase_id
        self.step_sec = int(step_sec)
        self.timeout_sec = float(timeout_sec)
        self.select_timeout_sec = float(select_timeout_sec)
        self.retries = int(retries)
        self.backoff_base_sec = float(backoff_base_sec)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = float(timeout if timeout is not None else self.timeout_sec)
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
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_base_sec * (2**attempt))
        raise RuntimeError(f"BOPTEST request failed after {self.retries} attempts: {method} {url}") from last_error

    def probe(self) -> None:
        self._request_json("GET", "/version", timeout=min(self.timeout_sec, 20.0))

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

    def initialize(self, testid: str, start_time_sec: float, warmup_sec: float = 0.0) -> None:
        self._request_json("PUT", f"/step/{testid}", payload={"step": self.step_sec}, timeout=30.0)
        self._request_json(
            "PUT",
            f"/initialize/{testid}",
            payload={"start_time": float(start_time_sec), "warmup_period": float(warmup_sec)},
            timeout=self.select_timeout_sec,
        )

    def advance(self, testid: str, actions: dict[str, Any]) -> dict[str, Any]:
        data = self._request_json("POST", f"/advance/{testid}", payload=actions)
        return data.get("payload", data)

    def stop(self, testid: str) -> None:
        try:
            self._request_json("PUT", f"/stop/{testid}", payload={}, timeout=10.0)
        except Exception:
            pass


def build_bestest_air_command(a0: float, a1: float) -> dict[str, Any]:
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": action_to_fan(float(a1)),
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": action_to_t_supply(float(a0)) + 273.15,
    }


def load_schedule(path: Path) -> StepSchedule:
    frame = pd.read_csv(path)
    required = ["sim_time_sec", "t_zone", "t_amb", "hour", "day", "a0_raw", "a1_raw"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Schedule file is missing required columns: {missing}")
    return StepSchedule(frame=frame[required].dropna().reset_index(drop=True))


def summarize_timings(
    *,
    backend: str,
    status: str,
    episodes: int,
    steps_per_episode: int,
    step_sec: int,
    step_times: list[float],
    episode_times: list[float],
    total_time_sec: float,
    device: str,
    torch_num_threads: int,
    boptest_url: str,
    notes: str = "",
) -> dict[str, Any]:
    steps_completed = len(step_times)
    step_ms = np.asarray(step_times, dtype=float) * 1000.0
    episode_sec = np.asarray(episode_times, dtype=float)
    env_steps_per_sec = steps_completed / total_time_sec if total_time_sec > 0 and steps_completed else np.nan
    return {
        "backend": backend,
        "status": status,
        "episodes_requested": episodes,
        "steps_per_episode": steps_per_episode,
        "steps_completed": steps_completed,
        "step_sec": step_sec,
        "total_time_sec": total_time_sec,
        "env_steps_per_sec": env_steps_per_sec,
        "mean_raw_step_ms": float(np.mean(step_ms)) if steps_completed else np.nan,
        "median_raw_step_ms": float(np.median(step_ms)) if steps_completed else np.nan,
        "p95_raw_step_ms": float(np.percentile(step_ms, 95)) if steps_completed else np.nan,
        "mean_episode_time_sec": float(np.mean(episode_sec)) if len(episode_sec) else np.nan,
        "device": device,
        "torch_num_threads": torch_num_threads,
        "boptest_url": boptest_url,
        "notes": notes,
    }


def benchmark_surrogate(
    *,
    backend: str,
    kind: str,
    schedule: StepSchedule,
    episodes: int,
    steps_per_episode: int,
    step_sec: int,
    v3_model: Path,
    v35_summary: Path,
    device: str,
    torch_num_threads: int,
    warmup_steps: int,
    boptest_url: str,
    verbose: bool,
) -> dict[str, Any]:
    if verbose:
        print(f"[surrogate] loading {backend} ({kind})", flush=True)
    adapter = load_direct_tsup_adapter(
        kind=kind,
        legacy_model_path=v3_model,
        summary_json=v35_summary,
        runtime_step_sec=step_sec,
        legacy_step_sec=3600,
        device=device,
    )

    warm_t_zone = float(schedule.row(0)["t_zone"])
    for warm_idx in range(max(0, warmup_steps)):
        row = schedule.row(warm_idx)
        out = adapter.step_with_aux_numpy(
            t_zone=warm_t_zone,
            t_amb=float(row["t_amb"]),
            hour=float(row["hour"]),
            day=float(row["day"]),
            a0=float(row["a0_raw"]),
            a1=float(row["a1_raw"]),
            device=device,
        )
        warm_t_zone = float(out["t_next"])

    step_times: list[float] = []
    episode_times: list[float] = []
    total_started = time.perf_counter()
    for episode in range(episodes):
        if verbose and (episode == 0 or (episode + 1) % 10 == 0 or episode + 1 == episodes):
            print(f"[surrogate] {backend}: episode {episode + 1}/{episodes}", flush=True)
        episode_started = time.perf_counter()
        base_index = episode * steps_per_episode
        t_zone = float(schedule.row(base_index)["t_zone"])
        for step in range(steps_per_episode):
            row = schedule.row(base_index + step)
            started = time.perf_counter()
            out = adapter.step_with_aux_numpy(
                t_zone=t_zone,
                t_amb=float(row["t_amb"]),
                hour=float(row["hour"]),
                day=float(row["day"]),
                a0=float(row["a0_raw"]),
                a1=float(row["a1_raw"]),
                device=device,
            )
            step_times.append(time.perf_counter() - started)
            t_zone = float(out["t_next"])
        episode_times.append(time.perf_counter() - episode_started)
    total_time_sec = time.perf_counter() - total_started

    return summarize_timings(
        backend=backend,
        status="ok",
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        step_sec=step_sec,
        step_times=step_times,
        episode_times=episode_times,
        total_time_sec=total_time_sec,
        device=device,
        torch_num_threads=torch_num_threads,
        boptest_url=boptest_url,
        notes="in-process CPU surrogate rollout; model load and warmup excluded",
    )


def benchmark_boptest(
    *,
    schedule: StepSchedule,
    episodes: int,
    steps_per_episode: int,
    step_sec: int,
    boptest_url: str,
    testcase_id: str,
    timeout_sec: float,
    select_timeout_sec: float,
    retries: int,
    backoff_base_sec: float,
    device: str,
    torch_num_threads: int,
    verbose: bool,
) -> dict[str, Any]:
    client = BOPTESTClient(
        base_url=boptest_url,
        testcase_id=testcase_id,
        step_sec=step_sec,
        timeout_sec=timeout_sec,
        select_timeout_sec=select_timeout_sec,
        retries=retries,
        backoff_base_sec=backoff_base_sec,
    )
    if verbose:
        print(f"[boptest] probing {boptest_url}/version", flush=True)
    client.probe()
    step_times: list[float] = []
    episode_times: list[float] = []
    total_started = time.perf_counter()
    for episode in range(episodes):
        if verbose:
            print(f"[boptest] episode {episode + 1}/{episodes}: selecting testcase", flush=True)
        episode_started = time.perf_counter()
        base_index = episode * steps_per_episode
        testid = client.select_testcase()
        try:
            if verbose:
                print(f"[boptest] episode {episode + 1}/{episodes}: initializing {testid}", flush=True)
            client.initialize(testid, start_time_sec=float(schedule.row(base_index)["sim_time_sec"]), warmup_sec=0.0)
            for step in range(steps_per_episode):
                if verbose and (step == 0 or (step + 1) % 24 == 0 or step + 1 == steps_per_episode):
                    print(f"[boptest] episode {episode + 1}/{episodes}: advance {step + 1}/{steps_per_episode}", flush=True)
                row = schedule.row(base_index + step)
                actions = build_bestest_air_command(float(row["a0_raw"]), float(row["a1_raw"]))
                started = time.perf_counter()
                client.advance(testid, actions)
                step_times.append(time.perf_counter() - started)
        finally:
            client.stop(testid)
        episode_times.append(time.perf_counter() - episode_started)
    total_time_sec = time.perf_counter() - total_started

    return summarize_timings(
        backend="boptest_rte_http",
        status="ok",
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        step_sec=step_sec,
        step_times=step_times,
        episode_times=episode_times,
        total_time_sec=total_time_sec,
        device=device,
        torch_num_threads=torch_num_threads,
        boptest_url=boptest_url,
        notes="BOPTEST RTE HTTP API; episode time includes select, step setup, initialize, advances, and stop",
    )


def failed_row(
    *,
    backend: str,
    episodes: int,
    steps_per_episode: int,
    step_sec: int,
    device: str,
    torch_num_threads: int,
    boptest_url: str,
    error: Exception,
) -> dict[str, Any]:
    row = summarize_timings(
        backend=backend,
        status="failed",
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        step_sec=step_sec,
        step_times=[],
        episode_times=[],
        total_time_sec=np.nan,
        device=device,
        torch_num_threads=torch_num_threads,
        boptest_url=boptest_url,
        notes=str(error),
    )
    return row


def add_speedups(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table["speedup_vs_boptest_rte"] = np.nan
    ref = table.loc[(table["backend"] == "boptest_rte_http") & (table["status"] == "ok"), "env_steps_per_sec"]
    if len(ref) == 1 and float(ref.iloc[0]) > 0:
        ref_steps_per_sec = float(ref.iloc[0])
        ok = table["status"] == "ok"
        table.loc[ok, "speedup_vs_boptest_rte"] = table.loc[ok, "env_steps_per_sec"] / ref_steps_per_sec
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark live BOPTEST RTE HTTP stepping against local v3/v3.5/hybrid surrogate stepping."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=96)
    parser.add_argument("--step-sec", type=int, default=900)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--surrogate-path", type=Path, default=DEFAULT_V3_MODEL)
    parser.add_argument("--surrogate-summary-json", type=Path, default=DEFAULT_V35_SUMMARY)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--testcase-id", default="bestest_air")
    parser.add_argument("--skip-boptest", action="store_true")
    parser.add_argument("--boptest-timeout-sec", type=float, default=30.0)
    parser.add_argument("--boptest-select-timeout-sec", type=float, default=120.0)
    parser.add_argument("--boptest-retries", type=int, default=1)
    parser.add_argument("--boptest-backoff-base-sec", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise ValueError("This benchmark is intended for honest single-machine CPU comparison; use --device cpu.")
    torch.set_num_threads(int(args.torch_num_threads))

    schedule = load_schedule(args.data)
    rows: list[dict[str, Any]] = []
    verbose = not args.quiet

    if not args.skip_boptest:
        try:
            rows.append(
                benchmark_boptest(
                    schedule=schedule,
                    episodes=args.episodes,
                    steps_per_episode=args.steps_per_episode,
                    step_sec=args.step_sec,
                    boptest_url=args.boptest_url,
                    testcase_id=args.testcase_id,
                    timeout_sec=args.boptest_timeout_sec,
                    select_timeout_sec=args.boptest_select_timeout_sec,
                    retries=args.boptest_retries,
                    backoff_base_sec=args.boptest_backoff_base_sec,
                    device=args.device,
                    torch_num_threads=args.torch_num_threads,
                    verbose=verbose,
                )
            )
        except Exception as exc:
            rows.append(
                failed_row(
                    backend="boptest_rte_http",
                    episodes=args.episodes,
                    steps_per_episode=args.steps_per_episode,
                    step_sec=args.step_sec,
                    device=args.device,
                    torch_num_threads=args.torch_num_threads,
                    boptest_url=args.boptest_url,
                    error=exc,
                )
            )

    for backend, kind in [
        ("v3_surrogate", "legacy_v3"),
        ("v35_calibrated_surrogate", "v35_calibrated"),
        ("hybrid_v3_v35_surrogate", "hybrid_v3_v35"),
    ]:
        try:
            rows.append(
                benchmark_surrogate(
                    backend=backend,
                    kind=kind,
                    schedule=schedule,
                    episodes=args.episodes,
                    steps_per_episode=args.steps_per_episode,
                    step_sec=args.step_sec,
                    v3_model=args.surrogate_path,
                    v35_summary=args.surrogate_summary_json,
                    device=args.device,
                    torch_num_threads=args.torch_num_threads,
                    warmup_steps=args.warmup_steps,
                    boptest_url=args.boptest_url,
                    verbose=verbose,
                )
            )
        except Exception as exc:
            rows.append(
                failed_row(
                    backend=backend,
                    episodes=args.episodes,
                    steps_per_episode=args.steps_per_episode,
                    step_sec=args.step_sec,
                    device=args.device,
                    torch_num_threads=args.torch_num_threads,
                    boptest_url=args.boptest_url,
                    error=exc,
                )
            )

    table = add_speedups(rows)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved speed benchmark table: {args.output_table}")


if __name__ == "__main__":
    main()
