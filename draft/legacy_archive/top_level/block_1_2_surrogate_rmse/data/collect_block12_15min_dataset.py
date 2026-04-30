from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_1_2_surrogate_rmse.workflow_config import (
    BLOCK_NAME,
    COMFORT_CENTER_C,
    COMFORT_HIGH_C,
    COMFORT_LOW_C,
    DEFAULT_COLLECTED_DATASET_CSV,
    DEFAULT_COLLECTED_OUTPUT_DIR,
    DEFAULT_SAFE_T_ZONE_MAX_C,
    DEFAULT_SAFE_T_ZONE_MIN_C,
    DEFAULT_TRAIN_ABS_DELTA_T_MAX_C,
    DEFAULT_TRAIN_P_TOTAL_MAX_W,
    DEFAULT_TRAIN_T_ZONE_MAX_C,
    DEFAULT_TRAIN_T_ZONE_MIN_C,
    SURROGATE_STEP_SEC,
)
from envs.tsup_features import action_to_fan, action_to_t_supply


DEFAULT_BASE_URL = "http://web:8000"
DEFAULT_TESTCASE = "bestest_air"
DEFAULT_STEP_SEC = SURROGATE_STEP_SEC
DEFAULT_STEPS_PER_EPISODE = 2016  # 21 days at 15 min
DEFAULT_WARMUP_SEC = 86400  # 1 day
DEFAULT_SEASONS = ["winter", "spring", "summer", "autumn"]
DEFAULT_POLICIES = [
    "random",
    "heat",
    "cool",
    "mixed",
    "thermostatic_noise",
    "pulse",
]

DEFAULT_OUTPUT_CSV = DEFAULT_COLLECTED_DATASET_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_COLLECTED_OUTPUT_DIR

SEASON_START_TIMES_SEC = {
    "winter": 0.0,
    "spring": 90.0 * 86400.0,
    "summer": 180.0 * 86400.0,
    "autumn": 270.0 * 86400.0,
}

HEATING_SETPOINT_C = 15.0
COOLING_SETPOINT_C = 40.0
CONTROLLER_SOURCE = "surrogate_collect_15min"


@dataclass
class EpisodeSpec:
    season: str
    policy: str
    seed: int
    start_time_sec: float


@dataclass
class EpisodeResult:
    episode_id: str
    season: str
    policy: str
    seed: int
    rows: int
    step_sec: int
    t_zone_min_c: float
    t_zone_max_c: float
    t_amb_mean_c: float
    mean_power_w: float
    mean_abs_delta_t_c: float


@dataclass
class PolicyState:
    rng: np.random.Generator
    hold_steps_left: int = 0
    held_a0: float = 0.0
    held_a1: float = 0.0


class BOPTESTClient:
    def __init__(
        self,
        base_url: str,
        testcase: str,
        step_sec: int,
        timeout_sec: float = 60.0,
        select_timeout_sec: float = 300.0,
        retries: int = 3,
        backoff_base_sec: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.testcase = testcase
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

    def check_connectivity(self) -> dict[str, Any]:
        return self._request_json("GET", "/version", timeout=min(self.timeout_sec, 10.0))

    def select_testcase(self) -> str:
        data = self._request_json(
            "POST",
            f"/testcases/{self.testcase}/select",
            payload={},
            timeout=self.select_timeout_sec,
        )
        testid = data.get("testid")
        if not testid:
            raise RuntimeError(f"Could not obtain testid from BOPTEST response: {data}")
        return str(testid)

    def initialize(self, testid: str, start_time_sec: float, warmup_sec: float) -> None:
        self._request_json("PUT", f"/step/{testid}", payload={"step": int(self.step_sec)}, timeout=30.0)
        self._request_json(
            "PUT",
            f"/initialize/{testid}",
            payload={"start_time": float(start_time_sec), "warmup_period": float(warmup_sec)},
            timeout=self.select_timeout_sec,
        )

    def advance(self, testid: str, actions: dict[str, Any] | None = None) -> dict[str, Any]:
        data = self._request_json("POST", f"/advance/{testid}", payload=actions or {})
        return data.get("payload", data)

    def stop(self, testid: str) -> None:
        try:
            self._request_json("PUT", f"/stop/{testid}", payload={}, timeout=10.0)
        except Exception:
            pass


def flatten_boptest_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return float(value)


def k_to_c(value: float) -> float:
    value = float(value)
    return value - 273.15 if value > 200.0 else value


def clamp_action(a0: float, a1: float) -> tuple[float, float]:
    return float(np.clip(a0, -1.0, 1.0)), float(np.clip(a1, -1.0, 1.0))


def normalize_t_supply(t_supply_c: float) -> float:
    return float(2.0 * (float(t_supply_c) - 18.0) / (35.0 - 18.0) - 1.0)


def normalize_fan(fan_u: float) -> float:
    return float(2.0 * np.clip(float(fan_u), 0.0, 1.0) - 1.0)


def apply_safety_envelope(
    a0: float,
    a1: float,
    t_zone_c: float,
    t_amb_c: float,
    safe_t_zone_min_c: float,
    safe_t_zone_max_c: float,
) -> tuple[float, float]:
    if t_zone_c < safe_t_zone_min_c:
        error_c = safe_t_zone_min_c - t_zone_c
        t_supply_target_c = np.clip(
            25.5 + 2.0 * error_c + 0.05 * (COMFORT_CENTER_C - t_amb_c),
            COMFORT_LOW_C + 3.0,
            35.0,
        )
        fan_target_u = np.clip(0.7 + 0.08 * error_c, 0.5, 1.0)
        return clamp_action(normalize_t_supply(t_supply_target_c), normalize_fan(fan_target_u))

    if t_zone_c > safe_t_zone_max_c:
        error_c = t_zone_c - safe_t_zone_max_c
        t_supply_target_c = np.clip(20.0 - 1.2 * error_c, 18.0, COMFORT_LOW_C + 1.0)
        fan_target_u = np.clip(0.25 + 0.03 * error_c, 0.1, 0.55)
        return clamp_action(normalize_t_supply(t_supply_target_c), normalize_fan(fan_target_u))

    return clamp_action(a0, a1)


def band_tracking_action(
    t_zone_c: float,
    t_amb_c: float,
    rng: np.random.Generator,
    heating_focus: bool,
) -> tuple[float, float]:
    if t_zone_c < COMFORT_LOW_C:
        undershoot_c = COMFORT_LOW_C - t_zone_c
        weather_term_c = np.clip(COMFORT_CENTER_C - t_amb_c, -8.0, 18.0)
        base_tsup_c = 24.5 if heating_focus else 23.5
        t_supply_target_c = np.clip(
            base_tsup_c + 1.9 * undershoot_c + 0.10 * weather_term_c + rng.normal(0.0, 0.35 if heating_focus else 0.55),
            20.0,
            35.0,
        )
        base_fan_u = 0.32 if heating_focus else 0.28
        fan_target_u = np.clip(
            base_fan_u + 0.08 * undershoot_c + 0.01 * weather_term_c + rng.normal(0.0, 0.04 if heating_focus else 0.06),
            0.05,
            0.95,
        )
    elif t_zone_c > COMFORT_HIGH_C:
        overshoot_c = t_zone_c - COMFORT_HIGH_C
        t_supply_target_c = np.clip(
            20.5 - 0.8 * overshoot_c + rng.normal(0.0, 0.30 if heating_focus else 0.45),
            18.0,
            COMFORT_CENTER_C,
        )
        fan_target_u = np.clip(
            0.22 + 0.03 * overshoot_c + rng.normal(0.0, 0.03 if heating_focus else 0.05),
            0.0,
            0.65,
        )
    else:
        center_error_c = COMFORT_CENTER_C - t_zone_c
        weather_term_c = np.clip(COMFORT_LOW_C - t_amb_c, -6.0, 16.0)
        t_supply_target_c = np.clip(
            COMFORT_CENTER_C + 0.9 * center_error_c + 0.08 * weather_term_c + rng.normal(0.0, 0.20 if heating_focus else 0.30),
            19.0,
            29.0,
        )
        fan_target_u = np.clip(
            0.20 + 0.04 * center_error_c + 0.006 * weather_term_c + rng.normal(0.0, 0.02 if heating_focus else 0.04),
            0.05,
            0.75,
        )

    return clamp_action(normalize_t_supply(t_supply_target_c), normalize_fan(fan_target_u))


def build_bestest_air_command(a0: float, a1: float) -> tuple[dict[str, Any], float, float]:
    t_supply_c = action_to_t_supply(a0)
    fan_u = action_to_fan(a1)
    command = {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": fan_u,
        "fcu_oveTSup_activate": 1,
        "fcu_oveTSup_u": t_supply_c + 273.15,
    }
    return command, float(t_supply_c), float(fan_u)


def choose_action(policy: str, state: PolicyState, t_zone_c: float, t_amb_c: float) -> tuple[float, float]:
    rng = state.rng

    if policy == "random":
        return clamp_action(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))

    if policy == "heat":
        return clamp_action(rng.uniform(0.25, 1.0), rng.uniform(0.15, 1.0))

    if policy == "cool":
        return clamp_action(rng.uniform(-1.0, -0.25), rng.uniform(0.15, 1.0))

    if policy == "mixed":
        return clamp_action(rng.uniform(-0.55, 0.55), rng.uniform(-0.4, 0.7))

    if policy == "thermostatic_noise":
        return band_tracking_action(t_zone_c=t_zone_c, t_amb_c=t_amb_c, rng=rng, heating_focus=False)

    if policy == "pulse":
        if state.hold_steps_left <= 0:
            tsup_levels = np.array([18.0, 20.0, 23.0, 26.0, 30.0, 35.0], dtype=float)
            fan_levels = np.array([0.0, 0.2, 0.45, 0.7, 1.0], dtype=float)
            state.held_a0 = normalize_t_supply(float(rng.choice(tsup_levels)))
            state.held_a1 = normalize_fan(float(rng.choice(fan_levels)))
            state.hold_steps_left = int(rng.integers(4, 25))
        state.hold_steps_left -= 1
        return clamp_action(state.held_a0, state.held_a1)

    raise ValueError(f"Unsupported policy: {policy}")


def choose_action_profile(
    policy: str,
    profile: str,
    state: PolicyState,
    t_zone_c: float,
    t_amb_c: float,
) -> tuple[float, float]:
    if profile == "broad":
        return choose_action(policy=policy, state=state, t_zone_c=t_zone_c, t_amb_c=t_amb_c)

    if profile == "heating_focus":
        rng = state.rng

        if policy == "random":
            return clamp_action(rng.uniform(-0.15, 0.75), rng.uniform(-0.15, 0.85))

        if policy == "heat":
            return clamp_action(rng.uniform(0.15, 0.85), rng.uniform(0.25, 0.95))

        if policy == "cool":
            return clamp_action(rng.uniform(-0.35, 0.05), rng.uniform(0.0, 0.45))

        if policy == "mixed":
            return clamp_action(rng.uniform(-0.1, 0.6), rng.uniform(0.0, 0.8))

        if policy == "thermostatic_noise":
            return band_tracking_action(t_zone_c=t_zone_c, t_amb_c=t_amb_c, rng=rng, heating_focus=True)

        if policy == "pulse":
            if state.hold_steps_left <= 0:
                tsup_levels = np.array([20.0, 21.5, 23.0, 24.5, 26.5, 29.0], dtype=float)
                fan_levels = np.array([0.15, 0.35, 0.55, 0.75], dtype=float)
                state.held_a0 = normalize_t_supply(float(rng.choice(tsup_levels)))
                state.held_a1 = normalize_fan(float(rng.choice(fan_levels)))
                state.hold_steps_left = int(rng.integers(4, 13))
            state.hold_steps_left -= 1
            return clamp_action(state.held_a0, state.held_a1)

    raise ValueError(f"Unsupported profile '{profile}' for policy '{policy}'")


def filter_dataset_for_training(
    df: pd.DataFrame,
    train_t_zone_min_c: float,
    train_t_zone_max_c: float,
    train_abs_delta_t_max_c: float,
    train_p_total_max_w: float,
) -> pd.DataFrame:
    keep_mask = (
        df["t_zone"].between(train_t_zone_min_c, train_t_zone_max_c)
        & df["t_zone_next"].between(train_t_zone_min_c, train_t_zone_max_c)
        & (df["delta_t"].abs() <= train_abs_delta_t_max_c)
        & df["p_total"].between(0.0, train_p_total_max_w)
    )
    return df.loc[keep_mask].reset_index(drop=True)


def parse_csv_list(raw: str) -> list[str]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def build_episode_specs(seasons: list[str], policies: list[str], base_seed: int) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    episode_idx = 0
    for season in seasons:
        if season not in SEASON_START_TIMES_SEC:
            raise ValueError(
                f"Unknown season '{season}'. Available: {sorted(SEASON_START_TIMES_SEC.keys())}"
            )
        for policy in policies:
            specs.append(
                EpisodeSpec(
                    season=season,
                    policy=policy,
                    seed=int(base_seed + episode_idx),
                    start_time_sec=float(SEASON_START_TIMES_SEC[season]),
                )
            )
            episode_idx += 1
    return specs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_episode(
    client: BOPTESTClient,
    spec: EpisodeSpec,
    steps_per_episode: int,
    warmup_sec: int,
    episodes_dir: Path,
    profile: str,
    safe_t_zone_min_c: float,
    safe_t_zone_max_c: float,
) -> tuple[pd.DataFrame, EpisodeResult]:
    episode_id = f"{spec.season}__{spec.policy}__seed{spec.seed}"
    rng = np.random.default_rng(spec.seed)
    policy_state = PolicyState(rng=rng)

    print("-" * 88)
    print(
        f"[COLLECT] episode={episode_id} | start_day={spec.start_time_sec / 86400.0:.1f} "
        f"| steps={steps_per_episode} | step_sec={client.step_sec}"
    )

    testid = client.select_testcase()
    client.initialize(testid=testid, start_time_sec=spec.start_time_sec, warmup_sec=warmup_sec)
    payload = client.advance(testid, {})

    rows: list[dict[str, Any]] = []
    t_start = time.time()

    try:
        for step in range(int(steps_per_episode)):
            sim_time_sec = flatten_boptest_value(payload, "time")
            t_zone_c = k_to_c(flatten_boptest_value(payload, "zon_reaTRooAir_y"))
            t_amb_c = k_to_c(flatten_boptest_value(payload, "zon_weaSta_reaWeaTDryBul_y"))
            hour = (sim_time_sec / 3600.0) % 24.0
            day = (sim_time_sec / 86400.0) % 365.0

            a0, a1 = choose_action_profile(
                policy=spec.policy,
                profile=profile,
                state=policy_state,
                t_zone_c=t_zone_c,
                t_amb_c=t_amb_c,
            )
            a0, a1 = apply_safety_envelope(
                a0=a0,
                a1=a1,
                t_zone_c=t_zone_c,
                t_amb_c=t_amb_c,
                safe_t_zone_min_c=safe_t_zone_min_c,
                safe_t_zone_max_c=safe_t_zone_max_c,
            )
            command, t_supply_cmd_c, fan_cmd_u = build_bestest_air_command(a0, a1)
            payload_next = client.advance(testid, command)

            t_zone_next_c = k_to_c(flatten_boptest_value(payload_next, "zon_reaTRooAir_y"))
            p_cool_w = flatten_boptest_value(payload_next, "fcu_reaPCoo_y")
            p_fan_w = flatten_boptest_value(payload_next, "fcu_reaPFan_y")
            p_total_w = p_cool_w + p_fan_w

            rows.append(
                {
                    "episode_id": episode_id,
                    "step": int(step),
                    "step_sec": int(client.step_sec),
                    "sim_time_sec": float(sim_time_sec),
                    "t_zone": float(t_zone_c),
                    "t_amb": float(t_amb_c),
                    "hour": float(hour),
                    "day": float(day),
                    "a0_raw": float(a0),
                    "a1_raw": float(a1),
                    "t_supply_cmd_c": float(t_supply_cmd_c),
                    "fan_cmd_u": float(fan_cmd_u),
                    "t_zone_next": float(t_zone_next_c),
                    "delta_t": float(t_zone_next_c - t_zone_c),
                    "p_total": float(p_total_w),
                    "p_cool_w": float(p_cool_w),
                    "p_fan_w": float(p_fan_w),
                    "policy": spec.policy,
                    "season": spec.season,
                    "controller_source": CONTROLLER_SOURCE,
                    "testcase": client.testcase,
            "collector": "collect_surrogate_15min_boptest_data.py",
                }
            )

            payload = payload_next

            if step == 0 or (step + 1) % 288 == 0 or (step + 1) == int(steps_per_episode):
                elapsed = max(time.time() - t_start, 1e-6)
                fps = float((step + 1) / elapsed)
                print(
                    f"  step {step + 1:4d}/{steps_per_episode} | Tz={t_zone_c:5.2f} C "
                    f"| Tamb={t_amb_c:6.2f} C | Tsup={t_supply_cmd_c:5.1f} C "
                    f"| P={p_total_w:7.1f} W | {fps:5.2f} steps/s"
                )
    finally:
        client.stop(testid)

    episode_df = pd.DataFrame(rows)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(episodes_dir / f"{episode_id}.csv", index=False)

    result = EpisodeResult(
        episode_id=episode_id,
        season=spec.season,
        policy=spec.policy,
        seed=int(spec.seed),
        rows=int(len(episode_df)),
        step_sec=int(client.step_sec),
        t_zone_min_c=float(episode_df["t_zone"].min()),
        t_zone_max_c=float(episode_df["t_zone"].max()),
        t_amb_mean_c=float(episode_df["t_amb"].mean()),
        mean_power_w=float(episode_df["p_total"].mean()),
        mean_abs_delta_t_c=float(episode_df["delta_t"].abs().mean()),
    )
    return episode_df, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a new 15-minute direct-TSup surrogate dataset from live BOPTEST."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--testcase", default=DEFAULT_TESTCASE)
    parser.add_argument("--step-sec", type=int, default=DEFAULT_STEP_SEC)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--warmup-sec", type=int, default=DEFAULT_WARMUP_SEC)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seasons", default=",".join(DEFAULT_SEASONS))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--profile", choices=["broad", "heating_focus"], default="heating_focus")
    parser.add_argument("--safe-t-zone-min-c", type=float, default=DEFAULT_SAFE_T_ZONE_MIN_C)
    parser.add_argument("--safe-t-zone-max-c", type=float, default=DEFAULT_SAFE_T_ZONE_MAX_C)
    parser.add_argument("--write-train-subset", action="store_true")
    parser.add_argument("--train-subset-csv", default=None)
    parser.add_argument("--train-t-zone-min-c", type=float, default=DEFAULT_TRAIN_T_ZONE_MIN_C)
    parser.add_argument("--train-t-zone-max-c", type=float, default=DEFAULT_TRAIN_T_ZONE_MAX_C)
    parser.add_argument("--train-abs-delta-t-max-c", type=float, default=DEFAULT_TRAIN_ABS_DELTA_T_MAX_C)
    parser.add_argument("--train-p-total-max-w", type=float, default=DEFAULT_TRAIN_P_TOTAL_MAX_W)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    seasons = parse_csv_list(args.seasons)
    policies = parse_csv_list(args.policies)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    episodes_dir = output_dir / "episodes"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_episode_specs(seasons=seasons, policies=policies, base_seed=int(args.base_seed))
    collection_config = {
        "block": BLOCK_NAME,
        "dataset_kind": "collected_15min_direct_tsup",
        "base_url": args.base_url,
        "testcase": args.testcase,
        "step_sec": int(args.step_sec),
        "steps_per_episode": int(args.steps_per_episode),
        "warmup_sec": int(args.warmup_sec),
        "base_seed": int(args.base_seed),
        "seasons": seasons,
        "policies": policies,
        "profile": args.profile,
        "safe_t_zone_min_c": float(args.safe_t_zone_min_c),
        "safe_t_zone_max_c": float(args.safe_t_zone_max_c),
        "comfort_band_c": [COMFORT_LOW_C, COMFORT_HIGH_C],
        "write_train_subset": bool(args.write_train_subset),
        "train_subset_csv": str(args.train_subset_csv) if args.train_subset_csv else None,
        "train_t_zone_min_c": float(args.train_t_zone_min_c),
        "train_t_zone_max_c": float(args.train_t_zone_max_c),
        "train_abs_delta_t_max_c": float(args.train_abs_delta_t_max_c),
        "train_p_total_max_w": float(args.train_p_total_max_w),
        "episodes": len(specs),
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "notes": [
            "This is the active live BOPTEST collector for Block 1.2.",
            "The collector is centered on the 21-24 C comfort band instead of a fixed 22 C target.",
            "It produces per-episode dumps suitable for 15-minute surrogate retraining and later v3.5 calibration.",
        ],
    }
    write_json(output_dir / "collection_config.json", collection_config)

    print("=" * 88)
    print("BLOCK 1.2 COLLECT 15-MINUTE SURROGATE DATASET")
    print("=" * 88)
    print(f"Base URL:          {args.base_url}")
    print(f"Testcase:          {args.testcase}")
    print(f"Step sec:          {args.step_sec}")
    print(f"Steps/episode:     {args.steps_per_episode}")
    print(f"Warmup sec:        {args.warmup_sec}")
    print(f"Seasons:           {seasons}")
    print(f"Policies:          {policies}")
    print(f"Profile:           {args.profile}")
    print(f"Comfort band:      [{COMFORT_LOW_C}, {COMFORT_HIGH_C}] C")
    print(f"Safety envelope:   [{args.safe_t_zone_min_c}, {args.safe_t_zone_max_c}] C")
    print(f"Total episodes:    {len(specs)}")
    print(f"Expected rows:     {len(specs) * int(args.steps_per_episode):,}")
    print(f"Output CSV:        {output_csv}")
    print(f"Output dir:        {output_dir}")

    client = BOPTESTClient(
        base_url=args.base_url,
        testcase=args.testcase,
        step_sec=int(args.step_sec),
    )
    version = client.check_connectivity()
    print(f"BOPTEST version:   {version.get('payload', {}).get('version', 'unknown')}")

    all_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    t_total = time.time()

    for spec in specs:
        episode_df, episode_result = collect_episode(
            client=client,
            spec=spec,
            steps_per_episode=int(args.steps_per_episode),
            warmup_sec=int(args.warmup_sec),
            episodes_dir=episodes_dir,
            profile=args.profile,
            safe_t_zone_min_c=float(args.safe_t_zone_min_c),
            safe_t_zone_max_c=float(args.safe_t_zone_max_c),
        )
        all_frames.append(episode_df)
        summary_rows.append(
            {
                "episode_id": episode_result.episode_id,
                "season": episode_result.season,
                "policy": episode_result.policy,
                "seed": episode_result.seed,
                "rows": episode_result.rows,
                "step_sec": episode_result.step_sec,
                "t_zone_min_c": episode_result.t_zone_min_c,
                "t_zone_max_c": episode_result.t_zone_max_c,
                "t_amb_mean_c": episode_result.t_amb_mean_c,
                "mean_power_w": episode_result.mean_power_w,
                "mean_abs_delta_t_c": episode_result.mean_abs_delta_t_c,
            }
        )

        combined_intermediate = pd.concat(all_frames, ignore_index=True)
        combined_intermediate.to_csv(output_csv, index=False)
        pd.DataFrame(summary_rows).to_csv(output_dir / "episode_summary.csv", index=False)

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset.to_csv(output_csv, index=False)
    episode_summary = pd.DataFrame(summary_rows).sort_values(["season", "policy"]).reset_index(drop=True)
    episode_summary.to_csv(output_dir / "episode_summary.csv", index=False)

    train_subset_csv = None
    train_subset_rows = None
    if args.write_train_subset:
        if args.train_subset_csv:
            train_subset_csv = Path(args.train_subset_csv)
        else:
            train_subset_csv = output_csv.with_name(f"{output_csv.stem}_train_subset.csv")
        train_subset_csv.parent.mkdir(parents=True, exist_ok=True)
        train_subset = filter_dataset_for_training(
            df=dataset,
            train_t_zone_min_c=float(args.train_t_zone_min_c),
            train_t_zone_max_c=float(args.train_t_zone_max_c),
            train_abs_delta_t_max_c=float(args.train_abs_delta_t_max_c),
            train_p_total_max_w=float(args.train_p_total_max_w),
        )
        train_subset.to_csv(train_subset_csv, index=False)
        train_subset_rows = int(len(train_subset))

    elapsed_min = (time.time() - t_total) / 60.0
    dataset_summary = {
        "block": BLOCK_NAME,
        "dataset_kind": "collected_15min_direct_tsup",
        "rows": int(len(dataset)),
        "episodes": int(dataset["episode_id"].nunique()),
        "step_sec_unique": sorted(dataset["step_sec"].astype(int).unique().tolist()),
        "season_values": sorted(dataset["season"].astype(str).unique().tolist()),
        "policy_values": sorted(dataset["policy"].astype(str).unique().tolist()),
        "t_zone_range_c": [float(dataset["t_zone"].min()), float(dataset["t_zone"].max())],
        "t_amb_range_c": [float(dataset["t_amb"].min()), float(dataset["t_amb"].max())],
        "delta_t_range_c": [float(dataset["delta_t"].min()), float(dataset["delta_t"].max())],
        "p_total_range_w": [float(dataset["p_total"].min()), float(dataset["p_total"].max())],
        "mean_abs_delta_t_c": float(dataset["delta_t"].abs().mean()),
        "mean_power_w": float(dataset["p_total"].mean()),
        "profile": args.profile,
        "safe_t_zone_min_c": float(args.safe_t_zone_min_c),
        "safe_t_zone_max_c": float(args.safe_t_zone_max_c),
        "comfort_band_c": [COMFORT_LOW_C, COMFORT_HIGH_C],
        "elapsed_min": float(elapsed_min),
        "output_csv": str(output_csv),
        "episode_summary_csv": str(output_dir / "episode_summary.csv"),
        "collection_config_json": str(output_dir / "collection_config.json"),
        "train_subset_csv": str(train_subset_csv) if train_subset_csv else None,
        "train_subset_rows": train_subset_rows,
    }
    write_json(output_dir / "dataset_summary.json", dataset_summary)

    print("=" * 88)
    print("BLOCK 1.2 COLLECTION COMPLETE")
    print("=" * 88)
    print(f"Rows:              {len(dataset):,}")
    print(f"Episodes:          {dataset['episode_id'].nunique()}")
    print(f"T_zone range:      [{dataset['t_zone'].min():.2f}, {dataset['t_zone'].max():.2f}] C")
    print(f"T_amb range:       [{dataset['t_amb'].min():.2f}, {dataset['t_amb'].max():.2f}] C")
    print(f"|delta_t| mean:    {dataset['delta_t'].abs().mean():.4f} C")
    print(f"Mean power:        {dataset['p_total'].mean():.1f} W")
    print(f"Elapsed:           {elapsed_min:.1f} min")
    print(f"Output CSV:        {output_csv}")
    if train_subset_csv is not None:
        print(f"Train subset CSV:  {train_subset_csv} ({train_subset_rows:,} rows)")
    print(f"Episode summary:   {output_dir / 'episode_summary.csv'}")
    print(f"Dataset summary:   {output_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
