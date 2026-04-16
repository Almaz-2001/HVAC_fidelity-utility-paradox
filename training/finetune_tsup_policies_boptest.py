from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.tsup_features import (
    EXTENDED_TSUP_OBS_DIM,
    SUPPORTED_TSUP_OBS_DIMS,
    WeatherLookup,
    action_to_fan,
    action_to_t_supply,
    build_tsup_obs,
    infer_tsup_model_obs_dim,
    resolve_weather_csv,
)


T_LOW = 21.0
T_HIGH = 25.0
T_TARGET = 22.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
COOLING_SETPOINT_C = 40.0
HEATING_SETPOINT_C = 15.0
FORECAST_HORIZONS = (1, 3, 6)

BASE_UNDERSHOOT_WEIGHT = 1.30
BASE_OVERSHOOT_WEIGHT = 1.10
WINTER_UNDERSHOOT_WEIGHT = 1.90
HOT_OVERSHOOT_WEIGHT = 1.35
WINTER_AMB_THRESHOLD = 10.0
HOT_AMB_THRESHOLD = 24.0
ENERGY_NEAR_TARGET_SCALE = 3.0e-5
LAMBDA_SMOOTH = 0.05


THERMOSTATIC_MODEL_CANDIDATES = [
    ROOT / "models" / "ppo_thermostatic.zip",
    Path("/app/models/ppo_thermostatic.zip"),
]
WINTER_MODEL_CANDIDATES = [
    ROOT / "models" / "ppo_winter_final.zip",
    Path("/app/models/ppo_winter_final.zip"),
]
SUMMER_MODEL_CANDIDATES = [
    ROOT / "models" / "ppo_summer_final.zip",
    Path("/app/models/ppo_summer_final.zip"),
]


@dataclass
class AnchorWindows:
    peak_heat_start: int
    typical_heat_start: int
    peak_cool_start: int
    typical_cool_start: int


def resolve_existing_path(candidates: list[Path]) -> str:
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def parse_agents(raw: str) -> list[str]:
    agents = [item.strip() for item in raw.split(",") if item.strip()]
    if not agents:
        raise ValueError("At least one agent must be specified.")
    return agents


def parse_start_times(raw: str | None) -> list[int]:
    if not raw:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(float(token)))
    return values


def derive_anchor_windows(weather_csv: str, duration_days: int, heating_threshold_c: float = 12.0) -> AnchorWindows:
    df = pd.read_csv(weather_csv)
    grouped = (
        df.groupby("day", as_index=False)["t_amb"]
        .mean()
        .rename(columns={"t_amb": "daily_mean_t_amb_c"})
        .sort_values("day")
        .reset_index(drop=True)
    )
    grouped = grouped[(grouped["day"] >= 0) & (grouped["day"] <= 365 - duration_days)].copy()
    if grouped.empty:
        raise RuntimeError(f"Could not derive anchor windows from weather csv: {weather_csv}")

    peak_heat = grouped.loc[grouped["daily_mean_t_amb_c"].idxmin()]
    peak_cool = grouped.loc[grouped["daily_mean_t_amb_c"].idxmax()]

    heating_days = grouped[grouped["daily_mean_t_amb_c"] <= heating_threshold_c].copy()
    if heating_days.empty:
        heating_days = grouped.copy()
    heating_target = float(heating_days["daily_mean_t_amb_c"].median())
    heating_days["distance"] = np.abs(heating_days["daily_mean_t_amb_c"] - heating_target)
    typical_heat = heating_days.sort_values(["distance", "day"]).iloc[0]

    cooling_days = grouped[grouped["daily_mean_t_amb_c"] >= 18.0].copy()
    if cooling_days.empty:
        cooling_days = grouped.copy()
    cooling_target = float(cooling_days["daily_mean_t_amb_c"].median())
    cooling_days["distance"] = np.abs(cooling_days["daily_mean_t_amb_c"] - cooling_target)
    typical_cool = cooling_days.sort_values(["distance", "day"]).iloc[0]

    return AnchorWindows(
        peak_heat_start=int(peak_heat["day"]) * 86400,
        typical_heat_start=int(typical_heat["day"]) * 86400,
        peak_cool_start=int(peak_cool["day"]) * 86400,
        typical_cool_start=int(typical_cool["day"]) * 86400,
    )


class BOPTESTDirectTSupEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        url: str,
        testcase: str,
        step_sec: int,
        episode_days: int,
        start_time_choices: list[int],
        jitter_days: float = 0.0,
        seed: int | None = None,
        obs_dim: int = EXTENDED_TSUP_OBS_DIM,
    ) -> None:
        super().__init__()
        self.url = url.rstrip("/")
        self.testcase = testcase
        self.step_sec = int(step_sec)
        self.episode_days = int(episode_days)
        self.max_steps = int(round(self.episode_days * 86400 / self.step_sec))
        self.start_time_choices = [int(v) for v in start_time_choices]
        self.jitter_sec = int(round(float(jitter_days) * 86400.0))
        self.obs_dim = int(obs_dim)
        if self.obs_dim not in SUPPORTED_TSUP_OBS_DIMS:
            raise RuntimeError(
                f"Unsupported observation dim {self.obs_dim}. Supported dims: {sorted(SUPPORTED_TSUP_OBS_DIMS)}."
            )
        self.include_forecast = self.obs_dim == EXTENDED_TSUP_OBS_DIM
        self.current_step = 0
        self.testid: str | None = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_t_zone: float | None = None
        self.weather = WeatherLookup(resolve_weather_csv())
        self.rng = np.random.RandomState(seed if seed is not None else 42)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
        url = f"{self.url}{path}"
        for attempt in range(3):
            try:
                if method == "POST":
                    response = self.session.post(url, json=payload or {}, timeout=timeout)
                elif method == "PUT":
                    response = self.session.put(url, json=payload or {}, timeout=timeout)
                elif method == "GET":
                    response = self.session.get(url, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code in (500, 502, 503, 504):
                    time.sleep(2**attempt)
                    continue

                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"Request failed unexpectedly: {url}")

    @staticmethod
    def _get_val(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key, 0.0)
        if isinstance(value, dict):
            value = value.get("value", 0.0)
        return float(value)

    def _sample_start_time(self) -> int:
        if self.start_time_choices:
            base = int(self.rng.choice(self.start_time_choices))
        else:
            base = 0
        if self.jitter_sec > 0:
            base = max(0, base + int(self.rng.randint(-self.jitter_sec, self.jitter_sec + 1)))
        return base

    def _make_obs(self, payload: dict[str, Any], prev_action: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        t_zone = self._get_val(payload, "zon_reaTRooAir_y") - 273.15
        co2 = self._get_val(payload, "zon_reaCO2RooAir_y")
        p_cool = self._get_val(payload, "fcu_reaPCoo_y")
        p_fan = self._get_val(payload, "fcu_reaPFan_y")
        p_heat = self._get_val(payload, "fcu_reaPHea_y")
        t_amb_k = self._get_val(payload, "zon_weaSta_reaWeaTDryBul_y")
        t_amb = t_amb_k - 273.15 if t_amb_k > 200.0 else t_amb_k
        sim_time = self._get_val(payload, "time")
        hour = (sim_time / 3600.0) % 24.0
        day = (sim_time / 86400.0) % 365.0
        p_total = p_cool + p_fan + p_heat
        prev_t_supply = action_to_t_supply(float(prev_action[0])) if prev_action is not None else 0.5 * (T_SUPPLY_LOW + T_SUPPLY_HIGH)
        delta_t_zone = 0.0 if self.prev_t_zone is None else (t_zone - self.prev_t_zone)

        obs = build_tsup_obs(
            t_zone,
            co2,
            p_total,
            prev_t_supply,
            t_amb,
            hour,
            day,
            prev_action if prev_action is not None else np.zeros(2, dtype=np.float32),
            delta_t_zone,
            self.weather,
            include_forecast=self.include_forecast,
        )
        self.prev_t_zone = t_zone
        info = {
            "zone_temp": float(t_zone),
            "t_amb": float(t_amb),
            "power": float(p_total),
            "hour": float(hour),
            "day": float(day),
            "sim_time": float(sim_time),
            "delta_t_zone": float(delta_t_zone),
        }
        return obs, info

    def _compute_reward(self, info: dict[str, float], action: np.ndarray) -> float:
        t_zone = float(info["zone_temp"])
        t_amb = float(info["t_amb"])
        p_total = float(info["power"])
        hour = float(info["hour"])
        day = float(info["day"])

        error = t_zone - T_TARGET
        abs_error = abs(error)
        step_scale = self.step_sec / 3600.0
        cold_pressure = min(self.weather.forecast(hour, day, h) for h in FORECAST_HORIZONS)
        winter_like = (
            t_amb <= WINTER_AMB_THRESHOLD
            or cold_pressure <= WINTER_AMB_THRESHOLD
            or day < 59.0
            or day >= 334.0
        )
        hot_like = t_amb >= HOT_AMB_THRESHOLD

        if abs_error <= 1.0:
            r_track = -0.5 * (abs_error**2)
        else:
            r_track = -abs_error + 0.5

        if error < 0.0:
            weight = WINTER_UNDERSHOOT_WEIGHT if winter_like else BASE_UNDERSHOOT_WEIGHT
            r_track *= weight
        elif error > 0.0:
            weight = HOT_OVERSHOOT_WEIGHT if hot_like else BASE_OVERSHOOT_WEIGHT
            r_track *= weight

        if abs_error <= 0.5:
            r_track += 0.75 * (1.0 - abs_error / 0.5)
        elif T_LOW <= t_zone <= T_HIGH:
            r_track += 0.05

        if t_zone < 19.0:
            r_track -= 2.5 * (19.0 - t_zone)

        delta = np.asarray(action, dtype=np.float32) - self.prev_action
        r_smooth = -LAMBDA_SMOOTH * float(np.sum(delta**2))

        if abs_error < 0.75:
            r_power = -ENERGY_NEAR_TARGET_SCALE * p_total * step_scale
        else:
            r_power = 0.0

        return step_scale * (r_track + r_smooth) + r_power

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10.0)
            except Exception:
                pass

        data = self._request("POST", f"/testcases/{self.testcase}/select", timeout=300.0)
        self.testid = data.get("testid")
        if not self.testid:
            raise RuntimeError(f"BOPTEST did not return testid: {data}")

        self._request("PUT", f"/step/{self.testid}", {"step": self.step_sec}, timeout=30.0)
        start_time = self._sample_start_time()
        self._request(
            "PUT",
            f"/initialize/{self.testid}",
            {"start_time": start_time, "warmup_period": 0},
            timeout=300.0,
        )
        payload = self._request("POST", f"/advance/{self.testid}", {})
        payload = payload.get("payload", payload)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_t_zone = None
        self.current_step = 0
        obs, info = self._make_obs(payload, self.prev_action)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        t_supply = action_to_t_supply(float(action[0]))
        fan_u = action_to_fan(float(action[1]))
        command = {
            "con_oveTSetCoo_activate": 1,
            "con_oveTSetCoo_u": COOLING_SETPOINT_C + 273.15,
            "con_oveTSetHea_activate": 1,
            "con_oveTSetHea_u": HEATING_SETPOINT_C + 273.15,
            "fcu_oveFan_activate": 1,
            "fcu_oveFan_u": fan_u,
            "fcu_oveTSup_activate": 1,
            "fcu_oveTSup_u": t_supply + 273.15,
        }

        payload = self._request("POST", f"/advance/{self.testid}", command)
        payload = payload.get("payload", payload)
        obs, info = self._make_obs(payload, action)
        reward = self._compute_reward(info, action)
        self.prev_action = action.copy()
        self.current_step += 1

        truncated = self.current_step >= self.max_steps
        terminated = False
        info = dict(info)
        info["t_supply_cmd"] = float(t_supply)
        info["fan_u"] = float(fan_u)
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.testid:
            try:
                self._request("PUT", f"/stop/{self.testid}", timeout=10.0)
            except Exception:
                pass
        self.session.close()


def validate_model_obs_dim(model_path: str) -> int:
    return infer_tsup_model_obs_dim(model_path)


def default_model_for_agent(agent: str, args: argparse.Namespace) -> str:
    if agent == "thermostatic":
        return args.thermostatic_model
    if agent == "winter":
        return args.winter_model
    if agent == "summer":
        return args.summer_model
    raise ValueError(f"Unknown agent: {agent}")


def default_steps_for_agent(agent: str, args: argparse.Namespace) -> int:
    if agent == "thermostatic":
        return args.steps_thermostatic
    if agent == "winter":
        return args.steps_winter
    if agent == "summer":
        return args.steps_summer
    raise ValueError(f"Unknown agent: {agent}")


def default_anchor_times(agent: str, anchors: AnchorWindows, explicit: list[int]) -> list[int]:
    if explicit:
        return explicit
    if agent == "thermostatic":
        return [anchors.peak_heat_start, anchors.typical_heat_start]
    if agent == "winter":
        return [anchors.peak_heat_start, anchors.typical_heat_start]
    if agent == "summer":
        return [anchors.peak_cool_start, anchors.typical_cool_start]
    raise ValueError(f"Unknown agent: {agent}")


def finetune_one_agent(
    agent: str,
    model_path: str,
    out_dir: Path,
    args: argparse.Namespace,
    start_time_choices: list[int],
) -> Path:
    model_obs_dim = validate_model_obs_dim(model_path)

    env = BOPTESTDirectTSupEnv(
        url=args.boptest_url,
        testcase=args.testcase_id,
        step_sec=args.step_sec,
        episode_days=args.episode_days,
        start_time_choices=start_time_choices,
        jitter_days=args.jitter_days,
        seed=args.seed,
        obs_dim=model_obs_dim,
    )
    env = Monitor(env)

    print(f"\n{'=' * 72}")
    print(f"FINE-TUNE {agent.upper()} ON BOPTEST")
    print(f"{'=' * 72}")
    print(f"Model:        {model_path}")
    print(f"Step sec:     {args.step_sec}")
    print(f"Obs dim:      {model_obs_dim}")
    print(f"Episode days: {args.episode_days}")
    print(f"Episode len:  {int(round(args.episode_days * 86400 / args.step_sec))} steps")
    print(f"Anchors:      {start_time_choices}")
    print(f"Jitter days:  {args.jitter_days}")

    obs, _ = env.reset(seed=args.seed)
    print(f"Obs sample OK: shape={obs.shape}")

    custom_objects = {
        "clip_range": lambda _: args.learning_rate * 0.0 + 0.2,
        "lr_schedule": lambda _: args.learning_rate,
    }
    model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
    model.learning_rate = args.learning_rate
    model.lr_schedule = lambda _: args.learning_rate
    model.n_epochs = args.ppo_epochs

    steps = default_steps_for_agent(agent, args)
    start = time.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=1)
    elapsed_min = (time.time() - start) / 60.0

    save_path = out_dir / f"{agent}_step{args.step_sec}_finetuned"
    model.save(str(save_path))
    try:
        env.close()
    except Exception:
        pass

    print(f"Done: {elapsed_min:.1f} min")
    print(f"Saved: {save_path}.zip")
    return save_path.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune direct-TSup thermostatic/HDRL policies on real BOPTEST with configurable step size."
    )
    parser.add_argument("--agents", default="thermostatic,winter", help="Comma-separated: thermostatic,winter,summer")
    parser.add_argument("--boptest-url", default=os.environ.get("BOPTEST_URL", "http://web:8000"))
    parser.add_argument("--testcase-id", default="bestest_air")
    parser.add_argument("--step-sec", type=int, default=900)
    parser.add_argument("--episode-days", type=int, default=14)
    parser.add_argument("--jitter-days", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--steps-thermostatic", type=int, default=120_000)
    parser.add_argument("--steps-winter", type=int, default=80_000)
    parser.add_argument("--steps-summer", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heating-threshold-c", type=float, default=12.0)
    parser.add_argument("--start-times", default=None, help="Optional comma-separated explicit start times in seconds.")
    parser.add_argument("--out-dir", default="outputs/boptest_15min_policy_finetune")
    parser.add_argument("--thermostatic-model", default=resolve_existing_path(THERMOSTATIC_MODEL_CANDIDATES))
    parser.add_argument("--winter-model", default=resolve_existing_path(WINTER_MODEL_CANDIDATES))
    parser.add_argument("--summer-model", default=resolve_existing_path(SUMMER_MODEL_CANDIDATES))
    args = parser.parse_args()

    agents = parse_agents(args.agents)
    explicit_start_times = parse_start_times(args.start_times)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weather_csv = resolve_weather_csv()
    anchors = derive_anchor_windows(
        weather_csv=weather_csv,
        duration_days=args.episode_days,
        heating_threshold_c=args.heating_threshold_c,
    )

    snapshot = {
        "agents": agents,
        "boptest_url": args.boptest_url,
        "testcase_id": args.testcase_id,
        "step_sec": args.step_sec,
        "episode_days": args.episode_days,
        "jitter_days": args.jitter_days,
        "learning_rate": args.learning_rate,
        "ppo_epochs": args.ppo_epochs,
        "steps_thermostatic": args.steps_thermostatic,
        "steps_winter": args.steps_winter,
        "steps_summer": args.steps_summer,
        "seed": args.seed,
        "weather_csv": weather_csv,
        "anchors": anchors.__dict__,
        "explicit_start_times": explicit_start_times,
        "thermostatic_model": args.thermostatic_model,
        "winter_model": args.winter_model,
        "summer_model": args.summer_model,
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("=" * 72)
    print("15-MINUTE DIRECT-TSUP BOPTEST FINE-TUNE")
    print("=" * 72)
    print(f"Agents:        {agents}")
    print(f"Step sec:      {args.step_sec}")
    print(f"Episode days:  {args.episode_days}")
    print(f"Weather csv:   {weather_csv}")
    print(f"Anchors:       {anchors}")
    print(f"Output dir:    {out_dir}")

    saved_paths: dict[str, str] = {}
    for agent in agents:
        model_path = default_model_for_agent(agent, args)
        start_time_choices = default_anchor_times(agent, anchors, explicit_start_times)
        saved = finetune_one_agent(
            agent=agent,
            model_path=model_path,
            out_dir=out_dir,
            args=args,
            start_time_choices=start_time_choices,
        )
        saved_paths[agent] = str(saved)

    (out_dir / "saved_models.json").write_text(json.dumps(saved_paths, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("FINE-TUNE COMPLETE")
    print(f"{'=' * 72}")
    for agent, path in saved_paths.items():
        print(f"{agent}: {path}")
    print("\nNext step example:")
    print(
        "python evaluation/benchmark_bestest_air_article7_style.py "
        f"--step-sec {args.step_sec} "
        "--controllers thermostatic,hdrl "
        f"--thermostatic-model {saved_paths.get('thermostatic', args.thermostatic_model)} "
        f"--hdrl-winter-model {saved_paths.get('winter', args.winter_model)} "
        f"--hdrl-summer-model {saved_paths.get('summer', args.summer_model)} "
        "--output-dir outputs/bestest_air_article7_style_15min"
    )


if __name__ == "__main__":
    main()
