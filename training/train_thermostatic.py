"""
training/train_thermostatic.py

Comfort-first thermostatic PPO baseline on the direct-TSup surrogate.

This file intentionally stays close to the strongest validated paper baseline:
  - direct TSup control using the tsupply-trained surrogate
  - unbiased weather and forecast lookup
  - 17-feature observation with P_total, previous TSup, cyclic time,
    previous action and delta-T history
  - simple target-tracking reward with explicit winter underheating bias

Usage:
    python training/train_thermostatic.py
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from gymnasium import Env, spaces
from gymnasium.wrappers import TimeLimit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


T_TARGET = 22.0
COMFORT_BAND_LOW = 21.0
COMFORT_BAND_HIGH = 25.0
LAMBDA_SMOOTH = 0.05
TRACK_BONUS_DEADBAND = 0.5
FORECAST_HORIZONS = [1, 3, 6, 12, 24]
N_PHYS = 5
N_TIME = 4
N_FORECAST = len(FORECAST_HORIZONS)
N_HISTORY = 3  # prev_action(2) + delta_t(1)
N_OBS_M1 = N_PHYS + N_TIME + N_HISTORY
N_OBS_M23 = N_PHYS + N_TIME + N_FORECAST + N_HISTORY
BASE_UNDERSHOOT_WEIGHT = 1.30
BASE_OVERSHOOT_WEIGHT = 1.10
WINTER_UNDERSHOOT_WEIGHT = 1.85
WINTER_AMB_THRESHOLD = 10.0
ENERGY_NEAR_TARGET_SCALE = 3.0e-5



# Normalization
PHYS_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
PHYS_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)
T_AMB_LOW = -30.0
T_AMB_HIGH = 45.0
DELTA_T_MAX = 5.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
SURROGATE_KIND_DEFAULT = "legacy_v3"
SURROGATE_PATH_DEFAULT = "/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt"
WEATHER_CSV_PRIMARY = "/app/data/surrogate_v2/boptest_v2_tsupply.csv"
WEATHER_CSV_FALLBACK = "/app/data/surrogate_v2/boptest_v2_all.csv"


def resolve_torch_device(env_var_name: str, default: str) -> torch.device:
    raw = os.environ.get(env_var_name, default).strip().lower()
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raw = "cpu"
    return torch.device(raw)


def resolve_weather_csv():
    for path in (WEATHER_CSV_PRIMARY, WEATHER_CSV_FALLBACK):
        if os.path.exists(path):
            return path
    return WEATHER_CSV_PRIMARY


class RealWeatherLookup:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = resolve_weather_csv()
        self.grid = np.zeros((366, 24), dtype=np.float32)
        self.count = np.zeros((366, 24), dtype=np.int32)
        self.daily_mean = np.full(366, 10.0, dtype=np.float32)
        self.available = False
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            d = int(row["day"]) % 366
            h = int(row["hour"]) % 24
            t = float(row["t_amb"])
            if -30.0 < t < 50.0:
                self.grid[d, h] += t
                self.count[d, h] += 1

        mask = self.count > 0
        if not np.any(mask):
            return

        self.grid[mask] /= self.count[mask]
        for d in range(366):
            for h in range(24):
                if self.count[d, h] == 0:
                    for off in range(1, 30):
                        dp = (d - off) % 366
                        dn = (d + off) % 366
                        if self.count[dp, h] > 0:
                            self.grid[d, h] = self.grid[dp, h]
                            break
                        if self.count[dn, h] > 0:
                            self.grid[d, h] = self.grid[dn, h]
                            break

        for d in range(366):
            valid_hours = self.count[d] > 0
            if np.any(valid_hours):
                self.daily_mean[d] = float(self.grid[d, valid_hours].mean())
            else:
                self.daily_mean[d] = float(self.grid[d].mean())

        self.available = True

    def get(self, hour, day, noise_std=1.5):
        if not self.available:
            return 10.0
        d = int(day) % 366
        h = int(hour) % 24
        h_next = (h + 1) % 24
        frac = hour - int(hour)
        t = float(self.grid[d, h]) * (1.0 - frac) + float(self.grid[d, h_next]) * frac
        return float(np.clip(t + np.random.normal(0.0, noise_std), -30.0, 45.0))

    def forecast(self, hour, day, horizon_hours):
        if not self.available:
            return 10.0
        fh = (hour + horizon_hours) % 24
        fd = day + horizon_hours / 24.0
        d = int(fd) % 366
        h = int(fh) % 24
        return float(self.grid[d, h])


def norm_t_amb(t):
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


class ThermostaticEnv(Env):
    """
    17-feature thermostatic env with richer temporal context.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate_path=None,
        surrogate_kind: str = SURROGATE_KIND_DEFAULT,
        surrogate_summary_json: str | None = None,
        surrogate_checkpoint: str | None = None,
        surrogate_base_model: str | None = None,
        dr_enabled=True,
        season_sampling: str = "uniform",
        heating_threshold_c: float = 12.0,
        cooling_threshold_c: float = 24.0,
        heating_bias_prob: float = 0.7,
        cooling_bias_prob: float = 0.7,
        include_forecast: bool = True,
    ):
        super().__init__()

        self.include_forecast = bool(include_forecast)
        obs_dim = N_OBS_M23 if self.include_forecast else N_OBS_M1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if surrogate_path is None:
            surrogate_path = SURROGATE_PATH_DEFAULT
        self.device = resolve_torch_device("THERMOSTATIC_SURROGATE_DEVICE", "cpu")
        self.model = load_direct_tsup_adapter(
            kind=surrogate_kind,
            legacy_model_path=surrogate_path,
            summary_json=surrogate_summary_json,
            checkpoint_path=surrogate_checkpoint,
            base_model_path=surrogate_base_model,
            device=self.device,
        )
        self.surrogate_meta = self.model.describe()

        self.weather = RealWeatherLookup()
        self.dr_enabled = dr_enabled
        self.season_sampling = season_sampling
        self.heating_threshold_c = float(heating_threshold_c)
        self.cooling_threshold_c = float(cooling_threshold_c)
        self.heating_bias_prob = float(np.clip(heating_bias_prob, 0.0, 1.0))
        self.cooling_bias_prob = float(np.clip(cooling_bias_prob, 0.0, 1.0))
        self.dt = 3600
        self.max_steps = 336
        self.step_count = 0
        self._all_days, self._heating_days, self._cooling_days = self._build_candidate_days()

        self._t_zone = 20.0
        self._t_amb = 10.0
        self._co2 = 800.0
        self._p_total = 0.0
        self._t_supply_prev = 0.5 * (T_SUPPLY_LOW + T_SUPPLY_HIGH)
        self._time = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._delta_t_zone = 0.0
        self._errors = []

    def _build_candidate_days(self):
        all_days = np.arange(366, dtype=np.float32)
        if not self.weather.available:
            return all_days, all_days, all_days

        heating_days = np.where(self.weather.daily_mean <= self.heating_threshold_c)[0].astype(np.float32)
        cooling_days = np.where(self.weather.daily_mean >= self.cooling_threshold_c)[0].astype(np.float32)

        if len(heating_days) == 0:
            heating_days = all_days
        if len(cooling_days) == 0:
            cooling_days = all_days
        return all_days, heating_days, cooling_days

    @property
    def _hour(self):
        return (self._time / 3600.0) % 24.0

    @property
    def _day(self):
        return (self._time / 86400.0) % 365.0

    def _make_obs(self):
        phys = np.array(
            [self._t_zone, self._co2, self._p_total, self._t_supply_prev, self._t_amb],
            dtype=np.float32,
        )
        phys_norm = np.clip(2.0 * (phys - PHYS_LOW) / (PHYS_HIGH - PHYS_LOW) - 1.0, -1.0, 1.0)

        hour_sin, hour_cos = encode_hour_cyc(self._hour)
        day_sin, day_cos = encode_day_cyc(self._day)

        forecasts = []
        if self.include_forecast:
            for horizon in FORECAST_HORIZONS:
                forecasts.append(norm_t_amb(self.weather.forecast(self._hour, self._day, horizon)))

        blocks = [
            phys_norm,
            [hour_sin, hour_cos, day_sin, day_cos],
        ]
        if self.include_forecast:
            blocks.append(forecasts)
        blocks.extend(
            [
                self._prev_action,
                [norm_delta_t(self._delta_t_zone)],
            ]
        )
        obs = np.concatenate(blocks).astype(np.float32)
        return obs

    def _sample_start_day(self):
        if self.season_sampling == "uniform":
            return float(np.random.uniform(0.0, 365.0))
        if self.season_sampling == "heating_only":
            return float(np.random.choice(self._heating_days))
        if self.season_sampling == "cooling_only":
            return float(np.random.choice(self._cooling_days))
        if self.season_sampling == "heating_bias":
            if np.random.rand() < self.heating_bias_prob:
                return float(np.random.choice(self._heating_days))
            return float(np.random.choice(self._all_days))
        if self.season_sampling == "cooling_bias":
            if np.random.rand() < self.cooling_bias_prob:
                return float(np.random.choice(self._cooling_days))
            return float(np.random.choice(self._all_days))
        return float(np.random.uniform(0.0, 365.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._t_zone = float(np.random.uniform(15.0, 28.0)) if self.dr_enabled else 20.0
        self._co2 = 800.0 + np.random.normal(0.0, 50.0)
        self._p_total = 0.0
        self._t_supply_prev = 0.5 * (T_SUPPLY_LOW + T_SUPPLY_HIGH)
        self._start_day = self._sample_start_day() if self.dr_enabled else 180.0
        self._time = self._start_day * 86400.0
        self._t_amb = self.weather.get(self._hour, self._day, 1.5 if self.dr_enabled else 0.5)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._delta_t_zone = 0.0
        self.step_count = 0
        self._errors = []
        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a0, a1 = float(action[0]), float(action[1])
        t_supply = action_to_t_supply(a0)
        fan = action_to_fan(a1)

        self._time += self.dt
        self._t_amb = self.weather.get(self._hour, self._day, 1.5 if self.dr_enabled else 0.5)

        with torch.no_grad():
            t_next, p_total = self.model(
                torch.tensor([self._t_zone], device=self.device),
                torch.tensor([self._t_amb], device=self.device),
                torch.tensor([self._hour], device=self.device),
                torch.tensor([self._day], device=self.device),
                torch.tensor([a0], device=self.device),
                torch.tensor([a1], device=self.device),
            )

        t_prev = self._t_zone
        self._t_zone = float(t_next[0])
        self._delta_t_zone = self._t_zone - t_prev
        p = float(p_total[0])
        self._co2 = float(np.clip(self._co2 - 50.0 * fan + 10.0, 400.0, 2000.0))
        self._p_total = p
        self._t_supply_prev = t_supply

        error = self._t_zone - T_TARGET
        abs_error = abs(error)
        cold_pressure = min(self.weather.forecast(self._hour, self._day, h) for h in (1, 3, 6))
        winter_like = (
            self._t_amb <= WINTER_AMB_THRESHOLD
            or cold_pressure <= WINTER_AMB_THRESHOLD
            or self._day < 59.0
            or self._day >= 334.0
        )

        if abs_error <= 1.0:
            r_track = -0.5 * (abs_error ** 2)
        else:
            r_track = -abs_error + 0.5

        if error < 0.0:
            weight = WINTER_UNDERSHOOT_WEIGHT if winter_like else BASE_UNDERSHOOT_WEIGHT
            r_track *= weight
        elif error > 0.0:
            r_track *= BASE_OVERSHOOT_WEIGHT

        if abs_error <= TRACK_BONUS_DEADBAND:
            r_track += 0.75 * (1.0 - abs_error / TRACK_BONUS_DEADBAND)
        if self._t_zone < 19.0:
            r_track -= 2.0 * (19.0 - self._t_zone)

        delta = action - self._prev_action
        r_smooth = -LAMBDA_SMOOTH * float(np.sum(delta ** 2))

        if abs_error < 0.75:
            r_power = -ENERGY_NEAR_TARGET_SCALE * p
        else:
            r_power = 0.0

        reward = r_track + r_smooth + r_power

        self._prev_action = action.copy()
        self._errors.append(abs_error)
        self.step_count += 1

        return self._make_obs(), float(reward), False, self.step_count >= self.max_steps, {
            "zone_temp": self._t_zone,
            "t_amb": self._t_amb,
            "error": abs_error,
            "power": p,
            "delta_t_zone": self._delta_t_zone,
            "t_supply_cmd": t_supply,
            "fan_u": fan,
            "cold_pressure": cold_pressure,
            "winter_like": winter_like,
        }

    def close(self):
        pass


def make_env(rank, seed=42, env_kwargs=None):
    def _init():
        env = ThermostaticEnv(dr_enabled=True, **(env_kwargs or {}))
        env = TimeLimit(env, max_episode_steps=336)
        env.reset(seed=seed + rank)
        return env

    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train the thermostatic direct-TSup PPO baseline.")
    parser.add_argument("--article22-variant", choices=["m1", "m2", "m3"], default=None)
    parser.add_argument("--extractor", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--surrogate-kind", choices=["legacy_v3", "v35_raw", "v35_calibrated"], default=SURROGATE_KIND_DEFAULT)
    parser.add_argument("--surrogate-path", default=SURROGATE_PATH_DEFAULT)
    parser.add_argument("--surrogate-summary-json", default=None)
    parser.add_argument("--surrogate-checkpoint", default=None)
    parser.add_argument("--surrogate-base-model", default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--season-sampling",
        choices=["uniform", "heating_only", "cooling_only", "heating_bias", "cooling_bias"],
        default="uniform",
    )
    parser.add_argument("--heating-threshold-c", type=float, default=12.0)
    parser.add_argument("--cooling-threshold-c", type=float, default=24.0)
    parser.add_argument("--heating-bias-prob", type=float, default=0.7)
    parser.add_argument("--cooling-bias-prob", type=float, default=0.7)
    parser.add_argument("--save-name", default=None, help="Model basename without .zip")
    parser.add_argument("--tb-log", default=None, help="TensorBoard log dir")
    return parser.parse_args()


def main():
    args = parse_args()
    policy_device = resolve_torch_device("THERMOSTATIC_POLICY_DEVICE", "auto")
    surrogate_device = resolve_torch_device("THERMOSTATIC_SURROGATE_DEVICE", "cpu")
    extractor_name = args.extractor.lower().strip()
    include_forecast = True

    if args.article22_variant == "m1":
        extractor_name = "mlp"
        include_forecast = False
    elif args.article22_variant == "m2":
        extractor_name = "mlp"
        include_forecast = True
    elif args.article22_variant == "m3":
        extractor_name = "gru"
        include_forecast = True

    print("=" * 60)
    print("THERMOSTATIC PAPER BASELINE (direct TSup)")
    print(f"Target: {T_TARGET}C | Direct supply-temperature control")
    obs_dim = N_OBS_M23 if include_forecast else N_OBS_M1
    print(f"Obs: {obs_dim}F [T_zone, CO2, P_total, TSup_prev, T_amb + time/history" + ("/forecast]" if include_forecast else "]"))
    print(f"Forecast horizons: {FORECAST_HORIZONS}h")
    print(f"TSup range: [{T_SUPPLY_LOW}, {T_SUPPLY_HIGH}]C")
    print(f"Surrogate kind: {args.surrogate_kind}")
    print(f"Surrogate path: {args.surrogate_path}")
    if args.surrogate_summary_json:
        print(f"V3.5 summary: {args.surrogate_summary_json}")
    if args.surrogate_checkpoint:
        print(f"V3.5 checkpoint: {args.surrogate_checkpoint}")
    print("Reward: target-aware comfort + winter undershoot bias + light near-target power penalty")
    if args.article22_variant is not None:
        print(f"Article 22 variant: {args.article22_variant.upper()}")
    print(f"Feature extractor: {extractor_name}")
    print(f"Policy device: {policy_device}")
    print(f"Surrogate device: {surrogate_device}")
    print("=" * 60)

    num_envs = int(args.num_envs)
    total_steps = int(args.total_steps)
    seed = int(args.seed)
    env_kwargs = {
        "season_sampling": args.season_sampling,
        "heating_threshold_c": args.heating_threshold_c,
        "cooling_threshold_c": args.cooling_threshold_c,
        "heating_bias_prob": args.heating_bias_prob,
        "cooling_bias_prob": args.cooling_bias_prob,
        "include_forecast": include_forecast,
        "surrogate_kind": args.surrogate_kind,
        "surrogate_path": args.surrogate_path,
        "surrogate_summary_json": args.surrogate_summary_json,
        "surrogate_checkpoint": args.surrogate_checkpoint,
        "surrogate_base_model": args.surrogate_base_model,
    }

    print(f"Season sampling: {args.season_sampling}")
    if args.season_sampling == "heating_only":
        print(f"Heating threshold: {args.heating_threshold_c}C daily mean ambient")
    elif args.season_sampling == "cooling_only":
        print(f"Cooling threshold: {args.cooling_threshold_c}C daily mean ambient")
    elif args.season_sampling == "heating_bias":
        print(
            f"Heating threshold: {args.heating_threshold_c}C daily mean ambient | "
            f"bias prob: {args.heating_bias_prob}"
        )
    elif args.season_sampling == "cooling_bias":
        print(
            f"Cooling threshold: {args.cooling_threshold_c}C daily mean ambient | "
            f"bias prob: {args.cooling_bias_prob}"
        )

    vec_env = SubprocVecEnv([make_env(i, seed=seed, env_kwargs=env_kwargs) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = {"net_arch": [256, 256]}
    tb_log = args.tb_log or "./logs/thermostatic_v3/"
    save_name = args.save_name or "ppo_thermostatic"

    if extractor_name == "gru":
        from training.gru_policy import WeatherGRUExtractor

        # Current observation layout:
        # 0:5 physical, 5:9 cyclic time, 9:14 forecast horizons, 14:16 prev action, 16 delta-T.
        policy_kwargs = {
            "features_extractor_class": WeatherGRUExtractor,
            "features_extractor_kwargs": {
                "forecast_start": N_PHYS + N_TIME,
                "forecast_len": len(FORECAST_HORIZONS),
                "state_hidden": 128,
                "gru_hidden": 32,
                "combined_hidden": 128,
                "features_dim": 128,
            },
            "net_arch": [128, 128],
        }
        tb_log = args.tb_log or "./logs/thermostatic_gru/"
        save_name = args.save_name or "ppo_thermostatic_gru"
    elif args.article22_variant == "m1":
        save_name = args.save_name or "ppo_thermostatic_article22_m1"
    elif args.article22_variant == "m2":
        save_name = args.save_name or "ppo_thermostatic_article22_m2"
    elif args.article22_variant == "m3":
        save_name = args.save_name or "ppo_thermostatic_article22_m3"

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device=str(policy_device),
        verbose=1,
        tensorboard_log=tb_log,
    )

    os.makedirs("./models/", exist_ok=True)
    t0 = time.time()
    print(f"\nTraining: {num_envs} envs x {total_steps / 1e6:.0f}M steps...")

    model.learn(total_timesteps=total_steps)
    elapsed = (time.time() - t0) / 60

    model.save(f"./models/{save_name}")
    vec_env.close()
    print(f"\nDone: {elapsed:.1f} min -> models/{save_name}.zip")

    print(f"\n{'=' * 60}")
    print("SURROGATE EVAL")
    print(f"{'=' * 60}")
    eval_env_kwargs = dict(env_kwargs)
    env = ThermostaticEnv(dr_enabled=False, **eval_env_kwargs)
    env = TimeLimit(env, max_episode_steps=336)

    all_errors = []
    all_actions = []
    for ep in range(10):
        obs, _ = env.reset(seed=ep * 100)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            all_errors.append(info["error"])
            all_actions.append(action.copy())

    errors = np.array(all_errors)
    action_arr = np.array(all_actions)
    rmse = np.sqrt(np.mean(errors ** 2))
    within_1 = (errors < 1.0).mean() * 100
    within_05 = (errors < 0.5).mean() * 100
    delta_a = np.mean(np.linalg.norm(np.diff(action_arr, axis=0), axis=1)) if len(action_arr) > 1 else 0.0

    print(f"  RMSE: {rmse:.3f}C | +-1C: {within_1:.1f}% | +-0.5C: {within_05:.1f}%")
    print(f"  Smoothness ||da||: {delta_a:.4f}")
    print(f"  Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
