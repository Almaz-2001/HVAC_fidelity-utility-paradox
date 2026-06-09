from __future__ import annotations

import json
import os
import zipfile
from typing import Optional

import numpy as np
import pandas as pd


FORECAST_HORIZONS = [1, 3, 6, 12, 24]
BASIC_TSUP_OBS_DIM = 5
TIME_TSUP_OBS_DIM = 4
HISTORY_TSUP_OBS_DIM = 3
NO_FORECAST_TSUP_OBS_DIM = BASIC_TSUP_OBS_DIM + TIME_TSUP_OBS_DIM + HISTORY_TSUP_OBS_DIM
EXTENDED_TSUP_OBS_DIM = NO_FORECAST_TSUP_OBS_DIM + len(FORECAST_HORIZONS)
SUPPORTED_TSUP_OBS_DIMS = frozenset({NO_FORECAST_TSUP_OBS_DIM, EXTENDED_TSUP_OBS_DIM})
SUPPORTED_TSUP_OBS_ABLATIONS = frozenset(
    {
        "none",
        "no_power",
        "no_delta_t",
        "no_power_no_delta_t",
        "no_prev_action",
        "no_temp_history",
    }
)
SUPPORTED_DELTA_FEATURE_MODES = frozenset({"raw", "causal_smooth"})
SUPPORTED_POWER_FEATURE_MODES = frozenset({"raw", "clipped_log"})
SUPPORTED_T_ZONE_FEATURE_MODES = frozenset({"raw", "comfort_centered"})

PHYS_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
PHYS_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)
T_AMB_LOW = -30.0
T_AMB_HIGH = 45.0
DELTA_T_MAX = 5.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0
P_TOTAL_CLIPPED_LOG_MAX = 2500.0
T_ZONE_COMFORT_CENTER_C = 22.5
T_ZONE_COMFORT_SCALE_C = 3.0

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEATHER_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "data", "surrogate_v2", "boptest_v2_tsupply.csv"),
    os.path.join(_PROJECT_ROOT, "data", "surrogate_v2", "boptest_v2_all.csv"),
    "/app/data/surrogate_v2/boptest_v2_tsupply.csv",
    "/app/data/surrogate_v2/boptest_v2_all.csv",
]


def resolve_weather_csv() -> str:
    for path in _WEATHER_CANDIDATES:
        if os.path.exists(path):
            return path
    return _WEATHER_CANDIDATES[0]


class WeatherLookup:
    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            csv_path = resolve_weather_csv()
        self.csv_path = csv_path
        self.grid = np.zeros((366, 24), dtype=np.float32)
        self.count = np.zeros((366, 24), dtype=np.int32)
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
        self.available = True

    def get(self, hour: float, day: float, noise_std: float = 0.0) -> float:
        if not self.available:
            return 10.0
        d = int(day) % 366
        h = int(hour) % 24
        h_next = (h + 1) % 24
        frac = hour - int(hour)
        t = float(self.grid[d, h]) * (1.0 - frac) + float(self.grid[d, h_next]) * frac
        if noise_std > 0.0:
            t += np.random.normal(0.0, noise_std)
        return float(np.clip(t, -30.0, 45.0))

    def forecast(self, hour: float, day: float, horizon_hours: int) -> float:
        if not self.available:
            return 10.0
        fh = (hour + horizon_hours) % 24
        fd = day + horizon_hours / 24.0
        d = int(fd) % 366
        h = int(fh) % 24
        return float(self.grid[d, h])


def action_to_t_supply(a0: float, t_supply_low: float = T_SUPPLY_LOW, t_supply_high: float = T_SUPPLY_HIGH) -> float:
    return t_supply_low + (float(a0) + 1.0) * 0.5 * (t_supply_high - t_supply_low)


def action_to_fan(a1: float) -> float:
    return float(np.clip((float(a1) + 1.0) * 0.5, 0.0, 1.0))


def encode_hour_cyc(hour: float) -> tuple[float, float]:
    rad = 2.0 * np.pi * (hour / 24.0)
    return float(np.sin(rad)), float(np.cos(rad))


def encode_day_cyc(day: float) -> tuple[float, float]:
    rad = 2.0 * np.pi * (day / 365.0)
    return float(np.sin(rad)), float(np.cos(rad))


def norm_t_amb(t_amb: float) -> float:
    return float(np.clip(2.0 * (t_amb - T_AMB_LOW) / (T_AMB_HIGH - T_AMB_LOW) - 1.0, -1.0, 1.0))


def norm_delta_t(delta_t: float) -> float:
    return float(np.clip(delta_t / DELTA_T_MAX, -1.0, 1.0))


def encode_delta_t_feature(delta_t: float, mode: str | None = None) -> float:
    feature_mode = str(mode or "raw").strip().lower()
    raw_delta = float(delta_t)
    if feature_mode == "raw":
        return norm_delta_t(raw_delta)
    if feature_mode == "causal_smooth":
        clipped = float(np.clip(raw_delta, -1.5, 1.5))
        return float(np.tanh(clipped / 1.25))
    raise ValueError(
        f"Unsupported delta feature mode '{mode}'. "
        f"Supported: {sorted(SUPPORTED_DELTA_FEATURE_MODES)}."
    )


def encode_p_total_feature(p_total: float, mode: str | None = None) -> float:
    feature_mode = str(mode or "raw").strip().lower()
    raw_power = float(max(p_total, 0.0))
    if feature_mode == "raw":
        return float(np.clip(2.0 * (raw_power - PHYS_LOW[2]) / (PHYS_HIGH[2] - PHYS_LOW[2]) - 1.0, -1.0, 1.0))
    if feature_mode == "clipped_log":
        clipped = float(min(raw_power, P_TOTAL_CLIPPED_LOG_MAX))
        return float(2.0 * np.log1p(clipped) / np.log1p(P_TOTAL_CLIPPED_LOG_MAX) - 1.0)
    raise ValueError(
        f"Unsupported power feature mode '{mode}'. "
        f"Supported: {sorted(SUPPORTED_POWER_FEATURE_MODES)}."
    )


def encode_t_zone_feature(t_zone: float, mode: str | None = None) -> float:
    feature_mode = str(mode or "raw").strip().lower()
    raw_t_zone = float(t_zone)
    if feature_mode == "raw":
        return float(np.clip(2.0 * (raw_t_zone - PHYS_LOW[0]) / (PHYS_HIGH[0] - PHYS_LOW[0]) - 1.0, -1.0, 1.0))
    if feature_mode == "comfort_centered":
        return float(np.clip((raw_t_zone - T_ZONE_COMFORT_CENTER_C) / T_ZONE_COMFORT_SCALE_C, -1.0, 1.0))
    raise ValueError(
        f"Unsupported t_zone feature mode '{mode}'. "
        f"Supported: {sorted(SUPPORTED_T_ZONE_FEATURE_MODES)}."
    )


def build_basic_tsup_obs(
    t_zone: float,
    co2: float,
    p_total: float,
    t_supply_prev: float,
    t_amb: float,
    t_zone_feature_mode: str = "raw",
    power_feature_mode: str = "raw",
) -> np.ndarray:
    raw = np.array([t_zone, co2, p_total, t_supply_prev, t_amb], dtype=np.float32)
    obs = 2.0 * (raw - PHYS_LOW) / (PHYS_HIGH - PHYS_LOW) - 1.0
    obs[0] = encode_t_zone_feature(t_zone, mode=t_zone_feature_mode)
    obs[2] = encode_p_total_feature(p_total, mode=power_feature_mode)
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def apply_tsup_obs_ablation(obs: np.ndarray, obs_ablation: str | None = None) -> np.ndarray:
    mode = str(obs_ablation or "none").strip().lower()
    out = np.asarray(obs, dtype=np.float32).copy()
    if mode == "none":
        return out
    if mode == "no_power":
        out[2] = 0.0
        return out
    if mode == "no_delta_t":
        out[-1] = 0.0
        return out
    if mode == "no_power_no_delta_t":
        out[2] = 0.0
        out[-1] = 0.0
        return out
    if mode == "no_prev_action":
        out[-3:-1] = 0.0
        return out
    if mode == "no_temp_history":
        out[3] = 0.0
        out[-3:] = 0.0
        return out
    raise ValueError(
        f"Unsupported tsup observation ablation '{obs_ablation}'. "
        f"Supported: {sorted(SUPPORTED_TSUP_OBS_ABLATIONS)}."
    )


def build_tsup_obs(
    t_zone: float,
    co2: float,
    p_total: float,
    t_supply_prev: float,
    t_amb: float,
    hour: float,
    day: float,
    prev_action: np.ndarray,
    delta_t_zone: float,
    weather: WeatherLookup,
    include_forecast: bool = True,
    obs_ablation: str | None = None,
    delta_feature_mode: str = "raw",
    t_zone_feature_mode: str = "raw",
    power_feature_mode: str = "raw",
) -> np.ndarray:
    phys_norm = build_basic_tsup_obs(
        t_zone,
        co2,
        p_total,
        t_supply_prev,
        t_amb,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )
    hour_sin, hour_cos = encode_hour_cyc(hour)
    day_sin, day_cos = encode_day_cyc(day)
    blocks: list[np.ndarray] = [
        phys_norm,
        np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32),
    ]
    if include_forecast:
        forecasts = [norm_t_amb(weather.forecast(hour, day, horizon)) for horizon in FORECAST_HORIZONS]
        blocks.append(np.array(forecasts, dtype=np.float32))
    blocks.extend(
        [
            np.asarray(prev_action, dtype=np.float32),
            np.array([encode_delta_t_feature(delta_t_zone, mode=delta_feature_mode)], dtype=np.float32),
        ]
    )
    obs = np.concatenate(blocks).astype(np.float32)
    obs = apply_tsup_obs_ablation(obs, obs_ablation=obs_ablation)
    return np.clip(obs, -1.0, 1.0)


def build_extended_tsup_obs(
    t_zone: float,
    co2: float,
    p_total: float,
    t_supply_prev: float,
    t_amb: float,
    hour: float,
    day: float,
    prev_action: np.ndarray,
    delta_t_zone: float,
    weather: WeatherLookup,
    obs_ablation: str | None = None,
    delta_feature_mode: str = "raw",
    t_zone_feature_mode: str = "raw",
    power_feature_mode: str = "raw",
) -> np.ndarray:
    return build_tsup_obs(
        t_zone=t_zone,
        co2=co2,
        p_total=p_total,
        t_supply_prev=t_supply_prev,
        t_amb=t_amb,
        hour=hour,
        day=day,
        prev_action=prev_action,
        delta_t_zone=delta_t_zone,
        weather=weather,
        include_forecast=True,
        obs_ablation=obs_ablation,
        delta_feature_mode=delta_feature_mode,
        t_zone_feature_mode=t_zone_feature_mode,
        power_feature_mode=power_feature_mode,
    )


def infer_tsup_model_obs_dim(model_path: str, default_obs_dim: int = EXTENDED_TSUP_OBS_DIM) -> int:
    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))
    obs_shape = data.get("observation_space", {}).get("_shape", [default_obs_dim])
    obs_dim = int(obs_shape[0] if isinstance(obs_shape, list) else obs_shape)
    if obs_dim not in SUPPORTED_TSUP_OBS_DIMS:
        raise RuntimeError(
            f"Model {model_path} has unsupported obs dim {obs_dim}. "
            f"Supported dims: {sorted(SUPPORTED_TSUP_OBS_DIMS)}."
        )
    return obs_dim
