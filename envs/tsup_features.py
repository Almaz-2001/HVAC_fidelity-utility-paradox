from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


FORECAST_HORIZONS = [1, 3, 6, 12, 24]
EXTENDED_TSUP_OBS_DIM = 17
BASIC_TSUP_OBS_DIM = 5

PHYS_LOW = np.array([5.0, 400.0, 0.0, 18.0, -30.0], dtype=np.float32)
PHYS_HIGH = np.array([40.0, 2000.0, 5500.0, 35.0, 45.0], dtype=np.float32)
T_AMB_LOW = -30.0
T_AMB_HIGH = 45.0
DELTA_T_MAX = 5.0
T_SUPPLY_LOW = 18.0
T_SUPPLY_HIGH = 35.0

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


def build_basic_tsup_obs(t_zone: float, co2: float, p_total: float, t_supply_prev: float, t_amb: float) -> np.ndarray:
    raw = np.array([t_zone, co2, p_total, t_supply_prev, t_amb], dtype=np.float32)
    obs = 2.0 * (raw - PHYS_LOW) / (PHYS_HIGH - PHYS_LOW) - 1.0
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


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
) -> np.ndarray:
    phys_norm = build_basic_tsup_obs(t_zone, co2, p_total, t_supply_prev, t_amb)
    hour_sin, hour_cos = encode_hour_cyc(hour)
    day_sin, day_cos = encode_day_cyc(day)
    forecasts = [norm_t_amb(weather.forecast(hour, day, horizon)) for horizon in FORECAST_HORIZONS]
    obs = np.concatenate(
        [
            phys_norm,
            np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32),
            np.array(forecasts, dtype=np.float32),
            np.asarray(prev_action, dtype=np.float32),
            np.array([norm_delta_t(delta_t_zone)], dtype=np.float32),
        ]
    ).astype(np.float32)
    return np.clip(obs, -1.0, 1.0)
