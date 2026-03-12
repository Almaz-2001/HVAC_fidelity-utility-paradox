"""
surrogate/synthetic_data_gen.py

Генератор синтетических данных "реального" здания для Фазы 2.

Три реалистичных артефакта телеметрии:
  1. Gaussian Noise:     sigma = 0.3 C  (случайный шум датчика PT100)
  2. Telemetry Latency:  задержка наблюдения 1-3 шага (1-3 часа)
  3. Sensor Bias:        постоянное смещение +0.5 C (дрейф калибровки)

"Реальное" здание отличается от surrogate:
  - C_zon_real = 4.2e5 J/K  (surrogate знает 5.3e5, разница 21%)
  - теплопотери через стены  (surrogate не моделирует)
  - нелинейный КПД HVAC

Запуск:
  python surrogate/synthetic_data_gen.py --steps 500 --seed 42
  python surrogate/synthetic_data_gen.py --steps 500 --no_latency --no_bias
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Tuple


class RealBuildingParams:
    C_ZON            = 4.2e5   # J/K
    R_WALL            = 0.015   # K/W
    COP_BASE          = 2.5
    COP_SLOPE         = 0.05
    T_AMB_MEAN        = 5.0
    T_AMB_STD         = 3.0
    Q_INTERNAL_MEAN   = 50.0
    Q_INTERNAL_STD    = 20.0
    # Artifact 1: Gaussian Noise
    SENSOR_TEMP_STD   = 0.3    # C
    SENSOR_POWER_REL  = 0.02   # relative
    # Artifact 2: Telemetry Latency
    LATENCY_MIN       = 1
    LATENCY_MAX       = 3
    # Artifact 3: Sensor Bias
    SENSOR_BIAS       = +0.5   # C
    DT                = 3600.0  # s


def _real_step(t_zone, a0, a1, t_amb, q_int, p):
    fan   = float(np.clip((a1 + 1.0) / 2.0, 0.0, 1.0))
    sp    = float(np.clip((a0 + 1.0) / 2.0, 0.0, 1.0))
    cop   = max(1.0, p.COP_BASE + p.COP_SLOPE * (t_zone - t_amb))
    p_e   = 1406.0 * fan * sp
    q_hvac = p_e * cop
    q_wall = (t_zone - t_amb) / p.R_WALL
    t_next = t_zone + (p.DT / p.C_ZON) * (q_hvac - q_wall + q_int)
    return float(np.clip(t_next, 10.0, 40.0)), p_e


class TelemetrySimulator:
    """
    Имитирует BMS с тремя слоями артефактов:

    t_true
      → [Bias: +0.5 C]       систематическая ошибка калибровки
      → [Noise: N(0, 0.3)]   случайный шум датчика PT100
      → [Buffer: delay=1-3]  задержка передачи данных
      → t_observed
    """

    def __init__(self, params, rng, use_noise, use_latency, use_bias, latency=None):
        self.p           = params
        self.rng         = rng
        self.use_noise   = use_noise
        self.use_latency = use_latency
        self.use_bias    = use_bias
        self.latency     = latency if latency else int(
            rng.integers(params.LATENCY_MIN, params.LATENCY_MAX + 1)
        )
        self.buffer      = deque(maxlen=self.latency + 1)

        print(f"[TELEMETRY] Artifacts:")
        print(f"  1. Gaussian Noise: {'ON' if use_noise else 'OFF'}  sigma={params.SENSOR_TEMP_STD}C")
        print(f"  2. Latency:        {'ON' if use_latency else 'OFF'}  delay={self.latency} steps")
        print(f"  3. Sensor Bias:    {'ON' if use_bias else 'OFF'}  bias={params.SENSOR_BIAS:+.1f}C")

    def observe(self, t_true):
        t = t_true
        if self.use_bias:
            t += self.p.SENSOR_BIAS
        if self.use_noise:
            t += float(self.rng.normal(0.0, self.p.SENSOR_TEMP_STD))
        if self.use_latency:
            self.buffer.append(t)
            t = self.buffer[0]
        return float(t)

    def observe_power(self, p_true):
        if self.use_noise:
            return max(0.0, p_true * (1.0 + float(self.rng.normal(0.0, self.p.SENSOR_POWER_REL))))
        return p_true


def generate(n_steps=500, seed=42, output_dir="/app/data/surrogate",
             policy="mixed", t_init=18.0,
             use_noise=True, use_latency=True, use_bias=True):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(seed)
    params = RealBuildingParams()

    print(f"\n[SYNTH] Real building: C_zon={params.C_ZON:.2e} J/K (surrogate=5.3e5, diff=21%)")
    print(f"[SYNTH] Steps={n_steps}, seed={seed}, policy={policy}\n")

    tel    = TelemetrySimulator(params, rng, use_noise, use_latency, use_bias)
    t_zone = t_init
    rows   = []

    for step in range(n_steps):
        t_amb = float(rng.normal(params.T_AMB_MEAN, params.T_AMB_STD))
        q_int = float(max(0.0, rng.normal(params.Q_INTERNAL_MEAN, params.Q_INTERNAL_STD)))

        if policy == "random":
            a0 = float(rng.uniform(-1, 1))
            a1 = float(rng.uniform(-1, 1))
        elif policy == "comfort":
            a0 = float(np.clip((0.5 if t_zone < 21.0 else -0.3) + rng.normal(0, 0.1), -1, 1))
            a1 = float(np.clip((0.8 if t_zone < 20.0 else 0.3)  + rng.normal(0, 0.1), -1, 1))
        else:
            if rng.random() < 0.5:
                a0, a1 = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
            else:
                a0 = float(np.clip((0.5 if t_zone < 21.0 else -0.3) + rng.normal(0, 0.1), -1, 1))
                a1 = float(np.clip((0.8 if t_zone < 20.0 else 0.3)  + rng.normal(0, 0.1), -1, 1))

        t_next_true, p_true = _real_step(t_zone, a0, a1, t_amb, q_int, params)

        t_zone_obs = tel.observe(t_zone)
        t_next_obs = tel.observe(t_next_true)
        p_obs      = tel.observe_power(p_true)

        rows.append({
            "step":          step,
            "t_zone":        round(t_zone_obs,   4),
            "t_zone_next":   round(t_next_obs,   4),
            "p_total":       round(p_obs,          2),
            "a0_raw":        round(a0,              5),
            "a1_raw":        round(a1,              5),
            "t_amb":         round(t_amb,            3),
            "q_internal":    round(q_int,            2),
            # Ground truth (для валидации)
            "t_zone_true":   round(t_zone,           4),
            "t_next_true":   round(t_next_true,      4),
            "p_true":        round(p_true,            2),
            "latency_steps": tel.latency if use_latency else 0,
            "sensor_bias":   params.SENSOR_BIAS if use_bias else 0.0,
        })

        t_zone = t_next_true

        if step % 100 == 0:
            print(f"[SYNTH] step={step:4d}  T_true={t_zone:.1f}C  "
                  f"T_obs={t_zone_obs:.1f}C  "
                  f"bias={t_zone_obs - t_zone:+.2f}C  P={p_true:.0f}W")

    df = pd.DataFrame(rows)
    suffix = ("_noise" if use_noise else "") + \
             ("_latency" if use_latency else "") + \
             ("_bias" if use_bias else "")
    out = os.path.join(output_dir, f"synthetic_real_{n_steps}{suffix}.csv")
    df.to_csv(out, index=False)

    bias_meas = (df["t_zone"] - df["t_zone_true"]).mean()
    print(f"\n{'='*55}")
    print(f"SYNTHETIC DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Rows: {len(df)}  ->  {out}")
    print(f"\n  True dT:  mean={( df['t_next_true']-df['t_zone_true']).mean():.4f}  "
          f"std={(df['t_next_true']-df['t_zone_true']).std():.4f} C/step")
    print(f"  Obs  dT:  mean={(df['t_zone_next']-df['t_zone']).mean():.4f}  "
          f"std={(df['t_zone_next']-df['t_zone']).std():.4f} C/step")
    print(f"\n  Artifacts effect:")
    print(f"    Sensor bias measured:  {bias_meas:+.4f} C  (true={params.SENSOR_BIAS:+.1f} C)")
    print(f"    Latency:               {tel.latency} steps")
    print(f"\n  Calibration targets:")
    print(f"    C_zon_real  = {params.C_ZON:.3e} J/K  (surrogate: 5.300e+05)")
    print(f"    bias_T_real = {params.SENSOR_BIAS:+.3f} C  <-- inverse_problem.py should find this")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--policy",     default="mixed",
                        choices=["random", "comfort", "mixed"])
    parser.add_argument("--output_dir", default="/app/data/surrogate")
    parser.add_argument("--t_init",     type=float, default=18.0)
    parser.add_argument("--no_noise",   action="store_true")
    parser.add_argument("--no_latency", action="store_true")
    parser.add_argument("--no_bias",    action="store_true")
    args = parser.parse_args()

    generate(
        n_steps=args.steps, seed=args.seed,
        output_dir=args.output_dir, policy=args.policy, t_init=args.t_init,
        use_noise=not args.no_noise,
        use_latency=not args.no_latency,
        use_bias=not args.no_bias,
    )

if __name__ == "__main__":
    main()