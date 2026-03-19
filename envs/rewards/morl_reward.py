"""
envs/rewards/morl_reward.py

Multi-Objective reward with asymmetric comfort penalty.

v2 improvements:
  - Quadratic penalty for violations > 2°C (exponential urgency)
  - Asymmetric: undershoot (cold) penalized 1.5x more than overshoot (hot)
    Reason: occupant discomfort is higher when cold; heating is harder than cooling
  - Bonus for being inside comfort band (positive reinforcement)
  - Energy penalty only when inside comfort band (don't punish heating during recovery)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MORLWeights:
    comfort: float = 0.8
    energy: float = 0.2


class MORLReward:

    def __init__(
        self,
        temp_low: float,
        temp_high: float,
        energy_scale: float = 2e-4,
        weights: MORLWeights = MORLWeights(),
        cold_penalty_mult: float = 1.5,
        quadratic_threshold: float = 2.0,
        comfort_bonus: float = 0.1,
    ):
        assert temp_low < temp_high, "temp_low must be < temp_high"
        self.temp_low = float(temp_low)
        self.temp_high = float(temp_high)
        self.energy_scale = float(energy_scale)
        self.w = weights
        self.cold_penalty_mult = cold_penalty_mult
        self.quadratic_threshold = quadratic_threshold
        self.comfort_bonus = comfort_bonus
        self.t_mid = (temp_low + temp_high) / 2.0

    def compute(self, zone_temp: float, hvac_power: float) -> Tuple[float, Dict[str, float]]:

        # --- Comfort reward ---
        if self.temp_low <= zone_temp <= self.temp_high:
            # Inside comfort band: small positive bonus
            # Closer to midpoint = higher bonus
            dist_to_mid = abs(zone_temp - self.t_mid)
            half_range = (self.temp_high - self.temp_low) / 2.0
            r_comfort = self.comfort_bonus * (1.0 - dist_to_mid / half_range)
        else:
            # Outside comfort band: negative penalty
            if zone_temp < self.temp_low:
                # COLD violation — more dangerous
                deviation = self.temp_low - zone_temp

                if deviation > self.quadratic_threshold:
                    # Quadratic penalty for severe cold: -d² effect
                    linear_part = self.quadratic_threshold
                    quad_part = (deviation - self.quadratic_threshold) ** 2
                    r_comfort = -(linear_part + quad_part) * self.cold_penalty_mult
                else:
                    # Linear penalty for mild cold
                    r_comfort = -deviation * self.cold_penalty_mult
            else:
                # HOT violation — less severe (standard linear)
                deviation = zone_temp - self.temp_high
                if deviation > self.quadratic_threshold:
                    linear_part = self.quadratic_threshold
                    quad_part = (deviation - self.quadratic_threshold) ** 2
                    r_comfort = -(linear_part + quad_part)
                else:
                    r_comfort = -deviation

        # --- Energy reward ---
        # Only penalize energy when in comfort band
        # During recovery (outside band), agent should heat freely
        if self.temp_low <= zone_temp <= self.temp_high:
            r_energy = -float(hvac_power) * self.energy_scale
        else:
            # Outside comfort: don't penalize energy use
            # This encourages the agent to USE energy to recover
            r_energy = 0.0

        scalar = self.w.comfort * r_comfort + self.w.energy * r_energy

        vec = {
            "comfort": float(r_comfort),
            "energy": float(r_energy),
            "zone_temp": float(zone_temp),
            "hvac_power": float(hvac_power),
            "w_comfort": float(self.w.comfort),
            "w_energy": float(self.w.energy),
        }
        return scalar, vec