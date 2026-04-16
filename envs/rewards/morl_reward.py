from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple


@dataclass
class MORLWeights:
    comfort: float = 0.8
    energy: float = 0.2


class MORLReward:
    """
    Legacy Sinergym MORL reward.

    The thesis-era setup made energy effectively free outside the comfort band,
    which causes most Pareto weight settings to collapse to the same policy.
    This version keeps the asymmetric comfort logic but adds:
    - smooth comfort shaping inside the band around its midpoint
    - a reduced but non-zero energy cost during recovery
    - a soft penalty for very large HVAC power spikes
    """

    def __init__(
        self,
        temp_low: float,
        temp_high: float,
        energy_scale: float = 1.2e-4,
        weights: MORLWeights = MORLWeights(),
        cold_penalty_mult: float = 1.5,
        quadratic_threshold: float = 2.0,
        comfort_bonus: float = 0.02,
        inband_temp_penalty_scale: float = 0.25,
        recovery_energy_mult: float = 0.60,
        peak_power_threshold: float = 700.0,
        peak_power_scale: float = 8e-5,
        action_penalty_scale: float = 0.03,
        reward_shape: str = "linear",
        target_temp: float | None = None,
        exp_alpha: float = 2.5,
        exp_scale: float = 0.04,
        gaussian_sigma: float = 1.0,
        gaussian_peak: float = 1.0,
        gaussian_offset: float = 0.2,
        cubic_scale: float = 0.60,
    ):
        assert temp_low < temp_high, "temp_low must be < temp_high"
        self.temp_low = float(temp_low)
        self.temp_high = float(temp_high)
        self.energy_scale = float(energy_scale)
        self.w = weights
        self.cold_penalty_mult = float(cold_penalty_mult)
        self.quadratic_threshold = float(quadratic_threshold)
        self.comfort_bonus = float(comfort_bonus)
        self.inband_temp_penalty_scale = float(inband_temp_penalty_scale)
        self.recovery_energy_mult = float(recovery_energy_mult)
        self.peak_power_threshold = float(peak_power_threshold)
        self.peak_power_scale = float(peak_power_scale)
        self.action_penalty_scale = float(action_penalty_scale)
        self.t_mid = 0.5 * (self.temp_low + self.temp_high)
        self.half_range = max(0.5 * (self.temp_high - self.temp_low), 1e-6)
        self.reward_shape = str(reward_shape).strip().lower()
        self.target_temp = float(target_temp) if target_temp is not None else self.t_mid
        self.exp_alpha = float(exp_alpha)
        self.exp_scale = float(exp_scale)
        self.gaussian_sigma = max(float(gaussian_sigma), 1e-6)
        self.gaussian_peak = float(gaussian_peak)
        self.gaussian_offset = float(gaussian_offset)
        self.cubic_scale = float(cubic_scale)

    def _outside_band_penalty(self, zone_temp: float) -> float:
        if self.temp_low <= zone_temp <= self.temp_high:
            return 0.0

        if zone_temp < self.temp_low:
            deviation = self.temp_low - zone_temp
            if deviation > self.quadratic_threshold:
                linear_part = self.quadratic_threshold
                quad_part = (deviation - self.quadratic_threshold) ** 2
                return -(linear_part + quad_part) * self.cold_penalty_mult
            return -deviation * self.cold_penalty_mult

        deviation = zone_temp - self.temp_high
        if deviation > self.quadratic_threshold:
            linear_part = self.quadratic_threshold
            quad_part = (deviation - self.quadratic_threshold) ** 2
            return -(linear_part + quad_part)
        return -deviation

    def _apply_cold_asymmetry(self, zone_temp: float, score: float) -> float:
        if zone_temp < self.target_temp and score < 0.0:
            return score * self.cold_penalty_mult
        return score

    def _comfort_linear(self, zone_temp: float) -> float:
        if self.temp_low <= zone_temp <= self.temp_high:
            dist_norm = abs(zone_temp - self.t_mid) / self.half_range
            return self.comfort_bonus - self.inband_temp_penalty_scale * (dist_norm ** 2)
        return self._outside_band_penalty(zone_temp)

    def _comfort_exponential(self, zone_temp: float) -> float:
        deviation_norm = abs(zone_temp - self.target_temp) / self.half_range
        score = self.comfort_bonus - self.exp_scale * (math.exp(self.exp_alpha * deviation_norm) - 1.0)
        score = self._apply_cold_asymmetry(zone_temp, score)
        return score + self._outside_band_penalty(zone_temp)

    def _comfort_gaussian(self, zone_temp: float) -> float:
        deviation = abs(zone_temp - self.target_temp)
        score = self.gaussian_peak * math.exp(-0.5 * ((deviation / self.gaussian_sigma) ** 2)) - self.gaussian_offset
        score = self._apply_cold_asymmetry(zone_temp, score)
        return score + self._outside_band_penalty(zone_temp)

    def _comfort_cubic(self, zone_temp: float) -> float:
        deviation_norm = abs(zone_temp - self.target_temp) / self.half_range
        score = self.comfort_bonus - self.cubic_scale * (deviation_norm ** 3)
        score = self._apply_cold_asymmetry(zone_temp, score)
        return score + self._outside_band_penalty(zone_temp)

    def _comfort_term(self, zone_temp: float) -> float:
        if self.reward_shape == "linear":
            return self._comfort_linear(zone_temp)
        if self.reward_shape == "exponential":
            return self._comfort_exponential(zone_temp)
        if self.reward_shape == "gaussian":
            return self._comfort_gaussian(zone_temp)
        if self.reward_shape == "cubic":
            return self._comfort_cubic(zone_temp)
        raise ValueError(f"Unsupported reward_shape: {self.reward_shape}")

    def _energy_term(self, zone_temp: float, hvac_power: float) -> float:
        in_band = self.temp_low <= zone_temp <= self.temp_high
        scale = self.energy_scale if in_band else self.energy_scale * self.recovery_energy_mult
        r_energy = -float(hvac_power) * scale

        if hvac_power > self.peak_power_threshold:
            r_energy -= (float(hvac_power) - self.peak_power_threshold) * self.peak_power_scale

        return r_energy

    def compute(self, zone_temp: float, hvac_power: float, action_mag: float | None = None) -> Tuple[float, Dict[str, float]]:
        r_comfort = self._comfort_term(float(zone_temp))
        r_energy = self._energy_term(float(zone_temp), float(hvac_power))
        r_control = -float(action_mag) * self.action_penalty_scale if action_mag is not None else 0.0

        scalar = self.w.comfort * r_comfort + self.w.energy * (r_energy + r_control)

        vec = {
            "comfort": float(r_comfort),
            "energy": float(r_energy),
            "control": float(r_control),
            "zone_temp": float(zone_temp),
            "hvac_power": float(hvac_power),
            "reward_shape": self.reward_shape,
            "target_temp": float(self.target_temp),
            "w_comfort": float(self.w.comfort),
            "w_energy": float(self.w.energy),
        }
        return scalar, vec
