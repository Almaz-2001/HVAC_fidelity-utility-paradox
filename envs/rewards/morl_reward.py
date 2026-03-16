from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MORLWeights:
    comfort: float = 0.6
    energy: float = 0.4


class MORLReward:
    """
    
    """

    def __init__(
        self,
        temp_low: float,
        temp_high: float,
        energy_scale: float = 1e-6,
        weights: MORLWeights = MORLWeights(),
    ):
        assert temp_low < temp_high, "temp_low must be < temp_high"
        self.temp_low = float(temp_low)
        self.temp_high = float(temp_high)
        self.energy_scale = float(energy_scale)
        self.w = weights

    def compute(self, zone_temp: float, hvac_power: float) -> Tuple[float, Dict[str, float]]:
        # Comfort penalty: 0 внутри диапазона, отрицательный штраф вне диапазона
        if self.temp_low <= zone_temp <= self.temp_high:
            r_comfort = 0.0
        else:
            d = min(abs(zone_temp - self.temp_low), abs(zone_temp - self.temp_high))
            r_comfort = -float(d)

        # Energy penalty: отрицательный, хотим минимизировать потребление
        r_energy = -float(hvac_power) * self.energy_scale

        scalar = self.w.comfort * r_comfort + self.w.energy * r_energy

        vec = {
            "comfort": r_comfort,
            "energy": r_energy,
            "zone_temp": float(zone_temp),
            "hvac_power": float(hvac_power),
            "w_comfort": float(self.w.comfort),
            "w_energy": float(self.w.energy),
        }
        return scalar, vec
