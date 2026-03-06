"""
envs/backends/surrogate_backend.py

SurrogateBackend — быстрая замена BOPTESTBackend для обучения агента.

Использует обученный RC Neural ODE surrogate вместо BOPTEST FMU.
Скорость: ~1,400,000 шагов/сек vs ~1 шаг/сек у BOPTEST.

Конфигурация в env.yaml:
    backend: surrogate
    surrogate_path: /app/outputs/surrogate/rc_node_best.pt

Observation space (идентично BOPTESTBackend):
    [T_zone_norm, CO2_norm, P_cool_norm, P_fan_norm] ∈ [-1, 1]^4

Action space (идентично BOPTESTBackend):
    [a0_setpoint, a1_fan] ∈ [-1, 1]^2
"""

from __future__ import annotations

import os
import numpy as np
import torch
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from envs.base_env import HVACBaseEnv
from surrogate.rc_node import RCNeuralODE


# Физические границы (идентично BOPTESTBackend)
_OBS_LOW  = np.array([15.0,  400.0,    0.0,   0.0], dtype=np.float32)
_OBS_HIGH = np.array([35.0, 2000.0, 5000.0, 500.0], dtype=np.float32)

# Зона комфорта
_T_LOW  = 21.0   # °C
_T_HIGH = 25.0   # °C

# Начальные условия (типичные для bestest_air)
_T_INIT_MEAN = 18.0   # °C — холодный старт
_T_INIT_STD  = 2.0    # °C — небольшой разброс между эпизодами
_CO2_INIT    = 800.0  # ppm
_P_COOL_INIT = 0.0    # W
_P_FAN_INIT  = 0.0    # W


class SurrogateBackend(HVACBaseEnv):
    """
    Gymnasium-совместимая среда на основе RC Neural ODE surrogate.

    Воспроизводит интерфейс BOPTESTBackend:
        - observation_space: Box(-1, 1, shape=(4,))
        - action_space:      Box(-1, 1, shape=(2,))
        - step() возвращает (obs, reward, terminated, truncated, info)
        - info содержит reward_vector и safety метрику
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.cfg = config

        # Параметры среды
        self.max_episode_steps = int(config.get("max_episode_steps", 2000))
        self.step_count        = 0

        # MORL веса
        morl = config.get("morl", {})
        self.w_comfort   = float(morl.get("w_comfort",   0.8))
        self.w_energy    = float(morl.get("w_energy",    0.2))
        self.energy_scale = float(morl.get("energy_scale", 2e-4))
        self.temp_low    = float(morl.get("temp_low",  _T_LOW))
        self.temp_high   = float(morl.get("temp_high", _T_HIGH))

        # Spaces (идентично BOPTESTBackend)
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Загружаем surrogate модель
        surrogate_path = config.get(
            "surrogate_path",
            "/app/outputs/surrogate/rc_node_best.pt"
        )
        self.model  = self._load_surrogate(surrogate_path)
        self.device = next(self.model.parameters()).device

        # Состояние среды
        self._t_zone  = _T_INIT_MEAN
        self._co2     = _CO2_INIT
        self._p_cool  = _P_COOL_INIT
        self._p_fan   = _P_FAN_INIT

        # Safety metric счётчики (Wang et al., 2024)
        self._total_steps      = 0
        self._violation_steps  = 0
        self._max_overshoot    = 0.0
        self._max_undershoot   = 0.0

        print(f"[SURROGATE] Loaded: {surrogate_path}")
        print(f"[SURROGATE] w_comfort={self.w_comfort}, "
              f"w_energy={self.w_energy}, "
              f"energy_scale={self.energy_scale}")

    # -----------------------------------------------------------------------
    # Загрузка модели
    # -----------------------------------------------------------------------

    def _load_surrogate(self, path: str) -> RCNeuralODE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Surrogate model not found: {path}\n"
                "Run: python surrogate/train_surrogate.py first"
            )
        checkpoint = torch.load(path, map_location="cpu")
        model = RCNeuralODE(
            hidden_dim=checkpoint.get("hidden_dim", 64),
            n_layers=checkpoint.get("n_layers", 3),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.update_normalization(
            checkpoint.get("t_mean", 20.0),
            checkpoint.get("t_std",  3.3),
        )
        model.eval()
        return model

    # -----------------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        # Случайная начальная температура для разнообразия эпизодов
        self._t_zone = np.random.normal(_T_INIT_MEAN, _T_INIT_STD)
        self._t_zone = float(np.clip(self._t_zone, _OBS_LOW[0], _OBS_HIGH[0]))
        self._co2    = _CO2_INIT + np.random.normal(0, 50)
        self._p_cool = 0.0
        self._p_fan  = 0.0

        self.step_count = 0

        # Сброс safety счётчиков
        self._total_steps     = 0
        self._violation_steps = 0
        self._max_overshoot   = 0.0
        self._max_undershoot  = 0.0

        obs  = self._make_obs()
        info = {}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        a0 = float(action[0])
        a1 = float(action[1]) if len(action) > 1 else 0.0

        # Прогоняем surrogate
        t_next, p_total = self._surrogate_step(a0, a1)

        # Обновляем CO2 (простая модель: снижается при вентиляции)
        fan_signal = float(np.clip((a1 + 1.0) / 2.0, 0.0, 1.0))
        co2_change = -50.0 * fan_signal + 10.0   # вентиляция снижает CO2
        self._co2  = float(np.clip(self._co2 + co2_change, 400, 2000))

        # Мощность по компонентам (приближение)
        self._p_cool = float(p_total * 0.85)
        self._p_fan  = float(p_total * 0.15)

        # Обновляем состояние
        self._t_zone = float(t_next)

        # Награды
        r_comfort = self._comfort_reward(self._t_zone)
        r_energy  = -self.energy_scale * p_total
        reward    = self.w_comfort * r_comfort + self.w_energy * r_energy

        # Safety metric
        self._update_safety(self._t_zone)

        self.step_count += 1
        terminated = False
        truncated  = self.step_count >= self.max_episode_steps

        obs  = self._make_obs()
        info = {
            "reward_vector": {
                "comfort":    r_comfort,
                "energy":     r_energy,
                "zone_temp":  self._t_zone,
                "hvac_power": p_total,
                "w_comfort":  self.w_comfort,
                "w_energy":   self.w_energy,
            },
            "safety": self.get_safety_metric(),
        }

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        pass

    # -----------------------------------------------------------------------
    # Surrogate forward pass
    # -----------------------------------------------------------------------

    def _surrogate_step(
        self, a0: float, a1: float
    ) -> Tuple[float, float]:
        """Один шаг RC Neural ODE."""
        with torch.no_grad():
            t  = torch.tensor([self._t_zone], dtype=torch.float32)
            a0_t = torch.tensor([a0],         dtype=torch.float32)
            a1_t = torch.tensor([a1],         dtype=torch.float32)
            t_next, p_total = self.model(t, a0_t, a1_t)
        return float(t_next[0]), float(p_total[0])

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        raw = np.array([
            self._t_zone,
            self._co2,
            self._p_cool,
            self._p_fan,
        ], dtype=np.float32)
        obs = 2.0 * (raw - _OBS_LOW) / (_OBS_HIGH - _OBS_LOW) - 1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _comfort_reward(self, t_c: float) -> float:
        if t_c < self.temp_low:
            return -(self.temp_low - t_c)
        elif t_c > self.temp_high:
            return -(t_c - self.temp_high)
        return 0.0

    # -----------------------------------------------------------------------
    # Safety metric (Wang et al., 2024)
    # -----------------------------------------------------------------------

    def _update_safety(self, t_c: float) -> None:
        self._total_steps += 1
        if t_c > self.temp_high:
            self._violation_steps += 1
            overshoot = (t_c - self.temp_high) / self.temp_high
            self._max_overshoot = max(self._max_overshoot, overshoot)
        elif t_c < self.temp_low:
            self._violation_steps += 1
            undershoot = (self.temp_low - t_c) / self.temp_low
            self._max_undershoot = max(self._max_undershoot, undershoot)

    def get_safety_metric(self) -> dict:
        if self._total_steps == 0:
            return {"r_time": 0.0, "r_sev": 0.0, "m_s": 0.0,
                    "violation_steps": 0, "total_steps": 0}
        r_time = self._violation_steps / self._total_steps
        r_sev  = max(self._max_overshoot, self._max_undershoot)
        m_s    = r_time + r_sev
        return {
            "r_time":          r_time,
            "r_sev":           r_sev,
            "m_s":             m_s,
            "violation_steps": self._violation_steps,
            "total_steps":     self._total_steps,
        }