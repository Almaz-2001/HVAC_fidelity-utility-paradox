"""Receding-horizon MPC baseline planning on the project's surrogates.

Why this exists
---------------
Every controller comparison in the study is against BOPTEST's built-in PI, which
is a weak reference. `layers/safety/fallback.py::SurrogateMPCFallback` is not a
usable substitute: it was written as a runtime safety layer, it freezes ambient
temperature across the horizon, and it optimises a single action pair that is
then held for every horizon step. Reported as "an MPC baseline" that would be a
straw man.

This module implements the baseline properly:

* the decision variable is a SEQUENCE of H action pairs, not one pair;
* ambient temperature comes from the forecast at each horizon step, from the
  same WeatherLookup the RL agent's 17-D observation uses, so neither controller
  gets privileged information;
* only the first action is applied and the problem is re-solved at the next
  control step (receding horizon), warm-started from the shifted previous
  solution;
* the horizon is specified in HOURS and converted per backend using that
  backend's native step, so "6 h" means the same physical lookahead on the
  hourly BB surrogate and on the 15-minute GB twin.

The objective mirrors the structure of the RL training reward -- quadratic
comfort-band violation plus normalised energy -- rather than the evaluation
metric m_s. Optimising the evaluation metric directly would flatter the
baseline and make the comparison meaningless.

The point of the experiment is the cross-backend contrast: the same planner on
the coarse black-box surrogate and on the accurate calibrated twin. See
configs/mpc_baseline_preregistration.yaml for the hypothesis and its thresholds,
fixed before any of this was run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class RecedingHorizonMPC:
    """Gradient-based receding-horizon planner over a differentiable surrogate."""

    def __init__(
        self,
        adapter: torch.nn.Module,
        weather: Any,
        *,
        model_step_sec: float,
        horizon_hours: float = 6.0,
        n_iters: int = 60,
        lr: float = 0.08,
        t_low: float = 21.0,
        t_high: float = 24.0,
        lambda_comfort: float = 60.0,
        lambda_energy: float = 1.0,
        warm_start: bool = True,
    ) -> None:
        self.model = adapter.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.weather = weather
        self.model_step_sec = float(model_step_sec)
        self.horizon = max(1, int(round(horizon_hours * 3600.0 / self.model_step_sec)))
        self.horizon_hours = float(horizon_hours)
        self.n_iters = int(n_iters)
        self.lr = float(lr)
        self.t_low = float(t_low)
        self.t_high = float(t_high)
        self.lambda_comfort = float(lambda_comfort)
        self.lambda_energy = float(lambda_energy)
        self.warm_start = bool(warm_start)
        self.p_max = float(getattr(self.model, "P_MAX", 5500.0))
        self._prev: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def _initial_sequence(self, t_zone: float) -> torch.Tensor:
        """Unconstrained (atanh-space) starting point, shape (H, 2)."""
        if self.warm_start and self._prev is not None and self._prev.shape[0] == self.horizon:
            # shift the previous plan one step and repeat its tail
            seq = torch.roll(self._prev.detach(), shifts=-1, dims=0)
            seq[-1] = self._prev.detach()[-1]
            return seq.clone().requires_grad_(True)

        t_mid = 0.5 * (self.t_low + self.t_high)
        a0 = float(np.clip((t_mid - t_zone) / 3.0, -0.8, 0.8))
        seq = torch.zeros((self.horizon, 2), dtype=torch.float32)
        seq[:, 0] = math.atanh(min(max(a0, -0.99), 0.99))
        seq[:, 1] = math.atanh(0.3)
        return seq.requires_grad_(True)

    def _forecast_amb(self, hour: float, day: float, step: int) -> float:
        """Ambient at horizon step `step`, from the same source the agent sees."""
        dt_h = self.model_step_sec / 3600.0
        h = (hour + step * dt_h) % 24.0
        d = day + (hour + step * dt_h) / 24.0
        if self.weather is not None and getattr(self.weather, "available", False):
            return float(self.weather.get(h, d))
        return float("nan")

    # ------------------------------------------------------------------ #
    def compute(self, state: dict[str, float]) -> np.ndarray:
        t_zone = float(state.get("t_zone", 22.0))
        t_amb_now = float(state.get("t_amb", 10.0))
        hour = float(state.get("hour", 12.0))
        day = float(state.get("day", 180.0))

        amb = []
        for k in range(self.horizon):
            a = self._forecast_amb(hour, day, k)
            amb.append(t_amb_now if math.isnan(a) else a)
        amb_t = torch.tensor(amb, dtype=torch.float32)

        dt_h = self.model_step_sec / 3600.0
        hours_t = torch.tensor([(hour + k * dt_h) % 24.0 for k in range(self.horizon)],
                               dtype=torch.float32)
        days_t = torch.tensor([day + (hour + k * dt_h) / 24.0 for k in range(self.horizon)],
                              dtype=torch.float32)

        seq = self._initial_sequence(t_zone)
        opt = torch.optim.Adam([seq], lr=self.lr)

        best_cost = float("inf")
        best_seq = seq.detach().clone()

        for _ in range(self.n_iters):
            opt.zero_grad()
            a = torch.tanh(seq)
            t_curr = torch.tensor([t_zone], dtype=torch.float32)
            energy = torch.zeros(())
            comfort = torch.zeros(())

            for k in range(self.horizon):
                t_curr, p = self.model(
                    t_curr,
                    amb_t[k:k + 1],
                    hours_t[k:k + 1],
                    days_t[k:k + 1],
                    a[k, 0:1],
                    a[k, 1:2],
                )
                energy = energy + p.sum()
                comfort = comfort + (torch.relu(self.t_low - t_curr) ** 2).sum() \
                                  + (torch.relu(t_curr - self.t_high) ** 2).sum()

            cost = (self.lambda_comfort * comfort
                    + self.lambda_energy * energy / (self.p_max * self.horizon))
            cost.backward()
            opt.step()

            # Keep the unconstrained variable off the tanh saturation shelf.
            # Without this the optimiser can push seq far enough that
            # dtanh/dseq ~ 0, the gradient dies, and the warm start carries the
            # dead sequence into every later control step: measured as a0 pinned
            # at -1.000 from step 0 for a whole 14-day window, on every backend,
            # producing bit-identical results for three different surrogates.
            # tanh(2.5) = 0.987, so the full command range stays reachable.
            with torch.no_grad():
                seq.clamp_(-2.5, 2.5)

            with torch.no_grad():
                c = float(cost.item())
                if c < best_cost:
                    best_cost = c
                    best_seq = seq.detach().clone()

        self._prev = best_seq
        first = torch.tanh(best_seq[0]).numpy()
        return np.clip(first, -1.0, 1.0).astype(np.float32)
