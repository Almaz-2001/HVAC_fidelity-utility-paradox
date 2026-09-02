"""Directional-validity audit of every planning surrogate.

The manuscript reports a directional check for the calibrated grey-box twin
(correct response sign in 100% of 400 sampled states). The black-box surrogates
were never audited the same way. The MPC action-sensitivity diagnostic suggested
the matched-resolution black box responds with an INVERTED sign -- a warmer
supply-temperature command producing a colder zone -- so this script settles it
on the same terms the twin was audited on.

Two probes, because they answer different questions:

  one-step   is the sign wrong in the learned transition itself?
  rollout    is the sign wrong after the horizon the planner and the policy see?

A model can be locally correct and globally inverted if the error compounds, so
both are reported. Fan command is swept as well, since holding it fixed at one
value could hide or manufacture the effect.

No BOPTEST needed.

    .venv/Scripts/python.exe evaluation/check_surrogate_response_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter   # noqa: E402
from envs.tsup_features import WeatherLookup                          # noqa: E402

PREREG = ROOT / "configs" / "mpc_baseline_preregistration.yaml"
N_STATES = 400
RNG = np.random.default_rng(42)


def sample_states(n: int) -> list[dict]:
    return [{"t_zone": float(z), "t_amb": float(a), "hour": float(h), "day": float(d)}
            for z, a, h, d in zip(RNG.uniform(16.0, 30.0, n),
                                  RNG.uniform(-20.0, 35.0, n),
                                  RNG.uniform(0.0, 24.0, n),
                                  RNG.uniform(0.0, 365.0, n))]


def response(adapter, weather, state, model_step_sec, a0, a1, n_steps):
    t = torch.tensor([state["t_zone"]], dtype=torch.float32)
    dt_h = model_step_sec / 3600.0
    with torch.no_grad():
        for k in range(n_steps):
            h = (state["hour"] + k * dt_h) % 24.0
            d = state["day"] + (state["hour"] + k * dt_h) / 24.0
            amb = weather.get(h, d) if getattr(weather, "available", False) else state["t_amb"]
            t, _ = adapter(t,
                           torch.tensor([float(amb)], dtype=torch.float32),
                           torch.tensor([h], dtype=torch.float32),
                           torch.tensor([d], dtype=torch.float32),
                           torch.tensor([a0], dtype=torch.float32),
                           torch.tensor([a1], dtype=torch.float32))
    return float(t.item())


def main() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    weather_csv = ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv"
    weather = WeatherLookup(str(weather_csv) if weather_csv.exists() else None)
    states = sample_states(N_STATES)

    print(f"Directional validity over {N_STATES} sampled states.")
    print("Correct sign = raising the supply-temperature command does not lower the zone "
          "temperature.\n")
    header = f"{'backend':<20}{'fan':>6}{'one-step':>12}{'6 h rollout':>14}"
    print(header)
    print("-" * len(header))

    for backend in prereg["protocol"]["backends"]:
        adapter = load_direct_tsup_adapter(
            kind=backend["surrogate_kind"],
            legacy_model_path=str(ROOT / backend["path"]) if backend.get("path") else None,
            summary_json=str(ROOT / backend["summary_json"]) if backend.get("summary_json") else None,
            device="cpu",
        ).eval()
        step = backend["model_step_sec"]
        n_roll = max(1, int(round(6.0 * 3600.0 / step)))

        for a1 in (0.0, 0.5, 1.0):
            ok_one = ok_roll = 0
            for s in states:
                lo1 = response(adapter, weather, s, step, -0.9, a1, 1)
                hi1 = response(adapter, weather, s, step, +0.9, a1, 1)
                if hi1 >= lo1 - 1e-6:
                    ok_one += 1
                loR = response(adapter, weather, s, step, -0.9, a1, n_roll)
                hiR = response(adapter, weather, s, step, +0.9, a1, n_roll)
                if hiR >= loR - 1e-6:
                    ok_roll += 1
            print(f"{backend['id']:<20}{a1:>6.1f}"
                  f"{100 * ok_one / N_STATES:>11.1f}%"
                  f"{100 * ok_roll / N_STATES:>13.1f}%")
        print()


if __name__ == "__main__":
    main()
