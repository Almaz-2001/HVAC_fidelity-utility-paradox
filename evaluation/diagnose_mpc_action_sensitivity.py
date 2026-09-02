"""Is the MPC objective actually sensitive to the action, on each surrogate?

The planner smoke test showed the matched-resolution black box commanding almost
the same supply temperature for a cold zone and a hot one. Two explanations fit:
a bug, or the paper's own mechanism -- a surrogate whose per-step increment is
small moves the zone so little over the horizon that the planning objective is
nearly flat in the action, leaving the optimiser nothing to descend.

This script separates them. It holds the state fixed, sweeps the first action of
the horizon across its range, and reports how much the horizon cost and the
terminal zone temperature actually move. A backend where both are flat cannot be
planned on, for the same reason it cannot be learned on.

No BOPTEST needed; runs in about a minute.

    .venv/Scripts/python.exe evaluation/diagnose_mpc_action_sensitivity.py
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
from envs.tsup_features import WeatherLookup, action_to_t_supply     # noqa: E402

PREREG = ROOT / "configs" / "mpc_baseline_preregistration.yaml"

STATES = [
    ("cold  (T_zone 18 C, T_amb -10 C)", {"t_zone": 18.0, "t_amb": -10.0, "hour": 6.0, "day": 10.0}),
    ("mild  (T_zone 22 C, T_amb   5 C)", {"t_zone": 22.0, "t_amb": 5.0, "hour": 12.0, "day": 40.0}),
]


def horizon_response(adapter, weather, state, model_step_sec, horizon_hours, a0):
    """Terminal zone temperature and total energy after holding action a0."""
    n = max(1, int(round(horizon_hours * 3600.0 / model_step_sec)))
    dt_h = model_step_sec / 3600.0
    t = torch.tensor([state["t_zone"]], dtype=torch.float32)
    energy = 0.0
    with torch.no_grad():
        for k in range(n):
            h = (state["hour"] + k * dt_h) % 24.0
            d = state["day"] + (state["hour"] + k * dt_h) / 24.0
            amb = weather.get(h, d) if getattr(weather, "available", False) else state["t_amb"]
            t, p = adapter(
                t,
                torch.tensor([float(amb)], dtype=torch.float32),
                torch.tensor([h], dtype=torch.float32),
                torch.tensor([d], dtype=torch.float32),
                torch.tensor([a0], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            )
            energy += float(p.sum())
    return float(t.item()), energy


def main() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    horizon_hours = prereg["protocol"]["planner"]["horizon_hours"]

    weather_csv = ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv"
    weather = WeatherLookup(str(weather_csv) if weather_csv.exists() else None)

    grid = np.linspace(-1.0, 1.0, 9)
    print(f"Horizon {horizon_hours} h. Terminal zone temperature as the held action sweeps "
          f"the full command range.\n")

    for backend in prereg["protocol"]["backends"]:
        adapter = load_direct_tsup_adapter(
            kind=backend["surrogate_kind"],
            legacy_model_path=str(ROOT / backend["path"]) if backend.get("path") else None,
            summary_json=str(ROOT / backend["summary_json"]) if backend.get("summary_json") else None,
            device="cpu",
        ).eval()

        print(f"=== {backend['id']}  (model step {backend['model_step_sec']:.0f} s, "
              f"rollout RMSE {backend['rollout_rmse_c']} C) ===")
        for label, state in STATES:
            temps = [horizon_response(adapter, weather, state,
                                      backend["model_step_sec"], horizon_hours, float(a))[0]
                     for a in grid]
            span = max(temps) - min(temps)
            print(f"  {label}")
            print("    T_sup cmd (C): " + " ".join(f"{action_to_t_supply(float(a)):6.1f}" for a in grid))
            print("    T_zone end(C): " + " ".join(f"{t:6.2f}" for t in temps))
            verdict = "flat -- unplannable" if span < 0.5 else ("weak" if span < 2.0 else "responsive")
            print(f"    authority over the horizon: {span:5.2f} C  [{verdict}]")
        print()


if __name__ == "__main__":
    main()
