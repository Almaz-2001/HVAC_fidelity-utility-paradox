"""Decisive audit of the matched-resolution BB response sign.

Three independent probes, because a claim this consequential should not rest on
one call path:

  1. adapter.forward(...)            -- what evaluation/mpc_baseline.py uses
  2. adapter.step_with_aux_numpy(...)-- what envs/backends/surrogate_backend.py
                                        calls during RL training, i.e. exactly
                                        what the policy experienced
  3. the frozen live-BOPTEST trace of the controller trained on this surrogate
                                     -- if the model taught an inverted policy,
                                        the deployed actions should be inverted
                                        against the zone error

Probe 3 is the one that matters: it closes the chain from model pathology to the
observed controller failure, or breaks it.

    .venv/Scripts/python.exe evaluation/audit_matched_bb_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter   # noqa: E402
from envs.tsup_features import action_to_t_supply                    # noqa: E402

BACKENDS = {
    "BB hourly (usable)": dict(kind="legacy_v3", step=3600.0,
                               path="outputs/surrogate_v2/rc_node_v3_tsupply.pt"),
    "BB matched 15-min": dict(kind="legacy_v3", step=900.0,
                              path="outputs/surrogate_v3_15min_matched/rc_node_v3_15min_matched.pt"),
    "GB calibrated": dict(kind="v35_calibrated", step=900.0, path=None,
                          summary="outputs/surrogate_v35_inverse_boptest_15min_power_head_only/"
                                  "calibration_summary_boptest_v35.json"),
}

TRACES = {
    "BB hourly (usable)": "outputs/bestest_air_article7_style_15min/traces/{w}_thermostatic.csv",
    "BB matched 15-min": "outputs/bestest_air_pure_v3_15min/traces/{w}_thermostatic.csv",
}

N = 400
RNG = np.random.default_rng(42)


def load(name: str):
    b = BACKENDS[name]
    return load_direct_tsup_adapter(
        kind=b["kind"],
        legacy_model_path=str(ROOT / b["path"]) if b.get("path") else None,
        summary_json=str(ROOT / b["summary"]) if b.get("summary") else None,
        device="cpu",
    ).eval()


def probe_model(name: str) -> None:
    adapter = load(name)
    states = list(zip(RNG.uniform(16, 30, N), RNG.uniform(-20, 35, N),
                      RNG.uniform(0, 24, N), RNG.uniform(0, 365, N)))
    ok_fwd = ok_np = 0
    dT_fwd = []
    for z, a, h, d in states:
        args = dict(t_zone=float(z), t_amb=float(a), hour=float(h), day=float(d), a1=0.5)
        lo_np = adapter.step_with_aux_numpy(**args, a0=-0.9)["t_next"]
        hi_np = adapter.step_with_aux_numpy(**args, a0=+0.9)["t_next"]
        if hi_np >= lo_np - 1e-6:
            ok_np += 1

        with torch.no_grad():
            t = lambda v: torch.tensor([float(v)], dtype=torch.float32)   # noqa: E731
            lo_f = adapter(t(z), t(a), t(h), t(d), t(-0.9), t(0.5))[0].item()
            hi_f = adapter(t(z), t(a), t(h), t(d), t(+0.9), t(0.5))[0].item()
        if hi_f >= lo_f - 1e-6:
            ok_fwd += 1
        dT_fwd.append(hi_f - lo_f)

    print(f"  {name:<22} forward {100*ok_fwd/N:5.1f}%   "
          f"step_with_aux_numpy {100*ok_np/N:5.1f}%   "
          f"median dT(hot cmd - cold cmd) {np.median(dT_fwd):+.3f} C")


def probe_trace(name: str) -> None:
    """On the live building, did the deployed policy heat when the zone was cold?"""
    for window in ("peak_heat_window", "typical_heat_window"):
        path = ROOT / TRACES[name].format(w=window)
        if not path.exists():
            print(f"  {name:<22} {window:<20} trace not found: {path.relative_to(ROOT)}")
            continue
        df = pd.read_csv(path)
        if "t_supply_cmd_c" not in df or "t_zone_c" not in df:
            print(f"  {name:<22} {window:<20} trace lacks the needed columns")
            continue
        cold = df[df.t_zone_c < 21.0]
        hot = df[df.t_zone_c > 24.0]
        if len(cold) < 5 or len(hot) < 5:
            print(f"  {name:<22} {window:<20} too few out-of-band samples "
                  f"(cold {len(cold)}, hot {len(hot)})")
            continue
        mc, mh = cold.t_supply_cmd_c.mean(), hot.t_supply_cmd_c.mean()
        verdict = "sensible" if mc > mh else "INVERTED"
        print(f"  {name:<22} {window:<20} when cold -> T_sup {mc:5.1f} C | "
              f"when hot -> T_sup {mh:5.1f} C   [{verdict}]  "
              f"(n={len(cold)}/{len(hot)})")


def main() -> None:
    print("1-2. Model response sign, two call paths, 400 sampled states")
    print("     (correct = a hotter supply command does not cool the zone)\n")
    for name in BACKENDS:
        probe_model(name)

    print("\n3. Deployed behaviour on live BOPTEST, from the frozen traces")
    print("   (correct = the policy commands a hotter supply when the zone is below band)\n")
    for name in TRACES:
        probe_trace(name)


if __name__ == "__main__":
    main()
