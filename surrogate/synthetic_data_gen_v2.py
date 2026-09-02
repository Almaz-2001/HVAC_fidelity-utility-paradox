from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surrogate.rc_node_v2 import RCNeuralODEv2


@dataclass
class SyntheticBenchmarkParamsV2:
    surrogate_czon_ref: float = 5.3e5
    c_zon_true: float = 4.2e5
    bias_t: float = 0.5
    noise_t_std: float = 0.08
    power_scale: float = 1.04
    power_bias_w: float = 35.0
    power_noise_rel: float = 0.015

    @property
    def scale_dT_true(self) -> float:
        return float(self.surrogate_czon_ref / self.c_zon_true)


def _load_driver_slice(
    csv_path: str,
    n_steps: int,
    policy: str | None,
    season: str | None,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if policy:
        df = df[df["policy"] == policy]
    if season:
        df = df[df["season"] == season]
    if len(df) < n_steps:
        raise ValueError(
            f"Driver dataset slice too small: requested {n_steps}, available {len(df)}"
        )

    rng = np.random.default_rng(seed)
    if {"policy", "season"}.issubset(df.columns):
        df = df.sort_values(["season", "policy", "step"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Use one contiguous slice to keep trajectories coherent.
    start_max = len(df) - n_steps
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    return df.iloc[start:start + n_steps].reset_index(drop=True)


def _surrogate_forward(
    driver_df: pd.DataFrame,
    ckpt_path: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=int(ckpt.get("hidden_dim", 64))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        t_zone = torch.tensor(driver_df["t_zone"].values, dtype=torch.float32, device=device)
        t_amb = torch.tensor(driver_df["t_amb"].values, dtype=torch.float32, device=device)
        hour = torch.tensor(driver_df["hour"].values, dtype=torch.float32, device=device)
        day = torch.tensor(driver_df["day"].values, dtype=torch.float32, device=device)
        a0 = torch.tensor(driver_df["a0_raw"].values, dtype=torch.float32, device=device)
        a1 = torch.tensor(driver_df["a1_raw"].values, dtype=torch.float32, device=device)
        t_next, p_total = model(t_zone, t_amb, hour, day, a0, a1)
    return t_next.cpu().numpy(), p_total.cpu().numpy()


def generate_v2(
    n_steps: int = 1200,
    seed: int = 42,
    output_dir: str = "outputs/synthetic_surrogate_v2",
    policy: str | None = "mixed",
    season: str | None = None,
    driver_csv: str = "data/surrogate_v2/boptest_v2_tsupply.csv",
    model_path: str = "outputs/surrogate_v2/rc_node_v3_tsupply.pt",
    c_zon_true: float = 4.2e5,
    surrogate_czon_ref: float = 5.3e5,
    bias_t: float = 0.5,
    noise_t_std: float = 0.08,
    power_scale: float = 1.04,
    power_bias_w: float = 35.0,
    power_noise_rel: float = 0.015,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = SyntheticBenchmarkParamsV2(
        surrogate_czon_ref=surrogate_czon_ref,
        c_zon_true=c_zon_true,
        bias_t=bias_t,
        noise_t_std=noise_t_std,
        power_scale=power_scale,
        power_bias_w=power_bias_w,
        power_noise_rel=power_noise_rel,
    )

    driver_df = _load_driver_slice(
        csv_path=driver_csv,
        n_steps=n_steps,
        policy=policy,
        season=season,
        seed=seed,
    )
    t_next_surr, p_surr = _surrogate_forward(driver_df, model_path, device)

    t_zone = driver_df["t_zone"].to_numpy(dtype=float)
    delta_surr = t_next_surr - t_zone
    t_next_syn = t_zone + params.scale_dT_true * delta_surr + params.bias_t
    if noise_t_std > 0:
        t_next_syn = t_next_syn + rng.normal(0.0, noise_t_std, size=len(t_next_syn))
    t_next_syn = np.clip(t_next_syn, 15.0, 35.0)

    p_syn = params.power_scale * p_surr + params.power_bias_w
    if power_noise_rel > 0:
        p_syn = p_syn * (1.0 + rng.normal(0.0, power_noise_rel, size=len(p_syn)))
    p_syn = np.clip(p_syn, 0.0, None)

    synthetic_df = driver_df.copy()
    synthetic_df["t_zone_next"] = np.round(t_next_syn, 4)
    synthetic_df["delta_t"] = np.round(t_next_syn - t_zone, 4)
    synthetic_df["p_total"] = np.round(p_syn, 2)
    synthetic_df["t_zone_next_driver"] = np.round(driver_df["t_zone_next"].to_numpy(dtype=float), 4)
    synthetic_df["t_zone_next_surrogate"] = np.round(t_next_surr, 4)
    synthetic_df["p_driver"] = np.round(driver_df["p_total"].to_numpy(dtype=float), 2)
    synthetic_df["p_surrogate"] = np.round(p_surr, 2)
    synthetic_df["scale_dT_true"] = params.scale_dT_true
    synthetic_df["bias_t_true"] = params.bias_t
    synthetic_df["power_scale_true"] = params.power_scale
    synthetic_df["power_bias_true_w"] = params.power_bias_w
    synthetic_df["c_zon_true"] = params.c_zon_true
    synthetic_df["surrogate_czon_ref"] = params.surrogate_czon_ref

    suffix_policy = policy or "all"
    suffix_season = season or "all"
    out_csv = out_dir / f"synthetic_v2_driver_{suffix_policy}_{suffix_season}_{n_steps}.csv"
    synthetic_df.to_csv(out_csv, index=False)

    summary = {
        "rows": int(len(synthetic_df)),
        "driver_csv": driver_csv,
        "model_path": model_path,
        "policy_filter": policy,
        "season_filter": season,
        "seed": seed,
        "scale_dT_true": params.scale_dT_true,
        "bias_t_true_c": params.bias_t,
        "power_scale_true": params.power_scale,
        "power_bias_true_w": params.power_bias_w,
        "c_zon_true_j_per_k": params.c_zon_true,
        "surrogate_czon_ref_j_per_k": params.surrogate_czon_ref,
        "t_next_surrogate_rmse_vs_driver_c": float(
            np.sqrt(np.mean((t_next_surr - driver_df["t_zone_next"].to_numpy(dtype=float)) ** 2))
        ),
        "synthetic_delta_mean_c": float(np.mean(t_next_syn - t_zone)),
        "synthetic_delta_std_c": float(np.std(t_next_syn - t_zone)),
    }
    out_json = out_dir / "synthetic_v2_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[SYNTH_V2] driver={driver_csv}")
    print(f"[SYNTH_V2] checkpoint={model_path}")
    print(f"[SYNTH_V2] policy={policy or 'all'} season={season or 'all'} rows={len(synthetic_df)}")
    print(f"[SYNTH_V2] C_zon_true={params.c_zon_true:.3e} J/K, surrogate_ref={params.surrogate_czon_ref:.3e} J/K")
    print(f"[SYNTH_V2] scale_dT_true={params.scale_dT_true:.6f}, bias_t_true={params.bias_t:+.3f} C")
    print(f"[SYNTH_V2] saved_csv={out_csv}")
    print(f"[SYNTH_V2] saved_summary={out_json}")
    return str(out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a stable inverse-calibration benchmark for RCNeuralODEv2 using surrogate-based synthetic distortion."
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", default="mixed")
    parser.add_argument("--season", default=None)
    parser.add_argument("--output_dir", default="outputs/synthetic_surrogate_v2")
    parser.add_argument("--driver_csv", default="data/surrogate_v2/boptest_v2_tsupply.csv")
    parser.add_argument("--model_path", default="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--c_zon_true", type=float, default=4.2e5)
    parser.add_argument("--surrogate_czon_ref", type=float, default=5.3e5)
    parser.add_argument("--bias_t", type=float, default=0.5)
    parser.add_argument("--noise_t_std", type=float, default=0.08)
    parser.add_argument("--power_scale", type=float, default=1.04)
    parser.add_argument("--power_bias_w", type=float, default=35.0)
    parser.add_argument("--power_noise_rel", type=float, default=0.015)
    args = parser.parse_args()

    generate_v2(
        n_steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        policy=args.policy,
        season=args.season,
        driver_csv=args.driver_csv,
        model_path=args.model_path,
        c_zon_true=args.c_zon_true,
        surrogate_czon_ref=args.surrogate_czon_ref,
        bias_t=args.bias_t,
        noise_t_std=args.noise_t_std,
        power_scale=args.power_scale,
        power_bias_w=args.power_bias_w,
        power_noise_rel=args.power_noise_rel,
    )


if __name__ == "__main__":
    main()
