from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from legacy_sinergym_main import run_baselines_only, run_eval, run_one
from evaluation.visualize_results_live_sinergym import collect_default_policy_metrics, summarize_metrics


DEFAULT_CONFIG_DIR = "configs/legacy_sinergym"
DEFAULT_BASE_OUTPUT = "outputs/legacy_sinergym_gaussian_sweep"
DEFAULT_WEIGHTS = "0.90:0.10,0.92:0.08,0.95:0.05,0.97:0.03"
DEFAULT_SIGMAS = "1.0,1.25,1.5"
DEFAULT_OFFSETS = "0.10,0.15,0.20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated legacy Sinergym smoke-tests for gaussian comfort reward variants."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS)
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS)
    parser.add_argument("--peak", type=float, default=1.0)
    parser.add_argument("--target-temp", type=float, default=22.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--baseline-steps", type=int, default=500)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT)
    return parser.parse_args()


def _parse_weights(raw: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        comfort_raw, energy_raw = item.split(":", 1)
        pairs.append((float(comfort_raw), float(energy_raw)))
    return pairs


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _build_cfg(
    base_cfg: dict,
    out_dir: Path,
    w_comfort: float,
    w_energy: float,
    sigma: float,
    offset: float,
    peak: float,
    target_temp: float,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["_config_dir"] = base_cfg.get("_config_dir", DEFAULT_CONFIG_DIR)
    cfg["env"]["output_dir"] = str(out_dir)
    morl_cfg = cfg["env"]["morl"]
    morl_cfg["reward_shape"] = "gaussian"
    morl_cfg["w_comfort"] = float(w_comfort)
    morl_cfg["w_energy"] = float(w_energy)
    morl_cfg["gaussian_sigma"] = float(sigma)
    morl_cfg["gaussian_offset"] = float(offset)
    morl_cfg["gaussian_peak"] = float(peak)
    morl_cfg["target_temp"] = float(target_temp)
    return cfg


def _write_summary(
    base_dir: Path,
    w_comfort: float,
    w_energy: float,
    sigma: float,
    offset: float,
    peak: float,
    target_temp: float,
) -> dict[str, float | str]:
    metrics_df, _ = collect_default_policy_metrics(base_dir)
    row: dict[str, float | str] = {
        "reward_shape": "gaussian",
        "w_comfort": float(w_comfort),
        "w_energy": float(w_energy),
        "gaussian_sigma": float(sigma),
        "gaussian_offset": float(offset),
        "gaussian_peak": float(peak),
        "target_temp": float(target_temp),
    }
    if metrics_df.empty:
        return row

    summary_df = summarize_metrics(metrics_df, "policy")
    summary_path = base_dir / "live_figures" / "live_baseline_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    for policy in ("ppo", "rule_based", "random", "zero_hold"):
        sub = summary_df[summary_df["policy"] == policy]
        if sub.empty:
            continue
        row[f"{policy}_power"] = float(sub["mean_hvac_power_w_mean"].iloc[0])
        row[f"{policy}_comfort"] = float(sub["mean_comfort_penalty_mean"].iloc[0])
        row[f"{policy}_temp"] = float(sub["mean_zone_temp_c_mean"].iloc[0])
        row[f"{policy}_in_band_pct"] = float(sub["comfort_in_band_pct_mean"].iloc[0])
    return row


def main() -> None:
    args = parse_args()
    weights = _parse_weights(args.weights)
    sigmas = _parse_floats(args.sigmas)
    offsets = _parse_floats(args.offsets)

    base_cfg = load_all_configs(args.config_dir)
    base_cfg["_config_dir"] = args.config_dir

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for w_comfort, w_energy in weights:
        for sigma in sigmas:
            for offset in offsets:
                run_name = (
                    f"gaussian_wc_{_slug(w_comfort)}_we_{_slug(w_energy)}"
                    f"_sig_{_slug(sigma)}_off_{_slug(offset)}"
                )
                out_dir = base_output_dir / run_name
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_cfg(
                    base_cfg=base_cfg,
                    out_dir=out_dir,
                    w_comfort=w_comfort,
                    w_energy=w_energy,
                    sigma=sigma,
                    offset=offset,
                    peak=args.peak,
                    target_temp=args.target_temp,
                )

                print("\n" + "=" * 84)
                print(
                    f"LEGACY GAUSSIAN SMOKE TEST | wc={w_comfort:.2f} | we={w_energy:.2f} "
                    f"| sigma={sigma:.2f} | offset={offset:.2f} | peak={args.peak:.2f}"
                )
                print(f"Output dir: {out_dir}")
                print("=" * 84)

                run_one(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps, run_baselines_flag=False)
                run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
                run_baselines_only(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps)

                summary_rows.append(
                    _write_summary(
                        base_dir=out_dir,
                        w_comfort=w_comfort,
                        w_energy=w_energy,
                        sigma=sigma,
                        offset=offset,
                        peak=args.peak,
                        target_temp=args.target_temp,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / "gaussian_smoke_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
