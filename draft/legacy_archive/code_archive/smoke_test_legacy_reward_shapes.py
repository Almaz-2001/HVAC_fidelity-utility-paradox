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
DEFAULT_BASE_OUTPUT = "outputs/legacy_sinergym_reward_sweep"
DEFAULT_SHAPES = "linear,exponential,gaussian,cubic"
DEFAULT_WEIGHTS = "0.85:0.15,0.90:0.10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated legacy Sinergym smoke-tests for multiple reward shapes and comfort-energy weights."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--baseline-steps", type=int, default=500)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--target-temp", type=float, default=None)
    return parser.parse_args()


def _parse_shapes(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_weights(raw: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        comfort_raw, energy_raw = item.split(":", 1)
        pairs.append((float(comfort_raw), float(energy_raw)))
    return pairs


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _build_cfg(
    base_cfg: dict,
    out_dir: Path,
    reward_shape: str,
    w_comfort: float,
    w_energy: float,
    target_temp: float | None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["_config_dir"] = base_cfg.get("_config_dir", DEFAULT_CONFIG_DIR)
    cfg["env"]["output_dir"] = str(out_dir)
    morl_cfg = cfg["env"]["morl"]
    morl_cfg["reward_shape"] = reward_shape
    morl_cfg["w_comfort"] = float(w_comfort)
    morl_cfg["w_energy"] = float(w_energy)
    if target_temp is not None:
        morl_cfg["target_temp"] = float(target_temp)
    return cfg


def _write_summary(base_dir: Path, reward_shape: str, w_comfort: float, w_energy: float) -> dict[str, float | str]:
    metrics_df, _ = collect_default_policy_metrics(base_dir)
    row: dict[str, float | str] = {
        "reward_shape": reward_shape,
        "w_comfort": float(w_comfort),
        "w_energy": float(w_energy),
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
    shapes = _parse_shapes(args.shapes)
    weights = _parse_weights(args.weights)

    base_cfg = load_all_configs(args.config_dir)
    base_cfg["_config_dir"] = args.config_dir

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for reward_shape in shapes:
        for w_comfort, w_energy in weights:
            run_name = f"shape_{reward_shape}_wc_{_slug(w_comfort)}_we_{_slug(w_energy)}"
            out_dir = base_output_dir / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = _build_cfg(
                base_cfg=base_cfg,
                out_dir=out_dir,
                reward_shape=reward_shape,
                w_comfort=w_comfort,
                w_energy=w_energy,
                target_temp=args.target_temp,
            )

            print("\n" + "=" * 76)
            print(
                f"LEGACY REWARD-SHAPE SMOKE TEST | shape={reward_shape} "
                f"| w_comfort={w_comfort:.2f} | w_energy={w_energy:.2f}"
            )
            print(f"Output dir: {out_dir}")
            print("=" * 76)

            run_one(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps, run_baselines_flag=False)
            run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
            run_baselines_only(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps)

            summary_rows.append(_write_summary(out_dir, reward_shape, w_comfort, w_energy))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / "reward_shape_smoke_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
