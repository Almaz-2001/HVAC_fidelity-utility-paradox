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
DEFAULT_BASE_OUTPUT = "outputs/legacy_sinergym_tempgrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated legacy Sinergym smoke-tests for multiple temp_index candidates."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--indices", default="10,11,12,13")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--baseline-steps", type=int, default=500)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT)
    return parser.parse_args()


def _parse_indices(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _build_cfg(base_cfg: dict, out_dir: Path, temp_index: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["_config_dir"] = base_cfg.get("_config_dir", DEFAULT_CONFIG_DIR)
    cfg["env"]["output_dir"] = str(out_dir)
    cfg["env"]["morl"]["temp_index"] = int(temp_index)
    return cfg


def _write_summary(base_dir: Path, temp_index: int) -> dict[str, float | int]:
    metrics_df, _ = collect_default_policy_metrics(base_dir)
    if metrics_df.empty:
        return {"temp_index": temp_index}

    summary_df = summarize_metrics(metrics_df, "policy")
    summary_path = base_dir / "live_figures" / "live_baseline_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    row = {"temp_index": temp_index}
    for policy in ("ppo", "rule_based", "random", "zero_hold"):
        sub = summary_df[summary_df["policy"] == policy]
        if sub.empty:
            continue
        row[f"{policy}_power"] = float(sub["mean_hvac_power_w_mean"].iloc[0])
        row[f"{policy}_comfort"] = float(sub["mean_comfort_penalty_mean"].iloc[0])
        row[f"{policy}_temp"] = float(sub["mean_zone_temp_c_mean"].iloc[0])
    return row


def main() -> None:
    args = parse_args()
    indices = _parse_indices(args.indices)

    base_cfg = load_all_configs(args.config_dir)
    base_cfg["_config_dir"] = args.config_dir

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for temp_index in indices:
        out_dir = base_output_dir / f"temp_idx_{temp_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = _build_cfg(base_cfg, out_dir, temp_index)

        print("\n" + "=" * 72)
        print(f"LEGACY TEMP-INDEX SMOKE TEST | temp_index={temp_index}")
        print(f"Output dir: {out_dir}")
        print("=" * 72)

        run_one(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps, run_baselines_flag=False)
        run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
        run_baselines_only(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps)

        summary_rows.append(_write_summary(out_dir, temp_index))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / "temp_index_smoke_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
