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
DEFAULT_BASE_OUTPUT = "outputs/legacy_sinergym_bc_intensity_sweep"
DEFAULT_WARMSTART_STEPS = "10000,20000"
DEFAULT_WARMSTART_EPOCHS = "20,40"
DEFAULT_PPO_LRS = "0.00015,0.0001"
DEFAULT_BC_PAIRS = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated legacy Sinergym smoke-tests for BC intensity and PPO LR using the current gaussian reward baseline."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--bc-pairs", default=DEFAULT_BC_PAIRS)
    parser.add_argument("--warmstart-steps-grid", default=DEFAULT_WARMSTART_STEPS)
    parser.add_argument("--warmstart-epochs-grid", default=DEFAULT_WARMSTART_EPOCHS)
    parser.add_argument("--ppo-lr-grid", default=DEFAULT_PPO_LRS)
    parser.add_argument("--warmstart-batch-size", type=int, default=256)
    parser.add_argument("--warmstart-lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--baseline-steps", type=int, default=500)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT)
    return parser.parse_args()


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_bc_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        steps_raw, epochs_raw = item.split(":", 1)
        pairs.append((int(steps_raw), int(epochs_raw)))
    return pairs


def _slug(value: float | int) -> str:
    return str(value).replace(".", "p")


def _build_cfg(
    base_cfg: dict,
    out_dir: Path,
    *,
    warmstart_steps: int,
    warmstart_epochs: int,
    warmstart_batch_size: int,
    warmstart_lr: float,
    ppo_learning_rate: float,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["_config_dir"] = base_cfg.get("_config_dir", DEFAULT_CONFIG_DIR)

    cfg["env"]["output_dir"] = str(out_dir)

    train_cfg = cfg["train"]
    warm_cfg = dict(train_cfg.get("bc_warmstart", {}) or {})
    warm_cfg["enabled"] = True
    warm_cfg["steps"] = int(warmstart_steps)
    warm_cfg["epochs"] = int(warmstart_epochs)
    warm_cfg["batch_size"] = int(warmstart_batch_size)
    warm_cfg["learning_rate"] = float(warmstart_lr)
    train_cfg["bc_warmstart"] = warm_cfg
    cfg["train"] = train_cfg

    agent_cfg = cfg["agent"]
    ppo_cfg = dict(agent_cfg.get("ppo", {}) or {})
    ppo_cfg["learning_rate"] = float(ppo_learning_rate)
    agent_cfg["ppo"] = ppo_cfg
    cfg["agent"] = agent_cfg
    return cfg


def _write_summary(
    base_dir: Path,
    *,
    warmstart_steps: int,
    warmstart_epochs: int,
    warmstart_batch_size: int,
    warmstart_lr: float,
    ppo_learning_rate: float,
) -> dict[str, float | str]:
    metrics_df, _ = collect_default_policy_metrics(base_dir)
    row: dict[str, float | str] = {
        "reward_shape": "gaussian",
        "warmstart_steps": int(warmstart_steps),
        "warmstart_epochs": int(warmstart_epochs),
        "warmstart_batch_size": int(warmstart_batch_size),
        "warmstart_lr": float(warmstart_lr),
        "ppo_learning_rate": float(ppo_learning_rate),
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
    bc_pairs = _parse_bc_pairs(args.bc_pairs) if args.bc_pairs else []
    warmstart_steps_grid = _parse_ints(args.warmstart_steps_grid)
    warmstart_epochs_grid = _parse_ints(args.warmstart_epochs_grid)
    ppo_lr_grid = _parse_floats(args.ppo_lr_grid)

    base_cfg = load_all_configs(args.config_dir)
    base_cfg["_config_dir"] = args.config_dir

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if bc_pairs:
        bc_grid = bc_pairs
    else:
        bc_grid = [
            (warmstart_steps, warmstart_epochs)
            for warmstart_steps in warmstart_steps_grid
            for warmstart_epochs in warmstart_epochs_grid
        ]

    summary_rows = []
    for warmstart_steps, warmstart_epochs in bc_grid:
        for ppo_learning_rate in ppo_lr_grid:
                run_name = (
                    f"bc_steps_{_slug(warmstart_steps)}"
                    f"_ep_{_slug(warmstart_epochs)}"
                    f"_ppo_lr_{_slug(ppo_learning_rate)}"
                )
                out_dir = base_output_dir / run_name
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_cfg(
                    base_cfg=base_cfg,
                    out_dir=out_dir,
                    warmstart_steps=warmstart_steps,
                    warmstart_epochs=warmstart_epochs,
                    warmstart_batch_size=args.warmstart_batch_size,
                    warmstart_lr=args.warmstart_lr,
                    ppo_learning_rate=ppo_learning_rate,
                )

                print("\n" + "=" * 88)
                print(
                    "LEGACY BC-INTENSITY SMOKE TEST "
                    f"| bc_steps={warmstart_steps} | bc_epochs={warmstart_epochs} "
                    f"| ppo_lr={ppo_learning_rate:.6f}"
                )
                print(
                    f"Warm-start batch={args.warmstart_batch_size} "
                    f"| warmstart_lr={args.warmstart_lr}"
                )
                print(f"Output dir: {out_dir}")
                print("=" * 88)

                run_one(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps, run_baselines_flag=False)
                run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
                run_baselines_only(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps)

                summary_rows.append(
                    _write_summary(
                        base_dir=out_dir,
                        warmstart_steps=warmstart_steps,
                        warmstart_epochs=warmstart_epochs,
                        warmstart_batch_size=args.warmstart_batch_size,
                        warmstart_lr=args.warmstart_lr,
                        ppo_learning_rate=ppo_learning_rate,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / "bc_intensity_smoke_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
