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
DEFAULT_BASE_OUTPUT = "outputs/legacy_sinergym_exponential_bc_sweep"
DEFAULT_WEIGHTS = "0.70:0.30,0.80:0.20,0.90:0.10"
DEFAULT_EXP_ALPHAS = "1.0,1.5"
DEFAULT_EXP_SCALES = "0.02,0.04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated legacy Sinergym smoke-tests for exponential reward with RBC BC warm-start."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--exp-alphas", default=DEFAULT_EXP_ALPHAS)
    parser.add_argument("--exp-scales", default=DEFAULT_EXP_SCALES)
    parser.add_argument("--target-temp", type=float, default=22.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--baseline-steps", type=int, default=500)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--warmstart-steps", type=int, default=10000)
    parser.add_argument("--warmstart-epochs", type=int, default=20)
    parser.add_argument("--warmstart-batch-size", type=int, default=256)
    parser.add_argument("--warmstart-lr", type=float, default=0.001)
    parser.add_argument("--ppo-learning-rate", type=float, default=0.00015)
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
    *,
    w_comfort: float,
    w_energy: float,
    exp_alpha: float,
    exp_scale: float,
    target_temp: float,
    warmstart_steps: int,
    warmstart_epochs: int,
    warmstart_batch_size: int,
    warmstart_lr: float,
    ppo_learning_rate: float,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["_config_dir"] = base_cfg.get("_config_dir", DEFAULT_CONFIG_DIR)

    cfg["env"]["output_dir"] = str(out_dir)
    morl_cfg = cfg["env"]["morl"]
    morl_cfg["reward_shape"] = "exponential"
    morl_cfg["w_comfort"] = float(w_comfort)
    morl_cfg["w_energy"] = float(w_energy)
    morl_cfg["target_temp"] = float(target_temp)
    morl_cfg["exp_alpha"] = float(exp_alpha)
    morl_cfg["exp_scale"] = float(exp_scale)

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
    w_comfort: float,
    w_energy: float,
    exp_alpha: float,
    exp_scale: float,
    target_temp: float,
    warmstart_steps: int,
    warmstart_epochs: int,
    warmstart_batch_size: int,
    warmstart_lr: float,
    ppo_learning_rate: float,
) -> dict[str, float | str]:
    metrics_df, _ = collect_default_policy_metrics(base_dir)
    row: dict[str, float | str] = {
        "reward_shape": "exponential",
        "w_comfort": float(w_comfort),
        "w_energy": float(w_energy),
        "exp_alpha": float(exp_alpha),
        "exp_scale": float(exp_scale),
        "target_temp": float(target_temp),
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
    weights = _parse_weights(args.weights)
    exp_alphas = _parse_floats(args.exp_alphas)
    exp_scales = _parse_floats(args.exp_scales)

    base_cfg = load_all_configs(args.config_dir)
    base_cfg["_config_dir"] = args.config_dir

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for w_comfort, w_energy in weights:
        for exp_alpha in exp_alphas:
            for exp_scale in exp_scales:
                run_name = (
                    f"exp_bc_wc_{_slug(w_comfort)}_we_{_slug(w_energy)}"
                    f"_alpha_{_slug(exp_alpha)}_scale_{_slug(exp_scale)}"
                )
                out_dir = base_output_dir / run_name
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_cfg(
                    base_cfg=base_cfg,
                    out_dir=out_dir,
                    w_comfort=w_comfort,
                    w_energy=w_energy,
                    exp_alpha=exp_alpha,
                    exp_scale=exp_scale,
                    target_temp=args.target_temp,
                    warmstart_steps=args.warmstart_steps,
                    warmstart_epochs=args.warmstart_epochs,
                    warmstart_batch_size=args.warmstart_batch_size,
                    warmstart_lr=args.warmstart_lr,
                    ppo_learning_rate=args.ppo_learning_rate,
                )

                print("\n" + "=" * 90)
                print(
                    "LEGACY EXPONENTIAL+BC SMOKE TEST "
                    f"| wc={w_comfort:.2f} | we={w_energy:.2f} "
                    f"| exp_alpha={exp_alpha:.2f} | exp_scale={exp_scale:.3f}"
                )
                print(
                    f"Warm-start: steps={args.warmstart_steps} epochs={args.warmstart_epochs} "
                    f"batch={args.warmstart_batch_size} lr={args.warmstart_lr}"
                )
                print(f"PPO lr: {args.ppo_learning_rate}")
                print(f"Output dir: {out_dir}")
                print("=" * 90)

                run_one(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps, run_baselines_flag=False)
                run_eval(seed=args.seed, cfg=cfg, eval_steps=args.eval_steps)
                run_baselines_only(seed=args.seed, cfg=cfg, baseline_steps=args.baseline_steps)

                summary_rows.append(
                    _write_summary(
                        base_dir=out_dir,
                        w_comfort=w_comfort,
                        w_energy=w_energy,
                        exp_alpha=exp_alpha,
                        exp_scale=exp_scale,
                        target_temp=args.target_temp,
                        warmstart_steps=args.warmstart_steps,
                        warmstart_epochs=args.warmstart_epochs,
                        warmstart_batch_size=args.warmstart_batch_size,
                        warmstart_lr=args.warmstart_lr,
                        ppo_learning_rate=args.ppo_learning_rate,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_output_dir / "exponential_bc_smoke_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
