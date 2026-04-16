from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory
from training.train_ppo import MORLLogCallback, build_ppo, maybe_save_model


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class ObjectiveStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.count = 0
        self.sum_comfort = 0.0
        self.sum_energy = 0.0
        self.sum_safety = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos and len(infos) > 0:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            rv = info0.get("reward_vector") if isinstance(info0, dict) else None
            if isinstance(rv, dict):
                self.count += 1
                self.sum_comfort += float(rv.get("comfort", 0.0))
                self.sum_energy += float(rv.get("energy", 0.0))
                self.sum_safety += float(rv.get("safety", 0.0))
        return True

    def means(self) -> np.ndarray:
        denom = max(self.count, 1)
        return np.array(
            [
                self.sum_comfort / denom,
                self.sum_energy / denom,
                self.sum_safety / denom,
            ],
            dtype=np.float64,
        )


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def eram_weight_update(values: np.ndarray, prev_weights: np.ndarray, tau_w: float, adv_lr: float) -> np.ndarray:
    eps = 1e-8
    beta = 1.0 / (adv_lr * tau_w + 1.0)
    logits = -((1.0 + beta) / tau_w) * values + beta * np.log(np.clip(prev_weights, eps, None))
    return softmax(logits)


def parse_weights(text: str | None) -> np.ndarray:
    if not text:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("Weights must have three comma-separated values: comfort,energy,safety")
    arr = np.array(parts, dtype=np.float64)
    if np.any(arr < 0.0) or arr.sum() <= 0.0:
        raise ValueError("Weights must be non-negative and not all zero.")
    return arr / arr.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train true vector-reward MORL on the surrogate with ERAM-style adversarial weight updates.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing env.yaml, agent.yaml, train.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--chunk_steps", type=int, default=100_000)
    parser.add_argument("--tau_w", type=float, default=0.35)
    parser.add_argument("--adv_lr", type=float, default=1.0)
    parser.add_argument("--init_weights", default="0.34,0.33,0.33")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--surrogate-kind",
        choices=["legacy_v3", "v35_raw", "v35_calibrated"],
        default=None,
        help="Select which direct-TSup surrogate implementation to use.",
    )
    parser.add_argument("--surrogate-path", default=None, help="Legacy v3 base model path.")
    parser.add_argument("--surrogate-summary-json", default=None, help="Calibration summary JSON for v3.5 adapters.")
    parser.add_argument("--surrogate-checkpoint", default=None, help="Explicit v3.5 staged checkpoint path.")
    parser.add_argument("--surrogate-base-model", default=None, help="Explicit base v3 TSup checkpoint for v3.5 adapters.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    cfg = load_all_configs(args.config_dir)
    env_cfg = dict(cfg["env"])
    agent_cfg = dict(cfg["agent"])
    train_cfg = dict(cfg["train"])

    env_cfg["backend"] = "surrogate"
    env_cfg["control_mode"] = "tsup_direct"
    if args.surrogate_kind is not None:
        env_cfg["surrogate_kind"] = args.surrogate_kind
    if args.surrogate_path is not None:
        env_cfg["surrogate_path"] = args.surrogate_path
    if args.surrogate_summary_json is not None:
        env_cfg["surrogate_summary_json"] = args.surrogate_summary_json
    if args.surrogate_checkpoint is not None:
        env_cfg["surrogate_checkpoint"] = args.surrogate_checkpoint
    if args.surrogate_base_model is not None:
        env_cfg["surrogate_base_model"] = args.surrogate_base_model

    out_dir = Path(args.out_dir or f"outputs/morl_eram_seed{args.seed}")
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_cfg["seed"] = args.seed
    train_cfg["output_dir"] = str(out_dir)
    train_cfg["save_model"] = True
    train_cfg["save_path"] = str(models_dir)
    train_cfg["morl_csv_name"] = "morl_eram_log.csv"

    raw_env = EnvFactory.create(env_cfg)
    model = build_ppo(raw_env, agent_cfg)

    weights = parse_weights(args.init_weights)
    history_path = out_dir / "eram_weight_history.csv"
    snapshot = {
        "seed": args.seed,
        "iterations": args.iterations,
        "chunk_steps": args.chunk_steps,
        "tau_w": args.tau_w,
        "adv_lr": args.adv_lr,
        "init_weights": weights.tolist(),
        "env": env_cfg,
        "agent": agent_cfg,
        "train": train_cfg,
        "config_dir": str(Path(args.config_dir).resolve()),
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print("=" * 88)
    print("TRUE MORL PRETRAIN ON SURROGATE WITH ERAM-STYLE ADVERSARY")
    print("=" * 88)
    print(f"Seed:         {args.seed}")
    print(f"Iterations:   {args.iterations}")
    print(f"Chunk steps:  {args.chunk_steps:,}")
    print(f"tau_w:        {args.tau_w}")
    print(f"adv_lr:       {args.adv_lr}")
    print(f"Init weights: {weights.round(4).tolist()}")
    print(f"Output dir:   {out_dir}")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "chunk_steps",
                "mean_comfort",
                "mean_energy",
                "mean_safety",
                "w_comfort_before",
                "w_energy_before",
                "w_safety_before",
                "w_comfort_after",
                "w_energy_after",
                "w_safety_after",
            ],
        )
        writer.writeheader()

        for iteration in range(args.iterations):
            raw_env.set_objective_weights(weights[0], weights[1], weights[2])
            stats_cb = ObjectiveStatsCallback()
            csv_cb = MORLLogCallback(str(out_dir), f"morl_eram_iter{iteration:02d}.csv")
            callbacks = CallbackList([stats_cb, csv_cb])

            model.learn(
                total_timesteps=args.chunk_steps,
                callback=callbacks,
                reset_num_timesteps=(iteration == 0),
            )

            values = stats_cb.means()
            new_weights = eram_weight_update(values, weights, args.tau_w, args.adv_lr)

            writer.writerow(
                {
                    "iteration": iteration,
                    "chunk_steps": args.chunk_steps,
                    "mean_comfort": float(values[0]),
                    "mean_energy": float(values[1]),
                    "mean_safety": float(values[2]),
                    "w_comfort_before": float(weights[0]),
                    "w_energy_before": float(weights[1]),
                    "w_safety_before": float(weights[2]),
                    "w_comfort_after": float(new_weights[0]),
                    "w_energy_after": float(new_weights[1]),
                    "w_safety_after": float(new_weights[2]),
                }
            )
            f.flush()

            ckpt_path = models_dir / f"ppo_eram_iter{iteration:02d}.zip"
            model.save(str(ckpt_path))

            print(
                f"[Iter {iteration:02d}] "
                f"mean=[c {values[0]:.4f}, e {values[1]:.4f}, s {values[2]:.4f}] "
                f"weights {weights.round(4).tolist()} -> {new_weights.round(4).tolist()}"
            )
            weights = new_weights

    raw_env.set_objective_weights(weights[0], weights[1], weights[2])
    final_path = maybe_save_model(
        model,
        {
            "save_model": True,
            "output_dir": str(out_dir),
            "save_path": str(models_dir),
        },
    )
    (out_dir / "final_eram_weights.json").write_text(
        json.dumps(
            {
                "w_comfort": float(weights[0]),
                "w_energy": float(weights[1]),
                "w_safety": float(weights[2]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        raw_env.close()
    except Exception:
        pass

    print("=" * 88)
    print("ERAM SURROGATE TRAINING COMPLETE")
    print("=" * 88)
    print(f"Final weights: {weights.round(6).tolist()}")
    if final_path:
        print(f"Final model:   {final_path}")
    print(f"History CSV:   {history_path}")


if __name__ == "__main__":
    main()
