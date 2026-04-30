from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from evaluation.run_baselines import RuleBasedSinergymPolicy


def collect_rbc_dataset(
    env,
    *,
    steps: int,
    seed: int,
    out_dir: str | Path,
    dataset_name: str = "rbc_warmstart_dataset.csv",
) -> tuple[np.ndarray, np.ndarray, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / dataset_name

    obs, info = env.reset(seed=seed)
    info = dict(info or {})
    policy = RuleBasedSinergymPolicy(env)
    policy.reset()

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    csv_rows: list[dict[str, Any]] = []

    for step_idx in range(int(steps)):
        action = np.asarray(policy(obs, info), dtype=np.float32).reshape(-1)
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)

        obs_rows.append(obs_arr.copy())
        action_rows.append(action.copy())
        csv_rows.append(
            {
                "step": step_idx,
                "zone_temp": info.get("zone_temp"),
                "hvac_power": info.get("hvac_power"),
                "a0": float(action[0]) if action.size > 0 else np.nan,
                "a1": float(action[1]) if action.size > 1 else np.nan,
                **{f"obs_{i}": float(value) for i, value in enumerate(obs_arr)},
            }
        )

        obs, _, terminated, truncated, info = env.step(action)
        info = dict(info or {})
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
            info = dict(info or {})
            policy.reset()

    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(action_rows, dtype=np.float32),
        str(out_path),
    )


def behavior_clone_policy(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    out_dir: str | Path,
    seed: int,
) -> dict[str, float | int | str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = model.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)

    dataset_size = int(obs_tensor.shape[0])
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(learning_rate))

    history: list[dict[str, float | int]] = []
    model.policy.train()

    for epoch in range(int(epochs)):
        indices = rng.permutation(dataset_size)
        batch_losses: list[float] = []

        for start in range(0, dataset_size, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            batch_obs = obs_tensor[batch_idx].to(device)
            batch_actions = action_tensor[batch_idx].to(device)

            dist = model.policy.get_distribution(batch_obs)
            pred_actions = dist.distribution.mean
            loss = F.mse_loss(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append({"epoch": epoch + 1, "bc_loss": epoch_loss})

    model.policy.eval()

    history_path = out_dir / "bc_warmstart_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "dataset_size": dataset_size,
        "final_bc_loss": float(history[-1]["bc_loss"]) if history else float("nan"),
        "history_csv": str(history_path),
    }
    summary_path = out_dir / "bc_warmstart_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def maybe_run_rbc_warmstart(
    model: PPO,
    env_factory,
    env_cfg: dict,
    warm_cfg: dict,
    *,
    seed: int,
    out_dir: str | Path,
) -> dict[str, float | int | str] | None:
    enabled = bool(warm_cfg.get("enabled", False))
    if not enabled:
        return None

    out_dir = Path(out_dir) / "bc_warmstart"
    out_dir.mkdir(parents=True, exist_ok=True)

    warm_env = env_factory.create(env_cfg)
    try:
        observations, actions, dataset_csv = collect_rbc_dataset(
            warm_env,
            steps=int(warm_cfg.get("steps", 10000)),
            seed=seed,
            out_dir=out_dir,
            dataset_name=str(warm_cfg.get("dataset_name", "rbc_warmstart_dataset.csv")),
        )
    finally:
        try:
            warm_env.close()
        except Exception:
            pass

    summary = behavior_clone_policy(
        model,
        observations,
        actions,
        epochs=int(warm_cfg.get("epochs", 20)),
        batch_size=int(warm_cfg.get("batch_size", 256)),
        learning_rate=float(warm_cfg.get("learning_rate", 1e-3)),
        out_dir=out_dir,
        seed=seed,
    )
    summary["dataset_csv"] = dataset_csv
    return summary
