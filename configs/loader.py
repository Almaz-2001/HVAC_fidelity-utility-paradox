from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data

def load_all_configs(base_dir: str | Path = "configs") -> Dict[str, Dict[str, Any]]:
    base_dir = Path(base_dir)
    env_cfg = load_yaml(base_dir / "env.yaml")
    agent_cfg = load_yaml(base_dir / "agent.yaml")
    train_cfg = load_yaml(base_dir / "train.yaml")
    return {"env": env_cfg, "agent": agent_cfg, "train": train_cfg}
