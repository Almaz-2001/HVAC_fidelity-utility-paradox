from __future__ import annotations
from typing import Dict, Any

from envs.base_env import HVACBaseEnv
from envs.backends.boptest_backend import BOPTESTBackend


class EnvFactory:
    @staticmethod
    def create(config: Dict[str, Any]) -> HVACBaseEnv:
        backend = (config.get("backend") or "sinergym").lower()

        if backend == "boptest":
            return BOPTESTBackend(config)

        if backend == "surrogate":
            from envs.backends.surrogate_backend import SurrogateBackend
            return SurrogateBackend(config)

        raise ValueError(
            f"Unknown backend: {backend}."
            f"Available: boptest, surrogate"
        )