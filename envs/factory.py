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

        # Sinergym оставлен как заглушка — раскомментируй когда понадобится
        # if backend == "sinergym":
        #     from envs.backends.sinergym_backend import SinergymBackend
        #     return SinergymBackend(config)

        raise ValueError(
            f"Unknown backend: {backend}. Currently implemented: boptest"
        )