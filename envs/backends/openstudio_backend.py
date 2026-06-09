from __future__ import annotations
from typing import Any, Dict

from envs.base_env import HVACBaseEnv


class OpenStudioBackend(HVACBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raise NotImplementedError("OpenStudio backend is not implemented yet.")
