from typing import Any, Tuple, Optional, Dict
import gymnasium as gym


class HVACBaseEnv(gym.Env):
    """
    
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Any, Dict]:
        super().reset(seed=seed)
        raise NotImplementedError

    def step(self, action: Any):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    def render(self):
        return None

    def close(self):
        return None