import os
from stable_baselines3.common.utils import set_random_seed
from envs.factory import EnvFactory
from gymnasium.wrappers import TimeLimit

def make_surrogate_env(env_id, rank, seed=0):
    def _init():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_surrogate_path = os.path.join(project_root, "outputs", "surrogate_v2", "rc_node_v2_best.pt")
        
        config = {
            "backend": "surrogate",
            "surrogate_path": local_surrogate_path
        }
        
        env = EnvFactory.create(config)
        env = TimeLimit(env, max_episode_steps=336)
        
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    
    set_random_seed(seed)
    return _init