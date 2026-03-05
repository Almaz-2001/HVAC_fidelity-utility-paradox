from __future__ import annotations

from logging import info
from typing import Dict, Any, Optional
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from training.morl_logger import MORLCSVLogger


class MORLLogCallback(BaseCallback):
    """
    Логирует info['reward_vector'] в CSV на каждом step.
    Работает с DummyVecEnv: infos = list[dict], берём infos[0].
    """

    def __init__(self, out_dir: str = "outputs", filename: str = "morl_log.csv", verbose: int = 0):
        super().__init__(verbose)
        Path(out_dir).mkdir(parents=True, exist_ok=True)  # ✅ чтобы файл точно мог создаться
        self.csv = MORLCSVLogger(out_dir=out_dir, filename=filename)
        self.step_id = 0

    def _extract_scalar_reward(self) -> float:
        """
        Пытаемся достать reward из SB3 callback locals.
        Возвращаем float или np.nan.
        """
        rewards = self.locals.get("rewards", None)
        if rewards is None:
            return float("nan")

        try:
            # часто rewards = np.array([r]) из VecEnv
            if isinstance(rewards, (list, tuple, np.ndarray)):
                return float(rewards[0])
            return float(rewards)
        except Exception:
            return float("nan")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)

        reward_scalar = self._extract_scalar_reward()

        if infos and len(infos) > 0:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            rv = info0.get("reward_vector", None) if isinstance(info0, dict) else None

            if isinstance(rv, dict):
                # fallback: если reward_scalar не получился, считаем сами из MORL-вектора
                if np.isnan(reward_scalar):
                    c = rv.get("comfort")
                    e = rv.get("energy")
                    wc = rv.get("w_comfort")
                    we = rv.get("w_energy")
                    if all(v is not None for v in (c, e, wc, we)):
                        try:
                            reward_scalar = float(wc) * float(c) + float(we) * float(e)
                        except Exception:
                            reward_scalar = float("nan")

                self.csv.log({
                    "step": self.step_id,
                    "reward_scalar": reward_scalar,  # ✅ всегда число (float или nan)
                    "comfort": rv.get("comfort"),
                    "energy": rv.get("energy"),
                    "zone_temp": rv.get("zone_temp"),
                    "hvac_power": rv.get("hvac_power"),
                    "w_comfort": rv.get("w_comfort"),
                    "w_energy": rv.get("w_energy"),
                })

        self.step_id += 1
        return True

    def _on_training_end(self) -> None:
        self.csv.close()
    """
    Логирует info['reward_vector'] в CSV на каждом step.
    Работает с DummyVecEnv: infos = list[dict], берём infos[0].
    """

    def __init__(self, out_dir: str = "outputs", filename: str = "morl_log.csv", verbose: int = 0):
        super().__init__(verbose)
        self.csv = MORLCSVLogger(out_dir=out_dir, filename=filename)
        self.step_id = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)

        # --- 1) Пытаемся взять reward из SB3 ---
        reward_scalar = None
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            try:
                # обычно rewards = np.array([r]) в VecEnv
                reward_scalar = float(rewards[0])
            except Exception:
                reward_scalar = float(rewards)

        # --- 2) Берём MORL reward_vector из info и делаем fallback ---
        if infos and len(infos) > 0:
            info0 = infos[0]
            rv = info0.get("reward_vector", None)

            if isinstance(rv, dict):
                # fallback: если SB3 reward не доступен, считаем сами
                if reward_scalar is None:
                    c = rv.get("comfort")
                    e = rv.get("energy")
                    wc = rv.get("w_comfort")
                    we = rv.get("w_energy")
                    if c is not None and e is not None and wc is not None and we is not None:
                        reward_scalar = float(wc) * float(c) + float(we) * float(e)

                self.csv.log({
                    "step": self.step_id,
                    "reward_scalar": reward_scalar,
                    "comfort": rv.get("comfort"),
                    "energy": rv.get("energy"),
                    "zone_temp": rv.get("zone_temp"),
                    "hvac_power": rv.get("hvac_power"),
                    "w_comfort": rv.get("w_comfort"),
                    "w_energy": rv.get("w_energy"),
                })

        self.step_id += 1
        return True

    def _on_training_end(self) -> None:
        self.csv.close()



def build_ppo(env, agent_cfg: Dict[str, Any]) -> PPO:
    algo = (agent_cfg.get("algorithm") or "PPO").upper()
    if algo != "PPO":
        raise ValueError(f"Unsupported algorithm: {algo}. Only PPO is implemented now.")

    policy = agent_cfg.get("policy", "MlpPolicy")
    device = agent_cfg.get("device", "cpu")
    ppo_params = agent_cfg.get("ppo", {}) or {}

    env = Monitor(env) 



    model = PPO(
        policy=policy,
        env=env,
        device=device,
        learning_rate=ppo_params.get("learning_rate", 3e-4),
        n_steps=ppo_params.get("n_steps", 2048),
        batch_size=ppo_params.get("batch_size", 64),
        n_epochs=ppo_params.get("n_epochs", 10),
        gamma=ppo_params.get("gamma", 0.99),
        gae_lambda=ppo_params.get("gae_lambda", 0.95),
        clip_range=ppo_params.get("clip_range", 0.2),
        ent_coef=ppo_params.get("ent_coef", 0.0),
        vf_coef=ppo_params.get("vf_coef", 0.5),
        verbose=1,
    )
    return model


def train_ppo(model: PPO, train_cfg: Dict[str, Any]) -> None:
    total_timesteps = int(train_cfg.get("total_timesteps", 1000))
    seed = train_cfg.get("seed")

    if seed is not None:
        model.set_random_seed(int(seed))

    enable_morl_csv = bool(train_cfg.get("morl_csv", True))
    
    out_dir = str(train_cfg.get("output_dir", "outputs"))
    filename = str(train_cfg.get("morl_csv_name", "morl_log.csv"))

    callback = MORLLogCallback(out_dir, filename) if enable_morl_csv else None

    model.learn(total_timesteps=total_timesteps, callback=callback,reset_num_timesteps=True)


def maybe_save_model(model: PPO, train_cfg: Dict[str, Any]) -> Optional[str]:
    if not bool(train_cfg.get("save_model", False)):
        return None

    # если save_path не задан — сохраняем рядом с логами
    output_dir = str(train_cfg.get("output_dir", "/app/outputs"))
    default_path = str(Path(output_dir) / "models")
    save_path = str(train_cfg.get("save_path", default_path))

    Path(save_path).mkdir(parents=True, exist_ok=True)

    file_path = str(Path(save_path) / "ppo_model.zip")
    model.save(file_path)
    return file_path

