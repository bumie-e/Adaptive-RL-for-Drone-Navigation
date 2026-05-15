import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class CoverageLogCallback(BaseCallback):
    """Logs mean patch coverage from info dicts to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._coverages: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "coverage" in info:
                self._coverages.append(info["coverage"])

        if len(self._coverages) >= 100:
            self.logger.record("env/mean_coverage", np.mean(self._coverages))
            self._coverages = []
        return True


def make_eval_callback(eval_env, cfg: dict) -> EvalCallback:
    log_cfg = cfg["logging"]
    return EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_cfg["model_dir"], "best"),
        log_path=log_cfg["log_dir"],
        eval_freq=log_cfg["eval_freq"],
        n_eval_episodes=log_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
