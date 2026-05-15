"""
Main training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --env-config configs/env_config.yaml \
                            --train-config configs/train_config.yaml \
                            --run-name my_experiment
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs import make_env
from utils.callbacks import CoverageLogCallback, make_eval_callback


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_single_env(env_cfg, backend, seed):
    def _init():
        return make_env(env_cfg, backend=backend)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    train_cfg = load_config(args.train_config)
    log_cfg = train_cfg["logging"]
    ppo_cfg = train_cfg["ppo"]
    backend = train_cfg["backend"]

    device = train_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    os.makedirs(log_cfg["log_dir"], exist_ok=True)
    os.makedirs(log_cfg["model_dir"], exist_ok=True)
    run_dir = os.path.join(log_cfg["log_dir"], args.run_name)

    n_envs = train_cfg["n_envs"]
    vec_env = make_vec_env(
        make_single_env(env_cfg, backend, seed=0),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = make_vec_env(
        make_single_env(env_cfg, backend, seed=999),
        n_envs=1,
    )

    policy_kwargs = dict(
        net_arch=ppo_cfg["policy_kwargs"]["net_arch"],
        activation_fn=torch.nn.Tanh
        if ppo_cfg["policy_kwargs"]["activation_fn"] == "tanh"
        else torch.nn.ReLU,
    )

    model = PPO(
        policy=ppo_cfg["policy"],
        env=vec_env,
        device=device,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=run_dir,
        verbose=1,
    )

    callbacks = [
        CoverageLogCallback(),
        make_eval_callback(eval_env, train_cfg),
    ]

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    model_path = os.path.join(log_cfg["model_dir"], f"{args.run_name}_final")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
