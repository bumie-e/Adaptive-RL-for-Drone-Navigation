"""
Evaluate a trained model and optionally render an episode.

Usage:
    python scripts/evaluate.py --model models/run_final --render
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
from stable_baselines3 import PPO

from envs import make_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model (no .zip)")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    train_cfg = load_config(args.train_config)
    backend = train_cfg["backend"]

    env = make_env(env_cfg, backend=backend, render=args.render)
    model = PPO.load(args.model, device="cpu")

    coverages = []
    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        coverages.append(info.get("coverage", 0.0))
        returns.append(ep_return)
        print(f"Episode {ep+1}: return={ep_return:.1f}, coverage={info.get('coverage', 0):.1%}")

    print(f"\nMean return : {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"Mean coverage: {np.mean(coverages):.1%} ± {np.std(coverages):.1%}")
    env.close()


if __name__ == "__main__":
    main()
