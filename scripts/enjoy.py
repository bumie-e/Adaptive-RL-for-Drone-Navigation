"""
Quick visual sanity check — runs the environment with random actions and GUI.
Use this before training to verify the scene looks correct.

Usage:
    python scripts/enjoy.py
    python scripts/enjoy.py --episodes 3 --steps 300
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import numpy as np
from envs import make_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    train_cfg = load_config(args.train_config)
    backend = train_cfg["backend"]

    print(f"Backend : {backend}")
    print(f"River   : {env_cfg['river']['length']}m × {env_cfg['river']['width']}m")
    print(f"Patches : {env_cfg['waste_patches']['num_patches']} in "
          f"{env_cfg['waste_patches']['num_clusters']} clusters")
    print(f"Obstacles: {env_cfg['obstacles']['num_trees']} trees, "
          f"{env_cfg['obstacles']['num_rocks']} rocks, "
          f"{env_cfg['obstacles']['num_rooftops']} rooftops, "
          f"{env_cfg['obstacles']['num_birds']} birds")
    print()

    env = make_env(env_cfg, backend=backend, render=True)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0

        print(f"--- Episode {ep + 1} ---")
        for step in range(args.steps):
            # gentle forward drift so the drone actually moves down the river
            action = np.array([0.0, 0.3, 0.0, 0.0], dtype=np.float32)
            action += env.action_space.sample() * 0.1   # small random jitter

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if step % 100 == 0:
                pos = obs[:3]
                print(f"  step {step:4d} | pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                      f"| reward={reward:+.3f} | coverage={info['coverage']:.0%}")

            if terminated or truncated:
                reason = "reached river end" if info["reached_end"] else "truncated"
                print(f"  Done ({reason}) at step {step}")
                break

            time.sleep(1 / 48)   # match ctrl_freq so GUI isn't too fast

        print(f"  Episode return: {ep_reward:.2f} | "
              f"final coverage: {info['coverage']:.0%}\n")

    env.close()


if __name__ == "__main__":
    main()
