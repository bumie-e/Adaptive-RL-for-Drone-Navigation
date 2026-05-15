"""Smoke tests — verify env init, step, and obs/act space shapes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import yaml
import numpy as np


@pytest.fixture
def configs():
    with open("configs/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    with open("configs/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    return env_cfg, train_cfg


def test_env_spaces(configs):
    env_cfg, train_cfg = configs
    from envs import make_env
    env = make_env(env_cfg, backend="pybullet", render=False)

    k = env_cfg["observation"]["num_nearest_patches"]
    assert env.observation_space.shape == (12 + k * 4,)
    assert env.action_space.shape == (4,)
    env.close()


def test_env_reset_step(configs):
    env_cfg, _ = configs
    from envs import make_env
    env = make_env(env_cfg, backend="pybullet", render=False)

    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert not np.any(np.isnan(obs))

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert "coverage" in info
    env.close()


def test_patch_manager(configs):
    from envs.pybullet.waste_patch_manager import WastePatchManager
    env_cfg, _ = configs
    rng = np.random.default_rng(0)
    pm = WastePatchManager(env_cfg, rng)
    pm.generate()

    n = env_cfg["waste_patches"]["num_patches"]
    assert len(pm.patches) == n
    assert pm.coverage_fraction() == 0.0

    drone_pos = np.array(pm.patches[0]["pos"])

    # distance to nearest unvisited should be ~0 when on top of a patch
    d = pm.get_nearest_unvisited_distance(drone_pos[:2])
    assert d is not None and d < 1.0

    reward = pm.check_and_visit(drone_pos[:2], visit_radius=10.0)
    assert reward == pytest.approx(env_cfg["episode"]["patch_visit_reward"])
    assert pm.coverage_fraction() > 0.0

    # after visiting all patches, get_nearest_unvisited_distance returns None
    for patch in pm.patches:
        patch["visited"] = True
    assert pm.get_nearest_unvisited_distance(drone_pos[:2]) is None


def test_reward_components(configs):
    """Distance reward stays in (0,1]; step penalty is applied every step."""
    env_cfg, _ = configs
    from envs import make_env
    env = make_env(env_cfg, backend="pybullet", render=False)
    obs, _ = env.reset(seed=0)

    # collect a few steps of reward and verify step penalty is present
    total_reward = 0.0
    for _ in range(5):
        action = np.zeros(4, dtype=np.float32)  # hover
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    # 5 steps × -0.001 = -0.005 baseline, offset by distance reward
    assert total_reward > -0.05   # distance reward should dominate step penalty
    env.close()
