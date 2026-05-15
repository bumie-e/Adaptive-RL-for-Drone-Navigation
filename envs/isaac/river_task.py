"""
Isaac Lab backend — placeholder for migration from PyBullet.

Migration checklist:
  1. Inherit from isaaclab.envs.DirectRLEnv (or ManagerBasedRLEnv).
  2. Port _load_river_terrain() using USD prims / PhysX materials.
  3. Port WastePatchManager to GPU tensors (patch positions as torch.Tensor
     on device, visit checks via batched distance ops).
  4. Keep observation_space / action_space shapes identical to RiverDroneEnv
     so the trained SB3 policy loads without modification.
  5. Switch train_config.yaml backend: "isaac" and raise n_envs to 1024+.

The DroneNavEnvBase contract (envs/base.py) is the shared interface — do not
change the obs/act dimensionality without updating both backends.
"""

from envs.base import DroneNavEnvBase


class RiverDroneTask(DroneNavEnvBase):
    def __init__(self, config: dict):
        raise NotImplementedError(
            "Isaac Lab backend is not yet implemented. "
            "Set backend='pybullet' in train_config.yaml."
        )

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
