"""
Shared constants and abstract interface that both the PyBullet and Isaac
backends must satisfy. Keeping this stable means migrating backends only
requires swapping the envs/pybullet/ implementation for envs/isaac/.
"""

from abc import ABC, abstractmethod
import numpy as np


RIVER_AXIS = 1        # Y-axis runs along the river length
CROSS_AXIS = 0        # X-axis is the river width
VERTICAL_AXIS = 2     # Z-axis is altitude


class DroneNavEnvBase(ABC):
    """
    Observation contract (flat numpy array):
        [0:12]   kinematic state  — pos(3) vel(3) rpy(3) ang_vel(3)
        [12:]    K patches × 4   — rel_x, rel_y, distance, is_visited

    Action contract:
        4D velocity target [vx, vy, vz, yaw_rate], normalised to [-1, 1]

    Space contract (gym-pybullet-drones pattern):
        Override _observationSpace() and _actionSpace() as methods,
        NOT as @property — BaseAviary assigns their return values to
        self.observation_space / self.action_space in __init__.
    """

    @abstractmethod
    def reset(self, seed=None):
        ...

    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def render(self):
        ...

    @abstractmethod
    def close(self):
        ...

    @staticmethod
    def obs_dim(num_nearest_patches: int) -> int:
        return 12 + num_nearest_patches * 4

    @staticmethod
    def act_dim() -> int:
        return 4
