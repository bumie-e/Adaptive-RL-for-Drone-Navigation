"""
RiverDroneEnv — PyBullet backend.

Reward structure:
  - Distance shaping  : 1/(1 + d/scale) in (0,1] toward nearest unvisited patch
  - Cluster visit     : +3.0  (first time drone enters a patch)
  - End of river      : +5.0  (episode terminates)
  - Obstacle hit      : -0.5  (with short cooldown to avoid per-step double-counting)
  - Step penalty      : -0.001
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from envs.pybullet.waste_patch_manager import WastePatchManager
from envs.pybullet.obstacle_manager import ObstacleManager
from envs.base import DroneNavEnvBase

_CTRL_FREQ = 48
_COLLISION_COOLDOWN_STEPS = 10   # steps before the same collision can penalise again


class RiverDroneEnv(BaseRLAviary, DroneNavEnvBase):

    def __init__(self, config: dict, render: bool = False, seed: int = 0):
        self._config = config
        self._ep_cfg = config["episode"]
        self._drone_cfg = config["drone"]
        self._river_cfg = config["river"]
        self._obs_cfg = config["observation"]

        self._num_patches_obs = self._obs_cfg["num_nearest_patches"]
        self._visit_radius = self._obs_cfg["visit_radius"]
        self._dist_scale = self._obs_cfg["distance_reward_scale"]
        self._river_end_threshold = 3.0    # metres from river end counts as "reached"
        self._max_steps = self._ep_cfg["max_steps"]

        self._rng = np.random.default_rng(seed)
        self._patch_manager = WastePatchManager(config, self._rng)
        self._obstacle_manager = ObstacleManager(config, self._rng)
        self._step_count = 0
        self._collision_cooldown = 0
        self._reached_end = False

        # generate before super().__init__ so _addObstacles can use them
        self._patch_manager.generate()
        self._obstacle_manager.generate()

        init_pos = np.array([[
            self._drone_cfg["init_x"],
            self._drone_cfg["init_y"],
            self._drone_cfg["init_z"],
        ]])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=init_pos,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=_CTRL_FREQ,
            gui=render,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

    # ------------------------------------------------------------------
    # Observation & Action spaces
    # BaseAviary calls these methods in __init__ and assigns the results
    # to self.observation_space / self.action_space directly, so they
    # must be methods, not properties.
    # ------------------------------------------------------------------

    def _observationSpace(self):
        obs_dim = DroneNavEnvBase.obs_dim(self._num_patches_obs)
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._patch_manager.remove_from_pybullet(self.CLIENT)
        self._obstacle_manager.remove_from_pybullet(self.CLIENT)

        self._patch_manager.generate()
        self._obstacle_manager.generate()

        self._step_count = 0
        self._collision_cooldown = 0
        self._reached_end = False

        obs, info = super().reset(seed=seed, options=options)
        self._reset_camera()
        return self._computeObs(), info

    def step(self, action):
        self._step_count += 1
        if self._collision_cooldown > 0:
            self._collision_cooldown -= 1

        self._obstacle_manager.update_birds(dt=1.0 / _CTRL_FREQ)

        # BaseRLAviary._preprocessAction indexes action as action[drone_idx, :]
        # so it must be 2D (NUM_DRONES, 4) even though our external space is flat (4,)
        obs, _, terminated, truncated, info = super().step(
            action.reshape(self.NUM_DRONES, 4)
        )
        self._follow_camera()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        return self._computeObs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _addObstacles(self):
        self._load_river_terrain()
        self._patch_manager.load_into_pybullet(self.CLIENT)
        self._obstacle_manager.load_into_pybullet(self.CLIENT)

    def _load_river_terrain(self):
        river = self._river_cfg
        length = river["length"]
        width = river["width"]
        bank_w = river["bank_width"]

        water_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[width / 2, length / 2, 0.01],
            rgbaColor=[0.1, 0.4, 0.8, 0.7],
            physicsClientId=self.CLIENT,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=water_vis,
            basePosition=[0.0, length / 2, 0.0],
            physicsClientId=self.CLIENT,
        )

        for side in [-1, 1]:
            bank_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[bank_w / 2, length / 2, 0.02],
                rgbaColor=[0.2, 0.6, 0.2, 1.0],
                physicsClientId=self.CLIENT,
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=bank_vis,
                basePosition=[side * (width / 2 + bank_w / 2), length / 2, 0.0],
                physicsClientId=self.CLIENT,
            )

    # ------------------------------------------------------------------
    # RL components
    # ------------------------------------------------------------------

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        kin = np.concatenate([
            state[0:3],    # pos
            state[10:13],  # vel
            state[7:10],   # rpy
            state[13:16],  # ang_vel
        ]).astype(np.float32)
        patch_obs = self._patch_manager.get_nearest_obs(state[0:3], self._num_patches_obs)
        return np.concatenate([kin, patch_obs])

    def _computeReward(self) -> float:
        state = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        drone_xy = state[0:2]
        reward = self._ep_cfg["step_penalty"]   # -0.001 per step

        # --- distance shaping toward nearest unvisited patch ----------------
        d = self._patch_manager.get_nearest_unvisited_distance(drone_xy)
        if d is not None:
            reward += 1.0 / (1.0 + d / self._dist_scale)   # (0, 1]
        else:
            # all patches visited — shape toward river end instead
            dist_to_end = max(0.0, self._river_cfg["length"] - drone_pos[1])
            reward += 1.0 / (1.0 + dist_to_end / self._dist_scale)

        # --- cluster visit reward: +3 per newly visited patch ---------------
        reward += self._patch_manager.check_and_visit(drone_xy, self._visit_radius)

        # --- end-of-river reward: +5 (one-shot) ----------------------------
        if not self._reached_end and self._is_at_river_end(drone_pos):
            reward += self._ep_cfg["end_of_river_reward"]
            self._reached_end = True

        # --- obstacle collision penalty: -0.5 (cooldown-gated) -------------
        if self._collision_cooldown == 0:
            if self._obstacle_manager.any_collision(self.DRONE_IDS[0]):
                reward += self._ep_cfg["obstacle_penalty"]
                self._collision_cooldown = _COLLISION_COOLDOWN_STEPS

        return float(reward)

    def _computeTerminated(self) -> bool:
        return self._reached_end

    def _computeTruncated(self) -> bool:
        state = self._getDroneStateVector(0)
        return (
            self._step_count >= self._max_steps
            or self._is_out_of_bounds(state[0:3])
        )

    def _computeInfo(self) -> dict:
        return {
            "coverage": self._patch_manager.coverage_fraction(),
            "step": self._step_count,
            "reached_end": self._reached_end,
        }

    # ------------------------------------------------------------------
    # Camera helpers (GUI only — no-ops when gui=False)
    # ------------------------------------------------------------------

    def _reset_camera(self):
        if not self.GUI:
            return
        p.resetDebugVisualizerCamera(
            cameraDistance=8,
            cameraYaw=30,
            cameraPitch=-25,
            cameraTargetPosition=[
                self._drone_cfg["init_x"],
                self._drone_cfg["init_y"],
                self._drone_cfg["init_z"],
            ],
            physicsClientId=self.CLIENT,
        )

    def _follow_camera(self):
        if not self.GUI:
            return
        state = self._getDroneStateVector(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=8,
            cameraYaw=30,
            cameraPitch=-25,
            cameraTargetPosition=[state[0], state[1], state[2]],
            physicsClientId=self.CLIENT,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_at_river_end(self, pos: np.ndarray) -> bool:
        return pos[1] >= self._river_cfg["length"] - self._river_end_threshold

    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        half_w = self._river_cfg["width"] / 2
        return (
            abs(pos[0]) > half_w
            or pos[1] < 0
            or pos[2] < 0.3
            or pos[2] > 30.0
        )
