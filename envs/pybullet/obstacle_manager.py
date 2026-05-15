"""
Manages static and dynamic obstacles in the river scene.

Static: trees (on banks), rocks (river edges), rooftops (beyond banks)
Dynamic: birds (patrol along river at altitude, updated each env step)

All bodies are created with both collision and visual shapes so that
PyBullet's getContactPoints() works for penalty detection.
"""

import numpy as np
import pybullet as p


class ObstacleManager:

    def __init__(self, config: dict, rng: np.random.Generator):
        self._cfg = config
        self._rng = rng
        self._river = config["river"]
        self._obs_cfg = config["obstacles"]

        self._static_bodies: list[int] = []
        self._birds: list[dict] = []   # {body_id, y_center, x, z, phase, speed}
        self._client: int | None = None

    # ------------------------------------------------------------------
    # Generation (call before load)
    # ------------------------------------------------------------------

    def generate(self):
        self._bird_params = self._generate_bird_params()
        self._static_specs = (
            self._generate_tree_specs()
            + self._generate_rock_specs()
            + self._generate_rooftop_specs()
        )

    # ------------------------------------------------------------------
    # PyBullet integration
    # ------------------------------------------------------------------

    def load_into_pybullet(self, client_id: int):
        self._client = client_id
        self._static_bodies.clear()
        self._birds.clear()

        for spec in self._static_specs:
            body = self._spawn_body(spec, client_id)
            self._static_bodies.append(body)

        for bp in self._bird_params:
            col = p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.3, physicsClientId=client_id
            )
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.3,
                rgbaColor=[0.25, 0.15, 0.05, 1.0],
                physicsClientId=client_id,
            )
            pos = [bp["x"], bp["y_center"], bp["z"]]
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=client_id,
            )
            self._birds.append({**bp, "body_id": body_id, "phase": 0.0})

    def remove_from_pybullet(self, client_id: int):
        for body_id in self._static_bodies:
            p.removeBody(body_id, physicsClientId=client_id)
        for bird in self._birds:
            p.removeBody(bird["body_id"], physicsClientId=client_id)
        self._static_bodies.clear()
        self._birds.clear()

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update_birds(self, dt: float = 1 / 48):
        """Move birds along their patrol paths. Call once per env step."""
        river_len = self._river["length"]
        for bird in self._birds:
            bird["phase"] += bird["speed"] * dt
            y = bird["y_center"] + bird["amplitude"] * np.sin(bird["phase"])
            y = float(np.clip(y, 2.0, river_len - 2.0))
            x = float(bird["x"] + 0.5 * np.sin(bird["phase"] * 0.3))
            z = float(bird["z"] + 0.3 * np.sin(bird["phase"] * 0.7))
            p.resetBasePositionAndOrientation(
                bird["body_id"],
                [x, y, z],
                [0, 0, 0, 1],
                physicsClientId=self._client,
            )

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def any_collision(self, drone_body_id: int) -> bool:
        """Return True if the drone is in contact with any obstacle."""
        all_bodies = self._static_bodies + [b["body_id"] for b in self._birds]
        for obs_id in all_bodies:
            contacts = p.getContactPoints(
                bodyA=drone_body_id,
                bodyB=obs_id,
                physicsClientId=self._client,
            )
            if contacts:
                return True
        return False

    def all_body_ids(self) -> list[int]:
        return self._static_bodies + [b["body_id"] for b in self._birds]

    # ------------------------------------------------------------------
    # Spec generators
    # ------------------------------------------------------------------

    def _generate_tree_specs(self) -> list[dict]:
        n = self._obs_cfg["num_trees"]
        river_len = self._river["length"]
        half_w = self._river["width"] / 2
        bank_w = self._river["bank_width"]
        specs = []
        for i in range(n):
            side = 1 if i % 2 == 0 else -1
            height = float(self._rng.uniform(3.0, 8.0))
            radius = float(self._rng.uniform(0.4, 1.0))
            x = side * (half_w + self._rng.uniform(0.5, bank_w - 0.5))
            y = float(self._rng.uniform(3.0, river_len - 3.0))
            specs.append({
                "shape": "cylinder",
                "radius": radius,
                "height": height,
                "pos": [x, y, height / 2],
                "color": [0.1, 0.35, 0.1, 1.0],
            })
        return specs

    def _generate_rock_specs(self) -> list[dict]:
        n = self._obs_cfg["num_rocks"]
        river_len = self._river["length"]
        half_w = self._river["width"] / 2
        specs = []
        for _ in range(n):
            radius = float(self._rng.uniform(0.4, 1.2))
            side = self._rng.choice([-1, 1])
            x = side * self._rng.uniform(half_w * 0.4, half_w * 0.9)
            y = float(self._rng.uniform(5.0, river_len - 5.0))
            specs.append({
                "shape": "sphere",
                "radius": radius,
                "pos": [x, y, radius],
                "color": [0.45, 0.45, 0.45, 1.0],
            })
        return specs

    def _generate_rooftop_specs(self) -> list[dict]:
        n = self._obs_cfg["num_rooftops"]
        river_len = self._river["length"]
        half_w = self._river["width"] / 2
        bank_w = self._river["bank_width"]
        specs = []
        for i in range(n):
            side = 1 if i % 2 == 0 else -1
            house_h = float(self._rng.uniform(5.0, 12.0))
            roof_hw = [
                float(self._rng.uniform(2.0, 5.0)),
                float(self._rng.uniform(2.0, 5.0)),
                0.4,
            ]
            x = side * (half_w + bank_w + self._rng.uniform(1.0, 4.0))
            y = float(self._rng.uniform(5.0, river_len - 5.0))
            specs.append({
                "shape": "box",
                "half_extents": roof_hw,
                "pos": [x, y, house_h],
                "color": [0.55, 0.28, 0.18, 1.0],
            })
        return specs

    def _generate_bird_params(self) -> list[dict]:
        n = self._obs_cfg["num_birds"]
        river_len = self._river["length"]
        half_w = self._river["width"] / 2
        birds = []
        for _ in range(n):
            birds.append({
                "x": float(self._rng.uniform(-half_w * 0.8, half_w * 0.8)),
                "y_center": float(self._rng.uniform(15.0, river_len - 15.0)),
                "z": float(self._rng.uniform(4.0, 12.0)),
                "amplitude": float(self._rng.uniform(5.0, 20.0)),
                "speed": float(self._rng.uniform(0.5, 2.0)),
            })
        return birds

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spawn_body(self, spec: dict, client_id: int) -> int:
        shape = spec["shape"]
        color = spec["color"]

        if shape == "cylinder":
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=spec["radius"],
                height=spec["height"],
                physicsClientId=client_id,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=spec["radius"],
                length=spec["height"],
                rgbaColor=color,
                physicsClientId=client_id,
            )
        elif shape == "sphere":
            col = p.createCollisionShape(
                p.GEOM_SPHERE, radius=spec["radius"], physicsClientId=client_id
            )
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=spec["radius"],
                rgbaColor=color,
                physicsClientId=client_id,
            )
        elif shape == "box":
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=spec["half_extents"],
                physicsClientId=client_id,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=spec["half_extents"],
                rgbaColor=color,
                physicsClientId=client_id,
            )
        else:
            raise ValueError(f"Unknown obstacle shape: {shape}")

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=spec["pos"],
            physicsClientId=client_id,
        )
