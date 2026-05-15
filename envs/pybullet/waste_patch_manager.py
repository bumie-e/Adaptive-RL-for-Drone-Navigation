import numpy as np
import pybullet as p
from scipy.spatial import KDTree


class WastePatchManager:
    """
    Generates plastic waste patches along a river and tracks coverage.

    Patches are clustered (gaussian blobs) to mimic realistic accumulation
    at eddies and bends. Positions are stored in world XY; the river runs
    along the Y-axis.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self._cfg = cfg
        self._rng = rng
        self.patches: list[dict] = []   # {pos, radius, visited, body_id}
        self._kdtree: KDTree | None = None
        self._client_id: int | None = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self):
        """Sample patch positions and radii. Call before reset()."""
        river = self._cfg["river"]
        wp = self._cfg["waste_patches"]

        river_len = river["length"]
        half_width = (river["width"] - 2 * river["bank_width"]) / 2.0

        num_patches = wp["num_patches"]
        num_clusters = wp["num_clusters"]
        cluster_std = wp["cluster_std"]

        # cluster centres spread along the river
        cluster_ys = self._rng.uniform(10.0, river_len - 10.0, num_clusters)
        cluster_xs = self._rng.uniform(-half_width * 0.5, half_width * 0.5, num_clusters)

        positions = []
        for i in range(num_patches):
            c = i % num_clusters
            x = np.clip(self._rng.normal(cluster_xs[c], cluster_std), -half_width, half_width)
            y = np.clip(self._rng.normal(cluster_ys[c], cluster_std), 2.0, river_len - 2.0)
            positions.append([x, y])

        radii = self._rng.uniform(
            wp["min_radius"], wp["max_radius"], num_patches
        )

        self.patches = [
            {"pos": np.array([positions[i][0], positions[i][1], 0.05]),
             "radius": radii[i],
             "visited": False,
             "body_id": None}
            for i in range(num_patches)
        ]
        self._rebuild_kdtree()

    def reset_visits(self):
        for p in self.patches:
            p["visited"] = False

    # ------------------------------------------------------------------
    # PyBullet integration
    # ------------------------------------------------------------------

    def load_into_pybullet(self, client_id: int):
        """Spawn visual-only cylinders for each patch in the physics world."""
        self._client_id = client_id
        for patch in self.patches:
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=patch["radius"],
                length=0.05,
                rgbaColor=[1.0, 0.5, 0.0, 0.8],   # orange = plastic waste
                physicsClientId=client_id,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                basePosition=patch["pos"].tolist(),
                physicsClientId=client_id,
            )
            patch["body_id"] = body

    def remove_from_pybullet(self, client_id: int):
        for patch in self.patches:
            if patch["body_id"] is not None:
                p.removeBody(patch["body_id"], physicsClientId=client_id)
                patch["body_id"] = None

    # ------------------------------------------------------------------
    # Coverage logic
    # ------------------------------------------------------------------

    def get_nearest_unvisited_distance(self, drone_xy: np.ndarray) -> float | None:
        """
        Returns distance to the nearest unvisited patch, or None if all are visited.
        Used by the distance-based reward shaping.
        """
        best = None
        for patch in self.patches:
            if not patch["visited"]:
                d = float(np.linalg.norm(drone_xy - patch["pos"][:2]))
                if best is None or d < best:
                    best = d
        return best

    def check_and_visit(self, drone_xy: np.ndarray, visit_radius: float) -> float:
        """
        Returns reward earned this step from newly visited patches.
        Marks patches within visit_radius as visited.
        """
        cfg = self._cfg["episode"]
        reward = 0.0
        for patch in self.patches:
            dist = np.linalg.norm(drone_xy - patch["pos"][:2])
            if dist <= visit_radius + patch["radius"]:
                if not patch["visited"]:
                    patch["visited"] = True
                    reward += cfg["patch_visit_reward"]
                    self._mark_visited_visual(patch)
                else:
                    reward += cfg["revisit_penalty"]
        return reward

    def coverage_fraction(self) -> float:
        visited = sum(1 for p in self.patches if p["visited"])
        return visited / max(len(self.patches), 1)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_nearest_obs(self, drone_pos: np.ndarray, k: int) -> np.ndarray:
        """
        Returns flat array of shape (k*4,):
          [rel_x, rel_y, distance, is_visited] for each of the k nearest patches.
        Pads with zeros if fewer than k patches exist.
        """
        if self._kdtree is None or len(self.patches) == 0:
            return np.zeros(k * 4, dtype=np.float32)

        drone_xy = drone_pos[:2]
        k_actual = min(k, len(self.patches))
        _, idxs = self._kdtree.query(drone_xy, k=k_actual)
        if k_actual == 1:
            idxs = [idxs]

        obs = np.zeros(k * 4, dtype=np.float32)
        for i, idx in enumerate(idxs):
            patch = self.patches[idx]
            rel = patch["pos"][:2] - drone_xy
            dist = float(np.linalg.norm(rel))
            obs[i * 4: i * 4 + 4] = [rel[0], rel[1], dist, float(patch["visited"])]
        return obs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_kdtree(self):
        if self.patches:
            pts = np.array([p["pos"][:2] for p in self.patches])
            self._kdtree = KDTree(pts)

    def _mark_visited_visual(self, patch: dict):
        if patch["body_id"] is not None and self._client_id is not None:
            p.changeVisualShape(
                patch["body_id"],
                -1,
                rgbaColor=[0.2, 0.8, 0.2, 0.6],   # green = visited
                physicsClientId=self._client_id,
            )
