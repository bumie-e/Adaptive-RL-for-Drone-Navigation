from envs.base import DroneNavEnvBase


def make_env(config: dict, backend: str = "pybullet", render: bool = False):
    if backend == "pybullet":
        from envs.pybullet.river_aviary import RiverDroneEnv
        return RiverDroneEnv(config, render=render)
    elif backend == "isaac":
        from envs.isaac.river_task import RiverDroneTask
        return RiverDroneTask(config)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'pybullet' or 'isaac'.")
