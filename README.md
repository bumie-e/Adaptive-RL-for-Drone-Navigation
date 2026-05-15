# Adaptive RL for Drone Navigation

RL agent trained to navigate a drone over a river and systematically survey plastic waste patches, using patch locations as guidance.

## Reward

I haven't added the reward function. For the first instance, I     
  want a distance based reward function. The closer it is to the     
  taget, the more ward it gts. This distance based reward is between 
   0 and 1. When if finds a cluster, it gets a reward of 3. When it  
  reaches the end of the river, it gets a reward of 5, the episode   
  ends. I want there to be obstacles too like bird flying, trees,    
  rocks, rooftops of houses that if it hits, it gets a negatve       
  reward of -0.5. Each step takes a -0.001 reward, to force it to    
  move forward.
  
## Architecture

```
Adaptive-RL-for-Drone-Navigation/
├── envs/
│   ├── base.py                  # Shared obs/act contract (backend-agnostic)
│   ├── pybullet/
│   │   ├── river_aviary.py      # gym-pybullet-drones environment
│   │   └── waste_patch_manager.py
│   └── isaac/
│       └── river_task.py        # Isaac Lab backend (placeholder)
├── configs/
│   ├── env_config.yaml          # River size, patch counts, rewards
│   └── train_config.yaml        # PPO hyperparams, device, n_envs
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   └── callbacks.py
└── tests/
    └── test_env.py
```

## Observation Space (32D default)

| Slice | Content |
|---|---|
| `[0:3]` | Drone position (x, y, z) |
| `[3:6]` | Drone velocity |
| `[6:9]` | Roll, pitch, yaw |
| `[9:12]` | Angular velocity |
| `[12:]` | 5 nearest patches × (rel_x, rel_y, distance, is_visited) |

## Quick Start

```bash
pip install -e .
pip install -r requirements.txt

# train (GPU policy, 8 parallel CPU envs)
python scripts/train.py --run-name baseline

# evaluate with rendering
python scripts/evaluate.py --model models/baseline_final --render

# run tests
pytest tests/
```

## GPU Usage

PyBullet physics runs on CPU. The policy network trains on GPU via PyTorch — Stable-Baselines3 handles this automatically when `device: cuda` is set in `train_config.yaml`. Use `n_envs: 8` (or more) to saturate the GPU with environment rollouts from parallel CPU workers.

## Migrating to Isaac Lab

1. Implement `envs/isaac/river_task.py` (see the checklist in that file).
2. Change `backend: isaac` in `train_config.yaml`.
3. Increase `n_envs` to 1024+ — Isaac runs all envs on GPU in parallel.
4. The trained policy loads without modification since both backends share the same observation/action space (defined in `envs/base.py`).
