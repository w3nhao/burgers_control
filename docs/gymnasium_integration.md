# Gymnasium Integration for Burgers Control

This document explains how the Burgers Control package has been updated to follow Gymnasium conventions while maintaining full backward compatibility and efficient vectorized computation.

## The Challenge

The original system had several non-standard features:

1. **Custom registration system** instead of Gymnasium's `register()`
2. **Always vectorized environments** (no single environment option)
3. **Configuration complexity** with custom `EnvironmentConfig` classes
4. **Non-standard creation** using `create_env()` instead of `gym.make()`

However, the vectorized environments were essential for:
- **Parallel tensor computation** with PyTorch
- **High-performance training** with thousands of parallel environments
- **Efficient GPU utilization**

## The Solution: Clever Wrapper Approach

Instead of rewriting the entire system, we implemented a clever workaround that follows Gymnasium conventions while maintaining functionality:

### 1. Single Environment Wrapper (`BurgersEnv`)

Created a thin wrapper around `BurgersOnTheFlyVecEnv` with `num_envs=1`:

```python
class BurgersEnv(Env):
    def __init__(self, spatial_size=128, num_time_points=10, ...):
        # Create VectorEnv with num_envs=1
        self.vec_env = BurgersOnTheFlyVecEnv(
            num_envs=1,
            spatial_size=spatial_size,
            # ... other params
        )
        
        # Expose standard Gymnasium interfaces
        self.observation_space = self.vec_env.single_observation_space
        self.action_space = self.vec_env.single_action_space
    
    def step(self, action):
        # Expand to batch dimension and extract single result
        action_batch = np.expand_dims(action, axis=0)
        obs, rewards, terminated, truncated, info = self.vec_env.step(action_batch)
        return obs[0], rewards[0], terminated[0], truncated[0], info_single
```

### 2. Standard Gymnasium Registration

Registered environments using standard Gymnasium patterns:

```python
from gymnasium.envs.registration import register

register(
    id="BurgersVec-v0",
    entry_point=make_burgers_env,
    kwargs={
        "spatial_size": 128,
        "num_time_points": 10,
        "viscosity": 0.01,
        "sim_time": 1.0,
        # ... other parameters
    },
    max_episode_steps=10,
)
```

### 3. Backward Compatibility Layer

Enhanced the existing `env_configs.py` to support both systems:

```python
def create_env(env_id: str, num_envs: int, **kwargs):
    # Try legacy configs first
    if env_id in ENV_CONFIGS:
        config = get_env_config(env_id)
        return config.create_env(num_envs, **kwargs)
    
    # Fallback to Gymnasium registration
    try:
        env_kwargs = get_environment_kwargs(env_id)
        env_kwargs.update(kwargs)
        return make_burgers_vec_env(num_envs=num_envs, **env_kwargs)
    except:
        raise ValueError(f"Environment {env_id} not found")
```

## Usage Patterns

### New Gymnasium Standard Way

```python
import gymnasium as gym
import burgers_control  # Triggers registration

# Single environment (internally uses VectorEnv with num_envs=1)
env = gym.make("BurgersVec-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Vectorized environment (direct usage)
from burgers_control import make_burgers_vec_env
vec_env = make_burgers_vec_env(num_envs=8, spatial_size=128)
```

### Legacy Compatibility (Still Works)

```python
from burgers_control.env_configs import create_env

# Old system still works
env = create_env("BurgersVec-v0", num_envs=8)
```

### PPO Training (Automatically Supports Both)

```python
# PPO script tries legacy first, falls back to Gymnasium
try:
    env = create_env(args.env_id, num_envs=args.num_envs)  # Legacy
except ValueError:
    env_kwargs = get_environment_kwargs(args.env_id)      # Gymnasium
    env = make_burgers_vec_env(num_envs=args.num_envs, **env_kwargs)
```

## Benefits of This Approach

### ✅ Follows Gymnasium Conventions
- Standard `gym.make()` interface
- Proper environment registration
- Compatible with Gymnasium tools and wrappers

### ✅ Maintains Performance
- VectorEnv still used for all actual computation
- No performance overhead for vectorized operations
- GPU acceleration preserved

### ✅ Full Backward Compatibility
- Existing code continues to work unchanged
- Legacy configuration system still functional
- No breaking changes

### ✅ Best of Both Worlds
- Single environment interface for Gymnasium compatibility
- Vectorized environments for high-performance training
- Automatic fallback between systems

## Registered Environments

| Environment ID | Spatial Size | Time Points | Sim Time | Reward Type |
|---------------|--------------|-------------|----------|-------------|
| `BurgersVec-v0` | 128 | 10 | 1.0 | exp_scaled_mse |
| `BurgersVec-v1` | 128 | 10 | 0.1 | exp_scaled_mse |
| `BurgersVec-debug` | 64 | 5 | 0.1 | exp_scaled_mse |

## File Structure

```
burgers_control/
├── burgers_env.py          # Single environment wrapper + factory functions
├── register.py             # Gymnasium registration system
├── env_configs.py          # Legacy system (enhanced for compatibility)
├── burgers_onthefly_env.py # Original VectorEnv (unchanged)
├── ppo.py                  # Updated to support both systems
└── __init__.py             # Triggers registration on import
```

## Migration Guide

### For New Code
Use standard Gymnasium patterns:
```python
import gymnasium as gym
import burgers_control

env = gym.make("BurgersVec-v0")  # Single environment
# or
from burgers_control import make_burgers_vec_env
vec_env = make_burgers_vec_env(num_envs=8)  # Vectorized
```

### For Existing Code
No changes needed! The legacy system continues to work:
```python
from burgers_control.env_configs import create_env
env = create_env("BurgersVec-v0", num_envs=8)  # Still works
```

This approach cleverly solves the Gymnasium convention problem without sacrificing the efficient vectorized computation that was essential for the project's performance requirements. 