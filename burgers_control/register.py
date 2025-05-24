"""
Gymnasium Environment Registration for Burgers Control

This module registers Burgers environments with Gymnasium following standard conventions.
It provides both single environment and vectorized environment access patterns.

Usage:
    import gymnasium as gym
    import burgers_control  # This triggers registration
    
    # Single environment (backed by VectorEnv with num_envs=1)
    env = gym.make("BurgersVec-v0")
    
    # Vectorized environment (using original VectorEnv)
    from burgers_control import make_burgers_vec_env
    vec_env = make_burgers_vec_env(num_envs=8, spatial_size=128, ...)
    
    # Or using the env configs (maintains compatibility)
    from burgers_control.env_configs import create_env
    vec_env = create_env("BurgersVec-v0", num_envs=8)
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from burgers_control.burgers_env import make_burgers_env


def register_burgers_environments():
    """
    Register Burgers environments with Gymnasium.
    
    This function registers the environment configurations as standard
    Gymnasium environments that can be created with gym.make().
    """
    
    # Register BurgersVec-v0 (original configuration)
    register(
        id="BurgersVec-v0",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 128,
            "num_time_points": 10,
            "viscosity": 0.01,
            "sim_time": 1.0,
            "time_step": 1e-4,
            "forcing_terms_scaling_factor": 1.0,
            "reward_type": "exp_scaled_mse",
            "mse_scaling_factor": 1e3,
        },
        max_episode_steps=10,  # num_time_points
    )
    
    # Register BurgersVec-v1 (modified with sim_time=0.1)
    register(
        id="BurgersVec-v1",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 128,
            "num_time_points": 10,
            "viscosity": 0.01,
            "sim_time": 0.1,  # Changed from 1.0 to 0.1
            "time_step": 1e-4,
            "forcing_terms_scaling_factor": 1.0,
            "reward_type": "exp_scaled_mse",
            "mse_scaling_factor": 1e3,
        },
        max_episode_steps=10,  # num_time_points
    )
    
    # Register a basic configuration for testing/debugging
    register(
        id="BurgersVec-debug",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 64,
            "num_time_points": 5,
            "viscosity": 0.01,
            "sim_time": 0.1,
            "time_step": 1e-4,
            "forcing_terms_scaling_factor": 1.0,
            "reward_type": "exp_scaled_mse",
            "mse_scaling_factor": 1e3,
        },
        max_episode_steps=5,
    )


def list_registered_environments():
    """
    List all registered Burgers environments.
    
    Returns:
        list: List of registered environment IDs
    """
    all_envs = gym.envs.registry.keys()
    burgers_envs = [env_id for env_id in all_envs if env_id.startswith("BurgersVec")]
    return sorted(burgers_envs)


def get_environment_kwargs(env_id: str):
    """
    Get the kwargs used to register an environment.
    
    Args:
        env_id: Environment ID (e.g., "BurgersVec-v0")
        
    Returns:
        dict: Environment kwargs
        
    Raises:
        ValueError: If environment is not registered
    """
    if env_id not in gym.envs.registry:
        available_envs = list_registered_environments()
        raise ValueError(f"Environment {env_id} not registered. Available: {available_envs}")
    
    spec = gym.envs.registry[env_id]
    return spec.kwargs.copy() if spec.kwargs else {}


# Register environments when module is imported
register_burgers_environments() 