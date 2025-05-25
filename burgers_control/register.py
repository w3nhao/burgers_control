"""
Gymnasium Environment Registration for Burgers Control

This module registers Burgers environments with Gymnasium and contains all legacy
compatibility functions. It provides both single environment and vectorized 
environment access patterns.

Usage:
    import gymnasium as gym
    import burgers_control  # This triggers registration
    
    # Single environment (backed by VectorEnv with num_envs=1)
    env = gym.make("BurgersVec-v0")
    
    # Vectorized environment (using original VectorEnv)
    from burgers_control import make_burgers_vec_env
    vec_env = make_burgers_vec_env(num_envs=8, spatial_size=128, ...)
    
    # Get preset configurations
    kwargs = get_environment_kwargs("BurgersVec-v0")
    vec_env = make_burgers_vec_env(num_envs=8192, **kwargs)

Adding New Environment Specifications:
    To add a new environment configuration, add a new register() call in 
    register_burgers_environments():
    
    register(
        id="BurgersVec-custom",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 256,        # Custom spatial resolution
            "num_time_points": 15,      # Custom episode length
            "viscosity": 0.005,         # Custom physics parameters
            "sim_time": 0.5,           # Custom simulation time
            "reward_type": "vanilla",   # Custom reward function
            # ... other parameters
        },
        max_episode_steps=15,  # Should match num_time_points
    )
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import warnings
from typing import Dict, Any

from burgers_control.burgers_onthefly_env import make_burgers_env


def register_burgers_environments():
    """
    Register Burgers environments with Gymnasium.
    
    This function registers the environment configurations as standard
    Gymnasium environments that can be created with gym.make().
    
    To add new environment specifications:
    1. Add a new register() call with your custom parameters
    2. Use a unique ID following the pattern "BurgersVec-<name>"
    3. Set max_episode_steps to match num_time_points
    4. All kwargs will be passed to make_burgers_env()
    """
    
    # Register BurgersVec-v0 
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
            "use_random_targets": False,
        },
        max_episode_steps=10,  # num_time_points
    )
    
    # Register BurgersVec-Tune (Converge Faseter)
    register(
        id="BurgersVec-V1",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 128,
            "num_time_points": 10,
            "viscosity": 0.01,
            "sim_time": 1.0,
            "time_step": 1e-4,
            "forcing_terms_scaling_factor": 1.0,
            "reward_type": "exp_scaled_mse",
            "mse_scaling_factor": 5e2,
            "use_random_targets": False,
        },
        max_episode_steps=10,  # num_time_points
    )
    
    # Register BurgersVec-v1 (faster simulation)
    register(
        id="BurgersVec-v2",
        entry_point=make_burgers_env,
        kwargs={
            "spatial_size": 128,
            "num_time_points": 10,
            "viscosity": 0.01,
            "sim_time": 0.1,  
            "time_step": 1e-4,
            "forcing_terms_scaling_factor": 1.0,
            "reward_type": "exp_scaled_mse",
            "mse_scaling_factor": 1e3,
            "use_random_targets": False,
        },
        max_episode_steps=10,  # num_time_points
    )
    
    # Register BurgersVec-debug (small environment for testing)
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
            "use_random_targets": False,
        },
        max_episode_steps=5,  # num_time_points
    )

def list_registered_environments():
    """
    List all registered Burgers environments.
    
    Returns:
        list: List of registered environment IDs
        
    Example:
        >>> from burgers_control import list_registered_environments
        >>> envs = list_registered_environments()
        >>> print(envs)
        ['BurgersVec-debug', 'BurgersVec-v0', 'BurgersVec-v1']
    """
    all_envs = gym.envs.registry.keys()
    burgers_envs = [env_id for env_id in all_envs if env_id.startswith("BurgersVec")]
    return sorted(burgers_envs)


def get_environment_kwargs(env_id: str):
    """
    Get the kwargs used to register an environment.
    
    This is useful for getting preset configurations to initialize
    vectorized environments with the same parameters.
    
    Args:
        env_id: Environment ID (e.g., "BurgersVec-v0")
        
    Returns:
        dict: Environment kwargs that can be passed to make_burgers_vec_env()
        
    Raises:
        ValueError: If environment is not registered
        
    Example:
        >>> from burgers_control import get_environment_kwargs, make_burgers_vec_env
        >>> kwargs = get_environment_kwargs("BurgersVec-v1")
        >>> vec_env = make_burgers_vec_env(num_envs=8192, **kwargs)
        >>> print(f"Created environment with spatial_size={kwargs['spatial_size']}")
    """
    if env_id not in gym.envs.registry:
        available_envs = list_registered_environments()
        raise ValueError(f"Environment {env_id} not registered. Available: {available_envs}")
    
    spec = gym.envs.registry[env_id]
    return spec.kwargs.copy() if spec.kwargs else {}


def add_environment_spec(env_id: str, **kwargs):
    """
    Dynamically add a new environment specification.
    
    This allows users to register new environment configurations at runtime
    without modifying the source code.
    
    Args:
        env_id: Unique environment ID (should start with "BurgersVec-")
        **kwargs: Environment parameters to pass to make_burgers_env()
        
    Example:
        >>> from burgers_control.register import add_environment_spec
        >>> add_environment_spec(
        ...     "BurgersVec-highres",
        ...     spatial_size=256,
        ...     num_time_points=20,
        ...     viscosity=0.005,
        ...     sim_time=0.5,
        ...     reward_type="vanilla"
        ... )
        >>> import gymnasium as gym
        >>> env = gym.make("BurgersVec-highres")
    """
    # Extract max_episode_steps from num_time_points if available
    max_episode_steps = kwargs.get('num_time_points', 10)
    
    register(
        id=env_id,
        entry_point=make_burgers_env,
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
    )


# ==============================================================================
# DEPRECATED LEGACY FUNCTIONS
# ==============================================================================
# The following functions are maintained for backward compatibility only.
# All new code should use the standard Gymnasium registration system above.

def create_env(*args, **kwargs):
    """
    DEPRECATED: Use burgers_control.make_burgers_vec_env() instead.
    
    This function is maintained for backward compatibility only.
    New code should use:
        from burgers_control import make_burgers_vec_env
        vec_env = make_burgers_vec_env(num_envs=8, **get_environment_kwargs("BurgersVec-v0"))
    """
    warnings.warn(
        "create_env() is deprecated. Use 'from burgers_control import make_burgers_vec_env' "
        "and 'get_environment_kwargs()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to new system
    from burgers_control.burgers_onthefly_env import make_burgers_vec_env
    
    # Handle both positional and keyword arguments
    if len(args) >= 2:
        env_id, num_envs = args[0], args[1]
    elif len(args) == 1 and 'num_envs' in kwargs:
        env_id = args[0]
        num_envs = kwargs.pop('num_envs')
    elif 'env_id' in kwargs and 'num_envs' in kwargs:
        env_id = kwargs.pop('env_id')
        num_envs = kwargs.pop('num_envs')
    else:
        raise ValueError("create_env requires env_id and num_envs as arguments")
    
    env_kwargs = get_environment_kwargs(env_id)
    env_kwargs.update(kwargs)
    return make_burgers_vec_env(num_envs=num_envs, **env_kwargs)


def get_env_config(*args, **kwargs):
    """DEPRECATED: Use burgers_control.get_environment_kwargs() instead."""
    warnings.warn(
        "get_env_config() is deprecated. Use 'from burgers_control import get_environment_kwargs' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("This function has been removed. Use get_environment_kwargs() instead.")


def list_env_configs() -> Dict[str, Dict[str, Any]]:
    """DEPRECATED: Use burgers_control.list_registered_environments() instead."""
    warnings.warn(
        "list_env_configs() is deprecated. Use 'from burgers_control import list_registered_environments' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    registered_envs = list_registered_environments()
    return {env_id: get_environment_kwargs(env_id) for env_id in registered_envs}


def register_env_config(*args, **kwargs):
    """DEPRECATED: Use add_environment_spec() or gymnasium.register() instead."""
    warnings.warn(
        "register_env_config() is deprecated. Use add_environment_spec() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    pass  # No-op for backward compatibility


class EnvironmentConfig:
    """DEPRECATED: Use gymnasium.register() with kwargs instead."""
    
    def __init__(self, **kwargs):
        warnings.warn(
            "EnvironmentConfig is deprecated. Use add_environment_spec() or gymnasium.register() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        pass 


# Register environments when module is imported
register_burgers_environments() 