"""
Environment Configuration Registry for Burgers Control

This module defines different environment configurations that can be used
across the codebase. Each configuration specifies all the parameters needed
to create a BurgersOnTheFlyVecEnv instance.

Available configurations:
- BurgersVec-v0: Original settings with sim_time=1.0
- BurgersVec-v1: Modified settings with sim_time=0.1
"""

from dataclasses import dataclass
from typing import Dict, Any
from burgers_control.burgers_onthefly_env import BurgersOnTheFlyVecEnv


@dataclass
class EnvironmentConfig:
    """Configuration for a Burgers environment."""
    spatial_size: int
    num_time_points: int
    viscosity: float
    sim_time: float
    time_step: float
    forcing_terms_scaling_factor: float
    reward_type: str
    mse_scaling_factor: float
    
    def create_env(self, num_envs: int, **kwargs) -> BurgersOnTheFlyVecEnv:
        """
        Create a BurgersOnTheFlyVecEnv instance with this configuration.
        
        Args:
            num_envs: Number of parallel environments
            **kwargs: Additional keyword arguments to override config values
            
        Returns:
            BurgersOnTheFlyVecEnv: The configured environment
        """
        # Start with configuration values
        env_kwargs = {
            'num_envs': num_envs,
            'spatial_size': self.spatial_size,
            'num_time_points': self.num_time_points,
            'viscosity': self.viscosity,
            'sim_time': self.sim_time,
            'time_step': self.time_step,
            'forcing_terms_scaling_factor': self.forcing_terms_scaling_factor,
            'reward_type': self.reward_type,
            'mse_scaling_factor': self.mse_scaling_factor,
        }
        
        # Override with any provided kwargs
        env_kwargs.update(kwargs)
        
        return BurgersOnTheFlyVecEnv(**env_kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'spatial_size': self.spatial_size,
            'num_time_points': self.num_time_points,
            'viscosity': self.viscosity,
            'sim_time': self.sim_time,
            'time_step': self.time_step,
            'forcing_terms_scaling_factor': self.forcing_terms_scaling_factor,
            'reward_type': self.reward_type,
            'mse_scaling_factor': self.mse_scaling_factor,
        }


# Environment configuration registry
ENV_CONFIGS: Dict[str, EnvironmentConfig] = {}


def register_env_config(env_id: str, config: EnvironmentConfig) -> None:
    """
    Register an environment configuration.
    
    Args:
        env_id: Environment identifier (e.g., "BurgersVec-v0")
        config: Environment configuration
    """
    ENV_CONFIGS[env_id] = config


def get_env_config(env_id: str) -> EnvironmentConfig:
    """
    Get an environment configuration by ID.
    
    Args:
        env_id: Environment identifier
        
    Returns:
        EnvironmentConfig: The requested configuration
        
    Raises:
        ValueError: If environment ID is not found
    """
    if env_id not in ENV_CONFIGS:
        available_envs = list(ENV_CONFIGS.keys())
        raise ValueError(f"Environment {env_id} not found. Available environments: {available_envs}")
    
    return ENV_CONFIGS[env_id]


def list_env_configs() -> Dict[str, Dict[str, Any]]:
    """
    List all available environment configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping env_id to config dict
    """
    return {env_id: config.to_dict() for env_id, config in ENV_CONFIGS.items()}


def create_env(env_id: str, num_envs: int, **kwargs) -> BurgersOnTheFlyVecEnv:
    """
    Create an environment using a registered configuration.
    
    Args:
        env_id: Environment identifier
        num_envs: Number of parallel environments
        **kwargs: Additional keyword arguments to override config values
        
    Returns:
        BurgersOnTheFlyVecEnv: The configured environment
    """
    config = get_env_config(env_id)
    return config.create_env(num_envs, **kwargs)


# Register V0 configuration (original settings)
register_env_config(
    "BurgersVec-v0",
    EnvironmentConfig(
        spatial_size=128,
        num_time_points=10,
        viscosity=0.01,
        sim_time=1.0,
        time_step=1e-4,
        forcing_terms_scaling_factor=1.0,
        reward_type="exp_scaled_mse",
        mse_scaling_factor=1e3,
    )
)

# Register V1 configuration (modified with sim_time=0.1)
register_env_config(
    "BurgersVec-v1",
    EnvironmentConfig(
        spatial_size=128,
        num_time_points=10,
        viscosity=0.01,
        sim_time=0.1,  # Changed from 1.0 to 0.1
        time_step=1e-4,
        forcing_terms_scaling_factor=1.0,
        reward_type="exp_scaled_mse",
        mse_scaling_factor=1e3,
    )
) 