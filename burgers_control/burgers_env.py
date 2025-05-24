import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv
from typing import Dict, Tuple, Any, Optional, Union

from burgers_control.burgers_onthefly_env import BurgersOnTheFlyVecEnv


class BurgersEnv(Env):
    """
    Single environment wrapper around BurgersOnTheFlyVecEnv.
    
    This class provides a standard Gymnasium Env interface while internally
    using a VectorEnv with num_envs=1. This allows for Gymnasium registration
    while maintaining the efficient tensor-based implementation.
    """
    
    def __init__(self,
                 spatial_size: int = 128,
                 num_time_points: int = 10,
                 viscosity: float = 0.01,
                 sim_time: float = 1.0,
                 time_step: float = 1e-4,
                 forcing_terms_scaling_factor: float = 1.0,
                 reward_type: str = "exp_scaled_mse",
                 mse_scaling_factor: float = 1e3):
        """
        Initialize the single Burgers environment.
        
        Args:
            spatial_size: Number of spatial points
            num_time_points: Number of time points for simulation
            viscosity: Viscosity coefficient
            sim_time: Total physical simulation time
            time_step: Physical simulation time step size
            forcing_terms_scaling_factor: Scaling factor for forcing terms
            reward_type: Type of reward function to use
            mse_scaling_factor: Scaling factor for MSE reward
        """
        # Create a VectorEnv with num_envs=1
        self.vec_env = BurgersOnTheFlyVecEnv(
            num_envs=1,
            spatial_size=spatial_size,
            num_time_points=num_time_points,
            viscosity=viscosity,
            sim_time=sim_time,
            time_step=time_step,
            forcing_terms_scaling_factor=forcing_terms_scaling_factor,
            reward_type=reward_type,
            mse_scaling_factor=mse_scaling_factor
        )
        
        # Expose the spaces from the vector environment
        self.observation_space = self.vec_env.single_observation_space
        self.action_space = self.vec_env.single_action_space
        
        # Store parameters for reproduction
        self.spatial_size = spatial_size
        self.num_time_points = num_time_points
        self.viscosity = viscosity
        self.sim_time = sim_time
        self.time_step = time_step
        self.forcing_terms_scaling_factor = forcing_terms_scaling_factor
        self.reward_type = reward_type
        self.mse_scaling_factor = mse_scaling_factor
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        # Reset the vectorized environment and extract single environment result
        obs, info = self.vec_env.reset(seed=seed, options=options)
        return obs[0], {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v for k, v in info.items()}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Expand action to batch dimension
        action_batch = np.expand_dims(action, axis=0)
        
        # Step the vectorized environment
        obs, rewards, terminations, truncations, info = self.vec_env.step(action_batch)
        
        # Extract single environment results
        obs_single = obs[0]
        reward_single = rewards[0]
        terminated_single = terminations[0]
        truncated_single = truncations[0]
        
        # Process info dict
        info_single = {}
        for k, v in info.items():
            if k == "final_info":
                # Handle final_info specially
                if v[0] is not None:
                    info_single.update(v[0])
            elif isinstance(v, np.ndarray) and v.ndim > 0:
                info_single[k] = v[0]
            else:
                info_single[k] = v
        
        return obs_single, reward_single, terminated_single, truncated_single, info_single
    
    def render(self):
        """Render the environment (if implemented in VectorEnv)."""
        return self.vec_env.render()
    
    def close(self):
        """Close the environment."""
        self.vec_env.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        return self.vec_env.seed(seed)
    
    def set_device(self, device):
        """Move environment tensors to specified device."""
        self.vec_env.set_device(device)


def make_burgers_env(**kwargs):
    """
    Factory function to create a Burgers environment.
    
    This function is designed to be used with gymnasium.register()
    
    Args:
        **kwargs: Environment parameters
        
    Returns:
        BurgersEnv: A single Burgers environment
    """
    return BurgersEnv(**kwargs)


def make_burgers_vec_env(num_envs: int = 1, **kwargs):
    """
    Factory function to create a vectorized Burgers environment.
    
    Args:
        num_envs: Number of parallel environments
        **kwargs: Environment parameters
        
    Returns:
        BurgersOnTheFlyVecEnv: A vectorized Burgers environment
    """
    return BurgersOnTheFlyVecEnv(num_envs=num_envs, **kwargs) 