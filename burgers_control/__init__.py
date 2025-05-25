"""
Burgers Control: A package for reinforcement learning-based control of the Burgers equation.

This package provides tools for training and evaluating RL agents to control
the Burgers partial differential equation using various neural network architectures.

Usage:
    import gymnasium as gym
    import burgers_control  # Triggers Gymnasium registration
    
    # Single environment (standard Gymnasium interface)
    env = gym.make("BurgersVec-v0")
    
    # Vectorized environment (high-performance training)
    from burgers_control import make_burgers_vec_env
    vec_env = make_burgers_vec_env(num_envs=8192, spatial_size=128)
"""

__version__ = "0.1.0"
__author__ = "Wenhao Deng"

# Import main modules for easy access
from .burgers import *
from .burgers_onthefly_env import *
from .layers import *

# Import Gymnasium registration (this triggers environment registration)
from . import register

# Import utility functions
from .utils import *

# Main exports for convenient access
from .burgers_onthefly_env import make_burgers_vec_env, make_burgers_env, BurgersOnTheFlyVecEnv
from .register import list_registered_environments, get_environment_kwargs, add_environment_spec

__all__ = [
    # Environment classes
    "BurgersOnTheFlyVecEnv",
    
    # Factory functions (recommended)
    "make_burgers_vec_env",
    "make_burgers_env",
    
    # Gymnasium integration (recommended)
    "list_registered_environments",
    "get_environment_kwargs",
    "add_environment_spec",
    
    # Other evaluation functions
    "evaluate_on_testset", 
    "evaluate_on_env",
    "pretrain_policy",
    
] 