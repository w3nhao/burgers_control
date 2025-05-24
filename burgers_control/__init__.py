"""
Burgers Control: A package for reinforcement learning-based control of the Burgers equation.

This package provides tools for training and evaluating RL agents to control
the Burgers partial differential equation using various neural network architectures.
"""

__version__ = "0.1.0"
__author__ = "Wenhao Deng"

# Import main modules for easy access
from .burgers import *
from .burgers_onthefly_env import *
from .burgers_env import *
from .ppo import *
from .layers import *

# Import environment configurations for backward compatibility
from .env_configs import *

# Import Gymnasium registration (this triggers environment registration)
from . import register

# Import utility functions
from .utils import *

# Additional imports for convenient access
from .burgers_env import make_burgers_vec_env
from .register import list_registered_environments, get_environment_kwargs

__all__ = [
    # Environment classes
    "BurgersOnTheFlyVecEnv",
    "BurgersEnv",
    
    # Factory functions
    "make_burgers_vec_env",
    "make_burgers_env",
    
    # Legacy environment configs (backward compatibility)
    "EnvironmentConfig",
    "register_env_config",
    "get_env_config", 
    "list_env_configs",
    "create_env",
    
    # Gymnasium integration
    "list_registered_environments",
    "get_environment_kwargs",
    
    # Other components
    "PPOAgent",
    "evaluate_on_testset", 
    "evaluate_on_env",
    "pretrain_policy",
] 