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
from .ppo import *
from .layers import *

# Import utility functions
from .utils import *

__all__ = [
    "BurgersEnvironment",
    "PPOAgent", 
    "evaluate_on_testset",
    "evaluate_on_env",
    "pretrain_policy",
] 