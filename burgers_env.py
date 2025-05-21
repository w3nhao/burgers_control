import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

from dataset import BurgersTest, test_file_path
from evaluation import burgers_solver


class BurgersEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, test_dataset_path=test_file_path, viscosity=0.01, num_time_points=10):
        """
        Initialize the Burgers Equation Environment
        
        Args:
            test_dataset_path: Path to test data file
            viscosity: Viscosity coefficient
            num_time_points: Number of time points for simulation
        """
        super().__init__()
        
        # Load test dataset
        self.test_dataset = BurgersTest(test_dataset_path)
        
        # Get the spatial size from the dataset
        self.spatial_size = self.test_dataset.data['observations'][0][0].shape[0]
        
        # Environment parameters
        self.viscosity = viscosity
        self.num_time_points = num_time_points
        self.current_time = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.spatial_size,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.spatial_size,), dtype=np.float32
        )
        
        # Current state and episode info
        self.current_state = None
        self.target_state = None
        self.episode_idx = None
        self.initial_state = None
        self.actions_history = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Choose a random episode from the test dataset
        self.episode_idx = self.np_random.integers(0, len(self.test_dataset))
        episode_data = self.test_dataset[self.episode_idx]
        
        # Get initial state and target state
        self.current_state = torch.tensor(episode_data['observations'][0]).float().unsqueeze(0)
        self.initial_state = self.current_state.clone()
        self.target_state = torch.tensor(episode_data['target']).float().unsqueeze(0)
        
        # Reset actions history
        self.actions_history = []
        
        # Reset time counter
        self.current_time = 0
        
        # Convert to numpy array for gym interface
        observation = self.current_state.squeeze().numpy()
        
        info = {
            "target_state": self.target_state.squeeze().numpy(),
            "episode_idx": self.episode_idx,
        }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Forcing term to apply
            
        Returns:
            observation: Next state
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Store the action
        self.actions_history.append(action)
        
        # Increment time
        self.current_time += 1
        
        # Check if the episode is done
        done = self.current_time >= self.num_time_points
        
        # For simulation purposes, we need to pad actions to match num_time_points
        # Create a forcing terms tensor with the correct dimensions
        forcing_terms = torch.zeros(1, self.num_time_points, self.spatial_size)
        
        # Fill in the actions we've taken so far
        for i, act in enumerate(self.actions_history):
            forcing_terms[0, i] = torch.tensor(act, dtype=torch.float32)
        
        # Simulate full trajectory, but we'll only use the state at current_time
        trajectory = burgers_solver(
            self.initial_state, 
            forcing_terms, 
            num_time_points=self.num_time_points
        )
        
        # Get the current state from the trajectory (based on current time)
        self.current_state = trajectory[:, self.current_time].reshape(1, self.spatial_size)
        
        if done:
            # For terminal state, calculate reward as negative MSE to target
            reward = -((self.current_state - self.target_state)**2).mean().item()
        else:
            # For intermediate states, use small negative reward
            reward = -0.1
        
        # Convert to numpy array for gym interface
        observation = self.current_state.squeeze().numpy()
        
        info = {
            "target_state": self.target_state.squeeze().numpy(),
            "time_step": self.current_time,
            "error": ((self.current_state - self.target_state)**2).mean().item(),
        }
        
        return observation, reward, done, False, info

    def render(self):
        """
        Render the environment
        """
        pass  # Implement visualization if needed

    def close(self):
        """
        Clean up environment resources
        """
        pass


# Factory function to create the environment
def make_burgers_env(test_dataset_path=test_file_path, viscosity=0.01, num_time_points=10):
    """
    Create a Burgers equation environment instance
    
    Args:
        test_dataset_path: Path to test data file
        viscosity: Viscosity coefficient
        num_time_points: Number of time points for simulation
        
    Returns:
        BurgersEnv: A Burgers equation environment
    """
    return BurgersEnv(test_dataset_path, viscosity, num_time_points) 