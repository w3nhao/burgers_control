import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from typing import Dict, Tuple, Any, Optional, List, Union

import torch.nn.functional as F
from dataset import BurgersTest, test_file_path

# Import existing utilities
from evaluation import create_differential_matrices_1d


class BurgersVecEnv(VectorEnv):
    """
    Vectorized version of the BurgersEnvClosedLoop that efficiently simulates 
    multiple environments in parallel using tensor operations.
    """
    
    def __init__(self, 
                 num_envs: int = 1, 
                 test_dataset_path=test_file_path, 
                 viscosity=0.01, 
                 num_time_points=None):
        """
        Initialize the vectorized Burgers Equation Environment
        
        Args:
            num_envs: Number of parallel environments
            test_dataset_path: Path to test data file
            viscosity: Viscosity coefficient
            num_time_points: Number of time points for simulation (if None, uses length from dataset)
        """
        # Load test dataset
        self.test_dataset = BurgersTest(test_dataset_path)
        
        # Get the spatial size from the dataset
        self.spatial_size = self.test_dataset.data['observations'][0][0].shape[0]
        
        # Determine number of time points from the dataset if not provided
        if num_time_points is None:
            # Get the length of observations for the first episode
            self.num_time_points = self.test_dataset.data['observations'][0].shape[0]
            print(f"Setting num_time_points to {self.num_time_points} based on dataset")
        else:
            self.num_time_points = num_time_points
        
        # Define action and observation spaces
        single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.spatial_size,), dtype=np.float32
        )
        single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.spatial_size,), dtype=np.float32
        )
        
        # Initialize the VectorEnv with proper spaces
        super().__init__(
            num_envs=num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space
        )
        
        # Environment parameters
        self.viscosity = viscosity
        self.current_time = np.zeros(num_envs, dtype=int)
        
        # Define domain and step sizes
        self.domain_min = 0.0
        self.domain_max = 1.0
        self.spatial_step = (self.domain_max - self.domain_min) / (self.spatial_size + 1)
        
        # Total simulation time and time step for numerical integration
        self.sim_time = 0.1  # Match the default in burgers_solver
        self.time_step = 1e-4  # Match the default in burgers_solver
        
        # Time step for simulation matches the total simulation time divided by num_time_points
        self.simulation_dt = self.sim_time / self.num_time_points
        
        # Steps per simulation time step
        self.steps_per_sim = int(self.simulation_dt / self.time_step)
        
        # Setup differential matrices for finite difference method
        self.first_deriv, self.second_deriv = create_differential_matrices_1d(
            self.spatial_size + 2, device='cpu'
        )
        
        # Prepare the differential matrices for efficient computation
        self._prepare_differential_matrices()
        
        # Current state and episode info - batched for all environments
        self.current_state = None  # Will be tensor of shape [num_envs, spatial_size]
        self.target_state = None   # Will be tensor of shape [num_envs, spatial_size]
        self.episode_idx = np.zeros(num_envs, dtype=int)
        self.initial_state = None  # Will be tensor of shape [num_envs, spatial_size]
        
        # Track actions for validation - list of lists, one per environment
        self.actions_history = [[] for _ in range(num_envs)]
        
        # Using CPU by default, can be moved to GPU if needed
        self.device = torch.device("cpu")

    def _prepare_differential_matrices(self):
        """Prepare differential matrices for efficient computation"""
        # Adjust boundary conditions for the matrices
        self.first_deriv.rows[0] = self.first_deriv.rows[0][:2]
        self.first_deriv.rows[-1] = self.first_deriv.rows[-1][-2:]
        self.first_deriv.data[0] = self.first_deriv.data[0][:2]
        self.first_deriv.data[-1] = self.first_deriv.data[-1][-2:]
        
        self.second_deriv.rows[0] = self.second_deriv.rows[0][:3]
        self.second_deriv.rows[-1] = self.second_deriv.rows[-1][-3:]
        self.second_deriv.data[0] = self.second_deriv.data[0][:3]
        self.second_deriv.data[-1] = self.second_deriv.data[-1][-3:]
        
        # Convert sparse matrices to tensor format for computation
        self.transport_indices = list(self.first_deriv.rows)
        self.transport_coeffs = torch.FloatTensor(np.stack(self.first_deriv.data) / (2 * self.spatial_step))
        self.diffusion_indices = list(self.second_deriv.rows)
        self.diffusion_coeffs = torch.FloatTensor(self.viscosity * np.stack(self.second_deriv.data) / self.spatial_step**2)

    def simulate(self, states, forcing_terms):
        """
        Simulate one step of the Burgers equation for multiple environments in parallel
        
        Args:
            states: Current states with shape [num_envs, spatial_size]
            forcing_terms: Forcing terms with shape [num_envs, spatial_size]
            
        Returns:
            next_states: States after one simulation step with shape [num_envs, spatial_size]
        """
        # Make sure state is properly padded with boundary conditions
        states_with_boundary = F.pad(states, (1, 1))
        
        # Process forcing term (already has the right shape)
        forcing_with_boundary = F.pad(forcing_terms, (1, 1))
        
        # Simulate the required number of small time steps
        for _ in range(self.steps_per_sim):
            # Remove boundary values and repad to maintain consistent size
            states_with_boundary = states_with_boundary[..., 1:-1]
            states_with_boundary = F.pad(states_with_boundary, (1, 1))
            
            # Calculate nonlinear transport and diffusion terms
            squared_state = states_with_boundary**2
            transport_term = torch.einsum('nsi,si->ns', squared_state[..., self.transport_indices], self.transport_coeffs)
            diffusion_term = torch.einsum('nsi,si->ns', states_with_boundary[..., self.diffusion_indices], self.diffusion_coeffs)
            
            # Update state using Euler step
            states_with_boundary = states_with_boundary + self.time_step * (
                -(1/2) * transport_term + 
                diffusion_term + 
                forcing_with_boundary
            )
        
        # Return state without boundary
        return states_with_boundary[..., 1:-1]

    def reset(self, 
              mask: Optional[np.ndarray] = None, 
              *, 
              seed: Optional[Union[int, List[int]]] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset specified environments or all environments if mask is None
        
        Args:
            mask: Boolean mask of which environments to reset (True = reset)
            seed: Random seed(s)
            options: Additional options for reset
            
        Returns:
            observations: Initial states for reset environments
            info: Additional information
        """
        mask = self._normalize_mask(mask)
        
        # Initialize random number generator with seed
        if seed is not None:
            if isinstance(seed, int):
                seeds = [seed + i for i in range(self.num_envs)]
            else:
                seeds = seed
        else:
            seeds = [None] * self.num_envs
        
        # If this is the first reset, initialize all environments
        if self.current_state is None:
            self.current_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            self.target_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            self.initial_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            mask = np.ones(self.num_envs, dtype=bool)
        
        # Reset environments according to mask
        for i in range(self.num_envs):
            if mask[i]:
                # Set random seed for this environment
                if seeds[i] is not None:
                    np.random.seed(seeds[i])
                
                # Choose a random episode from the test dataset
                self.episode_idx[i] = np.random.randint(0, len(self.test_dataset))
                episode_data = self.test_dataset[self.episode_idx[i]]
                
                # Get initial state and target state
                initial_state = torch.tensor(episode_data['observations'][0], device=self.device).float()
                target_state = torch.tensor(episode_data['target'], device=self.device).float()
                
                # Store states
                self.current_state[i] = initial_state
                self.initial_state[i] = initial_state
                self.target_state[i] = target_state
                
                # Reset actions history for this environment
                self.actions_history[i] = []
                
                # Reset time counter
                self.current_time[i] = 0
        
        # Convert to numpy for gym interface - must move to CPU first
        observations = self.current_state.cpu().numpy()
        
        # Prepare info dict
        info = {
            "target_state": self.target_state.cpu().numpy(),
            "episode_idx": self.episode_idx.copy(),
        }
        
        return observations, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Take a step in all environments using provided actions
        
        Args:
            actions: Actions to take in each environment with shape [num_envs, spatial_size]
            
        Returns:
            observations: Next states
            rewards: Rewards for the actions
            terminations: Whether episodes are terminated
            truncations: Whether episodes are truncated
            info: Additional information
        """
        # Store actions for validation
        for i in range(self.num_envs):
            self.actions_history[i].append(actions[i])
        
        # Convert actions to tensor
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        # Simulate one step forward in time
        self.current_state = self.simulate(self.current_state, actions_tensor)
        
        # Increment time for all environments
        self.current_time += 1
        
        # Check if episodes are done
        terminations = (self.current_time >= self.num_time_points)
        
        # Calculate rewards
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        
        # For terminated environments, calculate reward based on final state
        for i in range(self.num_envs):
            if terminations[i]:
                # Use exp to transform the mse to [0, 1]
                rewards[i] = np.exp(-((self.current_state[i] - self.target_state[i])**2).mean().cpu().item())
        
        # No truncations in this environment
        truncations = np.zeros(self.num_envs, dtype=bool)
        
        # Calculate errors for info
        errors = ((self.current_state - self.target_state)**2).mean(dim=1).cpu().numpy()
        
        # Convert states to numpy for gym interface - must move to CPU first
        observations = self.current_state.cpu().numpy()
        
        # Prepare info dict
        info = {
            "target_state": self.target_state.cpu().numpy(),
            "time_step": self.current_time.copy(),
            "error": errors,
        }
        
        return observations, rewards, terminations, truncations, info
    
    def _normalize_mask(self, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Normalize the mask to a boolean array of the correct shape
        
        Args:
            mask: Boolean mask or None
            
        Returns:
            normalized_mask: Boolean mask of shape [num_envs]
        """
        if mask is None:
            # Reset all environments if mask is None
            return np.ones(self.num_envs, dtype=bool)
        elif isinstance(mask, bool):
            # If a single boolean is provided, apply to all environments
            return np.full(self.num_envs, mask, dtype=bool)
        else:
            # Otherwise, ensure mask is a boolean array of correct shape
            if mask.shape != (self.num_envs,):
                raise ValueError(f"Mask shape {mask.shape} does not match number of environments {self.num_envs}")
            return mask.astype(bool)

    def set_device(self, device):
        """
        Move environment tensors to the specified device
        
        Args:
            device: PyTorch device to use
        """
        self.device = device
        self.transport_coeffs = self.transport_coeffs.to(device)
        self.diffusion_coeffs = self.diffusion_coeffs.to(device)
        
        if self.current_state is not None:
            self.current_state = self.current_state.to(device)
            self.target_state = self.target_state.to(device)
            self.initial_state = self.initial_state.to(device)

    def seed(self, seed: Optional[Union[int, List[int]]] = None):
        """
        Set random seed for the environments
        
        Args:
            seed: Seed for random number generator
        """
        if seed is None:
            return
        
        if isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = seed
            
        # Store seeds for reset
        self.seeds = seeds
        
        # Set seed for numpy
        np.random.seed(seeds[0])


# Factory function to create the environment
def make_burgers_vec_env(num_envs=1, test_dataset_path=test_file_path, viscosity=0.01, num_time_points=None):
    """
    Create a vectorized Burgers equation environment
    
    Args:
        num_envs: Number of parallel environments
        test_dataset_path: Path to test data file
        viscosity: Viscosity coefficient
        num_time_points: Number of time points for simulation (if None, uses length from dataset)
        
    Returns:
        BurgersVecEnv: A vectorized Burgers equation environment
    """
    return BurgersVecEnv(num_envs, test_dataset_path, viscosity, num_time_points) 