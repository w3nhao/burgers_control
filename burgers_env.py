import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import torch.nn.functional as F
from dataset import BurgersTest, test_file_path

# Import existing burgers_solver for comparison/validation
from evaluation import create_differential_matrices_1d, burgers_solver


class BurgersEnvClosedLoop(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, test_dataset_path=test_file_path, viscosity=0.01, num_time_points=None):
        """
        Initialize the Burgers Equation Environment with closed-loop simulation and sparse rewards
        
        Args:
            test_dataset_path: Path to test data file
            viscosity: Viscosity coefficient
            num_time_points: Number of time points for simulation (if None, uses length from dataset)
        """
        super().__init__()
        
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
        
        # Environment parameters
        self.viscosity = viscosity
        self.current_time = 0
        
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

    def simulate(self, state, forcing_term):
        """
        Simulate one step of the Burgers equation
        
        Args:
            state: Current state
            forcing_term: Forcing term to apply
            
        Returns:
            next_state: State after one simulation step
        """
        # Make sure state is properly padded with boundary conditions
        state_with_boundary = F.pad(state, (1, 1))
        
        # Process forcing term (already has the right shape)
        forcing_with_boundary = F.pad(forcing_term, (1, 1))
        
        # Simulate the required number of small time steps
        for _ in range(self.steps_per_sim):
            # Remove boundary values and repad to maintain consistent size
            state_with_boundary = state_with_boundary[..., 1:-1]
            state_with_boundary = F.pad(state_with_boundary, (1, 1))
            
            # Calculate nonlinear transport and diffusion terms
            squared_state = state_with_boundary**2
            transport_term = torch.einsum('nsi,si->ns', squared_state[..., self.transport_indices], self.transport_coeffs)
            diffusion_term = torch.einsum('nsi,si->ns', state_with_boundary[..., self.diffusion_indices], self.diffusion_coeffs)
            
            # Update state using Euler step
            state_with_boundary = state_with_boundary + self.time_step * (
                -(1/2) * transport_term + 
                diffusion_term + 
                forcing_with_boundary
            )
        
        # Return state without boundary
        return state_with_boundary[..., 1:-1]

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
        Take a step in the environment using closed-loop simulation with sparse rewards
        
        Args:
            action: Forcing term to apply
            
        Returns:
            observation: Next state
            reward: Reward for the action (sparse, mostly given at the end)
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Store the action for validation
        self.actions_history.append(action)
        
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        # Simulate one step forward in time
        self.current_state = self.simulate(self.current_state, action_tensor)
        
        # Increment time
        self.current_time += 1
        
        # Check if the episode is done
        done = self.current_time >= self.num_time_points
        
        if done:
            # Use exp to transform the mse to [0, 1]
            reward = np.exp(-((self.current_state - self.target_state)**2).mean()).item()
        else:
            # For intermediate states, use small negative reward
            reward = 0.0
        
        # Convert to numpy array for gym interface
        observation = self.current_state.squeeze().numpy()
        
        info = {
            "target_state": self.target_state.squeeze().numpy(),
            "time_step": self.current_time,
            "error": ((self.current_state - self.target_state)**2).mean().item(),
        }
        
        return observation, reward, done, False, info
        
    def validate_against_trajectory(self):
        """
        Validate the closed-loop simulation against the trajectory-based simulation
        
        Returns:
            bool: Whether the simulations match
        """
        if not self.actions_history:
            return False
            
        # Create a forcing terms tensor with the correct dimensions
        forcing_terms = torch.zeros(1, self.num_time_points, self.spatial_size)
        
        # Fill in the actions we've taken so far
        for i, act in enumerate(self.actions_history):
            forcing_terms[0, i] = torch.tensor(act, dtype=torch.float32)
        
        # Simulate using the trajectory-based approach
        trajectory = burgers_solver(
            self.initial_state, 
            forcing_terms, 
            num_time_points=self.num_time_points
        )
        
        # Get the current state from the trajectory
        trajectory_state = trajectory[:, self.current_time]
        
        # Compare with our closed-loop simulation state
        error = ((self.current_state - trajectory_state)**2).mean().item()
        
        print(f"Validation error: {error}")
        return error < 1e-6  # Small threshold for numerical precision issues

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
def make_burgers_env_closedloop(test_dataset_path=test_file_path, viscosity=0.01, num_time_points=None):
    """
    Create a closed-loop Burgers equation environment instance with sparse rewards
    
    Args:
        test_dataset_path: Path to test data file
        viscosity: Viscosity coefficient
        num_time_points: Number of time points for simulation (if None, uses length from dataset)
        
    Returns:
        BurgersEnvClosedLoop: A closed-loop Burgers equation environment with sparse rewards
    """
    return BurgersEnvClosedLoop(test_dataset_path, viscosity, num_time_points) 