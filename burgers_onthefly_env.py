import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from typing import Dict, Tuple, Any, Optional, List, Union

import torch.nn.functional as F

# Import functions from burgers.py
from burgers import (
    create_differential_matrices_1d,
    make_initial_conditions_and_varying_forcing_terms,
    simulate_burgers_equation
)

class BurgersOnTheFlyVecEnv(VectorEnv):
    """
    Vectorized Burgers Equation Environment that generates data on-the-fly.
    Instead of using pregenerated data, this environment creates new initial conditions
    and targets on each reset by simulating the system forward.
    """
    
    def __init__(self, 
                 num_envs: int = 1,
                 spatial_size: int = 128,
                 num_time_points: int = 10,
                 viscosity: float = 0.01,
                 sim_time: float = 0.1,
                 time_step: float = 1e-4,
                 forcing_terms_scaling_factor: float = 1.0,
                 reward_type: str = "vanilla",
                 mse_scaling_factor: float = 1e3
                 ):
        """
        Initialize the vectorized on-the-fly Burgers Equation Environment
        
        Args:
            num_envs: Number of parallel environments
            spatial_size: Number of spatial points
            num_time_points: Number of time points for simulation
            viscosity: Viscosity coefficient
            sim_time: Total physical simulation time
            time_step: Physical simulation time step size
            forcing_terms_scaling_factor: Scaling factor for forcing terms
            reward_type: Type of reward function to use (vanilla, inverse_mse, exp_scaled_mse)
            mse_scaling_factor: Scaling factor for MSE reward
        """
        self.spatial_size = spatial_size
        self.num_time_points = num_time_points
        self.reward_type = reward_type
        self.mse_scaling_factor = mse_scaling_factor
        
        # Define action and observation spaces
        single_action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.spatial_size,), dtype=np.float32
        )
        # Observation space now includes both current state and target state
        single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.spatial_size,), dtype=np.float32
        )
        
        # Initialize the VectorEnv with proper spaces
        super().__init__(
            num_envs=num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space
        )
        
        # Environment parameters
        self.viscosity = viscosity
        self.sim_time = sim_time
        self.time_step = time_step
        self.forcing_terms_scaling_factor = forcing_terms_scaling_factor
        self.current_time = np.zeros(num_envs, dtype=int)
        
        # Define domain and step sizes
        self.domain_min = 0.0
        self.domain_max = 1.0
        self.spatial_step = (self.domain_max - self.domain_min) / (self.spatial_size + 1)
        
        # Time step for simulation matches the total simulation time divided by num_time_points
        self.simulation_dt = self.sim_time / self.num_time_points
        
        # Steps per simulation time step
        self.steps_per_sim = int(self.simulation_dt / self.time_step)
        
        # Setup differential matrices for finite difference method
        self.first_deriv, self.second_deriv = create_differential_matrices_1d(
            self.spatial_size + 2
        )
        
        # Prepare the differential matrices for efficient computation
        self._prepare_differential_matrices()
        
        # Current state and episode info - batched for all environments
        self.current_state = None  # Will be tensor of shape [num_envs, spatial_size]
        self.target_state = None   # Will be tensor of shape [num_envs, spatial_size]
        self.initial_state = None  # Will be tensor of shape [num_envs, spatial_size]
        
        # Track actions for validation - list of lists, one per environment
        self.actions_history = [[] for _ in range(num_envs)]
        
        # Episode tracking for final info records
        self.cumulative_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=int)
        
        # Using CPU by default, can be moved to GPU if needed
        self.device = torch.device("cpu")
        
        # Random generator for reproducibility
        self.rng = np.random.RandomState()

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
        for sim_step in range(self.steps_per_sim):
            # Remove boundary values and repad to maintain consistent size
            states_with_boundary = states_with_boundary[..., 1:-1]
            states_with_boundary = F.pad(states_with_boundary, (1, 1))
            
            # Calculate nonlinear transport and diffusion terms
            squared_state = states_with_boundary**2
            transport_term = torch.einsum('nsi,si->ns', squared_state[..., self.transport_indices], self.transport_coeffs)
            diffusion_term = torch.einsum('nsi,si->ns', states_with_boundary[..., self.diffusion_indices], self.diffusion_coeffs)
            
            # Calculate update
            update_term = self.time_step * (
                -(1/2) * transport_term + 
                diffusion_term + 
                forcing_with_boundary
            )
            
            # Update state using Euler step
            states_with_boundary = states_with_boundary + update_term
        
        # Return state without boundary
        result = states_with_boundary[..., 1:-1]
        return result

    def reset(self, 
              mask: Optional[np.ndarray] = None, 
              *, 
              seed: Optional[Union[int, List[int]]] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset specified environments or all environments if mask is None.
        Generates new initial conditions and target states by simulating forward.
        
        Args:
            mask: Boolean mask of which environments to reset (True = reset)
            seed: Random seed(s)
            options: Additional options for reset
            
        Returns:
            observations: Initial states for reset environments
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            if isinstance(seed, int):
                self.rng = np.random.RandomState(seed)
            else:
                self.rng = np.random.RandomState(seed[0] if len(seed) > 0 else None)
        
        # If this is the first reset, initialize all environments
        if self.current_state is None:
            self.current_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            self.target_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            self.initial_state = torch.zeros((self.num_envs, self.spatial_size), device=self.device)
            mask = np.ones(self.num_envs, dtype=bool)
        else:
            # Normalize mask to reset only specified environments
            mask = self._normalize_mask(mask)
        
        # Count environments to reset
        num_to_reset = np.sum(mask)
        
        if num_to_reset > 0:
            # Generate initial conditions and forcing terms for environments to reset
            initial_conditions, forcing_terms = make_initial_conditions_and_varying_forcing_terms(
                num_to_reset, num_to_reset, self.spatial_size, self.num_time_points,
                scaling_factor=self.forcing_terms_scaling_factor, max_time=self.sim_time
            )
            
            # Move to device
            initial_conditions = initial_conditions.to(self.device)
            forcing_terms = forcing_terms.to(self.device)
            
            # Simulate forward to get target states
            trajectories = simulate_burgers_equation(
                initial_conditions, forcing_terms, self.viscosity, self.sim_time,
                time_step=self.time_step, num_time_points=self.num_time_points,
            )
            
            # Extract target states (final state of simulation)
            target_states = trajectories[:, -1, :]
            
            # Update states for environments being reset
            reset_idx = 0
            for i in range(self.num_envs):
                if mask[i]:
                    self.current_state[i] = initial_conditions[reset_idx]
                    self.initial_state[i] = initial_conditions[reset_idx]
                    self.target_state[i] = target_states[reset_idx]
                    
                    # Reset actions history
                    self.actions_history[i] = []
                    
                    # Reset time counter
                    self.current_time[i] = 0
                    
                    # Reset episode tracking
                    self.cumulative_rewards[i] = 0.0
                    self.episode_lengths[i] = 0
                    
                    reset_idx += 1
        
        # Convert to numpy for gym interface - must move to CPU first
        current_states = self.current_state.cpu().numpy()
        target_states = self.target_state.cpu().numpy()
        
        # Concatenate current state and target state for goal-conditioned observation
        observations = np.concatenate([current_states, target_states], axis=1)
        
        # Prepare info dict
        info = {
            "target_state": target_states,
            "initial_state": self.initial_state.cpu().numpy(),
        }
        
        return observations, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Take a step in all environments using provided actions
        
        Args:
            actions: Actions to take in each environment with shape [num_envs, spatial_size]
            
        Returns:
            observations: Next states (auto-reset for terminated environments)
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
        
        if self.reward_type == "vanilla":
            # Calculate rewards as negative MSE
            # This naturally maps MSE to (0, 1] range without manual scaling:
            # - MSE = 0 → reward = 0.0 (perfect match)
            # - MSE = 1 → reward = -1.0
            # - MSE = 9 → reward = -9.0
            # - MSE → ∞ → reward → -∞
            rewards = -((self.current_state - self.target_state)**2).mean(dim=1).cpu().numpy()
        elif self.reward_type == "inverse_mse":
            # Calculate rewards using inverse relationship: reward = 1 / (1 + MSE)
            # This naturally maps MSE to (0, 1] range without manual scaling:
            # - MSE = 0 → reward = 1.0 (perfect match)
            # - MSE = 1 → reward = 0.5 
            # - MSE = 9 → reward = 0.1
            # - MSE → ∞ → reward → 0
            mse_per_env = ((self.current_state - self.target_state)**2).mean(dim=1)  # Shape: [num_envs]
            rewards = (1.0 / (1.0 + mse_per_env)).cpu().numpy()  # Shape: [num_envs]
        elif self.reward_type == "exp_scaled_mse":
            # Calculate rewards using exponential relationship: reward = exp(-MSE * scaling_factor)
            # This naturally maps MSE to (0, 1] range without manual scaling:
            # - MSE = 0 → reward = 1.0 (perfect match)
            # - MSE = 1 → reward = 0.36787944117144233
            # - MSE = 9 → reward = 0.00012340980408667956
            # - MSE → ∞ → reward → 0
            mse_per_env = ((self.current_state - self.target_state)**2).mean(dim=1)  # Shape: [num_envs]
            rewards = torch.exp(-mse_per_env * self.mse_scaling_factor).cpu().numpy()  # Shape: [num_envs]
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        
        # Update episode tracking
        self.cumulative_rewards += rewards
        self.episode_lengths += 1
        
        # No truncations in this environment as mentioned in the requirements
        truncations = np.zeros(self.num_envs, dtype=bool)
        
        # Calculate errors for info
        errors = ((self.current_state - self.target_state)**2).mean(dim=1).cpu().numpy()
        
        # Convert states to numpy for gym interface - must move to CPU first
        current_states = self.current_state.cpu().numpy()
        target_states = self.target_state.cpu().numpy()
        
        # Concatenate current state and target state for goal-conditioned observation
        observations = np.concatenate([current_states, target_states], axis=1)
        
        # Prepare info dict
        info = {
            "current_state": current_states,
            "target_state": target_states,
            "time_step": self.current_time.copy(),
            "error": errors,
        }
        
        # Add final info for terminated environments and auto-reset
        if np.any(terminations):
            final_info = []
            for i in range(self.num_envs):
                if terminations[i]:
                    # Record final episode statistics
                    episode_info = {
                        "episode": {
                            "r": self.cumulative_rewards[i],  # Total episode reward in numpy float32
                            "l": self.episode_lengths[i],     # Episode length in numpy int32
                            "final_error": errors[i],         # Final MSE error in numpy float32
                            "final_mse": errors[i],           # Same as final_error (compatibility) in numpy float32
                            "initial_state": self.initial_state[i].cpu().numpy().copy(),
                            "final_state": self.current_state[i].cpu().numpy().copy(),
                            "target_state": self.target_state[i].cpu().numpy().copy(),
                            "initial_vs_target_mse": ((self.initial_state[i] - self.target_state[i])**2).mean().cpu().numpy(),
                            "final_vs_target_mse": errors[i], # Same as final_error (compatibility) in numpy float32
                            "improvement": ((self.initial_state[i] - self.target_state[i])**2).mean().cpu().numpy() - errors[i]
                        }
                    }
                    final_info.append(episode_info)
                else:
                    final_info.append(None)
            info["final_info"] = final_info
            
            # Auto-reset terminated environments
            reset_obs, reset_info = self.reset(mask=terminations)
            
            # Update observations for terminated environments with reset observations
            for i in range(self.num_envs):
                if terminations[i]:
                    observations[i] = reset_obs[i]
            
            # Update info with reset information for terminated environments
            info["target_state"] = self.target_state.cpu().numpy()  # Update with new targets
            info["time_step"] = self.current_time.copy()  # Update with reset time steps
        
        # Reset episode tracking for terminated environments (already done in reset)
        # Note: reset() already handles resetting cumulative_rewards and episode_lengths
        
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
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(seed[0] if len(seed) > 0 else None)


# Factory function to create the environment
def make_burgers_onthefly_vec_env(
    num_envs=1,
    spatial_size=64,
    num_time_points=10,
    viscosity=0.01,
    sim_time=0.1,
    time_step=1e-4,
    forcing_terms_scaling_factor=1.0
):
    """
    Create a vectorized Burgers equation environment with on-the-fly data generation
    
    Args:
        num_envs: Number of parallel environments
        spatial_size: Number of spatial points
        num_time_points: Number of time points for simulation
        viscosity: Viscosity coefficient
        sim_time: Total physical simulation time
        time_step: Physical simulation time step size
        forcing_terms_scaling_factor: Scaling factor for forcing terms
        
    Returns:
        BurgersOnTheFlyVecEnv: A vectorized Burgers equation environment
    """
    return BurgersOnTheFlyVecEnv(
        num_envs=num_envs,
        spatial_size=spatial_size,
        num_time_points=num_time_points,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        forcing_terms_scaling_factor=forcing_terms_scaling_factor
    ) 