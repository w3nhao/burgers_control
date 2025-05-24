import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm, trange
from functools import partial
import os
from datasets import Dataset, load_from_disk
import argparse
import random
import logging
import sys
from datetime import datetime

try:
    import h5py
except ImportError:
    print("You are not using h5py as we are not going to support h5py in the future.")
    pass


BURGERS_TRAIN_FILE_PATH = os.getenv("BURGERS_TRAIN_FILE_PATH")
BURGERS_TEST_FILE_PATH = os.getenv("BURGERS_TEST_FILE_PATH")

# ===============================
# Dataset loading utils
# ===============================

def discounted_cumsum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = torch.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

def get_squence_data(file_path=BURGERS_TRAIN_FILE_PATH):
    # Load dataset using Hugging Face datasets
    try:
        dataset = load_from_disk(file_path)
        dataset.set_format("torch")
        
        # Extract data from the dataset
        u_data = dataset['trajectories'][:40000]  # Limit to 40k as before
        f_data = dataset['actions'][:40000]
        
    except Exception as e:
        print(f"Warning: Could not load dataset from {file_path}. Error: {e}")
        print("Falling back to HDF5 format...")
        # Fallback to HDF5 if datasets format doesn't exist
        with h5py.File(file_path + ".h5", 'r') as hdf:
            print("Keys: ", list(hdf.keys()))
            u_data = torch.tensor(hdf['train']['pde_11-128'][:40000])
            f_data = torch.tensor(hdf['train']['pde_11-128_f'][:40000])
        
    # [s_0, s_1, s_2, ..., s_n - 1]
    # [a_0, a_1, a_2, ..., a_n - 1]
    # [r_1, r_2, r_3, ..., r_n]
    # [c_1, c_2, c_3, ..., c_n]
        
    rewards = -(u_data[:, -1].unsqueeze(1) - u_data[:, 1:]).square().mean(-1)
    
    terminals = np.zeros(rewards.shape, dtype=np.bool_)
    terminals[:, -1] = True

    data = dict(
        observations=u_data[:, :-1].numpy(),
        actions=f_data.numpy(),
        rewards=rewards.numpy(),
        targets=u_data[:, -1].numpy(),
    )
    return data

def get_test_data(file_path=BURGERS_TEST_FILE_PATH):
    # Load dataset using Hugging Face datasets
    try:
        dataset = load_from_disk(file_path)
        dataset.set_format("torch")
        
        # Extract data from the dataset
        u_test = dataset['trajectories']
        
    except Exception as e:
        print(f"Warning: Could not load dataset from {file_path}. Error: {e}")
        print("Falling back to HDF5 format...")
        # Fallback to HDF5 if datasets format doesn't exist
        with h5py.File(file_path + ".h5", 'r') as hdf:
            u_test = torch.tensor(hdf['test'][:])
    
    rewards = -((u_test[:, -1][:, None, :] - u_test[:, 1:]) ** 2).mean(-1)
    observations = u_test[:, :-1]
    targets = u_test[:, -1]
        
    data = dict(
        observations=observations.numpy(),
        actions=[None] * len(observations),
        rewards=rewards.numpy(),
        targets=targets.numpy(),
    )
    return data

class BurgersDataset(torch.utils.data.Dataset):
    def __init__(self, mode: str):
        assert mode in ["train", "test"]
        if mode == "train":
            self.data = get_squence_data(BURGERS_TRAIN_FILE_PATH)
        elif mode == "test":
            self.data = get_test_data(BURGERS_TEST_FILE_PATH)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def __len__(self):
        return len(self.data['observations'])
    
    def __getitem__(self, idx):
        observations = self.data['observations'][idx]
        actions = self.data['actions'][idx]
        rewards = self.data['rewards'][idx]
        targets = self.data['targets'][idx]
        
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            targets=targets,
        )

# ===============================
# Simulation utils
# ===============================

def create_differential_matrices_1d(grid_size):
    """
    Creates first and second derivative matrices for 1D finite difference approximations.
    
    Args:
        grid_size (int): Number of grid points
        device (str): Device to place tensors on ('cpu' or 'cuda')
        
    Returns:
        tuple: (first_derivative_matrix, second_derivative_matrix)
    """
    # First derivative matrix
    first_deriv = sp.diags([-1, 1], [-1, 1], shape=(grid_size, grid_size))  # Division by (2*dx) required later
    first_deriv = sp.lil_matrix(first_deriv)
    first_deriv[0, [0, 1, 2]] = [-3, 4, -1]                     # 2nd order forward difference
    first_deriv[grid_size-1, [grid_size-3, grid_size-2, grid_size-1]] = [1, -4, 3]  # 2nd order backward difference

    # Second derivative matrix
    second_deriv = sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size, grid_size))  # Division by dx^2 required
    second_deriv = sp.lil_matrix(second_deriv)
    second_deriv[0, [0, 1, 2, 3]] = [2, -5, 4, -1]                      # 2nd order forward difference
    second_deriv[grid_size-1, [grid_size-4, grid_size-3, grid_size-2, grid_size-1]] = [-1, 4, -5, 2]  # 2nd order backward difference
    
    return first_deriv, second_deriv

def simulate_burgers_equation(initial_conditions, forcing_terms, viscosity, sim_time, 
                             time_step=1e-4, num_time_points=10, print_progress=False):
    """
    Simulates Burgers' equation with forcing terms.
    
    Args:
        initial_conditions: (N, s) - Initial state for N samples with s spatial points
        forcing_terms: (N, Nt, s) - Forcing terms for N samples over Nt time points
        viscosity: Viscosity coefficient
        sim_time: Total physical simulation time
        time_step: Physical simulation time step size
        num_time_points: Number of time points to sample/record during the simulation.
                        The simulation uses much smaller time_step internally but only records
                        the state at evenly spaced intervals. The output will have num_time_points+1
                        time points (including the initial condition).
        print_progress: Whether to print progress bar
        
    Returns:
        tensor: Simulated trajectories (N, num_time_points+1, s)
    """
    assert forcing_terms.size()[1] == num_time_points, 'Number of time intervals must match forcing term dimensions'

    # Grid size
    spatial_size = initial_conditions.size(-1)
    num_time_steps = forcing_terms.size(1)

    num_initial = initial_conditions.size(0)
    num_forcing = forcing_terms.size(0)
    assert num_initial == num_forcing, "Number of initial conditions must match number of forcing terms"
    num_samples = num_forcing
     
    domain_min = 0.0
    domain_max = 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)

    # Number of steps to final time
    total_steps = math.ceil(sim_time / time_step)

    state = initial_conditions.reshape(num_samples, spatial_size)
    state = F.pad(state, (1, 1))  # Add boundary padding
    forcing_terms = forcing_terms.reshape(num_samples, num_time_steps, spatial_size)
    forcing_terms = F.pad(forcing_terms, (1, 1))  # Add boundary padding
    
    # Record solution every this number of steps
    record_interval = math.floor(total_steps / num_time_steps)
    
    first_deriv, second_deriv = create_differential_matrices_1d(spatial_size + 2)
    
    # Adjust boundary conditions for the matrices
    first_deriv.rows[0] = first_deriv.rows[0][:2]
    first_deriv.rows[-1] = first_deriv.rows[-1][-2:]
    first_deriv.data[0] = first_deriv.data[0][:2]
    first_deriv.data[-1] = first_deriv.data[-1][-2:]
    
    second_deriv.rows[0] = second_deriv.rows[0][:3]
    second_deriv.rows[-1] = second_deriv.rows[-1][-3:]
    second_deriv.data[0] = second_deriv.data[0][:3]
    second_deriv.data[-1] = second_deriv.data[-1][-3:]
    
    # Convert sparse matrices to tensor format for computation
    transport_indices = list(first_deriv.rows)
    transport_coeffs = torch.FloatTensor(np.stack(first_deriv.data) / (2 * spatial_step)).to(initial_conditions.device)
    diffusion_indices = list(second_deriv.rows)
    diffusion_coeffs = torch.FloatTensor(viscosity * np.stack(second_deriv.data) / spatial_step**2).to(initial_conditions.device)
    
    # Tensor for storing solution
    solution = torch.zeros(num_samples, spatial_size, num_time_steps, device=initial_conditions.device)
    
    # Counters and time tracking
    record_counter = 0
    physical_time = 0.0
    forcing_index = -1
    
    # Main simulation loop
    for step in trange(total_steps, desc="Simulating Burgers' equation", disable=not print_progress):
        # Remove boundary values and repad to maintain consistent size
        state = state[..., 1:-1]
        state = F.pad(state, (1, 1))
        
        # Calculate nonlinear transport and diffusion terms
        squared_state = state**2
        transport_term = torch.einsum('nsi,si->ns', squared_state[..., transport_indices], transport_coeffs)
        diffusion_term = torch.einsum('nsi,si->ns', state[..., diffusion_indices], diffusion_coeffs)
        
        # Update forcing index
        if step % record_interval == 0:
            forcing_index += 1
            
        # Update state using Euler step
        state = state + time_step * (
            -(1/2) * transport_term + 
            diffusion_term + 
            forcing_terms[:, forcing_index, :]
        )
        
        # Update physical time
        physical_time += time_step

        # Record solution at specified intervals
        if (step + 1) % record_interval == 0:
            solution[..., record_counter] = state[..., 1:-1]
            record_counter += 1

    # Reformat solution to (N, Nt, s)
    solution = solution.permute(0, 2, 1)
    
    # Add initial condition as first time point
    trajectory = torch.cat((initial_conditions.reshape(num_samples, 1, spatial_size), solution), dim=1)
    return trajectory

def simulate_burgers_one_time_point(state, forcing_term, transport_indices, transport_coeffs, 
                              diffusion_indices, diffusion_coeffs, time_step):
    """
    Simulates Burgers' equation for a single time step.
    
    Args:
        state (tensor): Current state with shape (N, s)
        forcing_term (tensor): Current forcing term with shape (N, s)
        transport_indices (list): Indices for transport term calculation
        transport_coeffs (tensor): Coefficients for transport term calculation
        diffusion_indices (list): Indices for diffusion term calculation
        diffusion_coeffs (tensor): Coefficients for diffusion term calculation
        time_step (float): Physical time step size
        
    Returns:
        tensor: Next state after one time step
    """
    # Remove boundary values and repad to maintain consistent size
    state = state[..., 1:-1]
    state = F.pad(state, (1, 1))
    
    # Calculate nonlinear transport and diffusion terms
    squared_state = state**2
    transport_term = torch.einsum('nsi,si->ns', squared_state[..., transport_indices], transport_coeffs)
    diffusion_term = torch.einsum('nsi,si->ns', state[..., diffusion_indices], diffusion_coeffs)
    
    # Update state using Euler step
    next_state = state + time_step * (
        -(1/2) * transport_term + 
        diffusion_term + 
        forcing_term
    )
    
    return next_state

def make_initial_conditions_and_varying_forcing_terms(num_initial_conditions, num_forcing_terms, spatial_size, 
                                           num_time_points, amplitude_compensation=2, 
                                           partial_control=None, scaling_factor=1.0, max_time=1.0, seed=None):
    """
    Generates initial conditions and time-varying forcing terms for Burgers' equation.
    
    Args:
        num_initial_conditions (int): Number of initial condition samples
        num_forcing_terms (int): Number of forcing term samples
        spatial_size (int): Number of spatial points
        num_time_points (int): Number of time points for forcing terms
        amplitude_compensation (float): Compensation factor for Gaussian in time domain
        partial_control (str): Partial control mode (None or 'front_rear_quarter')
        scaling_factor (float): Scaling factor for forcing terms
        max_time (float): Maximum simulation time
        seed (int): Random seed for reproducibility (optional)
        
    Returns:
        tuple: (initial_conditions, forcing_terms)
            - initial_conditions: tensor of shape (num_initial_conditions, spatial_size)
            - forcing_terms: tensor of shape (num_forcing_terms, num_time_points, spatial_size)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    # Define spatial domain
    domain_min = 0.0
    domain_max = 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    x = torch.linspace(domain_min + spatial_step, domain_max - spatial_step, spatial_size)

    # Define time domain
    time_min = 0.0
    time_step = (max_time - time_min) / (num_time_points + 1)
    time_points = torch.linspace(time_min + time_step, max_time - time_step, num_time_points)
    
    # Generate initial conditions with two Gaussian bumps
    # First Gaussian bump (positive)
    loc1 = np.random.uniform(0.2, 0.4, (num_initial_conditions, 1))
    amp1 = np.random.uniform(0, 2, (num_initial_conditions, 1))
    sig1 = np.random.uniform(0.05, 0.15, (num_initial_conditions, 1))
    gauss1 = amp1 * np.exp(-0.5 * (np.array(x.view(1, -1).repeat(num_initial_conditions, 1)) - loc1)**2 / sig1**2)

    # Second Gaussian bump (negative)
    loc2 = np.random.uniform(0.6, 0.8, (num_initial_conditions, 1))
    amp2 = np.random.uniform(-2, 0, (num_initial_conditions, 1))
    sig2 = np.random.uniform(0.05, 0.15, (num_initial_conditions, 1))
    gauss2 = amp2 * np.exp(-0.5 * (np.array(x.view(1, -1).repeat(num_initial_conditions, 1)) - loc2)**2 / sig2**2)

    # Combine the two Gaussian bumps
    initial_conditions = gauss1 + gauss2
    
    # Create spatial mask for partial control if needed
    if partial_control is None:
        spatial_mask = np.ones_like(x.view(1, 1, -1).repeat(num_forcing_terms, num_time_points, 1))
    elif partial_control == 'front_rear_quarter':
        spatial_mask = np.zeros_like(x.view(1, 1, -1).repeat(num_forcing_terms, num_time_points, 1))
        controllable_idx = np.hstack((
            np.arange(0, spatial_size // 4), 
            np.arange(3 * spatial_size // 4, spatial_size)
        ))
        spatial_mask[:, :, controllable_idx] = 1.0
        # Increase amplitude compensation for partial control
        amplitude_compensation *= 2
    else:
        raise ValueError('Invalid partial control mode')

    # Function to generate random forcing terms
    def generate_random_forcing(random_amplitude):
        if random_amplitude:
            amp = np.random.randint(2, size=(num_forcing_terms, 1, 1)) * np.random.uniform(-1.5, 1.5, (num_forcing_terms, 1, 1))
        else:
            amp = np.random.uniform(-1.5, 1.5, (num_forcing_terms, 1, 1))
        amp = torch.tensor(amp).repeat(1, num_time_points, spatial_size)

        # Spatial component
        loc_space = np.random.uniform(0, 1, (num_forcing_terms, 1, 1))
        sig_space = np.random.uniform(0.1, 0.4, (num_forcing_terms, 1, 1)) * 0.5
        exp_space = np.exp(-0.5 * (np.array(x.view(1, 1, -1).repeat(num_forcing_terms, num_time_points, 1)) - loc_space)**2 / sig_space**2)
        exp_space = exp_space * spatial_mask

        # Temporal component
        loc_time = np.random.uniform(0, 1, (num_forcing_terms, 1, 1))
        sig_time = np.random.uniform(0.1, 0.4, (num_forcing_terms, 1, 1)) * 0.5
        exp_time = amplitude_compensation * np.exp(-0.5 * (np.array(time_points.view(1, -1, 1).repeat(num_forcing_terms, 1, spatial_size)) - loc_time)**2 / sig_time**2)
        
        return amp * exp_space * exp_time
    
    # Generate forcing terms by summing multiple random components
    num_components = 7
    # Start with one non-random amplitude component to prevent zero forcing terms
    forcing_terms = generate_random_forcing(random_amplitude=False)
    for _ in range(num_components):
        forcing_terms += generate_random_forcing(random_amplitude=True)
    
    forcing_terms = forcing_terms.to(torch.float32)

    # Apply scaling factor if needed
    if scaling_factor != 1.0:
        forcing_terms = (forcing_terms * scaling_factor).clamp(-10.0, 10.0)
    
    return torch.tensor(initial_conditions, dtype=torch.float32), forcing_terms

# ===============================
# Data generation utils
# ===============================

def generate_training_data(num_trajectories=100000, num_time_points=10, spatial_size=128,
                          viscosity=0.01, sim_time=0.1, time_step=1e-4, seed=None, 
                          train_file_path=None):
    """
    Generate training data with initial conditions, intermediate states, and actions.
    
    Args:
        num_trajectories: Number of training trajectories to generate
        num_time_points: Number of time points in each trajectory
        spatial_size: Number of spatial grid points
        viscosity: Viscosity parameter
        sim_time: Total simulation time
        time_step: Time step for simulation
        seed: Random seed for reproducibility
        train_file_path: Path to save the training data
        
    Returns:
        tuple: (u_data, f_data) where
            - u_data: State trajectories (N, T+1, spatial_size)
            - f_data: Forcing terms (N, T, spatial_size)
    """
    log_info(f"Generating {num_trajectories} training trajectories...")
    log_info(f"Parameters: viscosity={viscosity}, sim_time={sim_time}, time_step={time_step}")
    log_info(f"Spatial size: {spatial_size}, Time points: {num_time_points}")
    if seed is not None:
        log_info(f"Random seed: {seed}")
    if train_file_path is not None:
        log_info(f"Will save training data to: {train_file_path}")
    
    # Generate initial conditions and forcing terms
    initial_conditions, forcing_terms = make_initial_conditions_and_varying_forcing_terms(
        num_initial_conditions=num_trajectories,
        num_forcing_terms=num_trajectories,
        spatial_size=spatial_size,
        num_time_points=num_time_points,
        scaling_factor=1.0,
        max_time=sim_time,
        seed=seed
    )
    
    log_info(f"Initial conditions shape: {initial_conditions.shape}")
    log_info(f"Forcing terms shape: {forcing_terms.shape}")
    
    # Run simulations in batches to manage memory
    batch_size = 1000  # Process 1000 trajectories at a time
    all_trajectories = []
    
    for batch_start in tqdm(range(0, num_trajectories, batch_size), desc="Simulating batches"):
        batch_end = min(batch_start + batch_size, num_trajectories)
        
        batch_initial = initial_conditions[batch_start:batch_end]
        batch_forcing = forcing_terms[batch_start:batch_end]
        
        # Run simulation for this batch
        batch_trajectory = simulate_burgers_equation(
            batch_initial,
            batch_forcing,
            viscosity=viscosity,
            sim_time=sim_time,
            time_step=time_step,
            num_time_points=num_time_points,
            print_progress=False
        )
        
        all_trajectories.append(batch_trajectory.cpu())
    
    # Concatenate all batches
    u_data = torch.cat(all_trajectories, dim=0)
    f_data = forcing_terms
    
    log_info(f"Generated training data:")
    log_info(f"  - u_data shape: {u_data.shape}")
    log_info(f"  - f_data shape: {f_data.shape}")
    
    # Save data if path is provided
    if train_file_path is not None:
        save_training_data_hf(u_data, f_data, train_file_path)
    
    return u_data, f_data

def generate_test_data(num_trajectories=50, num_time_points=10, spatial_size=128,
                      viscosity=0.01, sim_time=0.1, time_step=1e-4, seed=None,
                      test_file_path=None):
    """
    Generate test data with only initial and final states (no actions).
    
    Args:
        num_trajectories: Number of test trajectories to generate
        num_time_points: Number of time points in simulation
        spatial_size: Number of spatial grid points
        viscosity: Viscosity parameter
        sim_time: Total simulation time
        time_step: Time step for simulation
        seed: Random seed for reproducibility
        test_file_path: Path to save the test data
        
    Returns:
        torch.Tensor: Test trajectories (N, T+1, spatial_size) containing full trajectories
    """
    log_info(f"\nGenerating {num_trajectories} test trajectories...")
    if seed is not None:
        log_info(f"Random seed: {seed}")
    if test_file_path is not None:
        log_info(f"Will save test data to: {test_file_path}")
    
    # Generate initial conditions and forcing terms
    initial_conditions, forcing_terms = make_initial_conditions_and_varying_forcing_terms(
        num_initial_conditions=num_trajectories,
        num_forcing_terms=num_trajectories,
        spatial_size=spatial_size,
        num_time_points=num_time_points,
        scaling_factor=1.0,
        max_time=sim_time,
        seed=seed
    )
    
    # Run simulation
    test_trajectories = simulate_burgers_equation(
        initial_conditions,
        forcing_terms,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        num_time_points=num_time_points,
        print_progress=True
    )
    
    log_info(f"Generated test data shape: {test_trajectories.shape}")
    
    # Save data if path is provided
    if test_file_path is not None:
        save_test_data_hf(test_trajectories, test_file_path)
    
    return test_trajectories

def save_training_data_hf(u_data, f_data, file_path):
    """Save training data using Hugging Face datasets."""
    log_info(f"\nSaving training data to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert to numpy for datasets
    u_data_np = u_data.numpy()
    f_data_np = f_data.numpy()
    
    # Create dataset dictionary
    dataset_dict = {
        'trajectories': u_data_np,
        'actions': f_data_np,
        'num_trajectories': [u_data.shape[0]] * u_data.shape[0],
        'num_time_points': [u_data.shape[1] - 1] * u_data.shape[0],  # Excluding initial condition
        'spatial_size': [u_data.shape[2]] * u_data.shape[0],
        'viscosity': [0.01] * u_data.shape[0],
        'sim_time': [0.1] * u_data.shape[0],
        'time_step': [1e-4] * u_data.shape[0]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    dataset.save_to_disk(file_path)
    
    log_info(f"Saved training data with shape: {u_data.shape}")

def save_test_data_hf(test_data, file_path):
    """Save test data using Hugging Face datasets."""
    log_info(f"\nSaving test data to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert to numpy for datasets
    test_data_np = test_data.numpy()
    
    # Create dataset dictionary
    dataset_dict = {
        'trajectories': test_data_np,
        'num_trajectories': [test_data.shape[0]] * test_data.shape[0],
        'num_time_points': [test_data.shape[1] - 1] * test_data.shape[0],  # Excluding initial condition
        'spatial_size': [test_data.shape[2]] * test_data.shape[0],
        'viscosity': [0.01] * test_data.shape[0],
        'sim_time': [0.1] * test_data.shape[0],
        'time_step': [1e-4] * test_data.shape[0]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    dataset.save_to_disk(file_path)
    
    log_info(f"Saved test data with shape: {test_data.shape}")

def generate_small_dataset_for_testing(seed=42, train_file_path=None, test_file_path=None,
                                      log_file_path=None):
    """Generate a small test dataset for validation."""
    # Setup logging
    if log_file_path is None:
        log_file_path = f"burgers_small_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, actual_log_path = setup_logging(log_file_path, "small_dataset")
    
    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    # Small dataset parameters
    num_train_trajectories = 100
    num_test_trajectories = 10
    num_time_points = 10
    spatial_size = 128
    viscosity = 0.01
    sim_time = 0.1
    time_step = 1e-4
    
    # Default file paths if not provided
    if train_file_path is None:
        train_file_path = "../1d_burgers/burgers_train_small"
    if test_file_path is None:
        test_file_path = "../1d_burgers/unsafe_test_small"
    
    log_info("="*50)
    log_info("GENERATING SMALL TEST DATASET")
    log_info("="*50)
    log_info(f"Training trajectories: {num_train_trajectories}")
    log_info(f"Test trajectories: {num_test_trajectories}")
    log_info(f"Parameters: viscosity={viscosity}, sim_time={sim_time}, time_step={time_step}")
    log_info(f"Random seed: {seed}")
    log_info(f"Train file path: {train_file_path}")
    log_info(f"Test file path: {test_file_path}")
    log_info(f"Log file: {actual_log_path}")
    
    # Generate training data
    u_data, f_data = generate_training_data(
        num_trajectories=num_train_trajectories,
        num_time_points=num_time_points,
        spatial_size=spatial_size,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        seed=seed,
        train_file_path=train_file_path
    )
    
    # Generate test data
    test_data = generate_test_data(
        num_trajectories=num_test_trajectories,
        num_time_points=num_time_points,
        spatial_size=spatial_size,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        seed=seed + 1000,  # Different seed for test data
        test_file_path=test_file_path
    )
    
    log_info("\n" + "="*50)
    log_info("SMALL DATASET GENERATION COMPLETE")
    log_info("="*50)
    log_info(f"Training data shape: {u_data.shape}")
    log_info(f"Training actions shape: {f_data.shape}")
    log_info(f"Test data shape: {test_data.shape}")
    log_info(f"Files saved:")
    log_info(f"  - {train_file_path}")
    log_info(f"  - {test_file_path}")
    log_info(f"  - Log: {actual_log_path}")
    
    return train_file_path, test_file_path, actual_log_path

def generate_full_dataset(seed=42, train_file_path=None, test_file_path=None,
                         log_file_path=None):
    """Generate the full production dataset."""
    # Setup logging
    if log_file_path is None:
        log_file_path = f"burgers_full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, actual_log_path = setup_logging(log_file_path, "full_dataset")
    
    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    # Data generation parameters
    num_train_trajectories = 100000  # Full 1e5 trajectories
    num_test_trajectories = 50
    num_time_points = 10
    spatial_size = 128
    viscosity = 0.01
    sim_time = 0.1
    time_step = 1e-4
    
    # Default file paths if not provided
    if train_file_path is None:
        train_file_path = "../1d_burgers/burgers_train_new"
    if test_file_path is None:
        test_file_path = "../1d_burgers/unsafe_test_new"
    
    log_info("="*60)
    log_info("GENERATING NEW BURGERS EQUATION DATASET")
    log_info("="*60)
    log_info(f"Training trajectories: {num_train_trajectories}")
    log_info(f"Test trajectories: {num_test_trajectories}")
    log_info(f"Time points: {num_time_points}")
    log_info(f"Spatial size: {spatial_size}")
    log_info(f"Random seed: {seed}")
    log_info(f"Train file path: {train_file_path}")
    log_info(f"Test file path: {test_file_path}")
    log_info(f"Log file: {actual_log_path}")
    log_info(f"Simulation parameters:")
    log_info(f"  - Viscosity: {viscosity}")
    log_info(f"  - Simulation time: {sim_time}")
    log_info(f"  - Time step: {time_step}")
    log_info("="*60)
    
    # Generate training data
    u_data, f_data = generate_training_data(
        num_trajectories=num_train_trajectories,
        num_time_points=num_time_points,
        spatial_size=spatial_size,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        seed=seed,
        train_file_path=train_file_path
    )
    
    # Generate test data
    test_data = generate_test_data(
        num_trajectories=num_test_trajectories,
        num_time_points=num_time_points,
        spatial_size=spatial_size,
        viscosity=viscosity,
        sim_time=sim_time,
        time_step=time_step,
        seed=seed + 1000,  # Different seed for test data
        test_file_path=test_file_path
    )
    
    log_info("\n" + "="*60)
    log_info("DATA GENERATION COMPLETE")
    log_info("="*60)
    log_info(f"New training data saved to: {train_file_path}")
    log_info(f"New test data saved to: {test_file_path}")
    log_info(f"Generation log saved to: {actual_log_path}")
    log_info("\n" + "="*50)
    log_info("NEXT STEPS")
    log_info("="*50)
    log_info("1. The dataset paths in burgers.py are already updated to use the new data")
    log_info("2. You can now train your PPO agent with consistent data")
    log_info("3. Run environment validation with: python eval_on_testset.py")
    log_info("4. Train your agent with the new consistent dataset")
    
    return train_file_path, test_file_path, actual_log_path

# ===============================
# Evaluation and testing utils
# ===============================

# Default solver with pre-set parameters
burgers_solver = partial(simulate_burgers_equation, viscosity=0.01, sim_time=0.1, time_step=1e-4)

def evaluate_model_performance(num_episodes, initial_state, target_state, actions, device):
    """
    Evaluates model performance by comparing predicted states to target states.
    
    Args:
        num_episodes (int): Number of evaluation episodes
        initial_state (tensor): Initial state of the system
        target_state (tensor): Target state to reach
        actions (tensor): Actions/forcing terms to apply
        device (str): Device to run computation on
        
    Returns:
        float: Mean squared error between final state and target state
    """
    initial_state = initial_state.float().to(device)
    target_state = target_state.float().to(device)
    
    # Simulate system dynamics with the provided actions
    simulation_results = burgers_solver(initial_state, actions, num_time_points=num_episodes)
    final_state = simulation_results[:, -1]
    
    # Calculate mean squared error
    mse_per_sample = (final_state - target_state).square().mean(-1)
    mean_mse = mse_per_sample.mean()
    
    # Print evaluation metrics
    # r for per-sample MSE
    # J_actual_mse for average MSE
    print(f"Per-sample MSE: {mse_per_sample}")
    print(f"Average MSE: {mean_mse}")
    
    return mean_mse

def test_one_time_point_simulation(seed=42):
    """
    Tests that the one-time-point simulation produces the same results as the full simulation
    when applied sequentially.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        
    # Parameters
    num_samples = 2
    spatial_size = 64
    num_time_points = 5
    viscosity = 0.01
    sim_time = 1.0
    time_step = 1e-4
    
    # Generate initial conditions and forcing terms
    initial_conditions, forcing_terms = make_initial_conditions_and_varying_forcing_terms(
        num_samples, num_samples, spatial_size, num_time_points, scaling_factor=1.0, max_time=sim_time, seed=seed
    )
    
    # Run full simulation
    full_simulation = simulate_burgers_equation(
        initial_conditions, forcing_terms, viscosity, sim_time, 
        time_step=time_step, num_time_points=num_time_points
    )
    
    # Set up for step-by-step simulation
    domain_min = 0.0
    domain_max = 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    
    first_deriv, second_deriv = create_differential_matrices_1d(spatial_size + 2)
    
    # Adjust boundary conditions for the matrices
    first_deriv.rows[0] = first_deriv.rows[0][:2]
    first_deriv.rows[-1] = first_deriv.rows[-1][-2:]
    first_deriv.data[0] = first_deriv.data[0][:2]
    first_deriv.data[-1] = first_deriv.data[-1][-2:]
    
    second_deriv.rows[0] = second_deriv.rows[0][:3]
    second_deriv.rows[-1] = second_deriv.rows[-1][-3:]
    second_deriv.data[0] = second_deriv.data[0][:3]
    second_deriv.data[-1] = second_deriv.data[-1][-3:]
    
    # Convert sparse matrices to tensor format
    transport_indices = list(first_deriv.rows)
    transport_coeffs = torch.FloatTensor(np.stack(first_deriv.data) / (2 * spatial_step)).to(initial_conditions.device)
    diffusion_indices = list(second_deriv.rows)
    diffusion_coeffs = torch.FloatTensor(viscosity * np.stack(second_deriv.data) / spatial_step**2).to(initial_conditions.device)
    
    # Step-by-step simulation
    state = initial_conditions.reshape(num_samples, spatial_size)
    state = F.pad(state, (1, 1))  # Add boundary padding
    forcing_terms_padded = F.pad(forcing_terms, (1, 1))  # Add boundary padding
    
    step_by_step_results = [initial_conditions.clone()]
    
    # Number of simulation steps per recorded time point
    total_steps = math.ceil(sim_time / time_step)
    record_interval = math.floor(total_steps / num_time_points)
    
    # Main simulation loop
    for time_idx in range(num_time_points):
        for step in range(record_interval):
            # Get current forcing term
            current_forcing = forcing_terms_padded[:, time_idx, :]
            
            # Advance simulation by one time step
            state = simulate_burgers_one_time_point(
                state, current_forcing, transport_indices, transport_coeffs,
                diffusion_indices, diffusion_coeffs, time_step
            )
        
        # Record state at this time point
        step_by_step_results.append(state[..., 1:-1].clone())
    
    # Convert to tensor with shape (N, T, s)
    step_by_step_simulation = torch.stack(step_by_step_results, dim=1)
    
    # Compare results
    difference = torch.abs(full_simulation - step_by_step_simulation).max().item()
    log_info(f"Maximum difference between full and step-by-step simulation: {difference}")
    
    assert difference < 1e-5, "Step-by-step simulation does not match full simulation"
    log_info("Test passed: Step-by-step simulation matches full simulation")
    
    return full_simulation, step_by_step_simulation


# ===============================
# Logging utils
# ===============================

def setup_logging(log_file_path=None, mode="generation"):
    """
    Set up logging to capture all output to both console and file.
    
    Args:
        log_file_path (str): Path to save the log file
        mode (str): Mode description for log file naming
    """
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"burgers_{mode}_{timestamp}.log"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else ".", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('burgers_generation')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger, log_file_path

def log_info(message):
    """Helper function to log info messages."""
    logger = logging.getLogger('burgers_generation')
    if logger.hasHandlers():
        logger.info(message)
    else:
        print(message)
        
# ===============================
# Random seed utils
# ===============================

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    log_info(f"Random seeds set to: {seed}")

# ===============================
# Main function
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Burgers equation simulation and data generation")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "small", "full"],
                       help="Mode: 'test' runs simulation test, 'small' generates small dataset, 'full' generates full dataset")
    parser.add_argument("--validate", action="store_true", 
                       help="Run environment validation after generating small dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--train_file", type=str, default=None,
                       help="Path to save training data (default: auto-generated)")
    parser.add_argument("--test_file", type=str, default=None,
                       help="Path to save test data (default: auto-generated)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path to save generation log (default: auto-generated)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        # Run the original test
        log_file_path = args.log_file or f"burgers_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger, actual_log_path = setup_logging(log_file_path, "test")
        
        log_info("Running simulation validation test...")
        # Set seed for test reproducibility
        set_random_seeds(args.seed)
        test_one_time_point_simulation(args.seed)
        log_info(f"Test completed successfully. Log saved to: {actual_log_path}")
        
    elif args.mode == "small":
        # Generate small dataset for testing
        train_file, test_file, log_file = generate_small_dataset_for_testing(
            seed=args.seed,
            train_file_path=args.train_file,
            test_file_path=args.test_file,
            log_file_path=args.log_file
        )
        
        if args.validate:
            # Test with environment check
            log_info("\n" + "="*50)
            log_info("TESTING GENERATED DATA WITH ENVIRONMENT CHECK")
            log_info("="*50)
            
            # Temporarily update the paths
            original_train_path = BURGERS_TRAIN_FILE_PATH
            original_test_path = BURGERS_TEST_FILE_PATH
            
            # Update global variables
            globals()['BURGERS_TRAIN_FILE_PATH'] = train_file
            globals()['BURGERS_TEST_FILE_PATH'] = test_file
            
            try:
                # Test with our environment check (need to import here to avoid circular imports)
                sys.path.append('.')
                from eval_on_testset import test_environment_with_training_data
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                mean_mse, all_mse_values = test_environment_with_training_data(
                    device=device,
                    num_trajectories=5,  # Test with 5 trajectories
                    num_time_points=10,
                    viscosity=0.01,
                    sim_time=0.1,
                    time_step=1e-4
                )
                
                log_info(f"\nEnvironment check result: Mean MSE = {mean_mse:.10f}")
                if mean_mse < 1e-10:
                    log_info("✓ SUCCESS: Generated data is perfectly consistent!")
                else:
                    log_info("❌ WARNING: Generated data may have issues")
                    
            except ImportError:
                log_info("Note: eval_on_testset.py not found, skipping validation")
            finally:
                # Restore original paths
                globals()['BURGERS_TRAIN_FILE_PATH'] = original_train_path
                globals()['BURGERS_TEST_FILE_PATH'] = original_test_path
                
    elif args.mode == "full":
        # Generate full production dataset
        train_file, test_file, log_file = generate_full_dataset(
            seed=args.seed,
            train_file_path=args.train_file,
            test_file_path=args.test_file,
            log_file_path=args.log_file
        )
        
    log_info("\nDone!")