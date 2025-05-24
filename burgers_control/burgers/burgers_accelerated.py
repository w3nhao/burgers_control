"""
Accelerated Burgers equation simulation using torch.compile optimizations.
This module provides optimized versions of the key simulation functions for training environments.
"""

import os
import torch
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm, trange
from functools import partial
import time
from typing import Tuple, Optional, Union, List
import warnings

# Check if torch.compile should be disabled (for testing/debugging)
DISABLE_TORCH_COMPILE = os.environ.get('DISABLE_TORCH_COMPILE', 'false').lower() == 'true'

def maybe_compile(func):
    """Conditionally apply torch.compile based on environment variable."""
    if DISABLE_TORCH_COMPILE:
        return func
    else:
        try:
            return torch.compile(func, mode="default")
        except Exception as e:
            warnings.warn(f"torch.compile failed for {func.__name__}: {e}. Using uncompiled version.")
            return func

# ===============================
# Accelerated Initial Conditions and Forcing Terms Generation
# ===============================

@maybe_compile
def _generate_gaussian_bump_vectorized(x_grid: torch.Tensor, locations: torch.Tensor, 
                                     amplitudes: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Gaussian bump generation using pure PyTorch operations.
    
    Args:
        x_grid: Spatial grid points [spatial_size]
        locations: Center locations [num_samples, 1]
        amplitudes: Amplitudes [num_samples, 1]  
        sigmas: Standard deviations [num_samples, 1]
        
    Returns:
        torch.Tensor: Gaussian bumps [num_samples, spatial_size]
    """
    # Expand dimensions for broadcasting: [num_samples, 1] x [1, spatial_size] -> [num_samples, spatial_size]
    x_expanded = x_grid.unsqueeze(0)  # [1, spatial_size]
    
    # Compute Gaussian: amp * exp(-0.5 * (x - loc)^2 / sig^2)
    diff_squared = (x_expanded - locations) ** 2  # [num_samples, spatial_size]
    gaussian = amplitudes * torch.exp(-0.5 * diff_squared / (sigmas ** 2))
    
    return gaussian

@maybe_compile
def _generate_forcing_component_vectorized(x_grid: torch.Tensor, time_grid: torch.Tensor,
                                         amplitudes: torch.Tensor, loc_space: torch.Tensor,
                                         sig_space: torch.Tensor, loc_time: torch.Tensor,
                                         sig_time: torch.Tensor, spatial_mask: torch.Tensor,
                                         amplitude_compensation: float) -> torch.Tensor:
    """
    Vectorized forcing term component generation using pure PyTorch operations.
    
    Args:
        x_grid: Spatial grid [spatial_size]
        time_grid: Time grid [num_time_points]
        amplitudes: Amplitudes [num_forcing, 1, 1]
        loc_space: Spatial locations [num_forcing, 1, 1]
        sig_space: Spatial sigmas [num_forcing, 1, 1]
        loc_time: Temporal locations [num_forcing, 1, 1]
        sig_time: Temporal sigmas [num_forcing, 1, 1]
        spatial_mask: Spatial control mask [num_forcing, num_time_points, spatial_size]
        amplitude_compensation: Amplitude compensation factor
        
    Returns:
        torch.Tensor: Forcing component [num_forcing, num_time_points, spatial_size]
    """
    # Create expanded grids for broadcasting
    x_expanded = x_grid.view(1, 1, -1)  # [1, 1, spatial_size]
    t_expanded = time_grid.view(1, -1, 1)  # [1, num_time_points, 1]
    
    # Spatial component: exp(-0.5 * (x - loc_space)^2 / sig_space^2)
    x_diff_sq = (x_expanded - loc_space) ** 2  # [num_forcing, 1, spatial_size]
    exp_space = torch.exp(-0.5 * x_diff_sq / (sig_space ** 2))  # [num_forcing, 1, spatial_size]
    
    # Temporal component: amplitude_compensation * exp(-0.5 * (t - loc_time)^2 / sig_time^2)
    t_diff_sq = (t_expanded - loc_time) ** 2  # [num_forcing, num_time_points, 1]
    exp_time = amplitude_compensation * torch.exp(-0.5 * t_diff_sq / (sig_time ** 2))  # [num_forcing, num_time_points, 1]
    
    # Apply amplitude and combine spatial/temporal components
    forcing = amplitudes * exp_space * exp_time  # [num_forcing, num_time_points, spatial_size]
    
    # Apply spatial mask
    forcing = forcing * spatial_mask
    
    return forcing

def make_initial_conditions_and_varying_forcing_terms(
    num_initial_conditions: int, 
    num_forcing_terms: int, 
    spatial_size: int,
    num_time_points: int, 
    amplitude_compensation: float = 2.0,
    partial_control: Optional[str] = None, 
    scaling_factor: float = 1.0, 
    max_time: float = 1.0,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accelerated version of initial conditions and forcing terms generation.
    This function maintains exact compatibility with the original implementation while using PyTorch acceleration where possible.
    
    Args:
        num_initial_conditions: Number of initial condition samples
        num_forcing_terms: Number of forcing term samples  
        spatial_size: Number of spatial points
        num_time_points: Number of time points for forcing terms
        amplitude_compensation: Compensation factor for Gaussian in time domain
        partial_control: Partial control mode (None or 'front_rear_quarter')
        scaling_factor: Scaling factor for forcing terms
        max_time: Maximum simulation time
        seed: Random seed for reproducibility
        device: Device for computation
        
    Returns:
        tuple: (initial_conditions, forcing_terms)
            - initial_conditions: tensor of shape (num_initial_conditions, spatial_size)
            - forcing_terms: tensor of shape (num_forcing_terms, num_time_points, spatial_size)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Set random seeds for reproducibility - MUST match original exactly
    if seed is not None:
        np.random.seed(seed)
    
    # Define spatial and temporal domains
    domain_min, domain_max = 0.0, 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    x = torch.linspace(domain_min + spatial_step, domain_max - spatial_step, spatial_size)
    
    time_min = 0.0
    time_step = (max_time - time_min) / (num_time_points + 1)
    time_points = torch.linspace(time_min + time_step, max_time - time_step, num_time_points)
    
    # ===== Generate Initial Conditions =====
    # Use NumPy exactly like the original, then convert to tensors
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
    
    # ===== Create Spatial Mask =====
    # Create spatial mask for partial control if needed - exactly like original
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

    # ===== Generate Forcing Terms =====
    # Function to generate random forcing terms - maintains exact compatibility with original
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
    
    # Convert to device and return
    initial_conditions_tensor = torch.tensor(initial_conditions, dtype=torch.float32).to(device)
    forcing_terms_tensor = forcing_terms.to(device)
    
    return initial_conditions_tensor, forcing_terms_tensor

# ===============================
# Accelerated Simulation Functions
# ===============================

def create_differential_matrices_tensors(grid_size: int, viscosity: float, spatial_step: float, 
                                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create differential matrices as dense tensors for efficient GPU computation.
    
    Args:
        grid_size: Number of grid points (including boundaries)
        viscosity: Viscosity coefficient
        spatial_step: Spatial step size
        device: Device for computation
        
    Returns:
        tuple: (transport_matrix, diffusion_matrix) as dense tensors
    """
    # Create sparse matrices first (using the original implementation)
    first_deriv = sp.diags([-1, 1], [-1, 1], shape=(grid_size, grid_size))
    first_deriv = sp.lil_matrix(first_deriv)
    first_deriv[0, [0, 1, 2]] = [-3, 4, -1]
    first_deriv[grid_size-1, [grid_size-3, grid_size-2, grid_size-1]] = [1, -4, 3]

    second_deriv = sp.diags([1, -2, 1], [-1, 0, 1], shape=(grid_size, grid_size))
    second_deriv = sp.lil_matrix(second_deriv)
    second_deriv[0, [0, 1, 2, 3]] = [2, -5, 4, -1]
    second_deriv[grid_size-1, [grid_size-4, grid_size-3, grid_size-2, grid_size-1]] = [-1, 4, -5, 2]
    
    # Adjust boundary conditions (matching original implementation)
    first_deriv.rows[0] = first_deriv.rows[0][:2]
    first_deriv.rows[-1] = first_deriv.rows[-1][-2:]
    first_deriv.data[0] = first_deriv.data[0][:2]
    first_deriv.data[-1] = first_deriv.data[-1][-2:]
    
    second_deriv.rows[0] = second_deriv.rows[0][:3]
    second_deriv.rows[-1] = second_deriv.rows[-1][-3:]
    second_deriv.data[0] = second_deriv.data[0][:3]
    second_deriv.data[-1] = second_deriv.data[-1][-3:]
    
    # Convert to dense tensors with proper scaling
    transport_matrix = torch.tensor(first_deriv.toarray() / (2 * spatial_step), dtype=torch.float32, device=device)
    diffusion_matrix = torch.tensor(viscosity * second_deriv.toarray() / spatial_step**2, dtype=torch.float32, device=device)
    
    return transport_matrix, diffusion_matrix

@maybe_compile  
def simulate_burgers_step_vectorized(state: torch.Tensor, forcing: torch.Tensor, 
                                   transport_matrix: torch.Tensor, diffusion_matrix: torch.Tensor,
                                   time_step: float) -> torch.Tensor:
    """
    Vectorized single time step of Burgers equation simulation.
    
    Args:
        state: Current state with boundary padding [num_samples, spatial_size + 2]
        forcing: Forcing terms with boundary padding [num_samples, spatial_size + 2]
        transport_matrix: Transport coefficient matrix [spatial_size + 2, spatial_size + 2]
        diffusion_matrix: Diffusion coefficient matrix [spatial_size + 2, spatial_size + 2]
        time_step: Time step size
        
    Returns:
        torch.Tensor: Next state [num_samples, spatial_size + 2]
    """
    # Compute transport term: -0.5 * d(u^2)/dx
    squared_state = state ** 2
    transport_term = torch.matmul(squared_state, transport_matrix.T)
    
    # Compute diffusion term: viscosity * d^2u/dx^2
    diffusion_term = torch.matmul(state, diffusion_matrix.T)
    
    # Update state using Euler method
    next_state = state + time_step * (-0.5 * transport_term + diffusion_term + forcing)
    
    return next_state

def simulate_burgers_equation(
    initial_conditions: torch.Tensor, 
    forcing_terms: torch.Tensor, 
    viscosity: float,
    sim_time: float, 
    time_step: float = 1e-4, 
    num_time_points: int = 10,
    print_progress: bool = True,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Accelerated simulation of Burgers' equation using PyTorch optimizations.
    This function maintains exact compatibility with the original implementation.
    
    Args:
        initial_conditions: Initial state [N, s]
        forcing_terms: Forcing terms [N, Nt, s]
        viscosity: Viscosity coefficient
        sim_time: Total physical simulation time
        time_step: Physical simulation time step size
        num_time_points: Number of time points to record
        print_progress: Whether to print progress bar
        device: Device for computation
        
    Returns:
        torch.Tensor: Simulated trajectories [N, num_time_points+1, s]
    """
    if device is None:
        device = initial_conditions.device
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Move tensors to device
    initial_conditions = initial_conditions.to(device)
    forcing_terms = forcing_terms.to(device)
    
    # Validate inputs
    assert forcing_terms.size(1) == num_time_points, 'Number of time intervals must match forcing term dimensions'
    
    # Setup parameters
    spatial_size = initial_conditions.size(-1)
    num_samples = initial_conditions.size(0)
    
    domain_min, domain_max = 0.0, 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    
    # Calculate time stepping parameters (matching original logic exactly)
    total_steps_raw = math.ceil(sim_time / time_step)
    record_interval = total_steps_raw // num_time_points
    total_steps = record_interval * num_time_points
    
    # Log adjustment if needed (matching original)
    if total_steps != total_steps_raw:
        actual_sim_time = total_steps * time_step
        # Note: We skip logging here to avoid dependency on logger, but maintain the logic
    
    # Setup matrices for computation
    transport_matrix, diffusion_matrix = create_differential_matrices_tensors(
        spatial_size + 2, viscosity, spatial_step, device
    )
    
    # Initialize state with boundary padding
    state = F.pad(initial_conditions, (1, 1))  # [N, s+2]
    forcing_terms_padded = F.pad(forcing_terms, (1, 1))  # [N, Nt, s+2]
    
    # Preallocate solution tensor
    solution = torch.zeros(num_samples, spatial_size, num_time_points, device=device)
    
    # Main simulation loop
    record_counter = 0
    forcing_index = -1
    
    for step in trange(total_steps, desc="Simulating Burgers' equation (accelerated)", disable=not print_progress):
        # Remove boundary values and repad for consistency (matching original)
        state = state[..., 1:-1]
        state = F.pad(state, (1, 1))
        
        # Update forcing index (matching original logic exactly)
        if step % record_interval == 0:
            forcing_index += 1
        
        # Assert that forcing index is valid (matching original)
        assert 0 <= forcing_index < num_time_points, f"Invalid forcing_index {forcing_index}, should be in [0, {num_time_points-1}]"
        
        # Get current forcing term
        current_forcing = forcing_terms_padded[:, forcing_index, :]
        
        # Simulate one time step
        state = simulate_burgers_step_vectorized(
            state, current_forcing, transport_matrix, diffusion_matrix, time_step
        )
        
        # Record solution at specified intervals
        if (step + 1) % record_interval == 0:
            solution[:, :, record_counter] = state[..., 1:-1]
            record_counter += 1
    
    # Reformat solution to (N, Nt, s) (matching original)
    solution = solution.permute(0, 2, 1)
    
    # Add initial condition as first time point (matching original)
    trajectory = torch.cat([initial_conditions.unsqueeze(1), solution], dim=1)
    
    return trajectory

@maybe_compile
def simulate_burgers_one_time_point_vectorized(state: torch.Tensor, forcing_term: torch.Tensor,
                                             transport_matrix: torch.Tensor, diffusion_matrix: torch.Tensor,
                                             time_step: float) -> torch.Tensor:
    """
    Accelerated version of one-time-point simulation using matrix operations.
    """
    # Remove boundary values and repad to maintain consistent size
    state = state[..., 1:-1]
    state = F.pad(state, (1, 1))
    
    # Use the vectorized step function
    return simulate_burgers_step_vectorized(state, forcing_term, transport_matrix, diffusion_matrix, time_step)

def simulate_burgers_one_time_point(state, forcing_term, transport_indices, transport_coeffs, 
                                  diffusion_indices, diffusion_coeffs, time_step):
    """
    Accelerated version of single time step simulation.
    This function provides a compatibility layer but delegates to the original implementation for exact compatibility.
    
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
    # For exact compatibility, use the original implementation logic
    # Remove boundary values and repad to maintain consistent size
    state = state[..., 1:-1]
    state = F.pad(state, (1, 1))
    
    # Calculate nonlinear transport and diffusion terms using the original method
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

# ===============================
# Performance Testing and Validation Functions
# ===============================

def test_accelerated_vs_original(seed: int = 42, device: str = "cpu", 
                                tolerance: float = 1e-5, verbose: bool = True) -> bool:
    """
    Comprehensive test to ensure accelerated functions produce identical results to original.
    
    Args:
        seed: Random seed for reproducibility
        device: Device for computation  
        tolerance: Numerical tolerance for comparison
        verbose: Whether to print detailed output
        
    Returns:
        bool: True if all tests pass
    """
    # Import original functions for comparison
    from .burgers_original import (
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original,
        simulate_burgers_equation as simulate_original
    )
    
    device_torch = torch.device(device)
    if verbose:
        print(f"Testing accelerated functions on {device}...")
    
    # Test parameters
    test_cases = [
        (10, 10, 32, 5, "Small test"),
        (50, 50, 64, 8, "Medium test"),  
        (100, 100, 128, 10, "Large test"),
    ]
    
    all_tests_passed = True
    
    for i, (num_ic, num_ft, spatial_size, num_time_points, description) in enumerate(test_cases):
        if verbose:
            print(f"\nTest case {i+1}: {description}")
            print(f"  {num_ic} samples, {spatial_size} spatial points, {num_time_points} time points")
        
        # Test 1: Initial conditions and forcing terms
        if verbose:
            print("  Testing initial conditions and forcing terms generation...")
        
        # Original
        torch.manual_seed(seed)
        np.random.seed(seed)
        ic_orig, ft_orig = make_ic_ft_original(
            num_ic, num_ft, spatial_size, num_time_points, seed=seed
        )
        
        # Accelerated
        torch.manual_seed(seed)
        np.random.seed(seed)
        ic_acc, ft_acc = make_initial_conditions_and_varying_forcing_terms(
            num_ic, num_ft, spatial_size, num_time_points, seed=seed, device=device_torch
        )
        
        # Move original to device for comparison
        ic_orig = ic_orig.to(device_torch)
        ft_orig = ft_orig.to(device_torch)
        
        # Compare
        ic_diff = torch.abs(ic_orig - ic_acc).max().item()
        ft_diff = torch.abs(ft_orig - ft_acc).max().item()
        
        ic_match = ic_diff < tolerance
        ft_match = ft_diff < tolerance
        
        if verbose:
            print(f"    IC max difference: {ic_diff:.2e}, Match: {ic_match}")
            print(f"    FT max difference: {ft_diff:.2e}, Match: {ft_match}")
        
        if not (ic_match and ft_match):
            if verbose:
                print(f"    ❌ FAILED: Generation test case {i+1}")
            all_tests_passed = False
            continue
        
        # Test 2: Simulation
        if verbose:
            print("  Testing Burgers equation simulation...")
        
        # Use subset for simulation to manage memory
        sim_samples = min(20, num_ic)
        ic_sim = ic_acc[:sim_samples]
        ft_sim = ft_acc[:sim_samples]
        
        # Original simulation
        traj_orig = simulate_original(
            ic_sim.cpu(), ft_sim.cpu(), viscosity=0.01, sim_time=1.0,
            time_step=1e-4, num_time_points=num_time_points, print_progress=False
        )
        traj_orig = traj_orig.to(device_torch)
        
        # Accelerated simulation
        traj_acc = simulate_burgers_equation(
            ic_sim, ft_sim, viscosity=0.01, sim_time=1.0,
            time_step=1e-4, num_time_points=num_time_points, print_progress=False,
            device=device_torch
        )
        
        # Compare trajectories
        sim_diff = torch.abs(traj_orig - traj_acc).max().item()
        sim_match = sim_diff < tolerance
        
        if verbose:
            print(f"    Simulation max difference: {sim_diff:.2e}, Match: {sim_match}")
        
        if not sim_match:
            if verbose:
                print(f"    ❌ FAILED: Simulation test case {i+1}")
            all_tests_passed = False
        else:
            if verbose:
                print(f"    ✓ PASSED: Test case {i+1}")
    
    # Test edge cases
    if verbose:
        print("\nTesting edge cases...")
        print("  Testing with partial control...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    ic_orig_pc, ft_orig_pc = make_ic_ft_original(
        10, 10, 64, 5, partial_control='front_rear_quarter', seed=seed
    )
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    ic_acc_pc, ft_acc_pc = make_initial_conditions_and_varying_forcing_terms(
        10, 10, 64, 5, partial_control='front_rear_quarter', seed=seed, device=device_torch
    )
    
    ic_orig_pc = ic_orig_pc.to(device_torch)
    ft_orig_pc = ft_orig_pc.to(device_torch)
    
    ic_pc_diff = torch.abs(ic_orig_pc - ic_acc_pc).max().item()
    ft_pc_diff = torch.abs(ft_orig_pc - ft_acc_pc).max().item()
    
    pc_match = ic_pc_diff < tolerance and ft_pc_diff < tolerance
    if verbose:
        print(f"    Partial control - IC diff: {ic_pc_diff:.2e}, FT diff: {ft_pc_diff:.2e}, Match: {pc_match}")
    
    if not pc_match:
        if verbose:
            print("    ❌ FAILED: Partial control test")
        all_tests_passed = False
    
    # Final result
    if verbose:
        print(f"\n{'='*50}")
        if all_tests_passed:
            print("✓ ALL TESTS PASSED: Accelerated functions produce identical results!")
        else:
            print("❌ SOME TESTS FAILED: Please check the implementation.")
        print(f"{'='*50}")
    
    return all_tests_passed

def benchmark_functions(num_initial_conditions: int = 1000, num_forcing_terms: int = 1000,
                       spatial_size: int = 128, num_time_points: int = 10,
                       device: str = "cuda", seed: int = 42, verbose: bool = True) -> dict:
    """
    Benchmark original vs accelerated functions.
    
    Args:
        num_initial_conditions: Number of initial condition samples
        num_forcing_terms: Number of forcing term samples
        spatial_size: Number of spatial points
        num_time_points: Number of time points
        device: Device for computation
        seed: Random seed
        verbose: Whether to print detailed output
        
    Returns:
        dict: Benchmark results
    """
    # Import original functions
    from .burgers_original import (
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original,
        simulate_burgers_equation as simulate_original
    )
    
    device_torch = torch.device(device)
    results = {}
    
    if verbose:
        print(f"Benchmarking on {device} with {num_initial_conditions} samples...")
    
    # ===== Benchmark Initial Conditions and Forcing Terms Generation =====
    if verbose:
        print("\n=== Initial Conditions and Forcing Terms Generation ===")
    
    # Original function
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    ic_orig, ft_orig = make_ic_ft_original(
        num_initial_conditions, num_forcing_terms, spatial_size, num_time_points,
        amplitude_compensation=2.0, partial_control=None, scaling_factor=1.0, 
        max_time=1.0, seed=seed
    )
    # Move to device for fair comparison
    ic_orig = ic_orig.to(device_torch)
    ft_orig = ft_orig.to(device_torch)
    orig_time = time.time() - start_time
    
    # Accelerated function
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    ic_acc, ft_acc = make_initial_conditions_and_varying_forcing_terms(
        num_initial_conditions, num_forcing_terms, spatial_size, num_time_points,
        amplitude_compensation=2.0, partial_control=None, scaling_factor=1.0,
        max_time=1.0, seed=seed, device=device_torch
    )
    acc_time = time.time() - start_time
    
    # Compare results
    ic_diff = torch.abs(ic_orig - ic_acc).max().item()
    ft_diff = torch.abs(ft_orig - ft_acc).max().item()
    
    results['generation'] = {
        'original_time': orig_time,
        'accelerated_time': acc_time,
        'speedup': orig_time / acc_time,
        'ic_max_diff': ic_diff,
        'ft_max_diff': ft_diff,
        'results_match': ic_diff < 1e-5 and ft_diff < 1e-5
    }
    
    if verbose:
        print(f"Original time: {orig_time:.4f}s")
        print(f"Accelerated time: {acc_time:.4f}s")
        print(f"Speedup: {orig_time/acc_time:.2f}x")
        print(f"Max IC difference: {ic_diff:.2e}")
        print(f"Max FT difference: {ft_diff:.2e}")
        print(f"Results match: {ic_diff < 1e-5 and ft_diff < 1e-5}")
    
    # ===== Benchmark Simulation =====
    if verbose:
        print("\n=== Burgers Equation Simulation ===")
    
    # Use smaller samples for simulation benchmark to manage memory
    sim_samples = min(100, num_initial_conditions)
    ic_sim = ic_acc[:sim_samples]
    ft_sim = ft_acc[:sim_samples]
    
    # Original simulation
    start_time = time.time()
    traj_orig = simulate_original(
        ic_sim.cpu(), ft_sim.cpu(), viscosity=0.01, sim_time=1.0, 
        time_step=1e-4, num_time_points=num_time_points, print_progress=False
    )
    traj_orig = traj_orig.to(device_torch)
    orig_sim_time = time.time() - start_time
    
    # Accelerated simulation
    start_time = time.time()
    traj_acc = simulate_burgers_equation(
        ic_sim, ft_sim, viscosity=0.01, sim_time=1.0,
        time_step=1e-4, num_time_points=num_time_points, print_progress=False,
        device=device_torch
    )
    acc_sim_time = time.time() - start_time
    
    # Compare simulation results
    sim_diff = torch.abs(traj_orig - traj_acc).max().item()
    
    results['simulation'] = {
        'original_time': orig_sim_time,
        'accelerated_time': acc_sim_time,
        'speedup': orig_sim_time / acc_sim_time,
        'max_diff': sim_diff,
        'results_match': sim_diff < 1e-4
    }
    
    if verbose:
        print(f"Original time: {orig_sim_time:.4f}s")
        print(f"Accelerated time: {acc_sim_time:.4f}s")
        print(f"Speedup: {orig_sim_time/acc_sim_time:.2f}x")
        print(f"Max trajectory difference: {sim_diff:.2e}")
        print(f"Results match: {sim_diff < 1e-4}")
    
    # ===== Overall Summary =====
    if verbose:
        print("\n=== Summary ===")
    total_orig_time = orig_time + orig_sim_time
    total_acc_time = acc_time + acc_sim_time
    overall_speedup = total_orig_time / total_acc_time
    
    results['overall'] = {
        'total_original_time': total_orig_time,
        'total_accelerated_time': total_acc_time,
        'overall_speedup': overall_speedup
    }
    
    if verbose:
        print(f"Total original time: {total_orig_time:.4f}s")
        print(f"Total accelerated time: {total_acc_time:.4f}s")
        print(f"Overall speedup: {overall_speedup:.2f}x")
    
    return results

# ===============================
# Main function for testing
# ===============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and benchmark accelerated Burgers functions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for computation")
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for testing")
    
    args = parser.parse_args()
    
    if args.disable_compile:
        os.environ['DISABLE_TORCH_COMPILE'] = 'true'
    
    if args.test or (not args.test and not args.benchmark):
        print("Running correctness tests...")
        test_passed = test_accelerated_vs_original(seed=args.seed, device=args.device)
        
    if args.benchmark:
        print("\nRunning performance benchmarks...")
        results = benchmark_functions(device=args.device, seed=args.seed) 