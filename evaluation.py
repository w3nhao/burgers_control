import torch
import math
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import tqdm
from functools import partial

def create_differential_matrices_1d(grid_size, device='cpu'):
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
                             time_step=1e-4, num_time_points=10, mode=None):
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
        mode: Simulation mode (must not be 'const')
        
    Returns:
        tensor: Simulated trajectories (N, num_time_points+1, s)
    """
    if mode != 'const':
        assert forcing_terms.size()[1] == num_time_points, 'Number of time intervals must match forcing term dimensions'
    else:
        raise ValueError("'const' mode is not supported")

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
    
    first_deriv, second_deriv = create_differential_matrices_1d(spatial_size + 2, device=initial_conditions.device)
    
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
    for step in tqdm.trange(total_steps, desc="Simulating Burgers' equation"):
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
