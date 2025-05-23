import torch
import math
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import tqdm
from functools import partial

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
    for step in tqdm.trange(total_steps, desc="Simulating Burgers' equation", disable=not print_progress):
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
                                           partial_control=None, scaling_factor=1.0, max_time=1.0):
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
        
    Returns:
        tuple: (initial_conditions, forcing_terms)
            - initial_conditions: tensor of shape (num_initial_conditions, spatial_size)
            - forcing_terms: tensor of shape (num_forcing_terms, num_time_points, spatial_size)
    """
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

def test_one_time_point_simulation():
    """
    Tests that the one-time-point simulation produces the same results as the full simulation
    when applied sequentially.
    """
    # Parameters
    num_samples = 2
    spatial_size = 64
    num_time_points = 5
    viscosity = 0.01
    sim_time = 1.0
    time_step = 1e-4
    
    # Generate initial conditions and forcing terms
    initial_conditions, forcing_terms = make_initial_conditions_and_varying_forcing_terms(
        num_samples, num_samples, spatial_size, num_time_points, scaling_factor=1.0, max_time=sim_time
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
    print(f"Maximum difference between full and step-by-step simulation: {difference}")
    
    assert difference < 1e-5, "Step-by-step simulation does not match full simulation"
    print("Test passed: Step-by-step simulation matches full simulation")
    
    return full_simulation, step_by_step_simulation

if __name__ == "__main__":
    test_one_time_point_simulation()