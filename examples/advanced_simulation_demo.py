import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from burgers import simulate_burgers_equation, create_differential_matrices_1d
from dataset import BurgersTest, test_file_path

def print_tensor_info(name, tensor):
    """Print detailed information about a tensor or array"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: Tensor shape {tensor.shape}, dtype {tensor.dtype}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: Array shape {tensor.shape}, dtype {tensor.dtype}")
    else:
        print(f"{name}: Type {type(tensor)}")

def visualize_simulations(simulations, titles, initial_state=None, forcing_terms=None, filename="advanced_simulation_comparison.png"):
    """Visualize multiple simulation results side by side"""
    num_sims = len(simulations)
    fig, axes = plt.subplots(2, num_sims, figsize=(5*num_sims, 10))
    
    # If only one simulation, reshape axes for consistent indexing
    if num_sims == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each simulation
    for i, (trajectory, title) in enumerate(zip(simulations, titles)):
        # Plot the evolution of states for each simulation
        for t in range(trajectory.shape[1]):
            alpha = 0.2 + 0.8 * (t / trajectory.shape[1])
            label = f"t={t}" if t == 0 or t == trajectory.shape[1]-1 else None
            axes[0, i].plot(trajectory[0, t].cpu().numpy(), alpha=alpha, label=label)
        
        if initial_state is not None:
            axes[0, i].plot(initial_state[0].cpu().numpy(), 'k--', label="Initial")
            
        axes[0, i].legend()
        axes[0, i].set_title(f"{title} - State Evolution")
        axes[0, i].set_xlabel("Spatial Position")
        axes[0, i].set_ylabel("State Value")
        
        # Plot the evolution of derivatives (spatial gradient)
        for t in range(trajectory.shape[1]):
            if t == 0 or t == trajectory.shape[1]-1:
                # Compute approximate spatial derivative using central differences
                state = trajectory[0, t].cpu().numpy()
                dx = 1.0 / (len(state) - 1)
                gradient = np.gradient(state, dx)
                label = f"t={t}"
                axes[1, i].plot(gradient, alpha=0.8, label=label)
        
        axes[1, i].legend()
        axes[1, i].set_title(f"{title} - Spatial Gradient")
        axes[1, i].set_xlabel("Spatial Position")
        axes[1, i].set_ylabel("Gradient Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join("examples", filename))
    plt.close()
    print(f"Visualization saved to 'examples/{filename}'")

if __name__ == "__main__":
    print("="*50)
    print("ADVANCED SIMULATION DEMONSTRATION")
    print("="*50)
    
    # Load test dataset
    test_dataset = BurgersTest(test_file_path)
    sample = test_dataset[0]
    initial_state = torch.tensor(sample['observations'][0]).float().unsqueeze(0)
    
    # Print initial state properties
    print_tensor_info("Initial state", initial_state)
    
    # Create varying forcing terms
    spatial_size = initial_state.shape[-1]
    batch_size = initial_state.shape[0]
    x = torch.linspace(0, 1, spatial_size)
    
    # 1. Zero forcing (natural evolution)
    zero_forcing = torch.zeros(batch_size, 10, spatial_size)
    
    # 2. Sinusoidal forcing
    t = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)  # Shape: [1, 10, 1]
    x_expanded = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, spatial_size]
    sin_forcing = 0.05 * torch.sin(2 * np.pi * x_expanded) * torch.sin(2 * np.pi * t)
    
    # 3. Gaussian pulse forcing (localized in space and time)
    x_mid = spatial_size // 2
    x_gaussian = torch.exp(-0.5 * ((x - 0.5) / 0.05)**2)
    t_gaussian = torch.exp(-0.5 * ((torch.linspace(0, 1, 10) - 0.2) / 0.1)**2)
    gaussian_forcing = torch.zeros(batch_size, 10, spatial_size)
    for i in range(10):
        gaussian_forcing[0, i] = 0.1 * x_gaussian * t_gaussian[i]
    
    # Simulation parameters to explore
    viscosities = [0.001, 0.01, 0.1]  # Low, medium, high viscosity
    time_steps = [1e-4, 1e-5]  # Different numerical time steps
    forcing_types = [zero_forcing, sin_forcing, gaussian_forcing]
    forcing_names = ["Zero Forcing", "Sinusoidal Forcing", "Gaussian Pulse"]
    
    print("\n1. EXPLORING DIFFERENT VISCOSITIES")
    viscosity_sims = []
    viscosity_titles = []
    
    for visc in viscosities:
        print(f"\nRunning simulation with viscosity = {visc}")
        trajectory = simulate_burgers_equation(
            initial_state, 
            zero_forcing,
            viscosity=visc,
            sim_time=0.1,
            time_step=1e-4,
            num_time_points=10,
            mode='non-const'
        )
        viscosity_sims.append(trajectory)
        viscosity_titles.append(f"Viscosity = {visc}")
        print_tensor_info(f"Trajectory (viscosity = {visc})", trajectory)
    
    # Visualize viscosity comparison
    visualize_simulations(
        viscosity_sims, 
        viscosity_titles,
        initial_state=initial_state,
        filename="viscosity_comparison.png"
    )
    
    print("\n2. EXPLORING DIFFERENT FORCING TERMS")
    forcing_sims = []
    forcing_titles = []
    
    for i, (forcing, name) in enumerate(zip(forcing_types, forcing_names)):
        print(f"\nRunning simulation with {name}")
        trajectory = simulate_burgers_equation(
            initial_state, 
            forcing,
            viscosity=0.01,
            sim_time=0.1,
            time_step=1e-4,
            num_time_points=10,
            mode='non-const'
        )
        forcing_sims.append(trajectory)
        forcing_titles.append(name)
        print_tensor_info(f"Trajectory ({name})", trajectory)
    
    # Visualize forcing comparison
    visualize_simulations(
        forcing_sims, 
        forcing_titles,
        initial_state=initial_state,
        filename="forcing_comparison.png"
    )
    
    print("\n3. EXPLORING SHOCK FORMATION (Low viscosity, longer time)")
    # Create a special initial condition prone to shock formation
    shock_initial = torch.zeros(1, spatial_size)
    shock_initial[0, spatial_size//4:3*spatial_size//4] = 1.0  # Step function
    
    # Create zero forcing with the correct number of time points for shock simulation
    shock_num_time_points = 20
    shock_zero_forcing = torch.zeros(batch_size, shock_num_time_points, spatial_size)
    
    shock_trajectory = simulate_burgers_equation(
        shock_initial, 
        shock_zero_forcing,
        viscosity=0.001,  # Low viscosity to encourage shock formation
        sim_time=0.2,     # Longer simulation time
        time_step=1e-5,   # Smaller time step for stability
        num_time_points=shock_num_time_points,
        mode='non-const'
    )
    
    # Visualize shock formation
    visualize_simulations(
        [shock_trajectory], 
        ["Shock Formation"],
        initial_state=shock_initial,
        filename="shock_formation.png"
    )
    
    print("="*50)
    print("ADVANCED SIMULATION COMPLETE")
    print("="*50) 