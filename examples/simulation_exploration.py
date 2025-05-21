import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from evaluation import create_differential_matrices_1d, simulate_burgers_equation, burgers_solver, evaluate_model_performance
from dataset import get_squence_data, train_file_path, test_file_path, BurgersTest

def print_tensor_info(name, tensor):
    """Print detailed information about a tensor or array"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: Tensor shape {tensor.shape}, dtype {tensor.dtype}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: Array shape {tensor.shape}, dtype {tensor.dtype}")
    else:
        print(f"{name}: Type {type(tensor)}")

print("="*50)
print("EXPLORING SIMULATION FUNCTIONS")
print("="*50)

# Create and inspect differential matrices
grid_size = 128 + 2  # Including boundary points
first_deriv, second_deriv = create_differential_matrices_1d(grid_size)
print(f"First derivative matrix type: {type(first_deriv)}")
print(f"First derivative matrix shape: {first_deriv.shape}")
print(f"Second derivative matrix type: {type(second_deriv)}")
print(f"Second derivative matrix shape: {second_deriv.shape}")

# Load some sample data for simulation
test_dataset = BurgersTest(test_file_path)
sample = test_dataset[0]
# Convert NumPy arrays to PyTorch tensors if needed
initial_state = torch.tensor(sample['observations'][0]).float().unsqueeze(0)  # Add batch dimension
print_tensor_info("Initial state", initial_state)

# Create dummy forcing terms for simulation
num_time_points = 10
spatial_size = initial_state.shape[-1]
batch_size = initial_state.shape[0]
dummy_forcing = torch.zeros(batch_size, num_time_points, spatial_size)
print_tensor_info("Forcing terms", dummy_forcing)

# Run simulation
print("\nRunning Burgers equation simulation...")
trajectory = simulate_burgers_equation(
    initial_state, 
    dummy_forcing,
    viscosity=0.01,
    sim_time=0.1,
    time_step=1e-4,
    num_time_points=num_time_points,
    mode='non-const'
)
print_tensor_info("Simulation trajectory", trajectory)
print(f"  - First time point matches initial: {torch.allclose(trajectory[:, 0], initial_state)}")
print(f"  - Number of time points: {trajectory.shape[1]} (expected {num_time_points+1})")

# Demonstrate using the default burgers_solver
print("\nRunning default burgers_solver...")
default_trajectory = burgers_solver(initial_state, dummy_forcing, num_time_points=num_time_points)
print_tensor_info("Default solver trajectory", default_trajectory)

# Demonstrate evaluation function
print("\nDemonstrating evaluation function...")
target_state = trajectory[:, -1].clone()  # Use last state as target for demonstration
evaluation_result = evaluate_model_performance(
    num_episodes=num_time_points,
    initial_state=initial_state,
    target_state=target_state,
    actions=dummy_forcing,
    device='cpu'
)
print(f"Evaluation MSE: {evaluation_result}")

print("="*50)
print("SIMULATION EXPLORATION COMPLETE")
print("="*50) 