#!/usr/bin/env python3
"""
Test script for evaluating trained PPO agents on pre-generated test dataset.

This script loads a saved agent and tests it on 50 trajectories from the test dataset,
using one-step simulation for environment evolution and calculating the final mean MSE.

Usage:
    python test_agent_on_dataset.py --checkpoint_path checkpoints/run_name/agent_final.pt
"""

import argparse
import torch
import numpy as np
import math
import h5py
from ppo import load_saved_agent
from burgers import (
    create_differential_matrices_1d, 
    simulate_burgers_one_time_point,
    get_test_data,
    BURGERS_TEST_FILE_PATH
)

def setup_simulation_matrices(spatial_size, viscosity, device):
    """
    Set up differential matrices for one-step simulation.
    
    Args:
        spatial_size (int): Number of spatial points
        viscosity (float): Viscosity coefficient
        device (torch.device): Device for computation
        
    Returns:
        tuple: (transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, time_step, spatial_step)
    """
    domain_min = 0.0
    domain_max = 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    
    # Create differential matrices
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
    transport_coeffs = torch.FloatTensor(np.stack(first_deriv.data) / (2 * spatial_step)).to(device)
    diffusion_indices = list(second_deriv.rows)
    diffusion_coeffs = torch.FloatTensor(viscosity * np.stack(second_deriv.data) / spatial_step**2).to(device)
    
    return transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, spatial_step

def test_agent_on_dataset(agent, device, num_trajectories=50, num_time_points=10, 
                         viscosity=0.01, sim_time=0.1, time_step=1e-4):
    """
    Test agent on pre-generated test dataset using one-step simulation.
    
    Args:
        agent: Trained PPO agent
        device: Device for computation
        num_trajectories: Number of test trajectories to evaluate
        num_time_points: Number of time points in simulation
        viscosity: Viscosity parameter
        sim_time: Total simulation time
        time_step: Time step for simulation
        
    Returns:
        float: Mean MSE over all test trajectories
    """
    # Load test data
    test_data = get_test_data(BURGERS_TEST_FILE_PATH)
    
    # Extract initial states and target states for the first num_trajectories
    initial_states = test_data['observations'][:num_trajectories, 0, :]  # Shape: (N, spatial_size)
    target_states = test_data['targets'][:num_trajectories, :]           # Shape: (N, spatial_size)
    
    spatial_size = initial_states.shape[1]
    print(f"Testing on {num_trajectories} trajectories")
    print(f"Initial states shape: {initial_states.shape}")
    print(f"Target states shape: {target_states.shape}")
    print(f"Spatial size: {spatial_size}")
    
    # Setup simulation matrices
    transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, spatial_step = \
        setup_simulation_matrices(spatial_size, viscosity, device)
    
    # Calculate simulation parameters
    total_steps = math.ceil(sim_time / time_step)
    record_interval = math.floor(total_steps / num_time_points)
    
    # Convert to tensors and move to device
    initial_states = torch.tensor(initial_states, device=device, dtype=torch.float32)
    target_states = torch.tensor(target_states, device=device, dtype=torch.float32)
    
    # Set agent to evaluation mode
    agent.eval()
    
    all_final_states = []
    all_mse_values = []
    
    print(f"\n" + "="*50)
    print("RUNNING AGENT EVALUATION")
    print("="*50)
    
    for traj_idx in range(num_trajectories):
        # Get initial state for this trajectory
        current_state = initial_states[traj_idx:traj_idx+1]  # Shape: (1, spatial_size)
        target_state = target_states[traj_idx:traj_idx+1]    # Shape: (1, spatial_size)
        
        # Pad state for boundary conditions
        state = torch.nn.functional.pad(current_state, (1, 1))  # Add boundary padding
        
        # Run simulation for num_time_points
        for time_idx in range(num_time_points):
            # Get observation for agent (remove padding)
            obs = state[..., 1:-1]  # Shape: (1, spatial_size)
            
            # Get action from agent
            with torch.no_grad():
                action = agent.actor_mean(obs)  # Use mean action for evaluation
            
            # Pad action for simulation
            action_padded = torch.nn.functional.pad(action, (1, 1))
            
            # Run multiple time steps with this action
            for step in range(record_interval):
                state = simulate_burgers_one_time_point(
                    state, action_padded, transport_indices, transport_coeffs,
                    diffusion_indices, diffusion_coeffs, time_step
                )
        
        # Get final state (remove padding)
        final_state = state[..., 1:-1]
        all_final_states.append(final_state)
        
        # Calculate MSE for this trajectory
        mse = ((final_state - target_state) ** 2).mean().item()
        all_mse_values.append(mse)
        
        if (traj_idx + 1) % 10 == 0:
            print(f"Completed trajectory {traj_idx + 1}/{num_trajectories}, MSE: {mse:.6f}")
    
    # Calculate overall statistics
    mean_mse = np.mean(all_mse_values)
    std_mse = np.std(all_mse_values)
    min_mse = np.min(all_mse_values)
    max_mse = np.max(all_mse_values)
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"J_mean_mse (Mean MSE over {num_trajectories} trajectories): {mean_mse:.6f}")
    print(f"Standard deviation: {std_mse:.6f}")
    print(f"Minimum MSE: {min_mse:.6f}")
    print(f"Maximum MSE: {max_mse:.6f}")
    
    return mean_mse, all_mse_values

def main():
    parser = argparse.ArgumentParser(description="Test a saved PPO agent on pre-generated test dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the saved agent checkpoint")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to run on (cuda:0, cpu, or auto)")
    parser.add_argument("--num_trajectories", type=int, default=50,
                       help="Number of trajectories from test dataset to evaluate")
    parser.add_argument("--num_time_points", type=int, default=10,
                       help="Number of time points for simulation")
    parser.add_argument("--viscosity", type=float, default=0.01,
                       help="Viscosity parameter for simulation")
    parser.add_argument("--sim_time", type=float, default=0.1,
                       help="Total simulation time")
    parser.add_argument("--time_step", type=float, default=1e-4,
                       help="Time step for simulation")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load the saved agent
    print(f"Loading agent from: {args.checkpoint_path}")
    agent, metadata = load_saved_agent(args.checkpoint_path, device=device)
    
    # Print some info about the loaded agent
    print("\n" + "="*50)
    print("LOADED AGENT INFO")
    print("="*50)
    print(f"Training iteration: {metadata.get('iteration', 'unknown')}")
    print(f"Global step: {metadata.get('global_step', 'unknown')}")
    print(f"Episode return mean: {metadata.get('episode_return_mean', 'unknown')}")
    print(f"Version: {metadata.get('version', 'unknown')}")
    print(f"PyTorch version: {metadata.get('torch_version', 'unknown')}")
    
    # Test the agent on the dataset
    mean_mse, all_mse_values = test_agent_on_dataset(
        agent=agent,
        device=device,
        num_trajectories=args.num_trajectories,
        num_time_points=args.num_time_points,
        viscosity=args.viscosity,
        sim_time=args.sim_time,
        time_step=args.time_step
    )
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Tested agent on {args.num_trajectories} trajectories from test dataset")
    print(f"Final J_mean_mse: {mean_mse:.6f}")

if __name__ == "__main__":
    main() 