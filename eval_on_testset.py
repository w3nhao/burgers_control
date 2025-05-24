#!/usr/bin/env python3
"""
Test script for evaluating trained PPO agents on pre-generated test dataset.

This script loads a saved agent and tests it on 50 trajectories from the test dataset,
using one-step simulation for environment evolution and calculating the final mean MSE.

If no checkpoint path is provided, it performs an environment check using the first 50
trajectories from the training dataset with their provided actions.

Usage:
    python test_agent_on_dataset.py --checkpoint_path checkpoints/run_name/agent_final.pt
    python test_agent_on_dataset.py  # Environment check mode
"""

import argparse
import torch
import numpy as np
import math
from ppo import load_saved_agent
from burgers import (
    create_differential_matrices_1d, 
    simulate_burgers_one_time_point,
    get_test_data,
    get_squence_data,
    BURGERS_TEST_FILE_PATH,
    BURGERS_TRAIN_FILE_PATH,
    burgers_solver
)
from utils.utils import setup_logging, get_logger_functions

setup_logging(logger_name="eval_on_testset")
log_info, log_warning, log_error = get_logger_functions("eval_on_testset")

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
    log_info(f"Testing on {num_trajectories} trajectories")
    log_info(f"Initial states shape: {initial_states.shape}")
    log_info(f"Target states shape: {target_states.shape}")
    log_info(f"Spatial size: {spatial_size}")
    
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
    
    log_info("="*50)
    log_info("RUNNING AGENT EVALUATION")
    log_info("="*50)
    
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
        
        # Show progress - more frequent for small numbers of trajectories
        if num_trajectories <= 10 or (traj_idx + 1) % 10 == 0:
            log_info(f"Completed trajectory {traj_idx + 1}/{num_trajectories}, MSE: {mse:.6f}")
    
    # Calculate overall statistics
    mean_mse = np.mean(all_mse_values)
    std_mse = np.std(all_mse_values)
    min_mse = np.min(all_mse_values)
    max_mse = np.max(all_mse_values)
    
    log_info("="*50)
    log_info("FINAL RESULTS")
    log_info("="*50)
    log_info(f"J_mean_mse (Mean MSE over {num_trajectories} trajectories): {mean_mse:.6f}")
    log_info(f"Standard deviation: {std_mse:.6f}")
    log_info(f"Minimum MSE: {min_mse:.6f}")
    log_info(f"Maximum MSE: {max_mse:.6f}")
    
    return mean_mse, all_mse_values

def test_environment_with_training_data(device, num_trajectories=50, num_time_points=10,
                                      viscosity=0.01, sim_time=0.1, time_step=1e-4):
    """
    Test environment simulation using actions from training dataset.
    This serves as a sanity check for the environment implementation.
    
    Args:
        device: Device for computation  
        num_trajectories: Number of training trajectories to test
        num_time_points: Number of time points in simulation
        viscosity: Viscosity parameter
        sim_time: Total simulation time  
        time_step: Time step for simulation
        
    Returns:
        float: Mean MSE between simulated final states and training final states
    """
    # Load training data
    train_data = get_squence_data(BURGERS_TRAIN_FILE_PATH)
    
    # Extract data for the first num_trajectories
    observations = train_data['observations'][:num_trajectories]  # Shape: (N, T-1, spatial_size)
    actions = train_data['actions'][:num_trajectories]            # Shape: (N, T, spatial_size) 
    targets = train_data['targets'][:num_trajectories]            # Shape: (N, spatial_size)
    
    # Get initial states (first observation for each trajectory)
    initial_states = observations[:, 0, :]  # Shape: (N, spatial_size)
    
    spatial_size = initial_states.shape[1]
    log_info(f"Testing environment on {num_trajectories} training trajectories")
    log_info(f"Initial states shape: {initial_states.shape}")
    log_info(f"Actions shape: {actions.shape}")
    log_info(f"Target states shape: {targets.shape}")
    log_info(f"Spatial size: {spatial_size}")
    
    # Debug: Print simulation parameters
    log_info("Simulation parameters:")
    log_info(f"  - Viscosity: {viscosity}")
    log_info(f"  - Simulation time: {sim_time}")
    log_info(f"  - Time step: {time_step}")
    log_info(f"  - Number of time points: {num_time_points}")
    
    # Setup simulation matrices
    transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, spatial_step = \
        setup_simulation_matrices(spatial_size, viscosity, device)
    
    # Calculate simulation parameters
    total_steps = math.ceil(sim_time / time_step)
    record_interval = math.floor(total_steps / num_time_points)
    
    log_info(f"  - Total simulation steps: {total_steps}")
    log_info(f"  - Record interval: {record_interval}")
    log_info(f"  - Actual steps per time point: {record_interval}")
    log_info(f"  - Spatial step: {spatial_step:.6f}")
    
    # Convert to tensors and move to device
    initial_states = torch.tensor(initial_states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.float32)
    targets = torch.tensor(targets, device=device, dtype=torch.float32)
    
    all_final_states = []
    all_mse_values = []
    all_original_mse_values = []
    
    log_info("="*50)
    log_info("RUNNING ENVIRONMENT CHECK")
    log_info("="*50)
    
    # Test first trajectory with detailed debugging
    for traj_idx in range(min(1, num_trajectories)):  # Debug first trajectory only
        log_info(f"--- Debugging trajectory {traj_idx} ---")
        
        # Get initial state and actions for this trajectory
        current_state = initial_states[traj_idx:traj_idx+1]  # Shape: (1, spatial_size)
        traj_actions = actions[traj_idx]                      # Shape: (T, spatial_size)
        target_state = targets[traj_idx:traj_idx+1]          # Shape: (1, spatial_size)
        
        log_info(f"Initial state stats: mean={current_state.mean().item():.6f}, std={current_state.std().item():.6f}")
        log_info(f"Target state stats: mean={target_state.mean().item():.6f}, std={target_state.std().item():.6f}")
        
        # COMPARISON 1: Use original burgers_solver
        log_info("=== Testing original burgers_solver ===")
        traj_actions_expanded = traj_actions.unsqueeze(0)  # Add batch dimension: (1, T, spatial_size)
        original_trajectory = burgers_solver(current_state, traj_actions_expanded, num_time_points=num_time_points)
        original_final_state = original_trajectory[:, -1, :]  # Shape: (1, spatial_size)
        original_mse = ((original_final_state - target_state) ** 2).mean().item()
        
        log_info(f"Original solver final state stats: mean={original_final_state.mean().item():.6f}, std={original_final_state.std().item():.6f}")
        log_info(f"Original solver MSE: {original_mse:.10f}")
        
        # COMPARISON 2: Use our step-by-step implementation
        log_info("=== Testing step-by-step implementation ===")
        
        # Pad state for boundary conditions
        state = torch.nn.functional.pad(current_state, (1, 1))  # Add boundary padding
        
        # Debug: Store intermediate states
        intermediate_states = [current_state.clone()]
        
        # Run simulation for num_time_points using provided actions
        for time_idx in range(num_time_points):
            # Get action for this time step
            action = traj_actions[time_idx:time_idx+1]  # Shape: (1, spatial_size)
            
            log_info(f"Time {time_idx}: action stats: mean={action.mean().item():.6f}, std={action.std().item():.6f}")
            
            # Pad action for simulation  
            action_padded = torch.nn.functional.pad(action, (1, 1))
            
            # Run multiple time steps with this action
            for step in range(record_interval):
                state = simulate_burgers_one_time_point(
                    state, action_padded, transport_indices, transport_coeffs,
                    diffusion_indices, diffusion_coeffs, time_step
                )
            
            # Store intermediate state (remove padding)
            intermediate_state = state[..., 1:-1]
            intermediate_states.append(intermediate_state.clone())
            log_info(f"  After simulation: state stats: mean={intermediate_state.mean().item():.6f}, std={intermediate_state.std().item():.6f}")
        
        # Get final state (remove padding)
        final_state = state[..., 1:-1]
        
        # Compare with target
        mse = ((final_state - target_state) ** 2).mean().item()
        max_abs_diff = (final_state - target_state).abs().max().item()
        
        log_info("Final comparison:")
        log_info(f"  - Step-by-step final state stats: mean={final_state.mean().item():.6f}, std={final_state.std().item():.6f}")
        log_info(f"  - Step-by-step MSE: {mse:.10f}")
        log_info(f"  - Max absolute difference from target: {max_abs_diff:.10f}")
        
        # Compare the two methods
        method_diff = (final_state - original_final_state).abs().max().item()
        log_info(f"  - Max difference between methods: {method_diff:.10f}")
        
        # Store for overall statistics
        all_final_states.append(final_state)
        all_mse_values.append(mse)
        all_original_mse_values.append(original_mse)
        
        # Break after first trajectory for detailed debugging
        break
    
    # Process remaining trajectories without detailed debugging
    for traj_idx in range(1, num_trajectories):
        # Get initial state and actions for this trajectory
        current_state = initial_states[traj_idx:traj_idx+1]  # Shape: (1, spatial_size)
        traj_actions = actions[traj_idx]                      # Shape: (T, spatial_size)
        target_state = targets[traj_idx:traj_idx+1]          # Shape: (1, spatial_size)
        
        # Test with original burgers_solver
        traj_actions_expanded = traj_actions.unsqueeze(0)  # Add batch dimension
        original_trajectory = burgers_solver(current_state, traj_actions_expanded, num_time_points=num_time_points)
        original_final_state = original_trajectory[:, -1, :]
        original_mse = ((original_final_state - target_state) ** 2).mean().item()
        all_original_mse_values.append(original_mse)
        
        # Test with step-by-step implementation
        # Pad state for boundary conditions
        state = torch.nn.functional.pad(current_state, (1, 1))  # Add boundary padding
        
        # Run simulation for num_time_points using provided actions
        for time_idx in range(num_time_points):
            # Get action for this time step
            action = traj_actions[time_idx:time_idx+1]  # Shape: (1, spatial_size)
            
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
        
        # Calculate MSE between simulated final state and training target
        mse = ((final_state - target_state) ** 2).mean().item()
        all_mse_values.append(mse)
        
        # Show progress - more frequent for small numbers of trajectories
        if num_trajectories <= 10 or (traj_idx + 1) % 10 == 0:
            log_info(f"Completed trajectory {traj_idx + 1}/{num_trajectories}, Step-by-step MSE: {mse:.10f}, Original MSE: {original_mse:.10f}")
    
    # Calculate overall statistics
    mean_mse = np.mean(all_mse_values)
    std_mse = np.std(all_mse_values)
    min_mse = np.min(all_mse_values)
    max_mse = np.max(all_mse_values)
    
    mean_original_mse = np.mean(all_original_mse_values)
    std_original_mse = np.std(all_original_mse_values)
    min_original_mse = np.min(all_original_mse_values)
    max_original_mse = np.max(all_original_mse_values)
    
    log_info("="*50)
    log_info("ENVIRONMENT CHECK RESULTS")
    log_info("="*50)
    log_info("STEP-BY-STEP IMPLEMENTATION:")
    log_info(f"  J_env_check_mse (Mean MSE over {num_trajectories} trajectories): {mean_mse:.10f}")
    log_info(f"  Standard deviation: {std_mse:.10f}")
    log_info(f"  Minimum MSE: {min_mse:.10f}")
    log_info(f"  Maximum MSE: {max_mse:.10f}")
    
    log_info("ORIGINAL BURGERS_SOLVER:")
    log_info(f"  J_original_mse (Mean MSE over {num_trajectories} trajectories): {mean_original_mse:.10f}")
    log_info(f"  Standard deviation: {std_original_mse:.10f}")
    log_info(f"  Minimum MSE: {min_original_mse:.10f}")
    log_info(f"  Maximum MSE: {max_original_mse:.10f}")
    
    if mean_original_mse < 1e-10:
        log_info("✓ EXCELLENT: Original solver MSE is extremely low - training data is consistent!")
    elif mean_original_mse < 1e-6:
        log_info("✓ GOOD: Original solver MSE is very low - training data is mostly consistent.")
    elif mean_original_mse < 1e-3:
        log_info("⚠ WARNING: Original solver MSE is moderate - training data may have some inconsistencies.")
    else:
        log_info("❌ TRAINING DATA ISSUE: Original solver MSE is high - training data was generated with different parameters!")
        log_info("   This indicates the training data targets don't match the current simulation parameters.")
    
    # Check if our implementation matches the original solver exactly
    method_differences = [abs(step_mse - orig_mse) for step_mse, orig_mse in zip(all_mse_values, all_original_mse_values)]
    max_method_diff = max(method_differences)
    
    if max_method_diff < 1e-10:
        log_info("✓ EXCELLENT: Step-by-step implementation is PERFECT - matches original solver exactly!")
        log_info("   Max difference between methods across all trajectories: {:.2e}".format(max_method_diff))
    elif max_method_diff < 1e-6:
        log_info("✓ GOOD: Step-by-step implementation is very accurate - closely matches original solver.")
        log_info("   Max difference between methods: {:.2e}".format(max_method_diff))
    else:
        log_info("❌ IMPLEMENTATION ERROR: Step-by-step implementation differs from original solver.")
        log_info("   Max difference between methods: {:.2e}".format(max_method_diff))
    
    log_info("="*70)
    log_info("DIAGNOSIS:")
    if max_method_diff < 1e-10 and mean_original_mse > 1e-3:
        log_info("✓ Environment implementation is CORRECT (perfectly matches reference solver)")
        log_info("❌ Training data appears to be generated with different simulation parameters")
        log_info("   - Both solvers give identical results")
        log_info("   - Neither matches the training targets")
        log_info("   - This suggests the training data was created with different:")
        log_info("     * Viscosity coefficient")
        log_info("     * Time step size") 
        log_info("     * Simulation time")
        log_info("     * Numerical scheme")
    elif max_method_diff < 1e-10 and mean_original_mse < 1e-6:
        log_info("✓ Environment implementation is CORRECT")
        log_info("✓ Training data is consistent with current parameters")
    else:
        log_info("❌ Issues detected that need investigation")
    log_info("="*70)
    
    return mean_mse, all_mse_values

def main():
    parser = argparse.ArgumentParser(description="Test a saved PPO agent on pre-generated test dataset")
    parser.add_argument("--checkpoint_path", type=str, required=False,
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
    
    log_info(f"Using device: {device}")
    
    # Load the saved agent
    if args.checkpoint_path:
        log_info(f"Loading agent from: {args.checkpoint_path}")
        agent, metadata = load_saved_agent(args.checkpoint_path, device=device)
        
        # Print some info about the loaded agent
        log_info("="*50)
        log_info("LOADED AGENT INFO")
        log_info("="*50)
        log_info(f"Training iteration: {metadata.get('iteration', 'unknown')}")
        log_info(f"Global step: {metadata.get('global_step', 'unknown')}")
        log_info(f"Episode return mean: {metadata.get('episode_return_mean', 'unknown')}")
        log_info(f"Version: {metadata.get('version', 'unknown')}")
        log_info(f"PyTorch version: {metadata.get('torch_version', 'unknown')}")
        
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
        
        log_info("="*50)
        log_info("SUMMARY")
        log_info("="*50)
        log_info(f"Tested agent on {args.num_trajectories} trajectories from test dataset")
        log_info(f"Final J_mean_mse: {mean_mse:.6f}")
        
    else:
        # Environment check mode - use training data with provided actions
        log_info("No checkpoint path provided. Running environment check mode.")
        log_info("This will test the environment simulation using training data actions.")
        
        mean_mse, all_mse_values = test_environment_with_training_data(
            device=device,
            num_trajectories=args.num_trajectories,
            num_time_points=args.num_time_points,
            viscosity=args.viscosity,
            sim_time=args.sim_time,
            time_step=args.time_step
        )
        
        log_info("="*50)
        log_info("SUMMARY")
        log_info("="*50)
        log_info(f"Environment check completed on {args.num_trajectories} trajectories from training dataset")
        log_info(f"Final J_env_check_mse: {mean_mse:.10f}")
        log_info("Low MSE indicates accurate environment implementation.")

if __name__ == "__main__":
    main() 