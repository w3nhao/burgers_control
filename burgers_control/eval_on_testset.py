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
import os
import argparse
import torch
import numpy as np
import math

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not available, relying on system environment variables")

from burgers_control.ppo import load_saved_agent
from burgers_control.burgers import (
    create_differential_matrices_1d, 
    simulate_burgers_one_time_point,
    get_test_data,
    get_squence_data,
    get_test_data_with_metadata,
    get_training_data_with_metadata,
    burgers_solver
)
from burgers_control.utils.utils import setup_logging, get_logger_functions


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
    transport_coeffs = torch.tensor(np.stack(first_deriv.data) / (2 * spatial_step), dtype=torch.float32, device=device)
    diffusion_indices = list(second_deriv.rows)
    diffusion_coeffs = torch.tensor(float(viscosity) * np.stack(second_deriv.data) / spatial_step**2, dtype=torch.float32, device=device)
    
    return transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, spatial_step

def test_agent_on_dataset(agent, agent_metadata, device, test_file_path, num_trajectories=50, mode="final_state"):
    """
    Test agent on pre-generated test dataset using one-step simulation.
    Environment parameters are extracted from agent metadata and validated against test dataset.
    
    Args:
        agent: Trained PPO agent
        agent_metadata: Metadata from saved agent containing training parameters
        device: Device for computation
        test_file_path: Path to the test dataset file
        num_trajectories: Number of test trajectories to evaluate
        mode: Target mode for goal-conditioned agent
            - "final_state": Use final target state as goal for all time steps (default)
            - "next_state": Use next state in sequence as target for each time step
        
    Returns:
        tuple: (mean_mse, all_mse_values)
    """
    if mode not in ["final_state", "next_state"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'final_state' or 'next_state'")
    
    # Extract environment parameters from agent metadata
    nested_metadata = agent_metadata.get('metadata', {})
    training_args = nested_metadata.get('args', {})
    
    agent_params = {
        'num_time_points': training_args.get('num_time_points', 10),
        'viscosity': training_args.get('viscosity', 0.01),
        'sim_time': training_args.get('sim_time', 1.0),
        'time_step': training_args.get('time_step', 1e-4),
        'spatial_size': training_args.get('spatial_size', 128)
    }
    
    log_info("="*50)
    log_info("AGENT ENVIRONMENT PARAMETERS")
    log_info("="*50)
    for param, value in agent_params.items():
        log_info(f"  {param}: {value}")
    
    # Load test data and extract its metadata
    test_data, test_metadata = get_test_data_with_metadata(test_file_path)
    
    log_info("="*50)  
    log_info("TEST DATASET PARAMETERS")
    log_info("="*50)
    for param, value in test_metadata.items():
        log_info(f"  {param}: {value}")
    
    # Validate that agent and test dataset parameters match
    log_info("="*50)
    log_info("PARAMETER VALIDATION")
    log_info("="*50)
    
    params_to_check = ['num_time_points', 'viscosity', 'sim_time', 'time_step', 'spatial_size']
    mismatch_found = False
    
    for param in params_to_check:
        agent_val = agent_params.get(param)
        test_val = test_metadata.get(param)
        
        if agent_val is not None and test_val is not None:
            if isinstance(agent_val, float) and isinstance(test_val, float):
                # Use relative tolerance for floating point comparison
                if abs(agent_val - test_val) > max(1e-9 * max(abs(agent_val), abs(test_val)), 1e-12):
                    log_error(f"❌ MISMATCH: {param} - Agent: {agent_val}, Test dataset: {test_val}")
                    mismatch_found = True
                else:
                    log_info(f"✓ MATCH: {param} - {agent_val}")
            else:
                if agent_val != test_val:
                    log_error(f"❌ MISMATCH: {param} - Agent: {agent_val}, Test dataset: {test_val}")
                    mismatch_found = True
                else:
                    log_info(f"✓ MATCH: {param} - {agent_val}")
        else:
            log_warning(f"⚠ MISSING: {param} - Agent: {agent_val}, Test dataset: {test_val}")
    
    if mismatch_found:
        log_error("❌ CRITICAL: Parameter mismatch detected!")
        log_error("The agent was trained with different environment parameters than the test dataset.")
        log_error("Results may not be meaningful. Consider retraining or using correct datasets.")
        raise ValueError("Environment parameter mismatch between agent and test dataset")
    else:
        log_info("✓ ALL PARAMETERS MATCH - proceeding with evaluation")
    
    # Use agent parameters for simulation
    num_time_points = agent_params['num_time_points']
    viscosity = agent_params['viscosity']
    sim_time = agent_params['sim_time']
    time_step = agent_params['time_step']
    expected_spatial_size = agent_params['spatial_size']
    
    # Extract initial states and target states for the first num_trajectories
    initial_states = test_data['observations'][:num_trajectories, 0, :]  # Shape: (N, spatial_size)
    final_targets = test_data['targets'][:num_trajectories, :]           # Shape: (N, spatial_size)
    
    # For next_state mode, we also need the full observation sequences
    if mode == "next_state":
        full_observations = test_data['observations'][:num_trajectories]  # Shape: (N, T-1, spatial_size)
    
    spatial_size = initial_states.shape[1]
    
    # Validate spatial size matches agent expectation
    if spatial_size != expected_spatial_size:
        log_error(f"❌ CRITICAL: Spatial size mismatch!")
        log_error(f"Agent expects spatial_size={expected_spatial_size}, but test data has {spatial_size}")
        raise ValueError(f"Spatial size mismatch: agent expects {expected_spatial_size}, got {spatial_size}")
    
    log_info("="*50)
    log_info(f"TESTING AGENT ON DATASET - {mode.upper()} MODE")
    log_info("="*50)
    log_info(f"Testing on {num_trajectories} trajectories")
    log_info(f"Initial states shape: {initial_states.shape}")
    log_info(f"Final target states shape: {final_targets.shape}")
    if mode == "next_state":
        log_info(f"Full observations shape: {full_observations.shape}")
    log_info(f"Using simulation parameters from agent:")
    log_info(f"  - Spatial size: {spatial_size}")
    log_info(f"  - Num time points: {num_time_points}")
    log_info(f"  - Viscosity: {viscosity}")
    log_info(f"  - Sim time: {sim_time}")
    log_info(f"  - Time step: {time_step}")
    
    # Setup simulation matrices
    transport_indices, transport_coeffs, diffusion_indices, diffusion_coeffs, spatial_step = \
        setup_simulation_matrices(spatial_size, viscosity, device)
    
    # Calculate simulation parameters
    total_steps = math.ceil(sim_time / time_step)
    record_interval = math.floor(total_steps / num_time_points)
    
    # Convert to tensors and move to device
    initial_states = torch.tensor(initial_states, device=device, dtype=torch.float32)
    final_targets = torch.tensor(final_targets, device=device, dtype=torch.float32)
    
    if mode == "next_state":
        full_observations = torch.tensor(full_observations, device=device, dtype=torch.float32)
    
    # Set agent to evaluation mode
    agent.eval()
    
    all_final_states = []
    all_mse_values = []
    
    log_info("="*50)
    log_info(f"RUNNING AGENT EVALUATION - {mode.upper()} MODE")
    log_info("="*50)
    
    for traj_idx in range(num_trajectories):
        # Get initial state for this trajectory
        current_state = initial_states[traj_idx:traj_idx+1]  # Shape: (1, spatial_size)
        final_target = final_targets[traj_idx:traj_idx+1]    # Shape: (1, spatial_size)
        
        # Pad state for boundary conditions
        state = torch.nn.functional.pad(current_state, (1, 1))  # Add boundary padding
        
        # Run simulation for num_time_points
        for time_idx in range(num_time_points):
            # Get observation for agent (remove padding)
            current_obs = state[..., 1:-1]  # Shape: (1, spatial_size)
            
            # Determine target state based on mode
            if mode == "final_state":
                # Use final target state for all time steps
                target_state = final_target
            elif mode == "next_state":
                # Use next state in sequence as target
                if time_idx < num_time_points - 1:
                    # Use next observation from the test data
                    next_obs_idx = min(time_idx + 1, full_observations.shape[1] - 1)
                    target_state = full_observations[traj_idx:traj_idx+1, next_obs_idx, :]  # Shape: (1, spatial_size)
                else:
                    # For the last time step, use final target
                    target_state = final_target
            
            # Create goal-conditioned observation: concatenate current state and target state
            obs = torch.cat([current_obs, target_state], dim=-1)  # Shape: (1, 2*spatial_size)
            
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
        
        # Calculate MSE for this trajectory (always against final target)
        mse = ((final_state - final_target) ** 2).mean().item()
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
    log_info(f"Mode: {mode}")
    log_info(f"J_mean_mse (Mean MSE over {num_trajectories} trajectories): {mean_mse:.6f}")
    log_info(f"Standard deviation: {std_mse:.6f}")
    log_info(f"Minimum MSE: {min_mse:.6f}")
    log_info(f"Maximum MSE: {max_mse:.6f}")
    
    return mean_mse, all_mse_values

def test_environment_with_training_data(train_file_path, device, num_trajectories=50):
    """
    Test environment simulation using actions from training dataset.
    Environment parameters are extracted from the dataset metadata.
    This serves as a sanity check for the environment implementation.
    
    Args:
        train_file_path: Path to the training dataset file
        device: Device for computation  
        num_trajectories: Number of training trajectories to test
        
    Returns:
        tuple: (mean_mse, all_mse_values)
    """
    # Load training data with metadata
    train_data, train_metadata = get_training_data_with_metadata(train_file_path)
    
    # Extract environment parameters from dataset metadata
    num_time_points = train_metadata.get('num_time_points', 10)
    viscosity = train_metadata.get('viscosity', 0.01)
    sim_time = train_metadata.get('sim_time', 1.0)
    time_step = train_metadata.get('time_step', 1e-4)
    expected_spatial_size = train_metadata.get('spatial_size', 128)
    
    log_info("="*50)
    log_info("TRAINING DATASET PARAMETERS")
    log_info("="*50)
    log_info(f"  - Num time points: {num_time_points}")
    log_info(f"  - Viscosity: {viscosity}")
    log_info(f"  - Simulation time: {sim_time}")
    log_info(f"  - Time step: {time_step}")
    log_info(f"  - Expected spatial size: {expected_spatial_size}")
    
    # Extract data for the first num_trajectories
    observations = train_data['observations'][:num_trajectories]  # Shape: (N, T-1, spatial_size)
    actions = train_data['actions'][:num_trajectories]            # Shape: (N, T, spatial_size) 
    targets = train_data['targets'][:num_trajectories]            # Shape: (N, spatial_size)
    
    # Get initial states (first observation for each trajectory)
    initial_states = observations[:, 0, :]  # Shape: (N, spatial_size)
    
    spatial_size = initial_states.shape[1]
    
    # Validate spatial size
    if spatial_size != expected_spatial_size:
        log_warning(f"⚠ Spatial size mismatch: expected {expected_spatial_size}, got {spatial_size}")
        log_warning("Using actual spatial size from data")
    
    log_info("="*50)
    log_info("ENVIRONMENT CHECK SETUP")
    log_info("="*50)
    log_info(f"Testing environment on {num_trajectories} training trajectories")
    log_info(f"Initial states shape: {initial_states.shape}")
    log_info(f"Actions shape: {actions.shape}")
    log_info(f"Target states shape: {targets.shape}")
    log_info(f"Spatial size: {spatial_size}")
    log_info("Using environment parameters from dataset metadata:")
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
    parser.add_argument("--train_file_path", type=str, required=False,
                       help="Path to the training dataset file (required for environment check mode)")
    parser.add_argument("--test_file_path", type=str, required=False,
                       help="Path to the test dataset file (required for agent evaluation mode)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to run on (cuda:0, cpu, or auto)")
    parser.add_argument("--num_trajectories", type=int, default=50,
                       help="Number of trajectories from test dataset to evaluate")
    parser.add_argument("--mode", type=str, default="final_state",
                       choices=["final_state", "next_state"],
                       help="Target mode for goal-conditioned agent: 'final_state' (use final target for all steps) or 'next_state' (use next state as target)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    log_info(f"Using device: {device}")
    
    # Load the saved agent
    if args.checkpoint_path:
        if not args.test_file_path:
            raise ValueError("--test_file_path is required when using --checkpoint_path for agent evaluation")
            
        log_info(f"Loading agent from: {args.checkpoint_path}")
        agent, metadata = load_saved_agent(args.checkpoint_path, device=device)
        
        # Print some info about the loaded agent
        log_info("="*50)
        log_info("LOADED AGENT INFO")
        log_info("="*50)
        # Access nested metadata structure
        nested_metadata = metadata.get('metadata', {})
        log_info(f"Training iteration: {nested_metadata.get('iteration', 'unknown')}")
        log_info(f"Global step: {nested_metadata.get('global_step', 'unknown')}")
        log_info(f"Episode return mean: {nested_metadata.get('episode_return_mean', 'unknown')}")
        log_info(f"Version: {metadata.get('version', 'unknown')}")
        log_info(f"PyTorch version: {metadata.get('torch_version', 'unknown')}")
        
        # Print training arguments if available
        training_args = nested_metadata.get('args', {})
        if training_args:
            log_info(f"Training spatial size: {training_args.get('spatial_size', 'unknown')}")
            log_info(f"Training viscosity: {training_args.get('viscosity', 'unknown')}")
            log_info(f"Training reward type: {training_args.get('reward_type', 'unknown')}")
            log_info(f"Training num_time_points: {training_args.get('num_time_points', 'unknown')}")
            log_info(f"Training sim_time: {training_args.get('sim_time', 'unknown')}")
            log_info(f"Training time_step: {training_args.get('time_step', 'unknown')}")
        
        # Test the agent on the dataset
        mean_mse, all_mse_values = test_agent_on_dataset(
            agent=agent,
            agent_metadata=metadata,
            device=device,
            test_file_path=args.test_file_path,
            num_trajectories=args.num_trajectories,
            mode=args.mode
        )
        
        log_info("="*50)
        log_info("SUMMARY")
        log_info("="*50)
        log_info(f"Tested agent on {args.num_trajectories} trajectories from test dataset")
        log_info(f"Environment parameters were automatically extracted from agent metadata")
        log_info(f"Test dataset parameters were validated against agent parameters")
        log_info(f"Final J_mean_mse: {mean_mse:.6f}")
        
    else:
        # Environment check mode - use training data with provided actions
        if not args.train_file_path:
            raise ValueError("--train_file_path is required when running environment check mode (no --checkpoint_path provided)")
            
        log_info("No checkpoint path provided. Running environment check mode.")
        log_info("This will test the environment simulation using training data actions.")
        log_info("Environment parameters will be automatically extracted from dataset metadata.")
        
        mean_mse, all_mse_values = test_environment_with_training_data(
            train_file_path=args.train_file_path,
            device=device,
            num_trajectories=args.num_trajectories
        )
        
        log_info("="*50)
        log_info("SUMMARY")
        log_info("="*50)
        log_info(f"Environment check completed on {args.num_trajectories} trajectories from training dataset")
        log_info(f"Environment parameters were automatically extracted from dataset metadata")
        log_info(f"Final J_env_check_mse: {mean_mse:.10f}")
        log_info("Low MSE indicates accurate environment implementation.")

if __name__ == "__main__":
    main() 