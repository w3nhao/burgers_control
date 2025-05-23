#!/usr/bin/env python3
"""
Example script showing how to load and use saved PPO agents.

Usage:
    python load_agent_example.py --checkpoint_path checkpoints/run_name/agent_final.pt
"""

import argparse
import torch
import numpy as np
from ppo import load_saved_agent
from burgers_onthefly_env import BurgersOnTheFlyVecEnv

def main():
    parser = argparse.ArgumentParser(description="Load and test a saved PPO agent")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the saved agent checkpoint")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to run on (cuda:0, cpu, or auto)")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to run for testing")
    
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
    
    # Get environment parameters from saved args
    saved_args = metadata.get('args', {})
    if saved_args:
        print("\nEnvironment configuration:")
        print(f"  Spatial size: {saved_args.get('spatial_size', 128)}")
        print(f"  Num time points: {saved_args.get('num_time_points', 10)}")
        print(f"  Viscosity: {saved_args.get('viscosity', 0.01)}")
        print(f"  Reward type: {saved_args.get('reward_type', 'exp_scaled_mse')}")
    
    # Create environment with same parameters as training
    env = BurgersOnTheFlyVecEnv(
        num_envs=1,  # Single environment for testing
        spatial_size=saved_args.get('spatial_size', 128),
        num_time_points=saved_args.get('num_time_points', 10),
        viscosity=saved_args.get('viscosity', 0.01),
        sim_time=saved_args.get('sim_time', 0.1),
        time_step=saved_args.get('time_step', 1e-4),
        forcing_terms_scaling_factor=saved_args.get('forcing_terms_scaling_factor', 1.0),
        reward_type=saved_args.get('reward_type', 'exp_scaled_mse'),
        mse_scaling_factor=saved_args.get('mse_scaling_factor', 1e3)
    )
    env.set_device(device)
    
    print(f"\n" + "="*50)
    print("TESTING AGENT")
    print("="*50)
    
    # Test the agent
    agent.eval()  # Set to evaluation mode
    episode_returns = []
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        
        episode_return = 0
        step = 0
        done = False
        
        while not done and step < 1000:  # Max steps per episode
            with torch.no_grad():
                # Get action from agent (deterministic - using mean)
                action_mean = agent.actor_mean(obs)
                action = action_mean  # Use mean action for evaluation
                
            # Take step in environment
            action_np = action.cpu().numpy()
            next_obs, reward, termination, truncation, info = env.step(action_np)
            
            episode_return += reward[0]  # Single environment
            done = termination[0] or truncation[0]
            
            obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            step += 1
        
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.4f}, Steps = {step}")
    
    print(f"\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Average return over {args.num_episodes} episodes: {np.mean(episode_returns):.4f}")
    print(f"Standard deviation: {np.std(episode_returns):.4f}")
    print(f"Best episode return: {np.max(episode_returns):.4f}")
    print(f"Worst episode return: {np.min(episode_returns):.4f}")
    
    env.close()

if __name__ == "__main__":
    main() 