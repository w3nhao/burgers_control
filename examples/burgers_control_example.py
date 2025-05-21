import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import BurgersTest, test_file_path
from evaluation import burgers_solver

class SimpleController:
    """A basic controller for the Burgers equation"""
    def __init__(self, target_state, amplitude=0.1):
        self.target_state = target_state
        self.amplitude = amplitude
    
    def compute_control(self, current_state, spatial_size):
        """Computes a simple feedback control action"""
        # Compute error between current state and target
        error = self.target_state - current_state
        
        # Apply a simple proportional control (scaled by amplitude)
        control = self.amplitude * error
        
        return control

def visualize_trajectory(initial_state, trajectory, target_state, forcing_terms, title="Burgers Equation Trajectory"):
    """Visualize the evolution of the system"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot the evolution of the state
    axes[0].set_title("State Evolution")
    for i in range(trajectory.shape[1]):
        alpha = 0.2 + 0.8 * (i / trajectory.shape[1])
        axes[0].plot(trajectory[0, i].cpu().numpy(), alpha=alpha, 
                  label=f"t={i}" if i == 0 or i == trajectory.shape[1]-1 else None)
    
    axes[0].plot(initial_state[0].cpu().numpy(), 'b--', label="Initial")
    axes[0].plot(target_state[0].cpu().numpy(), 'r--', label="Target")
    axes[0].legend()
    axes[0].set_xlabel("Spatial Position")
    axes[0].set_ylabel("State Value")
    
    # Plot the error over time
    axes[1].set_title("Error vs Target Over Time")
    errors = []
    for i in range(trajectory.shape[1]):
        mse = ((trajectory[0, i] - target_state[0])**2).mean().item()
        errors.append(mse)
    axes[1].plot(errors)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Mean Squared Error")
    
    # Plot the control inputs
    axes[2].set_title("Control Inputs")
    for i in range(forcing_terms.shape[1]):
        alpha = 0.2 + 0.8 * (i / forcing_terms.shape[1])
        axes[2].plot(forcing_terms[0, i].cpu().numpy(), alpha=alpha, 
                  label=f"t={i}" if i == 0 or i == forcing_terms.shape[1]-1 else None)
    axes[2].legend()
    axes[2].set_xlabel("Spatial Position")
    axes[2].set_ylabel("Control Amplitude")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("examples/burgers_control_example.png")
    plt.close()

if __name__ == "__main__":
    print("="*50)
    print("BURGERS EQUATION CONTROL EXAMPLE")
    print("="*50)
    
    # Load test dataset
    test_dataset = BurgersTest(test_file_path)
    sample = test_dataset[0]
    
    # Extract initial and target states
    initial_state = torch.tensor(sample['observations'][0]).float().unsqueeze(0)
    target_state = torch.tensor(sample['target']).float().unsqueeze(0)
    
    # Print tensor shapes
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Target state shape: {target_state.shape}")
    
    # Create a controller
    controller = SimpleController(target_state, amplitude=0.05)
    
    # Setup simulation parameters
    num_time_points = 20
    spatial_size = initial_state.shape[-1]
    batch_size = initial_state.shape[0]
    
    # Precompute control actions (forcing terms)
    forcing_terms = torch.zeros(batch_size, num_time_points, spatial_size)
    
    # Compute initial control action
    current_state = initial_state.clone()
    for t in range(num_time_points):
        forcing_terms[:, t] = controller.compute_control(current_state, spatial_size)
        
        # Create a temporary forcing tensor for closed-loop simulation
        temp_forcing = torch.zeros(batch_size, 1, spatial_size)
        temp_forcing[:, 0] = forcing_terms[:, t]
        
        # Simulate one step to get the next state for control computation
        if t < num_time_points - 1:  # No need to simulate on the last iteration
            temp_trajectory = burgers_solver(current_state, temp_forcing, num_time_points=1)
            current_state = temp_trajectory[:, -1]
    
    # Run full simulation with all precomputed control actions
    print("\nRunning controlled Burgers equation simulation...")
    trajectory = burgers_solver(initial_state, forcing_terms, num_time_points=num_time_points)
    
    # Calculate final error
    final_state = trajectory[:, -1]
    final_mse = ((final_state - target_state)**2).mean().item()
    print(f"Final MSE: {final_mse}")
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_trajectory(initial_state, trajectory, target_state, forcing_terms, 
                        title="Burgers Equation Control Example")
    print("Visualization saved to 'examples/burgers_control_example.png'")
    
    print("="*50)
    print("CONTROL EXAMPLE COMPLETE")
    print("="*50) 