import torch
import numpy as np
import argparse
from dataset import BurgersTest, test_file_path
from evaluation import evaluate_model_performance
from burgers_env import BurgersEnvClosedLoop
from ppo import Agent

def evaluate_policy(model_path, test_dataset_path=test_file_path, num_episodes=10, device=None):
    """
    Evaluate a trained PPO policy model on the test dataset.
    
    Args:
        model_path (str): Path to the trained model file
        test_dataset_path (str): Path to the test dataset
        num_episodes (int): Number of episodes to evaluate
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        float: Mean squared error between final states and target states
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = BurgersTest(test_dataset_path)
    
    # Create environment to get observation/action dimensions
    env = BurgersEnvClosedLoop(test_dataset_path=test_dataset_path)
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    
    # Create agent with same architecture as used in training
    agent = Agent(n_obs, n_act, device=device)
    
    # Load model weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Loaded model from {model_path}")
    
    # Evaluation metrics
    total_mse = 0.0
    all_actions = []
    
    # Evaluate on test dataset
    for i in range(min(len(test_dataset), num_episodes)):
        print(f"Evaluating episode {i+1}/{min(len(test_dataset), num_episodes)}")
        
        # Get test sample
        sample = test_dataset[i]
        initial_state = torch.tensor(sample['observations'][0], dtype=torch.float32, device=device).unsqueeze(0)
        target_state = torch.tensor(sample['target'], dtype=torch.float32, device=device).unsqueeze(0)
        
        # Reset environment
        env.reset(seed=i)
        obs = initial_state.squeeze().cpu().numpy()
        
        # Initialize tensor to store actions
        num_time_points = env.num_time_points
        actions = torch.zeros(1, num_time_points, n_act, device=device)
        
        # Generate actions using the policy
        for t in range(num_time_points):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            # Store action
            actions[0, t] = action.squeeze()
            
            # Step environment
            next_obs, _, done, _, _ = env.step(action.cpu().numpy()[0])
            obs = next_obs
            
            if done:
                break
        
        # Store actions for this episode
        all_actions.append(actions.cpu().numpy())
        
        # Evaluate model performance
        mse = evaluate_model_performance(
            num_time_points, 
            initial_state, 
            target_state,
            actions,
            device
        )
        
        total_mse += mse.item()
    
    # Calculate average MSE
    avg_mse = total_mse / min(len(test_dataset), num_episodes)
    print(f"Average MSE across {min(len(test_dataset), num_episodes)} episodes: {avg_mse}")
    
    return avg_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on Burgers equation")
    parser.add_argument("--model_path", type=str, default="ppo_burgers_closedloop_model.pt", 
                        help="Path to the trained model file")
    parser.add_argument("--test_dataset", type=str, default=test_file_path, 
                        help="Path to the test dataset")
    parser.add_argument("--num_episodes", type=int, default=10, 
                        help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run evaluation on ('cpu' or 'cuda')")
    
    args = parser.parse_args()
    
    evaluate_policy(
        model_path=args.model_path,
        test_dataset_path=args.test_dataset,
        num_episodes=args.num_episodes,
        device=args.device
    ) 