import os
import torch
import numpy as np
import gymnasium as gym
import tqdm
from burgers_env_onestep import BurgersEnvOneStep, make_burgers_env_onestep

# Import PPO functionality
from ppo import Args, Agent, layer_init

def run_ppo_burgers_onestep():
    """
    Run PPO training using the one-step Burgers environment
    """
    # Set up parameters
    args = Args()
    args.num_envs = 1
    args.total_timesteps = 200000  # Reasonable training length
    args.num_steps = 64           # Steps per rollout
    args.num_minibatches = 2      # Fewer minibatches for faster training
    args.update_epochs = 4        # Update epochs
    args.learning_rate = 1e-3
    
    # Compute dynamic parameters
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create environment directly
    env = BurgersEnvOneStep()
    
    # Observation and action space dimensions
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    
    # Create PPO agent
    agent = Agent(n_obs, n_act, device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
    
    # Define step function to interact with environment
    def step_func(action_tensor):
        action = action_tensor.cpu().numpy()
        next_obs, reward, done, truncated, info = env.step(action[0])
        return (
            torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32, device=device),
            torch.tensor([done or truncated], dtype=torch.bool, device=device),
            info
        )
    
    # Training loop
    obs, _ = env.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Convert done tensor to correct format
    done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    final_mse_values = []
    
    progress_bar = tqdm.tqdm(range(args.num_iterations))
    for iteration in progress_bar:
        # Collect rollouts
        rollout_rewards = []
        ts = []
        
        for step in range(args.num_steps):
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            
            # Execute action in environment - this is now a true one-step simulation
            next_obs, reward, next_done, infos = step_func(action)
            
            # Store transition
            ts.append({
                "obs": obs,
                "dones": done,
                "vals": value.flatten(),
                "actions": action,
                "logprobs": logprob,
                "rewards": reward,
            })
            
            obs = next_obs
            done = next_done
            
            rollout_rewards.append(reward.item())
            
            # If episode is done, store metrics and reset environment
            if done.any():
                episode_length = step + 1
                episode_reward = sum(rollout_rewards)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if 'error' in infos:
                    final_mse_values.append(infos['error'])
                
                # Reset for next episode
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                rollout_rewards = []
        
        # Convert list of transitions to tensors
        keys = ts[0].keys()
        container = {k: torch.stack([t[k] for t in ts]) for k in keys}
        
        # Apply Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(container["vals"])
            lastgaelam = 0
            
            for t in range(args.num_steps - 1, -1, -1):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - container["dones"][t + 1].float()
                    nextvalues = container["vals"][t + 1]
                
                delta = container["rewards"][t] + args.gamma * nextvalues * nextnonterminal - container["vals"][t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + container["vals"]
        
        # Flatten data for minibatch updates
        b_obs = container["obs"].reshape(-1, n_obs)
        b_logprobs = container["logprobs"].reshape(-1)
        b_actions = container["actions"].reshape(-1, n_act)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = container["vals"].reshape(-1)
        
        # Optimize policy and value network
        batch_indices = torch.randperm(args.batch_size)
        
        for epoch in range(args.update_epochs):
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_indices[start:end]
                
                # Compute PPO loss
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # PPO clipped objective function
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Update progress bar with metrics
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            mean_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            mean_mse = np.mean(final_mse_values[-10:]) if final_mse_values else 0
            
            progress_bar.set_description(
                f"Iteration {iteration}/{args.num_iterations}, "
                f"Mean Reward: {mean_reward:.3f}, "
                f"Mean Length: {mean_length:.1f}, "
                f"Mean Final MSE: {mean_mse:.6f}"
            )
    
    # Save the model
    torch.save(agent.state_dict(), "ppo_burgers_onestep_model.pt")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_rewards = []
    eval_mse = []
    for _ in range(10):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            
            obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0])
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward
            done = done or truncated
            
            if done and 'error' in info:
                eval_mse.append(info['error'])
        
        eval_rewards.append(total_reward)
    
    print(f"Evaluation rewards: {eval_rewards}")
    print(f"Mean evaluation reward: {np.mean(eval_rewards):.3f}")
    print(f"Mean final MSE: {np.mean(eval_mse):.6f}")
    
    # Validate that our implementation matches expectations
    print("\nValidating implementation...")
    env.reset()
    
    # Generate a sequence of actions
    actions = [np.random.uniform(-0.5, 0.5, size=env.action_space.shape) for _ in range(env.num_time_points)]
    
    # Apply actions and track states
    states = []
    for action in actions:
        next_state, _, _, _, _ = env.step(action)
        states.append(next_state.copy())
    
    # Validate against trajectory-based implementation
    validation_result = env.validate_against_trajectory()
    print(f"Final validation result: {validation_result}")
    
    return agent

if __name__ == "__main__":
    run_ppo_burgers_onestep() 