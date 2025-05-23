# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb

# Import utility function to load environment variables
from utils.utils import load_environment_variables
load_environment_variables()

from burgers_onthefly_env import BurgersOnTheFlyVecEnv
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.normal import Normal
from layers import MLP, get_activation_fn
from pretrain_policy import PolicyNetwork
from utils.save_load import save_load

import datetime

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: int = 1
    """cuda device to use"""

    # Environment specific arguments
    env_id: str = "BurgersVec-v0"
    """the id of the environment"""
    spatial_size: int = 128
    """the spatial size of the environment"""
    num_time_points: int = 10
    """the number of time points of trajectory in the environment"""
    viscosity: float = 0.01
    """the viscosity of the Burgers' equation"""
    sim_time: float = 0.1
    """the total time of simulation"""
    time_step: float = 1e-4
    """the time step of simulation"""
    forcing_terms_scaling_factor: float = 1.0
    """the scaling factor of the forcing terms"""
    reward_type: str = "exp_scaled_mse"
    """the type of reward function to use"""
    mse_scaling_factor: float = 1e3
    """the scaling factor of the MSE reward"""
    
    # Model specific arguments
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024])
    """the hidden dimensions of the MLP"""
    act_fn: str = "gelu"
    """the activation function of the MLP"""
    
    # Pretrained policy loading arguments
    pretrained_policy_path: Optional[str] = None
    """path to pretrained policy model (None to start from scratch)"""
    freeze_policy_layers: bool = False
    """whether to freeze some layers of the pretrained policy during training"""
    policy_learning_rate_multiplier: float = 1.0
    """learning rate multiplier for pretrained policy parameters"""
    
    # Critic specific arguments (since critic is not pretrained)
    critic_hidden_dims: Optional[List[int]] = None
    """hidden dimensions for critic network (None to use same as policy)"""
    critic_act_fn: Optional[str] = None
    """activation function for critic network (None to use same as policy)"""
    critic_learning_rate_multiplier: float = 1.0
    """learning rate multiplier for critic parameters"""
    
    # Algorithm specific arguments
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_envs: int = 8192
    """the number of parallel game environments"""
    num_steps: int = 10
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 256
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 1e-5
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""
    
    # Model saving arguments
    save_every: int = 100
    """save agent every N iterations (0 to disable saving)"""
    save_dir: str = "checkpoints"
    """directory to save agent checkpoints"""
    save_final: bool = True
    """whether to save the final agent at the end of training"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@save_load(version="1.0.0")
class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None, hidden_dims=None, act_fn=None, 
                 critic_hidden_dims=None, critic_act_fn=None,
                 pretrained_policy_path=None):
        super().__init__()
        
        # n_obs now includes both current state and target state (goal-conditioned)
        # n_obs = 2 * spatial_size (current_state + target_state)
        
        # Set default critic parameters if not provided
        if critic_hidden_dims is None:
            critic_hidden_dims = hidden_dims
        if critic_act_fn is None:
            critic_act_fn = act_fn
        
        # Use MLP for critic network (outputs 1 value)
        self.critic = MLP(
            in_dim=n_obs,
            out_dim=1,
            hidden_dims=critic_hidden_dims,
            act_fn=get_activation_fn(critic_act_fn),
            norm_fn=nn.Identity,
            dropout_rate=0.0,
            use_input_residual=True,
            use_bias=True
        )
        
        # Load pretrained policy or create new one
        if pretrained_policy_path is not None:
            print(f"Loading pretrained policy from {pretrained_policy_path}")
            # Load the pretrained policy network
            pretrained_policy, metadata = PolicyNetwork.init_and_load(
                pretrained_policy_path, 
                device=device
            )
            print(f"Loaded pretrained policy with metadata: {metadata}")
            
            # Use the pretrained policy network as actor_mean
            self.actor_mean = pretrained_policy.policy_net
            print("Successfully loaded pretrained policy network")
            
        else:
            print("Creating new policy network from scratch")
            # Use MLP for actor mean network (outputs n_act actions)
            self.actor_mean = MLP(
                in_dim=n_obs,
                out_dim=n_act,
                hidden_dims=hidden_dims,
                act_fn=get_activation_fn(act_fn),
                norm_fn=nn.Identity,
                dropout_rate=0.0,
                use_input_residual=True,
                use_bias=True
            )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))
        
        if device is not None:
            self.critic = self.critic.to(device)
            self.actor_mean = self.actor_mean.to(device)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def gae(next_obs, next_done, container):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        action, logprob, _, value = policy(obs=obs)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)
        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"].reshape(()))
                # max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            # desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    
    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

def load_saved_agent(checkpoint_path: str, device: Optional[torch.device] = None) -> Tuple[Agent, Dict[str, Any]]:
    """
    Load a saved PPO agent from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved agent checkpoint
        device: Device to load the agent on (if None, uses same device as saved)
        
    Returns:
        Tuple of (loaded_agent, metadata)
        
    Example:
        agent, metadata = load_saved_agent("checkpoints/run_name/agent_final.pt")
        print(f"Loaded agent from iteration {metadata['iteration']}")
    """
    agent, metadata = Agent.init_and_load(checkpoint_path, device=device)
    print(f"Loaded agent from {checkpoint_path}")
    print(f"Metadata: {metadata}")
    return agent, metadata

if __name__ == "__main__":
    # Load environment variables from .env file
    from utils.utils import load_environment_variables
    load_environment_variables()
    
    args = tyro.cli(Args)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    
    exp_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include pretrained policy info in run name
    pretrain_suffix = ""
    if args.pretrained_policy_path:
        pretrain_suffix = "_pretrained"
    
    run_name = f"{args.env_id}__{args.exp_name}{pretrain_suffix}__{args.seed}__{exp_start_time}"

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup save directory
    save_path = None
    if args.save_every > 0 or args.save_final:
        save_path = os.path.join(args.save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"Save directory: {save_path}")
    
    ####### Environment setup #######
    # Create vectorized environment
    if args.env_id == "BurgersVec-v0":
        env = BurgersOnTheFlyVecEnv(
            num_envs=args.num_envs,
            spatial_size=args.spatial_size,
            num_time_points=args.num_time_points,
            viscosity=args.viscosity,
            sim_time=args.sim_time,
            time_step=args.time_step,
            forcing_terms_scaling_factor=args.forcing_terms_scaling_factor,
            reward_type=args.reward_type,
            mse_scaling_factor=args.mse_scaling_factor
        )
        # Move environment to the same device as the neural network
        env.set_device(device)
    else:
        raise ValueError(f"Environment {args.env_id} not found")
    # Observation and action space dimensions
    n_obs = env.single_observation_space.shape[0]
    n_act = env.single_action_space.shape[0]

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action_tensor):
        action = action_tensor.cpu().numpy()
        next_obs, rewards, terminations, truncations, info = env.step(action)
        return (
            torch.tensor(next_obs, dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.logical_or(terminations, truncations), dtype=torch.bool, device=device),
            info
        )
        
    ####### Agent #######
    agent = Agent(
        n_obs, 
        n_act, 
        device=device, 
        hidden_dims=args.hidden_dims, 
        act_fn=args.act_fn,
        critic_hidden_dims=args.critic_hidden_dims,
        critic_act_fn=args.critic_act_fn,
        pretrained_policy_path=args.pretrained_policy_path
    )
    
    # Make a version of agent with detached params
    agent_inference = Agent(
        n_obs, 
        n_act, 
        device=device, 
        hidden_dims=args.hidden_dims, 
        act_fn=args.act_fn,
        critic_hidden_dims=args.critic_hidden_dims,
        critic_act_fn=args.critic_act_fn,
        pretrained_policy_path=args.pretrained_policy_path
    )
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    # Create separate parameter groups for policy and critic with different learning rates
    policy_params = []
    critic_params = []
    
    # Add actor parameters to policy group
    for name, param in agent.named_parameters():
        if 'actor_mean' in name or 'actor_logstd' in name:
            policy_params.append(param)
        elif 'critic' in name:
            critic_params.append(param)
        else:
            # Default to policy group for any other parameters
            policy_params.append(param)
    
    # Setup parameter groups with different learning rates
    param_groups = [
        {
            'params': policy_params, 
            'lr': torch.tensor(args.learning_rate * args.policy_learning_rate_multiplier, device=device),
            'name': 'policy'
        },
        {
            'params': critic_params, 
            'lr': torch.tensor(args.learning_rate * args.critic_learning_rate_multiplier, device=device),
            'name': 'critic'
        }
    ]
    
    optimizer = optim.Adam(
        param_groups,
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )
    
    print(f"Policy parameters: {len(policy_params)}")
    print(f"Critic parameters: {len(critic_params)}")
    print(f"Policy learning rate: {args.learning_rate * args.policy_learning_rate_multiplier}")
    print(f"Critic learning rate: {args.learning_rate * args.critic_learning_rate_multiplier}")

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        if device.type == 'cuda':
            policy = CudaGraphModule(policy, device=device)
            # gae = CudaGraphModule(gae, device=device)
            update = CudaGraphModule(update, device=device)
        else:
            print("Warning: CUDA graphs requested but CUDA is not available. Skipping CUDA graph optimization.")
            args.cudagraphs = False

    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    # max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    # desc = ""
    global_step_burnin = None
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            # Update learning rates for all parameter groups
            for group in optimizer.param_groups:
                if group['name'] == 'policy':
                    lrnow = frac * args.learning_rate * args.policy_learning_rate_multiplier
                else:  # critic
                    lrnow = frac * args.learning_rate * args.critic_learning_rate_multiplier
                group["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        if global_step_burnin is not None and iteration % 10 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            
            # Handle empty avg_returns to avoid NaN
            if len(avg_returns) > 0:
                avg_returns_t = torch.tensor(avg_returns).mean()
                episode_return_mean = np.array(avg_returns).mean()
            else:
                print("WARNING: No episodes completed yet")
                avg_returns_t = torch.tensor(0.0)
                episode_return_mean = 0.0

            with torch.no_grad():
                logs = {
                    "episode_return": episode_return_mean,
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "gn": out["gn"].mean(),
                }

            # Get learning rates for each group
            policy_lr = optimizer.param_groups[0]["lr"]
            critic_lr = optimizer.param_groups[1]["lr"]
            
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f} ({len(avg_returns)} episodes), "
                f"policy_lr: {policy_lr: 4.2f}, critic_lr: {critic_lr: 4.2f}"
            )
            wandb.log(
                {
                    "speed": speed, 
                    "episode_return": avg_returns_t, 
                    "r": r, 
                    "r_max": r_max, 
                    "policy_lr": policy_lr,
                    "critic_lr": critic_lr,
                    **logs
                }, 
                step=global_step
            )
            
        # Save agent checkpoint at specified intervals
        if args.save_every > 0 and iteration % args.save_every == 0 and save_path is not None:
            checkpoint_path = os.path.join(save_path, f"agent_iteration_{iteration}.pt")
            try:
                agent.save(
                    checkpoint_path,
                    save_optimizer=True,
                    optimizer=optimizer,
                    metadata={
                        "iteration": iteration,
                        "global_step": global_step,
                        "episode_return_mean": episode_return_mean if len(avg_returns) > 0 else 0.0,
                        "args": vars(args)
                    }
                )
                print(f"Saved agent checkpoint at iteration {iteration}: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at iteration {iteration}: {e}")
            
    # Save final agent checkpoint
    if args.save_final and save_path is not None:
        final_checkpoint_path = os.path.join(save_path, "agent_final.pt")
        try:
            agent.save(
                final_checkpoint_path,
                save_optimizer=True,
                optimizer=optimizer,
                metadata={
                    "iteration": args.num_iterations,
                    "global_step": global_step,
                    "episode_return_mean": episode_return_mean if len(avg_returns) > 0 else 0.0,
                    "args": vars(args),
                    "final_checkpoint": True
                }
            )
            print(f"Saved final agent checkpoint: {final_checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to save final checkpoint: {e}")
            
    env.close() 