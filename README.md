# Burgers Control: Reinforcement Learning for PDE Control

A Python package for controlling the 1D Burgers equation using Proximal Policy Optimization (PPO) with on-the-fly data generation.

## Overview

`burgers_control` implements reinforcement learning agents to control the 1D Burgers equation:

```
u_t + 0.5*(u²)_x = ν*u_xx + f(x,t)
```

where:
- `u` is the solution
- `ν` is the viscosity (default: 0.01)
- `f(x,t)` represents control inputs/forcing terms

The system generates training data on-the-fly and trains policies to steer PDE solutions from initial states to target states. The agents are goal-conditioned, receiving observations that include both the current state and target state, enabling flexible control towards specified goals.

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/w3nhao/burgers-control.git
cd burgers_control

# Install in development mode
pip install -e .
```

This installs the package and provides console scripts for easy usage.

### Dependencies

The package automatically installs required dependencies including:
- PyTorch (≥1.9.0)
- Gymnasium
- Weights & Biases
- TensorDict

## Quick Start

After installation, you can use the package in three ways:

### 1. Using Console Scripts (Recommended)

```bash
# Train PPO agent from scratch
burgers-train \
    --num_envs 8192 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5

# Evaluate trained agent on test dataset  
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_trajectories 50 \
    --mode final_state

# Test agent in environment
burgers-eval-env \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_episodes 5
```

It is highly recommended to tune the `learning_rate`, `ent_coef`, `num_minibatches` and `update_epochs` to get the best performance.

### 2. Using the Package API

```python
import burgers_control
from burgers_control import BurgersOnTheFlyVecEnv, load_saved_agent

# Create environment
env = BurgersOnTheFlyVecEnv(num_envs=1024, spatial_size=128)

# Load trained agent
agent, metadata = load_saved_agent("path/to/checkpoint.pt")
```

### 3. Running Module Scripts

```bash
# Train PPO agent
python -m burgers_control.ppo --num_envs 8192

# Evaluate on test dataset
python -m burgers_control.eval_on_testset --checkpoint_path path/to/agent.pt

# Environment evaluation 
python -m burgers_control.eval_on_env --checkpoint_path path/to/agent.pt
```

## Core Components

### Environment (`burgers_control.burgers_onthefly_env`)
- **BurgersOnTheFlyVecEnv**: Vectorized gymnasium environment
- Generates data on-the-fly with random initial conditions
- Goal-conditioned observations: `[current_state, target_state]`
- Multiple reward functions: vanilla, inverse_mse, exp_scaled_mse

### PPO Agent (`burgers_control.ppo`)
- **Agent**: Actor-critic neural network architecture
- Supports policy pretraining initialization
- Automatic checkpoint saving and loading
- Integration with Weights & Biases logging

### Policy Pretraining (`burgers_control.pretrain_policy`)
- **PolicyNetwork**: Supervised learning on state transitions
- Trains policy to predict actions from `(s_prev, s_next)` pairs
- Compatible with PPO agent initialization

### Simulation Engine (`burgers_control.burgers`)
- Finite difference PDE simulation
- Dataset generation and loading utilities
- Support for HDF5 and Hugging Face datasets

### Utilities (`burgers_control.utils`)
- Model saving/loading with metadata
- Logging configuration
- Environment variable management

## Training Workflows

### 1. Basic PPO Training

```bash
burgers-train \
    --exp_name "basic_ppo" \
    --num_envs 4096 \
    --total_timesteps 10000000 \
    --learning_rate 1e-5 \
    --save_every 100
```

### 2. Training with Policy Pretraining

First, pretrain the policy:
```bash
python -m burgers_control.pretrain_policy \
    --exp_name "pretrain_experiment" \
    --num_epochs 500 \
    --learning_rate 5e-4 \
    --batch_size 512 \
    --save_dir "pretrained_models"
```

Then use it to initialize PPO:
```bash
burgers-train \
    --exp_name "ppo_with_pretrain" \
    --pretrained_policy_path pretrained_models/pretrain_experiment__1__TIMESTAMP_best.pt \
    --policy_learning_rate_multiplier 0.1 \
    --total_timesteps 50000000
```

### 3. Hyperparameter Tuning

```bash
burgers-train \
    --num_envs 8192 \
    --learning_rate 3e-5 \
    --ent_coef 1e-4 \
    --num_minibatches 512 \
    --update_epochs 20 \
    --clip_coef 0.1
```

## Evaluation

### Test Dataset Evaluation

The evaluation script supports two goal-conditioned modes:

#### Final State Mode (Default)
Uses the final target state as the goal for all time steps during evaluation.

```bash
# Evaluate agent on 50 test trajectories (final_state mode)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_trajectories 50 \
    --mode final_state \
    --device cuda:0
```

#### Next State Mode
Uses the next state in the trajectory sequence as the target for each time step.

```bash
# Evaluate agent using next state as target (more challenging)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_trajectories 50 \
    --mode next_state \
    --device cuda:0
```

#### Mode Comparison
- **final_state**: Agent knows the final target throughout the trajectory (easier task)
- **next_state**: Agent must follow intermediate states step-by-step (harder task)

The next_state mode typically results in higher MSE as it requires more precise control at each timestep.

#### Environment Validation
```bash
# Environment validation (should give MSE ≈ 0)
burgers-eval --num_trajectories 10
```

### Interactive Environment Testing

```bash
# Test agent in environment for 10 episodes
burgers-eval-env \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_episodes 10 \
    --device auto
```

## Configuration Options

### Environment Parameters
- `spatial_size`: Spatial grid points (default: 128)
- `num_time_points`: Time steps per episode (default: 10)
- `viscosity`: PDE viscosity coefficient (default: 0.01)
- `sim_time`: Physical simulation time (default: 0.1)
- `reward_type`: Reward function ("vanilla", "inverse_mse", "exp_scaled_mse")

### Evaluation Parameters
- `mode`: Goal-conditioned evaluation mode ("final_state", "next_state")
  - `final_state`: Use final target state as goal for all time steps (default)
  - `next_state`: Use next state in sequence as target for each time step
- `num_trajectories`: Number of test trajectories to evaluate (default: 50)
- `device`: Computation device ("cuda:0", "cpu", "auto")

### PPO Hyperparameters
- `num_envs`: Parallel environments (default: 8192)
- `learning_rate`: Learning rate (default: 1e-5)
- `num_steps`: Steps per rollout (default: 10)
- `update_epochs`: PPO update epochs (default: 10)
- `clip_coef`: PPO clipping coefficient (default: 0.2)

### Model Architecture
- `hidden_dims`: MLP layers (default: [1024, 1024, 1024])
- `act_fn`: Activation function (default: "gelu")

## Package Structure

```
burgers_control/
├── __init__.py                 # Package initialization
├── burgers.py                  # PDE simulation and datasets  
├── burgers_onthefly_env.py     # Gymnasium environment
├── ppo.py                      # PPO training script
├── pretrain_policy.py          # Policy pretraining
├── eval_on_testset.py          # Test dataset evaluation
├── eval_on_env.py              # Environment evaluation
├── layers.py                   # Neural network layers
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── save_load.py           # Model persistence
│   └── utils.py               # General utilities
└── mmdit/                      # MMDIT neural architectures
    ├── __init__.py
    ├── adaptive_attention.py
    ├── mmdit_generalized_pytorch.py
    └── mmdit_pytorch.py

# Project root
├── setup.py                    # Package configuration
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── LICENSE                     # License information
├── .env-example                # Environment variables template
├── scripts/                    # Shell scripts
└── examples/                   # Example usage (ignored by git)
```

## Environment Variables

Create a `.env` file for configuration:

```bash
cp .env-example .env
```

Edit `.env` with your settings:
```bash
# Weights & Biases
WANDB_API_KEY=your_wandb_key
WANDB_ENTITY=your_entity

# Data paths (optional)
BURGERS_TRAIN_FILE_PATH=/path/to/training/data
BURGERS_TEST_FILE_PATH=/path/to/test/data
```

## Advanced Usage

### Python API Examples

```python
import torch
from burgers_control import BurgersOnTheFlyVecEnv, load_saved_agent

# Create custom environment
env = BurgersOnTheFlyVecEnv(
    num_envs=512,
    spatial_size=64,
    reward_type="exp_scaled_mse",
    mse_scaling_factor=2e3,
    viscosity=0.005
)

# Load and test agent
agent, metadata = load_saved_agent("checkpoint.pt")
agent.eval()

obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32)

# Get action (deterministic)
with torch.no_grad():
    action = agent.actor_mean(obs)
```

### Custom Training Loop

```python
from burgers_control.ppo import Agent, Args
from burgers_control.burgers_onthefly_env import BurgersOnTheFlyVecEnv

# Custom configuration
args = Args(
    num_envs=2048,
    learning_rate=5e-6,
    total_timesteps=1000000
)

# Create environment and agent
env = BurgersOnTheFlyVecEnv(num_envs=args.num_envs)
agent = Agent(n_obs=env.observation_space.shape[0], 
              n_act=env.action_space.shape[0])

# ... implement training loop
```

## Monitoring and Logging

Training automatically logs to Weights & Biases:
- Episode returns and lengths
- Learning rates (policy & critic)
- Policy and value losses  
- KL divergence and entropy
- Training speed (steps/second)

## Troubleshooting

### Validation Test
Run environment validation to ensure correct installation:
```bash
burgers-eval --num_trajectories 5
```
Expected: MSE ≈ 0.0 (indicates perfect simulation consistency)

### Common Issues

1. **Slow training**: Enable compilation with `--compile`
2. **Package import errors**: Ensure installed with `pip install -e .`
3. **Verify package installation**: `python -c "import burgers_control; print('OK')"`
4. **Review logs**: `logs/` directory

## Citation

If you use this package in your research, please cite:

```bibtex
@software{burgers_control,
  title={Burgers Control},
  author={Wenhao Deng},
  year={2025},
  url={https://github.com/w3nhao/burgers-control}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.