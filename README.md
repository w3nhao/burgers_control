# Burgers Equation PPO Control

A reinforcement learning system for controlling the 1D Burgers equation using Proximal Policy Optimization (PPO).

## Overview

This project implements PPO agents to control the 1D Burgers equation:

```
u_t + 0.5*(u²)_x = ν*u_xx + f(x,t)
```

where:
- `u` is the solution
- `ν` is the viscosity (0.01)
- `f(x,t)` represents control inputs/forcing terms

The system generates training data on-the-fly and trains policies to steer the PDE solution from initial states to target states.

## Installation

```bash
pip install -r requirements.txt
```

## Core Components

### Environment (`burgers_onthefly_env.py`)
- **BurgersOnTheFlyVecEnv**: Vectorized gymnasium environment that generates data on-the-fly
- Simulates Burgers equation with control inputs
- Goal-conditioned observations: `[current_state, target_state]`
- Action space: forcing terms applied to the PDE

### PPO Training (`ppo.py`)
- **Agent**: Neural network with actor-critic architecture
- Supports policy pretraining initialization
- Automatic checkpoint saving and loading
- Integration with Weights & Biases logging

### Policy Pretraining (`pretrain_policy.py`)
- **PolicyNetwork**: Supervised learning on state transitions
- Trains policy to predict actions from `(s_prev, s_next)` pairs
- Can be loaded as initialization for PPO training

### Simulation Engine (`burgers.py`)
- PDE simulation using finite difference methods
- Dataset generation and loading utilities
- Fallback support for HDF5 and Hugging Face datasets formats

## Quick Start

### 1. Train PPO Agent from Scratch

```bash
python ppo.py \
    --num_envs 8192 \
    --num_minibatches 512 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5
```

It is highly recommended to tune the `learning_rate`, `ent_coef`, `num_minibatches` and `update_epochs` to get the best performance.

### 2. Train with Policy Pretraining

First, pretrain the policy:
```bash
python pretrain_policy.py \
    --num_epochs 100 \
    --learning_rate 5e-4 \
    --batch_size 512
```

Then train PPO with the pretrained policy:
```bash
python ppo.py \
    --pretrained_policy_path pretrained_models/pretrain_policy__1__TIMESTAMP_best.pt \
    --policy_learning_rate_multiplier 0.5 \
    --total_timesteps 50000000
```

### 3. Evaluate Trained Agent

```bash
# Test on pre-generated test dataset
python eval_on_testset.py \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_trajectories 50

# Environment validation (uses training data actions)
python eval_on_testset.py --num_trajectories 10
```

## Key Features

### On-the-Fly Data Generation
- No need for pre-generated datasets
- Dynamic initial conditions and targets
- Configurable simulation parameters

### Multiple Reward Functions
- `vanilla`: Negative MSE to target
- `inverse_mse`: Inverse MSE reward
- `exp_scaled_mse`: Exponentially scaled MSE

### Checkpoint Management
- Automatic saving during training (`--save_every`)
- Complete training state preservation
- Easy model loading and evaluation

### Policy Pretraining
- Supervised learning on expert trajectories
- Compatible with PPO actor architecture
- Improved sample efficiency

## Configuration Options

### Environment Parameters
- `spatial_size`: Number of spatial grid points (default: 128)
- `num_time_points`: Time steps per episode (default: 10)
- `viscosity`: PDE viscosity coefficient (default: 0.01)
- `sim_time`: Physical simulation time (default: 0.1)
- `reward_type`: Reward function type (default: "exp_scaled_mse")

### PPO Hyperparameters
- `num_envs`: Number of parallel environments (default: 8192)
- `learning_rate`: Learning rate (default: 1e-5)
- `num_steps`: Steps per rollout (default: 10)
- `update_epochs`: PPO update epochs (default: 10)

### Model Architecture
- `hidden_dims`: MLP hidden layer sizes (default: [1024, 1024, 1024])
- `act_fn`: Activation function (default: "gelu")

## File Structure

```
.
├── burgers_onthefly_env.py    # Gymnasium environment
├── ppo.py                     # PPO training script
├── pretrain_policy.py         # Policy pretraining
├── burgers.py                 # PDE simulation engine
├── eval_on_testset.py         # Evaluation scripts
├── eval_on_env.py             # Environment evaluation
├── layers.py                  # Neural network layers
├── requirements.txt           # Dependencies
├── utils/
│   ├── save_load.py          # Model saving/loading utilities
│   └── utils.py              # General utilities
└── examples/                  # Example scripts (see examples/README.md)
```

## Environment Configuration

Create a `.env` file for Weights & Biases integration:

```bash
cp .env-example .env
# Edit .env with your wandb credentials
```

## Advanced Usage

### Custom Environment Configuration

```python
from burgers_onthefly_env import BurgersOnTheFlyVecEnv

env = BurgersOnTheFlyVecEnv(
    num_envs=1024,
    spatial_size=64,
    reward_type="inverse_mse",
    mse_scaling_factor=1e3
)
```

### Loading and Testing Saved Models

```python
from ppo import load_saved_agent

# Load agent
agent, metadata = load_saved_agent("path/to/checkpoint.pt")

# Check training info
print(f"Training iteration: {metadata['iteration']}")
print(f"Episode return: {metadata['episode_return_mean']}")
```

## Monitoring

Training metrics are logged to Weights & Biases:
- Episode returns and lengths
- Learning rates
- Policy and value losses
- KL divergence and entropy

## Performance Tips

1. **Batch Size**: Use large `num_envs` (4096+) for stable training
2. **Learning Rate**: Start with 1e-5, adjust based on training progress
3. **Pretraining**: Use policy pretraining for faster convergence
4. **Hardware**: Use GPU with adequate memory for large batch sizes

## Troubleshooting

### Validation

Run environment validation to ensure correct setup:
```bash
python eval_on_testset.py --num_trajectories 5
```

Expected output: MSE ≈ 0.0 (perfect simulation consistency)