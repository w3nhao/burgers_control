# Burgers Equation Simulation and Control

This repository contains code for simulating and controlling the 1D Burgers equation with forcing terms:

```
u_t + 0.5*(u²)_x = ν*u_xx + f(x,t)
```

where u is the solution, ν is the viscosity, and f(x,t) represents forcing terms or control inputs.

## Documentation

For complete documentation of this project, please refer to the [comprehensive documentation](./comprehensive_documentation.md).

## Data Files

The repository requires the following data files (not included in the repository):
- `../1d_burgers/burgers_train.h5` - Training data
- `../1d_burgers/unsafe_test.h5` - Test data

## File Structure

### Core Files
- `dataset.py` - Data loading and processing utilities
- `evaluation.py` - Simulation and evaluation functions
- `comprehensive_documentation.md` - Complete documentation

### Examples (in `examples/` directory)
- `data_exploration.py` - Script to explore dataset structure 
- `data_exploration_detailed.py` - Detailed tensor shape exploration
- `simulation_exploration.py` - Examples of simulation functions
- `burgers_control_example.py` - Example implementation of a controller
- `advanced_simulation_demo.py` - Advanced simulation features and parameter exploration

## Usage

### 1. Exploring Data

```python
from dataset import get_squence_data, train_file_path
data = get_squence_data(train_file_path)
print(data.keys())  # observations, actions, rewards, terminals, timeouts
```

### 2. Simulating Burgers Equation

```python
import torch
from evaluation import burgers_solver

# Create initial condition and forcing terms
initial_state = torch.randn(1, 128)  # Batch size 1, 128 spatial points
forcing_terms = torch.zeros(1, 10, 128)  # No forcing

# Run simulation
trajectory = burgers_solver(initial_state, forcing_terms, num_time_points=10)
```

### 3. Implementing Control

```python
from dataset import BurgersTest, test_file_path
from evaluation import burgers_solver

# Load test data
test_dataset = BurgersTest(test_file_path)
sample = test_dataset[0]
initial_state = torch.tensor(sample['observations'][0]).float().unsqueeze(0)
target_state = torch.tensor(sample['target']).float().unsqueeze(0)

# Create control inputs (see examples/burgers_control_example.py for full implementation)
forcing_terms = torch.zeros(1, 10, 128)  # Replace with actual control logic

# Run simulation with control
trajectory = burgers_solver(initial_state, forcing_terms, num_time_points=10)
```

## Running Examples

You can run the examples in several ways:

### Running Individual Examples

```bash
# Basic data exploration
python examples/data_exploration.py

# Detailed data exploration
python examples/data_exploration_detailed.py

# Basic simulation examples
python examples/simulation_exploration.py

# Control example with visualization
python examples/burgers_control_example.py

# Advanced simulation demonstrations
python examples/advanced_simulation_demo.py
```

### Using the Examples Module

The `examples` directory includes a module interface for easily running examples:

```bash
# List and run a specific example
python -m examples data_exploration

# Run the control example
python -m examples burgers_control_example 

# Run all examples
python -m examples all
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- h5py
- SciPy
- tqdm
- matplotlib (for visualization)
- python-dotenv (for environment variable management)

## Environment Configuration

This project uses dotenv for managing environment variables, particularly for Weights & Biases (wandb) integration.

1. Create a `.env` file in the project root based on the provided `.env-example`:

```bash
cp .env-example .env
```

2. Edit the `.env` file with your actual wandb credentials:

```
# Weights & Biases Configuration
WANDB_API_KEY=your_actual_api_key_here
WANDB_BASE_URL=https://api.wandb.ai  # Or your custom wandb URL if applicable
```

The environment variables will be loaded automatically when running any of the training scripts.


# Policy Pretraining System for Burgers Equation

This directory contains a policy pretraining system that allows you to first train a policy using supervised learning on the BurgersDataset, then continue training with PPO.

## Overview

The pretraining system consists of two main scripts:

1. **`pretrain_policy.py`**: Trains a policy network using supervised learning on state transition data
2. **`ppo_pretrain_policy.py`**: Modified PPO script that can load and continue training a pretrained policy

## How It Works

### Pretraining Phase (`pretrain_policy.py`)

The pretraining script creates a supervised learning dataset from the BurgersDataset by:

1. **Creating State Transition Pairs**: For each trajectory in the dataset, it creates tuples of `(s_prev, s_next, action)` where:
   - `s_prev`: State at time t
   - `s_next`: State at time t+1  
   - `action`: The action/forcing term that caused the transition from `s_prev` to `s_next`

2. **Training Objective**: The policy network learns to predict the action given the concatenated state transition `[s_prev, s_next]`

3. **Network Architecture**: Uses the same MLP architecture as the PPO actor network for compatibility

### PPO Fine-tuning Phase (`ppo_pretrain_policy.py`)

The modified PPO script can:

1. **Load Pretrained Policy**: Load the pretrained policy weights and use them as initialization for the actor network
2. **Separate Learning Rates**: Use different learning rates for policy (pretrained) and critic (from scratch) parameters
3. **Continue Training**: Continue training with PPO reinforcement learning

## Usage

### Step 1: Pretrain the Policy

```bash
# Basic pretraining with default parameters
python pretrain_policy.py

# Customize pretraining parameters
python pretrain_policy.py --learning_rate 1e-4 --batch_size 512 --num_epochs 50

# Limit training data for faster experimentation
python pretrain_policy.py --max_samples 10000

# Specify GPU device
python pretrain_policy.py --cuda 0
```

**Key Arguments for Pretraining:**
- `--learning_rate`: Learning rate for supervised learning (default: 1e-4)
- `--batch_size`: Batch size for training (default: 512)
- `--num_epochs`: Number of training epochs (default: 100)
- `--max_samples`: Limit number of state transition samples (default: None for all)
- `--train_split`: Fraction of data for training vs validation (default: 0.8)
- `--save_dir`: Directory to save models (default: "pretrained_models")

### Step 2: Continue Training with PPO

```bash
# Train PPO from scratch (baseline)
python ppo_pretrain_policy.py

# Train PPO with pretrained policy
python ppo_pretrain_policy.py --pretrained_policy_path pretrained_models/pretrain_policy__1__20240101_120000_best.pt

# Customize learning rates for policy vs critic
python ppo_pretrain_policy.py \
    --pretrained_policy_path pretrained_models/pretrain_policy__1__20240101_120000_best.pt \
    --policy_learning_rate_multiplier 0.1 \
    --critic_learning_rate_multiplier 1.0
```

**Key Arguments for PPO with Pretraining:**
- `--pretrained_policy_path`: Path to pretrained policy model (None for training from scratch)
- `--policy_learning_rate_multiplier`: LR multiplier for policy parameters (default: 0.1)
- `--critic_learning_rate_multiplier`: LR multiplier for critic parameters (default: 1.0)
- `--critic_hidden_dims`: Hidden dimensions for critic (None to use same as policy)
- `--critic_act_fn`: Activation function for critic (None to use same as policy)

## Expected Benefits

1. **Faster Convergence**: Pretrained policy should converge faster than training from scratch
2. **Better Sample Efficiency**: Policy starts with reasonable behavior learned from demonstrations
3. **More Stable Training**: Pretraining provides a good initialization point for RL

## File Structure

```
pretrained_models/           # Directory for saved pretrained models
├── pretrain_policy__1__20240101_120000_best.pt      # Best model during training
├── pretrain_policy__1__20240101_120000_final.pt     # Final model after training
└── pretrain_policy__1__20240101_120000_epoch_10.pt  # Periodic checkpoints
```

## Model Compatibility

The pretrained policy network has:
- **Input**: Concatenated state transition `[s_prev, s_next]` (shape: 2 × spatial_size)
- **Output**: Predicted action/forcing terms (shape: spatial_size)
- **Architecture**: Same MLP structure as PPO actor for seamless integration

## Monitoring Training

Both scripts log to Weights & Biases (wandb):
- **Pretraining**: Project "burgers_policy_pretrain"
- **PPO**: Project "ppo_continuous_action"

Key metrics to monitor:
- **Pretraining**: `train_loss`, `val_loss`, `learning_rate`
- **PPO**: `episode_return`, `policy_lr`, `critic_lr`, `r` (reward)

## Example Workflow

```bash
# 1. Pretrain policy for 50 epochs
python pretrain_policy.py --num_epochs 50 --learning_rate 1e-4

# 2. Find the best model path from the output
# Example: pretrained_models/pretrain_policy__1__20240101_120000_best.pt

# 3. Train PPO with pretrained policy
python ppo_pretrain_policy.py \
    --pretrained_policy_path pretrained_models/pretrain_policy__1__20240101_120000_best.pt \
    --policy_learning_rate_multiplier 0.1 \
    --num_envs 4096 \
    --total_timesteps 50000000

# 4. Compare with baseline (no pretraining)
python ppo_pretrain_policy.py \
    --num_envs 4096 \
    --total_timesteps 50000000
```

## Notes

- The pretraining splits the train dataset internally (80/20 by default) - it doesn't use the test dataset
- The critic is always trained from scratch since it's not pretrained
- Different learning rates for policy vs critic allow fine-tuning of pretrained vs randomly initialized components
- All models are saved with the `save_load` decorator for easy loading and metadata tracking 