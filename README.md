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
# Train PPO agent from scratch with V0 configuration (default)
burgers-train \
    --env_id BurgersVec-v0 \
    --num_envs 8192 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5

# Train with V1 configuration (shorter simulation time)
burgers-train \
    --env_id BurgersVec-v1 \
    --num_envs 8192 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5

# Evaluate trained agent on test dataset (requires explicit test file path)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --test_file_path /path/to/test/dataset \
    --num_trajectories 50 \
    --mode final_state

# Environment validation using training data (requires explicit training file path)
burgers-eval \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 50

# Test agent in environment
burgers-eval-env \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_episodes 5

# Pretrain policy with supervised learning (requires explicit training file path)
burgers-pretrain \
    --train_file_path /path/to/training/dataset \
    --exp_name "policy_pretrain" \
    --num_epochs 500
```

It is highly recommended to tune the `learning_rate`, `ent_coef`, `num_minibatches` and `update_epochs` to get the best performance.

### 2. Using the Package API

```python
import burgers_control
from burgers_control import load_saved_agent
from burgers_control.env_configs import create_env

# Create environment using configuration system
env = create_env('BurgersVec-v0', num_envs=1024)  # Original settings
env_fast = create_env('BurgersVec-v1', num_envs=1024)  # Shorter simulation

# Or create environment directly (legacy method)
from burgers_control import BurgersOnTheFlyVecEnv
env_legacy = BurgersOnTheFlyVecEnv(num_envs=1024, spatial_size=128)

# Load trained agent
agent, metadata = load_saved_agent("path/to/checkpoint.pt")
```

### 3. Running Module Scripts

```bash
# Train PPO agent with default V0 configuration
python -m burgers_control.ppo --env_id BurgersVec-v0 --num_envs 8192

# Train PPO agent with V1 configuration (faster simulation)
python -m burgers_control.ppo --env_id BurgersVec-v1 --num_envs 8192

# Evaluate on test dataset (requires explicit --test_file_path)
python -m burgers_control.eval_on_testset \
    --checkpoint_path path/to/agent.pt \
    --test_file_path path/to/test/dataset

# Environment validation (requires explicit --train_file_path)
python -m burgers_control.eval_on_testset \
    --train_file_path path/to/training/dataset

# Environment evaluation 
python -m burgers_control.eval_on_env --checkpoint_path path/to/agent.pt

# Policy pretraining (requires explicit --train_file_path)
python -m burgers_control.pretrain_policy \
    --train_file_path path/to/training/dataset \
    --exp_name "pretrain_experiment"
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

## Data Generation

**Important Note:** All data generation commands require explicit output file paths specified as parameters. File paths are no longer configured via environment variables.

The package includes comprehensive tools for generating Burgers equation datasets with controlled trajectories and forcing terms. This is essential for training agents that can control PDE dynamics.

### Quick Start with Data Generation

```bash
# Generate small test dataset (100 training, 10 test trajectories)
python -m burgers_control.burgers \
    --mode small \
    --train_file "/path/to/save/burgers_train_small" \
    --test_file "/path/to/save/burgers_test_small" \
    --validate

# Generate full production dataset (100k training, 50 test trajectories)  
python -m burgers_control.burgers \
    --mode full \
    --train_file "/path/to/save/burgers_train_full" \
    --test_file "/path/to/save/burgers_test_full" \
    --batch_size 8192

# Generate custom dataset with specific parameters
python -m burgers_control.burgers \
    --mode full \
    --train_file "/path/to/save/custom_train" \
    --test_file "/path/to/save/custom_test" \
    --num_train_trajectories 50000 \
    --num_test_trajectories 25 \
    --spatial_size 256 \
    --viscosity 0.02 \
    --sim_time 0.8 \
    --seed 123

# Test simulation consistency (uses default temporary files)
python -m burgers_control.burgers --mode test
```

### Data Generation Modes

#### 1. Small Dataset (Development & Testing)
Generates a small dataset for development and validation:

```bash
python -m burgers_control.burgers \
    --mode small \
    --seed 42 \
    --validate \
    --train_file "../1d_burgers/burgers_train_small" \
    --test_file "../1d_burgers/unsafe_test_small"
```

**Parameters:**
- 100 training trajectories 
- 10 test trajectories
- Includes automatic validation with environment consistency check
- Fast generation for testing workflows

#### 2. Full Production Dataset
Generates the complete dataset for training production models:

```bash
python -m burgers_control.burgers \
    --mode full \
    --seed 42 \
    --batch_size 8192 \
    --train_file "../1d_burgers/burgers_train_new" \
    --test_file "../1d_burgers/unsafe_test_new"
```

**Parameters:**
- 100,000 training trajectories
- 50 test trajectories  
- Memory-efficient batch processing
- Full logging and progress tracking

#### 3. Simulation Validation
Tests that simulation algorithms are working correctly:

```bash
python -m burgers_control.burgers --mode test --seed 42
```

Validates that step-by-step simulation matches full trajectory simulation (should have MSE < 1e-5).

### Generated Data Structure

The data generation creates two main outputs:

#### Training Data Format
- **Trajectories**: `(N, T+1, spatial_size)` - State evolution over time
- **Actions**: `(N, T, spatial_size)` - Forcing terms at each time step  
- **Rewards**: `(N, T)` - Rewards for each transition
- **Targets**: `(N, spatial_size)` - Final target states

#### Test Data Format  
- **Trajectories**: `(N, T+1, spatial_size)` - Full state trajectories
- **Targets**: `(N, spatial_size)` - Target states (final states)
- Used for evaluation without actions (ground truth comparison)

### Data Generation Parameters

#### Dataset Size Parameters
```bash
--num_train_trajectories 100000  # Number of training trajectories (default: 100000)
--num_test_trajectories 50       # Number of test trajectories (default: 50)
```

#### Physical Parameters
```bash
--spatial_size 128          # Spatial grid points (default: 128)
--num_time_points 10        # Time steps per trajectory (default: 10)  
--viscosity 0.01            # PDE viscosity coefficient (default: 0.01)
--sim_time 1.0              # Physical simulation time (default: 1.0)
--time_step 1e-4            # Simulation time step (default: 1e-4)
```

#### Generation Parameters
```bash
--seed 42                   # Random seed for reproducibility
--batch_size 8192          # Trajectories per batch (memory management)
--train_file path/to/train # Custom training data path
--test_file path/to/test   # Custom test data path  
--log_file path/to/log     # Custom log file path
```

#### Custom Dataset Examples

**Small Dataset for Development:**
```bash
python -m burgers_control.burgers \
    --mode small \
    --num_train_trajectories 100 \
    --num_test_trajectories 10 \
    --spatial_size 64 \
    --sim_time 0.5 \
    --seed 42
```

**Custom Physical Parameters:**
```bash
python -m burgers_control.burgers \
    --mode full \
    --num_train_trajectories 50000 \
    --viscosity 0.02 \
    --spatial_size 256 \
    --num_time_points 20 \
    --sim_time 2.0
```

**High-Resolution Dataset:**
```bash
python -m burgers_control.burgers \
    --mode full \
    --spatial_size 512 \
    --num_time_points 50 \
    --time_step 5e-5 \
    --batch_size 1024  # Smaller batches for memory management
```

All parameters are stored in the dataset metadata for reproducibility and consistency checking.

### Data Storage Formats

The package supports two storage formats:

#### 1. Hugging Face Datasets (Default)
```python
from datasets import load_from_disk

# Load training data
dataset = load_from_disk("../1d_burgers/burgers_train_new")
dataset.set_format("torch")

trajectories = dataset['trajectories']  # State evolution
actions = dataset['actions']           # Forcing terms

# Access simulation parameters stored in metadata
print("Viscosity:", dataset[0]['viscosity'])
print("Simulation time:", dataset[0]['sim_time'])
print("Spatial size:", dataset[0]['spatial_size'])
print("Time step:", dataset[0]['time_step'])
```

**Metadata Storage**: All simulation parameters (viscosity, sim_time, time_step, spatial_size, num_time_points) are automatically stored with each trajectory for reproducibility and consistency validation.

#### 2. HDF5 Format (Fallback)
We are not going to use this format in the future.
```python
import h5py

# Load training data  
with h5py.File("../1d_burgers/burgers_train_new.h5", 'r') as hdf:
    u_data = hdf['train']['pde_11-128'][:]     # Trajectories
    f_data = hdf['train']['pde_11-128_f'][:]   # Actions
```

### Python API for Data Generation

#### Generate Custom Training Data
```python
from burgers_control.burgers import generate_training_data

u_data, f_data = generate_training_data(
    num_trajectories=1000,
    num_time_points=10,
    spatial_size=128,
    viscosity=0.01,
    sim_time=1.0,
    time_step=1e-4,
    seed=42,
    train_file_path="../data/custom_train",
    batch_size=500
)
```

#### Generate Test Data
```python
from burgers_control.burgers import generate_test_data

test_trajectories = generate_test_data(
    num_trajectories=50,
    num_time_points=10,
    spatial_size=128,
    viscosity=0.01,
    sim_time=1.0,
    time_step=1e-4,
    seed=42,
    test_file_path="../data/custom_test"
)
```

#### Generate Full Production Dataset
```python
from burgers_control.burgers import generate_full_dataset

train_file, test_file, log_file = generate_full_dataset(
    seed=42,
    num_train_trajectories=100000,
    num_test_trajectories=50,
    num_time_points=10,
    spatial_size=128,
    viscosity=0.01,
    sim_time=1.0,
    time_step=1e-4,
    batch_size=8192,
    train_file_path="../data/production_train",
    test_file_path="../data/production_test"
)
```

#### Load Data with File Paths
```python
from burgers_control.burgers import get_training_data_with_metadata, get_test_data_with_metadata

# Load training data (requires file path)
train_data, train_metadata = get_training_data_with_metadata("/path/to/training/dataset")

# Load test data (requires file path)
test_data, test_metadata = get_test_data_with_metadata("/path/to/test/dataset")

# Use BurgersDataset (requires file path parameters)
from burgers_control.burgers import BurgersDataset

# For training data
train_dataset = BurgersDataset(mode="train", train_file_path="/path/to/training/dataset")

# For test data  
test_dataset = BurgersDataset(mode="test", test_file_path="/path/to/test/dataset")
```

#### Evaluation Functions with File Paths
```python
from burgers_control.eval_on_testset import test_agent_on_dataset, test_environment_with_training_data

# Test agent on dataset (requires test file path)
mean_mse, all_mse_values = test_agent_on_dataset(
    agent=agent,
    agent_metadata=metadata,
    device=device,
    test_file_path="/path/to/test/dataset",
    num_trajectories=50,
    mode="final_state"
)

# Environment validation (requires training file path)
mean_mse, all_mse_values = test_environment_with_training_data(
    train_file_path="/path/to/training/dataset",
    device=device,
    num_trajectories=50
)
```

#### Policy Pretraining with File Paths
```python
from burgers_control.pretrain_policy import StateTransitionDataset

# Create dataset for pretraining (requires training file path)
dataset = StateTransitionDataset(
    train_file_path="/path/to/training/dataset",
    max_samples=10000
)
```

### Data Validation

#### Environment Consistency Check
```bash
# Should return MSE ≈ 0 (perfect consistency)
python -m burgers_control.eval_on_testset \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 5
```

#### Simulation Algorithm Validation  
```bash
# Validates step-by-step vs full simulation
python -m burgers_control.burgers --mode test
```

### Generated Data Characteristics

#### Initial Conditions
- **Dual Gaussian bumps**: Positive and negative Gaussian distributions
- **Spatial locations**: Randomly positioned in domain [0, 1]
- **Amplitudes**: Random magnitudes creating diverse initial states
- **Ensures variety**: Wide range of initial PDE states for robust training

#### Forcing Terms (Actions)
- **Spatio-temporal structure**: Gaussian distributions in both space and time
- **Multi-component**: Sum of multiple random Gaussian components
- **Controllable regions**: Option for partial spatial control
- **Realistic dynamics**: Physically meaningful forcing patterns

#### Simulation Quality
- **High-order finite differences**: 2nd order boundary conditions
- **Stable time stepping**: Euler method with small time steps
- **Boundary handling**: Proper zero Dirichlet boundary conditions
- **Validation**: Extensive consistency checking

### Memory Management

For large dataset generation:

```bash
# Use smaller batch sizes for limited memory
python -m burgers_control.burgers --mode full --batch_size 1024

# Monitor memory usage during generation
python -m burgers_control.burgers --mode full --batch_size 8192
```

The batch processing automatically manages memory by:
- Processing trajectories in chunks
- Releasing memory between batches  
- Concatenating results efficiently
- Providing progress tracking

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
    --train_file_path /path/to/training/dataset \
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

**Important Note:** All evaluation commands require explicit file path parameters. Use `--test_file_path` for agent evaluation on test datasets and `--train_file_path` for environment validation.

### Test Dataset Evaluation

The evaluation script supports two goal-conditioned modes:

#### Final State Mode (Default)
Uses the final target state as the goal for all time steps during evaluation.

```bash
# Evaluate agent on 50 test trajectories (final_state mode)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --test_file_path /path/to/test/dataset \
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
    --test_file_path /path/to/test/dataset \
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
burgers-eval \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 50
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

### Environment Configuration System

The package now uses a centralized environment configuration system defined in `burgers_control/env_configs.py`. Instead of specifying individual environment parameters, you select from pre-registered environment configurations:

#### Available Environment Configurations

- **`BurgersVec-v0`** (Original settings):
  - `spatial_size`: 128
  - `num_time_points`: 10
  - `viscosity`: 0.01
  - `sim_time`: 1.0
  - `time_step`: 1e-4
  - `forcing_terms_scaling_factor`: 1.0
  - `reward_type`: "exp_scaled_mse"
  - `mse_scaling_factor`: 1e3

- **`BurgersVec-v1`** (Shorter simulation time):
  - Same as V0, but with `sim_time`: 0.1 (10x faster simulation)

#### Creating Custom Environment Configurations

You can easily add your own environment configurations by editing `burgers_control/env_configs.py`. Here's how:

1. **Define a new configuration**:
```python
# Add to burgers_control/env_configs.py
register_env_config(
    "BurgersVec-v2",  # Your custom environment ID
    EnvironmentConfig(
        spatial_size=256,           # Higher resolution
        num_time_points=20,         # More time steps
        viscosity=0.005,           # Lower viscosity
        sim_time=2.0,              # Longer simulation
        time_step=5e-5,            # Smaller time step
        forcing_terms_scaling_factor=2.0,  # Stronger forcing
        reward_type="inverse_mse",  # Different reward function
        mse_scaling_factor=5e3,    # Different scaling
    )
)
```

2. **Register configurations for different scenarios**:
```python
# High-resolution configuration
register_env_config(
    "BurgersVec-HighRes",
    EnvironmentConfig(
        spatial_size=512,
        num_time_points=10,
        viscosity=0.01,
        sim_time=1.0,
        time_step=1e-4,
        forcing_terms_scaling_factor=1.0,
        reward_type="exp_scaled_mse",
        mse_scaling_factor=1e3,
    )
)

# Low-viscosity configuration (more turbulent)
register_env_config(
    "BurgersVec-Turbulent",
    EnvironmentConfig(
        spatial_size=128,
        num_time_points=15,
        viscosity=0.001,  # Much lower viscosity
        sim_time=0.5,
        time_step=1e-5,   # Smaller time step for stability
        forcing_terms_scaling_factor=0.5,
        reward_type="exp_scaled_mse",
        mse_scaling_factor=2e3,
    )
)

# Fast prototyping configuration
register_env_config(
    "BurgersVec-Fast",
    EnvironmentConfig(
        spatial_size=64,   # Lower resolution
        num_time_points=5, # Fewer steps
        viscosity=0.02,    # Higher viscosity for stability
        sim_time=0.1,      # Short simulation
        time_step=2e-4,    # Larger time step
        forcing_terms_scaling_factor=1.0,
        reward_type="inverse_mse",  # Simpler reward
        mse_scaling_factor=1e3,
    )
)
```

3. **Use your custom configurations**:
```bash
# Train with your custom high-resolution environment
burgers-train --env_id BurgersVec-HighRes --num_envs 4096

# Train with turbulent dynamics
burgers-train --env_id BurgersVec-Turbulent --num_envs 8192

# Quick prototyping with fast environment
burgers-train --env_id BurgersVec-Fast --num_envs 16384
```

#### Best Practices for Custom Configurations

**When to create custom configurations:**
- **Research experiments**: Different physical parameters (viscosity, simulation time)
- **Performance tuning**: Different spatial/temporal resolutions for speed vs. accuracy trade-offs
- **Specialized scenarios**: Different reward functions or scaling factors
- **Hardware constraints**: Smaller configurations for limited memory/compute

**Configuration naming conventions:**
- Use descriptive names: `BurgersVec-HighRes`, `BurgersVec-Turbulent`, `BurgersVec-Fast`
- Include version numbers for experiments: `BurgersVec-Exp1`, `BurgersVec-Exp2`
- Indicate key characteristics: `BurgersVec-LowVisc`, `BurgersVec-LongSim`

**Parameter considerations:**
- **Stability**: Lower viscosity requires smaller time steps (`time_step`)
- **Performance**: Higher resolution (`spatial_size`) needs more compute and memory
- **Training speed**: Shorter `sim_time` and fewer `num_time_points` for faster episodes
- **Reward tuning**: Different `reward_type` and `mse_scaling_factor` for learning dynamics

#### Usage

```bash
# Train with original settings (V0)
burgers-train --env_id BurgersVec-v0

# Train with shorter simulation time (V1)
burgers-train --env_id BurgersVec-v1
```

```python
# Programmatic usage
from burgers_control.env_configs import create_env, get_env_config, list_env_configs

# Create environments
env_v0 = create_env('BurgersVec-v0', num_envs=1024)
env_v1 = create_env('BurgersVec-v1', num_envs=1024)

# List all available configurations
configs = list_env_configs()
print(configs)

# Get specific configuration details
config = get_env_config('BurgersVec-v1')
print(f"V1 sim_time: {config.sim_time}")  # 0.1
```

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
├── env_configs.py              # Environment configuration registry
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
```

**Important Migration Note:** Dataset file paths are no longer configured via environment variables. All scripts now require explicit file path parameters:

- **Removed:** `BURGERS_TRAIN_FILE_PATH` and `BURGERS_TEST_FILE_PATH` environment variables
- **Required:** Use `--train_file_path` and `--test_file_path` command-line arguments
- **Migration:** Update your scripts to include explicit file path parameters

**Example migration:**
```bash
# Old approach (no longer supported)
export BURGERS_TRAIN_FILE_PATH="/data/burgers_train"
python -m burgers_control.eval_on_testset

# New approach (required)
python -m burgers_control.eval_on_testset \
    --train_file_path "/data/burgers_train"
```

This change ensures explicit data dependencies and makes scripts more portable and reproducible.

## Advanced Usage

### Python API Examples

```python
import torch
from burgers_control import load_saved_agent
from burgers_control.env_configs import create_env, get_env_config

# Create environment using configuration system (recommended)
env = create_env('BurgersVec-v0', num_envs=512)  # Original settings
env_fast = create_env('BurgersVec-v1', num_envs=512)  # Shorter simulation

# Create custom environment with overrides
env_custom = create_env('BurgersVec-v0', num_envs=512, 
                       reward_type="inverse_mse", 
                       mse_scaling_factor=2e3)

# Use custom registered configurations (if you added them to env_configs.py)
env_highres = create_env('BurgersVec-HighRes', num_envs=256)  # Fewer envs for high-res
env_turbulent = create_env('BurgersVec-Turbulent', num_envs=1024)

# Or create environment directly (legacy method)
from burgers_control import BurgersOnTheFlyVecEnv
env_legacy = BurgersOnTheFlyVecEnv(
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

### Working with Custom Configurations Programmatically

```python
from burgers_control.env_configs import (
    register_env_config, EnvironmentConfig, 
    get_env_config, list_env_configs, create_env
)

# Register a new configuration at runtime
register_env_config(
    "BurgersVec-Custom",
    EnvironmentConfig(
        spatial_size=256,
        num_time_points=15,
        viscosity=0.005,
        sim_time=1.5,
        time_step=7.5e-5,
        forcing_terms_scaling_factor=1.5,
        reward_type="vanilla",
        mse_scaling_factor=2e3,
    )
)

# List all available configurations
configs = list_env_configs()
print("Available environments:")
for env_id in configs.keys():
    config = get_env_config(env_id)
    print(f"  {env_id}: spatial_size={config.spatial_size}, sim_time={config.sim_time}")

# Create environment with your custom configuration
env = create_env('BurgersVec-Custom', num_envs=1024)

# Override specific parameters while keeping the base configuration
env_modified = create_env('BurgersVec-Custom', num_envs=512, 
                         spatial_size=128,  # Override to smaller size
                         viscosity=0.01)    # Override to higher viscosity
```

### Custom Training Loop

```python
from burgers_control.ppo import Agent, Args
from burgers_control.env_configs import create_env

# Custom configuration
args = Args(
    env_id='BurgersVec-v1',  # Use V1 for faster training
    num_envs=2048,
    learning_rate=5e-6,
    total_timesteps=1000000
)

# Create environment using configuration system
env = create_env(args.env_id, num_envs=args.num_envs)
agent = Agent(n_obs=env.single_observation_space.shape[0], 
              n_act=env.single_action_space.shape[0])

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
burgers-eval \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 5
```
Expected: MSE ≈ 0.0 (indicates perfect simulation consistency)

### Common Issues

1. **Slow training**: Enable compilation with `--compile`
2. **Package import errors**: Ensure installed with `pip install -e .`
3. **Verify package installation**: `python -c "import burgers_control; print('OK')"`
4. **Missing file path parameters**: All evaluation and pretraining scripts now require explicit `--train_file_path` or `--test_file_path` arguments
5. **Review logs**: `logs/` directory

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