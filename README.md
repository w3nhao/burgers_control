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



# PPO Agent Checkpoint Save/Load Functionality

This document explains how to use the checkpoint save/load functionality that has been added to the PPO training script.

## Overview

The PPO agent now supports automatic saving and loading of checkpoints during training, including:
- Model weights (actor and critic networks)
- Optimizer state
- Training metadata (iteration, global step, episode returns, etc.)
- Complete training configuration

## Save Configuration

The following arguments control the saving behavior:

### Command Line Arguments

- `--save_every`: Save agent every N iterations (default: 1000, set to 0 to disable periodic saving)
- `--save_dir`: Directory to save agent checkpoints (default: "checkpoints")
- `--save_final`: Whether to save the final agent at the end of training (default: True)

### Examples

```bash
# Save every 500 iterations to custom directory
python ppo.py --save_every 500 --save_dir my_checkpoints

# Disable periodic saving, only save final model
python ppo.py --save_every 0 --save_final True

# Save every 2000 iterations
python ppo.py --save_every 2000
```

## Checkpoint Structure

Checkpoints are saved in the following directory structure:

```
checkpoints/
└── BurgersVec-v0__ppo__1__20240101_120000/
    ├── agent_iteration_1000.pt
    ├── agent_iteration_2000.pt
    ├── agent_iteration_3000.pt
    └── agent_final.pt
```

Each checkpoint file contains:
- Model state dict (actor_mean, actor_logstd, critic)
- Optimizer state dict
- Training metadata (iteration, global step, episode returns)
- Complete training arguments for reproducibility
- Version information for compatibility checking

## Loading Saved Agents

### Using the Helper Function

```python
from ppo import load_saved_agent

# Load a saved agent
agent, metadata = load_saved_agent("checkpoints/run_name/agent_final.pt")

# Print training information
print(f"Loaded agent from iteration {metadata['iteration']}")
print(f"Episode return mean: {metadata['episode_return_mean']}")
```

### Manual Loading

```python
from ppo import Agent

# Load agent using the built-in method
agent, metadata = Agent.init_and_load("path/to/checkpoint.pt", device="cuda:0")
```

## Testing Saved Agents

Use the provided example script to test saved agents:

```bash
# Test a saved agent
python load_agent_example.py --checkpoint_path checkpoints/run_name/agent_final.pt

# Test with specific device and number of episodes
python load_agent_example.py \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --device cuda:0 \
    --num_episodes 10
```

## Resuming Training

To resume training from a checkpoint, you would need to:

1. Load the saved agent
2. Extract the optimizer state
3. Continue training from the saved iteration

Example pattern:

```python
# Load checkpoint
agent, metadata = load_saved_agent("path/to/checkpoint.pt")

# Extract training state
start_iteration = metadata['iteration']
saved_args = metadata['args']

# Resume training from this point
# (implementation depends on your specific requirements)
```

## Best Practices

1. **Regular Saving**: Use `--save_every 1000` or similar to save regularly during long training runs
2. **Backup Important Checkpoints**: The save system creates backups automatically, but consider manual backups for critical runs
3. **Version Compatibility**: The save system includes version information to help with compatibility across different code versions
4. **Storage Management**: Monitor disk usage, as checkpoints can be large (especially with large networks and optimizer states)

## Metadata Information

Each checkpoint includes comprehensive metadata:

```python
metadata = {
    'iteration': 5000,
    'global_step': 1280000,
    'episode_return_mean': 45.67,
    'version': '1.0.0',
    'torch_version': '2.0.1',
    'args': {
        'spatial_size': 128,
        'hidden_dims': [1024, 1024, 1024],
        'learning_rate': 1e-5,
        # ... all training arguments
    }
}
```

This ensures you can always reproduce the exact training configuration and understand the context of any saved model. 


# Burgers Equation Dataset Generation

## Overview

The Burgers equation simulation and dataset generation functionality has been integrated into `burgers.py`. The system now uses Hugging Face's `datasets` library for data storage and loading, with automatic fallback to HDF5 format for backward compatibility.

## Installation

Make sure you have the required dependencies:
```bash
pip install datasets torch numpy scipy h5py tqdm
```

## Usage

### Command Line Interface

The `burgers.py` file now supports three modes of operation:

```bash
# Test mode - runs simulation validation (default)
python burgers.py --mode test

# Small dataset mode - generates 100 training + 10 test trajectories for testing
python burgers.py --mode small --validate

# Full dataset mode - generates 100k training + 50 test trajectories  
python burgers.py --mode full
```

### Modes Explained

#### 1. Test Mode (`--mode test`)
- Runs the original simulation validation test
- Verifies that step-by-step simulation matches full simulation
- Quick sanity check for the simulation implementation

#### 2. Small Dataset Mode (`--mode small`)
- Generates 100 training trajectories + 10 test trajectories
- Useful for testing and validation
- Add `--validate` flag to automatically run environment check
- Saves data to:
  - `../1d_burgers/burgers_train_small`
  - `../1d_burgers/unsafe_test_small`

#### 3. Full Dataset Mode (`--mode full`)
- Generates 100,000 training trajectories + 50 test trajectories
- Production dataset for training PPO agents
- Saves data to:
  - `../1d_burgers/burgers_train_new`
  - `../1d_burgers/unsafe_test_new`

### Dataset Format

The new datasets use Hugging Face's `datasets` library format:

```python
from datasets import load_from_disk

# Load training dataset
dataset = load_from_disk("../1d_burgers/burgers_train_new")
dataset.set_format("torch")  # Convert to PyTorch tensors

# Access data
trajectories = dataset['trajectories']  # Shape: (N, T+1, spatial_size)
actions = dataset['actions']            # Shape: (N, T, spatial_size)
```

### Simulation Parameters

All datasets are generated with consistent simulation parameters:
- **Viscosity**: 0.01
- **Simulation time**: 0.1 seconds
- **Time step**: 1e-4
- **Time points**: 10
- **Spatial size**: 128 points
- **Scaling factor**: 1.0

### Backward Compatibility

The system automatically falls back to HDF5 format if Hugging Face datasets are not found:

```python
# This works with both formats
from burgers import get_squence_data, get_test_data

train_data = get_squence_data()  # Tries datasets format first, falls back to HDF5
test_data = get_test_data()      # Same fallback mechanism
```

## File Structure

```
../1d_burgers/
├── burgers_train.h5              # Original HDF5 training data (legacy)
├── unsafe_test.h5                # Original HDF5 test data (legacy)
├── burgers_train_new/            # NEW: Hugging Face datasets training data
├── unsafe_test_new/              # NEW: Hugging Face datasets test data
├── burgers_train_small/          # Small test dataset (training)
└── unsafe_test_small/            # Small test dataset (test)
```

## Environment Validation

Use `eval_on_testset.py` to validate the environment implementation:

```bash
# Environment check using training data actions
python eval_on_testset.py --num_trajectories 5

# Agent evaluation (if you have a trained agent)
python eval_on_testset.py --checkpoint_path path/to/agent.pt --num_trajectories 50
```

## Data Generation Functions

The following functions are available for programmatic use:

```python
from burgers import (
    generate_training_data,
    generate_test_data, 
    save_training_data_hf,
    save_test_data_hf,
    generate_small_dataset_for_testing,
    generate_full_dataset
)

# Generate custom datasets
u_data, f_data = generate_training_data(num_trajectories=1000)
test_data = generate_test_data(num_trajectories=20)

# Save in Hugging Face format
save_training_data_hf(u_data, f_data, "path/to/train_dataset")
save_test_data_hf(test_data, "path/to/test_dataset")
```

## Migration from HDF5

If you're migrating from the old HDF5 format:

1. Generate new datasets: `python burgers.py --mode full`
2. The system automatically uses new datasets (paths updated in `burgers.py`)
3. Old HDF5 files remain as backup
4. All existing code continues to work due to automatic fallback

## Performance Notes

- **Small dataset**: ~1 second generation time
- **Full dataset**: ~10-30 minutes depending on hardware
- **Memory usage**: Batched processing (1000 trajectories per batch)
- **Storage**: Datasets format includes compression and metadata

## Validation Results

Perfect consistency validation (MSE = 0.0) confirms:
- ✅ Environment implementation is correct
- ✅ Generated data matches simulation parameters exactly
- ✅ Step-by-step simulation matches reference solver
- ✅ Training data will be consistent for agent training

## Next Steps

1. Generate full dataset: `python burgers.py --mode full`
2. Validate environment: `python eval_on_testset.py`
3. Train your PPO agent with consistent data
4. Enjoy improved training performance! 


# Burgers Equation Simulation and Control

## Overview

This project provides a comprehensive set of tools for simulating and controlling the 1D Burgers equation with forcing terms. The governing equation is:

```
u_t + 0.5*(u²)_x = ν*u_xx + f(x,t)
```

Where:
- `u` is the solution
- `ν` is the viscosity
- `f(x,t)` represents forcing terms or control inputs

This documentation combines information from all project documentation files to provide a complete reference for working with the codebase.

## Data Files

The project works with the following data files (not included in the repository):
- **Training Data**: `../1d_burgers/burgers_train.h5`
- **Test Data**: `../1d_burgers/unsafe_test.h5`

### HDF5 Data Structure

**Training File Structure**:
```
train/
  └── pde_11-128 - Shape: [N, T, s] - PDE solution data
  └── pde_11-128_f - Shape: [N, T, s] - Forcing terms
```

**Test File Structure**:
```
test/ - Shape: [N, T, s] - Test data with initial and final states
```

Where:
- `N`: Number of samples (40,000)
- `T`: Number of time steps (typically 11)
- `s`: Spatial grid size (128 points)

## Directory Structure

```
./
├── dataset.py                    # Data loading and processing utilities
├── evaluation.py                 # Simulation and evaluation functions
├── api_documentation.md          # Basic API documentation
├── api_tensor_documentation.md   # Comprehensive API documentation
├── README.md                     # Project overview and usage instructions
├── summary.md                    # Project summary
├── changes_summary.md            # Code reorganization summary
├── comprehensive_documentation.md # This file - complete documentation
└── examples/                     # Example scripts and demonstrations
    ├── __init__.py                      # Package initialization
    ├── __main__.py                      # Module interface for running examples
    ├── data_exploration.py              # Simple data exploration
    ├── data_exploration_detailed.py     # Detailed tensor analysis
    ├── simulation_exploration.py        # Basic simulation examples
    ├── burgers_control_example.py       # Control system implementation
    └── advanced_simulation_demo.py      # Advanced simulation features
```

## API Reference

### Data Loading Functions

#### `get_squence_data(file_path)`

Loads and processes data from HDF5 files into RL-style format.

**Input**: 
- `file_path`: Path to HDF5 file

**Output**: Python dictionary with the following keys:
- `observations`: NumPy array of shape `(N*T, s)` - Flattened state observations
- `actions`: NumPy array of shape `(N*T, s)` - Flattened forcing terms
- `rewards`: NumPy array of shape `(N*T,)` - Negative MSE between final and intermediate states
- `terminals`: Boolean NumPy array of shape `(N*T,)` - Episode ending markers
- `timeouts`: Boolean NumPy array of shape `(N*T,)` - Always zeros in current implementation

Where:
- `N*T`: Total number of flattened state observations
- `s`: Spatial grid size (128 points)

#### `BurgersTest(torch.utils.data.Dataset)`

PyTorch dataset for test data.

**Constructor Input**: 
- HDF5 file path

**Item Output**: Dictionary with the following keys:
- `observations`: Tensor of shape `(T-1, s)` - States across time steps (excluding final)
- `rewards`: Tensor of shape `(T-1,)` - Rewards at each time step
- `returns`: Tensor of shape `(T-1,)` - Cumulative discounted returns
- `target`: Tensor of shape `(s,)` - Target final state

### Utility Functions

#### `discounted_cumsum(x, gamma)`

Computes discounted cumulative sums of values.

**Input**: 
- `x`: Tensor of shape `(T,)` - Values to accumulate (rewards/costs)
- `gamma`: Float - Discount factor [0,1]

**Output**: Tensor of shape `(T,)` - Discounted cumulative sums

### Simulation Functions

#### `create_differential_matrices_1d(grid_size, device='cpu')`

Creates finite difference matrices for PDE simulation.

**Input**:
- `grid_size`: Integer - Number of grid points (typically 128+2 with boundaries)
- `device`: String - Computation device

**Output**: Tuple of sparse matrices:
- `first_deriv`: `scipy.sparse.lil_matrix` of shape `(grid_size, grid_size)` - First derivative matrix
- `second_deriv`: `scipy.sparse.lil_matrix` of shape `(grid_size, grid_size)` - Second derivative matrix

#### `simulate_burgers_equation(initial_conditions, forcing_terms, viscosity, sim_time, time_step, num_time_points, mode)`

Simulates Burgers equation with given parameters.

**Input**:
- `initial_conditions`: Tensor of shape `(N, s)` - Initial states
- `forcing_terms`: Tensor of shape `(N, Nt, s)` - Forcing terms over time 
- `viscosity`: Float - Viscosity coefficient (default: 0.01)
- `sim_time`: Float - Total physical simulation time (default: 0.1)
- `time_step`: Float - Physical time step size (default: 1e-4)
- `num_time_points`: Integer - Number of time points to record
- `mode`: String - Simulation mode (cannot be 'const')

**Output**: Tensor of shape `(N, num_time_points+1, s)` - Simulated trajectories including initial state

#### `burgers_solver(initial_conditions, forcing_terms, num_time_points=10)`

Default solver with preset parameters.

**Input**:
- `initial_conditions`: Tensor of shape `(N, s)` - Initial states
- `forcing_terms`: Tensor of shape `(N, Nt, s)` - Forcing terms over time
- `num_time_points`: Integer - Number of time points to record (default: 10)

**Output**: Tensor of shape `(N, num_time_points+1, s)` - Simulated trajectories

**Preset Parameters**:
- `viscosity`: 0.01
- `sim_time`: 0.1
- `time_step`: 1e-4

#### `evaluate_model_performance(num_episodes, initial_state, target_state, actions, device)`

Computes MSE between final and target states.

**Input**:
- `num_episodes`: Integer - Number of evaluation episodes
- `initial_state`: Tensor of shape `(N, s)` - Initial states
- `target_state`: Tensor of shape `(N, s)` - Target states
- `actions`: Tensor of shape `(N, T, s)` - Actions/forcing terms
- `device`: String - Computation device ('cpu' or 'cuda')

**Output**: Float - Mean squared error between final state and target

### Control Implementation

#### `SimpleController(target_state, amplitude=0.1)`

Control class for computing forcing terms.

**Method**: `compute_control(current_state, spatial_size)`
- **Input**:
  - `current_state`: Tensor of shape `(N, s)` - Current state
  - `spatial_size`: Integer - Number of spatial grid points
- **Output**: Tensor of shape `(N, s)` - Control action as a forcing term

#### `visualize_trajectory(initial_state, trajectory, target_state, forcing_terms, title)`

Visualizes simulation results.

**Input**:
- `initial_state`: Tensor of shape `(N, s)` - Initial state
- `trajectory`: Tensor of shape `(N, T, s)` - Simulated trajectory
- `target_state`: Tensor of shape `(N, s)` - Target state
- `forcing_terms`: Tensor of shape `(N, T, s)` - Control inputs
- `title`: String - Plot title

**Output**: Saves visualization to "burgers_control_example.png"

## Simulation Details

### Numerical Constants
- **Spatial Domain**: [0.0, 1.0]
- **Default Viscosity**: 0.01
- **Default Simulation Time**: 0.1
- **Default Time Step**: 1e-4
- **Default Grid Size**: 128 spatial points

### Simulation Process
1. Initial conditions are padded with boundary points
2. First and second derivative matrices are constructed for finite difference approximation
3. Burgers equation is solved using explicit time stepping:
   - u_t + 0.5*(u²)_x = ν*u_xx + f(x,t)
4. Forcing terms provide control inputs at each time step
5. State evolution is recorded at specified intervals

### Performance Metrics
- Mean Squared Error (MSE) between final and target states
- Per-sample MSE for detailed analysis

## Usage Examples

### Data Exploration

```python
from dataset import get_squence_data, train_file_path
data = get_squence_data(train_file_path)
print(data.keys())  # observations, actions, rewards, terminals, timeouts
```

### Simulating Burgers Equation

```python
import torch
from evaluation import burgers_solver

# Create initial condition and forcing terms
initial_state = torch.randn(1, 128)  # Batch size 1, 128 spatial points
forcing_terms = torch.zeros(1, 10, 128)  # No forcing

# Run simulation
trajectory = burgers_solver(initial_state, forcing_terms, num_time_points=10)
```

### Implementing Control

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

## Example Descriptions

1. **data_exploration.py**
   - Simple script to load and print data

2. **data_exploration_detailed.py**
   - Detailed analysis of dataset structure
   - Prints shape, type, and value ranges of tensors
   - Explores both training and test data

3. **simulation_exploration.py**
   - Basic demonstration of simulation functions
   - Shows how to create differential matrices
   - Demonstrates using the burgers_solver

4. **advanced_simulation_demo.py**
   - In-depth exploration of simulation parameters
   - Compares different viscosities
   - Tests various forcing terms
   - Demonstrates shock formation in Burgers equation
   - Includes visualizations of results

5. **burgers_control_example.py**
   - Demonstrates a simple control system for Burgers equation
   - Implements a feedback controller
   - Shows how to compute control actions
   - Visualizes controlled system behavior

6. **__main__.py**
   - Module interface for running examples
   - Can run individual examples or all examples at once
   - Usage: `python -m examples [example_name|all]`

## Key Features

1. **Data Loading**
   - Loading HDF5 data files
   - Converting to PyTorch tensors
   - Processing data for RL-style format

2. **Numerical Simulation**
   - Finite difference methods for PDEs
   - Handling of boundary conditions
   - Explicit time stepping for Burgers equation

3. **Control System Implementation**
   - State feedback control
   - Proportional control
   - Iterative simulation for control computation

4. **Visualization**
   - State evolution over time
   - Error tracking
   - Control input visualization
   - Spatial gradient analysis

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- h5py
- SciPy
- tqdm
- matplotlib (for visualization)

## Recent Changes

### Code Reorganization

The codebase has been reorganized to improve clarity and maintainability:

1. **Directory Structure Changes**:
   - Created an `examples/` directory
   - Moved all exploration and example scripts to this directory
   - Added package files (`__init__.py` and `__main__.py`) to make examples runnable as a module

2. **Import Fixes**:
   - Updated imports in all example files to work from their new location
   - Added `sys.path` manipulation to each example to support importing from the parent directory

3. **Path Updates**:
   - Updated file paths in scripts that save output files
   - Ensured visualizations are saved in the correct locations

4. **Documentation Updates**:
   - Updated README.md with the new directory structure
   - Updated summary.md with detailed information
   - Created examples/README.md to document the examples directory
   - Updated file structure descriptions in all documentation

### New Features

1. **Examples as a Module**:
   - Can now run examples with `python -m examples <example_name>`
   - Can run all examples with `python -m examples all`
   - Command-line help with `python -m examples --help`

2. **Better Documentation**:
   - Added detailed docstrings in the `__init__.py` file
   - Improved examples README with usage instructions
   - Standardized file headers and comments

## Potential Extensions

Potential extensions to this project:
1. Implement more sophisticated control algorithms (e.g., LQR, MPC)
2. Create a neural network model to learn the dynamics
3. Develop reinforcement learning agents for optimal control
4. Extend to 2D Burgers equation
5. Implement adjoint methods for sensitivity analysis and gradient computation 