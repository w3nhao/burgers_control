# Burgers Control: Reinforcement Learning for PDE Control

A Python package for controlling the 1D Burgers equation using Proximal Policy Optimization (PPO) with on-the-fly data generation and standard Gymnasium interfaces.

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

**Key Features:**
- ✅ **Standard Gymnasium Interface**: Compatible with `gym.make()` and ecosystem tools
- ✅ **High-Performance Vectorization**: Thousands of parallel environments with PyTorch
- ✅ **On-the-fly Data Generation**: No need for pre-generated datasets
- ✅ **Goal-Conditioned Control**: Flexible target specification
- ✅ **Multiple Reward Functions**: Various learning objectives

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
- PyTorch (≥2.0)
- Gymnasium 
- Weights & Biases
- TensorDict

## Quick Start

### Standard Gymnasium Interface

```python
import gymnasium as gym
import burgers_control  # Triggers environment registration

# Create single environment (standard interface)
env = gym.make("BurgersVec-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# List available environments
from burgers_control import list_registered_environments
print("Available environments:", list_registered_environments())
```

### High-Performance Vectorized Training

```python
from burgers_control import make_burgers_vec_env, get_environment_kwargs

# Method 1: Use preset configurations (recommended for fast training)
kwargs = get_environment_kwargs("BurgersVec-v4")  # Ultra-fast with random targets
vec_env = make_burgers_vec_env(
    num_envs=8192,     # Thousands of parallel environments
    **kwargs           # Apply preset: spatial_size=128, sim_time=0.1, use_random_targets=True
)

# Method 2: Use physics-accurate configuration for final evaluation
kwargs = get_environment_kwargs("BurgersVec-v1")  # Fast but physics-accurate
vec_env = make_burgers_vec_env(
    num_envs=1024,     # Fewer envs due to slower resets
    **kwargs           # Apply preset: use_random_targets=False
)

# Method 3: Create custom configuration
vec_env = make_burgers_vec_env(
    num_envs=8192,          # Thousands of parallel environments
    spatial_size=128,       # Spatial resolution
    num_time_points=10,     # Episode length
    sim_time=1.0,          # Physical simulation time
    use_random_targets=True, # Enable fast random targets (speedup)
    reward_type="exp_scaled_mse"
)

# Use with your favorite RL library
obs, info = vec_env.reset()
obs, rewards, terminated, truncated, info = vec_env.step(actions)
```

### Console Scripts (Recommended)

```bash
# Train PPO agent with ultra-fast random targets (recommended for development)
burgers-train \
    --env_id BurgersVec-v4 \
    --num_envs 8192 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5

# Train with physics-accurate targets (for final production models)
burgers-train \
    --env_id BurgersVec-v0 \
    --num_envs 1024 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5

# Evaluate trained agent
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --test_file_path /path/to/test/dataset \
    --num_trajectories 50

# Test agent in environment
burgers-eval-env \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_episodes 5
```

## Available Environments

| Environment ID | Spatial Size | Time Points | Sim Time | Random Targets | Description |
|---------------|--------------|-------------|----------|----------------|-------------|
| `BurgersVec-v0` | 128 | 10 | 1.0 | ❌ | Original settings (slower but physics-accurate) |
| `BurgersVec-v1` | 128 | 10 | 0.1 | ❌ | Faster simulation (speedup) |
| `BurgersVec-v3` | 128 | 10 | 1.0 | ✅ | **Random targets** (More speedup in resets) |
| `BurgersVec-v4` | 128 | 10 | 0.1 | ✅ | **Ultra-fast** random targets + fast simulation |
| `BurgersVec-debug` | 64 | 5 | 0.1 | ❌ | Small environment for testing |

### When to Use Each Environment

**Use Random Targets (V3/V4) for:**
- RL training (Much faster environment resets)
- Policy pretraining and development
- Large-scale experiments with many parallel environments
- Rapid prototyping and testing

**Use Original Targets (V0/V1/V2) for:**
- Final evaluation and validation
- Physics-accurate research requiring realistic target distributions
- Benchmarking against ground truth trajectories

### Environment Details

```python
from burgers_control import get_environment_kwargs

# Check environment parameters
params = get_environment_kwargs("BurgersVec-v0")
print(params)
# {'spatial_size': 128, 'num_time_points': 10, 'viscosity': 0.01, 
#  'sim_time': 1.0, 'time_step': 1e-4, 'reward_type': 'exp_scaled_mse', 
#  'use_random_targets': False, ...}

# Random targets environment (much faster)
params_v4 = get_environment_kwargs("BurgersVec-v4") 
print(params_v4)
# {'spatial_size': 128, 'num_time_points': 10, 'viscosity': 0.01,
#  'sim_time': 0.1, 'time_step': 1e-4, 'reward_type': 'exp_scaled_mse',
#  'use_random_targets': True, ...}  # Key difference!
```

**Key Parameters:**
- `use_random_targets`: When `True`, generates random target states directly instead of running expensive trajectory simulations
- `sim_time`: Physical simulation time - shorter times mean faster step computations
- `spatial_size`: Higher values = more detailed spatial resolution but slower computation
- `reward_type`: `"exp_scaled_mse"` (default), `"vanilla"`, or `"inverse_mse"`

## Adding Custom Environment Specifications

You can add new environment configurations in two ways:

### Method 1: Runtime Registration (Recommended)

```python
from burgers_control import add_environment_spec
import gymnasium as gym

# Add a custom high-resolution environment
add_environment_spec(
    "BurgersVec-highres",
    spatial_size=256,        # Higher spatial resolution
    num_time_points=20,      # Longer episodes  
    viscosity=0.005,         # Lower viscosity (more turbulent)
    sim_time=0.5,           # Custom simulation time
    use_random_targets=True, # Enable fast random targets for training
    reward_type="vanilla",   # Different reward function
    mse_scaling_factor=2e3
)

# Now use it like any registered environment
env = gym.make("BurgersVec-highres")

# Or get the preset for vectorized training
from burgers_control import get_environment_kwargs, make_burgers_vec_env
kwargs = get_environment_kwargs("BurgersVec-highres")
vec_env = make_burgers_vec_env(num_envs=1024, **kwargs)
```

### Method 2: Source Code Registration

Edit `burgers_control/register.py` and add to `register_burgers_environments()`:

```python
# Add this inside register_burgers_environments()
register(
    id="BurgersVec-custom",
    entry_point=make_burgers_env,
    kwargs={
        "spatial_size": 256,
        "num_time_points": 15,
        "viscosity": 0.02,
        "sim_time": 2.0,
        "reward_type": "inverse_mse",
        # ... other parameters
    },
    max_episode_steps=15,  # Should match num_time_points
)
```

## Core Components

### Environment System
- **`BurgersEnv`**: Single environment with standard Gymnasium interface
- **`BurgersOnTheFlyVecEnv`**: Vectorized environment for high-performance training
- **Automatic Registration**: Standard `gym.make()` interface
- **Goal-Conditioned**: Observations include both current and target states

### Training & Evaluation
- **PPO Agent**: Actor-critic with customizable architectures
- **Policy Pretraining**: Supervised learning initialization
- **Multiple Reward Functions**: `vanilla`, `inverse_mse`, `exp_scaled_mse`
- **Automatic Checkpointing**: Save/load with metadata

### Data Generation
- **On-the-fly Generation**: No pre-generated datasets required
- **Realistic Initial Conditions**: Dual Gaussian bumps with random parameters
- **Physical Forcing**: Spatio-temporal Gaussian forcing terms
- **Batch Processing**: Memory-efficient generation for large datasets

## Training Workflows

### 1. Basic PPO Training

```bash
# Ultra-fast training with V4 environment (recommended for development and most use cases)
burgers-train \
    --env_id BurgersVec-v4 \
    --num_envs 8192 \
    --total_timesteps 10000000 \
    --learning_rate 1e-5 \
    --save_every 100

# Fast training with V1 environment (physics-accurate but slower)
burgers-train \
    --env_id BurgersVec-v1 \
    --num_envs 4096 \
    --total_timesteps 10000000 \
    --learning_rate 1e-5 \
    --save_every 100

# Production training with V0 environment (full complexity but slowest)  
burgers-train \
    --env_id BurgersVec-v0 \
    --num_envs 1024 \
    --total_timesteps 50000000 \
    --learning_rate 1e-5
```

**Recommended Training Strategy:**
1. **Development**: Use `BurgersVec-v4` for rapid prototyping (faster resets)
2. **Validation**: Test on `BurgersVec-v1` for physics accuracy
3. **Final Training**: Optionally fine-tune on `BurgersVec-v0` for maximum realism

### 2. Training with Policy Pretraining

First, generate training data and pretrain the policy:
```bash
# Generate training data
python -m burgers_control.burgers \
    --mode full \
    --train_file "/path/to/burgers_train" \
    --test_file "/path/to/burgers_test" \
    --num_train_trajectories 50000

# Pretrain policy with supervised learning
burgers-pretrain \
    --train_file_path /path/to/burgers_train \
    --exp_name "policy_pretrain" \
    --num_epochs 500 \
    --learning_rate 5e-4
```

Then use pretrained policy to initialize PPO:
```bash
burgers-train \
    --env_id BurgersVec-v1 \
    --pretrained_policy_path pretrained_models/policy_pretrain_best.pt \
    --policy_learning_rate_multiplier 0.1 \
    --total_timesteps 50000000
```

### 3. Hyperparameter Tuning

```bash
burgers-train \
    --env_id BurgersVec-v1 \
    --num_envs 8192 \
    --learning_rate 3e-5 \
    --ent_coef 1e-4 \
    --num_minibatches 512 \
    --update_epochs 20 \
    --clip_coef 0.1 \
    --hidden_dims 1024 1024 1024 \
    --act_fn gelu
```

It is highly recommended to tune the `learning_rate`, `ent_coef`, `num_minibatches` and `update_epochs` to get the best performance.

## Evaluation

### Test Dataset Evaluation

```bash
# Evaluate agent on test dataset (final_state mode - default)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --test_file_path /path/to/test/dataset \
    --num_trajectories 50 \
    --mode final_state

# More challenging evaluation (next_state mode)
burgers-eval \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --test_file_path /path/to/test/dataset \
    --mode next_state
```

**Evaluation Modes:**
- `final_state`: Agent knows the final target throughout the trajectory (easier)
- `next_state`: Agent must follow intermediate states step-by-step (harder)

### Environment Validation

```bash
# Environment consistency check (should give MSE ≈ 0)
burgers-eval \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 50
```

### Interactive Testing

```bash
# Test agent in live environment
burgers-eval-env \
    --checkpoint_path checkpoints/run_name/agent_final.pt \
    --num_episodes 10 \
    --device auto
```

## Data Generation (Optional)

While the package generates data on-the-fly for training, you can also generate datasets for evaluation and pretraining:

### Quick Dataset Generation

```bash
# Small dataset for development
python -m burgers_control.burgers \
    --mode small \
    --train_file "/path/to/burgers_train_small" \
    --test_file "/path/to/burgers_test_small" \
    --validate

# Full production dataset
python -m burgers_control.burgers \
    --mode full \
    --train_file "/path/to/burgers_train_full" \
    --test_file "/path/to/burgers_test_full" \
    --batch_size 8192
```

### Custom Dataset Parameters

```bash
python -m burgers_control.burgers \
    --mode full \
    --train_file "/path/to/custom_train" \
    --test_file "/path/to/custom_test" \
    --num_train_trajectories 50000 \
    --num_test_trajectories 25 \
    --spatial_size 256 \
    --viscosity 0.02 \
    --sim_time 0.8
```

## Advanced Usage

### Custom Environment Creation

```python
from burgers_control import make_burgers_vec_env, get_environment_kwargs

# Method 1: Start with a fast preset and modify
kwargs = get_environment_kwargs("BurgersVec-v4")  # Get ultra-fast preset
kwargs.update({
    "spatial_size": 256,      # Higher resolution
    "num_time_points": 20,    # Longer episodes
    "viscosity": 0.005,       # Lower viscosity (more turbulent)
})
custom_env = make_burgers_vec_env(num_envs=1024, **kwargs)

# Method 2: Create entirely custom environment with random targets (fast)
fast_custom_env = make_burgers_vec_env(
    num_envs=8192,              # Many parallel environments
    spatial_size=256,           # Higher resolution
    num_time_points=20,         # Longer episodes
    viscosity=0.005,            # Lower viscosity (more turbulent)
    sim_time=2.0,              # Longer physical time
    use_random_targets=True,    # Enable fast random targets (speedup)
    reward_type="inverse_mse",  # Different reward function
    mse_scaling_factor=2e3
)

# Method 3: Create physics-accurate environment (slower but realistic)
accurate_custom_env = make_burgers_vec_env(
    num_envs=512,               # Fewer environments due to slower resets
    spatial_size=256,           # Higher resolution
    num_time_points=20,         # Longer episodes
    viscosity=0.005,            # Lower viscosity (more turbulent)
    sim_time=2.0,              # Longer physical time
    use_random_targets=False,   # Physics-accurate targets (slow but realistic)
    reward_type="inverse_mse",  # Different reward function
    mse_scaling_factor=2e3
)

# Method 4: Register and reuse custom configuration
from burgers_control import add_environment_spec
add_environment_spec(
    "BurgersVec-mytask",
    spatial_size=256,
    num_time_points=20,
    viscosity=0.005,
    sim_time=2.0,
    use_random_targets=True,    # Fast for training
    reward_type="inverse_mse"
)

# Now reuse this configuration anywhere
mytask_kwargs = get_environment_kwargs("BurgersVec-mytask")
env1 = make_burgers_vec_env(num_envs=64, **mytask_kwargs)    # Small for testing
env2 = make_burgers_vec_env(num_envs=8192, **mytask_kwargs)  # Large for training
```

### Load and Test Agents

```python
from burgers_control.ppo import load_saved_agent
import torch

# Load trained agent
agent, metadata = load_saved_agent("checkpoint.pt")
agent.eval()

# Create environment and test
import gymnasium as gym
env = gym.make("BurgersVec-v0")
obs, _ = env.reset()

# Get action (deterministic)
with torch.no_grad():
    action = agent.actor_mean(torch.tensor(obs, dtype=torch.float32))
```

### Integration with Other RL Libraries

```python
# Use with Stable-Baselines3
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("BurgersVec-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Use with CleanRL patterns
from burgers_control import make_burgers_vec_env
vec_env = make_burgers_vec_env(num_envs=64)  # Standard CleanRL setup
```

## Package Structure

```
burgers_control/
├── __init__.py                 # Package initialization with Gymnasium registration
├── burgers.py                  # PDE simulation and dataset generation
├── burgers_onthefly_env.py     # Vectorized environment (core implementation)
├── register.py                 # Gymnasium environment registration + legacy compatibility
├── ppo.py                      # PPO training script
├── pretrain_policy.py          # Policy pretraining with supervised learning
├── eval_on_testset.py          # Test dataset evaluation
├── eval_on_env.py              # Environment evaluation
├── layers.py                   # Neural network layers
├── utils/                      # Utility modules
│   ├── save_load.py           # Model persistence with metadata
│   └── utils.py               # General utilities
└── mmdit/                      # MMDIT neural architectures
    ├── adaptive_attention.py
    ├── mmdit_generalized_pytorch.py
    └── mmdit_pytorch.py
```

## Monitoring and Logging

Training automatically integrates with Weights & Biases:

```bash
# Set up W&B (optional)
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your_entity
```

Logged metrics include:
- Episode returns and lengths
- Learning rates (policy & critic)
- Policy and value losses
- KL divergence and entropy
- Training speed (steps/second)

## Environment Variables

Create a `.env` file for configuration:

```bash
cp .env-example .env
```

Edit `.env`:
```bash
# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
WANDB_ENTITY=your_entity
```

## Troubleshooting

### Quick Verification

```bash
# Test package installation
python -c "import burgers_control; print('✓ Package installed')"

# Test Gymnasium registration
python -c "import gymnasium as gym; import burgers_control; env = gym.make('BurgersVec-v0'); print('✓ Gymnasium integration working')"

# Test environment consistency
burgers-eval \
    --train_file_path /path/to/training/dataset \
    --num_trajectories 5
# Expected: MSE ≈ 0.0
```

### Common Issues

1. **Import errors**: Ensure installed with `pip install -e .`
2. **Slow training**: Use `BurgersVec-v4` for faster simulation or enable compilation with `--compile`
3. **Memory issues**: Reduce `num_envs` or use smaller environments (`BurgersVec-debug`)
4. **Environment not found**: Check `burgers_control.list_registered_environments()`

## Citation

If you use this package in your research, please cite:

```bibtex
@software{burgers_control,
  title={Burgers Control: Reinforcement Learning for PDE Control},
  author={Wenhao Deng},
  year={2024},
  url={https://github.com/w3nhao/burgers-control}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.