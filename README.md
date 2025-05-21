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
