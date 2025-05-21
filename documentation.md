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