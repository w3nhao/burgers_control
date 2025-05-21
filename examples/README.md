# Burgers Equation Examples

This directory contains example scripts demonstrating various aspects of the Burgers equation simulation, data exploration, and control.

## Available Examples

- **data_exploration.py**: Simple exploration of the dataset structure and contents.
- **data_exploration_detailed.py**: Detailed analysis of tensor shapes, data types, and value ranges.
- **simulation_exploration.py**: Basic demonstration of the Burgers equation simulation functions.
- **advanced_simulation_demo.py**: Exploration of simulation parameters (viscosity, forcing terms) and demonstration of shock wave formation.
- **burgers_control_example.py**: Implementation of a simple control system for the Burgers equation.

## Running Examples

You can run the examples in several ways:

### Individual Examples

Run any example directly with Python:

```bash
python data_exploration.py
python simulation_exploration.py
python burgers_control_example.py
# etc.
```

### Using the Module Interface

This directory is also a Python module, allowing you to run examples using the module interface:

```bash
# From the parent directory
python -m examples data_exploration
python -m examples simulation_exploration
python -m examples burgers_control_example

# Run all examples
python -m examples all
```

## Example Output

- **burgers_control_example.py**: Generates a visualization saved to `examples/burgers_control_example.png`
- **advanced_simulation_demo.py**: Generates visualizations:
  - `examples/viscosity_comparison.png`: Comparison of different viscosity coefficients
  - `examples/forcing_comparison.png`: Comparison of different forcing terms
  - `examples/shock_formation.png`: Demonstration of shock wave formation

## Requirements

All examples require the main packages listed in the parent directory's README.md:
- PyTorch
- NumPy
- h5py
- SciPy
- tqdm
- matplotlib (for visualization) 