# Burgers equation simulation package
# Contains both original and accelerated implementations

# Import from original implementation (for slow tasks like dataset generation and evaluation)
from .burgers_original import (
    # Dataset loading utils
    discounted_cumsum,
    get_squence_data,
    get_training_data_with_metadata,
    get_test_data,
    get_test_data_with_metadata,
    BurgersDataset,
    
    # Simulation utils (original for dataset generation)
    create_differential_matrices_1d,
    simulate_burgers_equation as simulate_burgers_equation_original,
    simulate_burgers_one_time_point as simulate_burgers_one_time_point_original,
    make_initial_conditions_and_varying_forcing_terms as make_initial_conditions_and_varying_forcing_terms_original,
    
    # Data generation utils
    generate_training_data,
    generate_test_data,
    save_training_data_hf,
    save_test_data_hf,
    generate_small_dataset_for_testing,
    generate_full_dataset,
    
    # Evaluation and testing utils
    burgers_solver,
    evaluate_model_performance,
    test_one_time_point_simulation,
    test_temporal_discretization_fix,
    
    # Random seed utils
    set_random_seeds
)

# Import from accelerated implementation (for fast tasks like training)
try:
    from .burgers_accelerated import (
        simulate_burgers_equation as simulate_burgers_equation_accelerated,
        simulate_burgers_one_time_point as simulate_burgers_one_time_point_accelerated,
        make_initial_conditions_and_varying_forcing_terms as make_initial_conditions_and_varying_forcing_terms_accelerated,
        
        # Test functions
        test_accelerated_vs_original,
        benchmark_functions
    )
    ACCELERATED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Accelerated functions not available: {e}")
    ACCELERATED_AVAILABLE = False
    
    # Fallback to original implementations
    simulate_burgers_equation_accelerated = simulate_burgers_equation_original
    simulate_burgers_one_time_point_accelerated = simulate_burgers_one_time_point_original
    make_initial_conditions_and_varying_forcing_terms_accelerated = make_initial_conditions_and_varying_forcing_terms_original

# Default exports for backward compatibility
simulate_burgers_equation = simulate_burgers_equation_original
simulate_burgers_one_time_point = simulate_burgers_one_time_point_original
make_initial_conditions_and_varying_forcing_terms = make_initial_conditions_and_varying_forcing_terms_original

__all__ = [
    # Dataset loading utils
    'discounted_cumsum',
    'get_squence_data', 
    'get_training_data_with_metadata',
    'get_test_data',
    'get_test_data_with_metadata',
    'BurgersDataset',
    
    # Simulation utils - both versions
    'create_differential_matrices_1d',
    'simulate_burgers_equation',
    'simulate_burgers_equation_original',
    'simulate_burgers_equation_accelerated',
    'simulate_burgers_one_time_point',
    'simulate_burgers_one_time_point_original', 
    'simulate_burgers_one_time_point_accelerated',
    'make_initial_conditions_and_varying_forcing_terms',
    'make_initial_conditions_and_varying_forcing_terms_original',
    'make_initial_conditions_and_varying_forcing_terms_accelerated',
    
    # Data generation utils
    'generate_training_data',
    'generate_test_data',
    'save_training_data_hf',
    'save_test_data_hf',
    'generate_small_dataset_for_testing',
    'generate_full_dataset',
    
    # Evaluation and testing utils
    'burgers_solver',
    'evaluate_model_performance',
    'test_one_time_point_simulation',
    'test_temporal_discretization_fix',
    
    # Random seed utils
    'set_random_seeds',
    
    # Testing and benchmarking
    'test_accelerated_vs_original',
    'benchmark_functions',
    'ACCELERATED_AVAILABLE'
] 