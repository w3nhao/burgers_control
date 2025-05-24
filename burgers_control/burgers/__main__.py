#!/usr/bin/env python3
"""
Main entry point for the burgers_control.burgers module.

This allows the module to be called as:
    python -m burgers_control.burgers --mode full --train_file /path/to/train --test_file /path/to/test

This preserves the same interface described in the README.
"""

import argparse
import sys
import os

# Import everything needed from burgers_original
from .burgers_original import *

def main():
    """Main function that replicates the argparse logic from burgers_original.py"""
    parser = argparse.ArgumentParser(description="Burgers equation simulation and data generation")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "small", "full", "test_temporal"],
                       help="Mode: 'test' runs simulation test, 'small' generates small dataset, 'full' generates full dataset, 'test_temporal' runs temporal discretization tests")
    parser.add_argument("--validate", action="store_true", 
                       help="Run environment validation after generating small dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--train_file", type=str, default=None,
                       help="Path to save training data (default: auto-generated)")
    parser.add_argument("--test_file", type=str, default=None,
                       help="Path to save test data (default: auto-generated)")
    parser.add_argument("--batch_size", type=int, default=8192,
                       help="Number of trajectories to process at a time (default: 8192)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path to save generation log (default: auto-generated)")
    
    # Dataset generation parameters
    parser.add_argument("--num_train_trajectories", type=int, default=100000,
                       help="Number of training trajectories to generate (default: 100000)")
    parser.add_argument("--num_test_trajectories", type=int, default=50,
                       help="Number of test trajectories to generate (default: 50)")
    parser.add_argument("--num_time_points", type=int, default=10,
                       help="Number of time points in each trajectory (default: 10)")
    parser.add_argument("--spatial_size", type=int, default=128,
                       help="Number of spatial grid points (default: 128)")
    parser.add_argument("--viscosity", type=float, default=0.01,
                       help="Viscosity coefficient (default: 0.01)")
    parser.add_argument("--sim_time", type=float, default=1.0,
                       help="Total simulation time (default: 1.0)")
    parser.add_argument("--time_step", type=float, default=1e-4,
                       help="Time step for simulation (default: 1e-4)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        # Run the original test
        logger, actual_log_path = setup_logging(logger_name="burgers_generation", mode="test")
        
        log_info("Running simulation validation test...")
        # Set seed for test reproducibility
        set_random_seeds(args.seed)
        test_one_time_point_simulation(args.seed)
        log_info(f"Test completed successfully. Log saved to: {actual_log_path}")
        
    elif args.mode == "test_temporal":
        # Run the new temporal discretization tests
        logger, actual_log_path = setup_logging(logger_name="burgers_generation", mode="temporal_test")
        
        log_info("Running temporal discretization fix tests...")
        set_random_seeds(args.seed)
        
        # Run comprehensive test
        test_temporal_discretization_fix(args.seed)
        
        log_info(f"All temporal tests completed successfully. Log saved to: {actual_log_path}")
        
    elif args.mode == "small":
        # Generate small dataset for testing - use smaller defaults if not specified
        small_train = min(args.num_train_trajectories, 1000)  # Cap at 1000 for small dataset
        small_test = min(args.num_test_trajectories, 50)      # Cap at 50 for small dataset
        
        train_file, test_file, log_file = generate_small_dataset_for_testing(
            seed=args.seed,
            train_file_path=args.train_file,
            test_file_path=args.test_file,
            log_file_path=args.log_file,
            num_train_trajectories=small_train,
            num_test_trajectories=small_test,
            num_time_points=args.num_time_points,
            spatial_size=args.spatial_size,
            viscosity=args.viscosity,
            sim_time=args.sim_time,
            time_step=args.time_step
        )
        
        if args.validate:
            # Test with environment check
            log_info("="*50)
            log_info("TESTING GENERATED DATA WITH ENVIRONMENT CHECK")
            log_info("="*50)
            
            try:
                # Test with our environment check (need to import here to avoid circular imports)
                sys.path.append('.')
                from burgers_control.eval_on_testset import test_environment_with_training_data
                
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                mean_mse, all_mse_values = test_environment_with_training_data(
                    train_file_path=train_file,
                    device=device,
                    num_trajectories=5,  # Test with 5 trajectories
                    num_time_points=10,
                    viscosity=0.01,
                    sim_time=1.0,
                    time_step=1e-4
                )
                
                log_info(f"Environment check result: Mean MSE = {mean_mse:.10f}")
                if mean_mse < 1e-10:
                    log_info("✓ SUCCESS: Generated data is perfectly consistent!")
                else:
                    log_info("❌ WARNING: Generated data may have issues")
                    
            except ImportError:
                log_info("Note: eval_on_testset.py not found, skipping validation")
                
    elif args.mode == "full":
        # Generate full production dataset
        train_file, test_file, log_file = generate_full_dataset(
            seed=args.seed,
            train_file_path=args.train_file,
            test_file_path=args.test_file,
            log_file_path=args.log_file,
            batch_size=args.batch_size,
            num_train_trajectories=args.num_train_trajectories,
            num_test_trajectories=args.num_test_trajectories,
            num_time_points=args.num_time_points,
            spatial_size=args.spatial_size,
            viscosity=args.viscosity,
            sim_time=args.sim_time,
            time_step=args.time_step
        )
        
    log_info("Done!")

if __name__ == "__main__":
    main() 