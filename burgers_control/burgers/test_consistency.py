#!/usr/bin/env python3
"""
Comprehensive test suite to validate consistency between original and accelerated Burgers implementations.

This script tests all critical functions to ensure that the accelerated versions produce
identical results to the original implementations within numerical precision.
"""

import os
import sys
import torch
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple

# Add the parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_initial_conditions_and_forcing_terms(seed: int = 42, tolerance: float = 1e-6, 
                                             verbose: bool = True) -> bool:
    """Test that accelerated and original initial condition/forcing term generation are identical."""
    
    from .burgers_original import make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original
    from .burgers_accelerated import make_initial_conditions_and_varying_forcing_terms as make_ic_ft_accelerated
    
    if verbose:
        print("Testing initial conditions and forcing terms generation...")
    
    test_cases = [
        # (num_ic, num_ft, spatial_size, num_time_points, partial_control, description)
        (10, 10, 32, 5, None, "Small test, no control"),
        (50, 50, 64, 8, None, "Medium test, no control"), 
        (20, 20, 128, 10, None, "Large spatial, no control"),
        (15, 15, 64, 12, 'front_rear_quarter', "Medium test, partial control"),
        (25, 25, 32, 6, 'front_rear_quarter', "Small test, partial control"),
    ]
    
    all_passed = True
    
    for i, (num_ic, num_ft, spatial_size, num_time_points, partial_control, description) in enumerate(test_cases):
        if verbose:
            print(f"  Test {i+1}: {description}")
        
        # Test on both CPU and GPU if available
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            if verbose:
                print(f"    Device: {device}")
            
            # Original implementation
            torch.manual_seed(seed)
            np.random.seed(seed)
            ic_orig, ft_orig = make_ic_ft_original(
                num_ic, num_ft, spatial_size, num_time_points,
                partial_control=partial_control, seed=seed
            )
            
            # Accelerated implementation  
            torch.manual_seed(seed)
            np.random.seed(seed)
            ic_acc, ft_acc = make_ic_ft_accelerated(
                num_ic, num_ft, spatial_size, num_time_points,
                partial_control=partial_control, seed=seed, device=device
            )
            
            # Move original to device for comparison
            ic_orig = ic_orig.to(device)
            ft_orig = ft_orig.to(device)
            
            # Compare results
            ic_diff = torch.abs(ic_orig - ic_acc).max().item()
            ft_diff = torch.abs(ft_orig - ft_acc).max().item()
            
            ic_match = ic_diff < tolerance
            ft_match = ft_diff < tolerance
            
            if verbose:
                print(f"      IC max difference: {ic_diff:.2e}, Match: {ic_match}")
                print(f"      FT max difference: {ft_diff:.2e}, Match: {ft_match}")
            
            if not (ic_match and ft_match):
                print(f"    ❌ FAILED: Test {i+1} on {device}")
                all_passed = False
            elif verbose:
                print(f"    ✓ PASSED: Test {i+1} on {device}")
    
    return all_passed

def test_burgers_simulation(seed: int = 42, tolerance: float = 1e-5, verbose: bool = True) -> bool:
    """Test that accelerated and original Burgers simulation are identical."""
    
    from .burgers_original import (
        simulate_burgers_equation as simulate_original,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original
    )
    from .burgers_accelerated import (
        simulate_burgers_equation as simulate_accelerated,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_accelerated
    )
    
    if verbose:
        print("Testing Burgers equation simulation...")
    
    test_cases = [
        # (num_samples, spatial_size, num_time_points, viscosity, sim_time, time_step, description)
        (5, 32, 5, 0.01, 0.5, 1e-4, "Small quick test"),
        (10, 64, 8, 0.01, 1.0, 1e-4, "Medium test"),
        (15, 128, 10, 0.01, 1.0, 1e-4, "Large test"),
        (8, 64, 6, 0.005, 0.8, 2e-4, "Different parameters"),
    ]
    
    all_passed = True
    
    for i, (num_samples, spatial_size, num_time_points, viscosity, sim_time, time_step, description) in enumerate(test_cases):
        if verbose:
            print(f"  Test {i+1}: {description}")
            print(f"    Params: {num_samples} samples, {spatial_size} spatial, {num_time_points} time points")
        
        # Test on both CPU and GPU if available
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            if verbose:
                print(f"    Device: {device}")
            
            # Generate test data
            torch.manual_seed(seed)
            np.random.seed(seed)
            ic, ft = make_ic_ft_original(
                num_samples, num_samples, spatial_size, num_time_points, seed=seed
            )
            
            # Original simulation
            traj_orig = simulate_original(
                ic, ft, viscosity=viscosity, sim_time=sim_time,
                time_step=time_step, num_time_points=num_time_points, print_progress=False
            )
            
            # Accelerated simulation
            traj_acc = simulate_accelerated(
                ic.to(device), ft.to(device), viscosity=viscosity, sim_time=sim_time,
                time_step=time_step, num_time_points=num_time_points, print_progress=False,
                device=device
            )
            
            # Move original to device for comparison
            traj_orig = traj_orig.to(device)
            
            # Compare trajectories
            sim_diff = torch.abs(traj_orig - traj_acc).max().item()
            sim_match = sim_diff < tolerance
            
            if verbose:
                print(f"      Trajectory max difference: {sim_diff:.2e}, Match: {sim_match}")
            
            if not sim_match:
                print(f"    ❌ FAILED: Simulation test {i+1} on {device}")
                all_passed = False
            elif verbose:
                print(f"    ✓ PASSED: Simulation test {i+1} on {device}")
    
    return all_passed

def test_one_time_point_simulation(seed: int = 42, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Test the one-time-point simulation function."""
    
    from .burgers_original import (
        simulate_burgers_one_time_point as simulate_one_step_original,
        create_differential_matrices_1d,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original
    )
    from .burgers_accelerated import simulate_burgers_one_time_point as simulate_one_step_accelerated
    
    if verbose:
        print("Testing one-time-point simulation...")
    
    # Test parameters
    spatial_size = 64
    num_samples = 10
    num_time_points = 5
    viscosity = 0.01
    time_step = 1e-4
    
    # Generate test data
    torch.manual_seed(seed)
    np.random.seed(seed)
    ic, ft = make_ic_ft_original(num_samples, num_samples, spatial_size, num_time_points, seed=seed)
    
    # Setup differential matrices (original format)
    domain_min, domain_max = 0.0, 1.0
    spatial_step = (domain_max - domain_min) / (spatial_size + 1)
    
    first_deriv, second_deriv = create_differential_matrices_1d(spatial_size + 2)
    
    # Adjust boundary conditions
    first_deriv.rows[0] = first_deriv.rows[0][:2]
    first_deriv.rows[-1] = first_deriv.rows[-1][-2:]
    first_deriv.data[0] = first_deriv.data[0][:2]
    first_deriv.data[-1] = first_deriv.data[-1][-2:]
    
    second_deriv.rows[0] = second_deriv.rows[0][:3]
    second_deriv.rows[-1] = second_deriv.rows[-1][-3:]
    second_deriv.data[0] = second_deriv.data[0][:3]
    second_deriv.data[-1] = second_deriv.data[-1][-3:]
    
    # Convert to tensor format
    transport_indices = list(first_deriv.rows)
    transport_coeffs = torch.FloatTensor(np.stack(first_deriv.data) / (2 * spatial_step))
    diffusion_indices = list(second_deriv.rows)
    diffusion_coeffs = torch.FloatTensor(viscosity * np.stack(second_deriv.data) / spatial_step**2)
    
    all_passed = True
    
    # Test on both CPU and GPU if available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        if verbose:
            print(f"  Device: {device}")
        
        # Move data to device
        state = torch.nn.functional.pad(ic.to(device), (1, 1))
        forcing = torch.nn.functional.pad(ft[:, 0, :].to(device), (1, 1))
        transport_coeffs_dev = transport_coeffs.to(device)
        diffusion_coeffs_dev = diffusion_coeffs.to(device)
        
        # Original one-step simulation
        next_state_orig = simulate_one_step_original(
            state, forcing, transport_indices, transport_coeffs_dev,
            diffusion_indices, diffusion_coeffs_dev, time_step
        )
        
        # Accelerated one-step simulation
        next_state_acc = simulate_one_step_accelerated(
            state, forcing, transport_indices, transport_coeffs_dev,
            diffusion_indices, diffusion_coeffs_dev, time_step
        )
        
        # Compare results
        step_diff = torch.abs(next_state_orig - next_state_acc).max().item()
        step_match = step_diff < tolerance
        
        if verbose:
            print(f"    One-step max difference: {step_diff:.2e}, Match: {step_match}")
        
        if not step_match:
            print(f"  ❌ FAILED: One-step simulation on {device}")
            all_passed = False
        elif verbose:
            print(f"  ✓ PASSED: One-step simulation on {device}")
    
    return all_passed

def test_numerical_stability(seed: int = 42, tolerance: float = 1e-4, verbose: bool = True) -> bool:
    """Test numerical stability across different parameter ranges."""
    
    from .burgers_original import (
        simulate_burgers_equation as simulate_original,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original
    )
    from .burgers_accelerated import (
        simulate_burgers_equation as simulate_accelerated
    )
    
    if verbose:
        print("Testing numerical stability...")
    
    # Test cases with different parameter ranges that might cause instability
    test_cases = [
        # (viscosity, scaling_factor, description)
        (0.001, 0.1, "Low viscosity, small forcing"),
        (0.1, 1.0, "High viscosity, normal forcing"),
        (0.01, 2.0, "Normal viscosity, large forcing"),
        (0.005, 0.5, "Medium viscosity, medium forcing"),
    ]
    
    all_passed = True
    
    for i, (viscosity, scaling_factor, description) in enumerate(test_cases):
        if verbose:
            print(f"  Test {i+1}: {description}")
        
        # Generate test data
        num_samples = 8
        spatial_size = 64
        num_time_points = 8
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        ic, ft = make_ic_ft_original(
            num_samples, num_samples, spatial_size, num_time_points,
            scaling_factor=scaling_factor, seed=seed
        )
        
        # Test on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Original simulation
            traj_orig = simulate_original(
                ic, ft, viscosity=viscosity, sim_time=1.0,
                time_step=1e-4, num_time_points=num_time_points, print_progress=False
            )
            
            # Accelerated simulation
            traj_acc = simulate_accelerated(
                ic.to(device), ft.to(device), viscosity=viscosity, sim_time=1.0,
                time_step=1e-4, num_time_points=num_time_points, print_progress=False,
                device=device
            )
            
            # Move original to device for comparison
            traj_orig = traj_orig.to(device)
            
            # Check for NaN or Inf values
            if torch.isnan(traj_orig).any() or torch.isinf(traj_orig).any():
                print(f"  ❌ FAILED: Original simulation has NaN/Inf values")
                all_passed = False
                continue
            
            if torch.isnan(traj_acc).any() or torch.isinf(traj_acc).any():
                print(f"  ❌ FAILED: Accelerated simulation has NaN/Inf values")
                all_passed = False
                continue
            
            # Compare trajectories
            sim_diff = torch.abs(traj_orig - traj_acc).max().item()
            sim_match = sim_diff < tolerance
            
            if verbose:
                print(f"    Max difference: {sim_diff:.2e}, Match: {sim_match}")
            
            if not sim_match:
                print(f"  ❌ FAILED: Stability test {i+1}")
                all_passed = False
            elif verbose:
                print(f"  ✓ PASSED: Stability test {i+1}")
                
        except Exception as e:
            print(f"  ❌ FAILED: Stability test {i+1} - Exception: {e}")
            all_passed = False
    
    return all_passed

def benchmark_performance(verbose: bool = True) -> Dict:
    """Benchmark performance differences between implementations."""
    
    from .burgers_original import (
        simulate_burgers_equation as simulate_original,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_original
    )
    from .burgers_accelerated import (
        simulate_burgers_equation as simulate_accelerated,
        make_initial_conditions_and_varying_forcing_terms as make_ic_ft_accelerated
    )
    
    if verbose:
        print("Benchmarking performance...")
    
    # Benchmark parameters
    num_samples = 100
    spatial_size = 128
    num_time_points = 10
    seed = 42
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"  Using device: {device}")
    
    results = {}
    
    # Benchmark initial condition and forcing term generation
    if verbose:
        print("  Benchmarking IC/FT generation...")
    
    # Original
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    ic_orig, ft_orig = make_ic_ft_original(num_samples, num_samples, spatial_size, num_time_points, seed=seed)
    ic_orig = ic_orig.to(device)
    ft_orig = ft_orig.to(device)
    orig_gen_time = time.time() - start_time
    
    # Accelerated
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    ic_acc, ft_acc = make_ic_ft_accelerated(num_samples, num_samples, spatial_size, num_time_points, seed=seed, device=device)
    acc_gen_time = time.time() - start_time
    
    results['generation'] = {
        'original_time': orig_gen_time,
        'accelerated_time': acc_gen_time,
        'speedup': orig_gen_time / acc_gen_time if acc_gen_time > 0 else float('inf')
    }
    
    if verbose:
        print(f"    Original: {orig_gen_time:.4f}s, Accelerated: {acc_gen_time:.4f}s, Speedup: {results['generation']['speedup']:.2f}x")
    
    # Benchmark simulation
    if verbose:
        print("  Benchmarking simulation...")
    
    # Use smaller sample size for simulation benchmark
    sim_samples = min(50, num_samples)
    ic_sim = ic_acc[:sim_samples]
    ft_sim = ft_acc[:sim_samples]
    
    # Original
    start_time = time.time()
    traj_orig = simulate_original(
        ic_sim.cpu(), ft_sim.cpu(), viscosity=0.01, sim_time=1.0,
        time_step=1e-4, num_time_points=num_time_points, print_progress=False
    )
    orig_sim_time = time.time() - start_time
    
    # Accelerated
    start_time = time.time()
    traj_acc = simulate_accelerated(
        ic_sim, ft_sim, viscosity=0.01, sim_time=1.0,
        time_step=1e-4, num_time_points=num_time_points, print_progress=False,
        device=device
    )
    acc_sim_time = time.time() - start_time
    
    results['simulation'] = {
        'original_time': orig_sim_time,
        'accelerated_time': acc_sim_time,
        'speedup': orig_sim_time / acc_sim_time if acc_sim_time > 0 else float('inf')
    }
    
    if verbose:
        print(f"    Original: {orig_sim_time:.4f}s, Accelerated: {acc_sim_time:.4f}s, Speedup: {results['simulation']['speedup']:.2f}x")
    
    # Overall results
    total_orig = orig_gen_time + orig_sim_time
    total_acc = acc_gen_time + acc_sim_time
    
    results['overall'] = {
        'total_original_time': total_orig,
        'total_accelerated_time': total_acc,
        'overall_speedup': total_orig / total_acc if total_acc > 0 else float('inf')
    }
    
    if verbose:
        print(f"  Overall: Original {total_orig:.4f}s, Accelerated {total_acc:.4f}s, Speedup: {results['overall']['overall_speedup']:.2f}x")
    
    return results

def run_comprehensive_tests(verbose: bool = True) -> bool:
    """Run all tests and return overall pass/fail status."""
    
    print("="*70)
    print("COMPREHENSIVE BURGERS IMPLEMENTATION CONSISTENCY TESTS")
    print("="*70)
    
    # List of all tests to run
    tests = [
        ("Initial Conditions & Forcing Terms", test_initial_conditions_and_forcing_terms),
        ("Burgers Equation Simulation", test_burgers_simulation),
        ("One-Time-Point Simulation", test_one_time_point_simulation),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 50)
        
        try:
            passed = test_func(verbose=verbose)
            results.append(passed)
            
            if passed:
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
    
    # Run performance benchmark
    print(f"\nPerformance Benchmark:")
    print("-" * 50)
    try:
        benchmark_results = benchmark_performance(verbose=verbose)
        print("✓ Performance Benchmark: COMPLETED")
    except Exception as e:
        print(f"❌ Performance Benchmark: ERROR - {e}")
    
    # Summary
    all_passed = all(results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n✅ The accelerated implementation produces identical results to the original!")
        print("✅ You can safely use the accelerated functions for training.")
    else:
        print("\n❌ Some tests failed. Please check the accelerated implementation.")
    
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test consistency between original and accelerated Burgers implementations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Numerical tolerance for comparisons")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for testing")
    
    args = parser.parse_args()
    
    if args.disable_compile:
        os.environ['DISABLE_TORCH_COMPILE'] = 'true'
    
    # Set global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run comprehensive tests
    success = run_comprehensive_tests(verbose=args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 