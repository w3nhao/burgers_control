import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_squence_data, train_file_path, test_file_path, BurgersTest
import torch
import h5py
import numpy as np

def print_tensor_info(name, tensor):
    """Print detailed information about a tensor or array"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: Tensor shape {tensor.shape}, dtype {tensor.dtype}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}: Array shape {tensor.shape}, dtype {tensor.dtype}")
    else:
        print(f"{name}: Type {type(tensor)}")

# Explore training data
print("="*50)
print("EXPLORING TRAINING DATA")
print("="*50)

train_data = get_squence_data(train_file_path)

for key, value in train_data.items():
    print_tensor_info(f"train_data['{key}']", value)
    if isinstance(value, np.ndarray) and len(value) > 0:
        print(f"  - Sample values: {value[:3]}")
        print(f"  - Value range: [{np.min(value)}, {np.max(value)}]")
    print()

# Explore test data
print("="*50)
print("EXPLORING TEST DATA")
print("="*50)

test_dataset = BurgersTest(test_file_path)
print(f"Test dataset size: {len(test_dataset)}")

# Sample one item
sample_item = test_dataset[0]
for key, value in sample_item.items():
    print_tensor_info(f"test_item['{key}']", value)
    if isinstance(value, torch.Tensor) and len(value) > 0:
        print(f"  - Sample values: {value[:3].tolist() if value.dim() <= 1 else value[0, :3].tolist()}")
        print(f"  - Value range: [{torch.min(value).item()}, {torch.max(value).item()}]")
    print()

# Explore original HDF5 files for validation
print("="*50)
print("EXPLORING RAW HDF5 FILES")
print("="*50)

# Training file
with h5py.File(train_file_path, 'r') as hdf:
    print(f"Training file keys: {list(hdf.keys())}")
    print(f"Training file['train'] keys: {list(hdf['train'].keys())}")
    pde_data = hdf['train']['pde_11-128'][:]
    f_data = hdf['train']['pde_11-128_f'][:]
    print_tensor_info("Raw training pde_data", pde_data)
    print_tensor_info("Raw training f_data", f_data)

# Test file
with h5py.File(test_file_path, 'r') as hdf:
    print(f"Test file keys: {list(hdf.keys())}")
    test_data = hdf['test'][:]
    print_tensor_info("Raw test data", test_data)

print("="*50)
print("DATA EXPLORATION COMPLETE")
print("="*50) 