import h5py
import torch
import numpy as np

import torch

train_file_path = "../1d_burgers/burgers_train.h5"
test_file_path = "../1d_burgers/unsafe_test.h5"

def discounted_cumsum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = torch.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

def get_squence_data(file_path):
    with h5py.File(file_path, 'r') as hdf:
        print("Keys: ", list(hdf.keys()))
        u_data = torch.tensor(hdf['train']['pde_11-128'][:40000])
        f_data = torch.tensor(hdf['train']['pde_11-128_f'][:40000])
        
    # [s_0, s_1, s_2, ..., s_n - 1]
    # [a_0, a_1, a_2, ..., a_n - 1]
    # [r_1, r_2, r_3, ..., r_n]
    # [c_1, c_2, c_3, ..., c_n]
        
    rewards = -(u_data[:, -1].unsqueeze(1) - u_data[:, 1:]).square().mean(-1)
    
    terminals = np.zeros(rewards.shape, dtype=np.bool_)
    terminals[:, -1] = True

    data = dict(
        observations=u_data[:, :-1].reshape(-1, u_data.shape[-1]).numpy(),
        actions=f_data.reshape(-1, f_data.shape[-1]).numpy(),
        rewards=rewards.reshape(-1).numpy(),
        terminals=terminals.reshape(-1),
        timeouts=np.zeros_like(terminals).reshape(-1),
    )
    return data


class BurgersTest(torch.utils.data.Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as hdf:
            u_test = hdf['test'][:]
        
        rewards = -((u_test[:, -1][:, None, :] - u_test[:, 1:]) ** 2).mean(-1)
        observations = u_test[:, :-1]
        targets = u_test[:, -1]
        
        self.data = dict(
            observations=observations,
            rewards=rewards,
            targets=targets,
        )
        
    def __len__(self):
        return len(self.data['observations'])
    
    def __getitem__(self, idx):
        observations = self.data['observations'][idx]
        rewards = self.data['rewards'][idx]
        returns = discounted_cumsum(torch.tensor(rewards), gamma=1)
        target = self.data['targets'][idx]
        
        return dict(
            observations=observations,
            rewards=rewards,
            returns=returns,
            target=target,
        )
        
        