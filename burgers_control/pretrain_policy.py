"""
Goal-Conditioned Policy Pretraining for Burgers' Equation Control

This script pretrains a policy network for the goal-conditioned Burgers' equation control task.
The environment provides observations as [current_state, target_state] and expects actions (forcing terms).

Key aspects of goal-conditioning:
- Environment observations: [current_state, target_state] (shape: 2 * spatial_size)
- Policy input: [current_state, target_state] (same as environment)
- Policy output: action/forcing terms (shape: spatial_size)
- Dataset: Creates (observation, action) pairs from historical trajectories
  where observation = [state_t, target] and action = action_t

This ensures the pretrained policy is compatible with the goal-conditioned RL environment.
"""

import os
import random
import time
import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import tqdm
import tyro
import wandb
from burgers_control.layers import MLP, get_activation_fn
from burgers_control.burgers import BurgersDataset
from burgers_control.utils.utils import load_environment_variables


# Import save_load decorator for model persistence
from burgers_control.utils.save_load import save_load

@dataclass
class PretrainArgs:
    exp_name: str = "pretrain_policy"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: int = 7
    """cuda device to use"""
    
    # Model specific arguments
    spatial_size: int = 128
    """the spatial size of the environment"""
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024])
    """the hidden dimensions of the MLP"""
    act_fn: str = "gelu"
    """the activation function of the MLP"""
    
    # Training specific arguments
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    batch_size: int = 512
    """the batch size for training"""
    num_epochs: int = 2000
    """number of training epochs"""
    train_split: float = 0.8
    """fraction of data to use for training (rest for validation)"""
    weight_decay: float = 1e-5
    """weight decay for optimizer"""
    scheduler_patience: int = 100
    """patience for learning rate scheduler"""
    early_stopping_patience: int = 20
    """patience for early stopping"""
    
    # Logging and saving
    log_interval: int = 10
    """interval for logging training progress"""
    save_interval: int = 200
    """interval for saving model checkpoints (in epochs)"""
    save_dir: str = "pretrained_models"
    """directory to save pretrained models"""
    
    # Data specific
    max_samples: Optional[int] = None
    """maximum number of samples to use (None for all)"""
    train_file_path: str = ""
    """path to the training dataset file"""

class StateTransitionDataset(Dataset):
    """
    Dataset that creates (current_state, target, action) tuples from BurgersDataset.
    
    This matches the goal-conditioned environment interface where observations are
    [current_state, target_state] and we predict the action that was taken.
    The idea is to train the policy to predict the action given the current state and target goal,
    i.e., given (s_t, target), predict action a_t.
    """
    
    def __init__(self, train_file_path: str, max_samples: Optional[int] = None):
        # Load the burgers dataset
        if not train_file_path:
            raise ValueError("train_file_path must be provided")
        burgers_data = BurgersDataset(mode="train", train_file_path=train_file_path)
        
        self.observations = []  # Will store [current_state, target]
        self.actions = []
        
        print("Creating goal-conditioned state-action dataset...")
        
        # Process each trajectory
        for idx in tqdm.trange(len(burgers_data), desc="Processing trajectories"):
            sample = burgers_data[idx]
            observations = sample['observations']  # Shape: (T-1, spatial_size)
            actions = sample['actions']           # Shape: (T, spatial_size)
            target = sample['targets']            # Shape: (spatial_size,) - final target state
            
            # Create goal-conditioned (state, target) -> action pairs
            # Each observation at time t with the target and corresponding action
            for t in range(len(observations)):
                current_state = observations[t]   # s_t
                action = actions[t]               # a_t (action taken from s_t)
                
                # Concatenate current_state and target as observation (goal-conditioned)
                observation = np.concatenate([current_state, target])
                
                self.observations.append(observation)
                self.actions.append(action)
                
                # Limit number of samples if specified
                if max_samples is not None and len(self.observations) >= max_samples:
                    break
            
            if max_samples is not None and len(self.observations) >= max_samples:
                break
        
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        
        print(f"Created goal-conditioned dataset with {len(self.observations)} (state, target) -> action samples")
        print(f"Observations shape: {self.observations.shape}")
        print(f"Actions shape: {self.actions.shape}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.tensor(self.observations[idx], dtype=torch.float32),
            'action': torch.tensor(self.actions[idx], dtype=torch.float32)
        }

@save_load(version="1.0.0")
class PolicyNetwork(nn.Module):
    """
    Goal-conditioned policy network that predicts actions given current state and target.
    Uses the same architecture as the actor network in PPO.
    Matches the environment interface where observations are [current_state, target_state].
    """
    
    def __init__(self, spatial_size: int, hidden_dims: List[int], act_fn: str = "gelu"):
        super().__init__()
        
        # Input is concatenation of current_state and target (goal-conditioned)
        # This matches the environment observation space
        input_dim = 2 * spatial_size
        # Output is the action (forcing terms)
        output_dim = spatial_size
        
        self.policy_net = MLP(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dims=hidden_dims,
            act_fn=get_activation_fn(act_fn),
            norm_fn=nn.Identity,
            dropout_rate=0.0,
            use_input_residual=True,
            use_bias=True
        )
    
    def forward(self, observation):
        """
        Forward pass to predict action from goal-conditioned observation.
        
        Args:
            observation: Tensor of shape (..., 2*spatial_size) containing [current_state, target]
            
        Returns:
            Predicted action of shape (..., spatial_size)
        """
        return self.policy_net(observation)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        observations = batch['observation'].to(device)
        target_actions = batch['action'].to(device)
        
        optimizer.zero_grad()
        
        predicted_actions = model(observations)
        loss = criterion(predicted_actions, target_actions)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observation'].to(device)
            target_actions = batch['action'].to(device)
            
            predicted_actions = model(observations)
            loss = criterion(predicted_actions, target_actions)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    # Load environment variables
    load_environment_variables()
    
    args = tyro.cli(PretrainArgs)
    
    # Validate required arguments
    if not args.train_file_path:
        raise ValueError("train_file_path must be provided")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup experiment name with timestamp
    exp_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}__{args.seed}__{exp_start_time}"
    
    # Initialize wandb
    wandb.init(
        project="burgers_policy_pretrain",
        name=run_name,
        config=vars(args),
        save_code=True,
    )
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = StateTransitionDataset(train_file_path=args.train_file_path, max_samples=args.max_samples)
    
    # Split dataset into train and validation
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = PolicyNetwork(
        spatial_size=args.spatial_size,
        hidden_dims=args.hidden_dims,
        act_fn=args.act_fn
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,
        factor=0.5
    )
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    previous_lr = args.learning_rate  # Track LR changes
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate and check for changes
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f"ReduceLROnPlateau: reducing learning rate to {current_lr:.2e}")
            previous_lr = current_lr
        
        epoch_time = time.time() - start_time
        
        # Logging
        if epoch % args.log_interval == 0 or epoch == args.num_epochs - 1:
            print(f"Epoch {epoch:3d}/{args.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Wandb logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            best_model_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
            model.save(best_model_path, overwrite=True)
            print(f"Saved new best model with val_loss: {val_loss:.6f}")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint periodically
        if epoch % args.save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(args.save_dir, f"{run_name}_epoch_{epoch}.pt")
            model.save(checkpoint_path, overwrite=True)
        
        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs (no improvement for {args.early_stopping_patience} epochs)")
            break
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"{run_name}_final.pt")
    model.save(final_model_path, overwrite=True)
    
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved in: {args.save_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    main() 