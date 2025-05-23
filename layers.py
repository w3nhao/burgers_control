import torch
import torch.nn as nn
from typing import List, Type, Dict

def get_activation_fn(act_fn: str) -> Type[nn.Module]:
    """
    Get activation function class from string name.
    
    Args:
        act_fn (str): Name of the activation function
        
    Returns:
        Type[nn.Module]: PyTorch activation function class
        
    Raises:
        ValueError: If activation function name is not supported
        
    Supported activation functions:
        - relu, tanh, gelu, sigmoid, elu, swish, softplus, softsign
        - selu, mish, hardswish, hardsigmoid, hardtanh
    """
    activation_map: Dict[str, Type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "swish": nn.SiLU,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "selu": nn.SELU,
        "mish": nn.Mish,
        "hardswish": nn.Hardswish,
        "hardsigmoid": nn.Hardsigmoid,
        "hardtanh": nn.Hardtanh,
    }
    
    act_fn_lower = act_fn.lower().strip()
    if act_fn_lower not in activation_map:
        supported_acts = ", ".join(sorted(activation_map.keys()))
        raise ValueError(
            f"Unsupported activation function: '{act_fn}'. "
            f"Supported functions: {supported_acts}"
        )
    
    return activation_map[act_fn_lower]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional residual connections from input to each hidden layer.
    
    This implementation creates a feed-forward network where each hidden layer can optionally
    receive the original input concatenated with the previous layer's output, creating
    dense connections similar to DenseNet architecture.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        hidden_dims (List[int]): List of hidden layer dimensions
        act_fn (Type[nn.Module], optional): Activation function class. Defaults to nn.GELU
        norm_fn (Type[nn.Module], optional): Normalization function class. Defaults to nn.Identity
        dropout_rate (float, optional): Dropout rate. If 0.0, no dropout is applied. Defaults to 0.0
        use_input_residual (bool, optional): Whether to concatenate input to each hidden layer. Defaults to True
        use_bias (bool, optional): Whether to use bias in linear layers. Defaults to True
        
    Example:
        >>> mlp = MLP(in_dim=10, out_dim=2, hidden_dims=[64, 32], dropout_rate=0.1)
        >>> x = torch.randn(32, 10)
        >>> output = mlp(x)
        >>> print(output.shape)  # torch.Size([32, 2])
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        act_fn: Type[nn.Module] = nn.GELU,
        norm_fn: Type[nn.Module] = nn.Identity,
        dropout_rate: float = 0.0,
        use_input_residual: bool = True,
        use_bias: bool = True
    ):
        super().__init__()
        
        # Validate inputs
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.use_input_residual = use_input_residual
        self.dropout_rate = dropout_rate
        
        # First hidden layer
        first_hidden_dim = hidden_dims[0]
        self.input_layer = self._create_layer(
            in_features=in_dim,
            out_features=first_hidden_dim,
            act_fn=act_fn,
            norm_fn=norm_fn,
            use_bias=use_bias
        )
        
        # Additional hidden layers
        self.hidden_layers = nn.ModuleList()
        current_dim = first_hidden_dim
        
        for hidden_dim in hidden_dims[1:]:
            input_features = current_dim + (in_dim if use_input_residual else 0)
            layer = self._create_layer(
                in_features=input_features,
                out_features=hidden_dim,
                act_fn=act_fn,
                norm_fn=norm_fn,
                use_bias=use_bias
            )
            self.hidden_layers.append(layer)
            current_dim = hidden_dim
        
        # Output layer (no activation or normalization)
        final_input_dim = current_dim + (in_dim if use_input_residual and hidden_dims else current_dim)
        if not hidden_dims or not use_input_residual:
            final_input_dim = current_dim
            
        self.output_layer = nn.Linear(final_input_dim, out_dim, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        
    def _create_layer(
        self,
        in_features: int,
        out_features: int,
        act_fn: Type[nn.Module],
        norm_fn: Type[nn.Module],
        use_bias: bool
    ) -> nn.Sequential:
        """Create a layer with linear transformation, normalization, and activation."""
        components = [nn.Linear(in_features, out_features, bias=use_bias)]
        
        # Add normalization if not Identity
        if norm_fn != nn.Identity:
            components.append(norm_fn(out_features))
        
        # Add activation
        components.append(act_fn())
        
        return nn.Sequential(*components)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim)
        """
        original_shape = x.shape
        
        # Flatten to 2D if needed (batch_size, features)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Store original input for residual connections
        original_input = x
        
        # First hidden layer
        x = self.input_layer(x)
        x = self.dropout(x)
        
        # Additional hidden layers with optional input residual connections
        for layer in self.hidden_layers:
            if self.use_input_residual:
                x = layer(torch.cat([x, original_input], dim=-1))
            else:
                x = layer(x)
            x = self.dropout(x)
        
        # Output layer
        if self.use_input_residual and self.hidden_dims:
            x = self.output_layer(torch.cat([x, original_input], dim=-1))
        else:
            x = self.output_layer(x)
        
        # Reshape to original shape (except last dimension)
        if len(original_shape) > 2:
            x = x.view(*original_shape[:-1], self.out_dim)
        elif len(original_shape) == 1:
            x = x.squeeze(0)
            
        return x
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"MLP(in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"hidden_dims={self.hidden_dims}, use_input_residual={self.use_input_residual}, "
            f"dropout_rate={self.dropout_rate}, params={self.get_num_parameters():,})"
        )