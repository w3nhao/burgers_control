import pickle
import logging
from functools import wraps
from pathlib import Path
from packaging import version as packaging_version
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module

# Configure logger
logger = logging.getLogger(__name__)

# helpers

def exists(v: Any) -> bool:
    """Check if a value is not None."""
    return v is not None

def ensure_path(path: Union[str, Path]) -> Path:
    """Ensure path is a Path object and resolve it."""
    return Path(path).resolve()

def validate_config_serializable(config: Tuple[Tuple[Any, ...], Dict[str, Any]]) -> None:
    """Validate that the configuration can be pickled and unpickled."""
    try:
        pickle.dumps(config)
    except (pickle.PickleError, TypeError) as e:
        raise ValueError(f"Configuration is not serializable: {e}")

def save_load(
    save_method_name: str = 'save',
    load_method_name: str = 'load',
    config_instance_var_name: str = '_config',
    init_and_load_classmethod_name: str = 'init_and_load',
    version: Optional[str] = None,
    backup_on_overwrite: bool = True,
    device_aware_loading: bool = True
):
    """
    Decorator to add save/load functionality to PyTorch nn.Module subclasses.
    
    This decorator automatically captures constructor arguments and provides methods
    to save and load both model weights and configuration, enabling complete model
    persistence and restoration.
    
    Args:
        save_method_name: Name of the save method to add to the class
        load_method_name: Name of the load method to add to the class  
        config_instance_var_name: Name of the instance variable to store config
        init_and_load_classmethod_name: Name of the class method for init+load
        version: Version string for compatibility checking
        backup_on_overwrite: Whether to create backup when overwriting existing files
        device_aware_loading: Whether to handle device placement during loading
        
    Returns:
        The decorated class with save/load methods added
        
    Raises:
        AssertionError: If the decorated class is not a subclass of torch.nn.Module
        
    Example:
        ```python
        @save_load(version="1.0.0")
        class MyModel(nn.Module):
            def __init__(self, hidden_dim: int, num_layers: int):
                super().__init__()
                self.net = nn.Linear(hidden_dim, num_layers)
                
        # Usage
        model = MyModel(128, 3)
        model.save("model.pt")
        
        # Load existing model
        loaded_model = MyModel.init_and_load("model.pt")
        ```
    """
    def _save_load(klass: type) -> type:
        if not issubclass(klass, Module):
            raise TypeError('save_load should decorate a subclass of torch.nn.Module')

        _orig_init = klass.__init__

        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            # Capture configuration before calling original init
            config = (args, kwargs)
            
            # Validate that config is serializable
            validate_config_serializable(config)
            
            # Store pickled config
            _config = pickle.dumps(config)
            setattr(self, config_instance_var_name, _config)
            
            # Call original init
            _orig_init(self, *args, **kwargs)

        def _save(
            self, 
            path: Union[str, Path], 
            overwrite: bool = True,
            save_optimizer: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> None:
            """
            Save model state, configuration, and optional metadata to file.
            
            Args:
                path: Path to save the model
                overwrite: Whether to overwrite existing files
                save_optimizer: Whether to save optimizer state
                optimizer: Optimizer instance to save (required if save_optimizer=True)
                metadata: Additional metadata to save with the model
            """
            path = ensure_path(path)
            
            # Create backup if file exists and backup is enabled
            if path.exists():
                if not overwrite:
                    raise FileExistsError(f"File {path} already exists and overwrite=False")
                if backup_on_overwrite:
                    backup_path = path.with_suffix(path.suffix + '.backup')
                    logger.info(f"Creating backup at {backup_path}")
                    path.rename(backup_path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build package to save
            pkg = {
                'model': self.state_dict(),
                'config': getattr(self, config_instance_var_name),
                'version': version,
                'save_timestamp': torch.tensor(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0),
                'torch_version': torch.__version__,
                'metadata': metadata or {}
            }
            
            # Add optimizer state if requested
            if save_optimizer:
                if optimizer is None:
                    raise ValueError("optimizer must be provided when save_optimizer=True")
                pkg['optimizer'] = optimizer.state_dict()
                pkg['optimizer_class'] = optimizer.__class__.__name__
            
            try:
                torch.save(pkg, str(path))
                logger.info(f"Model saved successfully to {path}")
            except Exception as e:
                logger.error(f"Failed to save model to {path}: {e}")
                raise

        def _load(
            self, 
            path: Union[str, Path], 
            strict: bool = True,
            load_optimizer: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            map_location: Optional[Union[str, torch.device]] = None
        ) -> Dict[str, Any]:
            """
            Load model state from file.
            
            Args:
                path: Path to the saved model
                strict: Whether to strictly enforce state dict key matching
                load_optimizer: Whether to load optimizer state
                optimizer: Optimizer instance to load state into
                map_location: Device to map tensors to during loading
                
            Returns:
                Dictionary containing metadata from the saved file
            """
            path = ensure_path(path)
            
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            
            try:
                # Determine map_location
                if map_location is None and device_aware_loading:
                    # Try to preserve the device of existing parameters
                    if hasattr(self, 'parameters') and any(True for _ in self.parameters()):
                        map_location = next(self.parameters()).device
                    else:
                        map_location = 'cpu'
                
                pkg = torch.load(str(path), map_location=map_location, weights_only=False)
                logger.info(f"Model loaded from {path}")
                
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
                raise
            
            # Version compatibility check
            if exists(version) and exists(pkg.get('version')):
                saved_version = pkg['version']
                if packaging_version.parse(version) != packaging_version.parse(saved_version):
                    logger.warning(
                        f"Version mismatch: loading model saved with version {saved_version}, "
                        f"but current version is {version}"
                    )
            
            # Load model state
            try:
                self.load_state_dict(pkg['model'], strict=strict)
            except Exception as e:
                logger.error(f"Failed to load model state dict: {e}")
                raise
            
            # Load optimizer state if requested
            if load_optimizer:
                if optimizer is None:
                    raise ValueError("optimizer must be provided when load_optimizer=True")
                if 'optimizer' not in pkg:
                    logger.warning("No optimizer state found in saved model")
                else:
                    try:
                        optimizer.load_state_dict(pkg['optimizer'])
                        logger.info("Optimizer state loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}")
            
            # Return metadata for inspection
            return {
                'version': pkg.get('version'),
                'torch_version': pkg.get('torch_version'),
                'metadata': pkg.get('metadata', {}),
                'save_timestamp': pkg.get('save_timestamp')
            }

        @classmethod
        def _init_and_load_from(
            cls, 
            path: Union[str, Path], 
            strict: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            **override_kwargs
        ) -> Tuple[Module, Dict[str, Any]]:
            """
            Initialize a new instance and load state from file.
            
            Args:
                path: Path to the saved model
                strict: Whether to strictly enforce state dict key matching
                device: Device to place the model on after loading
                **override_kwargs: Keyword arguments to override in the saved config
                
            Returns:
                Tuple of (model_instance, metadata)
            """
            path = ensure_path(path)
            
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
                
            try:
                pkg = torch.load(str(path), map_location='cpu', weights_only=False)
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {path}: {e}")
                raise
                
            if 'config' not in pkg:
                raise ValueError('Model configuration not found in checkpoint. '
                               'The model may have been saved without the save_load decorator.')
            
            try:
                config = pickle.loads(pkg['config'])
                args, kwargs = config
                
                # Apply any overrides
                kwargs.update(override_kwargs)
                
                logger.info(f"Initializing {cls.__name__} with saved configuration")
                model = cls(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Failed to initialize model from saved config: {e}")
                raise
            
            # Load the state
            metadata = _load(model, path, strict=strict)
            
            # Move to specified device
            if device is not None:
                model = model.to(device)
                logger.info(f"Model moved to device: {device}")
            
            return model, metadata

        # Set decorated methods
        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load