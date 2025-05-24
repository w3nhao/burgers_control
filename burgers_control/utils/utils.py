import os
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv

def load_environment_variables():
    """
    Load environment variables from .env file.
    
    This function should be called early in your application to ensure
    environment variables like WANDB_API_KEY and WANDB_BASE_URL are set.
    """
        
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Check if required WANDB environment variables are set
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is not set. Please create a .env file based on .env-example")
    
    wandb_base_url = os.getenv('WANDB_BASE_URL')
    if not wandb_base_url:
        print("Warning: WANDB_BASE_URL is not set. Using default URL.")
        raise ValueError("WANDB_BASE_URL is not set. Please create a .env file based on .env-example")

# ===============================
# General-purpose logging utilities
# ===============================

def setup_logging(log_file_path=None, logger_name="app", mode="default"):
    """
    Set up logging to capture all output to both console and file.
    
    Args:
        log_file_path (str): Path to save the log file. If None, auto-generates based on mode and timestamp.
        logger_name (str): Name of the logger (default: "app")
        mode (str): Mode description for log file naming when log_file_path is None (default: "default")
        
    Returns:
        tuple: (logger, actual_log_file_path)
    """
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(f"./logs/{logger_name}", exist_ok=True)
    
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"./logs/{logger_name}/{mode}_{timestamp}.log"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else ".", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger, log_file_path

def get_logger_functions(logger_name="app"):
    """
    Factory function that returns pre-configured logging functions for a specific logger.
    
    This is the recommended way to use logging - call this once per module and use the
    returned functions throughout that module.
    
    Args:
        logger_name (str): Name of the logger to use
        
    Returns:
        tuple: (log_info, log_warning, log_error) functions with logger_name pre-configured
        
    Example:
        log_info, log_warning, log_error = get_logger_functions("my_module")
        log_info("This is an info message")
        log_warning("This is a warning")
        log_error("This is an error")
    """
    def log_info_fn(message):
        """Log info message to the pre-configured logger."""
        logger = logging.getLogger(logger_name)
        if logger.hasHandlers():
            logger.info(message)
        else:
            print(message)
    
    def log_warning_fn(message):
        """Log warning message to the pre-configured logger."""
        logger = logging.getLogger(logger_name)
        if logger.hasHandlers():
            logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def log_error_fn(message):
        """Log error message to the pre-configured logger."""
        logger = logging.getLogger(logger_name)
        if logger.hasHandlers():
            logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    return log_info_fn, log_warning_fn, log_error_fn

# Legacy functions - kept for backward compatibility
# New code should use get_logger_functions() instead
def log_info(message, logger_name="app"):
    """
    Helper function to log info messages.
    
    Args:
        message (str): Message to log
        logger_name (str): Name of the logger to use (default: "app")
        
    Note: This is the legacy function. New code should use get_logger_functions() instead.
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.info(message)
    else:
        print(message)

def log_warning(message, logger_name="app"):
    """
    Helper function to log warning messages.
    
    Args:
        message (str): Message to log
        logger_name (str): Name of the logger to use (default: "app")
        
    Note: This is the legacy function. New code should use get_logger_functions() instead.
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.warning(message)
    else:
        print(f"WARNING: {message}")

def log_error(message, logger_name="app"):
    """
    Helper function to log error messages.
    
    Args:
        message (str): Message to log
        logger_name (str): Name of the logger to use (default: "app")
        
    Note: This is the legacy function. New code should use get_logger_functions() instead.
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.error(message)
    else:
        print(f"ERROR: {message}") 