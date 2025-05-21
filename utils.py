import os
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
        print("Warning: WANDB_API_KEY is not set. Please create a .env file based on .env-example")
    
    wandb_base_url = os.getenv('WANDB_BASE_URL')
    if not wandb_base_url:
        print("Warning: WANDB_BASE_URL is not set. Using default URL.") 