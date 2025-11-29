import os
import random
import numpy as np
import torch

def set_seed(seed_value=42):
    # Set a seed for reproducibility.
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 

def init_qwen_api():
    # Set up the API configuration for Qwen.
    # Ensure you have the QWEN_API_KEY environment variable set.
    api_key = os.environ.get("QWEN_API_KEY")
    
    if not api_key:
        # You can also manually replace "YOUR_API_KEY_HERE" with your actual key if you don't want to use environment variables.
        api_key = "YOUR_API_KEY_HERE"
        
    return {
        "api_key": api_key,
        "api_endpoint": "https://chat-ai.academiccloud.de/v1",
        "model": "qwen2.5-vl-72b-instruct"
    } 