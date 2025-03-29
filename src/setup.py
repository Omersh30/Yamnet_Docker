import os
import shutil
from pathlib import Path

def setup_yamnet():
    """Set up YAMNet model directory structure."""
    models_dir = Path("models")
    yamnet_dir = models_dir / "yamnet"
    
    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    yamnet_dir.mkdir(exist_ok=True)
    
    # Create a placeholder for the YAMNet model
    yamnet_path = yamnet_dir / "yamnet_model"
    if not yamnet_path.exists():
        yamnet_path.mkdir(exist_ok=True)
        print("Created YAMNet model directory structure")
    else:
        print("YAMNet model directory already exists")

def setup_models():
    """Set up all required models."""
    print("Setting up models...")
    setup_yamnet()
    print("Setup complete!")
    print("\nPlease ensure you have the following files in place:")
    print("1. models/llama-2-7b-chat.gguf (Llama model)")
    print("2. models/yamnet/yamnet_model/ (YAMNet model directory)")

if __name__ == "__main__":
    setup_models() 