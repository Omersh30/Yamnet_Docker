import os
import requests
from tqdm import tqdm

def download_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Model URL
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.gguf"
    model_path = "models/llama-2-7b-chat.gguf"
    
    # Download the model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Downloading Llama 2 model to {model_path}...")
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        print("Download complete!")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model() 