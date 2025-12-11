"""
Model Downloader - Downloads ML model from Hugging Face
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_model_if_needed():
    """Download ML model from Hugging Face if not exists locally"""
    
    # Model configuration
    MODEL_DIR = Path("models/saved_models")
    MODEL_PATH = MODEL_DIR / "model_v1.h5"
    
    # Hugging Face direct download URL
    # Format: https://huggingface.co/USERNAME/REPO/resolve/main/model_v1.h5
    HUGGINGFACE_URL = "https://huggingface.co/Lynard/music-genre-model/resolve/main/model_v1.h5?download=true"
    
    # Check if model exists and is valid
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            print(f"Model already exists at {MODEL_PATH} ({size_mb:.2f} MB)")
            return True
        else:
            print(f" Model file corrupt ({size_mb:.2f} MB). Re-downloading...")
            MODEL_PATH.unlink()
    
    # Create directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model
    print(f"📥 Downloading model from Hugging Face...")
    print(f"   URL: {HUGGINGFACE_URL}")
    print(f"   Target: {MODEL_PATH}")
    print(f"   This may take 5-10 minutes...")
    
    try:
        # Stream download with progress
        response = requests.get(HUGGINGFACE_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        print(f"   File size: {total_size / (1024*1024):.2f} MB")
        
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress every 50MB
                    if downloaded % (50 * 1024 * 1024) < block_size:
                        progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"   Progress: {progress:.1f}% ({mb_downloaded:.1f} MB)")
        
        # Verify
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                print(f"Model downloaded successfully! ({size_mb:.2f} MB)")
                return True
            else:
                print(f"Download incomplete ({size_mb:.2f} MB)")
                return False
        else:
            print(" Model file not created")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f" Download error: {e}")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = download_model_if_needed()
    exit(0 if success else 1)
