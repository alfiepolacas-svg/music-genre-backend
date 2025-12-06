"""
Model Downloader - Downloads ML model from Google Drive
"""
import gdown
import os
from pathlib import Path

def download_model_if_needed():
    """Download ML model from Google Drive if not exists locally"""
    
    # Model configuration
    MODEL_DIR = Path("models/saved_models")
    MODEL_PATH = MODEL_DIR / "model_v1.h5"
    
    # Replace with your Google Drive file ID
    GOOGLE_DRIVE_FILE_ID = "1Qv4V0EXyyXc0IRNZRhptWApHyOvlP31-"
    
    # Check if model exists
    if MODEL_PATH.exists():
        print(f" Model already exists at {MODEL_PATH}")
        return True
    
    # Create directory if needed
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model
    print(f" Model not found. Downloading from Google Drive...")
    print(f"   Target: {MODEL_PATH}")
    
    try:
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, str(MODEL_PATH), quiet=False)
        
        # Verify download
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            print(f" Model downloaded successfully! ({size_mb:.2f} MB)")
            return True
        else:
            print(" Model download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f" Error downloading model: {e}")
        return False

if __name__ == "__main__":
    download_model_if_needed()
