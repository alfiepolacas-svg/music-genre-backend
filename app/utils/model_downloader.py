"""
Model Downloader - Downloads ML model from Google Drive
"""
import os
import requests
from pathlib import Path

def download_model_if_needed():
    """Download ML model from Google Drive if not exists locally"""
    
    # Model configuration
    MODEL_DIR = Path("models/saved_models")
    MODEL_PATH = MODEL_DIR / "model_v1.h5"
    
    # Google Drive direct download link
    # Get this from: Right-click file → Get link → Copy
    FILE_ID = "1Qv4V0EXyyXc0IRNZRhptWApHyOvlP31-"
    
    # Check if model exists
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at {MODEL_PATH}")
        return True
    
    # Create directory if needed
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model using requests (more reliable than gdown)
    print(f"📥 Model not found. Downloading from Google Drive...")
    print(f"   Target: {MODEL_PATH}")
    
    try:
        # Google Drive download URL
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        
        print("   Initiating download...")
        
        # Start session
        session = requests.Session()
        
        # First request
        response = session.get(url, stream=True)
        
        # Check for download warning (large files)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm={value}"
                response = session.get(url, stream=True)
                break
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress every 50MB
                    if downloaded % (50 * 1024 * 1024) == 0:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"   Downloaded: {mb_downloaded:.1f} MB...")
        
        # Verify download
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            print(f"✅ Model downloaded successfully! ({size_mb:.2f} MB)")
            return True
        else:
            print("❌ Model download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        
        # Fallback: try gdown
        try:
            print("   Trying alternative method (gdown)...")
            import gdown
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, str(MODEL_PATH), quiet=False)
            
            if MODEL_PATH.exists():
                size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
                print(f"✅ Model downloaded via gdown! ({size_mb:.2f} MB)")
                return True
        except Exception as e2:
            print(f"❌ Gdown also failed: {e2}")
        
        return False

if __name__ == "__main__":
    download_model_if_needed()
