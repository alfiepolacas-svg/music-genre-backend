"""
Model Downloader - Downloads ML model from Google Drive
Optimized for large files (700+ MB)
"""
import os
import gdown
from pathlib import Path

def download_model_if_needed():
    """Download ML model from Google Drive if not exists locally"""
    
    # Model configuration
    MODEL_DIR = Path("models/saved_models")
    MODEL_PATH = MODEL_DIR / "model_v1.h5"
    
    # Google Drive file ID
    FILE_ID = "1Qv4V0EXyyXc0IRNZRhptWApHyOvlP31-"
    
    # Check if model exists and is valid (> 100 MB)
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # Valid model should be > 100MB
            print(f"✅ Model already exists at {MODEL_PATH} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"⚠️  Model file exists but is too small ({size_mb:.2f} MB). Re-downloading...")
            MODEL_PATH.unlink()  # Delete corrupted file
    
    # Create directory if needed
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model
    print(f"📥 Model not found. Downloading from Google Drive...")
    print(f"   Target: {MODEL_PATH}")
    print(f"   This may take 5-10 minutes for a 708 MB file...")
    
    try:
        # Construct proper Google Drive URL
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        
        # Download with gdown - use fuzzy matching for large files
        output = str(MODEL_PATH)
        
        print("   Starting download...")
        gdown.download(
            url, 
            output, 
            quiet=False,
            fuzzy=True  # Handle large file warnings automatically
        )
        
        # Verify download
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            
            if size_mb > 100:  # Valid model
                print(f"✅ Model downloaded successfully! ({size_mb:.2f} MB)")
                return True
            else:
                print(f"❌ Download failed - file too small ({size_mb:.2f} MB)")
                print("   This usually means the download link returned an error page.")
                MODEL_PATH.unlink()  # Delete bad file
                return False
        else:
            print("❌ Model file not created after download")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        
        # Try alternative: download to temporary name first
        try:
            print("   Trying alternative download method...")
            import subprocess
            
            # Use gdown CLI directly
            result = subprocess.run(
                ["gdown", "--fuzzy", url, "-O", str(MODEL_PATH)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and MODEL_PATH.exists():
                size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    print(f"✅ Model downloaded via CLI! ({size_mb:.2f} MB)")
                    return True
            
            print(f"❌ Alternative method also failed")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
        except Exception as e2:
            print(f"❌ Alternative method error: {e2}")
        
        return False

if __name__ == "__main__":
    success = download_model_if_needed()
    exit(0 if success else 1)
