"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))
    DEBUG: bool = True
    
    # CORS - Fixed to accept string
    ALLOWED_ORIGINS: str = "*"
    
    # Upload
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_AUDIO_FORMATS: str = ".mp3,.wav,.m4a,.flac"
    UPLOAD_DIR: str = "uploads"
    
    # ML Model
    MODEL_PATH: str = "models/saved_models/genre_classifier.h5"
    SAMPLE_RATE: int = 22050
    AUDIO_DURATION: float = 30.0
    
    # Features
    N_MFCC: int = 13
    N_MELS: int = 128
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    
    # Convert string to list for CORS
    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert ALLOWED_ORIGINS string to list"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # Convert string to list for audio formats
    @property
    def allowed_audio_formats_list(self) -> List[str]:
        """Convert ALLOWED_AUDIO_FORMATS string to list"""
        return [fmt.strip() for fmt in self.ALLOWED_AUDIO_FORMATS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create necessary directories
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path("models/saved_models").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)