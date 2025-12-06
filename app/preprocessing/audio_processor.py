"""
Audio Preprocessing
"""
import librosa
import numpy as np
from typing import Tuple
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio preprocessing for ML model"""
    
    def __init__(
        self,
        sample_rate: int = None,
        duration: float = None
    ):
        self.sample_rate = sample_rate or settings.SAMPLE_RATE
        self.duration = duration or settings.AUDIO_DURATION
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            y, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.duration
            )
            logger.info(f"Loaded audio: {file_path}, shape: {y.shape}")
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]"""
        return librosa.util.normalize(y)
    
    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        target_length = int(self.sample_rate * self.duration)
        
        if len(y) < target_length:
            # Pad with zeros
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            # Truncate
            y = y[:target_length]
        
        return y
    
    def get_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
        except Exception as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0
    
    def preprocess(self, file_path: str) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Load audio
        y, sr = self.load_audio(file_path)
        
        # Normalize
        y = self.normalize_audio(y)
        
        # Pad or truncate
        y = self.pad_or_truncate(y)
        
        return y

# Global instance
audio_processor = AudioProcessor()