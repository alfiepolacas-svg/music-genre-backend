"""
Audio Utility Functions
"""
import librosa
import numpy as np
from typing import Tuple
import os

class AudioUtils:
    """Utility functions for audio processing"""
    
    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """Get audio file information"""
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                "sample_rate": sr,
                "duration": duration,
                "samples": len(y),
                "channels": 1,  # Mono after librosa.load
                "file_size": os.path.getsize(file_path)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def convert_to_mono(y: np.ndarray) -> np.ndarray:
        """Convert stereo to mono"""
        if y.ndim > 1:
            y = librosa.to_mono(y)
        return y
    
    @staticmethod
    def resample_audio(
        y: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return y
    
    @staticmethod
    def trim_silence(
        y: np.ndarray,
        top_db: int = 20
    ) -> np.ndarray:
        """Trim leading and trailing silence"""
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed
    
    @staticmethod
    def normalize_audio(y: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1]"""
        return librosa.util.normalize(y)

audio_utils = AudioUtils()