"""
Data Augmentation for Audio
"""
import librosa
import numpy as np
from typing import List

class AudioAugmenter:
    """Audio data augmentation"""
    
    @staticmethod
    def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random noise"""
        noise = np.random.randn(len(y))
        augmented = y + noise_factor * noise
        return augmented.astype(type(y[0]))
    
    @staticmethod
    def time_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """Shift audio in time"""
        shift = np.random.randint(int(len(y) * shift_max))
        return np.roll(y, shift)
    
    @staticmethod
    def pitch_shift(y: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
        """Shift pitch"""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(y: np.ndarray, rate: float = 1.2) -> np.ndarray:
        """Time stretch"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def augment(self, y: np.ndarray, sr: int) -> List[np.ndarray]:
        """Apply all augmentations"""
        augmented_data = [y]  # Original
        
        # Add variations
        augmented_data.append(self.add_noise(y))
        augmented_data.append(self.time_shift(y))
        augmented_data.append(self.pitch_shift(y, sr, 2))
        augmented_data.append(self.pitch_shift(y, sr, -2))
        
        return augmented_data

audio_augmenter = AudioAugmenter()