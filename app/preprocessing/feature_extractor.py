"""
Feature Extraction
"""
import librosa
import numpy as np
from typing import Dict, Tuple
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract audio features for genre classification"""
    
    def __init__(
        self,
        sample_rate: int = None,
        n_mfcc: int = None,
        n_mels: int = None
    ):
        self.sample_rate = sample_rate or settings.SAMPLE_RATE
        self.n_mfcc = n_mfcc or settings.N_MFCC
        self.n_mels = n_mels or settings.N_MELS
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        return np.mean(mfccs, axis=1)
    
    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Extract Chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        return np.mean(chroma, axis=1)
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        features = {}
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate
        )
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate
        )
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        return features
    
    def extract_rhythm_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract rhythm features"""
        features = {}
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        features['tempo'] = float(tempo)
        
        return features
    
    def extract_all_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract all features and concatenate
        Returns feature vector of size 36:
        - 13 MFCC coefficients
        - 12 Chroma features
        - 6 Spectral features
        - 1 Tempo
        """
        features = []
        
        # MFCC (13)
        mfcc = self.extract_mfcc(y)
        features.extend(mfcc)
        
        # Chroma (12)
        chroma = self.extract_chroma(y)
        features.extend(chroma)
        
        # Spectral (6)
        spectral = self.extract_spectral_features(y)
        features.extend(spectral.values())
        
        # Rhythm (1)
        rhythm = self.extract_rhythm_features(y)
        features.extend(rhythm.values())
        
        return np.array(features, dtype=np.float32)
    
    def extract_mel_spectrogram(
        self,
        y: np.ndarray,
        img_size: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Extract Mel-spectrogram for CNN input
        Returns: (height, width, 1) shaped array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=settings.N_FFT,
            hop_length=settings.HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize
        from scipy.ndimage import zoom
        zoom_factors = (
            img_size[0] / mel_spec_db.shape[0],
            img_size[1] / mel_spec_db.shape[1]
        )
        mel_spec_resized = zoom(mel_spec_db, zoom_factors)
        
        # Add channel dimension
        mel_spec_resized = np.expand_dims(mel_spec_resized, axis=-1)
        
        return mel_spec_resized

# Global instance
feature_extractor = FeatureExtractor()