import numpy as np
import tensorflow as tf
import librosa
from typing import Dict, Tuple, List
import logging
from app.preprocessing.audio_processor import audio_processor
from app.preprocessing.feature_extractor import feature_extractor
from app.core.constants import GENRES

logger = logging.getLogger(__name__)

class GenrePredictor:
    """
    ML Model predictor for genre classification
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try alternative paths
            try:
                alt_path = self.model_path.replace("models/saved_models/model_v1.h5", "models/best_model.h5")
                self.model = tf.keras.models.load_model(alt_path)
                logger.info(f"Model loaded from alternative path: {alt_path}")
            except Exception as e2:
                logger.error(f"Failed to load from alternative path: {e2}")
                # Create mock model for development
                self.model = None
    
    def predict(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict genre from audio file
        Returns: (predicted_genre, confidence, all_predictions)
        """
        try:
            # Preprocess audio
            audio = audio_processor.preprocess(audio_path)
            
            # Extract mel-spectrogram features (for CNN model)
            # Calculate natural dimensions based on audio length and hop_length
            # For 30s audio: frames = (22050 * 30) / 512 ≈ 1292
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=audio_processor.sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Add channel dimension: (128, time_frames) -> (128, time_frames, 1)
            features = np.expand_dims(mel_spec_db, axis=-1)
            
            # Reshape for model input (add batch dimension)
            features = np.expand_dims(features, axis=0)
            
            logger.info(f"Audio shape: {audio.shape}")
            logger.info(f"Mel-spectrogram shape: {mel_spec_db.shape}")
            logger.info(f"Final features shape: {features.shape}")
            logger.info(f"Expected model input: (None, 128, 1292, 1)")
            
            # Predict
            if self.model:
                predictions = self.model.predict(features, verbose=0)[0]
            else:
                # Mock prediction for development
                predictions = np.random.dirichlet(np.ones(len(GENRES)))
            
            # Get results with bounds checking
            predicted_idx = np.argmax(predictions)
            
            # Safety check for index bounds
            if predicted_idx >= len(GENRES):
                logger.error(f"Predicted index {predicted_idx} out of bounds for GENRES list (length: {len(GENRES)})")
                logger.error(f"Model output shape: {predictions.shape}, GENRES: {GENRES}")
                predicted_idx = 0  # Default to first genre
            
            predicted_genre = GENRES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            
            logger.info(f"Predicted: {predicted_genre} with confidence: {confidence:.3f}")
            logger.info(f"Model output length: {len(predictions)}, GENRES length: {len(GENRES)}")
            
            # Create predictions dictionary with length matching
            all_predictions = {}
            min_length = min(len(GENRES), len(predictions))
            
            for i in range(min_length):
                all_predictions[GENRES[i]] = float(predictions[i])
            
            # If there are more predictions than genres, ignore excess
            if len(predictions) > len(GENRES):
                logger.warning(f"Model returned {len(predictions)} predictions but only {len(GENRES)} genres available")
            
            # If there are more genres than predictions, pad with zeros
            if len(GENRES) > len(predictions):
                logger.warning(f"Only {len(predictions)} predictions for {len(GENRES)} genres, padding with zeros")
                for i in range(len(predictions), len(GENRES)):
                    all_predictions[GENRES[i]] = 0.0
            
            # Sort by confidence
            all_predictions = dict(
                sorted(all_predictions.items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
            
            return predicted_genre, confidence, all_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        audio_paths: List[str]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict genres for multiple audio files"""
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            results.append(result)
        return results

predictor = GenrePredictor(model_path="models/saved_models/model_v1.h5")