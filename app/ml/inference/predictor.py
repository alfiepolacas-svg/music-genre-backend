"""
Lazy-loading ML Predictor for Music Genre Classification
"""
import numpy as np
import tensorflow as tf
import librosa
from typing import Dict, Tuple, List
import logging
import os
from pathlib import Path
from app.preprocessing.audio_processor import audio_processor
from app.core.constants import GENRES

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

class GenrePredictor:
    """
    ML Model predictor for genre classification with lazy loading
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._loading = False
        logger.info(f"GenrePredictor initialized (model will load on first prediction)")
    
    @property
    def model(self):
        """Lazy load model on first access"""
        if self._model is None and not self._loading:
            self._loading = True
            try:
                logger.info("🔄 Loading model for first time...")
                
                # Check if model file exists
                model_file = Path(self.model_path)
                if not model_file.exists():
                    logger.warning(f"⚠️ Model not found at {self.model_path}")
                    
                    # Try to download model
                    from app.utils.model_downloader import download_model_if_needed
                    logger.info("📥 Attempting to download model...")
                    
                    if download_model_if_needed():
                        logger.info("✅ Model downloaded successfully")
                    else:
                        logger.error("❌ Model download failed")
                        self._model = None
                        return None
                
                # Load model
                if model_file.exists():
                    logger.info(f"📂 Loading model from {self.model_path}")
                    self._model = tf.keras.models.load_model(self.model_path)
                    logger.info(f"✅ Model loaded successfully!")
                    logger.info(f"   Input shape: {self._model.input_shape}")
                    logger.info(f"   Output shape: {self._model.output_shape}")
                else:
                    logger.error("❌ Model file still not found after download attempt")
                    self._model = None
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}", exc_info=True)
                self._model = None
            finally:
                self._loading = False
        
        return self._model
    
    def predict(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict genre from audio file
        Returns: (predicted_genre, confidence, all_predictions)
        """
        try:
            # Get model (triggers lazy load if needed)
            model = self.model
            
            if model is None:
                logger.warning("⚠️ Model not available, using mock prediction")
                return self._mock_prediction()
            
            # Preprocess audio
            logger.info(f"🎵 Processing audio: {audio_path}")
            audio = audio_processor.preprocess(audio_path)
            
            # Extract mel-spectrogram features (for CNN model)
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
            
            logger.info(f"   Audio shape: {audio.shape}")
            logger.info(f"   Mel-spectrogram shape: {mel_spec_db.shape}")
            logger.info(f"   Final features shape: {features.shape}")
            
            # Predict
            predictions = model.predict(features, verbose=0)[0]
            
            # Get results with bounds checking
            predicted_idx = np.argmax(predictions)
            
            # Safety check for index bounds
            if predicted_idx >= len(GENRES):
                logger.error(f"Predicted index {predicted_idx} out of bounds for GENRES list (length: {len(GENRES)})")
                predicted_idx = 0  # Default to first genre
            
            predicted_genre = GENRES[predicted_idx]
            confidence = float(predictions[predicted_idx])
            
            logger.info(f"✅ Predicted: {predicted_genre} with confidence: {confidence:.3f}")
            
            # Create predictions dictionary
            all_predictions = {}
            min_length = min(len(GENRES), len(predictions))
            
            for i in range(min_length):
                all_predictions[GENRES[i]] = float(predictions[i])
            
            # Handle length mismatches
            if len(predictions) > len(GENRES):
                logger.warning(f"Model returned {len(predictions)} predictions but only {len(GENRES)} genres available")
            
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
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            logger.warning("⚠️ Falling back to mock prediction")
            return self._mock_prediction()
    
    def _mock_prediction(self) -> Tuple[str, float, Dict[str, float]]:
        """
        Fallback mock prediction when model is unavailable
        """
        import random
        
        # Generate random predictions
        predictions = np.random.dirichlet(np.ones(len(GENRES)))
        
        predicted_idx = np.argmax(predictions)
        predicted_genre = GENRES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        all_predictions = {
            genre: float(pred)
            for genre, pred in zip(GENRES, predictions)
        }
        
        # Sort by confidence
        all_predictions = dict(
            sorted(all_predictions.items(), 
                  key=lambda x: x[1], 
                  reverse=True)
        )
        
        logger.info(f"🎲 Mock prediction: {predicted_genre} ({confidence:.3f})")
        
        return predicted_genre, confidence, all_predictions
    
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


# Singleton instance - model will load lazily on first prediction
predictor = GenrePredictor(model_path="models/saved_models/model_v1.h5")
