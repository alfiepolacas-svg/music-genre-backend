"""
Lightweight mock predictor for development/low-memory environments
"""
import random
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MockGenrePredictor:
    """
    Mock predictor that simulates ML predictions without loading a large model
    Useful for development and memory-constrained environments
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.genres = [
            "blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock"
        ]
        logger.info("MockGenrePredictor initialized - using simulated predictions")
    
    @property
    def model(self):
        """Mock model property - always returns True to indicate 'loaded'"""
        return True
    
    def predict(self, audio_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Mock prediction that returns random but realistic results
        """
        try:
            # Simulate processing time
            time.sleep(0.5)
            
            # Generate realistic predictions
            predictions = {}
            for genre in self.genres:
                predictions[genre] = random.uniform(0.01, 0.95)
            
            # Normalize to sum to 1.0
            total = sum(predictions.values())
            predictions = {k: v/total for k, v in predictions.items()}
            
            # Get top prediction
            top_genre = max(predictions.items(), key=lambda x: x[1])
            predicted_genre = top_genre[0]
            confidence = top_genre[1]
            
            logger.info(f"Mock prediction: {predicted_genre} ({confidence:.2%})")
            
            return predicted_genre, confidence, predictions
            
        except Exception as e:
            logger.error(f"Mock prediction failed: {e}")
            # Return default prediction
            return "pop", 0.5, {"pop": 0.5, "rock": 0.3, "jazz": 0.2}
    
    def is_model_loaded(self) -> bool:
        """Mock method - always returns True"""
        return True