"""
Post-processing for predictions
"""
import numpy as np
from typing import Dict, List, Tuple

class PredictionPostProcessor:
    """Post-process model predictions"""
    
    @staticmethod
    def apply_threshold(
        predictions: Dict[str, float],
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Filter predictions below threshold
        
        Args:
            predictions: Dictionary of genre -> probability
            threshold: Minimum probability to include
            
        Returns:
            Filtered predictions dictionary
        """
        return {
            genre: prob
            for genre, prob in predictions.items()
            if prob >= threshold
        }
    
    @staticmethod
    def get_top_k(
        predictions: Dict[str, float],
        k: int = 3
    ) -> Dict[str, float]:
        """
        Get top K predictions
        
        Args:
            predictions: Dictionary of genre -> probability
            k: Number of top predictions to return
            
        Returns:
            Dictionary with top K predictions
        """
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_preds[:k])
    
    @staticmethod
    def smooth_predictions(
        predictions: np.ndarray,
        alpha: float = 0.9
    ) -> np.ndarray:
        """
        Smooth predictions using exponential smoothing
        
        Args:
            predictions: Array of predictions
            alpha: Smoothing factor (0-1)
            
        Returns:
            Smoothed predictions array
        """
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]
        
        for i in range(1, len(predictions)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * predictions[i]
        
        return smoothed
    
    @staticmethod
    def normalize_predictions(
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize predictions to sum to 1.0
        
        Args:
            predictions: Dictionary of genre -> probability
            
        Returns:
            Normalized predictions
        """
        total = sum(predictions.values())
        if total == 0:
            return predictions
        
        return {
            genre: prob / total
            for genre, prob in predictions.items()
        }
    
    @staticmethod
    def get_confidence_level(confidence: float) -> str:
        """
        Get confidence level description
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence level string
        """
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    @staticmethod
    def format_prediction_response(
        genre_name: str,
        confidence: float,
        all_predictions: Dict[str, float],
        top_k: int = 5
    ) -> Dict:
        """
        Format prediction response for API
        
        Args:
            genre_name: Predicted genre name
            confidence: Confidence score
            all_predictions: All genre predictions
            top_k: Number of top predictions to include
            
        Returns:
            Formatted response dictionary
        """
        post_processor = PredictionPostProcessor()
        
        return {
            "predicted_genre": genre_name,
            "confidence": confidence,
            "confidence_level": post_processor.get_confidence_level(confidence),
            "top_predictions": post_processor.get_top_k(all_predictions, k=top_k),
            "all_predictions": all_predictions
        }

# Global instance
post_processor = PredictionPostProcessor()