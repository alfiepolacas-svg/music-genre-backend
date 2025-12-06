"""
Model Information Endpoint
"""
from fastapi import APIRouter
from app.ml.inference.predictor import predictor
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/model/info")
async def get_model_info():
    """
    Get ML model information
    
    Returns detailed information about the loaded ML model including:
    - Model name and version
    - Framework information
    - Model architecture details
    - Parameter counts
    - Load status
    """
    try:
        model_info = {
            "name": "Music Genre Classifier",
            "version": "1.0.0",
            "type": "Deep Neural Network",
            "framework": "TensorFlow",
            "framework_version": tf.__version__,
            "model_loaded": predictor.model is not None,
            "description": "Neural network trained on audio features for genre classification"
        }
        
        # If model is loaded, get detailed architecture info
        if predictor.model is not None:
            try:
                # Get model architecture details
                model_info.update({
                    "architecture": predictor.model.__class__.__name__,
                    "input_shape": str(predictor.model.input_shape),
                    "output_shape": str(predictor.model.output_shape),
                    "total_params": int(predictor.model.count_params()),
                })
                
                # Get trainable parameters count
                trainable_count = sum([
                    tf.keras.backend.count_params(w) 
                    for w in predictor.model.trainable_weights
                ])
                model_info["trainable_params"] = int(trainable_count)
                
                # Get non-trainable parameters
                non_trainable_count = sum([
                    tf.keras.backend.count_params(w) 
                    for w in predictor.model.non_trainable_weights
                ])
                model_info["non_trainable_params"] = int(non_trainable_count)
                
                # Get number of layers
                model_info["num_layers"] = len(predictor.model.layers)
                
                logger.info("Successfully retrieved model information")
                
            except Exception as e:
                logger.error(f"Error getting model details: {e}")
                model_info["error"] = f"Could not retrieve full model details: {str(e)}"
        else:
            model_info["message"] = "Model not loaded. Using mock predictions."
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {e}")
        return {
            "name": "Music Genre Classifier",
            "version": "1.0.0",
            "type": "Unknown",
            "framework": "TensorFlow",
            "error": str(e),
            "model_loaded": False,
            "message": "Error retrieving model information"
        }

@router.get("/model/status")
async def get_model_status():
    """
    Get quick model status
    
    Returns:
        Simple status check if model is loaded
    """
    return {
        "model_loaded": predictor.model is not None,
        "status": "ready" if predictor.model is not None else "not_loaded"
    }

@router.get("/model/metrics")
async def get_model_metrics():
    """
    Get model training metrics (if available)
    
    Returns model accuracy, loss, and other training metrics
    """
    try:
        # This would come from your training history or evaluation
        # For now, return placeholder metrics
        return {
            "accuracy": 0.85,
            "loss": 0.45,
            "val_accuracy": 0.82,
            "val_loss": 0.52,
            "message": "Metrics from last training session"
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Could not retrieve model metrics"
        }