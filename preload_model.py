#!/usr/bin/env python
"""
Pre-load ML model before starting the server
This ensures the model is downloaded and loaded before accepting requests
"""

import os
import sys
import logging
from app.ml.inference.predictor import predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preload_model():
    """Pre-load the ML model to avoid timeout issues"""
    try:
        logger.info("Starting model preload process...")
        
        # Force model loading by accessing the model property
        if predictor.model is None:
            logger.info("Model needs to be loaded...")
            # This will trigger the download if needed
            _ = predictor.model
            
        if predictor.model is not None:
            logger.info(" Model preloaded successfully!")
            logger.info(f"Model shape: {predictor.model.input_shape}")
            return True
        else:
            logger.error("Failed to preload model")
            return False
            
    except Exception as e:
        logger.error(f"Error during model preload: {e}")
        return False

if __name__ == "__main__":
    success = preload_model()
    sys.exit(0 if success else 1)