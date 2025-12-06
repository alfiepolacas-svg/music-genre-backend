"""
Custom Training Callbacks
"""
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)

class CustomLoggingCallback(keras.callbacks.Callback):
    """Custom callback for detailed logging"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch"""
        logs = logs or {}
        
        logger.info(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )

class ConfusionMatrixCallback(keras.callbacks.Callback):
    """Callback to log confusion matrix"""
    
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate confusion matrix at end of epoch"""
        if (epoch + 1) % 10 == 0:  # Every 10 epochs
            predictions = self.model.predict(self.X_val, verbose=0)
            # Log or save confusion matrix here
            pass