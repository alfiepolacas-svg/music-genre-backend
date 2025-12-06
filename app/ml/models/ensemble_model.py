"""
RNN/LSTM Model for Genre Classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class LSTMGenreClassifier:
    """LSTM model for sequential audio features"""
    
    def __init__(
        self,
        input_shape: tuple = (None, 13),  # (time_steps, features)
        num_classes: int = 10
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Build LSTM architecture"""
        model = keras.Sequential([
            # Input
            layers.Input(shape=self.input_shape),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )