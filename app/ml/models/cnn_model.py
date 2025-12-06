"""
CNN Model for Genre Classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

class CNNGenreClassifier:
    """CNN model using Mel-spectrograms"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 1),
        num_classes: int = 10
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Build CNN architecture"""
        model = keras.Sequential([
            # Input
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
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
    
    def summary(self):
        """Model summary"""
        if self.model:
            return self.model.summary()
        return "Model not built"


class DenseGenreClassifier:
    """Dense NN for extracted features"""
    
    def __init__(
        self,
        input_dim: int = 36,
        num_classes: int = 10
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Build Dense architecture"""
        model = keras.Sequential([
            # Input
            layers.Input(shape=(self.input_dim,)),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
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
    
    def summary(self):
        """Model summary"""
        if self.model:
            return self.model.summary()
        return "Model not built"