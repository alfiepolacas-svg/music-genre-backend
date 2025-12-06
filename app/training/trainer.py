"""
Genre Classification Model Trainer (CNN-based)
Trains CNN model on GTZAN dataset for music genre classification
"""
import os
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenreTrainer:
    """Trainer for music genre classification model"""
    
    def __init__(
        self,
        data_dir,
        model_dir,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.2,
        sample_rate=22050,
        duration=30,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        
        # Audio processing parameters
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.model = None
        self.label_encoder = None
        self.history = None
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load audio files and extract features"""
        logger.info("Loading and preprocessing data...")
        
        X = []
        y = []
        
        # Get all genre folders
        genre_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        
        for genre_folder in genre_folders:
            genre_name = genre_folder.name
            logger.info(f"Processing genre: {genre_name}")
            
            # Get all audio files in genre folder
            audio_files = list(genre_folder.glob("*.wav")) + list(genre_folder.glob("*.mp3"))
            
            for audio_file in audio_files:
                try:
                    # Load audio WITHOUT duration limit first
                    audio, sr = librosa.load(
                        audio_file,
                        sr=self.sample_rate
                    )
                    
                    # Fix length: Pad or trim to exact 30 seconds
                    target_length = self.sample_rate * self.duration
                    
                    if len(audio) > target_length:
                        # Trim if too long
                        audio = audio[:target_length]
                    else:
                        # Pad with zeros if too short
                        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                    
                    # Extract mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio,
                        sr=sr,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length
                    )
                    
                    # Convert to log scale (dB)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    X.append(mel_spec_db)
                    y.append(genre_name)
                    
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Reshape for CNN (add channel dimension)
        X = X[..., np.newaxis]
        
        logger.info(f"Data loaded: {X.shape}, Labels: {len(self.label_encoder.classes_)} classes")
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """Build CNN model for genre classification"""
        logger.info("Building model...")
        
        model = keras.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built with {num_classes} output classes")
        model.summary()
        
        return model
    
    def train(self):
        """Train the model"""
        logger.info("Starting training...")
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        
        # Build model
        self.model = self.build_model(
            input_shape=X_train.shape[1:],
            num_classes=len(self.label_encoder.classes_)
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete!")
        
        return self.history
    
    def evaluate(self):
        """Evaluate the model"""
        if self.model is None:
            logger.error("Model not trained yet!")
            return None
        
        logger.info("Evaluating model...")
        
        # For now, return training history metrics
        final_metrics = {
            'accuracy': self.history.history['val_accuracy'][-1],
            'loss': self.history.history['val_loss'][-1]
        }
        
        logger.info(f"Final Validation Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"Final Validation Loss: {final_metrics['loss']:.4f}")
        
        return final_metrics
    
    def save_model(self, model_path):
        """Save trained model and label encoder"""
        if self.model is None:
            logger.error("No model to save!")
            return
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save label encoder
        label_encoder_path = model_path.parent / 'label_encoder.json'
        with open(label_encoder_path, 'w') as f:
            json.dump({
                'classes': self.label_encoder.classes_.tolist()
            }, f)
        logger.info(f"Label encoder saved: {label_encoder_path}")
        
        # Save training config
        config_path = model_path.parent / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'n_mels': self.n_mels,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'num_classes': len(self.label_encoder.classes_),
                'classes': self.label_encoder.classes_.tolist()
            }, f, indent=2)
        logger.info(f"Config saved: {config_path}")