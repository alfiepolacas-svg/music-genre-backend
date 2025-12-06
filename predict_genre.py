"""
Simple script to test trained model with audio file
"""
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import json
import sys

class GenrePredictor:
    def __init__(self, model_path, config_path, label_encoder_path):
        """Initialize predictor"""
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
            self.classes = label_data['classes']
        
        print(f" Model loaded from {model_path}")
        print(f" Classes: {', '.join(self.classes)}")
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file same as training"""
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.config['sample_rate']
        )
        
        # Fix length to 30 seconds
        target_length = self.config['sample_rate'] * self.config['duration']
        
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.config['n_mels'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length']
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Reshape for model input
        mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]
        
        return mel_spec_db
    
    def predict(self, audio_path):
        """Predict genre of audio file"""
        print(f"\n Analyzing: {audio_path}")
        
        # Preprocess
        features = self.preprocess_audio(audio_path)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        print("\n Predictions:")
        print("-" * 40)
        for i, idx in enumerate(top_indices):
            genre = self.classes[idx]
            confidence = predictions[0][idx] * 100
            
            bar = " " * int(confidence / 2)
            print(f"{i+1}. {genre:12s} {confidence:5.2f}% {bar}")
        
        # Return top prediction
        top_genre = self.classes[top_indices[0]]
        top_confidence = predictions[0][top_indices[0]] * 100
        
        return top_genre, top_confidence


def main():
    """Main function"""
    print("=" * 60)
    print(" MUSIC GENRE PREDICTOR")
    print("=" * 60)
    
    # Paths
    model_path = "models/saved_models/model_v1.h5"
    config_path = "models/saved_models/training_config.json"
    label_encoder_path = "models/saved_models/label_encoder.json"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f" Error: Model not found at {model_path}")
        print("   Run 'py train_model.py' first to train the model.")
        return
    
    # Initialize predictor
    predictor = GenrePredictor(model_path, config_path, label_encoder_path)
    
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Default: use a sample from validation set
        print("\nℹ  No audio file specified. Usage:")
        print("   py predict_genre.py path/to/audio.wav")
        print("\n Searching for sample audio in data folder...")
        
        # Try to find a sample
        sample_path = None
        data_dir = Path("data/raw/gtzan")
        
        if data_dir.exists():
            for genre_dir in data_dir.iterdir():
                if genre_dir.is_dir():
                    audio_files = list(genre_dir.glob("*.wav")) + list(genre_dir.glob("*.mp3"))
                    if audio_files:
                        sample_path = str(audio_files[0])
                        break
        
        if sample_path:
            audio_path = sample_path
            print(f"   Using sample: {audio_path}")
        else:
            print(" No audio files found. Please specify a file:")
            print("   py predict_genre.py path/to/audio.wav")
            return
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f" Error: Audio file not found: {audio_path}")
        return
    
    # Predict
    try:
        genre, confidence = predictor.predict(audio_path)
        
        print("\n" + "=" * 60)
        print(f" RESULT: {genre.upper()} ({confidence:.2f}% confidence)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()