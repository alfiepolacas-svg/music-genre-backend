# Music Genre Classification Backend

ML-powered backend for music genre classification using FastAPI and TensorFlow.

## Features
- Audio genre classification (10 genres)
- RESTful API endpoints
- Real-time prediction
- Model training pipeline

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

## Usage
```bash
# Run server
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/v1/predict` - Predict genre from audio file
- `GET /api/v1/genres` - Get all available genres
- `GET /health` - Health check

## Project Structure
```
music_genre_ml_backend/
├── app/           # Application code
├── data/          # Dataset storage
├── models/        # Trained models
├── notebooks/     # Jupyter notebooks
└── scripts/       # Training scripts
```

## Development
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Train model
python scripts/train_model.py
```