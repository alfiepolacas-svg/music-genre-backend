from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import aiofiles
import os
import uuid
import logging

from app.ml.inference.predictor import predictor
from app.preprocessing.audio_processor import audio_processor
from app.schemas.prediction import PredictionResponse, ErrorResponse
from app.api.v1.deps import validate_audio_file, validate_file_size
from app.core.config import settings
from app.core.constants import GENRE_METADATA

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_genre(
    audio: UploadFile = File(..., description="Audio file to classify")
):
    """
    Predict music genre from audio file
    
    This endpoint matches the Flutter frontend:
    - ApiService.classifyAudio()
    - MlService.identifyGenre()
    
    Returns:
        PredictionResponse with genre classification results
    """
    file_path = None
    
    try:
        # Validate file
        validate_audio_file(audio)
        await validate_file_size(audio)
        
        # Save uploaded file temporarily
        file_ext = os.path.splitext(audio.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
        
        logger.info(f"Saving uploaded file: {temp_filename}")
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)
        
        # Get audio duration
        duration = audio_processor.get_duration(file_path)
        
        # Predict genre
        logger.info("Starting genre prediction...")
        genre_name, confidence, all_predictions = predictor.predict(file_path)
        
        # Get genre metadata
        genre_metadata = GENRE_METADATA.get(genre_name, {})
        genre_id = genre_metadata.get('id', genre_name.lower().replace(' ', '_'))
        
        # Create response
        response = PredictionResponse(
            genre_id=genre_id,
            genre_name=genre_name,
            confidence=confidence,
            all_predictions=all_predictions,
            audio_duration=duration,
            timestamp=datetime.now()
        )
        
        logger.info(f"Prediction successful: {genre_name} ({confidence:.2%})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Cleanup: delete temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

@router.post("/predict/batch")
async def predict_batch(
    audio_files: list[UploadFile] = File(...)
):
    """Predict genres for multiple audio files"""
    results = []
    
    for audio in audio_files:
        try:
            result = await predict_genre(audio)
            results.append({
                "filename": audio.filename,
                "prediction": result
            })
        except Exception as e:
            results.append({
                "filename": audio.filename,
                "error": str(e)
            })
    
    return {"results": results}