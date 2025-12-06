"""
Prediction schemas
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class PredictionResponse(BaseModel):
    """Response model for genre prediction"""
    genre_id: str = Field(..., description="Genre identifier")
    genre_name: str = Field(..., description="Genre name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    all_predictions: Dict[str, float] = Field(..., description="All genre predictions")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "genre_id": "rock",
                "genre_name": "Rock",
                "confidence": 0.87,
                "all_predictions": {
                    "Rock": 0.87,
                    "Metal": 0.08,
                    "Pop": 0.03,
                    "Jazz": 0.02
                },
                "audio_duration": 30.0,
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    status_code: int