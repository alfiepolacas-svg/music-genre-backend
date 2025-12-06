"""
API Dependencies
"""
from fastapi import HTTPException, UploadFile
import os
from app.core.config import settings

def validate_audio_file(file: UploadFile) -> bool:
    """Validate uploaded audio file"""
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_audio_formats_list:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
        )
    
    return True

async def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    # Read file to check size
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    return True