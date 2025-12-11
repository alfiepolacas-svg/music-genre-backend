"""
Helper script to run the server
"""
import uvicorn
import logging
import os

# Suppress TensorFlow warnings BEFORE importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        Music Genre Classification API Server                ║
    ║                                                              ║
    ║  Server running at: http://{settings.HOST}:{settings.PORT}                  ║
    ║  API Documentation: http://{settings.HOST}:{settings.PORT}/docs            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\nStarting Uvicorn server...\n")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

