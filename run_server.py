"""
Helper script to run the server
"""
import uvicorn
from app.core.config import settings
from app.utils.model_downloader import download_model_if_needed
import os

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
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"

    )
