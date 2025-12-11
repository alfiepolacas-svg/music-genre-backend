"""
FastAPI Main Application
"""
import os

# Suppress TensorFlow warnings BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from app.core.config import settings
from app.api.v1.endpoints import predict
from app.api.v1.endpoints import genres as genres_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Music Genre Classification API",
    description="ML-powered API for music genre classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predict.router,
    prefix="/api/v1",
    tags=["prediction"]
)

app.include_router(
    genres_router.router,
    prefix="/api/v1",
    tags=["genres"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Music Genre Classification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Railway
    IMPORTANT: Must respond quickly without loading the model
    """
    try:
        # Check if predictor exists (don't access .model property!)
        model_status = "not_loaded"
        
        # Only check if model is already loaded (don't trigger lazy load)
        if hasattr(predict.predictor, '_model') and predict.predictor._model is not None:
            model_status = "loaded"
        
        return {
            "status": "healthy",
            "service": "music-genre-api",
            "model_status": model_status,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        # Always return healthy even if there's an error
        # Railway just needs to know the service is responding
        return {
            "status": "healthy",
            "service": "music-genre-api",
            "model_status": "unknown"
        }

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info(" Music Genre Classification API started")
    logger.info(" Model will load on first prediction request")
    logger.info(f" Running on {settings.HOST}:{settings.PORT}")
    logger.info(f" Docs available at http://{settings.HOST}:{settings.PORT}/docs")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        timeout_keep_alive=300
    )

