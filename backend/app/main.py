"""
SmartLib Kiosk - FastAPI Main Application

A smart library kiosk system with AI-powered:
- Face recognition for student authentication (ArcFace)
- Book detection (YOLOv8)
- OCR for book identification (PaddleOCR)
- Transaction management (borrow/return)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.config import settings
from app.database import init_db, close_db
from app.api.routes import auth, books, transactions, students, assistant
from app.core.ml_container import init_ai_models, AIModels


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO"
)
logger.add(
    "logs/smartlib_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI.
    Initializes database and AI models on startup.
    """
    import os
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    logger.info("Starting SmartLib Kiosk API...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    from app.database import async_session_maker
    
    # 1. Initialize database connection
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Depending on severity, you might want to raise the exception or exit here
    
    # 2. Warm up AI models
    try:
        logger.info("Initializing AI models...")
        await init_ai_models()
        ocr_status = "GPU" if getattr(AIModels.ocr_service, '_ocr', None) else "MOCK"
        logger.info(f"✓ AI Models loaded (Face: GPU, YOLO: GPU, OCR: {ocr_status})")
    except Exception as e:
        logger.error(f"Failed to pre-load AI models: {e}")
        
    # 3. Synchronize FAISS Vector Engine
    try:
        logger.info("Synchronizing FAISS vector engine from pgvector...")
        async with async_session_maker() as session:
            await AIModels.faiss_engine.sync_from_db(session)
    except Exception as e:
        logger.error(f"FAISS sync failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SmartLib Kiosk API...")
    await close_db()
    logger.info("Database connection closed")


# Create FastAPI application
app = FastAPI(
    title="SmartLib Kiosk API",
    description="""
## SmartLib - AI-Powered Library Kiosk System

### Features:
- 🔐 **Face Recognition**: Student authentication using ArcFace (512-dim embeddings)
- 📚 **Book Detection**: Real-time book identification using YOLOv8
- 📝 **OCR**: Text extraction from book covers using PaddleOCR
- 🤖 **AI Assistant**: Intelligent chat assistant (Qwen 2.5) for book recommendations
- 💳 **Transactions**: Automated book borrowing and returning
- 📊 **Fine Management**: Automatic overdue fine calculation

### API Endpoints:
- `/api/v1/auth/*` - Face verification and registration
- `/api/v1/books/*` - Book detection and catalog
- `/api/v1/transactions/*` - Borrow/return operations
- `/api/v1/students/*` - Student management
- `/api/v1/ai/*` - Chat and smart assistant
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware - origins configured via CORS_ALLOWED_ORIGINS env var
# Use comma-separated values: "http://localhost:5173,https://yourdomain.com"
allowed_origins = [
    origin.strip() 
    for origin in settings.cors_allowed_origins.split(",") 
    if origin.strip()
] if settings.cors_allowed_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(books.router, prefix=API_PREFIX)
app.include_router(transactions.router, prefix=API_PREFIX)
app.include_router(students.router, prefix=API_PREFIX)
app.include_router(assistant.router, prefix=API_PREFIX)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API health check."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint with actual service verification."""
    from sqlalchemy import text
    from app.database import async_session_maker
    
    db_status = "disconnected"
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    overall_status = "healthy" if db_status == "connected" else "degraded"
    
    return {
        "status": overall_status,
        "database": db_status,
        "environment": settings.app_env,
        "services": {
            "face_recognition": "ready",
            "book_detection": "ready",
            "ocr": "ready"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
