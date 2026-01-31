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
from app.api.routes import auth, books, transactions, students


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
    """Application lifespan events."""
    # Startup
    logger.info("Starting SmartLib Kiosk API...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Production security warnings
    if settings.app_env == "production":
        if settings.secret_key == "change-this-in-production":
            logger.warning("⚠️ SECRET_KEY is using default value! Set a strong random key in production.")
        if settings.cors_allowed_origins == "*":
            logger.warning("⚠️ CORS is open to all origins! Set specific origins in production.")
        if settings.debug:
            logger.warning("⚠️ Debug mode is ON in production! Set DEBUG=false.")
    
    # Validate model files exist
    import os
    model_checks = {
        "Face Recognition (ArcFace)": settings.face_model_path,
        "Anti-Spoofing (MiniFASNet)": settings.antispoofing_model_path,
        "Book Detection (YOLOv8)": settings.yolo_model_path,
    }
    
    for model_name, model_path in model_checks.items():
        if os.path.exists(model_path):
            logger.info(f"✓ {model_name} model: {model_path}")
        else:
            logger.warning(f"⚠️ {model_name} model not found: {model_path} - Using fallback/mock mode")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
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
- 💳 **Transactions**: Automated book borrowing and returning
- 📊 **Fine Management**: Automatic overdue fine calculation

### API Endpoints:
- `/api/v1/auth/*` - Face verification and registration
- `/api/v1/books/*` - Book detection and catalog
- `/api/v1/transactions/*` - Borrow/return operations
- `/api/v1/students/*` - Student management

### Technology Stack:
- **Backend**: FastAPI + SQLAlchemy (async)
- **Database**: Supabase PostgreSQL
- **AI/ML**: InsightFace (ArcFace), Ultralytics (YOLOv8), PaddleOCR
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
