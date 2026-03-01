"""
SmartLib Kiosk - Configuration Settings
Supports Supabase PostgreSQL
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "SmartLib Kiosk"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = Field(
        default="change-this-in-production",
        description="⚠️ MUST be set to a strong random value in production"
    )
    
    # Supabase Database - MUST be provided via .env file
    # Format: postgresql+asyncpg://user:password@host:5432/database
    database_url: str = ""  # Required - provide via DATABASE_URL in .env
    supabase_url: str = ""  # Required - provide via SUPABASE_URL in .env
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    
    # CORS Settings - Environment-based for security
    # Development: "*" allowed
    # Production: Set to specific origins like "http://localhost:5173,https://yourdomain.com"
    cors_allowed_origins: str = Field(
        default="*",
        description="⚠️ Set specific origins in production for security"
    )
    
    # AI Models Paths
    face_model_path: str = "models/face_recognition/arcface_r100.onnx"
    antispoofing_model_path: str = "models/minifasnet.onnx"
    yolo_model_path: str = "models/book_detection/yolov8m_books.pt"
    ocr_lang: str = "vi"
    
    # Face Recognition Thresholds
    face_similarity_threshold: float = 0.45
    liveness_threshold: float = 0.60
    
    # Continuous Learning
    continuous_learning_threshold: float = 0.85
    continuous_learning_enabled: bool = True
    
    # Book Detection
    book_detection_confidence: float = 0.5
    
    # Transaction Settings
    max_borrow_days: int = 14
    fine_per_day: int = 10000  # VND
    max_books_per_student: int = 5
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    disable_model_source_check: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
