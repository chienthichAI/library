"""
SmartLib Kiosk - Authentication Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class FaceVerifyRequest(BaseModel):
    """Schema for face verification request (metadata only, image sent as file)."""
    timestamp: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-01-21T15:30:00Z"
            }
        }


class FaceVerifyResponse(BaseModel):
    """Schema for face verification response."""
    success: bool
    student_id: Optional[str]
    student_name: Optional[str]
    role: Optional[str] = "STUDENT"
    confidence: float
    liveness_score: float
    is_real_face: bool
    error_message: Optional[str]
    processing_time_ms: float
    quality_score: float = 1.0
    quality_issues: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "student_id": "FPT20240001",
                "student_name": "Nguyễn Văn A",
                "confidence": 0.9952,
                "liveness_score": 0.98,
                "is_real_face": True,
                "error_message": None,
                "processing_time_ms": 142.5,
                "quality_score": 0.95,
                "quality_issues": None
            }
        }


class FaceRegisterRequest(BaseModel):
    """Schema for face registration request."""
    student_id: str = Field(..., description="Student ID to register face for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "FPT20240001"
            }
        }


class FaceRegisterResponse(BaseModel):
    """Schema for face registration response."""
    success: bool
    message: str
    quality_score: float = 1.0
    total_embeddings: int = 0
    max_embeddings: int = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Đăng ký thành công! (1/5 ảnh)",
                "quality_score": 0.92,
                "total_embeddings": 1,
                "max_embeddings": 5
            }
        }
