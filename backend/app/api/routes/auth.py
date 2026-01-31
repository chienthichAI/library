"""
SmartLib Kiosk - Authentication API Routes
Face verification and registration endpoints with quality feedback
"""
import numpy as np
import cv2
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.database import get_db
from app.schemas.auth import FaceVerifyResponse, FaceRegisterResponse
from app.services.authentication_service import AuthenticationService

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Global service instance
_auth_service: AuthenticationService = None


def get_auth_service() -> AuthenticationService:
    """Get or create authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service


@router.post("/verify-face", response_model=FaceVerifyResponse)
async def verify_face(
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """
    Verify student identity using face recognition.
    
    **Enhanced Pipeline:**
    1. Check image quality (brightness, sharpness, face position)
    2. Detect face using RetinaFace
    3. Check liveness (anti-spoofing)
    4. Extract 512-dim embedding using ArcFace
    5. Match against ALL student embeddings (supports multiple per student)
    
    **Quality Feedback:**
    - Returns quality_score (0-1) indicating image quality
    - Returns quality_issues list if problems detected
    - Vietnamese error messages for user guidance
    
    **Returns:**
    - student_id and name if verified
    - confidence score (0-1)
    - liveness score
    - quality feedback
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Authenticate with quality check
        result = await auth_service.authenticate(img, db, check_quality=True)
        
        return FaceVerifyResponse(
            success=result.success,
            student_id=result.student_id,
            student_name=result.student_name,
            role=result.role,
            confidence=result.confidence,
            liveness_score=result.liveness_score,
            is_real_face=result.is_real_face,
            error_message=result.error_message,
            processing_time_ms=result.processing_time_ms,
            quality_score=result.quality_score,
            quality_issues=result.quality_issues
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-face", response_model=FaceRegisterResponse)
async def register_face(
    student_id: str = Form(..., description="Student ID"),
    image: UploadFile = File(..., description="Face image for registration"),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """
    Register a student's face for future authentication.
    
    **Multi-Embedding Support:**
    - Each student can register up to 5 face images
    - Different angles improve recognition accuracy
    - Duplicate detection prevents same angle re-registration
    
    **Quality Validation:**
    - Checks brightness, sharpness, face size
    - Rejects poor quality images with guidance
    - Performs liveness check
    
    **Returns:**
    - success status and message
    - quality_score of submitted image
    - total_embeddings count for this student
    - max_embeddings limit (5)
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Register face with quality check
        result = await auth_service.register_student_face(
            student_id, img, db, check_quality=True
        )
        
        return FaceRegisterResponse(
            success=result.success,
            message=result.message,
            quality_score=result.quality_score,
            total_embeddings=result.total_embeddings,
            max_embeddings=auth_service.MAX_EMBEDDINGS_PER_STUDENT
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-quality")
async def check_image_quality(
    image: UploadFile = File(..., description="Face image to check"),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """
    Check image quality without registration.
    
    Useful for real-time feedback during capture.
    
    **Returns:**
    - is_valid: Whether image meets quality requirements
    - overall_score: Combined quality score (0-1)
    - Individual scores for brightness, sharpness, centering
    - Vietnamese guidance message
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect face first
        faces = auth_service.face_detector.detect(img, max_faces=5)
        face_bbox = None
        num_faces = len(faces)
        
        if faces:
            # Use _select_best_face for consistency with authenticate()
            face = auth_service._select_best_face(img, faces)
            if face:
                face_bbox = (face.x1, face.y1, face.x2, face.y2)
        
        # Check quality
        result = auth_service.quality_checker.check(img, face_bbox, num_faces)
        
        return {
            "is_valid": result.is_valid,
            "overall_score": result.overall_score,
            "brightness_score": result.brightness_score,
            "sharpness_score": result.sharpness_score,
            "face_size_score": result.face_size_score,
            "centering_score": result.centering_score,
            "issues": [issue.value for issue in result.issues],
            "message": result.vietnamese_message,
            "recommendations": result.recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
