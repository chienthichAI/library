"""
SmartLib Kiosk - Authentication API Routes
Face verification and registration endpoints with quality feedback
"""
import numpy as np
import cv2
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.database import get_db
from app.schemas.auth import FaceVerifyResponse, FaceRegisterResponse
from app.services.authentication_service import AuthenticationService
from functools import lru_cache
from app.core.face_session import create_face_session_token

router = APIRouter(prefix="/auth", tags=["Authentication"])
FACE_SESSION_TTL_SECONDS = 300


@lru_cache()
def get_auth_service() -> AuthenticationService:
    """Get or create authentication service instance (cached singleton)."""
    return AuthenticationService()


@router.post("/verify-face", response_model=FaceVerifyResponse)
async def verify_face(
    image: Optional[UploadFile] = File(None, description="Single Face image (Legacy)"),
    images: List[UploadFile] = File(None, description="Multiple face images for rPPG"),
    ground_truth_student_id: Optional[str] = Form(None, description="Test mode label for evaluation"),
    source: Optional[str] = Form("api", description="Source tag for logging"),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """
    Verify student identity using face recognition.
    """
    try:
        if images and len(images) > 0:
            target_file = images[-1]  # using the last frame as the main one for auth
        elif image:
            target_file = image
            images = [image] # default to single frame list
        else:
            raise HTTPException(status_code=400, detail="No image provided")
            
        # Validate image size (P4-02: prevent OOM on Jetson 8GB)
        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
        contents = await target_file.read()
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
        # Decode all frames concurrently to improve processing speed (20-40ms target)
        async def _decode_frame(f):
            await f.seek(0)
            c = await f.read()
            arr = np.frombuffer(c, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        decoded_frames = await asyncio.gather(*[_decode_frame(f) for f in images])
        frames = [f for f in decoded_frames if f is not None]

        
        # Authenticate with quality check (P3-04: 15s timeout for ML inference)
        result = await asyncio.wait_for(
            auth_service.authenticate(
                img,
                db,
                check_quality=True,
                frames=frames,
                ground_truth_student_id=(ground_truth_student_id or "").strip() or None,
                source=(source or "api").strip() or "api",
            ),
            timeout=15.0
        )
        
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
            quality_issues=result.quality_issues,
            verification_token=(
                create_face_session_token(
                    student_id=result.student_id,
                    role=result.role or "STUDENT",
                    ttl_seconds=FACE_SESSION_TTL_SECONDS
                ) if result.success and result.student_id else None
            ),
            token_expires_in_seconds=FACE_SESSION_TTL_SECONDS if result.success and result.student_id else None
        )
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Face verification timed out (>15s)")
        raise HTTPException(status_code=504, detail="Face verification timed out. Please try again.")
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xác thực khuôn mặt")


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
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi đăng ký khuôn mặt")


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
        
        # Detect face first (skip heavy embedding extraction for just quality check)
        faces = auth_service.face_detector.detect(img, max_faces=5, extract_embedding=False)
        face_bbox = None
        landmarks = None
        face = None
        num_faces = len(faces)
        
        if faces:
            # Use _select_best_face for consistency with authenticate()
            face = auth_service._select_best_face(img, faces)
            if face:
                face_bbox = (face.x1, face.y1, face.x2, face.y2)
                landmarks = getattr(face, 'landmarks', None)
        
        # Check quality
        result = auth_service.quality_checker.check(img, face_bbox, num_faces, landmarks=landmarks)
        
        response_faces = []
        if faces:
            for f in faces:
                is_primary = (f == face) if face else False
                response_faces.append({
                    "x1": int(f.x1), "y1": int(f.y1),
                    "x2": int(f.x2), "y2": int(f.y2),
                    "confidence": float(f.confidence),
                    "is_primary": is_primary
                })

        return {
            "is_valid": bool(result.is_valid),
            "overall_score": float(result.overall_score),
            "brightness_score": float(result.brightness_score),
            "sharpness_score": float(result.sharpness_score),
            "face_size_score": float(result.face_size_score),
            "centering_score": float(result.centering_score),
            "issues": [issue.value for issue in result.issues],
            "message": result.vietnamese_message,
            "recommendations": result.recommendations,
            "faces": response_faces
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi kiểm tra chất lượng ảnh")

@router.websocket("/ws/stream")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """
    WebSocket endpoint for real-time face authentication with continuous streaming.
    Receives JPEG binary frames and returns JSON authentication/tracking updates.
    """
    await websocket.accept()
    ground_truth_student_id = (websocket.query_params.get("ground_truth_student_id") or "").strip() or None
    source = "ws_test" if ground_truth_student_id else "ws"
    logger.info("WebSocket client connected for real-time authentication")
    
    try:
        while True:
            # Receive image frame
            data = await websocket.receive_bytes()
            
            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                await websocket.send_json({"error": "Invalid frame data", "status": "error"})
                continue
                
            # Use tracker to detect and track faces
            faces = auth_service.face_detector.track(img, max_faces=5, extract_embedding=False)
            
            if not faces:
                await websocket.send_json({"status": "no_face_detected"})
                continue
                
            # Process the best face
            best_face = auth_service._select_best_face(img, faces)
            
            if not best_face:
                await websocket.send_json({"status": "no_prominent_face"})
                continue
                
            # Use the track ID for caching and identity persistence
            track_id = str(best_face.track_id) if best_face.track_id is not None else None
            
            # Keep websocket path simple: pass track_id for cache hits and
            # let authentication service run full pipeline when cache misses.
            result = await auth_service.authenticate(
                image=img, 
                db=db, 
                check_quality=True, 
                track_id=track_id,
                ground_truth_student_id=ground_truth_student_id,
                source=source,
            )
            
            # Send result back
            response_data = {
                "status": "success" if result.success else "failed",
                "track_id": track_id,
                "student_id": result.student_id,
                "student_name": result.student_name,
                "role": result.role,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "error_message": result.error_message,
                "quality_score": result.quality_score,
                "quality_issues": result.quality_issues,
                "verification_token": (
                    create_face_session_token(
                        student_id=result.student_id,
                        role=result.role or "STUDENT",
                        ttl_seconds=FACE_SESSION_TTL_SECONDS
                    ) if result.success and result.student_id else None
                ),
                "token_expires_in_seconds": FACE_SESSION_TTL_SECONDS if result.success and result.student_id else None
            }
            
            # Gửi kết quả về client
            await websocket.send_json(response_data)
            
            # Nếu xác thực thành công → chờ client nhận message rồi mới đóng
            if result.success:
                logger.info(f"Xác thực thành công cho {result.student_name}. Đóng WebSocket.")
                await asyncio.sleep(0.5)  # Cho client 500ms để xử lý message success
                await websocket.close(code=1000)
                break
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
