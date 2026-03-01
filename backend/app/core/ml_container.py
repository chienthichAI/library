"""
SmartLib Kiosk - AI Model Container
Centralized management for all AI/ML models to optimize VRAM and startup speed.
"""
from typing import Optional
from loguru import logger

from app.ml.face_detector import FaceDetector
from app.ml.face_recognition import FaceRecognizer
from app.ml.anti_spoofing import AntiSpoofing
from app.ml.book_detector import BookDetector
from app.ml.ocr_service import OCRService
from app.ml.faiss_engine import FaissEngine
from app.config import settings

class AIModels:
    """Container for singleton model instances."""
    face_detector: Optional[FaceDetector] = None
    face_recognizer: Optional[FaceRecognizer] = None
    anti_spoofing: Optional[AntiSpoofing] = None
    book_detector: Optional[BookDetector] = None
    ocr_service: Optional[OCRService] = None
    faiss_engine: Optional[FaissEngine] = None
    # Future: llm_service can be added here

async def init_ai_models():
    """Initialize all AI models and warm them up on the GPU."""
    logger.info("Initializing AI models on GPU...")
    
    # 1. Face Detector (InsightFace/ArcFace/RetinaFace)
    AIModels.face_detector = FaceDetector(
        model_name="buffalo_l",
        use_gpu=True
    )
    AIModels.face_detector.initialize()
    
    # 3. Anti-Spoofing (MiniFASNet)
    AIModels.anti_spoofing = AntiSpoofing(
        model_path=settings.antispoofing_model_path,
        threshold=settings.liveness_threshold,
        use_gpu=True
    )
    AIModels.anti_spoofing.initialize()

    # 4. Face Recognizer (ArcFace Standalone if needed)
    # Hitch to existing FaceAnalysis instance from face_detector to save VRAM (B05)
    AIModels.face_recognizer = FaceRecognizer(
        face_analysis_instance=getattr(AIModels.face_detector, '_model', None),
        use_gpu=True
    )
    AIModels.face_recognizer.initialize()
    
    # 5. Book Detector (YOLOv8)
    AIModels.book_detector = BookDetector(
        model_path=settings.yolo_model_path,
        use_gpu=True
    )
    AIModels.book_detector.initialize()
    
    # 6. OCR Service (PaddleOCR)
    AIModels.ocr_service = OCRService()
    AIModels.ocr_service.initialize()
    
    # 6. FAISS Search Engine
    AIModels.faiss_engine = FaissEngine()
    
    # Warm up models with dummy input (avoids first-request lag)
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    logger.info("Warming up AI models...")
    try:
        AIModels.face_detector.detect(dummy_img)
        # For AntiSpoofing and FaceRecognizer which need specific sizes
        dummy_112 = np.zeros((112, 112, 3), dtype=np.uint8)
        AIModels.face_recognizer.extract_embedding(dummy_112)
        AIModels.anti_spoofing.detect(dummy_112)
        AIModels.book_detector.detect(dummy_img)
        # Note: OCR might be slow to warm up, but it's okay
        logger.info("AI models warmed up successfully")
    except Exception as e:
        logger.warning(f"Warm up failed: {e}")

    return AIModels
