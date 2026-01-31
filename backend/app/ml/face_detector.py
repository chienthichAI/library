"""
SmartLib Kiosk - Face Detection using Deep Neural Networks

Uses RetinaFace for face detection and landmark localization.
RetinaFace is a state-of-the-art face detection model that provides:
- Face bounding boxes
- 5 facial landmarks (eyes, nose, mouth corners)
- High accuracy even in challenging conditions
"""
import os
# Disable InsightFace model source check to speed up startup
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available. Using mock face detector.")


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box and landmarks."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    aligned_face: Optional[np.ndarray] = None  # 112x112 aligned face
    embedding: Optional[np.ndarray] = None  # 512-dim embedding
    
    @property
    def x1(self) -> int:
        return int(self.bbox[0])
    
    @property
    def y1(self) -> int:
        return int(self.bbox[1])
    
    @property
    def x2(self) -> int:
        return int(self.bbox[2])
    
    @property
    def y2(self) -> int:
        return int(self.bbox[3])
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class FaceDetector:
    """
    Face Detection using RetinaFace (Deep Neural Network).
    
    Architecture:
    - Backbone: ResNet50 / MobileNet
    - Feature Pyramid Network (FPN) for multi-scale detection
    - Single-shot detector with facial landmarks
    
    Performance:
    - WiderFace Hard: 91.4% AP
    - Speed: ~30ms per frame on GPU
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize face detector.
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, antelopev2)
            det_size: Detection input size (width, height)
            det_thresh: Detection confidence threshold
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Load and initialize the face detection model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available. Using mock mode.")
            self._initialized = True
            return True
            
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            
            self._model = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            self._model.prepare(
                ctx_id=0 if self.use_gpu else -1,
                det_size=self.det_size,
                det_thresh=self.det_thresh
            )
            
            self._initialized = True
            logger.info(f"Face detector initialized: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, max_faces: int = 1) -> List[DetectedFace]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            max_faces: Maximum number of faces to return
            
        Returns:
            List of DetectedFace objects
        """
        if not self._initialized:
            self.initialize()
            
        if image is None or image.size == 0:
            logger.warning("Empty image provided to face detector")
            return []
            
        # Convert to RGB if needed (InsightFace expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        if not INSIGHTFACE_AVAILABLE or self._model is None:
            # Mock detection for testing
            return self._mock_detect(image)
            
        try:
            # Run detection
            faces = self._model.get(image_rgb)
            
            if not faces:
                return []
                
            # Sort by confidence and limit
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
            
            detected_faces = []
            for face in faces:
                detected = DetectedFace(
                    bbox=tuple(face.bbox.astype(int)),
                    confidence=float(face.det_score),
                    landmarks=face.landmark if hasattr(face, 'landmark') else None,
                    aligned_face=self._align_face(image_rgb, face) if hasattr(face, 'landmark') else None,
                    embedding=face.normed_embedding if hasattr(face, 'normed_embedding') else (face.embedding if hasattr(face, 'embedding') else None)
                )
                detected_faces.append(detected)
                
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _align_face(
        self, 
        image: np.ndarray, 
        face: Any,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """
        Align face using landmarks for face recognition.
        
        Uses affine transformation to align face to standard pose.
        Required for ArcFace embedding extraction.
        
        Args:
            image: RGB image
            face: InsightFace face object with landmarks
            target_size: Output size (112x112 for ArcFace)
            
        Returns:
            Aligned face image or None
        """
        try:
            from insightface.utils import face_align
            
            # Use 5-point landmarks for alignment (ArcFace standard)
            kps = None
            if hasattr(face, 'kps'):
                kps = face.kps
            elif hasattr(face, 'landmark'):
                kps = face.landmark
                
            if kps is not None and image is not None:
                # Ensure image is a valid numpy array with shape
                if not hasattr(image, 'shape') or len(image.shape) < 2:
                    logger.warning("Invalid image object passed to _align_face")
                    return None
                    
                # Ensure landmarks are in correct format (5, 2)
                if len(kps) > 5:
                    kps = kps[:5]
                
                aligned = face_align.norm_crop(image, kps, image_size=target_size[0])
                return aligned
            return None
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return None
    
    def _mock_detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Mock face detection for testing without InsightFace.
        Uses OpenCV Haar Cascade as fallback.
        """
        try:
            # Use OpenCV Haar Cascade as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            detected_faces = []
            for (x, y, w, h) in faces:
                detected = DetectedFace(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.85,
                    landmarks=None,
                    aligned_face=cv2.resize(image[y:y+h, x:x+w], (112, 112))
                )
                detected_faces.append(detected)
                
            return detected_faces
            
        except Exception as e:
            logger.error(f"Mock face detection failed: {e}")
            return []
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected faces on image for visualization.
        
        Args:
            image: Original BGR image
            faces: List of detected faces
            color: Bounding box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for face in faces:
            # Draw bounding box
            cv2.rectangle(
                result,
                (face.x1, face.y1),
                (face.x2, face.y2),
                color,
                thickness
            )
            
            # Draw confidence
            label = f"{face.confidence:.2f}"
            cv2.putText(
                result,
                label,
                (face.x1, face.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness
            )
            
            # Draw landmarks if available
            if face.landmarks is not None:
                for point in face.landmarks:
                    cv2.circle(result, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    
        return result
