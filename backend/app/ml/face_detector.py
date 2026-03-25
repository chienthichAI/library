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
    track_id: Optional[int] = None
    pose: Optional[Dict[str, float]] = None
    
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
        det_size: Tuple[int, int] = (320, 320),
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
        self.tracker = None
        
    def initialize(self) -> bool:
        """
        Load and initialize the face detection model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace dependency is missing. Face detector cannot run without real model.")
            self._initialized = False
            return False
            
        try:
            from pathlib import Path
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(Path.home() / '.insightface' / 'models' / self.model_name),
                    'trt_fp16_enable': True,
                    'trt_max_workspace_size': 2147483648,
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ] if self.use_gpu else ['CPUExecutionProvider']
            
            self._model = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            self._model.prepare(
                ctx_id=0 if self.use_gpu else -1,
                det_size=self.det_size,
                det_thresh=self.det_thresh
            )
            
            try:
                from norfair import Tracker
                self.tracker = Tracker(
                    distance_function="euclidean",
                    distance_threshold=100,
                    initialization_delay=2,
                    hit_counter_max=10
                )
            except (ImportError, TypeError) as tracker_err:
                logger.warning(f"Norfair tracker not available or incompatible: {tracker_err}. Face tracking disabled.")
                self.tracker = None
            
            self._initialized = True
            logger.info(f"Face detector initialized: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return False
    
    def detect(self, image: np.ndarray, max_faces: int = 1, extract_embedding: bool = True) -> List[DetectedFace]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            max_faces: Maximum number of faces to return
            extract_embedding: Whether to extract deep face features (ArcFace). False speeds up ~10x.
            
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
            logger.error("Face detector model is not initialized. Detection aborted.")
            return []
            
        try:
            import time
            t0 = time.time()
            # Run detection ONLY or full pipeline
            if extract_embedding:
                faces = self._model.get(image_rgb)
                logger.debug(f"InsightFace full pipeline took {(time.time() - t0)*1000:.2f}ms")
            else:
                det_model = self._model.models.get('detection')
                if det_model is None:
                    faces = self._model.get(image_rgb)
                else:
                    bboxes, kpss = det_model.detect(image_rgb)
                    logger.debug(f"InsightFace detection-only took {(time.time() - t0)*1000:.2f}ms")
                    faces = []
                    # bboxes shape is (N, 5), kpss is (N, 5, 2)
                    if bboxes is not None and bboxes.shape[0] > 0:
                        for i in range(bboxes.shape[0]):
                            class MockInsightFace:
                                bbox = bboxes[i, 0:4]
                                det_score = bboxes[i, 4]
                                kps = kpss[i] if kpss is not None else None
                                embedding = None
                                normed_embedding = None
                            faces.append(MockInsightFace())
            
            if not faces:
                return []
                
            # Sort by confidence and get a generous margin so _select_best_face can pick based on size/center
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
            if max_faces > 0:
                faces = faces[:max_faces * 2]
            
            detected_faces = []
            for face in faces:
                kps = getattr(face, 'kps', None)
                
                normed_emb = getattr(face, 'normed_embedding', None)
                emb = getattr(face, 'embedding', None)
                
                pose = FaceDetector._estimate_pose(kps) if kps is not None else None
                
                detected = DetectedFace(
                    bbox=tuple(face.bbox.astype(int)),
                    confidence=float(face.det_score),
                    landmarks=kps,
                    aligned_face=self._align_face(image_rgb, face) if kps is not None else None,
                    embedding=normed_emb if normed_emb is not None else emb,
                    pose=pose
                )
                detected_faces.append(detected)
                
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    @staticmethod
    def _estimate_pose(landmarks: np.ndarray) -> Dict[str, float]:
        """Estimate yaw, pitch roughly from 5 landmarks."""
        if landmarks is None or len(landmarks) < 5:
            return {"yaw": 0.0, "pitch": 0.0}
        le, re, nose, lm, rm = landmarks
        ec = (le + re) / 2.0
        mc = (lm + rm) / 2.0
        face_width = max(np.linalg.norm(le - re), 1e-5)
        nose_dist_from_center_x = nose[0] - ((ec[0] + mc[0]) / 2.0)
        yaw = (nose_dist_from_center_x / face_width) * 90.0
        
        face_height = max(np.linalg.norm(ec - mc), 1e-5)
        nose_dist_from_center_y = nose[1] - ((ec[1] + mc[1]) / 2.0)
        pitch = (nose_dist_from_center_y / face_height) * 90.0
        return {"yaw": yaw, "pitch": pitch}
    
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
            kps = getattr(face, 'kps', None)
            
            if kps is not None and image is not None:
                # Ensure image is a valid numpy array with shape
                if not hasattr(image, 'shape') or len(image.shape) < 2:
                    logger.warning("Invalid image object passed to _align_face")
                    return None
                    
                import numpy as np
                if not isinstance(kps, np.ndarray):
                    kps = np.array(kps)
                    
                if kps.shape != (5, 2):
                    logger.warning(f"Unexpected landmark shape: {kps.shape}. Needs (5, 2).")
                    return None
                
                aligned = face_align.norm_crop(image, kps, image_size=target_size[0])
                return aligned
            return None
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return None

    def track(self, image: np.ndarray, max_faces: int = 1, extract_embedding: bool = True) -> List[DetectedFace]:
        """Detect and track faces consistently across frames with smoothing."""
        detections = self.detect(image, max_faces=10, extract_embedding=extract_embedding)
        
        if self.tracker is None:
            if max_faces > 0:
                detections = sorted(detections, key=lambda x: x.confidence, reverse=True)[:max_faces]
            return detections
            
        from norfair import Detection
        norfair_detections = []
        for det in detections:
            # Quality filter before tracking
            if det.pose is not None:
                if abs(det.pose['yaw']) > 45.0 or abs(det.pose['pitch']) > 45.0:
                    continue # Skip highly angled faces
                    
            points = np.array([[det.x1, det.y1], [det.x2, det.y2]])
            norfair_detections.append(Detection(points=points, data=det))
            
        tracked_objects = self.tracker.update(detections=norfair_detections)
        
        tracked_faces = []
        for obj in tracked_objects:
            if obj.last_detection is not None:
                face = obj.last_detection.data
                # Update bbox with Kalman-smoothed estimate
                pts = obj.estimate
                face.bbox = (int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(pts[1][1]))
                face.track_id = int(obj.id)
                tracked_faces.append(face)
                
        if max_faces > 0:
            tracked_faces = sorted(tracked_faces, key=lambda x: x.confidence, reverse=True)[:max_faces]
            
        return tracked_faces
    
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
            
            # Draw confidence and tracking ID
            label = f"ID:{face.track_id}" if face.track_id is not None else ""
            label += f" {face.confidence:.2f}"
            if face.pose is not None:
                label += f" Y:{face.pose['yaw']:.0f} P:{face.pose['pitch']:.0f}"
                
            cv2.putText(
                result,
                label.strip(),
                (face.x1, face.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness
            )
            
            # Draw landmarks if available
            if face.landmarks is not None and getattr(face.landmarks, "ndim", 0) == 2:
                for point in face.landmarks:
                    cv2.circle(result, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    
        return result
