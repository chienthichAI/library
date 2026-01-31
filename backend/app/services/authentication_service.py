"""
SmartLib Kiosk - Authentication Service

Handles student authentication via face recognition.
Uses ArcFace + AntiSpoofing for secure verification.
Enhanced with image quality check and multiple embeddings support.
"""
import numpy as np
import hashlib
from typing import Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from loguru import logger

from app.models.student import Student, StudentStatus
from app.models.face_embedding import FaceEmbedding
from app.ml.face_detector import FaceDetector, DetectedFace
from app.ml.face_recognition import FaceRecognizer, FaceEmbeddingResult
from app.ml.anti_spoofing import AntiSpoofing, AntiSpoofingResult
from app.ml.quality_checker import ImageQualityChecker, QualityCheckResult
from app.config import settings


@dataclass
class AuthenticationResult:
    """Result of student authentication."""
    success: bool
    student_id: Optional[str]
    student_name: Optional[str]
    confidence: float
    liveness_score: float
    is_real_face: bool
    error_message: Optional[str]
    processing_time_ms: float
    role: Optional[str] = "STUDENT"
    quality_score: float = 1.0
    quality_issues: List[str] = None
    
    @property
    def is_authenticated(self) -> bool:
        return self.success and self.is_real_face


@dataclass
class FaceRegistrationResult:
    """Result of face registration."""
    success: bool
    message: str
    embedding_id: Optional[int]
    quality_score: float
    total_embeddings: int


class AuthenticationService:
    """
    Student Authentication Service using Face Recognition.
    
    Enhanced Pipeline:
    1. Check image quality (brightness, sharpness, face size)
    2. Detect face in image (RetinaFace)
    3. Check liveness (MiniFASNet anti-spoofing)
    4. Extract face embedding (ArcFace)
    5. Match against ALL student embeddings (supports multiple per student)
    6. Return authentication result with quality feedback
    """
    
    # Maximum embeddings per student for multi-angle registration
    MAX_EMBEDDINGS_PER_STUDENT = 5
    
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        anti_spoofing: Optional[AntiSpoofing] = None,
        quality_checker: Optional[ImageQualityChecker] = None
    ):
        """Initialize authentication service with all components."""
        self.similarity_threshold = settings.face_similarity_threshold
        self.liveness_threshold = settings.liveness_threshold
        
        # Initialize components with settings if not provided
        self.face_detector = face_detector or FaceDetector()
        self.face_recognizer = face_recognizer or FaceRecognizer(
            model_path=settings.face_model_path
        )
        self.anti_spoofing = anti_spoofing or AntiSpoofing(
            model_path=settings.antispoofing_model_path,
            threshold=self.liveness_threshold
        )
        self.quality_checker = quality_checker or ImageQualityChecker()
        
        # Ensure threshold is updated if component was passed in
        if self.anti_spoofing:
            self.anti_spoofing.threshold = self.liveness_threshold
        
    async def initialize(self) -> bool:
        """Initialize all ML models."""
        try:
            self.face_detector.initialize()
            self.face_recognizer.initialize()
            self.anti_spoofing.initialize()
            logger.info("Authentication service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize authentication service: {e}")
            return False
    
    async def authenticate(
        self,
        image: np.ndarray,
        db: AsyncSession,
        check_quality: bool = True
    ) -> AuthenticationResult:
        """
        Authenticate a student using face recognition.
        
        Args:
            image: BGR image from camera
            db: Database session
            check_quality: Whether to perform quality checks
            
        Returns:
            AuthenticationResult with success status and student info
        """
        import time
        start_time = time.time()
        quality_issues = []
        quality_score = 1.0
        
        try:
            # Step 1: Detect face FIRST (needed for quality check)
            faces = self.face_detector.detect(image, max_faces=5)
            num_faces = len(faces)
            
            if not faces:
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=0.0, is_real_face=False,
                    error_message="Không phát hiện khuôn mặt. Đưa mặt vào khung hình.",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=0.0
                )
            
            face = self._select_best_face(image, faces)
            if face is None:
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=0.0, is_real_face=False,
                    error_message="Không xác định được khuôn mặt chính. Đưa mặt vào giữa khung hình.",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=0.0
                )
                
            face_bbox = (face.x1, face.y1, face.x2, face.y2)
            
            # Step 2: Check image quality
            quality_score = 1.0
            if check_quality:
                quality_result = self.quality_checker.check(image, face_bbox, num_faces)
                quality_score = quality_result.overall_score
                quality_issues = [issue.value for issue in quality_result.issues]
                
                if not quality_result.is_valid:
                    return AuthenticationResult(
                        success=False, student_id=None, student_name=None, role=None,
                        confidence=0.0, liveness_score=0.0, is_real_face=False,
                        error_message=quality_result.vietnamese_message,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        quality_score=quality_score,
                        quality_issues=quality_issues
                    )
            
            # Step 3: Check liveness (anti-spoofing)
            face_crop = self._crop_face(image, face)
            liveness_result = self.anti_spoofing.detect(face_crop)
            liveness_score = liveness_result.liveness_score
            is_real = liveness_result.is_real
            
            if not is_real:
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=liveness_score,
                    is_real_face=False,
                    error_message=f"Phát hiện giả mạo: {liveness_result.spoof_type}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=quality_score
                )
                
            # Step 4: Extract embedding (Use pre-computed from detector if available)
            if face.embedding is not None:
                query_embedding = face.embedding
            else:
                aligned_face = face.aligned_face if face.aligned_face is not None else self._align_face_simple(image, face)
                embedding_result = self.face_recognizer.extract_embedding(aligned_face)
                if not embedding_result.is_valid:
                    return AuthenticationResult(
                        success=False, student_id=None, student_name=None, role=None,
                        confidence=0.0, liveness_score=liveness_score,
                        is_real_face=True,
                        error_message="Không thể trích xuất đặc trưng khuôn mặt",
                        processing_time_ms=(time.time() - start_time) * 1000,
                        quality_score=quality_score
                    )
                query_embedding = embedding_result.embedding
                
            # Step 5: Match against database (ALL embeddings)
            match_result = await self._find_matching_student(
                query_embedding, db
            )
            
            if match_result is None:
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=liveness_score,
                    is_real_face=True,
                    error_message="Không tìm thấy sinh viên. Vui lòng đăng ký trước.",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=quality_score
                )
                
            student, similarity = match_result
            
            # Update last login
            student.last_login = datetime.utcnow()
            await db.commit()
            
            return AuthenticationResult(
                success=True,
                student_id=student.student_id,
                student_name=student.full_name,
                role=student.role.value if hasattr(student.role, 'value') else student.role,
                confidence=similarity,
                liveness_score=liveness_score,
                is_real_face=is_real,
                error_message=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthenticationResult(
                success=False, student_id=None, student_name=None, role=None,
                confidence=0.0, liveness_score=0.0, is_real_face=False,
                error_message=f"Lỗi hệ thống: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=0.0
            )
    
    async def _find_matching_student(
        self,
        query_embedding: np.ndarray,
        db: AsyncSession
    ) -> Optional[Tuple[Student, float]]:
        """
        Find matching student by comparing against ALL embeddings.
        
        This handles multiple embeddings per student (different angles).
        The best match across all embeddings is returned.
        """
        # Use pgvector for optimized similarity search (Cosine Distance)
        # Note: pgvector operator <=> computes cosine distance (1 - cosine_similarity)
        # We want similarity >= threshold, which means distance <= (1 - threshold)
        
        distance_threshold = 1.0 - self.similarity_threshold
        
        stmt = (
            select(Student, FaceEmbedding, FaceEmbedding.embedding.cosine_distance(query_embedding).label("distance"))
            .join(FaceEmbedding, Student.student_id == FaceEmbedding.student_id)
            .where(
                and_(
                    Student.status == StudentStatus.ACTIVE.value,
                    FaceEmbedding.embedding.cosine_distance(query_embedding) <= distance_threshold
                )
            )
            .order_by(FaceEmbedding.embedding.cosine_distance(query_embedding))
            .limit(1)
        )
        
        result = await db.execute(stmt)
        row = result.first()
        
        if not row:
            return None
            
        student, face_emb, distance = row
        similarity = 1.0 - distance
        
        return student, similarity
    
    def _crop_face(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Crop face region from image."""
        return image[face.y1:face.y2, face.x1:face.x2]
    
    def _align_face_simple(
        self,
        image: np.ndarray,
        face: DetectedFace,
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """Simple face alignment without landmarks."""
        import cv2
        cropped = self._crop_face(image, face)
        return cv2.resize(cropped, target_size)
    
    async def register_student_face(
        self,
        student_id: str,
        face_image: np.ndarray,
        db: AsyncSession,
        check_quality: bool = True
    ) -> FaceRegistrationResult:
        """
        Register a student's face with quality validation.
        
        Supports multiple embeddings per student (up to MAX_EMBEDDINGS_PER_STUDENT).
        Each registration adds a new embedding for better recognition.
        
        Args:
            student_id: Student ID
            face_image: Face image
            db: Database session
            check_quality: Whether to validate image quality
            
        Returns:
            FaceRegistrationResult
        """
        try:
            # Check if student exists
            stmt = select(Student).where(Student.student_id == student_id)
            result = await db.execute(stmt)
            student = result.scalar_one_or_none()
            
            if not student:
                return FaceRegistrationResult(
                    success=False, message="Sinh viên không tồn tại",
                    embedding_id=None, quality_score=0.0, total_embeddings=0
                )
            
            # Count existing embeddings
            count_stmt = select(func.count()).where(FaceEmbedding.student_id == student_id)
            count_result = await db.execute(count_stmt)
            existing_count = count_result.scalar() or 0
            
            if existing_count >= self.MAX_EMBEDDINGS_PER_STUDENT:
                return FaceRegistrationResult(
                    success=False,
                    message=f"Đã đạt giới hạn {self.MAX_EMBEDDINGS_PER_STUDENT} ảnh đăng ký",
                    embedding_id=None, quality_score=1.0, total_embeddings=existing_count
                )
                
            # Detect face
            faces = self.face_detector.detect(face_image, max_faces=1)
            
            face = self._select_best_face(face_image, faces)
            
            if face is None:
                return FaceRegistrationResult(
                    success=False, message="Không xác định được khuôn mặt chính",
                    embedding_id=None, quality_score=0.0, total_embeddings=existing_count
                )
                
            face_bbox = (face.x1, face.y1, face.x2, face.y2)
            
            # Quality check
            quality_score = 1.0
            if check_quality:
                quality_result = self.quality_checker.check(face_image, face_bbox, 1)
                quality_score = quality_result.overall_score
                
                if not quality_result.is_valid:
                    return FaceRegistrationResult(
                        success=False,
                        message=quality_result.vietnamese_message,
                        embedding_id=None, quality_score=quality_score,
                        total_embeddings=existing_count
                    )
            
            # Check liveness during registration
            face_crop = self._crop_face(face_image, face)
            liveness = self.anti_spoofing.detect(face_crop)
            if not liveness.is_real:
                return FaceRegistrationResult(
                    success=False, message="Ảnh không hợp lệ (phát hiện giả mạo)",
                    embedding_id=None, quality_score=quality_score,
                    total_embeddings=existing_count
                )
                
            # Extract embedding (Use pre-computed from detector if available)
            if face.embedding is not None:
                embedding_val = face.embedding
                is_valid_emb = True
            else:
                aligned_face = face.aligned_face if face.aligned_face is not None else self._align_face_simple(face_image, face)
                embedding_result = self.face_recognizer.extract_embedding(aligned_face)
                embedding_val = embedding_result.embedding
                is_valid_emb = embedding_result.is_valid

            if not is_valid_emb:
                return FaceRegistrationResult(
                    success=False, message="Không thể trích xuất đặc trưng khuôn mặt",
                    embedding_id=None, quality_score=quality_score,
                    total_embeddings=existing_count
                )
            
            # Use embedding_val for registration
            from app.ml.face_recognition import FaceEmbeddingResult
            embedding_result = FaceEmbeddingResult(embedding=embedding_val, confidence=1.0, is_valid=True)
            
            # Check for duplicate embedding (same face already registered)
            is_duplicate = await self._check_duplicate_embedding(
                student_id, embedding_result.embedding, db
            )
            if is_duplicate:
                return FaceRegistrationResult(
                    success=False, message="Góc chụp này đã được đăng ký. Thử góc khác.",
                    embedding_id=None, quality_score=quality_score,
                    total_embeddings=existing_count
                )
                
            # Save embedding
            embedding_bytes = embedding_result.to_bytes()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()
            
            face_embedding = FaceEmbedding(
                student_id=student_id,
                embedding=embedding_bytes,
                embedding_hash=embedding_hash,
                quality_score=quality_score
            )
            
            db.add(face_embedding)
            await db.commit()
            await db.refresh(face_embedding)
            
            new_total = existing_count + 1
            
            return FaceRegistrationResult(
                success=True,
                message=f"Đăng ký thành công! ({new_total}/{self.MAX_EMBEDDINGS_PER_STUDENT} ảnh)",
                embedding_id=face_embedding.id,
                quality_score=quality_score,
                total_embeddings=new_total
            )
            
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            return FaceRegistrationResult(
                success=False, message=f"Lỗi hệ thống: {str(e)}",
                embedding_id=None, quality_score=0.0, total_embeddings=0
            )
    
    async def _check_duplicate_embedding(
        self,
        student_id: str,
        new_embedding: np.ndarray,
        db: AsyncSession,
        similarity_threshold: float = 0.95
    ) -> bool:
        """Check if a similar embedding already exists for this student."""
        # Calculate distance threshold (1 - similarity)
        distance_threshold = 1.0 - similarity_threshold
        
        # Check if any embedding for this student is close enough
        stmt = select(FaceEmbedding).where(
            and_(
                FaceEmbedding.student_id == student_id,
                FaceEmbedding.embedding.cosine_distance(new_embedding) <= distance_threshold
            )
        ).limit(1)
        
        result = await db.execute(stmt)
        return result.first() is not None
    
    async def register_multiple_faces(
        self,
        student_id: str,
        face_images: List[np.ndarray],
        db: AsyncSession
    ) -> Tuple[int, List[str]]:
        """
        Register multiple face images for a student.
        
        Args:
            student_id: Student ID
            face_images: List of face images (different angles)
            db: Database session
            
        Returns:
            Tuple of (successful_count, error_messages)
        """
        successful = 0
        errors = []
        
        for i, image in enumerate(face_images):
            result = await self.register_student_face(student_id, image, db)
            if result.success:
                successful += 1
            else:
                errors.append(f"Ảnh {i+1}: {result.message}")
                
        return successful, errors

    def _select_best_face(self, image: np.ndarray, faces: List[DetectedFace]) -> Optional[DetectedFace]:
        """
        Select the best face from multiple detections.
        
        When multiple faces are detected, automatically selects the PRIMARY face:
        - The person standing directly in front of the camera
        - Largest face (closest to camera)
        - Most centered in frame
        
        This allows the system to work even when there are other people in background.
        """
        if not faces:
            return None
            
        if len(faces) == 1:
            return faces[0]
        
        # Log when multiple faces detected
        logger.info(f"Multiple faces detected ({len(faces)}), auto-selecting best candidate...")
            
        # Get image center
        h, w = image.shape[:2]
        img_center_x, img_center_y = w / 2, h / 2
        
        best_face = None
        best_score = -1.0
        
        for face in faces:
            # Calculate face area (larger = closer to camera)
            face_area = face.width * face.height
            image_area = w * h
            area_ratio = face_area / image_area
            
            # Size score: larger face = higher score (max 1.0 when face is 30%+ of image)
            size_score = min(area_ratio / 0.30, 1.0)
            
            # Calculate distance from center (normalized 0-1)
            face_center_x, face_center_y = face.center
            dist = np.sqrt(((face_center_x - img_center_x)/w)**2 + ((face_center_y - img_center_y)/h)**2)
            
            # Center score: closer to center is better (max 1.0)
            center_score = max(0, 1.0 - dist * 2) 
            
            # Weighted total score - SIZE IS MORE IMPORTANT (person closest to camera)
            # Size: 50%, Center: 30%, Confidence: 20%
            total_score = size_score * 0.5 + center_score * 0.3 + face.confidence * 0.2
            
            logger.debug(f"Face candidate: size={area_ratio:.2%}, center_dist={dist:.2f}, score={total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        if best_face:
            best_area = (best_face.width * best_face.height) / (w * h)
            logger.info(f"Selected best face: size={best_area:.1%} of frame, score={best_score:.2f}")
            
        # Minimum threshold: face should be reasonably prominent
        if best_score < 0.3:
            logger.warning("No sufficiently prominent face found")
            return None
            
        return best_face

