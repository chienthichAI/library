"""
SmartLib Kiosk - Authentication Service

Handles student authentication via face recognition.
Uses ArcFace + AntiSpoofing for secure verification.
Enhanced with image quality check and multiple embeddings support.
"""
import numpy as np
import cv2
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
from app.services.face_eval_logger import get_face_eval_logger, FaceEvalLogEntry


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


from app.core.ml_container import AIModels


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
    
    # Maximum embeddings per student for multi-angle registration & continuous learning
    MAX_EMBEDDINGS_PER_STUDENT = 10
    
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
        
        # Use pre-loaded models from AIModels container by default
        self.face_detector = face_detector or AIModels.face_detector or FaceDetector()
        
        self.face_recognizer = face_recognizer or AIModels.face_recognizer or FaceRecognizer(
            face_analysis_instance=getattr(self.face_detector, '_model', None),
            model_path=settings.face_model_path if not getattr(self.face_detector, '_model', None) else None,
            use_gpu=settings.use_gpu
        )
        self.anti_spoofing = anti_spoofing or AIModels.anti_spoofing or AntiSpoofing(

            model_path=settings.antispoofing_model_path,
            threshold=self.liveness_threshold
        )
        self.quality_checker = quality_checker or ImageQualityChecker()
        
        # Tracking Cache: { track_id: {"result": AuthenticationResult, "timestamp": float} }
        self.track_cache = {}
        self.cache_ttl = 10.0  # Increased to 10s to prevent rapid re-auth triggers
        
        # Ensure threshold is updated if component was passed in
        if self.anti_spoofing:
            self.anti_spoofing.threshold = self.liveness_threshold
        
    async def initialize(self) -> bool:
        """Models are now initialized via lifespan + AIModels container."""
        # Just ensure everything is ready
        if not self.face_detector._initialized:
            self.face_detector.initialize()
        if not self.face_recognizer._initialized:
            self.face_recognizer.initialize()
        if not self.anti_spoofing._initialized:
            self.anti_spoofing.initialize()
        return True

    async def _write_audit_log(self, db: AsyncSession, event_type: str, start_time: float, 
                               student_id: str = None, error_msg: str = None,
                               conf: float = 0.0, liveness: float = 0.0, **kwargs):
        """Write audit log to DB and append evaluation CSV log."""
        from app.models.audit_log import AuditLog
        import time
        try:
            now = time.time()
            proc_time_ms = (now - start_time) * 1000
            details = {"error": error_msg, **kwargs}
            log_entry = AuditLog(
                event_type=event_type,
                student_id=student_id,
                similarity_score=conf,
                liveness_score=liveness,
                processing_time_ms=proc_time_ms,
                details=details
            )
            db.add(log_entry)

            # Also export to CSV for offline daily evaluation.
            try:
                eval_logger = get_face_eval_logger()
                eval_logger.log(
                    FaceEvalLogEntry(
                        event_type=event_type,
                        predicted_student_id=student_id or "UNKNOWN",
                        predicted_student_name=str(kwargs.get("predicted_student_name") or ""),
                        confidence=float(conf or 0.0),
                        liveness_score=float(liveness or 0.0),
                        is_real_face=bool((liveness or 0.0) >= self.liveness_threshold),
                        quality_score=float(kwargs.get("quality_score") or 0.0),
                        processing_time_ms=float(proc_time_ms),
                        error_message=error_msg or "",
                        source=str(kwargs.get("source") or "api"),
                        track_id=str(kwargs.get("track_id") or ""),
                        ground_truth_student_id=str(kwargs.get("ground_truth_student_id") or ""),
                    )
                )
            except Exception as csv_err:
                logger.warning(f"Failed to write face evaluation CSV log: {csv_err}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def authenticate(
        self,
        image: np.ndarray,
        db: AsyncSession,
        check_quality: bool = True,
        frames: Optional[List[np.ndarray]] = None,
        track_id: Optional[str] = None,
        pre_detected_face: Optional[DetectedFace] = None,
        ground_truth_student_id: Optional[str] = None,
        source: str = "api",
    ) -> AuthenticationResult:
        """
        Authenticate a student using face recognition.
        
        Args:
            image: BGR image from camera
            db: Database session
            check_quality: Whether to perform quality checks
            frames: Optional list of recent frames for temporal anti-spoofing (e.g. rPPG)
            track_id: Optional ID from object tracker to use cached result
            pre_detected_face: Optional DetectedFace to bypass face detection step
        Returns:
            AuthenticationResult with success status and student info
        """
        import time
        start_time = time.time()
        log_context = {
            "source": source,
            "track_id": track_id or "",
            "ground_truth_student_id": ground_truth_student_id or "",
        }
        
        if track_id is not None and track_id in self.track_cache:
            cache_entry = self.track_cache.get(track_id)
            if cache_entry and (start_time - cache_entry["timestamp"] < self.cache_ttl):
                logger.info(f"Using cached authentication for TrackID: {track_id}")
                return cache_entry["result"]
            elif cache_entry:
                del self.track_cache[track_id]
                
        quality_issues = []
        quality_score = 1.0
        
        try:
            # Step 1: Detect face FIRST (needed for quality check)
            t_det0 = time.time()
            if pre_detected_face is not None:
                faces = [pre_detected_face]
            else:
                faces = self.face_detector.detect(image, max_faces=5)
            t_det1 = time.time()
            logger.info(f"[Perf] FaceDetector.detect took {(t_det1-t_det0)*1000:.2f}ms")
            
            num_faces = len(faces)
            
            if not faces:
                error_msg = "Không phát hiện khuôn mặt. Đưa mặt vào khung hình."
                await self._write_audit_log(db, "AUTH_FAIL", start_time, error_msg=error_msg, quality_score=0.0, **log_context)
                await db.commit()
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=0.0, is_real_face=False,
                    error_message=error_msg,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=0.0
                )
            
            face = self._select_best_face(image, faces)
            if face is None:
                error_msg = "Không xác định được khuôn mặt chính. Đưa mặt vào giữa khung hình."
                await self._write_audit_log(db, "AUTH_FAIL", start_time, error_msg=error_msg, quality_score=0.0, **log_context)
                await db.commit()
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=0.0, is_real_face=False,
                    error_message=error_msg,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=0.0
                )
                
            face_bbox = (face.x1, face.y1, face.x2, face.y2)
            
            # Step 2: Check image quality
            quality_score = 1.0
            if check_quality:
                quality_result = self.quality_checker.check(image, face_bbox, num_faces, landmarks=getattr(face, 'landmarks', None))
                quality_score = quality_result.overall_score
                quality_issues = [issue.value for issue in quality_result.issues]
                
                if not quality_result.is_valid:
                    error_msg = quality_result.vietnamese_message
                    await self._write_audit_log(db, "AUTH_FAIL", start_time, error_msg=error_msg, quality_score=quality_score, quality_issues=quality_issues, **log_context)
                    await db.commit()
                    return AuthenticationResult(
                        success=False, student_id=None, student_name=None, role=None,
                        confidence=0.0, liveness_score=0.0, is_real_face=False,
                        error_message=error_msg,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        quality_score=quality_score,
                        quality_issues=quality_issues
                    )
            
            # Step 3: Check liveness (anti-spoofing)
            t_spoof0 = time.time()
            # MiniFASNet needs background context (shoulders/bg) so we scale crop by 2.7
            face_crop_liveness = self._crop_face(image, face, scale=2.7)
            liveness_result = self.anti_spoofing.detect(face_crop_liveness)
            
            # We now have multiple frames for temporal analysis (rPPG & Screen Flicker)
            from app.ml.anti_spoofing import extract_rppg_signal, detect_screen_flicker
            
            texture_conf = liveness_result.confidence
            rppg_score = 1.0
            flicker_score = 1.0
            
            if frames and len(frames) >= 10:
                logger.info(f"Running temporal liveness analysis on {len(frames)} frames")
                # face_bbox is (x1, y1, x2, y2)
                rppg_score = extract_rppg_signal(frames, face_bbox)
                flicker_score = detect_screen_flicker(frames, face_bbox)
            else:
                logger.info("No temporal frames provided or insufficient frames, using texture-only anti-spoof signals")
                rppg_score = texture_conf
                flicker_score = texture_conf
                
            # Combine Model + Temporal Signals (Defense in depth)
            # Weighted fusion
            # Texture confidence is heavily weighted because MiniFASNet is very accurate.
            # rPPG & flicker are supplementary signals to catch sophisticated masks.
            final_score = texture_conf * 0.7 + rppg_score * 0.2 + flicker_score * 0.1
            
            # If MiniFASNet strongly believes it is fake, don't let temporal features override it too much
            if texture_conf < 0.2:
                final_score = min(final_score, texture_conf * 1.5)
                
            # If temporal scores are reasonably good AND texture is on the fence, boost the final score
            if rppg_score > 0.15 and flicker_score > 0.3 and 0.2 <= texture_conf < 0.5:
                logger.info(f"Boosting liveness due to temporal features (rPPG: {rppg_score:.2f}, Flicker: {flicker_score:.2f})")
                final_score = max(final_score, rppg_score * 0.5 + flicker_score * 0.5)

            # Fast reject only if literally EVERYTHING fails
            if final_score < 0.1:
                spoof_t = "texture_and_temporal_fake"
            else:
                spoof_t = liveness_result.spoof_type if not liveness_result.is_real else "replay_fullscreen"
                
            # P4-05: Real face determination based on FUSED final_score (not raw MiniFASNet)
            # final_score = texture(70%) + rPPG(20%) + flicker(10%)
            liveness_score = final_score  # Use fused score, not raw model score
            is_real = liveness_score >= self.liveness_threshold
            
            t_spoof1 = time.time()
            logger.info(f"[Perf] AntiSpoofing.detect took {(t_spoof1-t_spoof0)*1000:.2f}ms | texture={texture_conf:.2f}, rPPG={rppg_score:.2f}, flicker={flicker_score:.2f}, final={liveness_score:.2f}, is_real={is_real}")
            
            if not is_real:
                error_msg = f"Phát hiện giả mạo: {spoof_t} (Score: {liveness_score:.2f})"
                await self._write_audit_log(db, "SPOOF_DETECTED", start_time, error_msg=error_msg, liveness=liveness_score, quality_score=quality_score, **log_context)
                await db.commit()
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=liveness_score,
                    is_real_face=False,
                    error_message=error_msg,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=quality_score
                )
                
            # Step 4: Extract embedding (Use pre-computed from detector if available)
            t_emb0 = time.time()
            if face.embedding is not None:
                query_embedding = face.embedding
            else:
                aligned_face = face.aligned_face if face.aligned_face is not None else self._align_face_simple(image, face)
                embedding_result = self.face_recognizer.extract_embedding(aligned_face)
                if not embedding_result.is_valid:
                    error_msg = "Không thể trích xuất đặc trưng khuôn mặt"
                    await self._write_audit_log(db, "AUTH_FAIL", start_time, error_msg=error_msg, liveness=liveness_score, quality_score=quality_score, **log_context)
                    await db.commit()
                    return AuthenticationResult(
                        success=False, student_id=None, student_name=None, role=None,
                        confidence=0.0, liveness_score=liveness_score,
                        is_real_face=True,
                        error_message=error_msg,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        quality_score=quality_score
                    )
                query_embedding = embedding_result.embedding
            t_emb1 = time.time()
            logger.info(f"[Perf] Embedding extraction took {(t_emb1-t_emb0)*1000:.2f}ms")
                
            # Step 5: Match against database (ALL embeddings)
            match_result = await self._find_matching_student(
                query_embedding, db
            )

            # If no match, retry with mirrored embedding to handle camera/register
            # orientation inconsistencies (common with front-camera mirror UX).
            if match_result is None:
                aligned_for_retry = face.aligned_face if face.aligned_face is not None else self._align_face_simple(image, face)
                try:
                    if aligned_for_retry is not None and aligned_for_retry.size > 0:
                        mirrored_face = cv2.flip(aligned_for_retry, 1)
                        mirrored_emb_result = self.face_recognizer.extract_embedding(mirrored_face)
                        if mirrored_emb_result.is_valid:
                            mirror_match = await self._find_matching_student(mirrored_emb_result.embedding, db)
                            if mirror_match is not None:
                                logger.info("Mirror-embedding fallback produced a valid face match")
                                match_result = mirror_match
                except Exception as mirror_err:
                    logger.warning(f"Mirror fallback matching failed: {mirror_err}")
            
            if match_result is None:
                error_msg = "Không tìm thấy sinh viên. Vui lòng đăng ký trước."
                await self._write_audit_log(db, "AUTH_MISMATCH", start_time, error_msg=error_msg, liveness=liveness_score, quality_score=quality_score, **log_context)
                await db.commit()
                return AuthenticationResult(
                    success=False, student_id=None, student_name=None, role=None,
                    confidence=0.0, liveness_score=liveness_score,
                    is_real_face=True,
                    error_message=error_msg,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    quality_score=quality_score
                )
                
            student_id, full_name, role, similarity = match_result
            
            # Step 6 (Layer 6): Continuous Learning
            if similarity >= settings.continuous_learning_threshold and is_real and settings.continuous_learning_enabled:
                await self._trigger_continuous_learning(student_id, query_embedding, quality_score, db)
            
            # Step 7: Update last login (Direct DB update for performance)
            from sqlalchemy import update
            from app.models.student import Student
            await db.execute(
                update(Student)
                .where(Student.student_id == student_id)
                .values(last_login=datetime.utcnow())
            )
            
            await self._write_audit_log(
                db,
                "AUTH_SUCCESS",
                start_time,
                student_id=student_id,
                conf=similarity,
                liveness=liveness_score,
                quality_score=quality_score,
                predicted_student_name=full_name,
                **log_context,
            )
            await db.commit()
            
            result_obj = AuthenticationResult(
                success=True,
                student_id=student_id,
                student_name=full_name,
                role=role.value if hasattr(role, 'value') else role,
                confidence=similarity,
                liveness_score=liveness_score,
                is_real_face=is_real,
                error_message=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=quality_score
            )
            
            if track_id is not None:
                self.track_cache[track_id] = {
                    "result": result_obj,
                    "timestamp": time.time()
                }
            
            return result_obj
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}", exc_info=True)
            error_msg = "Lỗi hệ thống. Vui lòng thử lại hoặc liên hệ quản trị viên."
            await self._write_audit_log(db, "SYSTEM_ERROR", start_time, error_msg=str(e), **log_context)
            await db.commit()
            return AuthenticationResult(
                success=False, student_id=None, student_name=None, role=None,
                confidence=0.0, liveness_score=0.0, is_real_face=False,
                error_message=error_msg,
                processing_time_ms=(time.time() - start_time) * 1000,
                quality_score=0.0
            )

    async def _trigger_continuous_learning(self, student_id: str, new_embedding: np.ndarray, quality_score: float, db: AsyncSession):
        """
        Layer 6: Continuous Learning.
        Auto-adds a new highly confident facial embedding to improve future recognition.
        Avoids adding duplicates if an identical vector already exists.
        """
        try:
            # Check current total
            count_stmt = select(func.count()).where(FaceEmbedding.student_id == student_id)
            count_result = await db.execute(count_stmt)
            existing_count = count_result.scalar() or 0
            
            # Only store up to MAX dynamically learned embeddings to prevent DB bloat
            if existing_count >= self.MAX_EMBEDDINGS_PER_STUDENT:
                return
                
            # Ensure it's not a duplicate perspective (threshold 0.95+)
            is_duplicate = await self._check_duplicate_embedding(student_id, new_embedding, db, similarity_threshold=0.95)
            if not is_duplicate:
                embedding_bytes = new_embedding.astype(np.float32).tobytes()
                embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()
                
                face_embedding = FaceEmbedding(
                    student_id=student_id,
                    embedding=new_embedding,
                    embedding_hash=embedding_hash,
                    quality_score=quality_score
                )
                db.add(face_embedding)
                
                # Add to FAISS cache without waiting for next full sync
                from app.core.ml_container import AIModels
                
                # Note: db.flush() is needed to get the sequence ID before commit
                await db.flush()
                if AIModels.faiss_engine:
                    # Fetch student metadata for the FAISS metadata cache
                    meta_stmt = select(Student.full_name, Student.role).where(Student.student_id == student_id)
                    meta_result = await db.execute(meta_stmt)
                    meta_row = meta_result.first()
                    metadata = {
                        "full_name": meta_row.full_name if meta_row else student_id,
                        "role": meta_row.role if meta_row else "STUDENT",
                    }
                    AIModels.faiss_engine.add_embedding(new_embedding, student_id, metadata)
                
                logger.info(f"Continuous Learning: Auto-registered new face vector for {student_id}")
        except Exception as e:
            logger.warning(f"Continuous learning step failed non-fatally: {e}")

    async def _find_matching_student(self, query_embedding: np.ndarray, db: AsyncSession) -> Optional[Tuple[str, str, str, float]]:
        """
        Match face embedding against the student database.

        Strategy (fast-path first):
        1. FAISS (in-memory, <1ms) — primary path.  Uses Inner Product on L2-normalized
           vectors, which equals cosine similarity.
        2. pgvector fallback — used only when FAISS is not ready (e.g., cold start,
           a brand-new registration that hasn't been synced yet).

        Returns (student_id, full_name, role, similarity) or None.
        """
        import time
        from app.core.ml_container import AIModels

        start_t = time.time()

        # ── PATH 1: FAISS (sub-millisecond) ─────────────────────────────────────
        faiss_engine = AIModels.faiss_engine
        if faiss_engine and faiss_engine.is_ready and faiss_engine.current_idx > 0:
            results = faiss_engine.search(query_embedding, top_k=3)

            if results:
                best = results[0]
                score: float = best["score"]  # cosine similarity (0–1)

                if score >= self.similarity_threshold:
                    elapsed_ms = (time.time() - start_t) * 1000
                    logger.debug(
                        f"FAISS match: {best['student_id']} ({best['full_name']}) "
                        f"| Sim: {score:.3f} | {elapsed_ms:.1f}ms"
                    )
                    return best["student_id"], best["full_name"], best.get("role", "STUDENT"), score

            # FAISS returned results but all below threshold — definitive no-match.
            elapsed_ms = (time.time() - start_t) * 1000
            best_score = results[0]["score"] if results else 0.0
            logger.debug(
                f"FAISS: no match above threshold {self.similarity_threshold} "
                f"(best={best_score:.3f}) | {elapsed_ms:.1f}ms"
            )
            return None

        # ── PATH 2: pgvector fallback (when FAISS not ready) ────────────────────
        logger.warning("FAISS not ready — falling back to pgvector search (slow path)")
        from app.models.face_embedding import FaceEmbedding
        from app.models.student import Student, StudentStatus
        from sqlalchemy import and_, select

        distance_threshold = 1.0 - self.similarity_threshold

        stmt = (
            select(
                Student.student_id,
                Student.full_name,
                Student.role,
                FaceEmbedding.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .join(FaceEmbedding, Student.student_id == FaceEmbedding.student_id)
            .where(
                and_(
                    Student.status == StudentStatus.ACTIVE.value,
                    FaceEmbedding.embedding.cosine_distance(query_embedding) <= distance_threshold,
                )
            )
            .order_by("distance")
            .limit(1)
        )

        result = await db.execute(stmt)
        row = result.first()

        elapsed_ms = (time.time() - start_t) * 1000
        if not row:
            logger.debug(f"pgvector fallback: no match | {elapsed_ms:.1f}ms")
            return None

        student_id, full_name, role, distance = row
        similarity = 1.0 - float(distance)
        logger.debug(
            f"pgvector fallback match: {student_id} ({full_name}) "
            f"| Sim: {similarity:.3f} | {elapsed_ms:.1f}ms"
        )
        return student_id, full_name, role, similarity
    
    def _crop_face(self, image: np.ndarray, face: DetectedFace, scale: float = 1.0) -> np.ndarray:
        """Crop face region from image. Scale > 1.0 captures more background context."""
        if scale == 1.0:
            return image[face.y1:face.y2, face.x1:face.x2]
            
        h, w = image.shape[:2]
        face_w = face.x2 - face.x1
        face_h = face.y2 - face.y1
        
        center_x = face.x1 + face_w // 2
        center_y = face.y1 + face_h // 2
        
        new_size = int(max(face_w, face_h) * scale)
        
        # Extremely crucial: Prevent dimension from exceeding image bounds so it stays strictly SQUARE.
        # MiniFASNet fails spectacularly if aspect ratio is stretched during resize
        new_size = min(new_size, h, w) 
        
        half_size = new_size // 2
        
        y1 = center_y - half_size
        y2 = y1 + new_size
        x1 = center_x - half_size
        x2 = x1 + new_size
        
        # Shift bounds to maintain square aspect ratio if near edges
        if y1 < 0:
            y2 = min(h, y2 - y1)
            y1 = 0
        elif y2 > h:
            y1 = max(0, y1 - (y2 - h))
            y2 = h
            
        if x1 < 0:
            x2 = min(w, x2 - x1)
            x1 = 0
        elif x2 > w:
            x1 = max(0, x1 - (x2 - w))
            x2 = w
        
        return image[y1:y2, x1:x2]
    
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
        import time
        start_time = time.time()
        
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
                error_msg = f"Đã đạt giới hạn {self.MAX_EMBEDDINGS_PER_STUDENT} ảnh đăng ký"
                await self._write_audit_log(db, "REGISTER_FAIL", start_time, student_id=student_id, error_msg=error_msg)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
                    embedding_id=None, quality_score=1.0, total_embeddings=existing_count
                )
                
            # Detect face
            faces = self.face_detector.detect(face_image, max_faces=1)
            
            if not faces:
                error_msg = "Không phát hiện khuôn mặt."
                await self._write_audit_log(db, "REGISTER_FAIL", start_time, student_id=student_id, error_msg=error_msg)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
                    embedding_id=None, quality_score=0.0,
                    total_embeddings=existing_count
                )
            
            # Select the largest, most central face
            face = self._select_best_face(face_image, faces)
            if face is None:
                error_msg = "Không xác định được khuôn mặt chính. Đưa mặt vào giữa khung hình."
                await self._write_audit_log(db, "REGISTER_FAIL", start_time, student_id=student_id, error_msg=error_msg)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
                    embedding_id=None, quality_score=0.0,
                    total_embeddings=existing_count
                )
                
            face_bbox = (face.x1, face.y1, face.x2, face.y2)
            
            # Quality check
            quality_score = 1.0
            if check_quality:
                quality_result = self.quality_checker.check(face_image, face_bbox, 1, landmarks=getattr(face, 'landmarks', None))
                quality_score = quality_result.overall_score
                
                if not quality_result.is_valid:
                    error_msg = quality_result.vietnamese_message
                    await self._write_audit_log(db, "REGISTER_FAIL", start_time, student_id=student_id, error_msg=error_msg, quality_score=quality_score)
                    await db.commit()
                    return FaceRegistrationResult(
                        success=False,
                        message=error_msg,
                        embedding_id=None, quality_score=quality_score,
                        total_embeddings=existing_count
                    )
            
            # Check liveness during registration
            face_crop_liveness = self._crop_face(face_image, face, scale=2.7)
            liveness_result = self.anti_spoofing.detect(face_crop_liveness)
            if not liveness_result.is_real:
                error_msg = f"Phát hiện giả mạo: {liveness_result.spoof_type}"
                await self._write_audit_log(db, "SPOOF_DETECTED_REGISTER", start_time, student_id=student_id, error_msg=error_msg, quality_score=quality_score)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
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
                error_msg = "Trích xuất đặc trưng thất bại."
                await self._write_audit_log(db, "REGISTER_FAIL", start_time, student_id=student_id, error_msg=error_msg, quality_score=quality_score)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
                    embedding_id=None, quality_score=quality_score,
                    total_embeddings=existing_count
                )
            
            # Use embedding_val for registration
            from app.ml.face_recognition import FaceEmbeddingResult
            embedding_result = FaceEmbeddingResult(embedding=embedding_val, confidence=1.0, is_valid=True)
            
            # Check for duplicate embedding (same face already registered)
            is_duplicate = await self._check_duplicate_embedding(
                student_id, embedding_result.embedding, db, similarity_threshold=0.95
            )
            if is_duplicate:
                error_msg = "Khuôn mặt này đã được đăng ký. Hãy thử góc mặt khác."
                # Don't technically fail audit heavily, but mark as duplicate
                await self._write_audit_log(db, "REGISTER_DUPLICATE", start_time, student_id=student_id, error_msg=error_msg, quality_score=quality_score)
                await db.commit()
                return FaceRegistrationResult(
                    success=False,
                    message=error_msg,
                    embedding_id=None, quality_score=quality_score,
                    total_embeddings=existing_count
                )
                
            # Save embedding
            embedding_bytes = embedding_result.to_bytes()
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()
            
            face_embedding = FaceEmbedding(
                student_id=student_id,
                embedding=embedding_result.embedding,
                embedding_hash=embedding_hash,
                quality_score=quality_score
            )
            
            db.add(face_embedding)
            await db.flush()  # Get ID without committing
            await db.refresh(face_embedding)
            
            # Sync FAISS immediately (B02)
            from app.core.ml_container import AIModels
            if AIModels.faiss_engine:
                # Embedding is already L2-normalized by extract_embedding()
                metadata = {
                    "full_name": student.full_name if student else student_id,
                    "role": student.role if student else "STUDENT"
                }
                AIModels.faiss_engine.add_embedding(embedding_result.embedding, student_id, metadata)
            
            new_total = existing_count + 1
            
            await self._write_audit_log(db, "REGISTER_SUCCESS", start_time, student_id=student_id, quality_score=quality_score)
            await db.commit()  # Single commit for both face_embedding and audit log
            
            return FaceRegistrationResult(
                success=True,
                message=f"Đăng ký thành công góc mặt ({new_total}/{self.MAX_EMBEDDINGS_PER_STUDENT}).",
                embedding_id=face_embedding.id,
                quality_score=quality_score,
                total_embeddings=new_total
            )
            
        except Exception as e:
            logger.error(f"Error registering face: {e}", exc_info=True)
            await self._write_audit_log(db, "SYSTEM_ERROR_REGISTER", start_time=start_time, student_id=student_id, error_msg=str(e))
            await db.commit()
            return FaceRegistrationResult(
                success=False,
                message="Lỗi hệ thống. Vui lòng thử lại.",
                embedding_id=None, quality_score=0.0,
                total_embeddings=0
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
            logger.warning(f"No sufficiently prominent face found. Best score was {best_score:.2f}")
            if best_face:
                area_ratio = (best_face.width * best_face.height) / (w * h)
                face_center_x, face_center_y = best_face.center
                dist = np.sqrt(((face_center_x - img_center_x)/w)**2 + ((face_center_y - img_center_y)/h)**2)
                size_score = min(area_ratio / 0.30, 1.0)
                center_score = max(0, 1.0 - dist * 2) 
                logger.debug(f"Rejecting face: size_score={size_score:.2f}, center_score={center_score:.2f}, confidence={best_face.confidence:.2f}")
            return None
            
        return best_face

