"""
SmartLib Kiosk - Image Quality Checker

Validates face images before processing to ensure good recognition accuracy.
Checks for:
- Brightness (too dark/bright)
- Sharpness (blur detection)
- Face size (too small/far)
- Face position (centered)
- Occlusion detection
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class QualityIssue(Enum):
    """Types of image quality issues."""
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    BLURRY = "blurry"
    FACE_TOO_SMALL = "face_too_small"
    FACE_NOT_CENTERED = "face_not_centered"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE = "no_face"
    LOW_CONTRAST = "low_contrast"
    PARTIAL_FACE = "partial_face"


@dataclass
class QualityCheckResult:
    """Result of image quality check."""
    is_valid: bool
    overall_score: float  # 0-1
    brightness_score: float
    sharpness_score: float
    face_size_score: float
    centering_score: float
    issues: List[QualityIssue]
    recommendations: List[str]
    
    @property
    def vietnamese_message(self) -> str:
        """Get user-friendly message in Vietnamese."""
        if self.is_valid:
            return "✓ Chất lượng ảnh tốt"
        
        messages = {
            QualityIssue.TOO_DARK: "Ánh sáng quá tối. Vui lòng di chuyển đến nơi sáng hơn.",
            QualityIssue.TOO_BRIGHT: "Ánh sáng quá chói. Tránh ánh sáng trực tiếp.",
            QualityIssue.BLURRY: "Ảnh bị mờ. Giữ yên khuôn mặt.",
            QualityIssue.FACE_TOO_SMALL: "Khuôn mặt quá nhỏ. Tiến lại gần camera.",
            QualityIssue.FACE_NOT_CENTERED: "Khuôn mặt chưa căn giữa. Điều chỉnh vị trí.",
            QualityIssue.MULTIPLE_FACES: "Phát hiện nhiều khuôn mặt. Chỉ một người.",
            QualityIssue.NO_FACE: "Không phát hiện khuôn mặt.",
            QualityIssue.LOW_CONTRAST: "Thiếu độ tương phản. Điều chỉnh ánh sáng.",
            QualityIssue.PARTIAL_FACE: "Khuôn mặt bị che khuất hoặc không đầy đủ."
        }
        
        return " | ".join([messages.get(issue, str(issue)) for issue in self.issues[:2]])


class ImageQualityChecker:
    """
    Checks image quality for face recognition.
    
    Thresholds are configurable for different environments.
    """
    
    def __init__(
        self,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_sharpness: float = 50.0,
        min_face_ratio: float = 0.15,  # Face should be at least 15% of image
        max_face_ratio: float = 0.70,  # Face should be at most 70% of image
        center_tolerance: float = 0.40  # Increased tolerance (40%) to allow tilted faces
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.min_face_ratio = min_face_ratio
        self.max_face_ratio = max_face_ratio
        self.center_tolerance = center_tolerance
        
    def check(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        num_faces: int = 1
    ) -> QualityCheckResult:
        """
        Check image quality.
        
        Args:
            image: BGR image
            face_bbox: (x1, y1, x2, y2) of detected face
            num_faces: Number of faces detected
            
        Returns:
            QualityCheckResult
        """
        issues = []
        recommendations = []
        
        # Check brightness
        brightness_score = self._check_brightness(image)
        if brightness_score < 0.3:
            issues.append(QualityIssue.TOO_DARK)
            recommendations.append("Increase lighting")
        elif brightness_score < 0.5:
            recommendations.append("Consider better lighting")
        if brightness_score > 0.95:
            issues.append(QualityIssue.TOO_BRIGHT)
            recommendations.append("Reduce direct light")
            
        # Check sharpness
        sharpness_score = self._check_sharpness(image)
        if sharpness_score < 0.3:
            issues.append(QualityIssue.BLURRY)
            recommendations.append("Hold camera steady")
        elif sharpness_score < 0.5:
            recommendations.append("Try to reduce motion")
            
        # Check face count
        if num_faces == 0:
            issues.append(QualityIssue.NO_FACE)
            recommendations.append("Position face in frame")
        # Note: Multiple faces are now ALLOWED - system auto-selects the best (largest, centered) face
        # No error is raised, just an informational note in logs
            
        # Check face size and position
        face_size_score = 1.0
        centering_score = 1.0
        
        if face_bbox is not None:
            face_size_score = self._check_face_size(image, face_bbox)
            if face_size_score < 0.3:
                issues.append(QualityIssue.FACE_TOO_SMALL)
                recommendations.append("Move closer to camera")
                
            centering_score = self._check_centering(image, face_bbox)
            if centering_score < 0.5:
                issues.append(QualityIssue.FACE_NOT_CENTERED)
                recommendations.append("Center your face")
                
            # Check for partial face (face extends beyond image)
            if self._is_partial_face(image, face_bbox):
                issues.append(QualityIssue.PARTIAL_FACE)
                recommendations.append("Move face fully into frame")
                
        # Check contrast
        contrast_score = self._check_contrast(image)
        if contrast_score < 0.3:
            issues.append(QualityIssue.LOW_CONTRAST)
            
        # Calculate overall score
        overall_score = (
            brightness_score * 0.25 +
            sharpness_score * 0.25 +
            face_size_score * 0.25 +
            centering_score * 0.25
        )
        
        # Determine if valid (no critical issues)
        # Note: MULTIPLE_FACES removed - system now auto-selects best face
        critical_issues = {
            QualityIssue.NO_FACE,
            QualityIssue.BLURRY,
            QualityIssue.PARTIAL_FACE
        }
        has_critical = any(issue in critical_issues for issue in issues)
        is_valid = overall_score >= 0.5 and not has_critical
        
        return QualityCheckResult(
            is_valid=is_valid,
            overall_score=overall_score,
            brightness_score=brightness_score,
            sharpness_score=sharpness_score,
            face_size_score=face_size_score,
            centering_score=centering_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _check_brightness(self, image: np.ndarray) -> float:
        """Check image brightness. Returns 0-1 score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)
        
        # Normalize to 0-1 based on ideal range (60-180)
        if mean_brightness < self.min_brightness:
            return mean_brightness / self.min_brightness * 0.5
        elif mean_brightness > self.max_brightness:
            return max(0, 1 - (mean_brightness - self.max_brightness) / 50)
        else:
            # In ideal range
            ideal_center = (self.min_brightness + self.max_brightness) / 2
            distance = abs(mean_brightness - ideal_center)
            max_distance = (self.max_brightness - self.min_brightness) / 2
            return 1 - (distance / max_distance) * 0.3
    
    def _check_sharpness(self, image: np.ndarray) -> float:
        """Check image sharpness using Laplacian variance. Returns 0-1 score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize (typical values: 50-500 for sharp images)
        score = min(laplacian_var / self.min_sharpness, 1.0)
        return max(0, min(score, 1.0))
    
    def _check_face_size(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> float:
        """Check if face size is appropriate. Returns 0-1 score."""
        x1, y1, x2, y2 = face_bbox
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / image_area
        
        if face_ratio < self.min_face_ratio:
            return face_ratio / self.min_face_ratio
        elif face_ratio > self.max_face_ratio:
            return max(0, 1 - (face_ratio - self.max_face_ratio) / 0.2)
        else:
            return 1.0
    
    def _check_centering(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> float:
        """Check if face is centered. Returns 0-1 score."""
        x1, y1, x2, y2 = face_bbox
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2
        
        # Calculate normalized distance from center
        dx = abs(face_center_x - image_center_x) / image.shape[1]
        dy = abs(face_center_y - image_center_y) / image.shape[0]
        
        distance = (dx + dy) / 2
        
        if distance < self.center_tolerance:
            return 1.0
        else:
            return max(0, 1 - (distance - self.center_tolerance) / 0.3)
    
    def _check_contrast(self, image: np.ndarray) -> float:
        """Check image contrast. Returns 0-1 score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contrast = gray.std()
        
        # Normalize (good contrast typically > 40)
        return min(contrast / 50, 1.0)
    
    def _is_partial_face(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if face is partially outside the image."""
        x1, y1, x2, y2 = face_bbox
        h, w = image.shape[:2]
        
        # Check if bbox extends significantly beyond image boundaries
        margin = 5  # pixel margin
        return (
            x1 < margin or
            y1 < margin or
            x2 > w - margin or
            y2 > h - margin
        )
