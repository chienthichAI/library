"""
SmartLib Kiosk - Anti-Spoofing / Liveness Detection

MiniFASNet is a lightweight face anti-spoofing network designed to prevent:
- Print attacks (photos)
- Replay attacks (videos)
- 3D mask attacks

Architecture:
- Backbone: MobileFaceNet (lightweight CNN)
- Multi-scale feature extraction
- Binary classification: Real vs Fake

Performance:
- ACER (Average Classification Error Rate): < 2%
- Speed: ~15ms per face on GPU
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass  
class AntiSpoofingResult:
    """Result of anti-spoofing detection."""
    is_real: bool
    confidence: float
    spoof_type: Optional[str] = None  # "print", "replay", "mask", or None
    
    @property
    def liveness_score(self) -> float:
        """Score indicating how likely the face is real (0-1)."""
        return self.confidence if self.is_real else 1.0 - self.confidence


class AntiSpoofing:
    """
    Face Anti-Spoofing using MiniFASNet Deep Neural Network.
    
    This model detects presentation attacks by analyzing:
    1. Texture patterns (print artifacts, screen moiré)
    2. Depth cues (2D vs 3D face)
    3. Temporal consistency (if video input)
    
    Architecture Details:
    - Input: 80x80x3 or 128x128x3 RGB image
    - Backbone: MobileFaceNet variant
    - Output: Binary classification (Real/Fake) with confidence
    
    The model uses:
    - Convolution layers for feature extraction
    - Depth-wise separable convolutions for efficiency
    - Global average pooling
    - Fully connected layer for classification
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (80, 80),
        threshold: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize anti-spoofing detector.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            threshold: Threshold for real/fake classification
            use_gpu: Whether to use GPU
        """
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.use_gpu = use_gpu
        self._session = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Load and initialize the anti-spoofing model."""
        if self._initialized:
            return True
            
        try:
            if self.model_path and Path(self.model_path).exists() and ONNX_AVAILABLE:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self._session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"Loaded anti-spoofing model from: {self.model_path}")
            else:
                logger.warning("Anti-spoofing model not available. Using heuristic detection.")
                
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize anti-spoofing: {e}")
            return False
    
    def detect(self, face_image: np.ndarray) -> AntiSpoofingResult:
        """
        Detect if a face image is real or spoofed.
        
        Args:
            face_image: Face image (RGB), preferably aligned
            
        Returns:
            AntiSpoofingResult with is_real flag and confidence
        """
        if not self._initialized:
            self.initialize()
            
        if face_image is None or face_image.size == 0:
            return AntiSpoofingResult(is_real=False, confidence=0.0, spoof_type="invalid_input")
            
        try:
            if self._session is not None:
                # Use ONNX model
                return self._run_model_inference(face_image)
            else:
                # Use heuristic-based detection
                return self._heuristic_detection(face_image)
                
        except Exception as e:
            logger.error(f"Anti-spoofing detection failed: {e}")
            return AntiSpoofingResult(is_real=False, confidence=0.0, spoof_type="error")
    
    def _run_model_inference(self, face_image: np.ndarray) -> AntiSpoofingResult:
        """Run MiniFASNet model inference."""
        # Preprocess
        face = cv2.resize(face_image, self.input_size)
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5  # Normalize to [-1, 1]
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # Add batch
        
        # Inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: face})
        
        # Parse output (assuming softmax output [fake_prob, real_prob])
        probs = outputs[0].flatten()
        
        if len(probs) >= 2:
            real_prob = probs[1]
            fake_prob = probs[0]
        else:
            real_prob = float(probs[0] > 0.5)
            fake_prob = 1.0 - real_prob
            
        is_real = real_prob >= self.threshold
        spoof_type = None if is_real else self._classify_spoof_type(probs)
        
        return AntiSpoofingResult(
            is_real=is_real,
            confidence=float(real_prob if is_real else fake_prob),
            spoof_type=spoof_type
        )
    
    def _heuristic_detection(self, face_image: np.ndarray) -> AntiSpoofingResult:
        """
        Heuristic-based anti-spoofing when no model is available.
        
        Uses multiple image quality metrics:
        1. Laplacian variance (blur detection)
        2. Color histogram analysis (print detection)
        3. Moiré pattern detection (screen detection)
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
            
        # Resize for consistent analysis
        gray = cv2.resize(gray, (128, 128))
        
        scores = []
        
        # 1. Blur detection using Laplacian variance
        # Real faces typically have more texture detail
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize
        scores.append(blur_score)
        
        # 2. Frequency analysis for moiré patterns
        # Screen replay attacks often show moiré patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Check for periodic patterns in high frequencies
        center = magnitude.shape[0] // 2
        high_freq = magnitude[center-40:center+40, center-40:center+40]
        freq_variance = np.var(high_freq)
        freq_score = 1.0 - min(freq_variance / 5.0, 1.0)
        scores.append(freq_score)
        
        # 3. Color distribution analysis (for color images)
        if len(face_image.shape) == 3:
            # Real faces have smooth color gradients
            color_var = np.var(face_image, axis=(0, 1)).mean()
            color_score = min(color_var / 2000.0, 1.0)
            scores.append(color_score)
            
        # 4. Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = 1.0 - min(abs(edge_density - 0.1) * 5, 1.0)
        scores.append(edge_score)
        
        # Combine scores
        final_score = np.mean(scores)
        is_real = final_score >= self.threshold
        
        # Determine likely spoof type
        spoof_type = None
        if not is_real:
            if blur_score < 0.3:
                spoof_type = "print"
            elif freq_score < 0.4:
                spoof_type = "replay"
            else:
                spoof_type = "unknown"
                
        return AntiSpoofingResult(
            is_real=is_real,
            confidence=float(final_score),
            spoof_type=spoof_type
        )
    
    def _classify_spoof_type(self, probs: np.ndarray) -> str:
        """Classify the type of spoof attack if model supports it."""
        # If model outputs multiple classes
        if len(probs) > 2:
            class_names = ["real", "print", "replay", "mask"]
            max_idx = np.argmax(probs)
            if max_idx < len(class_names):
                return class_names[max_idx]
        return "unknown"
    
    def detect_with_depth(
        self,
        face_rgb: np.ndarray,
        face_depth: Optional[np.ndarray] = None
    ) -> AntiSpoofingResult:
        """
        Enhanced anti-spoofing with depth information.
        
        Uses RGB + Depth for more robust detection against:
        - High-quality prints
        - 3D masks
        - Screen replays
        
        Args:
            face_rgb: RGB face image
            face_depth: Depth map (optional, from RGB-D camera)
            
        Returns:
            AntiSpoofingResult
        """
        # Get RGB-based result
        rgb_result = self.detect(face_rgb)
        
        if face_depth is None:
            return rgb_result
            
        # Analyze depth for additional verification
        try:
            depth = cv2.resize(face_depth, (64, 64))
            
            # Real faces have varying depth (nose protrudes)
            depth_variance = np.var(depth)
            depth_range = np.max(depth) - np.min(depth)
            
            # Check for realistic depth distribution
            is_depth_valid = depth_variance > 10 and depth_range > 20
            
            # Combine RGB and depth scores
            if is_depth_valid:
                combined_confidence = (rgb_result.confidence + 0.3) / 1.3
            else:
                combined_confidence = rgb_result.confidence * 0.7
                
            return AntiSpoofingResult(
                is_real=rgb_result.is_real and is_depth_valid,
                confidence=combined_confidence,
                spoof_type=rgb_result.spoof_type if not is_depth_valid else None
            )
            
        except Exception as e:
            logger.error(f"Depth analysis failed: {e}")
            return rgb_result
