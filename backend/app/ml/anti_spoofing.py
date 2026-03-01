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

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def extract_rppg_signal(frames: list, face_bbox: tuple) -> float:
    """
    frames: list of BGR frames (e.g. 15-30 frames)
    face_bbox: tuple of (x1, y1, x2, y2)
    Returns: liveness score 0-1 based on rPPG heart rate signal extraction.
    """
    if not SCIPY_AVAILABLE or len(frames) < 10:
        return 1.0  # Fail-open if not enough frames or missing scipy

    green_channel_means = []
    x1, y1, x2, y2 = face_bbox
    
    # Restrict to forehead/cheek to avoid eyes and mouth movement
    forehead_y1 = y1 + int((y2 - y1) * 0.1)
    forehead_y2 = y1 + int((y2 - y1) * 0.4)

    gray_prev = None
    for frame in frames:
        roi = frame[forehead_y1:forehead_y2, x1:x2]
        if roi.size == 0:
            continue
            
        # Optional: check if frame is exactly same as previous
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_prev is not None:
            if np.mean(cv2.absdiff(gray, gray_prev)) < 0.5:
                continue # Skip identical frame
                
        gray_prev = gray
            
        # Green channel is most sensitive to hemoglobin absorption
        g_mean = np.mean(roi[:, :, 1])
        green_channel_means.append(g_mean)
        
    if len(green_channel_means) < 10:
        return 1.0
        
    signal = np.array(green_channel_means)
    signal = signal - np.mean(signal)  # Detrend
    
    # Bandpass filter: heart rate frequencies 0.7-3.5 Hz (42-210 BPM)
    # Assuming ~30fps
    fps = 30
    nyq = fps / 2.0
    low, high = 0.7 / nyq, 3.5 / nyq
    try:
        b, a = butter(3, [low, high], btype='band')
        
        # Dynamically set padlen to avoid ValueError for short signals
        pad_len = min(3 * max(len(a), len(b)), len(signal) - 1)
        
        filtered = filtfilt(b, a, signal, padlen=pad_len)
        
        # Measure energy in the heart rate band vs out of band
        pulse_energy = np.var(filtered)
        noise_energy = np.var(signal - filtered)
        
        snr = pulse_energy / (noise_energy + 1e-6)
        liveness_score = min(snr / 2.0, 1.0)  # Normalize
        
        logger.info(f"rPPG Liveness check: SNR {snr:.4f} -> Score {liveness_score:.4f}")
        return liveness_score
    except Exception as e:
        logger.warning(f"rPPG filtering error: {e}")
        return 1.0

def detect_screen_flicker(frames: list, face_bbox: tuple) -> float:
    """
    Detect screen refresh flicker via temporal FFT.
    """
    if len(frames) < 10:
        return 1.0
        
    means = []
    x1, y1, x2, y2 = face_bbox
    gray_prev = None
    for frame in frames:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if gray_prev is not None:
                if np.mean(cv2.absdiff(gray, gray_prev)) < 0.5:
                    continue
            gray_prev = gray
            means.append(np.mean(roi))
            
    if len(means) < 10:
        return 1.0
        
    signal = np.array(means)
    signal = signal - np.mean(signal)
    
    fps = 30
    fft_vals = np.abs(np.fft.rfft(signal))
    
    if len(fft_vals) <= 1:
        return 1.0
    
    max_energy = np.max(fft_vals[1:])
    # For a short 15-frame window (0.5s), frequencies are very coarse.
    # We mainly look for high energy at the tail end (Nyquist = 15Hz).
    # Screen recordings often have strong aliasing artifacts showing up as high-freq noise.
    tail_energy = np.mean(fft_vals[len(fft_vals)//2:])
    
    screen_ratio = tail_energy / (max_energy + 1e-6)
    score = 1.0 - min(screen_ratio * 4.0, 1.0)
    logger.info(f"Screen Flicker check: Ratio {screen_ratio:.4f} -> Score {score:.4f}")
    return score


@dataclass  
class AntiSpoofingResult:
    """Result of anti-spoofing detection."""
    is_real: bool
    confidence: float
    spoof_type: Optional[str] = None  # "print", "replay", "mask", or None
    
    @property
    def liveness_score(self) -> float:
        """Xác suất là người thật, luôn trong [0, 1]."""
        return self.confidence


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
                import os
                if os.getenv("ALLOW_HEURISTIC_SPOOF", "false").lower() != "true":
                    raise FileNotFoundError(
                        f"Anti-spoofing model không tìm thấy tại {self.model_path}. "
                        "Không thể khởi động kiosk mà không có bảo vệ liveness. (Set ALLOW_HEURISTIC_SPOOF=true to override)"
                    )
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
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _run_model_inference(self, face_image: np.ndarray) -> AntiSpoofingResult:
        """Run MiniFASNet model inference."""
        # Preprocess
        face = cv2.resize(face_image, self.input_size)
        
        # MiniFASNet model expects BGR image (as read by cv2), unscaled [0, 255]
        face = face.astype(np.float32)
        
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # Add batch
        
        # Inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: face})
        
        probs = self._softmax(outputs[0].flatten())
            
        # MiniFASNet standard has 3 classes: [Fake(Print), Real(1), Fake(Replay/3D)]
        if len(probs) == 3:
            real_prob = probs[1]
            fake_print_prob = probs[0]
            fake_replay_prob = probs[2]
            
            fake_total = fake_print_prob + fake_replay_prob
            is_real = real_prob >= self.threshold
            
            spoof_type = None if is_real else ("print" if fake_print_prob > fake_replay_prob else "replay")
            
            logger.info(f"LIVENESS CHECK - Fake(Print): {fake_print_prob:.4f} | Real: {real_prob:.4f} | Fake(Replay): {fake_replay_prob:.4f} | Threshold: {self.threshold} | is_real: {is_real}")
            
            return AntiSpoofingResult(
                is_real=bool(is_real),
                confidence=float(real_prob),  # B03: Ensure high prob = real
                spoof_type=spoof_type
            )
        elif len(probs) >= 2:
            real_prob = probs[1]
            fake_prob = probs[0]
            
            is_real = real_prob >= self.threshold
            spoof_type = None if is_real else "fake"
            logger.info(f"LIVENESS CHECK - Type: 2-class | Fake: {fake_prob:.4f} | Real: {real_prob:.4f} | Threshold: {self.threshold} | is_real: {is_real}")
            return AntiSpoofingResult(
                is_real=bool(is_real),
                confidence=float(real_prob),
                spoof_type=spoof_type
            )
        else:
            real_prob = float(probs[0] > 0.5)
            is_real = real_prob >= self.threshold
            logger.info(f"LIVENESS CHECK - Type: 1-class | Real: {real_prob:.4f} | Threshold: {self.threshold} | is_real: {is_real}")
            return AntiSpoofingResult(
                is_real=bool(is_real),
                confidence=float(real_prob),
                spoof_type=None if is_real else "fake"
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
        
        # Calculate overall heuristic
        # 1. Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)
        scores.append(blur_score)
        
        # 2. Frequency analysis for moiré patterns
        # Screen replay attacks often show moiré patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Mask: only keep high frequency region
        mask = np.ones((rows, cols), np.uint8)
        mask[max(0, crow-20):crow+20, max(0, ccol-20):ccol+20] = 0
        
        high_freq_energy = np.sum(np.abs(f_shift) * mask)
        low_freq_energy = np.sum(np.abs(f_shift) * (1 - mask))
        
        # Fake (screen) has higher high_freq ratio
        moire_ratio = high_freq_energy / (low_freq_energy + 1e-6)
        freq_score = max(0.0, 1.0 - moire_ratio * 0.1)
        scores.append(freq_score)
        
        # 3. Enhance single-frame evaluation with Color distribution
        if len(face_image.shape) == 3:
            color_var = np.var(face_image, axis=(0, 1)).mean()
            color_score = min(color_var / 2000.0, 1.0)
            scores.append(color_score)
            
            # Additional logic: Check for flat green channel standard deviation
            # (often an indicator of generic digital displays)
            g_std = np.std(face_image[:, :, 1])
            if g_std < 10.0:
                scores.append(0.1) # Penalize heavily if very flat
                
        # Combine scores
        final_score = float(np.mean(scores))

        is_real = bool(final_score >= self.threshold)
        
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
            
            # Combine RGB and depth scores as independent gates
            if not is_depth_valid:
                return AntiSpoofingResult(
                    is_real=False,
                    confidence=rgb_result.confidence * 0.5,
                    spoof_type="3d_mask_or_flat"
                )
                
            return rgb_result
            
        except Exception as e:
            logger.error(f"Depth analysis failed: {e}")
            return rgb_result
