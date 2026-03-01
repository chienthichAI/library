"""
SmartLib Kiosk - Face Recognition using ArcFace Deep Neural Network

ArcFace (Additive Angular Margin Loss) is a state-of-the-art face recognition model.
Architecture:
- Backbone: ResNet100 (deep residual network)
- Loss: Additive Angular Margin Softmax (ArcFace)
- Output: 512-dimensional face embedding vector

Performance:
- LFW: 99.83% accuracy
- CFP-FP: 98.27% accuracy
- AgeDB-30: 98.28% accuracy
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List, Union, Any
from dataclasses import dataclass
from pathlib import Path
import struct
from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


@dataclass
class FaceEmbeddingResult:
    """Result of face embedding extraction."""
    embedding: np.ndarray  # 512-dimensional vector
    confidence: float
    is_valid: bool
    
    def to_bytes(self) -> bytes:
        """Convert embedding to bytes for database storage."""
        # Ensure L2 normalization before saving (Defensive)
        norm = np.linalg.norm(self.embedding)
        emb = self.embedding / (norm + 1e-6) if norm > 1e-6 else self.embedding
        return emb.astype(np.float32).tobytes()
    
    @staticmethod
    def from_bytes(data: bytes, expected_dim: int = 512) -> 'FaceEmbeddingResult':
        """Reconstruct embedding from bytes."""
        embedding = np.frombuffer(data, dtype=np.float32).copy()
        if len(embedding) != expected_dim:
            logger.error(f"Embedding dim mismatch: got {len(embedding)}, expected {expected_dim}")
            return FaceEmbeddingResult(
                embedding=np.zeros(expected_dim, dtype=np.float32),
                confidence=0.0,
                is_valid=False
            )
        return FaceEmbeddingResult(
            embedding=embedding.copy(),  # B08: Return copy to allow mutation
            confidence=1.0,
            is_valid=True
        )


class FaceRecognizer:
    """
    Face Recognition using ArcFace Deep Neural Network.
    
    The ArcFace model extracts a 512-dimensional embedding vector from
    a face image. These embeddings can be compared using cosine similarity
    to determine if two faces belong to the same person.
    
    Architecture Details:
    - Input: 112x112x3 RGB image (aligned face)
    - Backbone: ResNet100 with 100 layers
    - Output: 512-dim L2-normalized embedding
    - Loss: Additive Angular Margin Loss (ArcFace)
    
    Mathematical formulation:
    L = -log(exp(s*cos(θ_yi + m)) / (exp(s*cos(θ_yi + m)) + Σexp(s*cos(θ_j))))
    where s=64 (scale), m=0.5 (margin)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        face_analysis_instance: Optional[Any] = None,
        embedding_dim: int = 512,
        use_gpu: bool = True
    ):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to standalone ONNX model file (optional)
            face_analysis_instance: Instance of InsightFace's FaceAnalysis (avoids double loading)
            embedding_dim: Output embedding dimension (512 for ArcFace)
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.face_analysis_instance = face_analysis_instance
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        
        self._rec_model = None
        self._session = None
        self._initialized = False
        
        # CLAHE configuration (Contrast Limited Adaptive Histogram Equalization)
        # Only used as fallback if we don't have InsightFace's built-in processing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        
    def initialize(self) -> bool:
        """
        Load and initialize the ArcFace model.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
            
        try:
            if self.face_analysis_instance is not None and hasattr(self.face_analysis_instance, 'models'):
                # Extract recognition model from existing FaceAnalysis instance
                self._rec_model = self.face_analysis_instance.models.get('recognition')
                if self._rec_model is None:
                    # Search through models to find the recognizer
                    for model in self.face_analysis_instance.models.values():
                        if hasattr(model, 'get_feat'):
                            self._rec_model = model
                            break
                if self._rec_model is not None:
                    logger.info("FaceRecognizer hitched to existing FaceAnalysis instance")
            
            elif self.model_path and Path(self.model_path).exists() and ONNX_AVAILABLE:
                # Load custom standalone ONNX model
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self._session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"Loaded standalone ArcFace model from: {self.model_path}")
                
            else:
                logger.warning("No face recognition model available. Using mock mode.")
                
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize face recognizer: {e}")
            return False
    
    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> FaceEmbeddingResult:
        """
        Extract 512-dimensional embedding from aligned face.
        
        The embedding is L2-normalized, meaning ||embedding||_2 = 1.
        This allows using cosine similarity via simple dot product.
        
        Args:
            aligned_face: 112x112x3 aligned face image (RGB)
            
        Returns:
            FaceEmbeddingResult with 512-dim embedding
        """
        if not self._initialized:
            self.initialize()
            
        if aligned_face is None or aligned_face.size == 0:
            logger.warning("Empty face image provided")
            return FaceEmbeddingResult(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                confidence=0.0,
                is_valid=False
            )
            
        try:
            # Ensure correct size
            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(aligned_face, (112, 112))
            
            # Only use CLAHE for standalone ONNX models. 
            # InsightFace models have built-in optimal preprocessing.
            if self._rec_model is not None:
                embedding = self._run_insightface_inference(aligned_face)
            elif self._session is not None:
                aligned_face = self._apply_clahe(aligned_face)
                embedding = self._run_onnx_inference(aligned_face)
            else:
                aligned_face = self._apply_clahe(aligned_face)
                embedding = self._mock_embedding(aligned_face)
                
            # L2 normalize properly
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 0.01:
                embedding = embedding / (norm + 1e-6)
            
            return FaceEmbeddingResult(
                embedding=embedding.astype(np.float32),
                confidence=1.0,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return FaceEmbeddingResult(
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                confidence=0.0,
                is_valid=False
            )
    
    def _run_onnx_inference(self, face: np.ndarray) -> np.ndarray:
        """Run inference using ONNX Runtime."""
        # Preprocess: HWC -> CHW, normalize
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5  # Normalize to [-1, 1]
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: face})
        
        return output[0].flatten()
    
    def _run_insightface_inference(self, face: np.ndarray) -> np.ndarray:
        """Run straight to recognition model sub-node. Skips detection phase."""
        if self._rec_model is not None:
            # InsightFace recognizers typically expect RGB and native 112x112 layout
            embedding = self._rec_model.get_feat(face)
            return embedding.flatten()
            
        return self._mock_embedding(face)
    
    def _mock_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Generate mock embedding for testing.
        Uses image features to create reproducible embeddings.
        """
        if face is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        # Create deterministic embedding from image
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) if len(face.shape) == 3 else face
        resized = cv2.resize(gray, (16, 32))  # 512 pixels
        
        # Normalize to create embedding
        embedding = resized.flatten().astype(np.float32)
        # Safe normalization to avoid broadcasting errors
        mean_val = np.mean(embedding)
        std_val = np.std(embedding) + 1e-6
        embedding = (embedding - mean_val) / std_val
        
        # Ensure 512 dimensions
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
            
        return embedding
    
    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Cosine similarity measures the angle between two vectors:
        similarity = (A · B) / (||A|| * ||B||)
        
        Since embeddings are L2-normalized, this simplifies to dot product.
        
        Args:
            embedding1: First 512-dim embedding
            embedding2: Second 512-dim embedding
            
        Returns:
            Similarity score in range [-1, 1], where 1 = identical
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Guard against zero vectors which indicates failed extraction
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
            
        e1 = embedding1 / norm1
        e2 = embedding2 / norm2
        
        similarity = np.dot(e1, e2)
        
        return float(similarity)
    
    def is_same_person(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person.
        
        Threshold recommendations:
        - 0.5: High FAR (False Accept Rate), low FRR (False Reject Rate)
        - 0.6: Balanced (recommended for most applications)
        - 0.7: Low FAR, higher FRR (high security applications)
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        similarity = self.compare_embeddings(embedding1, embedding2)
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        This improves face recognition robustness in difficult lighting conditions
        (shadows, backlight, low light) by locally enhancing contrast.
        
        Process:
        1. RGB -> LAB color space
        2. Apply CLAHE to L-channel (Luminance)
        3. LAB -> RGB color space
        """
        try:
            # Check if image is color or grayscale
            if len(image.shape) == 3:
                # Convert to LAB
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L-channel
                l2 = self.clahe.apply(l)
                
                # Merge and convert back
                lab = cv2.merge((l2, a, b))
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale
                return self.clahe.apply(image)
        except Exception as e:
            logger.warning(f"CLAHE application failed: {e}")
            return image

    
