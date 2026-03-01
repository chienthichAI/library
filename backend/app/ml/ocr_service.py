"""
SmartLib Kiosk - OCR Service using PaddleOCR

PaddleOCR is used for extracting text from book covers.
Supports Vietnamese and English text recognition.

Architecture:
- Text Detection: DB (Differentiable Binarization)
- Text Recognition: CRNN (Convolutional Recurrent Neural Network)
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from loguru import logger

PADDLEOCR_AVAILABLE = False
try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    try:
        paddle.set_flags({'FLAGS_use_mkldnn': False})
    except:
        pass
except (ImportError, OSError, Exception, BaseException) as e:
    logger.warning(f"PaddleOCR failed to load (DLL conflict suspected): {e}")
    PADDLEOCR_AVAILABLE = False


@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    text: str
    confidence: float
    bbox: List[Tuple[int, int]]  # 4 corner points
    
    @property
    def bounding_rect(self) -> Tuple[int, int, int, int]:
        """Get bounding rectangle (x, y, w, h) safely."""
        if not self.bbox:
            return (0, 0, 0, 0)
        try:
            xs = [p[0] for p in self.bbox]
            ys = [p[1] for p in self.bbox]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            return (int(x), int(y), int(w), int(h))
        except (IndexError, TypeError, ValueError):
            return (0, 0, 0, 0)


@dataclass
class BookOCRResult:
    """Extracted book information from OCR."""
    title: Optional[str]
    author: Optional[str]
    publisher: Optional[str]
    isbn: Optional[str]
    all_text: List[OCRResult]
    confidence: float


class OCRService:
    """
    OCR Service using PaddleOCR Deep Neural Networks.
    
    PaddleOCR Pipeline:
    1. Text Detection (DB - Differentiable Binarization):
       - Finds text regions in image
       - Outputs polygon bounding boxes
       
    2. Text Recognition (CRNN):
       - Convolutional layers for feature extraction
       - Recurrent layers (LSTM) for sequence modeling
       - CTC decoder for character prediction
       
    Optimized for Vietnamese text recognition.
    """
    
    def __init__(
        self,
        lang: str = "vi",
        use_gpu: bool = True,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None
    ):
        """
        Initialize OCR service.
        
        Args:
            lang: Language code ("vi" for Vietnamese, "en" for English)
            use_gpu: Whether to use GPU
            det_model_dir: Custom detection model directory
            rec_model_dir: Custom recognition model directory
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self._ocr = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize PaddleOCR model."""
        if self._initialized:
            return True
            
        try:
            if PADDLEOCR_AVAILABLE:
                # Optimized for GPU if available, fallback to CPU
                kwargs = {
                    "lang": self.lang,
                    "det_limit_side_len": 480, # Increased resolution for GPU
                    "use_angle_cls": False
                }
                
                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        # PaddleOCR 3.x with Paddle 3.x handles device globally
                        import paddle
                        paddle.device.set_device('gpu:0')
                        kwargs["use_gpu"] = True
                        kwargs["enable_mkldnn"] = False
                        logger.info("PaddleOCR initializing on GPU")
                    else:
                        kwargs["use_gpu"] = False
                        kwargs["enable_mkldnn"] = True
                        logger.info("PaddleOCR falling back to CPU (No CUDA)")
                else:
                    kwargs["use_gpu"] = False
                    kwargs["enable_mkldnn"] = True
                
                self._ocr = PaddleOCR(**kwargs)
                logger.info(f"PaddleOCR initialized (GPU={kwargs.get('use_gpu', False)})")
            else:
                logger.warning("PaddleOCR not available. Using mock mode.")
                
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return False
    
    def extract_text(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5
    ) -> List[OCRResult]:
        """
        Extract text from image.
        
        Args:
            image: BGR image
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of OCRResult objects
        """
        if not self._initialized:
            self.initialize()
            
        if image is None or image.size == 0:
            return []
            
        try:
            if self._ocr is not None:
                return self._run_paddleocr(image, min_confidence)
            else:
                return self._mock_ocr(image)
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def _run_paddleocr(
        self,
        image: np.ndarray,
        min_confidence: float
    ) -> List[OCRResult]:
        """Run PaddleOCR inference with robust result parsing."""
        try:
            # Run OCR
            try:
                # Try with classifier
                result = self._ocr.ocr(image, cls=True)
            except Exception:
                # Fallback
                result = self._ocr.ocr(image)
            
            if not result or not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], list):
                return []
                
            ocr_results = []
            
            # PaddleOCR returns a list of results (one per image)
            # We only send one image, so we look at result[0] which should be a list of detections
            for line in result[0]:
                try:
                    # Case 1: Standard result structure [ [ [x,y],... ], (text, confidence) ]
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                            
                            if confidence >= min_confidence and text.strip():
                                # Convert bbox safely
                                bbox_int = []
                                if isinstance(bbox, (list, tuple)):
                                    bbox_int = [(int(p[0]), int(p[1])) for p in bbox if isinstance(p, (list, tuple)) and len(p) >= 2]
                                
                                ocr_results.append(OCRResult(text=text, confidence=confidence, bbox=bbox_int))
                                continue

                    # Case 2: Dict structure (newer versions) {'text': '...', 'confidence': 0.9, 'text_region': [...]}
                    if isinstance(line, dict):
                        text = str(line.get('text', ''))
                        confidence = float(line.get('confidence', 0.0))
                        bbox = line.get('text_region', [])
                        
                        if confidence >= min_confidence and text.strip():
                            bbox_int = [(int(p[0]), int(p[1])) for p in bbox if len(p) >= 2]
                            ocr_results.append(OCRResult(text=text, confidence=confidence, bbox=bbox_int))

                except (IndexError, TypeError, ValueError, KeyError) as e:
                    logger.debug(f"Skipping malformed OCR line: {e}")
                    continue
                    
            return ocr_results
            
        except Exception as e:
            logger.error(f"Internal PaddleOCR error: {e}")
            return []
    
    def _mock_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """Mock OCR for testing without PaddleOCR."""
        # Return empty or simulated results
        logger.debug("Using mock OCR")
        return []
    
    def extract_book_info(
        self,
        image: np.ndarray
    ) -> BookOCRResult:
        """
        Extract book information from cover image.
        
        Attempts to identify:
        - Title (usually the largest text)
        - Author (often below title)
        - Publisher (usually at bottom)
        - ISBN (if visible)
        
        Args:
            image: Book cover image
            
        Returns:
            BookOCRResult with extracted information
        """
        all_text = self.extract_text(image)
        
        if not all_text:
            return BookOCRResult(
                title=None,
                author=None,
                publisher=None,
                isbn=None,
                all_text=[],
                confidence=0.0
            )
            
        # Sort by Y position (top to bottom)
        sorted_by_y = sorted(all_text, key=lambda x: x.bounding_rect[1])
        
        # Sort by area (largest first) for title detection
        sorted_by_area = sorted(
            all_text,
            key=lambda x: x.bounding_rect[2] * x.bounding_rect[3],
            reverse=True
        )
        
        # Heuristics for book info extraction
        title = None
        author = None
        publisher = None
        isbn = None
        
        # Title is usually the largest text in upper half
        img_height = image.shape[0]
        upper_texts = [t for t in sorted_by_area if t.bounding_rect[1] < img_height * 0.6]
        if upper_texts:
            title = upper_texts[0].text
            
        # Author is often below title
        if title:
            title_y = next((t.bounding_rect[1] for t in all_text if t.text == title), 0)
            below_title = [t for t in sorted_by_y if t.bounding_rect[1] > title_y and t.text != title]
            if below_title:
                # Author often contains common patterns
                for t in below_title[:3]:
                    text_lower = t.text.lower()
                    if any(kw in text_lower for kw in ["tác giả", "author", "by"]):
                        author = t.text
                        break
                if not author and below_title:
                    author = below_title[0].text
                    
        # Publisher is usually at bottom
        lower_texts = [t for t in sorted_by_y if t.bounding_rect[1] > img_height * 0.7]
        for t in lower_texts:
            text_lower = t.text.lower()
            if any(kw in text_lower for kw in ["nhà xuất bản", "nxb", "publisher", "press"]):
                publisher = t.text
                break
                
        # ISBN pattern matching
        import re
        isbn_pattern = r'(?:ISBN[:\s-]*)?(\d{3}[-\s]?\d[-\s]?\d{3}[-\s]?\d{5}[-\s]?\d)'
        for t in all_text:
            match = re.search(isbn_pattern, t.text)
            if match:
                isbn = match.group(1).replace("-", "").replace(" ", "")
                break
                
        # Calculate overall confidence
        confidences = [t.confidence for t in all_text]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return BookOCRResult(
            title=self._clean_text(title),
            author=self._clean_text(author),
            publisher=self._clean_text(publisher),
            isbn=isbn,
            all_text=all_text,
            confidence=avg_confidence
        )
    
    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize extracted text."""
        if not text:
            return None
            
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common prefixes
        prefixes = ["tác giả:", "author:", "by ", "nxb:", "nhà xuất bản:"]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
                
        return text if text else None
    
    def draw_ocr_results(
        self,
        image: np.ndarray,
        results: List[OCRResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw OCR results on image."""
        output = image.copy()
        
        for result in results:
            # Draw polygon
            points = np.array(result.bbox, dtype=np.int32)
            cv2.polylines(output, [points], True, color, thickness)
            
            # Draw text
            x, y, _, _ = result.bounding_rect
            cv2.putText(
                output,
                result.text[:30],  # Truncate long text
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
        return output
