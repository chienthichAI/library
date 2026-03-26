"""
SmartLib Kiosk - Barcode Reader

Reads 1D and 2D barcodes from book images.
Supports:
- ISBN-13 (EAN-13)
- ISBN-10
- Code128, Code39
- QR codes
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except Exception as e:
    PYZBAR_AVAILABLE = False
    logger.warning(f"pyzbar could not be loaded (DLLs might be missing): {e}")


@dataclass
class BarcodeResult:
    """Result of barcode decoding."""
    data: str
    barcode_type: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    
    @property
    def is_isbn(self) -> bool:
        """Check if barcode is an ISBN."""
        # Accepts EAN13 (common for ISBN) and strings starting with 978/979
        return (
            self.barcode_type in ["EAN13", "EAN-13", "UPC", "UPCA", "ISBN"] and 
            (self.data.startswith("978") or self.data.startswith("979") or len(self.data) >= 10)
        )
    
    @property
    def isbn_13(self) -> Optional[str]:
        """Get ISBN-13 format."""
        if self.is_isbn:
            return self.data
        return None


class BarcodeReader:
    """
    Barcode reader using pyzbar/ZXing with image preprocessing.
    Falls back to OpenCV BarcodeDetector if pyzbar DLLs are missing.
    
    Features:
    - Multi-format support (1D and 2D)
    - Image preprocessing (CLAHE, Otsu, Adaptive, Sharpening)
    - Confidence scoring based on image quality
    """
    
    def __init__(self, supported_types: Optional[List[str]] = None):
        """
        Initialize barcode reader.
        
        Args:
            supported_types: List of barcode types to detect
                            (None = all types)
        """
        self.supported_types = supported_types or [
            "EAN13", "EAN8", "CODE128", "CODE39", "QRCODE"
        ]
        self._symbol_types = self._get_symbol_types()
        
        # Fallback detector
        self.cv2_barcode_detector = None
        if not PYZBAR_AVAILABLE:
            try:
                self.cv2_barcode_detector = cv2.barcode.BarcodeDetector()
                logger.info("Using OpenCV BarcodeDetector as fallback.")
            except AttributeError:
                logger.warning("cv2.barcode not available. Barcode reading will be disabled.")
        
    def _get_symbol_types(self) -> Optional[List]:
        """Get pyzbar symbol types for filtering."""
        if not PYZBAR_AVAILABLE:
            return None
            
        type_map = {
            "EAN13": ZBarSymbol.EAN13,
            "EAN8": ZBarSymbol.EAN8,
            "CODE128": ZBarSymbol.CODE128,
            "CODE39": ZBarSymbol.CODE39,
            "QRCODE": ZBarSymbol.QRCODE,
            "I25": ZBarSymbol.I25,
        }
        
        symbols = []
        for t in self.supported_types:
            if t.upper() in type_map:
                symbols.append(type_map[t.upper()])
                
        return symbols if symbols else None
    
    def read(self, image: np.ndarray) -> List[BarcodeResult]:
        """
        Read barcodes from an image.
        
        Args:
            image: BGR or grayscale image
            
        Returns:
            List of BarcodeResult objects
        """
        if image is None or image.size == 0:
            return []

        if not PYZBAR_AVAILABLE:
            if self.cv2_barcode_detector:
                return self._read_with_cv2(image)
            logger.warning("No barcode engines available.")
            return []
            
        results = []
        
        # Try multiple preprocessing methods
        preprocessed_images = self._preprocess(image)
        
        for processed in preprocessed_images:
            try:
                barcodes = pyzbar.decode(
                    processed,
                    symbols=self._symbol_types
                )
                
                for barcode in barcodes:
                    # Extract data
                    data = barcode.data.decode('utf-8', errors='ignore').strip()
                    barcode_type = barcode.type
                    
                    # Get bounding box
                    rect = barcode.rect
                    bbox = (rect.left, rect.top, rect.width, rect.height)
                    
                    # Calculate confidence based on quality
                    confidence = self._calculate_confidence(barcode, processed)
                    
                    result = BarcodeResult(
                        data=data,
                        barcode_type=barcode_type,
                        bbox=bbox,
                        confidence=confidence
                    )
                    
                    # Avoid duplicates or keep best confidence
                    existing_idx = next((i for i, r in enumerate(results) if r.data == data), None)
                    if existing_idx is None:
                        results.append(result)
                    elif result.confidence > results[existing_idx].confidence:
                        results[existing_idx] = result
                        
            except Exception as e:
                logger.debug(f"Barcode decode error: {e}")
                continue
                
        return results
    
    def _read_with_cv2(self, image: np.ndarray) -> List[BarcodeResult]:
        """Fallback reading using OpenCV BarcodeDetector."""
        if not self.cv2_barcode_detector:
            return []
            
        results = []
        # BarcodeDetector.detectAndDecode returns (ret, decoded_info, points, type)
        try:
            ret, decoded_info, points, types = self.cv2_barcode_detector.detectAndDecode(image)
            
            if ret:
                for i in range(len(decoded_info)):
                    data = decoded_info[i].strip()
                    if not data:
                        continue
                        
                    # OpenCV points are (4, 2)
                    pts = points[i].astype(int)
                    x = int(np.min(pts[:, 0]))
                    y = int(np.min(pts[:, 1]))
                    w = int(np.max(pts[:, 0]) - x)
                    h = int(np.max(pts[:, 1]) - y)
                    
                    results.append(BarcodeResult(
                        data=data,
                        barcode_type=str(types[i]),
                        bbox=(x, y, w, h),
                        confidence=0.9  # Fixed confidence for CV2
                    ))
        except Exception as e:
            logger.debug(f"OpenCV barcode decode error: {e}")
            
        return results

    def _preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple preprocessing methods for better detection.
        """
        preprocessed = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Original grayscale
        preprocessed.append(gray)
        
        # 2. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        preprocessed.append(enhanced)
        
        # 3. Binary threshold (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(binary)
        
        # 4. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        preprocessed.append(adaptive)
        
        # 5. Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        preprocessed.append(sharpened)
        
        return preprocessed
    
    def _calculate_confidence(self, barcode, image: np.ndarray) -> float:
        """Calculate confidence score based on detection quality."""
        # Base confidence
        confidence = 0.85
        
        # Adjust based on barcode size relative to image
        rect = barcode.rect
        image_area = image.shape[0] * image.shape[1]
        barcode_area = rect.width * rect.height
        size_ratio = barcode_area / image_area
        
        if size_ratio > 0.01:  # Barcode is reasonably sized
            confidence += 0.1
        if size_ratio > 0.05:
            confidence += 0.05
            
        return min(confidence, 1.0)
    
    def read_isbn(self, image: np.ndarray) -> Optional[str]:
        """
        Read ISBN from image.
        """
        results = self.read(image)
        
        for result in results:
            if result.is_isbn:
                return result.isbn_13
                
        # Check for any EAN-13 that might be ISBN
        for result in results:
            if result.barcode_type in ["EAN13", "EAN-13", "UPC", "UPCA"]:
                return result.data
                
        return None
    
    def draw_barcodes(
        self,
        image: np.ndarray,
        results: List[BarcodeResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw barcode detections on image."""
        output = image.copy()
        
        for result in results:
            x, y, w, h = result.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{result.barcode_type}: {result.data}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                       
        return output
