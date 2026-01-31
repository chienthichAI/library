"""
SmartLib Kiosk - Book Identification Service

Handles book detection, barcode reading, and OCR.
Uses YOLOv8 + pyzbar + PaddleOCR.
"""
import numpy as np
import cv2
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
import difflib

from app.models.book import Book, BookStatus
from app.ml.book_detector import BookDetector, BookDetectionResult, DetectedObject
from app.ml.barcode_reader import BarcodeReader, BarcodeResult
from app.ml.ocr_service import OCRService, BookOCRResult


@dataclass
class BookIdentificationResult:
    """Result of book identification."""
    success: bool
    book_id: Optional[str]
    title: Optional[str]
    author: Optional[str]
    barcode: Optional[str]
    status: Optional[str]
    detection_confidence: float
    barcode_confidence: float
    ocr_confidence: float
    error_message: Optional[str]
    processing_time_ms: float
    
    # Book database info if found
    book_exists: bool = False
    is_available: bool = False


class BookIdentificationService:
    """
    Book Identification Service using Computer Vision.
    
    Pipeline:
    1. Detect book in image (YOLOv8)
    2. Detect and read barcode (pyzbar)
    3. Extract text via OCR (PaddleOCR)
    4. Lookup book in database
    5. Return identification result
    """
    
    def __init__(
        self,
        book_detector: Optional[BookDetector] = None,
        barcode_reader: Optional[BarcodeReader] = None,
        ocr_service: Optional[OCRService] = None
    ):
        """
        Initialize book identification service.
        
        Args:
            book_detector: YOLOv8 book detector
            barcode_reader: Barcode reader
            ocr_service: OCR service for text extraction
        """
        self.book_detector = book_detector or BookDetector()
        self.barcode_reader = barcode_reader or BarcodeReader()
        self.ocr_service = ocr_service or OCRService()
        
    async def initialize(self) -> bool:
        """Initialize all ML models."""
        try:
            self.book_detector.initialize()
            self.ocr_service.initialize()
            logger.info("Book identification service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize book identification service: {e}")
            return False
    
    async def identify(
        self,
        image: np.ndarray,
        db: AsyncSession
    ) -> BookIdentificationResult:
        """
        Identify a book from an image.
        
        Args:
            image: BGR image from camera
            db: Database session
            
        Returns:
            BookIdentificationResult
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Detect book
            detection_result = self.book_detector.detect(image)
            
            book_image = image
            detection_confidence = 0.0
            
            if detection_result.has_book:
                book_detection = detection_result.primary_book
                detection_confidence = book_detection.confidence
                # Crop book region for further processing
                book_image = self.book_detector.crop_detection(image, book_detection)
            else:
                logger.info("No book detected by YOLO, falling back to full image processing.")
            
            # Step 2: Try barcode reading (primary and fastest method)
            barcode_result = await self._read_barcode(book_image)
            
            book = None
            book_id = None
            barcode_confidence = 0.0
            
            if barcode_result:
                book_id = barcode_result.data
                barcode_confidence = barcode_result.confidence
                # Step 3: Fast lookup
                book = await self._lookup_book(book_id, db)
                if book:
                    logger.info(f"Book found via barcode: {book.title}")
            
            # Step 4: OCR as backup (only if book not found via barcode)
            ocr_result = None
            if not book:
                logger.info("Starting OCR as backup...")
                ocr_result = await self._extract_text(book_image)
                
                if ocr_result and ocr_result.title:
                    # Step 5: Lookup via title
                    book = await self._search_book_by_title(ocr_result.title, db)
                    if book:
                        book_id = book.book_id
                        logger.info(f"Book found via OCR title: {book.title}")
            
            if not book:
                # Return partial result with OCR info
                return BookIdentificationResult(
                    success=False,
                    book_id=book_id,
                    title=ocr_result.title if ocr_result else None,
                    author=ocr_result.author if ocr_result else None,
                    barcode=book_id,
                    status=None,
                    detection_confidence=detection_confidence,
                    barcode_confidence=barcode_confidence,
                    ocr_confidence=ocr_result.confidence if ocr_result else 0.0,
                    error_message="Sách không có trong hệ thống",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    book_exists=False
                )
                
            return BookIdentificationResult(
                success=True,
                book_id=book.book_id,
                title=book.title,
                author=book.author,
                barcode=book.barcode,
                status=book.status.value,
                detection_confidence=detection_confidence,
                barcode_confidence=barcode_confidence,
                ocr_confidence=ocr_result.confidence if ocr_result else 0.0,
                error_message=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                book_exists=True,
                is_available=book.is_available
            )
            
        except Exception as e:
            logger.error(f"Book identification failed: {e}")
            return BookIdentificationResult(
                success=False,
                book_id=None,
                title=None,
                author=None,
                barcode=None,
                status=None,
                detection_confidence=0.0,
                barcode_confidence=0.0,
                ocr_confidence=0.0,
                error_message=f"Lỗi hệ thống: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _read_barcode(
        self,
        image: np.ndarray
    ) -> Optional[BarcodeResult]:
        """Ultra-robust barcode reading with multiple image enhancement passes."""
        # Pass 1: Raw image (Fastest)
        barcodes = self.barcode_reader.read(image)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Convert to gray for enhancements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pass 2: Brightness boost (Helpful for dark environments like yours)
        # Increase brightness and contrast
        bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        barcodes = self.barcode_reader.read(bright)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Pass 3: Adaptive Thresholding (Great for blurry/shiny barcodes)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        barcodes = self.barcode_reader.read(thresh)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Pass 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        barcodes = self.barcode_reader.read(cl1)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        return None

    def _pick_best_barcode(self, barcodes: List[BarcodeResult]) -> BarcodeResult:
        """Choose the most likely ISBN barcode."""
        for bc in barcodes:
            if bc.is_isbn:
                return bc
        return barcodes[0]
    
    async def _extract_text(
        self,
        image: np.ndarray
    ) -> Optional[BookOCRResult]:
        """Extract book info via OCR."""
        try:
            return self.ocr_service.extract_book_info(image)
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None
    
    async def _lookup_book(
        self,
        book_id: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """Lookup book by ID or barcode."""
        # Try by book_id
        stmt = select(Book).where(Book.book_id == book_id)
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()
        
        if book:
            return book
            
        # Try by barcode
        stmt = select(Book).where(Book.barcode == book_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _search_book_by_title(
        self,
        title: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """Search book by title with strict similarity check."""
        if not title or len(title) < 3:
            return None
            
        search_title = title.lower().strip()
        
        # 1. Broad search for candidates
        stmt = select(Book)
        result = await db.execute(stmt)
        all_books = result.scalars().all()
        
        best_match = None
        highest_ratio = 0.0
        
        for book in all_books:
            # Check similarity with book title
            ratio = difflib.SequenceMatcher(None, search_title, book.title.lower()).ratio()
            
            # Also check if it's a very clear substring
            if search_title in book.title.lower() or book.title.lower() in search_title:
                # Bonus for substring match
                ratio = max(ratio, 0.85)

            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = book
        
        # STRICT THRESHOLD: Only accept if 80% similar or higher
        if highest_ratio >= 0.8:
            logger.info(f"Book match found: {best_match.title} (Confidence: {highest_ratio:.2f})")
            return best_match
            
        logger.info(f"No strict match for title '{title}' (Best was {highest_ratio:.2f})")
        return None
    
    async def get_book_info(
        self,
        barcode: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """Get book information by barcode."""
        return await self._lookup_book(barcode, db)
