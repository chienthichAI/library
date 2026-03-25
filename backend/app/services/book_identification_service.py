"""
SmartLib Kiosk - Book Identification Service

Handles book detection and barcode reading.
Uses YOLOv8 + pyzbar.
"""
import numpy as np
import cv2
import asyncio
import time
import re
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from loguru import logger

from app.models.book import Book, BookStatus
from app.ml.book_detector import BookDetector, BookDetectionResult, DetectedObject
from app.ml.barcode_reader import BarcodeReader, BarcodeResult


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
    subject_category: Optional[str] = None
    description: Optional[str] = None


from app.core.ml_container import AIModels


class BookIdentificationService:
    """
    Book Identification Service using Computer Vision.
    
    Pipeline:
    1. Detect book in image (YOLOv8)
    2. Detect and read barcode (pyzbar)
    3. Lookup book in database
    4. Return identification result
    """
    
    def __init__(
        self,
        book_detector: Optional[BookDetector] = None,
        barcode_reader: Optional[BarcodeReader] = None
    ):
        """Initialize book identification service with pre-loaded components."""
        self.book_detector = book_detector or AIModels.book_detector or BookDetector()
        self.barcode_reader = barcode_reader or BarcodeReader()
        # OCR Service is no longer used for book identification as per user request
        self.ocr_service = None # or AIModels.ocr_service or OCRService()
        
    async def initialize(self) -> bool:
        """Models are now initialized via lifespan + AIModels container."""
        if not self.book_detector._initialized:
            self.book_detector.initialize()
        # OCR initialization skipped as it's disabled
        return True
    
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
                book_id = barcode_result.data.strip()
                barcode_confidence = barcode_result.confidence
                # Step 3: Fast lookup
                book = await self._lookup_book(book_id, db)
                if book:
                    logger.info(f"Book found via barcode: {book.title}")
            
            if not book:
                # Return partial result if barcode not found (OCR backup removed)
                return BookIdentificationResult(
                    success=False,
                    book_id=book_id,
                    title=None,
                    author=None,
                    barcode=book_id,
                    status=None,
                    detection_confidence=detection_confidence,
                    barcode_confidence=barcode_confidence,
                    ocr_confidence=0.0,
                    error_message="Không tìm thấy sách qua barcode hoặc sách chưa có trong hệ thống",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    book_exists=False,
                    subject_category=None,
                    description=None
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
                ocr_confidence=0.0,
                error_message=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                book_exists=True,
                is_available=book.is_available,
                subject_category=book.subject_category,
                description=book.description
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
        barcodes = await asyncio.to_thread(self.barcode_reader.read, image)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Convert to gray for enhancements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pass 2: Brightness boost (Helpful for dark environments like yours)
        # Increase brightness and contrast
        bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        barcodes = await asyncio.to_thread(self.barcode_reader.read, bright)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Pass 3: Adaptive Thresholding (Great for blurry/shiny barcodes)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        barcodes = await asyncio.to_thread(self.barcode_reader.read, thresh)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        # Pass 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        barcodes = await asyncio.to_thread(self.barcode_reader.read, cl1)
        if barcodes: return self._pick_best_barcode(barcodes)
        
        return None

    def _pick_best_barcode(self, barcodes: List[BarcodeResult]) -> BarcodeResult:
        """Choose the most likely ISBN barcode."""
        for bc in barcodes:
            if bc.is_isbn:
                return bc
        return barcodes[0]
    
    async def _lookup_book(
        self,
        book_id: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """Lookup book by ID/barcode/ISBN with normalization and ISBN fallback."""
        if not book_id:
            return None

        raw = book_id.strip()
        normalized = self._normalize_code(raw)
        candidates: Set[str] = {raw}
        if normalized:
            candidates.add(normalized)
            isbn13 = self._isbn10_to_isbn13(normalized)
            if isbn13:
                candidates.add(isbn13)

        # 1) Exact match against all identifiers
        stmt = select(Book).where(
            or_(
                Book.book_id.in_(list(candidates)),
                Book.barcode.in_(list(candidates)),
                Book.isbn_13.in_(list(candidates)),
                Book.isbn_10.in_(list(candidates)),
            )
        )
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()
        if book:
            return book

        # 2) Normalized match: ignore spaces/hyphens/case in DB identifiers
        if normalized:
            norm_book_id = func.replace(func.replace(func.upper(Book.book_id), "-", ""), " ", "")
            norm_barcode = func.replace(func.replace(func.upper(Book.barcode), "-", ""), " ", "")
            norm_isbn13 = func.replace(func.replace(func.upper(Book.isbn_13), "-", ""), " ", "")
            norm_isbn10 = func.replace(func.replace(func.upper(Book.isbn_10), "-", ""), " ", "")
            stmt = select(Book).where(
                or_(
                    norm_book_id == normalized,
                    norm_barcode == normalized,
                    norm_isbn13 == normalized,
                    norm_isbn10 == normalized,
                )
            ).limit(1)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

        return None

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Keep only alphanumeric characters and uppercase for robust matching."""
        return re.sub(r"[^0-9A-Za-z]", "", code or "").upper()

    @staticmethod
    def _isbn10_to_isbn13(isbn10: str) -> Optional[str]:
        """Convert ISBN-10 to ISBN-13 when possible."""
        clean = (isbn10 or "").upper()
        if not re.fullmatch(r"\d{9}[\dX]", clean):
            return None

        body = "978" + clean[:9]
        total = 0
        for idx, ch in enumerate(body):
            weight = 1 if idx % 2 == 0 else 3
            total += int(ch) * weight
        check = (10 - (total % 10)) % 10
        return f"{body}{check}"
    
    async def _search_book_by_title(
        self,
        title: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """
        Search book by title with high-performance fuzzy matching in Postgres.
        Uses pg_trgm extension for sub-second retrieval even with 10k+ books.
        """
        if not title or len(title) < 3:
            return None
            
        search_title = title.strip()
        
        # SQL Similarity Query (Layer 4 Optimized)
        # We use a threshold of 0.4 for the trigram match, then pick the best
        from sqlalchemy import func, or_
        
        stmt = select(Book).where(
            or_(
                Book.title.bool_op('%')(search_title), # Trigram similarity
                Book.title.ilike(f"%{search_title}%")  # Substring match
            )
        ).order_by(
            func.similarity(Book.title, search_title).desc()
        ).limit(1)
        
        try:
            result = await db.execute(stmt)
            book = result.scalar_one_or_none()
            
            if book:
                # Double check similarity score if needed for strictness
                # Note: SequenceMatcher ratio is different from SQL similarity
                # but SQL similarity is generally more reliable for library searches
                logger.info(f"Book match found via SQL similarity: {book.title}")
                return book
                
        except Exception as e:
            logger.error(f"SQL fuzzy search failed: {e}. Falling back to basic search.")
            # Emergency fallback to simple ILIKE (P4-04: escape wildcards)
            safe_title = search_title.replace("%", "\\%").replace("_", "\\_")
            stmt = select(Book).where(Book.title.ilike(f"%{safe_title}%")).limit(1)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()
            
        logger.info(f"No match for title '{title}' in database.")
        return None
    
    async def get_book_info(
        self,
        barcode: str,
        db: AsyncSession
    ) -> Optional[Book]:
        """Get book information by barcode."""
        return await self._lookup_book(barcode, db)
