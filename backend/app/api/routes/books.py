"""
SmartLib Kiosk - Books API Routes
Book detection and information endpoints
"""
import numpy as np
import cv2
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from loguru import logger

from app.database import get_db
from app.models.book import Book
from app.schemas.book import BookCreate, BookUpdate, BookResponse, BookIdentificationResponse
from app.services.book_identification_service import BookIdentificationService

router = APIRouter(prefix="/books", tags=["Books"])

# Global service instance
_book_service: BookIdentificationService = None


def get_book_service() -> BookIdentificationService:
    """Get or create book identification service instance."""
    global _book_service
    if _book_service is None:
        _book_service = BookIdentificationService()
    return _book_service


@router.post("/detect", response_model=BookIdentificationResponse)
async def detect_book(
    image: UploadFile = File(..., description="Book image (JPEG/PNG)"),
    db: AsyncSession = Depends(get_db),
    book_service: BookIdentificationService = Depends(get_book_service)
):
    """
    Detect and identify a book from an image.
    
    **Pipeline:**
    1. Detect book using YOLOv8
    2. Read barcode using pyzbar
    3. Extract text using PaddleOCR
    4. Lookup book in database
    
    **Returns:**
    - Book information if found
    - Detection confidence scores
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Identify book
        result = await book_service.identify(img, db)
        
        return BookIdentificationResponse(
            success=result.success,
            book_id=result.book_id,
            title=result.title,
            author=result.author,
            barcode=result.barcode,
            status=result.status,
            detection_confidence=result.detection_confidence,
            barcode_confidence=result.barcode_confidence,
            ocr_confidence=result.ocr_confidence,
            error_message=result.error_message,
            processing_time_ms=result.processing_time_ms,
            book_exists=result.book_exists,
            is_available=result.is_available
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Book detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{barcode}", response_model=BookResponse)
async def get_book_by_barcode(
    barcode: str = Path(..., description="Book barcode or ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get book information by barcode or ID.
    """
    try:
        # Try by book_id
        stmt = select(Book).where(Book.book_id == barcode)
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()
        
        if not book:
            # Try by barcode
            stmt = select(Book).where(Book.barcode == barcode)
            result = await db.execute(stmt)
            book = result.scalar_one_or_none()
            
        if not book:
            raise HTTPException(status_code=404, detail="Book not found")
            
        return BookResponse.model_validate(book)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get book error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[BookResponse])
async def list_books(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all books with optional filtering.
    """
    try:
        stmt = select(Book)
        
        if status:
            stmt = stmt.where(Book.status == status)
            
        stmt = stmt.offset(offset).limit(limit)
        
        result = await db.execute(stmt)
        books = result.scalars().all()
        
        return [BookResponse.model_validate(book) for book in books]
        
    except Exception as e:
        logger.error(f"List books error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=BookResponse, status_code=201)
async def create_book(
    book: BookCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new book in the library catalog.
    """
    try:
        new_book = Book(
            book_id=book.book_id,
            title=book.title,
            author=book.author,
            isbn_13=book.isbn_13,
            barcode=book.barcode,
            publisher=book.publisher,
            publication_year=book.publication_year,
            language=book.language,
            subject_category=book.subject_category
        )
        
        db.add(new_book)
        await db.commit()
        await db.refresh(new_book)
        
        return BookResponse.model_validate(new_book)
        
    except Exception as e:
        logger.error(f"Create book error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
