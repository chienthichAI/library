"""
SmartLib Kiosk - Books API Routes
Book detection and catalog management endpoints
"""
import numpy as np
import cv2
import asyncio
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional
from loguru import logger

from app.database import get_db
from app.models.book import Book, BookStatus
from app.schemas.book import BookCreate, BookUpdate, BookResponse, BookIdentificationResponse
from app.services.book_identification_service import BookIdentificationService
from functools import lru_cache
from app.core.face_session import require_admin_session

router = APIRouter(prefix="/books", tags=["Books"])


@lru_cache()
def get_book_service() -> BookIdentificationService:
    """Get or create book identification service instance (cached singleton)."""
    return BookIdentificationService()


# ---------------------------------------------------------------------------
# POST /detect  — AI pipeline (YOLOv8 + barcode)
# ---------------------------------------------------------------------------
@router.post("/detect", response_model=BookIdentificationResponse)
async def detect_book(
    image: UploadFile = File(..., description="Book image (JPEG/PNG)"),
    db: AsyncSession = Depends(get_db),
    book_service: BookIdentificationService = Depends(get_book_service)
):
    """
    Detect and identify a book from a camera image.

    Pipeline:
    1. Detect book using YOLOv8
    2. Read barcode with 4-pass enhancement (pyzbar)
    3. Lookup book in database

    Returns book info + detection confidence scores.
    """
    try:
        contents = await image.read()

        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 10MB)")

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Định dạng ảnh không hợp lệ")

        result = await asyncio.wait_for(
            book_service.identify(img, db),
            timeout=15.0
        )

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
            is_available=result.is_available,
            subject_category=result.subject_category,
            description=result.description
        )

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Book detection timed out (>15s)")
        raise HTTPException(status_code=504, detail="Nhận diện sách quá thời gian. Thử lại.")
    except Exception as e:
        logger.error(f"Book detection error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi nhận diện sách")


# ---------------------------------------------------------------------------
# GET /  — list books (pagination + status filter + search)
# Must be declared BEFORE /{barcode} to prevent FastAPI route shadowing
# ---------------------------------------------------------------------------
@router.get("/", response_model=dict)
async def list_books(
    status: Optional[str] = Query(None, description="Filter by status (AVAILABLE, BORROWED, …)"),
    search: Optional[str] = Query(None, description="Search by title, author, or book_id"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all books with optional filtering.

    Returns:
    - total: total matching records
    - books: paginated list
    """
    try:
        stmt = select(Book)
        count_stmt = select(func.count()).select_from(Book)

        if status:
            try:
                status_enum = BookStatus(status.upper())
                stmt = stmt.where(Book.status == status_enum)
                count_stmt = count_stmt.where(Book.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Trạng thái không hợp lệ: {status}")

        if search:
            pattern = f"%{search.strip()}%"
            condition = or_(
                Book.title.ilike(pattern),
                Book.author.ilike(pattern),
                Book.book_id.ilike(pattern),
                Book.barcode.ilike(pattern)
            )
            stmt = stmt.where(condition)
            count_stmt = count_stmt.where(condition)

        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(Book.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        books = result.scalars().all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "books": [BookResponse.model_validate(b) for b in books]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List books error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy danh sách sách")


# ---------------------------------------------------------------------------
# GET /{barcode}  — get single book by barcode or book_id
# ---------------------------------------------------------------------------
@router.get("/{barcode}", response_model=BookResponse)
async def get_book_by_barcode(
    barcode: str = Path(..., description="Book barcode or book_id"),
    db: AsyncSession = Depends(get_db)
):
    """Get book information by barcode or book_id."""
    try:
        stmt = select(Book).where(Book.book_id == barcode)
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()

        if not book:
            stmt = select(Book).where(Book.barcode == barcode)
            result = await db.execute(stmt)
            book = result.scalar_one_or_none()

        if not book:
            raise HTTPException(status_code=404, detail="Không tìm thấy sách")

        return BookResponse.model_validate(book)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get book error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy thông tin sách")


# ---------------------------------------------------------------------------
# POST / — create new book
# ---------------------------------------------------------------------------
@router.post("/", response_model=BookResponse, status_code=201)
async def create_book(
    book: BookCreate,
    db: AsyncSession = Depends(get_db),
    _claims=Depends(require_admin_session)
):
    """Create a new book in the library catalog."""
    try:
        # Duplicate book_id check
        existing = (await db.execute(
            select(Book).where(Book.book_id == book.book_id)
        )).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail=f"Mã sách '{book.book_id}' đã tồn tại trong hệ thống")

        # Duplicate barcode check
        existing_barcode = (await db.execute(
            select(Book).where(Book.barcode == book.barcode)
        )).scalar_one_or_none()
        if existing_barcode:
            raise HTTPException(
                status_code=400,
                detail=f"Barcode '{book.barcode}' đã được dùng bởi sách: {existing_barcode.title}"
            )

        new_book = Book(
            book_id=book.book_id,
            title=book.title,
            author=book.author,
            isbn_13=book.isbn_13,
            barcode=book.barcode,
            publisher=book.publisher,
            publication_year=book.publication_year,
            language=book.language,
            subject_category=book.subject_category,
            smart_category=book.smart_category,
            ai_category=book.ai_category,
        )

        db.add(new_book)
        await db.commit()

        await db.refresh(new_book)

        logger.info(f"Book created: {new_book.title} (barcode={new_book.barcode})")
        return BookResponse.model_validate(new_book)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create book error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi tạo sách")


# ---------------------------------------------------------------------------
# PUT /{barcode} — update book metadata or status
# ---------------------------------------------------------------------------
@router.put("/{barcode}", response_model=BookResponse)
async def update_book(
    barcode: str = Path(..., description="Book barcode or book_id"),
    update: BookUpdate = ...,
    db: AsyncSession = Depends(get_db),
    _claims=Depends(require_admin_session)
):
    """
    Update book metadata (title, author) or status.

    **Status transitions allowed:**
    - AVAILABLE → DAMAGED / LOST (admin sets damage)
    - DAMAGED → AVAILABLE (repaired)
    - BORROWED → cannot be changed directly (use return flow)
    """
    try:
        stmt = select(Book).where(or_(Book.book_id == barcode, Book.barcode == barcode))
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()

        if not book:
            raise HTTPException(status_code=404, detail="Không tìm thấy sách")

        if update.title is not None:
            book.title = update.title
        if update.author is not None:
            book.author = update.author
        if update.subject_category is not None:
            book.subject_category = update.subject_category
        if update.smart_category is not None:
            book.smart_category = update.smart_category
        if update.ai_category is not None:
            book.ai_category = update.ai_category

        if update.status is not None:
            # Protect borrowed books from accidental status override
            if book.status == BookStatus.BORROWED and update.status == BookStatus.AVAILABLE:
                raise HTTPException(
                    status_code=409,
                    detail="Không thể đổi trạng thái sách đang mượn sang AVAILABLE. Dùng quy trình trả sách."
                )
            book.status = BookStatus(update.status.value)

        await db.commit()
        await db.refresh(book)

        logger.info(f"Book updated: {book.book_id}")
        return BookResponse.model_validate(book)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update book error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi cập nhật sách")


# ---------------------------------------------------------------------------
# DELETE /{barcode} — remove book from catalog
# ---------------------------------------------------------------------------
@router.delete("/{barcode}", status_code=204)
async def delete_book(
    barcode: str = Path(..., description="Book barcode or book_id"),
    db: AsyncSession = Depends(get_db),
    _claims=Depends(require_admin_session)
):
    """
    Delete a book from the catalog.

    **Blocked if:** book is currently BORROWED (must be returned first).
    """
    try:
        stmt = select(Book).where(or_(Book.book_id == barcode, Book.barcode == barcode))
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()

        if not book:
            raise HTTPException(status_code=404, detail="Không tìm thấy sách")

        if book.status == BookStatus.BORROWED:
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa sách đang được mượn. Yêu cầu trả sách trước."
            )

        await db.delete(book)
        await db.commit()
        logger.info(f"Book deleted: {book.book_id} - {book.title}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete book error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xóa sách")
