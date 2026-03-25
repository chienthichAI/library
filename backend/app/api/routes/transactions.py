"""
SmartLib Kiosk - Transaction API Routes
Book borrowing, returning, renewal, and statistics endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from loguru import logger

from app.database import get_db
from app.schemas.transaction import (
    BorrowRequest, BorrowResponse,
    ReturnRequest, ReturnResponse,
    TransactionResponse, TransactionHistoryResponse
)
from app.services.transaction_service import TransactionService
from functools import lru_cache
from app.core.face_session import decode_face_session_token

router = APIRouter(prefix="/transactions", tags=["Transactions"])

def _validate_face_token(verification_token: str, student_id: str):
    try:
        claims = decode_face_session_token(verification_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Phiên xác thực khuôn mặt không hợp lệ hoặc đã hết hạn")
    token_student = claims.get("student_id")
    if token_student != student_id:
        raise HTTPException(status_code=403, detail="Token xác thực không khớp sinh viên")
    return claims


@lru_cache()
def get_transaction_service() -> TransactionService:
    """Get or create transaction service instance (cached singleton)."""
    return TransactionService()


# ---------------------------------------------------------------------------
# POST /borrow
# ---------------------------------------------------------------------------
@router.post("/borrow", response_model=BorrowResponse)
async def borrow_book(
    request: BorrowRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Borrow a book from the library.

    Requirements:
    - Student verified (via face recognition)
    - Book must be AVAILABLE
    - Student must not exceed borrowing limit
    - Student must have zero outstanding fines

    Returns transaction ID and due date.
    Note: Idempotent — if student already has the book, returns existing transaction.
    """
    try:
        _validate_face_token(request.verification_token, request.student_id)
        result = await transaction_service.borrow_book(
            student_id=request.student_id,
            book_id=request.book_id,
            db=db,
            kiosk_id=request.kiosk_id
        )

        return BorrowResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            book_title=result.book_title,
            due_date=result.due_date,
            error_message=result.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Borrow book error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý mượn sách")


# ---------------------------------------------------------------------------
# POST /return
# ---------------------------------------------------------------------------
@router.post("/return", response_model=ReturnResponse)
async def return_book(
    request: ReturnRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Return a borrowed book to the library.

    Process:
    - Validates active borrow transaction exists
    - Calculates overdue fine if applicable
    - Updates book status to AVAILABLE
    - Records return date and finalises transaction

    Returns days overdue and fine amount.
    """
    try:
        _validate_face_token(request.verification_token, request.student_id)
        result = await transaction_service.return_book(
            student_id=request.student_id,
            book_id=request.book_id,
            db=db,
            kiosk_id=request.kiosk_id
        )

        return ReturnResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            book_title=result.book_title,
            days_overdue=result.days_overdue,
            fine_amount=result.fine_amount,
            error_message=result.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Return book error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý trả sách")


# ---------------------------------------------------------------------------
# POST /renew — extend due date
# ---------------------------------------------------------------------------
@router.post("/renew", response_model=BorrowResponse)
async def renew_book(
    request: BorrowRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Renew an active borrow (extend due date by `max_borrow_days`).

    Rules:
    - Allowed only once per borrow cycle
    - Student must have zero outstanding fines
    - Book must still be actively borrowed by this student
    """
    try:
        _validate_face_token(request.verification_token, request.student_id)
        result = await transaction_service.renew_book(
            student_id=request.student_id,
            book_id=request.book_id,
            db=db,
            kiosk_id=request.kiosk_id
        )

        return BorrowResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            book_title=result.book_title,
            due_date=result.due_date,
            error_message=result.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Renew book error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý gia hạn sách")


# ---------------------------------------------------------------------------
# POST /validate-return — 2-phase check before actual return
# ---------------------------------------------------------------------------
@router.post("/validate-return")
async def validate_return(
    request: ReturnRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Validate if a student can return a specific book (pre-flight check).

    Use this before the actual /return call to avoid race conditions and
    present the fine preview to the student before they confirm.

    Returns:
    - can_return: whether the return is valid
    - estimated_fine: fine amount if overdue
    - transaction_id: active transaction ID if found
    """
    try:
        _validate_face_token(request.verification_token, request.student_id)
        transaction = await transaction_service._find_active_transaction(
            request.student_id, request.book_id, db
        )

        if transaction:
            return {
                "can_return": True,
                "transaction_id": transaction.transaction_id,
                "book_id": transaction.book_id,
                "borrow_date": transaction.borrow_date,
                "due_date": transaction.due_date,
                "is_overdue": transaction.is_overdue,
                "days_overdue": transaction.days_overdue,
                "estimated_fine": transaction.calculate_fine(transaction_service.fine_per_day)
            }
        else:
            return {
                "can_return": False,
                "error_message": "Không tìm thấy giao dịch mượn sách này. Sinh viên chưa mượn cuốn sách này.",
                "transaction_id": None
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate return error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi kiểm tra trả sách")


# ---------------------------------------------------------------------------
# GET /history/{student_id} — paginated transaction history
# ---------------------------------------------------------------------------
@router.get("/history/{student_id}", response_model=TransactionHistoryResponse)
async def get_transaction_history(
    student_id: str = Path(..., description="Student ID"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """Get transaction history for a student (paginated)."""
    try:
        transactions, total = await transaction_service.get_transaction_history(
            student_id=student_id,
            db=db,
            limit=limit,
            offset=offset
        )

        return TransactionHistoryResponse(
            total=total,
            transactions=[TransactionResponse.model_validate(t) for t in transactions]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get transaction history error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy lịch sử giao dịch")


# ---------------------------------------------------------------------------
# GET /stats/overdue — admin dashboard overdue stats
# ---------------------------------------------------------------------------
@router.get("/stats/overdue")
async def get_overdue_stats(
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Get system-wide overdue statistics.

    Returns:
    - total_overdue: number of overdue active transactions
    - total_fine_pending: total uncollected fine (VND)
    """
    try:
        return await transaction_service.get_overdue_summary(db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get overdue stats error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy thống kê quá hạn")
