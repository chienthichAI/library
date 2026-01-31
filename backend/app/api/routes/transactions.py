"""
SmartLib Kiosk - Transaction API Routes
Book borrowing and returning endpoints
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

router = APIRouter(prefix="/transactions", tags=["Transactions"])

# Global service instance
_transaction_service: TransactionService = None


def get_transaction_service() -> TransactionService:
    """Get or create transaction service instance."""
    global _transaction_service
    if _transaction_service is None:
        _transaction_service = TransactionService()
    return _transaction_service


@router.post("/borrow", response_model=BorrowResponse)
async def borrow_book(
    request: BorrowRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Borrow a book from the library.
    
    **Requirements:**
    - Student must be verified (via face recognition)
    - Book must be available
    - Student must not exceed borrowing limit
    - Student must have no outstanding fines
    
    **Returns:**
    - Transaction ID
    - Due date
    """
    try:
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
        
    except Exception as e:
        logger.error(f"Borrow book error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/return", response_model=ReturnResponse)
async def return_book(
    request: ReturnRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Return a borrowed book to the library.
    
    **Process:**
    - Validates active borrow transaction exists
    - Calculates overdue fine if applicable
    - Updates book status to available
    - Records return transaction
    
    **Returns:**
    - Days overdue (if any)
    - Fine amount (if applicable)
    """
    try:
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
        
    except Exception as e:
        logger.error(f"Return book error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{student_id}", response_model=TransactionHistoryResponse)
async def get_transaction_history(
    student_id: str = Path(..., description="Student ID"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Get transaction history for a student.
    """
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
        
    except Exception as e:
        logger.error(f"Get transaction history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-return")
async def validate_return(
    request: ReturnRequest,
    db: AsyncSession = Depends(get_db),
    transaction_service: TransactionService = Depends(get_transaction_service)
):
    """
    Validate if a student can return a specific book.
    
    Use this to check before attempting return to avoid race conditions.
    
    **Returns:**
    - can_return: Whether the return is valid
    - error_message: Reason if cannot return
    - transaction_id: Active transaction ID if found
    """
    try:
        # Check if student has active transaction for this book
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
                "estimated_fine": transaction.calculate_fine(transaction_service.fine_per_day)
            }
        else:
            return {
                "can_return": False,
                "error_message": "Không tìm thấy giao dịch mượn sách này. Sinh viên chưa mượn cuốn sách này.",
                "transaction_id": None
            }
            
    except Exception as e:
        logger.error(f"Validate return error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
