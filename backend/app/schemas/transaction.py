"""
SmartLib Kiosk - Transaction Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum


class TransactionType(str, Enum):
    BORROW = "BORROW"
    RETURN = "RETURN"
    RENEWAL = "RENEWAL"


class TransactionStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"
    CANCELLED = "CANCELLED"


class BorrowRequest(BaseModel):
    """Schema for borrow book request."""
    student_id: str = Field(..., description="Student ID (verified by face)")
    book_id: str = Field(..., description="Book ID or barcode")
    kiosk_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "FPT20240001",
                "book_id": "978-0-596-52068-7",
                "kiosk_id": "KIOSK_001"
            }
        }


class ReturnRequest(BaseModel):
    """Schema for return book request."""
    student_id: str = Field(..., description="Student ID (verified by face)")
    book_id: str = Field(..., description="Book ID or barcode")
    kiosk_id: Optional[str] = None


class TransactionResponse(BaseModel):
    """Schema for transaction response."""
    transaction_id: str
    student_id: str
    book_id: str
    transaction_type: TransactionType
    borrow_date: datetime
    due_date: date
    return_date: Optional[datetime]
    days_overdue: int
    fine_amount: float
    status: TransactionStatus
    created_at: datetime
    
    class Config:
        from_attributes = True


class BorrowResponse(BaseModel):
    """Schema for borrow book response."""
    success: bool
    transaction_id: Optional[str]
    book_title: Optional[str]
    due_date: Optional[date]
    error_message: Optional[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
                "book_title": "Advanced AI",
                "due_date": "2026-02-04",
                "error_message": None
            }
        }


class ReturnResponse(BaseModel):
    """Schema for return book response."""
    success: bool
    transaction_id: Optional[str]
    book_title: Optional[str]
    days_overdue: int
    fine_amount: float
    error_message: Optional[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
                "book_title": "Advanced AI",
                "days_overdue": 5,
                "fine_amount": 50000.0,
                "error_message": None
            }
        }


class TransactionHistoryResponse(BaseModel):
    """Schema for transaction history."""
    total: int
    transactions: List[TransactionResponse]
