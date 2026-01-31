"""
SmartLib Kiosk - Student Schemas
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class StudentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    GRADUATED = "GRADUATED"
    INACTIVE = "INACTIVE"


class StudentCreate(BaseModel):
    """Schema for creating a new student."""
    student_id: str = Field(..., min_length=1, max_length=20, description="Student ID")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=15)
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "FPT20240001",
                "full_name": "Nguyễn Văn A",
                "email": "a.nguyen@fpt.edu.vn",
                "phone": "0912345678"
            }
        }


class StudentUpdate(BaseModel):
    """Schema for updating student info."""
    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=15)
    status: Optional[StudentStatus] = None


class StudentResponse(BaseModel):
    """Schema for student response."""
    student_id: str
    full_name: str
    email: Optional[str]
    phone: Optional[str]
    status: StudentStatus
    fine_balance: float
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class BorrowedBook(BaseModel):
    """Schema for a borrowed book."""
    transaction_id: str
    book_id: str
    title: str
    borrow_date: datetime
    due_date: datetime
    days_left: int
    is_overdue: bool
    fine_amount: float


class StudentBorrowingInfoResponse(BaseModel):
    """Schema for student borrowing information."""
    student_id: str
    student_name: str
    currently_borrowed: int
    max_books: int
    fine_balance: float
    can_borrow: bool
    borrowed_books: List[BorrowedBook] = []
