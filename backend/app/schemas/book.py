"""
SmartLib Kiosk - Book Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date
from enum import Enum


class BookStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    BORROWED = "BORROWED"
    RESERVED = "RESERVED"
    DAMAGED = "DAMAGED"
    LOST = "LOST"


class BookCreate(BaseModel):
    """Schema for creating a new book."""
    book_id: str = Field(..., max_length=20, description="Book ID")
    title: str = Field(..., max_length=255, description="Book title")
    author: Optional[str] = Field(None, max_length=255)
    isbn_13: Optional[str] = Field(None, max_length=13)
    barcode: str = Field(..., max_length=50, description="Barcode for scanning")
    publisher: Optional[str] = Field(None, max_length=100)
    publication_year: Optional[int] = None
    language: str = Field(default="vi", max_length=20)
    subject_category: Optional[str] = Field(None, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "book_id": "978-0-596-52068-7",
                "title": "Advanced AI",
                "author": "Y. LeCun",
                "isbn_13": "9780596520687",
                "barcode": "9780596520687",
                "publisher": "O'Reilly",
                "publication_year": 2024,
                "language": "en",
                "subject_category": "Computer Science"
            }
        }


class BookUpdate(BaseModel):
    """Schema for updating book info."""
    title: Optional[str] = Field(None, max_length=255)
    author: Optional[str] = Field(None, max_length=255)
    status: Optional[BookStatus] = None


class BookResponse(BaseModel):
    """Schema for book response."""
    book_id: str
    title: str
    author: Optional[str]
    isbn_13: Optional[str]
    barcode: str
    publisher: Optional[str]
    publication_year: Optional[int]
    language: str
    status: BookStatus
    created_at: datetime
    
    class Config:
        from_attributes = True


class BookIdentificationResponse(BaseModel):
    """Schema for book identification result."""
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
    book_exists: bool
    is_available: bool
