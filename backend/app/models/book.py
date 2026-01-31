"""
SmartLib Kiosk - Book Model
Stores book information for the library catalog
"""
from datetime import datetime, date
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, DateTime, Date, Enum, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.database import Base

if TYPE_CHECKING:
    from app.models.transaction import Transaction


class BookStatus(str, enum.Enum):
    """Book availability status."""
    AVAILABLE = "AVAILABLE"
    BORROWED = "BORROWED"
    RESERVED = "RESERVED"
    DAMAGED = "DAMAGED"
    LOST = "LOST"


class Book(Base):
    """
    Book model - represents a book in the library.
    
    Contains catalog information and availability status.
    Identified by barcode/ISBN for detection.
    """
    __tablename__ = "books"
    
    # Primary key - Book ID (internal ID or ISBN)
    book_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    
    # Book information
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # ISBN codes
    isbn_13: Mapped[Optional[str]] = mapped_column(String(13), nullable=True, index=True)
    isbn_10: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    
    # Barcode for scanning
    barcode: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    
    # Library classification
    call_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Publication info
    publisher: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    publication_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    edition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    language: Mapped[str] = mapped_column(String(20), default="vi")
    pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Category/Subject
    subject_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Cover image path (for display and AI training)
    cover_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Availability status
    status: Mapped[BookStatus] = mapped_column(
        Enum(BookStatus, name="book_status"),
        default=BookStatus.AVAILABLE,
        index=True
    )
    
    # Inventory dates
    acquisition_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_inventory_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    transactions: Mapped[List["Transaction"]] = relationship(
        "Transaction",
        back_populates="book",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Book(id={self.book_id}, title={self.title}, status={self.status})>"
    
    @property
    def is_available(self) -> bool:
        """Check if book is available for borrowing."""
        return self.status == BookStatus.AVAILABLE
    
    @property
    def is_borrowed(self) -> bool:
        """Check if book is currently borrowed."""
        return self.status == BookStatus.BORROWED
