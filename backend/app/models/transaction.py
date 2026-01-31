"""
SmartLib Kiosk - Transaction Model
Stores borrow/return transactions
"""
from datetime import datetime, date
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, DateTime, Date, Enum, Integer, Float, Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.student import Student
    from app.models.book import Book


class TransactionType(str, enum.Enum):
    """Type of transaction."""
    BORROW = "BORROW"
    RETURN = "RETURN"
    RENEWAL = "RENEWAL"


class TransactionStatus(str, enum.Enum):
    """Transaction status."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"       # Book is borrowed
    COMPLETED = "COMPLETED" # Book has been returned
    OVERDUE = "OVERDUE"     # Past due date
    CANCELLED = "CANCELLED"


class Transaction(Base):
    """
    Transaction model - represents a borrow/return transaction.
    
    Tracks the complete lifecycle of a book loan:
    - Borrow date and due date
    - Return date (when applicable)
    - Fine calculation for overdue returns
    """
    __tablename__ = "transactions"
    
    # Primary key - UUID
    transaction_id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign keys
    student_id: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    book_id: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("books.book_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Transaction type
    transaction_type: Mapped[TransactionType] = mapped_column(
        String(20),
        nullable=False
    )
    
    # Dates
    borrow_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    return_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Overdue tracking
    days_overdue: Mapped[int] = mapped_column(Integer, default=0)
    
    # Fine information (VND)
    fine_amount: Mapped[float] = mapped_column(Float, default=0.0)
    fine_paid: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Status
    status: Mapped[TransactionStatus] = mapped_column(
        String(20),
        default=TransactionStatus.PENDING,
        index=True
    )
    
    # Kiosk that processed the transaction
    kiosk_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Snapshot path (captured image for audit)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="transactions")
    book: Mapped["Book"] = relationship("Book", back_populates="transactions")
    
    def __repr__(self) -> str:
        return f"<Transaction(id={self.transaction_id}, type={self.transaction_type}, status={self.status})>"
    
    @property
    def is_overdue(self) -> bool:
        """Check if transaction is overdue."""
        if self.return_date:
            return False
        return date.today() > self.due_date
    
    @property
    def is_active(self) -> bool:
        """Check if book is still borrowed (not returned)."""
        return self.status in [TransactionStatus.ACTIVE, TransactionStatus.OVERDUE]
    
    def calculate_fine(self, fine_per_day: float = 10000) -> float:
        """
        Calculate fine for overdue return.
        
        Args:
            fine_per_day: Fine amount per day (VND)
            
        Returns:
            Calculated fine amount
        """
        if not self.is_overdue and not self.return_date:
            return 0.0
        
        if self.return_date:
            return_d = self.return_date.date()
        else:
            return_d = date.today()
        
        days_over = (return_d - self.due_date).days
        if days_over > 0:
            self.days_overdue = days_over
            self.fine_amount = days_over * fine_per_day
        else:
            self.days_overdue = 0
            self.fine_amount = 0.0
            
        return self.fine_amount
