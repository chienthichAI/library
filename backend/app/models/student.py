"""
SmartLib Kiosk - Student Model
Stores student information and links to face embeddings
"""
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, DateTime, Enum, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.database import Base

if TYPE_CHECKING:
    from app.models.transaction import Transaction
    from app.models.face_embedding import FaceEmbedding


class StudentStatus(str, enum.Enum):
    """Student account status."""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    GRADUATED = "GRADUATED"
    INACTIVE = "INACTIVE"


class UserRole(str, enum.Enum):
    """User role for access control."""
    STUDENT = "STUDENT"
    ADMIN = "ADMIN"


class Student(Base):
    """
    Student model - represents a library user.
    
    Contains personal information and links to:
    - Face embeddings for recognition
    - Transaction history
    - Fine balance
    """
    __tablename__ = "students"
    
    # Primary key - Student ID (e.g., "FPT20240001")
    student_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    
    # Personal information
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    
    # Profile image path (for display)
    profile_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Account status and role
    status: Mapped[StudentStatus] = mapped_column(
        String(20), 
        default=StudentStatus.ACTIVE
    )
    
    role: Mapped[UserRole] = mapped_column(
        String(20),
        default=UserRole.STUDENT
    )
    
    # Fine balance (VND)
    fine_balance: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    # Relationships
    face_embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        back_populates="student",
        cascade="all, delete-orphan"
    )
    
    transactions: Mapped[List["Transaction"]] = relationship(
        "Transaction",
        back_populates="student",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Student(id={self.student_id}, name={self.full_name}, status={self.status})>"
    
    @property
    def is_active(self) -> bool:
        """Check if student account is active."""
        return self.status == StudentStatus.ACTIVE
    
    @property
    def has_outstanding_fines(self) -> bool:
        """Check if student has unpaid fines."""
        return self.fine_balance > 0
