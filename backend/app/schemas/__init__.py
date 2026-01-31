"""
SmartLib Kiosk - Pydantic Schemas Package
Request/Response models for API
"""
from app.schemas.student import (
    StudentCreate,
    StudentUpdate,
    StudentResponse,
    StudentBorrowingInfoResponse
)
from app.schemas.book import (
    BookCreate,
    BookUpdate,
    BookResponse,
    BookIdentificationResponse
)
from app.schemas.transaction import (
    BorrowRequest,
    ReturnRequest,
    TransactionResponse,
    BorrowResponse,
    ReturnResponse
)
from app.schemas.auth import (
    FaceVerifyRequest,
    FaceVerifyResponse,
    FaceRegisterRequest,
    FaceRegisterResponse
)

__all__ = [
    "StudentCreate", "StudentUpdate", "StudentResponse", "StudentBorrowingInfoResponse",
    "BookCreate", "BookUpdate", "BookResponse", "BookIdentificationResponse",
    "BorrowRequest", "ReturnRequest", "TransactionResponse", "BorrowResponse", "ReturnResponse",
    "FaceVerifyRequest", "FaceVerifyResponse", "FaceRegisterRequest", "FaceRegisterResponse"
]
