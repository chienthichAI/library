"""
SmartLib Kiosk - Backend Services Package
"""
from app.services.authentication_service import AuthenticationService
from app.services.book_identification_service import BookIdentificationService
from app.services.transaction_service import TransactionService
from app.services.embedding_service import embedding_service
from app.services.rag_service import rag_service
from app.services.intent_service import intent_service
from app.services.chat_service import chat_service
from app.services.llm_service import ai_assistant

__all__ = [
    "AuthenticationService",
    "BookIdentificationService",
    "TransactionService",
    "embedding_service",
    "rag_service",
    "intent_service",
    "chat_service",
    "ai_assistant",
]
