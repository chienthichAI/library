from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.sql import func
import uuid
from app.database import Base

class ChatHistory(Base):
    """
    SQLAlchemy model for the chat_history table.
    Used for storing conversation context for RAG.
    """
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), index=True, nullable=False)
    student_id = Column(String(50), nullable=True, index=True)
    role = Column(String(50), nullable=False)  # 'human', 'ai', 'system'
    content = Column(Text, nullable=False)
    extra_metadata = Column(JSON, nullable=True)     # Stores intent, sources, metrics, etc.
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
