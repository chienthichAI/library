from datetime import datetime, timezone
import uuid
from typing import Optional, Any
from sqlalchemy import String, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from app.database import Base

class ChatHistory(Base):
    """
    Model for storing chat interactions between users and the AI.
    Used for conversation memory and semantic caching.
    """
    __tablename__ = "chat_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    student_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False) # 'human', 'ai', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 1024-dim to match AITeamVN/Vietnamese_Embedding (bge-m3 base)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1024), nullable=True)
    extra_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<ChatHistory(id={self.id}, role={self.role}, session={self.session_id})>"
