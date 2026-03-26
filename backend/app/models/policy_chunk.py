from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from app.database import Base

class PolicyChunk(Base):
    """
    SQLAlchemy model for the policy_chunks table.
    Stores chunked text of library_policy.md and bge-m3 embeddings.
    """
    __tablename__ = "policy_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # bge-m3 produces 1024-dim vectors
    chunk_index = Column(Integer, nullable=False)
    section_title = Column(String(255), nullable=True)
    source = Column(String(100), default="library_policy", index=True)
