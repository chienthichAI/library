"""
SmartLib - Book Search Tool
Hybrid semantic + keyword book search backed by pgvector.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag_service import rag_service
from app.services.embedding_service import embedding_service
from app.config import settings


async def search_books(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    entities: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Hybrid book search using the unified search_books_hybrid in rag_service.
    """
    # Create the text search parameter
    search_text = query
    if entities:
        # If we have extracted explicit tags, enrich the search string potentially
        parts = []
        if entities.get("book_title"):
            parts.append(entities["book_title"])
        if entities.get("author"):
            parts.append(entities["author"])
        if entities.get("topic"):
            parts.append(entities["topic"])
        if parts:
            search_text = " ".join(parts)
            
    # Embed the query
    embedding = await embedding_service.embed(search_text)
    if not embedding:
        # If embedding fails (e.g. OpenAI network logic failing), use fallback
        return {
            "books": [],
            "context": "Không thể xử lý yêu cầu tìm kiếm lúc này.",
            "search_mode": "failed",
            "count": 0,
        }

    # Use the powerful SQL-based hybrid search
    books = await rag_service.search_books_hybrid(
        db=db,
        query_text=query,
        query_embedding=embedding,
        top_k=top_k,
        entities=entities,
        intent="book_search"
    )

    context = rag_service.format_books_for_context(books)
    
    return {
        "books": books,
        "context": context,
        "search_mode": "hybrid",
        "count": len(books),
    }
