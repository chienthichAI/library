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
    Hybrid book search: semantic first, keyword fallback.
    
    Args:
        db: Database session
        query: User's raw query
        top_k: Number of results
        entities: Extracted entities from intent detection
        
    Returns:
        Dict with books list and context string
    """
    books: list = []
    search_mode = "none"

    # Determine search query from entities if available
    search_text = query
    if entities:
        parts = []
        if entities.get("book_title"):
            parts.append(entities["book_title"])
        if entities.get("author"):
            parts.append(entities["author"])
        if entities.get("topic"):
            parts.append(entities["topic"])
        if parts:
            search_text = " ".join(parts)

    # Hybrid retrieval (semantic + keyword) with lightweight merge/rerank.
    # This typically improves precision vs "semantic only then fallback".
    semantic_books: list = []
    keyword_books: list = []

    # Compute semantic embeddings (optional; semantic can fail if books.embedding isn't ready)
    embedding = await embedding_service.embed(search_text)
    if embedding:
        semantic_books = await rag_service.search_books_semantic(db, embedding, top_k=max(10, top_k * 2))
        if semantic_books:
            # Filter out very low similarity results (below 0.3)
            semantic_books = [b for b in semantic_books if b.get("similarity", 0) >= 0.3]

    # Keyword retrieval always (cheap) for hybrid rerank
    keyword_books = await rag_service.search_books_keyword(db, search_text, top_k=max(10, top_k * 2))

    if semantic_books and keyword_books:
        alpha = settings.hybrid_semantic_weight  # semantic weight (from config or .env)
        merged: Dict[str, Dict[str, Any]] = {}

        for b in semantic_books:
            merged[b["book_id"]] = {
                "book": b,
                "sem": float(b.get("similarity", 0.0) or 0.0),
                "kw": 0.0,
            }

        for b in keyword_books:
            bid = b["book_id"]
            if bid not in merged:
                merged[bid] = {
                    "book": b,
                    "sem": 0.0,
                    "kw": float(b.get("similarity", 0.0) or 0.0),
                }
            else:
                merged[bid]["kw"] = float(b.get("similarity", 0.0) or 0.0)

        # Merge score and rerank
        scored_books = []
        for item in merged.values():
            score = alpha * item["sem"] + (1.0 - alpha) * item["kw"]
            book = item["book"]
            book["similarity"] = round(score, 3)  # reuse similarity field for ranking/context
            scored_books.append(book)

        scored_books.sort(key=lambda x: float(x.get("similarity", 0.0) or 0.0), reverse=True)
        books = scored_books[:top_k]
        search_mode = "hybrid"
        logger.info(f"Hybrid search found {len(books)} books for '{search_text}'")
    elif semantic_books:
        books = semantic_books[:top_k]
        search_mode = "semantic"
        logger.info(f"Semantic search found {len(books)} books for '{search_text}'")
    else:
        books = keyword_books[:top_k]
        search_mode = "keyword"
        logger.info(f"Keyword search found {len(books)} books for '{search_text}'")

    context = rag_service.format_books_for_context(books)
    return {
        "books": books,
        "context": context,
        "search_mode": search_mode,
        "count": len(books),
    }
