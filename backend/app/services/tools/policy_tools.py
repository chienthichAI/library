"""
SmartLib - Policy Query Tool
Retrieve relevant library policy chunks using vector similarity.
"""
from typing import Dict, Any, Optional, List
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding_service import embedding_service
from app.services.rag_service import rag_service


async def query_policy(
    db: AsyncSession,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve relevant policy chunks for a user's question.

    Returns:
        Dict with:
          - found (bool): True only when DB returned high-confidence chunks
          - chunks (list): Raw chunk dicts
          - context (str): Formatted text to inject into TOOL CONTEXT,
                           or empty string when nothing relevant was found.
    """
    try:
        # Generate embedding for the query
        embedding = await embedding_service.embed(query)

        if not embedding:
            logger.warning("Could not embed policy query, returning empty.")
            return {
                "found": False,
                "chunks": [],
                "context": "",
            }

        # Search policy_chunks table (returns reranked results)
        chunks = await rag_service.search_policy(db, query, embedding, top_k=top_k)

        if not chunks:
            logger.info(f"[Policy] No chunks returned for query: '{query[:60]}'")
            return {
                "found": False,
                "chunks": [],
                "context": "",
            }

        # Filter chunks by minimum similarity — only include truly relevant ones.
        # Using 0.40 to avoid noise; if reranker is active the rerank_score matters more.
        MIN_SIM = 0.40
        relevant_chunks = [
            c for c in chunks
            if c.get("rerank_score", c.get("similarity", 0)) >= MIN_SIM
        ]

        if not relevant_chunks:
            logger.info(
                f"[Policy] {len(chunks)} chunks found but all below similarity threshold {MIN_SIM}. "
                f"Top score: {chunks[0].get('rerank_score', chunks[0].get('similarity', 'N/A'))}"
            )
            return {
                "found": False,
                "chunks": [],
                "context": "",
            }

        context = rag_service.format_policy_for_context(relevant_chunks)

        logger.info(f"[Policy] Returning {len(relevant_chunks)} relevant chunks for query: '{query[:60]}'")
        return {
            "found": True,
            "chunks": relevant_chunks,
            "context": context,
        }

    except Exception as e:
        logger.error(f"Policy query failed: {e}")
        return {
            "found": False,
            "chunks": [],
            "context": "",
        }
