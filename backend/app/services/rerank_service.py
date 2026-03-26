"""
SmartLib - Rerank Service
Uses thanhtantran/Vietnamese_Reranker (BGE-M3 base) to refine search results.
"""
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
from app.rag.reranker import RerankerService  # Use standard cross-encoder


class RerankingService:
    """
    Reranks documents from initial search (Top-K) to select most relevant results.
    Perfect for resolving complex queries in Vietnamese.
    """
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = RerankerService.get_reranker()
        return self._model

    async def rerank_books(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of book dictionary objects based on their descriptions/titles.
        
        Args:
            query: The user query string
            candidates: List of book items from initial (hybrid) search
            limit: Final number of results to return
            
        Returns:
            Reranked list of books
        """
        if not candidates:
            return []
            
        model = self._get_model()
        if not model:
            logger.warning("[Rerank] Reranker model not initialized, skipping rerank.")
            return candidates[:limit]

        try:
            # Prepare pairs: (query, document)
            # We combine title + description for the document part
            pairs = []
            for item in candidates:
                doc_text = f"Tên sách: {item['title']} - Tác giả: {item.get('author', 'Không rõ')} - Thể loại: {item.get('category', 'Chưa phân loại')} - Mô tả: {item.get('description', '')}"
                pairs.append([query, doc_text[:1024]]) # Truncate to save tokens

            # Predict scores — CrossEncoder.predict is synchronous, run in threadpool
            scores = await asyncio.to_thread(model.predict, pairs)

            # Combine scores with items
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            # Sort by rerank score DESC
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug(f"[Rerank] Top score: {candidates[0]['rerank_score'] if candidates else 'N/A'}")
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"[Rerank] Extraction failed: {e}")
            return candidates[:limit]

    async def rerank_policies(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Reranks policy chunks based on relevance."""
        if not chunks:
            return []
            
        model = self._get_model()
        if not model:
            return chunks[:limit]

        try:
            pairs = [[query, c["text"][:1024]] for c in chunks]
            scores = await asyncio.to_thread(model.predict, pairs)

            for i, score in enumerate(scores):
                chunks[i]["rerank_score"] = float(score)

            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return chunks[:limit]
        except Exception as e:
            logger.error(f"[Rerank] Policy rerank failed: {e}")
            return chunks[:limit]


# Singleton instance
reranking_service = RerankingService()
