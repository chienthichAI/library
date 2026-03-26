"""
SmartLib - Embedding Service
Handles text embedding using vietnamese-sbert for 768d vectors.
"""
from typing import Optional, List
from loguru import logger
from app.rag.embeddings import EmbeddingsService


class EmbeddingService:
    """
    Embedding service using keepitreal/vietnamese-sbert.
    Produces 768-dim vectors consistent with the database schema.
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = EmbeddingsService.get_embeddings()
        return self._model

    async def close(self):
        """No-op for local model."""
        pass

    async def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text (Vietnamese or English)
            
        Returns:
            768-dim float vector or None on error
        """
        if not text or not text.strip():
            return None

        try:
            model = self._get_model()
            # aembed_query returns a list or ndarray
            embedding = await model.aembed_query(text.strip())
            
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            
            return list(embedding)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in parallel.
        """
        import asyncio
        return list(await asyncio.gather(*[self.embed(t) for t in texts]))


# Singleton instance
embedding_service = EmbeddingService()

