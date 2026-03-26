"""
SmartLib - Embedding Service
Handles text embedding using bge-m3 model via Ollama.
bge-m3 produces 1024-dim embeddings, ideal for multilingual text (Vietnamese/English).
"""
import httpx
from typing import Optional, List
from loguru import logger


class EmbeddingService:
    """
    Embedding service using BAAI/bge-m3 via Ollama.
    
    bge-m3 supports:
    - 1024-dim dense embeddings
    - Multilingual including Vietnamese
    - Very strong semantic understanding
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3"):
        self.base_url = base_url
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create reusable HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        """Close HTTP client on shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text (Vietnamese or English)
            
        Returns:
            1024-dim float vector or None on error
        """
        if not text or not text.strip():
            logger.warning("embed() called with empty text")
            return None

        payload = {"model": self.model, "prompt": text.strip()}

        try:
            client = await self._get_client()
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if embedding:
                logger.debug(f"Embedded text ({len(text)} chars) → {len(embedding)}-dim vector")
            return embedding
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama at %s. "
                "Ensure Ollama is running and bge-m3 is pulled.", self.base_url
            )
            return None
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in parallel.
        
        Args:
            texts: List of input strings
            
        Returns:
            List of embedding vectors (None for failed)
        """
        import asyncio
        return list(await asyncio.gather(*[self.embed(t) for t in texts]))


# Singleton instance
embedding_service = EmbeddingService()
