"""
SmartLib Kiosk - AI Assistant Service
Handles natural language interactions and smart book search using LLM (Qwen 2.5).
"""
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger
import json

class LlmService:
    """
    Service to interact with local LLM via Ollama.
    Updated to use Qwen 3 (8B Instruct) for superior reasoning and consultation.
    Uses a persistent httpx.AsyncClient with connection pooling.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url
        self.model = model
        self.timeout = 90.0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        """Close the persistent HTTP client. Call during app shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Send a chat request to Ollama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        try:
            client = await self._get_client()
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM Chat failed: {e}")
            return {"error": str(e), "message": {"content": "Xin lỗi, tôi đang gặp trục trặc kỹ thuật. Vui lòng thử lại sau."}}

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get text embedding using Ollama (if embedding model exists)."""
        payload = {
            "model": "qwen3-embedding:0.6b",
            "prompt": text
        }
        
        try:
            client = await self._get_client()
            response = await client.post("/api/embeddings", json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json().get("embedding")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def create_system_prompt(self, context_books: List[Any] = None) -> str:
        """Create a system prompt tailored for a Library Kiosk Assistant."""
        prompt = (
            "Bạn là trợ lý ảo thông minh của Thư viện SmartLib. "
            "Nhiệm vụ của bạn là hỗ trợ sinh viên mượn/trả sách, tìm kiếm tài liệu và giải đáp các thắc mắc về nội dung sách. "
            "Hãy trả lời một cách lịch sự, thân thiện và ngắn gọn (vì sinh viên đang đứng ở Kiosk)."
        )
        
        if context_books:
            books_info = "\n".join([f"- {b.title} (Tác giả: {b.author})" for b in context_books])
            prompt += f"\n\nDanh sách sách liên quan bạn có thể gợi ý:\n{books_info}"
            
        return prompt

ai_assistant = LlmService()
