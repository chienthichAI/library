"""
SmartLib Kiosk - AI Assistant Service
Handles LLM interactions via Ollama using qwen3.5:2b.
Updated for RAG pipeline with think-tag stripping.
"""
import os
import re
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger
from app.config import settings


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen 3.5 Instruct output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class LlmService:
    """
    Service to interact with local LLM via Ollama.
    Model: configurable via constructor/environment.
    Embedding: bge-m3 (via EmbeddingService, not here).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:2b",
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = 300.0  # Increased to 300s for slower environments
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

    async def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request to Ollama and return parsed response.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict or 1024,
            }
        }
        if format:
            payload["format"] = format

        try:
            client = await self._get_client()
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            raw = response.json()
            
            logger.debug(f"[Ollama Response RAW] {raw}")

            # Native Ollama response format: {"message": {"role": "assistant", "content": "..."}}
            data = raw
            
            # Strip think tags from response content
            if "message" in data and "content" in data["message"]:
                raw_content = data["message"]["content"] or ""
                stripped_content = strip_think_tags(raw_content)
                # If stripping would empty the reply, keep the original content.
                data["message"]["content"] = (
                    stripped_content if stripped_content else raw_content
                )

            return data

        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama at %s. "
                "Ensure Ollama is running: `ollama serve`", self.base_url
            )
            return {
                "error": "Ollama not available",
                "message": {
                    "content": "Xin lỗi, hệ thống AI đang không hoạt động. Vui lòng thử lại sau."
                },
            }
        except Exception as e:
            logger.error(f"LLM Chat Error ({type(e).__name__}): {e}")
            return {
                "error": str(e),
                "message": {
                    "content": "Xin lỗi, tôi đang gặp trục trặc kỹ thuật. Vui lòng thử lại sau."
                },
            }

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding via Ollama.
        NOTE: Use EmbeddingService (bge-m3) for RAG embedding instead.
        This is kept for backward compatibility only.
        """
        try:
            from app.services.embedding_service import embedding_service
            return await embedding_service.embed(text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def create_system_prompt(self, context_books: List[Any] = None) -> str:
        """
        Create a basic system prompt. 
        NOTE: For full RAG, ChatService builds the prompt dynamically.
        """
        prompt = (
            "Bạn là trợ lý ảo thông minh của Thư viện SmartLib. "
            "Nhiệm vụ của bạn là hỗ trợ sinh viên mượn/trả sách, tìm kiếm tài liệu "
            "và giải đáp các thắc mắc về quy định thư viện. "
            "Hãy trả lời một cách lịch sự, thân thiện và ngắn gọn bằng tiếng Việt."
        )

        if context_books:
            books_info = "\n".join(
                [f"- {b.title} (Tác giả: {b.author})" for b in context_books]
            )
            prompt += f"\n\nDanh sách sách liên quan:\n{books_info}"

        return prompt


# Two separate LLMs:
# - `ai_chat_assistant`: used for RAG generation (Vietnamese model)
# - `ai_intent_classifier`: used for JSON intent fallback (more JSON-stable model)
ai_chat_assistant = LlmService(
    base_url=settings.ollama_base_url,
    model=settings.llm_chat_model,
)

ai_intent_classifier = LlmService(
    base_url=settings.ollama_base_url,
    model=settings.llm_intent_model,
)

# Backward-compatible alias (some modules may still import `ai_assistant`)
ai_assistant = ai_chat_assistant

