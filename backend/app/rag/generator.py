import logging
import os
from typing import List, Optional

import httpx
import json

logger = logging.getLogger(__name__)


def _format_docs(docs):
    """Combine retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class GeneratorService:
    """
    Generation Layer: Combines context and query into Prompt, uses local LLM (Ollama) to generate answer.
    """
    
    def __init__(self, retriever, *, model: Optional[str] = None, base_url: Optional[str] = None):
        self.retriever = retriever
        self.model = model or os.getenv("OLLAMA_MODEL", "Qwen2-3B-RAG")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    async def generate_response(self, query: str):
        logger.info(f"Generating response for query: {query}")

        docs = await self.retriever.ainvoke(query)
        context = _format_docs(docs) if docs else ""

        system = (
            "Bạn là trợ lý AI của thư viện SmartLib. "
            "Hãy trả lời tiếng Việt, ngắn gọn, đúng trọng tâm. "
            "Nếu context không đủ, hãy nói bạn chưa chắc và gợi ý người dùng cung cấp thêm thông tin."
        )

        prompt = (
            f"{system}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST", 
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                full_content = ""
                import json
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message") or {}
                        content = msg.get("content") or chunk.get("response")
                        if content:
                            full_content += content
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                
                return full_content or "Xin lỗi, hiện tại tôi chưa thể tạo câu trả lời."
