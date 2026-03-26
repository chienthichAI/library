"""
SmartLib Kiosk - AI Assistant Routes (RAG-powered)
Full RAG chatbot endpoints using chat_service orchestration.
"""
import uuid
import httpx
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from loguru import logger

from app.services.chat_service import chat_service
from app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings


router = APIRouter(prefix="/ai", tags=["AI Assistant"])


# === Request / Response Schemas ===

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity. Auto-generated if not provided."
    )
    student_id: Optional[str] = Field(
        default=None,
        description="Authenticated student ID (e.g., SE001). Required for debt checks."
    )


class ChatResponse(BaseModel):
    reply: str
    intent: str
    entities: Dict[str, Any]
    sources: List[Dict[str, Any]]
    session_id: str
    success: bool


class HistoryMessage(BaseModel):
    id: str
    role: str  # 'human' or 'ai'
    content: str
    created_at: str


# === Endpoints ===

@router.get("/health", tags=["AI Assistant"])
async def ai_health_check():
    """
    Health check for AI services (Ollama LLM + vietnamese-sbert embedding).
    
    Returns the status of each AI backend service independently.
    Frontend can use this to show a warning when the chatbot is unavailable.
    """
    ollama_url = settings.ollama_base_url
    llm_model = settings.llm_chat_model
    embed_model = "vietnamese-sbert"

    async def check_ollama_tags() -> dict:
        """Check if Ollama is reachable and the LLM model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                resp.raise_for_status()
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                llm_ready = any(llm_model in m for m in models)
                # Note: vietnamese-sbert is handled locally, but we report it here for UI consistency
                embed_ready = True 
                return {
                    "ollama": "online",
                    "llm_model": llm_model,
                    "llm_ready": llm_ready,
                    "embed_model": embed_model,
                    "embed_ready": embed_ready,
                    "available_models": models,
                }
        except httpx.ConnectError:
            return {"ollama": "offline", "llm_ready": False, "embed_ready": False, "error": "Cannot connect to Ollama"}
        except Exception as e:
            return {"ollama": "error", "llm_ready": False, "embed_ready": False, "error": str(e)}

    result = await check_ollama_tags()
    overall = "healthy" if result.get("llm_ready") and result.get("embed_ready") else "degraded"

    return {
        "status": overall,
        "services": result,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    request: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Full RAG chat endpoint.
    
    Pipeline: embed → intent → tool retrieval → LLM response → save history
    
    Intents handled:
    - **book_search**: Tìm sách theo chủ đề/tác giả/nội dung
    - **stock_check**: Kiểm tra sách có sẵn không
    - **debt_check**: Kiểm tra nợ phạt sinh viên
    - **policy_query**: Hỏi quy định thư viện
    - **general_chat**: Trò chuyện thông thường
    """
    # Auto-generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await chat_service.process_message(
            db=db,
            message=request.message,
            session_id=session_id,
            student_id=request.student_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return ChatResponse(
            reply="Rất tiếc, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
            intent="error",
            entities={},
            sources=[],
            session_id=session_id,
            success=False,
        )


@router.get("/session/{session_id}")
async def get_session_history(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """
    Get full chat history for a session.
    
    Returns messages in chronological order.
    """
    try:
        history = await chat_service.get_history(db, session_id, limit=limit)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": [
                {
                    "id": str(msg.get("id", "")),
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "created_at": str(msg.get("created_at", "")),
                    "metadata": msg.get("metadata", {}),
                }
                for msg in history
            ],
        }
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail="Cannot retrieve chat history.")


@router.delete("/session/{session_id}")
async def clear_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Clear all messages for a specific session.
    
    Useful for starting a fresh conversation.
    """
    try:
        success = await chat_service.clear_history(db, session_id)
        if success:
            return {"message": f"Session '{session_id}' cleared successfully.", "success": True}
        raise HTTPException(status_code=500, detail="Failed to clear session.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail="Cannot clear session history.")


@router.get("/intents")
async def list_intents():
    """
    List all supported chatbot intents and example queries.
    Useful for frontend to display chat suggestions.
    """
    return {
        "intents": [
            {
                "name": "book_search",
                "description": "Tìm sách theo chủ đề, tác giả hoặc nội dung",
                "examples": [
                    "Tìm sách về lập trình Python",
                    "Sách kinh tế vi mô của Mankiw",
                    "Gợi ý sách về trí tuệ nhân tạo",
                ],
            },
            {
                "name": "stock_check",
                "description": "Kiểm tra sách có sẵn để mượn không",
                "examples": [
                    "Sách Giải tích còn không?",
                    "Cuốn Clean Code còn sẵn không?",
                    "Ai đang mượn sách Cơ sở dữ liệu?",
                ],
            },
            {
                "name": "debt_check",
                "description": "Kiểm tra tiền phạt và sách đang mượn",
                "examples": [
                    "Kiểm tra nợ của tôi",
                    "Tôi đang nợ bao nhiêu tiền?",
                    "Sách nào tôi đang mượn?",
                ],
            },
            {
                "name": "policy_query",
                "description": "Hỏi về quy định thư viện",
                "examples": [
                    "Mượn sách được bao nhiêu ngày?",
                    "Phí phạt quá hạn là bao nhiêu?",
                    "Thư viện mở cửa lúc mấy giờ?",
                ],
            },
            {
                "name": "general_chat",
                "description": "Trò chuyện thông thường hoặc câu hỏi khác",
                "examples": [
                    "Xin chào!",
                    "Bạn có thể giúp gì cho tôi?",
                    "Cảm ơn bạn!",
                ],
            },
        ]
    }
