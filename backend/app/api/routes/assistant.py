"""
SmartLib Kiosk - AI Assistant Routes
Natural language chat and smart search endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel

from app.services.llm_service import ai_assistant
from app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.book import Book

router = APIRouter(prefix="/ai", tags=["AI Assistant"])

class ChatRequest(BaseModel):
    message: str
    context_type: Optional[str] = "general" # general, search, book_info
    student_id: Optional[str] = None

@router.post("/chat")
async def chat_with_assistant(
    request: ChatRequest = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the AI Assistant (Qwen 3 8B).
    Focus: Smart book recommendation based on library database.
    """
    try:
        # 1. Extract keywords or intent for searching the database
        # We ask the LLM briefly (or use simple processing) to find search terms
        search_query = request.message.lower()
        
        # 2. Query Database for relevant books
        # We look for matches in title, author or subject category
        stmt = select(Book).filter(
            (Book.title.ilike(f"%{search_query}%")) | 
            (Book.subject_category.ilike(f"%{search_query}%")) |
            (Book.description.ilike(f"%{search_query}%"))
        ).limit(10)
        
        result = await db.execute(stmt)
        available_books = result.scalars().all()
        
        # 3. Build context for the Assistant
        # We tell the AI exactly what books we have so it doesn't "hallucinate" external books
        book_context = ""
        if available_books:
            book_context = "Dưới đây là danh sách sách hiện có trong thư viện phù hợp với yêu cầu:\n"
            for b in available_books:
                status_str = "Sẵn sàng mượn" if b.status == "AVAILABLE" else "Đã được mượn"
                book_context += f"- Tên: {b.title} | Tác giả: {b.author} | Thể loại: {b.subject_category} | Trạng thái: {status_str}\n"
        else:
            book_context = "Hiện tại không tìm thấy sách nào khớp chính xác trong kho dữ liệu cho yêu cầu này."

        system_prompt = (
            "Bạn là 'Trợ lý học thuật' thông minh của Thư viện trường Đại học. "
            "Phong cách trả lời: Chuyên nghiệp, hỗ trợ, sử dụng đại từ 'Mình' và 'Bạn' một cách thân thiện. "
            "Nhiệm vụ: Phân tích danh sách sách dưới đây và gợi ý cho sinh viên dựa trên nhu cầu học tập/nghiên cứu của họ. "
            "Nguyên tắc cốt lõi: CHỈ gợi ý sách có trong danh sách CONTEXT. Tuyệt đối không giới thiệu sách ngoài hệ thống. "
            "Khi không có sách khớp, hãy nói: 'Thư viện hiện chưa có đầu sách chính xác này, bạn có muốn mình tìm các chủ đề tương tự không?' "
            "Hãy trả lời bằng tiếng Việt, súc tích và tập trung vào lợi ích cho việc học của sinh viên.\n\n"
            f"CONTEXT (Kho sách hiện có):\n{book_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        # 4. Request LLM (Qwen 3 8B)
        response = await ai_assistant.chat(messages)
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
            
        # 5. Format suggested books metadata for frontend UI
        suggestions = [
            {
                "book_id": b.book_id,
                "title": b.title,
                "author": b.author,
                "status": b.status,
                "category": b.subject_category
            } for b in available_books
        ]
        
        return {
            "reply": response["message"]["content"],
            "suggestions": suggestions,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"AI Assistant Error: {e}")
        return {
            "reply": "Rất tiếc, hệ thống tư vấn sách đang gặp sự cố. Bạn có thể tìm sách thủ công bằng mã barcode.",
            "success": False
        }
