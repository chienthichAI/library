from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel
from app.rag.pipeline import RAGPipeline
from app.services.chat_service import chat_service
from app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
import os
import shutil
import uuid
from app.core.face_session import require_face_session, require_admin_session
from typing import Dict, Any

router = APIRouter(tags=["Chatbot RAG"])

# Keep the legacy pipeline instance for /upload-docs (ephemeral RAG demo)
rag_pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    is_scanner_trigger: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = []

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    claims: Dict[str, Any] = Depends(require_face_session)
):
    """
    Consolidated Chat Endpoint.
    Exclusive for logged-in users. Personalizes response using student_id from face session.
    """
    try:
        student_id = claims.get("student_id")
        # Use provided session_id or fallback to a default
        session_id = request.session_id or f"chat-session-{student_id}"
        
        result = await chat_service.process_message(
            db=db,
            message=request.query,
            session_id=session_id,
            student_id=student_id,
            metadata=request.metadata or {}
        )
        
        # Return full orchestrated result
        return {
            "answer": result.get("reply", "Xin lỗi, mình không tìm thấy câu trả lời."),
            "is_scanner_trigger": result.get("is_scanner_trigger", False),
            "metadata": result.get("metadata", {}),
            "suggestions": result.get("suggestions", [])
        }
    except Exception as e:
        import traceback
        from loguru import logger
        logger.error(f"Chat error for student {claims.get('student_id')}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý chatbot.")

@router.delete("/history")
async def clear_chat_history(
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    claims: Dict[str, Any] = Depends(require_face_session)
):
    """Clear chat history for the current user."""
    student_id = claims.get("student_id")
    target_session = session_id or f"chat-session-{student_id}"
    success = await chat_service.clear_history(db, target_session)
    if success:
        return {"message": "Lịch sử trò chuyện đã được xóa."}
    raise HTTPException(status_code=500, detail="Không thể xóa lịch sử trò chuyện.")

@router.get("/history")
async def get_chat_history(
    session_id: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    claims: Dict[str, Any] = Depends(require_face_session)
):
    """Retrieve chat history for the current user."""
    student_id = claims.get("student_id")
    # Priority: query param > default student-based session
    target_session = session_id or f"chat-session-{student_id}"
    
    history = await chat_service.get_history(db, target_session, limit=limit)
    return history

@router.post("/upload-docs")
async def upload_document(
    file: UploadFile = File(...),
    claims: Dict[str, Any] = Depends(require_admin_session)
):
    try:
        # Lưu file tạm thời để xử lý Loader
        temp_dir = "tests/temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Lấy định dạng file (pdf, csv)
        ext = file.filename.split('.')[-1].lower()
        
        # Tiến hành Ingestion Pipeline
        chunks_created = rag_pipeline.ingest_document(temp_file_path, doc_type=ext)
        
        # Có thể xóa file tạm nếu không cần lưu trữ gốc
        os.remove(temp_file_path)
        
        return {
            "message": "Tài liệu đã được AI học thành công!", 
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
