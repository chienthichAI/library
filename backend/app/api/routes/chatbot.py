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

router = APIRouter(tags=["Chatbot RAG"])

# Keep the legacy pipeline instance for /upload-docs (ephemeral RAG demo)
rag_pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    student_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Consolidated Chat Endpoint.
    Now uses the smarter ChatService (Intent + AI Entities + Hybrid Search).
    """
    try:
        # Use provided session_id or fallback to a default
        session_id = request.session_id or "legacy-chatbot-demo"
        student_id = request.student_id
        
        result = await chat_service.process_message(
            db=db,
            message=request.query,
            session_id=session_id,
            student_id=student_id
        )
        
        # Map 'reply' from ChatService to 'answer' for legacy compatibility
        return {"answer": result.get("reply", "Xin lỗi, mình không tìm thấy câu trả lời.")}
    except Exception as e:
        import traceback
        from loguru import logger
        logger.error(f"Consolidated Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý chatbot.")

@router.post("/upload-docs")
async def upload_document(file: UploadFile = File(...)):
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
