"""
SmartLib - Policy Query Tool
Retrieve relevant library policy chunks using vector similarity.
"""
from typing import Dict, Any, Optional, List
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedding_service import embedding_service
from app.services.rag_service import rag_service


async def query_policy(
    db: AsyncSession,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve relevant policy chunks for a user's question.
    
    Args:
        db: Database session
        query: User's question about library policy
        top_k: Number of chunks to retrieve
        
    Returns:
        Dict with policy chunks and context string
    """
    try:
        # Generate embedding for the query
        embedding = await embedding_service.embed(query)

        if not embedding:
            logger.warning("Could not embed policy query, returning empty.")
            return {
                "found": False,
                "chunks": [],
                "context": "Không thể tìm kiếm thông tin quy định lúc này. Vui lòng liên hệ thủ thư để được hỗ trợ.",
            }

        # Search policy_chunks table
        chunks = await rag_service.search_policy(db, embedding, top_k=top_k)

        if not chunks:
            # Fallback: return general policy summary
            return {
                "found": False,
                "chunks": [],
                "context": _get_general_policy_summary(),
            }

        # Filter chunks by minimum similarity
        relevant_chunks = [c for c in chunks if c.get("similarity", 0) >= 0.35]
        
        if not relevant_chunks:
            relevant_chunks = chunks[:2]  # Take top 2 even if low similarity

        context = rag_service.format_policy_for_context(relevant_chunks)

        return {
            "found": True,
            "chunks": relevant_chunks,
            "context": context,
        }

    except Exception as e:
        logger.error(f"Policy query failed: {e}")
        return {
            "found": False,
            "chunks": [],
            "context": _get_general_policy_summary(),
        }


def _get_general_policy_summary() -> str:
    """Return general policy summary as hardcoded fallback when DB search fails."""
    return """⚠️ *(Thông tin tổng quát — không tìm được dữ liệu từ hệ thống lúc này)*

**Quy định cơ bản của Thư viện SmartLib:**

- 📚 Mỗi sinh viên được mượn tối đa **5 quyển** cùng lúc
- ⏰ Thời hạn mượn tiêu chuẩn: **14 ngày**
- 🔄 Gia hạn tối đa **2 lần**, mỗi lần thêm **7 ngày**
- 💰 Phí phạt quá hạn: **10.000 VNĐ/ngày/quyển**
- 🕐 Giờ mở cửa: Thứ 2-6: 7:00-21:00 | Thứ 7: 8:00-17:00 | CN: Đóng cửa

💡 Để xem quy định chính xác và đầy đủ nhất, vui lòng liên hệ thủ thư hoặc hỏi lại mình khi hệ thống ổn định."""
