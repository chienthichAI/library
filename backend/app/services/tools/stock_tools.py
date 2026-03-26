"""
SmartLib - Stock Check Tool
Check availability of specific books in the library.
"""
from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.embedding_service import embedding_service
from app.services.rag_service import rag_service


async def check_book_stock(
    db: AsyncSession,
    query: str,
    entities: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Check if a specific book is available for borrowing.
    
    Searches by book title/ID, returns availability status and due dates
    for currently borrowed copies.
    
    Args:
        db: Database session
        query: User query mentioning specific book
        entities: Extracted entities (book_title, etc.)
        
    Returns:
        Dict with stock info and context string
    """
    search_term = query
    if entities and entities.get("book_title"):
        search_term = entities["book_title"]

    try:
        # Search by title similarity (keyword)
        sql = text("""
            SELECT 
                b.book_id, b.title, b.author, b.status,
                t.due_date, t.student_id, t.days_overdue
            FROM books b
            LEFT JOIN transactions t ON (
                b.book_id = t.book_id 
                AND t.status IN ('ACTIVE', 'OVERDUE')
            )
            WHERE 
                b.title ILIKE :q OR 
                b.book_id = :exact
            ORDER BY 
                CASE WHEN b.title ILIKE :exact_like THEN 1 ELSE 2 END
            LIMIT 5
        """)

        pattern = f"%{search_term}%"
        result = await db.execute(sql, {
            "q": pattern,
            "exact": search_term,
            "exact_like": pattern,  # Fix: ILIKE needs wildcard % to match substrings
        })
        rows = result.mappings().all()

        if not rows:
            # Fallback: semantic search
            emb = await embedding_service.embed(search_term)
            if emb:
                books = await rag_service.search_books_semantic(db, emb, top_k=3)
                if books:
                    # Format stock response from semantic results
                    context_lines = [
                        "Không tìm thấy sách chính xác, nhưng đây là sách tương tự:\n"
                    ]
                    for b in books:
                        status_map = {
                            "AVAILABLE": "✅ Có thể mượn",
                            "BORROWED": "❌ Đang được mượn",
                            "RESERVED": "🔒 Đang giữ chỗ",
                            "DAMAGED": "⚠️ Hư hỏng",
                            "LOST": "🚫 Mất",
                        }
                        status_str = status_map.get(str(b["status"]), str(b["status"]))
                        context_lines.append(f"- **{b['title']}** ({b['author']}): {status_str}")
                    return {
                        "found": True,
                        "books": books,
                        "context": "\n".join(context_lines),
                    }

            return {
                "found": False,
                "books": [],
                "context": f"Không tìm thấy sách '{search_term}' trong hệ thống. Bạn có thể thử tìm kiếm bằng tên khác hoặc hỏi mình tìm sách về chủ đề tương tự.",
            }

        # Build stock report
        context_lines = []
        books_data = []
        seen_ids = set()

        for row in rows:
            book_id = row["book_id"]
            if book_id in seen_ids:
                continue
            seen_ids.add(book_id)

            status = str(row["status"])
            status_map = {
                "AVAILABLE": "✅ **Có thể mượn ngay**",
                "BORROWED": "❌ **Đang được mượn**",
                "RESERVED": "🔒 **Đang giữ chỗ**",
                "DAMAGED": "⚠️ **Đang sửa chữa**",
                "LOST": "🚫 **Đã mất**",
            }
            status_str = status_map.get(status, status)

            line = f"📚 **{row['title']}** - {row['author'] or 'Không rõ tác giả'}\n   Trạng thái: {status_str}"
            
            if status == "BORROWED" and row.get("due_date"):
                line += f"\n   📅 Hạn trả: {row['due_date']}"
                if row.get("days_overdue", 0) > 0:
                    line += f" (Quá hạn {row['days_overdue']} ngày)"

            context_lines.append(line)
            books_data.append(dict(row))

        return {
            "found": True,
            "books": books_data,
            "context": "\n\n".join(context_lines),
        }

    except Exception as e:
        logger.error(f"Stock check failed: {e}")
        await db.rollback()
        return {
            "found": False,
            "books": [],
            "context": "Xin lỗi, không thể kiểm tra trạng thái sách lúc này. Vui lòng thử lại sau.",
        }
