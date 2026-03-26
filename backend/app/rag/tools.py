import json
from loguru import logger
from langchain_core.tools import tool
from sqlalchemy import select, or_, func
from app.database import async_session_maker
from app.models.book import Book, BookStatus
from app.models.student import Student
from app.models.transaction import Transaction, TransactionStatus
from underthesea import word_tokenize, pos_tag
import re

@tool
async def search_books(query: str) -> str:
    """Tra cứu sách theo tên, tác giả hoặc thể loại trong thư viện. Sử dụng công cụ này khi sinh viên muốn tìm một cuốn sách. Hãy chỉ truyền vào các KEYWORD quan trọng (tên sách, tác giả), không truyền cả câu hỏi của sinh viên."""
    logger.info(f"Tool executed: search_books with raw query '{query}'")
    try:
        # Pre-process query to extract meaningful keywords, excluding library noise
        tagged_tokens = pos_tag(query.lower())
        allowed_tags = {'N', 'NP', 'V', 'A', 'M', 'Np', 'Nc'}
        stop_keywords = {"tìm", "cần", "muốn", "sách", "cuốn", "bản", "giúp", "mình", "tôi", "cho"}
        
        keywords = []
        for token, tag in tagged_tokens:
            if tag in allowed_tags:
                clean = re.sub(r'[^\w\s]', '', token).strip()
                if clean and len(clean) > 1 and clean not in stop_keywords:
                    keywords.append(clean)
        
        search_term = " ".join(keywords) if keywords else query
        logger.info(f"Extracted keywords for DB query: '{search_term}'")

        async with async_session_maker() as session:
            # We use a more flexible ILIKE search with extracted keywords
            # For even better results, we could call RAGService.search_books_hybrid here
            # but that requires embeddings which tool call might not have yet.
            # So we stick to improved keyword-based ILIKE.
            
            # If multiple keywords, try to match any
            conditions = []
            if keywords:
                for kw in keywords[:3]: # Limit to top 3 keywords to avoid complex queries
                    conditions.append(Book.title.ilike(f"%{kw}%"))
                    conditions.append(Book.author.ilike(f"%{kw}%"))
                    conditions.append(Book.subject_category.ilike(f"%{kw}%"))
            
            # Fallback if no keywords extracted or no conditions built
            if not conditions:
                conditions = [
                    Book.title.ilike(f"%{query}%"),
                    Book.author.ilike(f"%{query}%"),
                    Book.subject_category.ilike(f"%{query}%")
                ]

            stmt = select(Book).where(or_(*conditions)).limit(5)
            
            result = await session.execute(stmt)
            books = result.scalars().all()
            
            if not books:
                return f"Không tìm thấy sách nào khớp với từ khóa '{search_term}'."
            
            response = [f"Tìm thấy {len(books)} kết quả cho '{search_term}':"]
            for b in books:
                status_vn = "Sẵn sàng mượn" if b.status == BookStatus.AVAILABLE else "Đang được mượn/Không khả dụng"
                author = b.author if b.author else "Khuyết danh"
                response.append(f"- '{b.title}' của {author} (Mã sách: {b.book_id}) - Trạng thái: {status_vn}")
                
            return "\n".join(response)
    except Exception as e:
        logger.error(f"Error in search_books tool: {e}")
        return "Xin lỗi, hệ thống đang gặp lỗi khi tra cứu sách."

@tool
async def check_student_info(student_id: str) -> str:
    """Kiểm tra thông tin sinh viên, bao gồm số lượng sách đang mượn và tiền phạt nếu có. Cần chạy công cụ này khi sinh viên muốn biết họ đang nợ sách gì, còn nợ bao nhiêu tiền, hoặc thông tin cá nhân."""
    logger.info(f"Tool executed: check_student_info for student_id '{student_id}'")
    try:
        async with async_session_maker() as session:
            # Get student info
            stmt = select(Student).where(Student.student_id == student_id)
            result = await session.execute(stmt)
            student = result.scalar_one_or_none()
            
            if not student:
                return f"Không tìm thấy sinh viên với mã '{student_id}' trong hệ thống."
                
            # Get active transactions
            t_stmt = select(Transaction).where(
                Transaction.student_id == student_id,
                Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
            )
            t_result = await session.execute(t_stmt)
            active_txs = t_result.scalars().all()
            
            info = [
                f"Thông tin sinh viên:",
                f"- Tên: {student.full_name}",
                f"- Mã SV: {student.student_id}",
                f"- Số sách đang mượn (chưa trả): {len(active_txs)} cuốn",
                f"- Tiền phạt đang nợ: {student.fine_balance:,.0f} VNĐ"
            ]
            
            if active_txs:
                info.append("\nChi tiết sách đang mượn:")
                for tx in active_txs:
                    # Also need book details to show name
                    b_stmt = select(Book).where(Book.book_id == tx.book_id)
                    b_res = await session.execute(b_stmt)
                    book = b_res.scalar_one_or_none()
                    title = book.title if book else "Không rõ"
                    
                    status_str = "Quá hạn!" if tx.status == TransactionStatus.OVERDUE else "Đang mượn"
                    info.append(f"  + Cuốn '{title}' (Mã sách: {tx.book_id}) - Hạn trả: {tx.due_date} [{status_str}]")
                    
            return "\n".join(info)
    except Exception as e:
        logger.error(f"Error in check_student_info tool: {e}")
        return "Xin lỗi, hệ thống đang gặp lỗi khi kiểm tra thông tin sinh viên."

# List of tools to pass to the agent
LIBRARY_TOOLS = [search_books, check_student_info]
