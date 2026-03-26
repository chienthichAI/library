"""
SmartLib - Action Tools (Gia hạn & Đặt trước / Renew & Reserve)

Provides chatbot-callable tools for:
- renew_book:   Extend due date of an active borrow (max 1 time per transaction)
- reserve_book: Add student to waiting queue when book is BORROWED/UNAVAILABLE
"""
from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.models.book import Book, BookStatus
from app.models.transaction import Transaction, TransactionStatus
from app.models.student import Student
from app.services.transaction_service import transaction_service


# ─────────────────────────────────────────────────────────────────────────────
# RENEW BOOK TOOL
# ─────────────────────────────────────────────────────────────────────────────

async def renew_book_tool(
    db: AsyncSession,
    student_id: str,
    entities: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Gia hạn sách đang mượn cho sinh viên.

    Logic:
    1. Lấy danh sách sách sinh viên đang mượn
    2. Nếu entities có book_title/topic → tìm sách khớp nhất
    3. Nếu chỉ có 1 cuốn → tự động gia hạn cuốn đó
    4. Nếu nhiều cuốn → liệt kê và yêu cầu chỉ rõ
    """
    entities = entities or {}

    # --- Step 1: Lấy sách đang mượn ---
    try:
        stmt = (
            select(Transaction, Book.title, Book.book_id)
            .join(Book, Transaction.book_id == Book.book_id)
            .where(
                and_(
                    Transaction.student_id == student_id,
                    Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
                )
            )
        )
        result = await db.execute(stmt)
        rows = result.all()
    except Exception as e:
        logger.error(f"[renew_tool] DB error: {e}")
        return {
            "success": False,
            "context": "❌ Lỗi hệ thống khi tra cứu sách đang mượn. Vui lòng thử lại.",
            "action": "renew",
        }

    if not rows:
        return {
            "success": False,
            "context": "📭 Bạn hiện không mượn cuốn sách nào để gia hạn.",
            "action": "renew",
        }

    # --- Step 2: Tìm sách khớp với entities ---
    target_row = None
    keyword = (
        entities.get("book_title") or
        entities.get("topic") or
        entities.get("book_keyword") or
        ""
    ).lower().strip()

    if keyword:
        # Fuzzy match against titles
        import difflib
        titles = [row[1].lower() for row in rows]
        matches = difflib.get_close_matches(keyword, titles, n=1, cutoff=0.3)
        if matches:
            matched_title = matches[0]
            for row in rows:
                if row[1].lower() == matched_title:
                    target_row = row
                    break
        else:
            # Try substring match
            for row in rows:
                if keyword in row[1].lower():
                    target_row = row
                    break

    # If no keyword match but exactly 1 book → auto-select
    if not target_row:
        if len(rows) == 1:
            target_row = rows[0]
        else:
            # Ambiguous — list all and ask user to clarify
            book_list = "\n".join(
                [f"- **{row[1]}** (Hạn: {row[0].due_date})" for row in rows]
            )
            return {
                "success": False,
                "context": (
                    f"📚 Bạn đang mượn **{len(rows)} cuốn**. Bạn muốn gia hạn cuốn nào?\n\n"
                    f"{book_list}\n\n"
                    "Hãy nói rõ tên sách để mình gia hạn nhé!"
                ),
                "action": "renew",
                "books": [{"title": row[1], "due_date": str(row[0].due_date)} for row in rows],
            }

    txn, book_title, book_id = target_row

    # --- Step 3: Thực hiện gia hạn ---
    renew_result = await transaction_service.renew_book(
        student_id=student_id,
        book_id=book_id,
        db=db,
    )

    if renew_result.success:
        context = (
            f"✅ **Gia hạn thành công!**\n\n"
            f"📖 **Sách:** {renew_result.book_title}\n"
            f"📅 **Hạn trả mới:** {renew_result.due_date}\n\n"
            f"*Lưu ý: Mỗi cuốn chỉ được gia hạn 1 lần.*"
        )
    else:
        context = (
            f"❌ **Không thể gia hạn:** {renew_result.error_message}\n\n"
            f"📖 Sách: {book_title}"
        )

    return {
        "success": renew_result.success,
        "context": context,
        "action": "renew",
        "book_title": book_title,
        "new_due_date": str(renew_result.due_date) if renew_result.due_date else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESERVE BOOK TOOL
# ─────────────────────────────────────────────────────────────────────────────

async def reserve_book_tool(
    db: AsyncSession,
    student_id: str,
    entities: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Đặt trước sách đang hết để được thông báo khi sách về.

    Logic:
    1. Tìm sách theo entities (title/topic)
    2. Kiểm tra trạng thái — sách phải đang BORROWED để đặt trước
    3. Kiểm tra sinh viên chưa đặt trước (tránh duplicate)
    4. Tạo reservation record trong DB
    """
    entities = entities or {}

    search_keyword = (
        entities.get("book_title") or
        entities.get("topic") or
        entities.get("book_keyword") or
        ""
    ).strip()

    if not search_keyword:
        return {
            "success": False,
            "context": "❓ Bạn muốn đặt trước sách nào? Hãy cho mình biết tên sách hoặc chủ đề nhé.",
            "action": "reserve",
        }

    # --- Step 1: Tìm sách ---
    try:
        stmt = select(Book).where(
            Book.title.ilike(f"%{search_keyword}%")
        ).limit(5)
        result = await db.execute(stmt)
        books = result.scalars().all()
    except Exception as e:
        logger.error(f"[reserve_tool] DB search error: {e}")
        return {
            "success": False,
            "context": "❌ Lỗi khi tìm sách. Vui lòng thử lại.",
            "action": "reserve",
        }

    if not books:
        return {
            "success": False,
            "context": (
                f"🔍 Không tìm thấy sách nào có tên **'{search_keyword}'** trong thư viện.\n"
                "Bạn có muốn mình tìm sách tương tự không?"
            ),
            "action": "reserve",
        }

    # --- Step 2: Tìm sách đang được mượn (có thể đặt trước) ---
    unavailable_books = [b for b in books if b.status == BookStatus.BORROWED]
    available_books = [b for b in books if b.status == BookStatus.AVAILABLE]

    if available_books:
        # Sách đang có sẵn — không cần đặt trước
        book_list = ", ".join([f"**{b.title}**" for b in available_books[:3]])
        return {
            "success": False,
            "context": (
                f"📗 Tin vui! Sách bạn muốn hiện **đang có sẵn** trên kệ:\n{book_list}\n\n"
                "Bạn có thể mượn ngay tại quầy hoặc qua kiosk. Không cần đặt trước!"
            ),
            "action": "reserve",
        }

    if not unavailable_books:
        return {
            "success": False,
            "context": (
                f"📭 Sách **'{search_keyword}'** hiện không có trong thư viện hoặc đang bảo trì.\n"
                "Vui lòng liên hệ thủ thư để được hỗ trợ thêm."
            ),
            "action": "reserve",
        }

    # --- Step 3: Chọn cuốn để đặt trước (ưu tiên cuốn đầu tiên nếu nhiều) ---
    target_book = unavailable_books[0]

    # --- Step 4: Kiểm tra và tạo reservation ---
    try:
        # Check existing reservation
        check_stmt = select(Transaction).where(
            and_(
                Transaction.student_id == student_id,
                Transaction.book_id == target_book.book_id,
                Transaction.status == TransactionStatus.RESERVED,
            )
        )
        existing = await db.execute(check_stmt)
        if existing.scalar_one_or_none():
            return {
                "success": False,
                "context": (
                    f"📌 Bạn **đã đặt trước** cuốn **{target_book.title}** rồi.\n"
                    "Mình sẽ thông báo ngay khi sách được trả về!"
                ),
                "action": "reserve",
            }

        # Check student has no fines
        student_stmt = select(Student).where(Student.student_id == student_id)
        student_res = await db.execute(student_stmt)
        student = student_res.scalar_one_or_none()
        if student and student.fine_balance > 0:
            return {
                "success": False,
                "context": (
                    f"💳 Bạn đang có nợ phạt **{student.fine_balance:,.0f} VND**.\n"
                    "Vui lòng thanh toán nợ trước khi đặt trước sách nhé."
                ),
                "action": "reserve",
            }

        # Count current position in queue
        queue_stmt = select(func.count()).select_from(Transaction).where(
            and_(
                Transaction.book_id == target_book.book_id,
                Transaction.status == TransactionStatus.RESERVED,
            )
        )
        queue_result = await db.execute(queue_stmt)
        queue_position = (queue_result.scalar() or 0) + 1

        # Create reservation record
        reservation = Transaction(
            student_id=student_id,
            book_id=target_book.book_id,
            status=TransactionStatus.RESERVED,
        )
        db.add(reservation)
        await db.commit()

        context = (
            f"📌 **Đặt trước thành công!**\n\n"
            f"📖 **Sách:** {target_book.title}\n"
            f"👤 **Tác giả:** {target_book.author or 'Chưa rõ'}\n"
            f"🔢 **Vị trí hàng chờ:** #{queue_position}\n\n"
            f"⏰ Mình sẽ thông báo ngay khi sách được trả lại!"
        )

        return {
            "success": True,
            "context": context,
            "action": "reserve",
            "book_title": target_book.book_id,
            "queue_position": queue_position,
        }

    except Exception as e:
        logger.error(f"[reserve_tool] Create reservation error: {e}")
        await db.rollback()
        return {
            "success": False,
            "context": "❌ Lỗi hệ thống khi đặt trước sách. Vui lòng thử lại.",
            "action": "reserve",
        }


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION TOOL
# ─────────────────────────────────────────────────────────────────────────────

async def get_personalized_recommendations(
    db: AsyncSession,
    student_id: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Gợi ý sách cá nhân hóa dựa trên lịch sử mượn của sinh viên.

    Logic:
    1. Lấy 5 cuốn gần nhất đã mượn (kể cả đã trả)
    2. Dùng Semantic Search tìm sách tương tự
    3. Loại bỏ sách sinh viên đã từng mượn
    4. Trả về top_k gợi ý
    """
    try:
        # Lấy lịch sử mượn sách (kể cả đã trả)
        stmt = (
            select(Book.title, Book.subject_category, Book.smart_category)
            .join(Transaction, Transaction.book_id == Book.book_id)
            .where(Transaction.student_id == student_id)
            .order_by(Transaction.created_at.desc())
            .limit(5)
        )
        result = await db.execute(stmt)
        history = result.all()

        if not history:
            return {
                "context": "",
                "has_recommendations": False,
                "books": [],
            }

        # Tổng hợp chủ đề từ lịch sử
        topics = []
        for title, category, smart_cat in history:
            if title:
                topics.append(title)
            if category:
                topics.append(category)
            if smart_cat:
                topics.append(smart_cat)
        
        search_query = " ".join(set(topics[:6]))  # Deduplicated, max 6 terms

        # Lấy IDs sách đã từng mượn (để loại trừ)
        borrowed_stmt = (
            select(Transaction.book_id)
            .where(Transaction.student_id == student_id)
        )
        borrowed_result = await db.execute(borrowed_stmt)
        borrowed_ids = {row[0] for row in borrowed_result.all()}

        # Semantic Search
        from app.services.embedding_service import embedding_service
        from app.services.rag_service import rag_service

        embedding = await embedding_service.embed(search_query)
        if not embedding:
            return {"context": "", "has_recommendations": False, "books": []}

        candidates = await rag_service.search_books_semantic(db, embedding, top_k=top_k * 3)
        
        # Lọc sách đã mượn + chỉ lấy sách AVAILABLE
        recommendations = [
            b for b in candidates
            if b["book_id"] not in borrowed_ids
            and b.get("status") == "AVAILABLE"
        ][:top_k]

        if not recommendations:
            return {"context": "", "has_recommendations": False, "books": []}

        # Format context theo phong cách "Cards"
        context_lines = [
            f"💡 **Dựa trên sở thích gần đây của bạn** (cuốn: *{recent_title}*):\n"
        ]
        
        for book in recommendations:
            context_lines.append(
                f"- 📘 **{book['title']}**\n"
                f"  - 👤 Tác giả: *{book.get('author', 'Đang cập nhật')}*\n"
                f"  - 🏷️ Thể loại: `{book.get('subject_category', 'Chưa phân loại')}`"
            )

        return {
            "context": "\n".join(context_lines),
            "has_recommendations": True,
            "books": recommendations,
        }

    except Exception as e:
        logger.error(f"[recommendation_tool] Error: {e}")
        return {"context": "", "has_recommendations": False, "books": []}
