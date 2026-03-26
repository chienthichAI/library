"""
SmartLib - Debt Check Tool
Query student fine balance and overdue transaction details.
"""
from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import re


async def check_student_debt(
    db: AsyncSession,
    query: str,
    entities: Optional[Dict] = None,
    session_student_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check a student's fine balance and overdue transactions.
    
    Args:
        db: Database session
        query: User's raw query
        entities: Extracted entities (student_id)
        session_student_id: Student ID from authenticated session
        
    Returns:
        Dict with debt info and context string
    """
    # Resolve student ID - priority: authenticated session > entities > extracted from query
    student_id = (
        session_student_id
        or (entities or {}).get("student_id")
        or _extract_student_id(query)
    )

    if not student_id:
        return {
            "found": False,
            "student": None,
            "context": (
                "Để kiểm tra nợ phạt, bạn cần cung cấp mã sinh viên. "
                "Ví dụ: 'Kiểm tra nợ của sinh viên SE001' hoặc 'Mã sinh viên của tôi là SS123'."
            ),
        }

    try:
        # Query student info + fine balance
        student_sql = text("""
            SELECT student_id, full_name, fine_balance, status, email
            FROM students
            WHERE student_id ILIKE :sid
            LIMIT 1
        """)
        student_result = await db.execute(student_sql, {"sid": student_id})
        student = student_result.mappings().first()

        if not student:
            return {
                "found": False,
                "student": None,
                "context": (
                    f"Không tìm thấy sinh viên với mã **{student_id}**. "
                    "Vui lòng kiểm tra lại mã sinh viên."
                ),
            }

        # Query overdue / active transactions
        tx_sql = text("""
            SELECT 
                t.transaction_id, t.book_id, b.title AS book_title,
                t.borrow_date, t.due_date, t.status, t.days_overdue, t.fine_amount
            FROM transactions t
            JOIN books b ON t.book_id = b.book_id
            WHERE t.student_id = :sid
              AND t.status IN ('ACTIVE', 'OVERDUE')
            ORDER BY t.due_date ASC
        """)
        tx_result = await db.execute(tx_sql, {"sid": student["student_id"]})
        transactions = tx_result.mappings().all()

        # Build context
        fine_balance = float(student["fine_balance"] or 0)
        context_lines = [
            f"📋 **Thông tin sinh viên: {student['full_name']}** (Mã: {student['student_id']})",
            f"💰 **Số dư nợ phạt: {fine_balance:,.0f} VNĐ**",
        ]

        if fine_balance > 0:
            context_lines.append("⚠️ Bạn có nợ phạt chưa thanh toán. Vui lòng thanh toán tại quầy thư viện để khôi phục quyền mượn sách.")
        else:
            context_lines.append("✅ Bạn không có nợ phạt nào.")

        if transactions:
            context_lines.append(f"\n📚 **Sách đang mượn ({len(transactions)} quyển):**")
            for tx in transactions:
                status_icon = "🔴" if str(tx["status"]) == "OVERDUE" else "📗"
                line = f"{status_icon} {tx['book_title']} — Hạn trả: {tx['due_date']}"
                if tx.get("days_overdue", 0) > 0:
                    fine = float(tx.get("fine_amount", 0))
                    line += f" (**Quá hạn {tx['days_overdue']} ngày, phạt: {fine:,.0f} VNĐ**)"
                context_lines.append(f"   - {line}")
        else:
            context_lines.append("📚 Hiện không có sách nào đang mượn.")

        return {
            "found": True,
            "student": dict(student),
            "transactions": [dict(t) for t in transactions],
            "fine_balance": fine_balance,
            "context": "\n".join(context_lines),
        }

    except Exception as e:
        logger.error(f"Debt check failed for student '{student_id}': {e}")
        await db.rollback()
        return {
            "found": False,
            "student": None,
            "context": "Xin lỗi, không thể truy vấn thông tin nợ phạt lúc này. Vui lòng thử lại sau.",
        }


def _extract_student_id(query: str) -> Optional[str]:
    """Extract student ID pattern like SE001, SS123 from free text.
    Allows optional whitespace between letter prefix and digits (e.g. 'SE 190047').
    """
    match = re.search(r"\b([A-Z]{2})\s*(\d{3,})\b", query, re.IGNORECASE)
    if not match:
        return None
    return f"{match.group(1).upper()}{match.group(2)}"
