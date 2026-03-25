"""
SmartLib Kiosk - Students API Routes
Student management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional
from datetime import datetime
from loguru import logger

from app.database import get_db
from app.models.student import Student, StudentStatus
from app.schemas.student import (
    StudentCreate, StudentUpdate, StudentResponse, StudentBorrowingInfoResponse
)
from app.services.transaction_service import TransactionService
from app.core.face_session import require_admin_session

router = APIRouter(prefix="/students", tags=["Students"])


# ---------------------------------------------------------------------------
# GET /  — list with pagination + search + status filter
# Must be declared BEFORE /{student_id} to avoid FastAPI route shadowing
# ---------------------------------------------------------------------------
@router.get("/", response_model=dict)
async def list_students(
    search: Optional[str] = Query(None, description="Search by name or student ID"),
    status: Optional[str] = Query(None, description="Filter by status (ACTIVE, SUSPENDED, …)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    List all students with optional search + status filter + pagination.

    Returns:
    - total: total matching records
    - students: paginated list
    """
    try:
        stmt = select(Student)
        count_stmt = select(func.count()).select_from(Student)

        # Filter by status
        if status:
            try:
                status_enum = StudentStatus(status.upper())
                stmt = stmt.where(Student.status == status_enum)
                count_stmt = count_stmt.where(Student.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status value: {status}")

        # Search by name or ID (case-insensitive)
        if search:
            pattern = f"%{search.strip()}%"
            condition = or_(
                Student.full_name.ilike(pattern),
                Student.student_id.ilike(pattern)
            )
            stmt = stmt.where(condition)
            count_stmt = count_stmt.where(condition)

        # Total count
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        # Paginated data
        stmt = stmt.order_by(Student.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(stmt)
        students = result.scalars().all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "students": [StudentResponse.model_validate(s) for s in students]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List students error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy danh sách sinh viên")


# ---------------------------------------------------------------------------
# GET /{student_id}
# ---------------------------------------------------------------------------
@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(
    student_id: str = Path(..., description="Student ID"),
    db: AsyncSession = Depends(get_db)
):
    """Get student information by ID."""
    try:
        stmt = select(Student).where(Student.student_id == student_id)
        result = await db.execute(stmt)
        student = result.scalar_one_or_none()

        if not student:
            raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")

        return StudentResponse.model_validate(student)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy thông tin sinh viên")


# ---------------------------------------------------------------------------
# GET /{student_id}/borrowing-info
# ---------------------------------------------------------------------------
@router.get("/{student_id}/borrowing-info", response_model=StudentBorrowingInfoResponse)
async def get_student_borrowing_info(
    student_id: str = Path(..., description="Student ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get student's current borrowing status.

    Returns:
    - Currently borrowed books count
    - Maximum allowed books
    - Outstanding fines
    - Whether student can borrow more books
    """
    try:
        transaction_service = TransactionService()
        info = await transaction_service.get_student_borrowing_info(student_id, db)

        if not info:
            raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")

        fine_per_day = transaction_service.fine_per_day

        return StudentBorrowingInfoResponse(
            student_id=info.student_id,
            student_name=info.student_name,
            currently_borrowed=info.currently_borrowed,
            max_books=info.max_books,
            fine_balance=info.fine_balance,
            can_borrow=info.can_borrow,
            borrowed_books=[
                {
                    "transaction_id": t.transaction_id,
                    "book_id": t.book_id,
                    "title": t.book.title if t.book else "N/A",
                    "borrow_date": t.borrow_date,
                    "due_date": t.due_date,
                    "days_left": max(0, (t.due_date - datetime.utcnow().date()).days) if t.due_date else 0,
                    "is_overdue": t.is_overdue,
                    "fine_amount": t.calculate_fine(fine_per_day)
                }
                for t in info.active_transactions
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get borrowing info error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi lấy thông tin mượn sách")


@router.post("/", response_model=StudentResponse, status_code=201)
async def create_student(
    student: StudentCreate,
    db: AsyncSession = Depends(get_db),
    _claims=Depends(require_admin_session)
):
    """Create a new student account."""
    try:
        from sqlalchemy import func as sqlfunc
        from app.models.face_embedding import FaceEmbedding

        # Check if student ID already exists
        stmt = select(Student).where(Student.student_id == student.student_id)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Check face embeddings
            stmt_faces = select(sqlfunc.count()).where(FaceEmbedding.student_id == student.student_id)
            faces_count = await db.execute(stmt_faces)
            count = faces_count.scalar() or 0

            if count == 0:
                # Resume interrupted registration — safe to overwrite basic info
                existing.full_name = student.full_name
                if student.email and student.email != existing.email:
                    stmt_email = select(Student).where(Student.email == student.email)
                    if (await db.execute(stmt_email)).scalar_one_or_none():
                        raise HTTPException(status_code=400, detail="Email này đã được sử dụng bởi sinh viên khác")
                if student.phone and student.phone != existing.phone:
                    stmt_phone = select(Student).where(Student.phone == student.phone)
                    if (await db.execute(stmt_phone)).scalar_one_or_none():
                        raise HTTPException(status_code=400, detail="Số điện thoại này đã được sử dụng")
                existing.email = student.email
                existing.phone = student.phone
                await db.commit()
                await db.refresh(existing)
                return StudentResponse.model_validate(existing)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mã sinh viên này đã tồn tại và đã đăng ký khuôn mặt trong hệ thống"
                )

        # Unique checks
        if student.email:
            stmt_email = select(Student).where(Student.email == student.email)
            if (await db.execute(stmt_email)).scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Email này đã được sử dụng")
        if student.phone:
            stmt_phone = select(Student).where(Student.phone == student.phone)
            if (await db.execute(stmt_phone)).scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Số điện thoại này đã được sử dụng")

        new_student = Student(
            student_id=student.student_id,
            full_name=student.full_name,
            email=student.email,
            phone=student.phone
        )

        db.add(new_student)
        await db.commit()
        await db.refresh(new_student)

        return StudentResponse.model_validate(new_student)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create student error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[StudentResponse])
async def list_students(
    db: AsyncSession = Depends(get_db)
):
    """
    Clear outstanding fine balance for a student (admin action).
    Used when a student pays at the counter and admin confirms manually.
    """
    try:
        stmt = select(Student).where(Student.student_id == student_id)
        result = await db.execute(stmt)
        student = result.scalar_one_or_none()

        if not student:
            raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")

        old_balance = student.fine_balance
        student.fine_balance = 0.0
        student.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(student)

        # Invalidate cache so it doesn't return stale fine balance
        from app.services.transaction_service import TransactionService
        TransactionService._student_cache.pop(student_id, None)

        logger.info(f"Fine cleared for {student_id}: {old_balance:,.0f} VND → 0")
        return StudentResponse.model_validate(student)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear fine error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xóa công nợ")
