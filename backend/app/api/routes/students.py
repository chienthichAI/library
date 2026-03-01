"""
SmartLib Kiosk - Students API Routes
Student management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime
from loguru import logger

from app.database import get_db
from app.models.student import Student
from app.schemas.student import (
    StudentCreate, StudentUpdate, StudentResponse, StudentBorrowingInfoResponse
)
from app.services.transaction_service import TransactionService

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(
    student_id: str = Path(..., description="Student ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get student information by ID.
    """
    try:
        stmt = select(Student).where(Student.student_id == student_id)
        result = await db.execute(stmt)
        student = result.scalar_one_or_none()
        
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
            
        return StudentResponse.model_validate(student)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{student_id}/borrowing-info", response_model=StudentBorrowingInfoResponse)
async def get_student_borrowing_info(
    student_id: str = Path(..., description="Student ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get student's current borrowing status.
    
    **Returns:**
    - Currently borrowed books count
    - Maximum allowed books
    - Outstanding fines
    - Whether student can borrow more books
    """
    try:
        transaction_service = TransactionService()
        info = await transaction_service.get_student_borrowing_info(student_id, db)
        
        if not info:
            raise HTTPException(status_code=404, detail="Student not found")
            
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
                    "days_left": (t.due_date - datetime.utcnow().date()).days if t.due_date else 0,
                    "is_overdue": t.is_overdue,
                    "fine_amount": t.calculate_fine(transaction_service.fine_per_day)
                }
                for t in info.active_transactions
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get borrowing info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=StudentResponse, status_code=201)
async def create_student(
    student: StudentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new student account.
    """
    try:
        from sqlalchemy import func
        from app.models.face_embedding import FaceEmbedding
        
        # Check if student ID already exists
        stmt = select(Student).where(Student.student_id == student.student_id)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Check if this student has any face embeddings
            stmt_faces = select(func.count()).where(FaceEmbedding.student_id == student.student_id)
            faces_count = await db.execute(stmt_faces)
            count = faces_count.scalar() or 0
            
            if count == 0:
                # Student exists but has no face embeddings (previous registration failed).
                # Overwrite their info and return it to resume the registration flow.
                existing.full_name = student.full_name
                # Only check email/phone conflicts if they are changed and conflict with ANOTHER student
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
                raise HTTPException(status_code=400, detail="Mã sinh viên này đã tồn tại và đã đăng ký khuôn mặt trong hệ thống")
            
        # Check if email exists
        if student.email:
            stmt_email = select(Student).where(Student.email == student.email)
            result_email = await db.execute(stmt_email)
            if result_email.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Email này đã được sử dụng bởi sinh viên khác")

        # Check if phone exists
        if student.phone:
            stmt_phone = select(Student).where(Student.phone == student.phone)
            result_phone = await db.execute(stmt_phone)
            if result_phone.scalar_one_or_none():
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
    List all students.
    """
    try:
        stmt = select(Student).limit(100)
        result = await db.execute(stmt)
        students = result.scalars().all()
        
        return [StudentResponse.model_validate(s) for s in students]
        
    except Exception as e:
        logger.error(f"List students error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
