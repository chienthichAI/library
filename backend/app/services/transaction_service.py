"""
SmartLib Kiosk - Transaction Service

Handles book borrowing and returning transactions.
Includes fine calculation for overdue returns.
"""
from typing import Optional, List, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_
from loguru import logger
from cachetools import TTLCache

from app.models.student import Student
from app.models.book import Book, BookStatus
from app.models.transaction import Transaction, TransactionType, TransactionStatus
from app.config import settings


@dataclass
class BorrowResult:
    """Result of book borrowing."""
    success: bool
    transaction_id: Optional[str]
    book_title: Optional[str]
    due_date: Optional[date]
    error_message: Optional[str]


@dataclass
class ReturnResult:
    """Result of book return."""
    success: bool
    transaction_id: Optional[str]
    book_title: Optional[str]
    days_overdue: int
    fine_amount: float
    error_message: Optional[str]


@dataclass
class StudentBorrowingInfo:
    """Current borrowing information for a student."""
    student_id: str
    student_name: str
    currently_borrowed: int
    max_books: int
    fine_balance: float
    can_borrow: bool
    active_transactions: List[Transaction]


class TransactionService:
    """
    Transaction Service for book borrowing and returning.
    
    Features:
    - Borrow book with validation
    - Return book with fine calculation
    - Transaction history
    - Fine management
    """
    
    def __init__(
        self,
        max_borrow_days: int = None,
        fine_per_day: float = None,
        max_books_per_student: int = None
    ):
        """
        Initialize transaction service.
        
        Args:
            max_borrow_days: Maximum days a book can be borrowed
            fine_per_day: Fine amount per day overdue (VND)
            max_books_per_student: Maximum books a student can borrow
        """
        self.max_borrow_days = max_borrow_days or settings.max_borrow_days
        self.fine_per_day = fine_per_day or settings.fine_per_day
        self.max_books_per_student = max_books_per_student or settings.max_books_per_student
    
    async def borrow_book(
        self,
        student_id: str,
        book_id: str,
        db: AsyncSession,
        kiosk_id: Optional[str] = None
    ) -> BorrowResult:
        """
        Process a book borrowing transaction.
        
        Args:
            student_id: Student ID
            book_id: Book ID or barcode
            db: Database session
            kiosk_id: Kiosk identifier
            
        Returns:
            BorrowResult
        """
        try:
            # Validate student
            student = await self._get_student(student_id, db)
            if not student:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message="Sinh viên không tồn tại"
                )
                
            if not student.is_active:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message="Tài khoản sinh viên bị khóa"
                )
                
            # Check student borrowing limit
            borrowing_info = await self.get_student_borrowing_info(student_id, db)
            if not borrowing_info.can_borrow:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message=f"Đã đạt giới hạn mượn sách ({self.max_books_per_student} cuốn)"
                )
                
            # Check outstanding fines
            if student.fine_balance > 0:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message=f"Còn nợ tiền phạt: {student.fine_balance:,.0f} VND"
                )
                
            # Validate and lock book row to prevent concurrent borrow race.
            # We bypass cache here to ensure fresh state under transaction lock.
            book_stmt = (
                select(Book)
                .where(or_(Book.book_id == book_id, Book.barcode == book_id))
                .with_for_update()
            )
            book_result = await db.execute(book_stmt)
            book = book_result.scalar_one_or_none()
            if not book:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message="Sách không tồn tại trong hệ thống"
                )
                
            if not book.is_available:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=book.title,
                    due_date=None,
                    error_message=f"Sách hiện không khả dụng (trạng thái: {book.status.value})"
                )
                
            # Check for duplicate active transaction (idempotent borrow) under lock
            # Real-world case: student scans the same book twice (slow UI, double tap)
            existing_stmt = (
                select(Transaction)
                .where(
                    and_(
                        Transaction.student_id == student_id,
                        Transaction.book_id == book.book_id,
                        Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
                    )
                )
                .with_for_update()
            )
            existing_result = await db.execute(existing_stmt)
            existing_txn = existing_result.scalar_one_or_none()
            if existing_txn:
                logger.warning(f"Idempotent borrow: student {student_id} already has book {book.book_id}")
                return BorrowResult(
                    success=True,
                    transaction_id=existing_txn.transaction_id,
                    book_title=book.title,
                    due_date=existing_txn.due_date,
                    error_message=None
                )

            # Create transaction
            now = datetime.utcnow()
            due_date = (now + timedelta(days=self.max_borrow_days)).date()
            
            transaction = Transaction(
                student_id=student_id,
                book_id=book.book_id,
                transaction_type=TransactionType.BORROW,
                borrow_date=now,
                due_date=due_date,
                status=TransactionStatus.ACTIVE,
                kiosk_id=kiosk_id
            )
            
            # Update book status
            book.status = BookStatus.BORROWED
            
            db.add(transaction)
            await db.commit()
            await db.refresh(transaction)
            
            # Invalidate stale cache entries after mutation
            self._book_cache.pop(book_id, None)
            self._book_cache.pop(book.book_id, None)
            self._student_cache.pop(student_id, None)
            
            logger.info(f"Book borrowed: {book.title} by {student.full_name}")
            
            return BorrowResult(
                success=True,
                transaction_id=transaction.transaction_id,
                book_title=book.title,
                due_date=due_date,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Borrow transaction failed: {e}")
            await db.rollback()
            return BorrowResult(
                success=False,
                transaction_id=None,
                book_title=None,
                due_date=None,
                error_message=f"Lỗi hệ thống: {str(e)}"
            )
    
    async def return_book(
        self,
        student_id: str,
        book_id: str,
        db: AsyncSession,
        kiosk_id: Optional[str] = None
    ) -> ReturnResult:
        """
        Process a book return transaction.
        
        Args:
            student_id: Student ID
            book_id: Book ID or barcode
            db: Database session
            kiosk_id: Kiosk identifier
            
        Returns:
            ReturnResult with fine information
        """
        try:
            # Find active transaction
            transaction = await self._find_active_transaction(student_id, book_id, db)
            
            if not transaction:
                return ReturnResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    days_overdue=0,
                    fine_amount=0,
                    error_message="Không tìm thấy giao dịch mượn sách"
                )
                
            # Get book info
            book = await self._get_book(book_id, db)
            
            # Calculate fine
            now = datetime.utcnow()
            transaction.return_date = now
            fine_amount = transaction.calculate_fine(self.fine_per_day)
            days_overdue = transaction.days_overdue
            
            # Update transaction status
            transaction.status = TransactionStatus.COMPLETED
            transaction.kiosk_id = kiosk_id
            
            # Update book status
            book.status = BookStatus.AVAILABLE
            
            # Update student fine balance if applicable
            if fine_amount > 0:
                # Bypass cache for mutation
                stmt_st = select(Student).where(Student.student_id == student_id)
                res_st = await db.execute(stmt_st)
                student = res_st.scalar_one_or_none()
                if student:
                    student.fine_balance += fine_amount
                
            await db.commit()
            
            # Invalidate stale cache entries after mutation
            self._book_cache.pop(book_id, None)
            self._book_cache.pop(book.book_id, None)
            self._student_cache.pop(student_id, None)
            
            logger.info(f"Book returned: {book.title} by student {student_id}, fine: {fine_amount}")
            
            return ReturnResult(
                success=True,
                transaction_id=transaction.transaction_id,
                book_title=book.title,
                days_overdue=days_overdue,
                fine_amount=fine_amount,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Return transaction failed: {e}")
            await db.rollback()
            return ReturnResult(
                success=False,
                transaction_id=None,
                book_title=None,
                days_overdue=0,
                fine_amount=0,
                error_message=f"Lỗi hệ thống: {str(e)}"
            )
    
    async def get_student_borrowing_info(
        self,
        student_id: str,
        db: AsyncSession
    ) -> Optional[StudentBorrowingInfo]:
        """Get current borrowing information for a student."""
        student = await self._get_student(student_id, db)
        if not student:
            return None
            
        # Get active transactions with book info
        from sqlalchemy.orm import joinedload
        stmt = (
            select(Transaction)
            .options(joinedload(Transaction.book))
            .where(
                and_(
                    Transaction.student_id == student_id,
                    Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
                )
            )
        )
        result = await db.execute(stmt)
        active_transactions = result.scalars().all()
        
        currently_borrowed = len(active_transactions)
        can_borrow = (
            currently_borrowed < self.max_books_per_student and
            student.fine_balance == 0 and
            student.is_active
        )
        
        return StudentBorrowingInfo(
            student_id=student.student_id,
            student_name=student.full_name,
            currently_borrowed=currently_borrowed,
            max_books=self.max_books_per_student,
            fine_balance=student.fine_balance,
            can_borrow=can_borrow,
            active_transactions=list(active_transactions)
        )
    
    async def renew_book(
        self,
        student_id: str,
        book_id: str,
        db: AsyncSession,
        kiosk_id: Optional[str] = None
    ) -> BorrowResult:
        """
        Renew an active borrow transaction — extend due date by max_borrow_days.

        Rules:
        - Student must have no outstanding fines
        - Book must have an active transaction
        - Renewal is only allowed once per transaction (to prevent abuse)
        """
        try:
            transaction = await self._find_active_transaction(student_id, book_id, db)
            if not transaction:
                return BorrowResult(
                    success=False,
                    transaction_id=None,
                    book_title=None,
                    due_date=None,
                    error_message="Không tìm thấy giao dịch mượn sách để gia hạn"
                )

            # Check if already renewed (renewal_count is now a proper DB column)
            if (transaction.renewal_count or 0) >= 1:
                return BorrowResult(
                    success=False,
                    transaction_id=transaction.transaction_id,
                    book_title=None,
                    due_date=transaction.due_date,
                    error_message="Sách đã được gia hạn 1 lần, không thể gia hạn thêm"
                )

            # Check fines
            student = await self._get_student(student_id, db)
            if student and student.fine_balance > 0:
                return BorrowResult(
                    success=False,
                    transaction_id=transaction.transaction_id,
                    book_title=None,
                    due_date=transaction.due_date,
                    error_message=f"Còn nợ tiền phạt {student.fine_balance:,.0f} VND, không thể gia hạn"
                )

            book = await self._get_book(book_id, db)
            new_due = (datetime.utcnow() + timedelta(days=self.max_borrow_days)).date()
            transaction.due_date = new_due
            transaction.status = TransactionStatus.ACTIVE  # Reset overdue flag if any

            transaction.renewal_count = (transaction.renewal_count or 0) + 1

            await db.commit()

            # Invalidate cache
            self._book_cache.pop(book_id, None)
            self._student_cache.pop(student_id, None)

            logger.info(f"Book renewed: {book.title if book else book_id} → new due {new_due}")
            return BorrowResult(
                success=True,
                transaction_id=transaction.transaction_id,
                book_title=book.title if book else book_id,
                due_date=new_due,
                error_message=None
            )

        except Exception as e:
            logger.error(f"Renew failed: {e}")
            await db.rollback()
            return BorrowResult(
                success=False,
                transaction_id=None,
                book_title=None,
                due_date=None,
                error_message=f"Lỗi hệ thống: {str(e)}"
            )

    async def get_overdue_summary(self, db: AsyncSession) -> dict:
        """
        Get system-wide overdue statistics for admin dashboard.

        Returns:
        - total_overdue: number of overdue transactions
        - total_fine_pending: total uncollected fine amount (VND)
        """
        try:
            from datetime import date as date_type
            today = date_type.today()

            stmt = select(Transaction).where(
                Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE]),
                Transaction.due_date < today
            )
            result = await db.execute(stmt)
            overdue_txns = result.scalars().all()

            total_fine = sum(t.calculate_fine(self.fine_per_day) for t in overdue_txns)

            return {
                "total_overdue": len(overdue_txns),
                "total_fine_pending": total_fine
            }
        except Exception as e:
            logger.error(f"Overdue summary error: {e}")
            return {"total_overdue": 0, "total_fine_pending": 0}

    async def get_transaction_history(
        self,
        student_id: str,
        db: AsyncSession,
        limit: int = 10,
        offset: int = 0
    ) -> Tuple[List[Transaction], int]:
        """
        Get transaction history for a student.
        
        Returns:
            Tuple of (transactions, total_count)
        """
        # Count total (SQL COUNT — O(1) memory)
        count_stmt = select(func.count()).select_from(Transaction).where(Transaction.student_id == student_id)
        result = await db.execute(count_stmt)
        total = result.scalar() or 0
        
        # Get paginated transactions
        stmt = (
            select(Transaction)
            .where(Transaction.student_id == student_id)
            .order_by(Transaction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        transactions = result.scalars().all()
        
        return list(transactions), total
    
    # In-memory TTL caches for hot-path lookups (P3-02)
    _student_cache = TTLCache(maxsize=200, ttl=60)
    _book_cache = TTLCache(maxsize=500, ttl=60)

    async def _get_student(self, student_id: str, db: AsyncSession) -> Optional[Student]:
        """Get student by ID (with TTL cache)."""
        if student_id in self._student_cache:
            return self._student_cache[student_id]
        stmt = select(Student).where(Student.student_id == student_id)
        result = await db.execute(stmt)
        student = result.scalar_one_or_none()
        if student:
            self._student_cache[student_id] = student
        return student
    
    async def _get_book(self, book_id: str, db: AsyncSession) -> Optional[Book]:
        """Get book by ID or barcode (with TTL cache)."""
        if book_id in self._book_cache:
            return self._book_cache[book_id]
        # Try by book_id
        stmt = select(Book).where(Book.book_id == book_id)
        result = await db.execute(stmt)
        book = result.scalar_one_or_none()
        
        if not book:
            # Try by barcode
            stmt = select(Book).where(Book.barcode == book_id)
            result = await db.execute(stmt)
            book = result.scalar_one_or_none()
        
        if book:
            self._book_cache[book_id] = book
        return book
    
    async def _find_active_transaction(
        self,
        student_id: str,
        book_id: str,
        db: AsyncSession
    ) -> Optional[Transaction]:
        """Find active transaction for student-book pair."""
        # Get book first (might be identified by barcode)
        book = await self._get_book(book_id, db)
        if not book:
            return None
            
        stmt = select(Transaction).where(
            and_(
                Transaction.student_id == student_id,
                Transaction.book_id == book.book_id,
                Transaction.status.in_([TransactionStatus.ACTIVE, TransactionStatus.OVERDUE])
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


# Singleton instance
transaction_service = TransactionService()
