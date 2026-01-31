import asyncio
import sys
import os

# Thêm đường dẫn để import được app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models.book import Book, BookStatus

async def add_book(book_id, title, author, barcode):
    db = SessionLocal()
    try:
        new_book = Book(
            book_id=book_id,
            title=title,
            author=author,
            barcode=barcode,
            status=BookStatus.AVAILABLE
        )
        db.add(new_book)
        await db.commit()
        print(f"✅ Đã thêm sách thành công: {title}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        await db.rollback()
    finally:
        await db.close()

if __name__ == "__main__":
    # Bạn có thể thay đổi thông tin sách ở đây
    BOOK_DATA = {
        "book_id": "MS001",
        "title": "Lập trình Python cơ bản",
        "author": "Nguyễn Văn A",
        "barcode": "123456789" # Mã này để camera quét
    }
    
    asyncio.run(add_book(**BOOK_DATA))
