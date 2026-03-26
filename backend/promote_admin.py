import asyncio
import sys
import os

# Add path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models.student import Student, UserRole
from sqlalchemy import select

async def promote_to_admin(student_id):
    db = SessionLocal()
    try:
        stmt = select(Student).where(Student.student_id == student_id)
        result = await db.execute(stmt)
        student = result.scalar_one_or_none()
        
        if not student:
            print(f"❌ Không tìm thấy sinh viên với ID: {student_id}")
            return
            
        student.role = UserRole.ADMIN
        await db.commit()
        print(f"✅ Đã nâng cấp {student.full_name} ({student_id}) lên thành ADMIN thành công!")
        print("Bây giờ bạn có thể dùng khuôn mặt của người này để đăng nhập vào giao diện Admin.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        await db.rollback()
    finally:
        await db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python promote_admin.py <student_id>")
    else:
        asyncio.run(promote_to_admin(sys.argv[1]))
