
import asyncio
import os
import sys
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

# Add backend to path for imports
sys.path.append(os.path.abspath("."))

from app.services.chat_service import chat_service
from app.config import settings

DATABASE_URL = "postgresql+asyncpg://postgres.gvgdispqeesizkgzduik:Chiendz098!@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"

async def test_scenarios():
    engine = create_async_engine(DATABASE_URL)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    session_id = "test-session-123"
    
    # 1. Get a test student
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT student_id, full_name FROM students LIMIT 1"))
        student = res.mappings().first()
        if not student:
            print("No students found in DB!")
            return
        
        sid = student['student_id']
        sname = student['full_name']
        print(f"Testing with Student: {sname} ({sid})")

        # Scenario 1: Book Search
        print("\n--- [Scenario 1: Book Search] ---")
        msg = "Tìm cho mình sách tiếng anh về IELTS"
        resp = await chat_service.process_message(db, msg, session_id, sid)
        print(f"User: {msg}")
        print(f"Intent: {resp['intent']}")
        print(f"Entities: {resp['entities']}")
        print(f"Reply: {resp['reply'][:200]}...")

        # Scenario 2: Debt Check (direct intent)
        print("\n--- [Scenario 2: Debt Check] ---")
        msg = "Kiểm tra nợ phạt của mình"
        resp = await chat_service.process_message(db, msg, session_id, sid)
        print(f"User: {msg}")
        print(f"Intent: {resp['intent']}")
        print(f"Reply: {resp['reply'][:200]}...")

        # Scenario 3: Policy Query
        print("\n--- [Scenario 3: Policy Query] ---")
        msg = "Mượn sách tối đa bao lâu?"
        resp = await chat_service.process_message(db, msg, session_id, sid)
        print(f"User: {msg}")
        print(f"Intent: {resp['intent']}")
        print(f"Reply: {resp['reply'][:200]}...")

        # Scenario 5: Return Book (Smart Procedure)
        print("\n--- [Scenario 5: Return Book] ---")
        msg = "Thủ tục trả sách thế nào?"
        resp = await chat_service.process_message(db, msg, session_id, sid)
        print(f"User: {msg}")
        print(f"Intent: {resp['intent']}")
        print(f"Reply: {resp['reply'][:200]}...")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_scenarios())
