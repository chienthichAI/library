import asyncio
from app.database import async_session_maker
from sqlalchemy import text
import logging

async def check():
    logging.basicConfig(level=logging.INFO)
    async with async_session_maker() as session:
        res = await session.execute(text("SELECT id, student_id FROM face_embeddings LIMIT 10"))
        print('All Embeddings:', res.all())
        
        res_students = await session.execute(text("SELECT student_id FROM students LIMIT 10"))
        print('All Students:', res_students.all())
        
if __name__ == "__main__":
    asyncio.run(check())
