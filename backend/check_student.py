import asyncio
from app.database import async_session_maker
from sqlalchemy import text

async def check():
    async with async_session_maker() as session:
        res1 = await session.execute(text("SELECT student_id, full_name FROM students WHERE student_id='QE190047'"))
        print('Student:', res1.all())
        
        res2 = await session.execute(text("SELECT id, student_id FROM face_embeddings WHERE student_id='QE190047'"))
        embeddings = res2.all()
        print('Embeddings:', embeddings)
        
        res3 = await session.execute(text("SELECT COUNT(*) FROM face_embeddings"))
        print('Total Embeddings:', res3.scalar())
        
if __name__ == "__main__":
    asyncio.run(check())
