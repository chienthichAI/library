import asyncio
from app.database import async_session_maker
from sqlalchemy import text

async def delete_student():
    async with async_session_maker() as session:
        await session.execute(text("DELETE FROM face_embeddings WHERE student_id='QE190047'"))
        await session.execute(text("DELETE FROM students WHERE student_id='QE190047'"))
        await session.commit()
        print("Deleted student QE190047")

if __name__ == "__main__":
    asyncio.run(delete_student())
