import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

async def test_db():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found!")
        return

    # Replace asyncpg driver if needed for local test
    engine = create_async_engine(db_url)
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT version()"))
        print(f"DB Version: {result.scalar()}")

if __name__ == "__main__":
    asyncio.run(test_db())
