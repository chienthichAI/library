
import asyncio
from app.database import async_session_maker
from sqlalchemy import text
import logging

async def enable_pg_trgm():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Attempting to enable pg_trgm extension...")
    async with async_session_maker() as session:
        try:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            await session.commit()
            logger.info("Successfully enabled pg_trgm extension.")
        except Exception as e:
            logger.error(f"Failed to enable pg_trgm: {e}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(enable_pg_trgm())
