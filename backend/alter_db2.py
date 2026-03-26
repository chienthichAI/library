import asyncio
import sys
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
sys.path.insert(0, ".")
from app.config import settings

async def main():
    print("Database URL:", settings.database_url)
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        print("Emptying table...")
        await conn.execute(text("DELETE FROM policy_chunks"))
        
        print("Altering policy_chunks table to vector(1024)...")
        await conn.execute(text("ALTER TABLE policy_chunks ALTER COLUMN embedding TYPE vector(1024)"))
    await engine.dispose()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
