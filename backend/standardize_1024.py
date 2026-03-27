
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def migrate():
    url = "postgresql+asyncpg://postgres.gvgdispqeesizkgzduik:Chiendz098!@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        print("Starting global migration to 1024 dimensions...")
        
        # 1. Update books table
        print("Dropping and recreating books.embedding as vector(1024)...")
        await conn.execute(text("ALTER TABLE books DROP COLUMN IF EXISTS embedding;"))
        await conn.execute(text("ALTER TABLE books ADD COLUMN embedding vector(1024);"))
        
        # 2. Update policy_chunks table
        print("Dropping and recreating policy_chunks.embedding as vector(1024)...")
        await conn.execute(text("ALTER TABLE policy_chunks DROP COLUMN IF EXISTS embedding;"))
        await conn.execute(text("ALTER TABLE policy_chunks ADD COLUMN embedding vector(1024);"))

        # 3. Update chat_history table (if it has embedding)
        print("Checking chat_history for embedding column...")
        await conn.execute(text("ALTER TABLE chat_history DROP COLUMN IF EXISTS embedding;"))
        await conn.execute(text("ALTER TABLE chat_history ADD COLUMN embedding vector(1024);"))

        await conn.commit()
        print("Migration to 1024d complete!")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate())
