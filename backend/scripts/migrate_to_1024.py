import asyncio
import sys
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

async def migrate():
    print("=" * 60)
    print("SmartLib - Database Migration: 768d -> 1024d")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=True)
    
    async with engine.begin() as conn:
        print("\n🗑️  Dropping old 768-dim embedding columns and indexes...")
        
        # Books table
        await conn.execute(text("DROP INDEX IF EXISTS books_embedding_idx;"))
        await conn.execute(text("ALTER TABLE books DROP COLUMN IF EXISTS embedding;"))
        await conn.execute(text("ALTER TABLE books ADD COLUMN embedding vector(1024);"))
        
        # Policy chunks table
        await conn.execute(text("ALTER TABLE policy_chunks DROP COLUMN IF EXISTS embedding;"))
        await conn.execute(text("ALTER TABLE policy_chunks ADD COLUMN embedding vector(1024);"))
        
        print("\n✅ New 1024-dim columns added successfully!")
        
    await engine.dispose()
    print("\n🎉 Migration complete. You can now run the indexing scripts.")

if __name__ == "__main__":
    asyncio.run(migrate())
