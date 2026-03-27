
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def check():
    url = "postgresql+asyncpg://postgres.gvgdispqeesizkgzduik:Chiendz098!@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        # Check policy_chunks
        res = await conn.execute(text("""
            SELECT column_name, udt_name, character_maximum_length 
            FROM information_schema.columns 
            WHERE table_name = 'policy_chunks' AND column_name = 'embedding';
        """))
        print(f"Policy Chunks Column: {res.fetchone()}")
        
        # Check actual data dim in a row
        res = await conn.execute(text("SELECT array_length(embedding::float[], 1) FROM policy_chunks WHERE embedding IS NOT NULL LIMIT 1;"))
        print(f"Actual Data Dim in Policy Chunks: {res.scalar()}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check())
