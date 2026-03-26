import asyncio, os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def main():
    load_dotenv('.env')
    DB_URL = os.getenv('DATABASE_URL')
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        try:
            # PostgreSQL command for array length of vector
            res = await conn.execute(text('SELECT vector_dims(embedding) FROM policy_chunks WHERE embedding IS NOT NULL LIMIT 1'))
            dim = res.scalar()
            print(f'Embedding dimension: {dim}')
        except Exception as e:
            print(f"Error checking dimension: {e}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
