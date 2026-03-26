import asyncio, os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def main():
    load_dotenv('.env')
    DB_URL = os.getenv('DATABASE_URL')
    if not DB_URL:
        print("DATABASE_URL not found")
        return
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        try:
            # PostgreSQL query to check column type info
            res = await conn.execute(text("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chat_history'::regclass AND attname = 'embedding'"))
            typmod = res.scalar()
            # attmod for vector is dimension + offset
            if typmod and typmod != -1:
                print(f"Chat history embedding dimension: {typmod}")
            else:
                print("Chat history embedding dimension: (flexible)")
        except Exception as e:
            print(f"Error checking dimension: {e}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
