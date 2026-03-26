import asyncio, os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def main():
    load_dotenv('.env')
    DB_URL = os.getenv('DATABASE_URL')
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        try:
            await conn.execute(text('ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS embedding vector(768)'))
            print('Successfully added embedding column to chat_history')
        except Exception as e:
            print(f"Error adding column: {e}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
