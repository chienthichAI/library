
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import sys
import os

sys.path.append(os.getcwd())

from app.config import settings

async def check():
    print(f"Checking database at: {settings.database_url}")
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT atttypmod 
            FROM pg_attribute 
            WHERE attrelid = 'chat_history'::regclass 
            AND attname = 'embedding';
        """))
        row = result.fetchone()
        if row:
            print(f"chat_history.embedding typmod: {row[0]}")
        else:
            print("Column not found")

if __name__ == "__main__":
    asyncio.run(check())
