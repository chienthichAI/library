
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import sys
import os

sys.path.append(os.getcwd())
from app.config import settings

async def check_data():
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT id, session_id, role, content FROM chat_history ORDER BY created_at DESC LIMIT 5"))
        rows = result.fetchall()
        print(f"Latest {len(rows)} messages in chat_history:")
        for row in rows:
            print(f"ID: {row[0]}, Session: {row[1]}, Role: {row[2]}, Content: {row[3][:30]}...")

if __name__ == "__main__":
    asyncio.run(check_data())
