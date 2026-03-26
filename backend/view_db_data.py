import asyncio, os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def main():
    load_dotenv('.env')
    DB_URL = os.getenv('DATABASE_URL')
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        print("Policy Chunks:")
        res = await conn.execute(text('SELECT section_title, chunk_text FROM policy_chunks LIMIT 3'))
        for row in res:
            print(f"- {row[0]}: {row[1][:100]}...")
            
        print("\nChat History:")
        res2 = await conn.execute(text('SELECT role, content FROM chat_history LIMIT 3'))
        for row in res2:
            print(f"- {row[0]}: {row[1][:100]}...")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
