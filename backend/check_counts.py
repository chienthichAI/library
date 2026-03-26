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
            res = await conn.execute(text('SELECT count(*) FROM policy_chunks'))
            print(f'Policy chunks count: {res.scalar()}')
        except Exception as e:
            print(f"Error checking policy_chunks: {e}")
            
        try:
            res2 = await conn.execute(text('SELECT count(*) FROM chat_history'))
            print(f'Chat history count: {res2.scalar()}')
        except Exception as e:
            print(f"Error checking chat_history: {e}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
