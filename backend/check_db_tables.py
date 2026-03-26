import asyncio, os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

async def check_columns(table_name):
    load_dotenv('.env')
    DB_URL = os.getenv('DATABASE_URL')
    if not DB_URL: return ""
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        result = await conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' ORDER BY ordinal_position"))
        cols = [f"{row[0]} ({row[1]})" for row in result]
        res = f"Columns in {table_name}: {', '.join(cols)}\n"
    await engine.dispose()
    return res

async def main():
    p = await check_columns('policy_chunks')
    c = await check_columns('chat_history')
    with open('db_inspect.txt', 'w', encoding='utf-8') as f:
        f.write(p + c)

if __name__ == "__main__":
    asyncio.run(main())
