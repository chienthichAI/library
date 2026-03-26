
import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://postgres.gvgdispqeesizkgzduik:Chiendz098!@aws-1-ap-south-1.pooler.supabase.com:5432/postgres')
    rows = await conn.fetch("SELECT language, subject_category FROM books LIMIT 5")
    for r in rows:
        print(f"Lang: {r[0]} | Cat: {r[1]}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
