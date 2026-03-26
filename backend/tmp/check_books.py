import asyncio
import asyncpg
import sys

async def main():
    try:
        conn = await asyncpg.connect('postgresql://postgres.gvgdispqeesizkgzduik:Chiendz098!@aws-1-ap-south-1.pooler.supabase.com:5432/postgres')
        rows = await conn.fetch("SELECT title, author, subject_category FROM books WHERE title ILIKE '%Cambridge IELTS%'")
        for r in rows:
            print(f"- {r[0]} | {r[1]} | {r[2]}")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
