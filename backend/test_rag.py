import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.services.rag_service import rag_service
from app.services.embedding_service import embedding_service
from app.config import settings

async def test_search():
    # Setup DB
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    query = "mượn sách tối đa được bao nhiêu ngày ạ"
    print(f"Query: {query}")
    
    # Embed
    try:
        emb = await embedding_service.embed(query)
        if not emb:
            print("Embedding failed: Service returned None")
            return
        print(f"Got embedding with dimension: {len(emb)}")
    except Exception as e:
        print(f"Embedding service exception: {e}")
        return
        
    async with async_session() as session:
        # Check raw count from WITHIN the script
        from sqlalchemy import text
        cnt_res = await session.execute(text("SELECT COUNT(*) FROM policy_chunks"))
        print(f"Direct DB Count from Script: {cnt_res.scalar()}")
        
        # Check if we can select anything with distance
        try:
            vec_str = "[" + ",".join(str(v) for v in emb) + "]"
            raw_res = await session.execute(text("SELECT id FROM policy_chunks ORDER BY embedding <=> cast(:v as vector) LIMIT 1"), {"v": vec_str})
            print(f"Raw SQL distance query result: {raw_res.all()}")
        except Exception as e:
            print(f"Raw SQL distance query failed: {e}")

        # Test 1: Policy
        print("\n--- Testing Policy Search ---")
        try:
            chunks = await rag_service.search_policy(session, emb, top_k=3)
            print(f"Search policy returned {len(chunks)} chunks")
            for i, c in enumerate(chunks):
                print(f"{i+1}. Similarity: {c.get('similarity')} | Text: {c.get('text')[:100]}...")
        except Exception as e:
            print(f"Search policy exception: {e}")
            
        # Test 2: Books
        print("\n--- Testing Book Search ---")
        try:
            books = await rag_service.search_books_semantic(session, emb, top_k=3)
            print(f"Search books returned {len(books)} books")
            for i, b in enumerate(books):
                print(f"{i+1}. Similarity: {b.get('similarity')} | Title: {b.get('title')}")
        except Exception as e:
            print(f"Search book exception: {e}")

if __name__ == "__main__":
    import sys
    # Add project root to path for relative imports if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    asyncio.run(test_search())
