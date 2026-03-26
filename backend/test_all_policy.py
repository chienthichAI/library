import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.intent_service import intent_service
from app.services.rag_service import rag_service
from app.services.embedding_service import embedding_service
from app.config import settings

async def test_all_policy_cases():
    # Setup DB
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    test_queries = [
        "Được mượn tối đa bao nhiêu quyển sách cùng lúc?",
        "Sách tài liệu tham khảo đặc biệt mượn được mấy ngày?",
        "Gia hạn sách được mấy lần và thêm được bao nhiêu ngày?",
        "Phí phạt trả sách quá hạn là bao nhiêu tiền một ngày?",
        "Làm mất sách thì phải đền bù thế nào?",
        "Phòng học nhóm mượn được tối đa bao lâu?",
        "Thư viện mở cửa lúc mấy giờ và đóng lúc mấy giờ?",
        "Có được mang đồ ăn vào phòng đọc không?",
        "Làm thế nào để được mượn sách tại kiosk?"
    ]
    
    print("="*80)
    print(f"{'QUESTION':<50} | {'INTENT':<15} | {'FOUND CHUNKS'}")
    print("="*80)
    
    async with async_session() as session:
        for query in test_queries:
            # 1. Intent classification
            intent_res = await intent_service.classify(query)
            intent = intent_res.get("intent", "unknown")
            
            # 2. Embedding
            emb = await embedding_service.embed(query)
            
            # 3. RAG Search
            if emb:
                chunks = await rag_service.search_policy(session, emb, top_k=2)
                chunk_info = f"{len(chunks)} (Top sim: {chunks[0]['similarity']:.3f})" if chunks else "0"
                
                # Check if the best chunk actually contains keywords related to the query
                # (Simple heuristic for test verification)
                best_text = chunks[0]['text'] if chunks else ""
            else:
                chunk_info = "Embedding Error"
                best_text = ""
                
            print(f"{query[:48]:<50} | {intent:<15} | {chunk_info}")
            if chunks:
                print(f"   -> Top Chunk: {best_text[:120]}...")
            print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_all_policy_cases())
