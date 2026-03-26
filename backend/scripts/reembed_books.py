import asyncio
import sys
import os

# Add parent directory to path to reach app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import async_session_maker
from app.models.book import Book
from app.rag.embeddings import EmbeddingsService
from sqlalchemy import select, update
from loguru import logger

async def reembed_all_books():
    logger.info("Starting book re-embedding process...")
    
    # 1. Initialize Embeddings Service
    embeddings_model = EmbeddingsService.get_embeddings()
    
    async with async_session_maker() as session:
        # 2. Fetch all books
        stmt = select(Book)
        result = await session.execute(stmt)
        books = result.scalars().all()
        
        logger.info(f"Found {len(books)} books to process.")
        
        updated_count = 0
        for b in books:
            # Create a rich description for embedding
            content = f"Tiêu đề: {b.title}. Tác giả: {b.author or 'Khuyết danh'}. Chủ đề: {b.subject_category or 'N/A'}. Mô tả: {b.description or ''}"
            
            # 3. Generate embedding
            try:
                # Use aembed_query for async if available, else embed_query
                embedding = await asyncio.to_thread(embeddings_model.embed_query, content)
                
                # Verify dimension
                if len(embedding) != 768:
                    logger.error(f"Dimension mismatch for {b.book_id}: expected 768, got {len(embedding)}")
                    continue
                
                # 4. Update the book
                b.embedding = embedding
                updated_count += 1
                
                if updated_count % 10 == 0:
                    logger.info(f"Progress: {updated_count}/{len(books)} books embedded.")
                    await session.commit()
            except Exception as e:
                logger.error(f"Failed to embed book {b.book_id}: {e}")
        
        await session.commit()
        logger.info(f"Process completed. Successfully re-embedded {updated_count} books.")

if __name__ == "__main__":
    asyncio.run(reembed_all_books())
