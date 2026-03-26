"""
SmartLib - Book Indexer (BGE-M3 -> pgvector)

Populate `books.embedding` (vector(1024)) for semantic search.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.models.book import Book
from app.rag.embeddings import EmbeddingsService


def build_book_embedding_text(book: Book) -> str:
    parts = [
        book.title,
        book.author or "",
        book.subject_category or "",
        book.description or "",
    ]
    # Keep it compact but informative
    return " ".join([p.strip() for p in parts if p and p.strip()])


async def main():
    print("=" * 60)
    print("SmartLib - Book Indexer (pgvector 1024d)")
    print("=" * 60)
    print(f"DB: {settings.database_url}")

    engine = create_async_engine(settings.database_url, echo=False)
    SessionMaker = async_sessionmaker(engine, expire_on_commit=False)
    
    print("\n⏳ Loading local AITeamVN/Vietnamese_Embedding model...")
    emb_model = EmbeddingsService.get_embeddings()

    async with SessionMaker() as db:
        books = (await db.execute(select(Book))).scalars().all()
        print(f"Found {len(books)} books to re-index")

        indexed = 0
        error_count = 0

        for i, book in enumerate(books):
            text_to_embed = build_book_embedding_text(book)
            if not text_to_embed.strip():
                continue

            print(f"   [{i+1}/{len(books)}] Embedding: {book.title[:40]}...", end=" ")
            
            try:
                # Use local model
                embedding = await asyncio.to_thread(emb_model.embed_query, text_to_embed)
                
                if embedding:
                    book.embedding = embedding
                    indexed += 1
                    print("✅ (1024d)")
                else:
                    print("❌ (Empty result)")
                    error_count += 1
            except Exception as e:
                print(f"❌ (Error: {e})")
                error_count += 1

            # Commit periodically
            if (i + 1) % 20 == 0:
                await db.commit()

        await db.commit()

    await engine.dispose()
    print(f"\n🎉 Done. Successfully re-indexed: {indexed}, Errors: {error_count}")


if __name__ == "__main__":
    asyncio.run(main())

