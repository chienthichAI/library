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

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"


def build_book_embedding_text(book: Book) -> str:
    parts = [
        book.title,
        book.author or "",
        book.subject_category or "",
        book.description or "",
    ]
    # Keep it compact to reduce embedding cost/latency
    return " ".join([p.strip() for p in parts if p and p.strip()])


async def embed_text(client: httpx.AsyncClient, text: str) -> Optional[list[float]]:
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        print(f"  ⚠️  Embedding failed: {e}")
        return None


async def main():
    print("=" * 60)
    print("SmartLib - Book Indexer (pgvector)")
    print("=" * 60)
    print(f"DB: {settings.database_url}")

    engine = create_async_engine(settings.database_url, echo=False)
    SessionMaker = async_sessionmaker(engine, expire_on_commit=False)

    async with httpx.AsyncClient() as client:
        async with SessionMaker() as db:
            books = (await db.execute(select(Book))).scalars().all()
            print(f"Found {len(books)} books")

            indexed = 0
            skipped = 0

            for i, book in enumerate(books):
                # Skip if embedding already exists
                if getattr(book, "embedding", None) is not None:
                    skipped += 1
                    continue

                text = build_book_embedding_text(book)
                if not text.strip():
                    skipped += 1
                    continue

                print(
                    f"   [{i+1}/{len(books)}] Embedding...",
                    end=" ",
                )
                embedding = await embed_text(client, text)
                if embedding:
                    book.embedding = embedding
                    indexed += 1
                    print("[OK]")
                else:
                    print("[FAIL]")

                # Commit periodically to keep transactions small
                if (i + 1) % 20 == 0:
                    await db.commit()

            await db.commit()

    await engine.dispose()
    print(f"\nDone. Indexed: {indexed}, Skipped: {skipped}")


if __name__ == "__main__":
    asyncio.run(main())

