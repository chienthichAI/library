"""
SmartLib - Library Policy Indexing Script (ORM Version)
"""
import asyncio
import sys
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# Import config and models
from app.config import settings
from app.models.policy_chunk import PolicyChunk
from app.rag.embeddings import EmbeddingsService


# === Configuration ===
POLICY_PATH = Path(__file__).parent.parent / "library_policy.md"
CHUNK_SIZE = 512  # SBERT works better with smaller chunks


def load_policy_text(filepath: Path) -> str:
    if not filepath.exists():
        raise FileNotFoundError(f"Policy file not found: {filepath}")
    return filepath.read_text(encoding="utf-8")


def chunk_by_section(text: str, max_chunk_size: int = CHUNK_SIZE) -> list[dict]:
    chunks = []
    current_section = "Tổng quan"
    chunk_index = 0
    sections = re.split(r"\n(?=#{1,2} )", text)

    for section in sections:
        if not section.strip():
            continue
        header_match = re.match(r"^(#{1,3})\s+(.+)", section.strip())
        if header_match:
            current_section = header_match.group(2).strip()

        if len(section) <= max_chunk_size:
            chunks.append({
                "chunk_text": section.strip(),
                "section_title": current_section,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        else:
            paragraphs = section.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) <= max_chunk_size:
                    buffer += ("\n\n" if buffer else "") + para
                else:
                    if buffer:
                        chunks.append({
                            "chunk_text": buffer.strip(),
                            "section_title": current_section,
                            "chunk_index": chunk_index,
                        })
                        chunk_index += 1
                    buffer = para
            if buffer.strip():
                chunks.append({
                    "chunk_text": buffer.strip(),
                    "section_title": current_section,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
    return chunks


async def main():
    print("=" * 60)
    print("SmartLib - Library Policy Indexer (Standardized 1024d)")
    print("=" * 60)

    # 1. Load policy file
    print(f"\n📄 Loading policy from: {POLICY_PATH}")
    policy_text = load_policy_text(POLICY_PATH)

    # 2. Chunk text
    CHUNK_SIZE = 768  # BGE-M3 handles larger context better
    chunks = chunk_by_section(policy_text, max_chunk_size=CHUNK_SIZE)
    print(f"✂️  Created {len(chunks)} chunks")

    # 3. Connection & Embeddings
    engine = create_async_engine(settings.database_url, echo=False)
    SessionMaker = async_sessionmaker(engine, expire_on_commit=False)
    
    print("\n⏳ Initializing Embedding Service (AITeamVN/Vietnamese_Embedding)...")
    emb_model = EmbeddingsService.get_embeddings()

    # 4. Clear existing
    print(f"\n🗑️  Clearing existing policy chunks...")
    async with SessionMaker() as db:
        await db.execute(delete(PolicyChunk)) # Clear all for clean re-index
        await db.commit()

    # 5. Embed and Index
    print(f"⚡ Indexing {len(chunks)} chunks...")
    success_count = 0

    async with SessionMaker() as db:
        for i, chunk_data in enumerate(chunks):
            print(f"   [{i+1}/{len(chunks)}] {chunk_data['section_title'][:35]}...", end=" ")
            
            try:
                # Use LangChain embed_query (sync) or aembed_query (async)
                embedding = await asyncio.to_thread(emb_model.embed_query, chunk_data["chunk_text"])
                
                if embedding:
                    chunk = PolicyChunk(
                        chunk_text=chunk_data["chunk_text"],
                        embedding=embedding,
                        chunk_index=chunk_data["chunk_index"],
                        section_title=chunk_data["section_title"],
                        source='library_policy'
                    )
                    db.add(chunk)
                    success_count += 1
                    print("✅ (1024d)")
                else:
                    print("❌ (No embedding)")
            except Exception as e:
                print(f"❌ (Error: {e})")
        
        await db.commit()

    print(f"\n🎉 Done! Indexed {success_count}/{len(chunks)} policy chunks with 1024d vectors.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

