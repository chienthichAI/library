"""
SmartLib - RAG (Retrieval-Augmented Generation) Service
Handles semantic search against books and policy chunks using pgvector.
"""
from typing import List, Optional, Dict, Any
from cachetools import TTLCache
from underthesea import word_tokenize, pos_tag
import re

from app.models.student import Student
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


class RAGService:
    """
    Retrieval service using pgvector for semantic similarity search.
    
    Supports:
    - Semantic book search via bge-m3 embeddings
    - Keyword fallback search for books
    - Policy document retrieval
    """

    async def search_books_semantic(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search books by vector similarity using pgvector cosine distance.
        
        Args:
            db: Database session
            query_embedding: 1024-dim query embedding from bge-m3
            top_k: Maximum results to return
            
        Returns:
            List of book dicts sorted by relevance
        """
        try:
            # pgvector cosine similarity: 1 - cosine_distance
            sql = text("""
                SELECT 
                    b.book_id, b.title, b.author, b.subject_category, 
                    b.smart_category, b.description, b.status, b.publisher,
                    b.publication_year, b.language,
                    t.due_date, t.days_overdue,
                    1 - (b.embedding <=> cast(:query_vec as vector)) AS similarity
                FROM books b
                LEFT JOIN transactions t ON (
                    b.book_id = t.book_id
                    AND t.status IN ('ACTIVE', 'OVERDUE')
                )
                WHERE b.embedding IS NOT NULL
                ORDER BY b.embedding <=> cast(:query_vec as vector)
                LIMIT :top_k
            """)

            # Convert list to pgvector format string
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            result = await db.execute(sql, {"query_vec": vec_str, "top_k": top_k})
            rows = result.mappings().all()

            books = []
            for row in rows:
                books.append({
                    "book_id": row["book_id"],
                    "title": row["title"],
                    "author": row["author"] or "Không rõ",
                    "category": row["smart_category"] or row["subject_category"] or "Chưa phân loại",
                    "description": row["description"] or "",
                    "status": row["status"],
                    "publisher": row["publisher"] or "",
                    "publication_year": row["publication_year"],
                    "language": row["language"] or "vi",
                    "similarity": round(float(row["similarity"]), 3),
                })
            
            logger.info(f"Semantic book search → {len(books)} results")
            return books
        except Exception as e:
            logger.error(f"Semantic book search failed: {e}")
            await db.rollback()  # Reset transaction state
            return []

    async def search_books_hybrid(
        self,
        db: AsyncSession,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        entities: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining:
        1. Exact ID Match (Highest Priority)
        2. Full-Text Search (Keyword)
        3. Vector Similarity (Semantic)
        
        Using Reciprocal Rank Fusion (RRF) logic.
        """
        try:
            # 1. Prepare query string and embedding
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            
            # 2. Execute a single powerful hybrid SQL query
            # We use CTE (Common Table Expressions) to rank each method
            sql = text("""
                WITH semantic_hits AS (
                    SELECT book_id, 1 - (embedding <=> cast(:query_vec as vector)) as score,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> cast(:query_vec as vector)) as rank
                    FROM books
                    WHERE embedding IS NOT NULL
                    LIMIT 20
                ),
                keyword_hits AS (
                    -- Rank based on Full-Text Search and high-precision category matches
                    SELECT book_id, 
                           ts_rank(fts, plainto_tsquery('simple', :query_text_ts)) +
                           ts_rank(fts, phraseto_tsquery('simple', :query_text_phrase)) +
                           (CASE 
                                WHEN subject_category ILIKE :topic_exact THEN 1.0
                                WHEN title ILIKE :topic_exact THEN 1.0
                                ELSE 0 
                            END) as internal_score,
                           ROW_NUMBER() OVER (
                               ORDER BY (
                                   ts_rank(fts, plainto_tsquery('simple', :query_text_ts)) + 
                                   ts_rank(fts, phraseto_tsquery('simple', :query_text_phrase)) +
                                   (CASE WHEN subject_category ILIKE :topic_exact THEN 1.0 ELSE 0 END)
                               ) DESC
                           ) as rank
                    FROM books
                    WHERE fts @@ plainto_tsquery('simple', :query_text_ts)
                       OR fts @@ phraseto_tsquery('simple', :query_text_phrase)
                       OR subject_category ILIKE :topic_exact
                       OR title ILIKE :topic_exact
                    LIMIT 20
                ),
                exact_hits AS (
                    -- Highest priority for specific book titles or IDs mentioned
                    SELECT book_id, 10.0 as score, 1 as rank
                    FROM books
                    WHERE book_id = :query_text 
                       OR title ILIKE :query_text_exact
                       OR (title ILIKE :topic_exact AND :topic_exact != '')
                )
                SELECT 
                    b.book_id, b.title, b.author, b.subject_category, 
                    b.smart_category, b.description, b.status, b.publisher,
                    b.publication_year, b.language,
                    t.due_date, t.days_overdue,
                    -- Combine scores: Boost exact matches, then mix semantic/keyword
                    -- Add a tiny boost for English/Vietnamese relevance based on entities
                    COALESCE(e.score, 0) + 
                    (1.0 / (60 + COALESCE(s.rank, 500))) + 
                    (1.0 / (60 + COALESCE(k.rank, 500))) +
                    (CASE WHEN :topic_exact ILIKE '%anh%' AND b.subject_category ILIKE '%Nhật%' THEN -0.5 ELSE 0 END) AS final_score
                FROM books b
                LEFT JOIN semantic_hits s ON b.book_id = s.book_id
                LEFT JOIN keyword_hits k ON b.book_id = k.book_id
                LEFT JOIN exact_hits e ON b.book_id = e.book_id
                LEFT JOIN transactions t ON (
                    b.book_id = t.book_id 
                    AND t.status IN ('ACTIVE', 'OVERDUE')
                )
                WHERE s.book_id IS NOT NULL OR k.book_id IS NOT NULL OR e.book_id IS NOT NULL
                ORDER BY final_score DESC
                LIMIT :top_k
            """)

            # Smart tokenization and filtering using POS tagging
            tagged_tokens = pos_tag(query_text.lower())
            
            # Words to keep (Nouns, Verbs, Adjectives, etc.)
            allowed_tags = {'N', 'NP', 'V', 'A', 'M', 'Np', 'Nc'}
            
            # First pass: Filter tokens accurately based on POS
            final_tokens = []
            for token, tag in tagged_tokens:
                if tag in allowed_tags:
                    cleaned_token = re.sub(r'[^\w\s]', '', token).strip()
                    if cleaned_token:
                        final_tokens.append(cleaned_token)
            
            # Second pass: Inject high-precision keywords from AI entities
            if entities:
                for key, val in entities.items():
                    if isinstance(val, str) and val.strip():
                        # We trust AI entities more, so we add them as is
                        final_tokens.append(val.strip())
                    elif isinstance(val, list):
                        final_tokens.extend([v.strip() for v in val if isinstance(v, str)])
            
            # Clean entities for matching
            topic_entity = entities.get("topic", "") if entities else ""
            title_entity = entities.get("book_title", "") if entities else ""
            
            # Use unique tokens for TS query
            ts_query = " ".join(list(set(final_tokens)))
            if not ts_query.strip():
                ts_query = "sách"
            
            # Collect and join entities for phrase matching
            phrase_query = ""
            if entities:
                phrase_query = " ".join([v for v in entities.values() if isinstance(v, str)])
            
            # Exact match pattern
            params = {
                "query_vec": vec_str,
                "query_text_ts": ts_query,
                "query_text_phrase": phrase_query or query_text, # Precise phrase matching
                "query_text": query_text,
                "query_text_exact": f"%{query_text}%",
                "topic_exact": f"%{topic_entity}%" if topic_entity else "",
                "top_k": top_k
            }

            result = await db.execute(sql, params)
            rows = result.mappings().all()

            books = []
            for row in rows:
                books.append({
                    "book_id": row["book_id"],
                    "title": row["title"],
                    "author": row["author"] or "Không rõ",
                    "category": row["smart_category"] or row["subject_category"] or "Chưa phân loại",
                    "description": row["description"] or "",
                    "status": row["status"],
                    "publisher": row["publisher"] or "",
                    "publication_year": row["publication_year"],
                    "language": row["language"] or "vi",
                    "due_date": str(row["due_date"]) if row["due_date"] else None,
                    "days_overdue": row["days_overdue"] or 0,
                    "similarity": round(float(row["final_score"]), 4),
                })
            
            logger.info(f"Hybrid book search '{query_text}' → {len(books)} results")
            return books
        except Exception as e:
            logger.error(f"Hybrid book search failed: {e}")
            await db.rollback()
            # Fallback to pure semantic if RRF fails
            return await self.search_books_semantic(db, query_embedding, top_k)

    async def search_policy(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Search library policy chunks by vector similarity.
        
        Args:
            db: Database session
            query_embedding: 1024-dim query embedding
            top_k: Maximum chunks to retrieve
            
        Returns:
            List of policy chunk dicts
        """
        try:
            sql = text("""
                SELECT 
                    id, chunk_text, section_title, chunk_index,
                    1 - (embedding <=> cast(:query_vec as vector)) AS similarity
                FROM policy_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> cast(:query_vec as vector)
                LIMIT :top_k
            """)

            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            result = await db.execute(sql, {"query_vec": vec_str, "top_k": top_k})
            rows = result.mappings().all()

            chunks = []
            for row in rows:
                chunks.append({
                    "id": row["id"],
                    "text": row["chunk_text"],
                    "section": row["section_title"] or "",
                    "similarity": round(float(row["similarity"]), 3),
                })
            
            logger.info(f"Policy search → {len(chunks)} chunks retrieved")
            return chunks
        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            await db.rollback()  # Reset transaction state
            return []

    def format_books_for_context(self, books: List[Dict[str, Any]]) -> str:
        """Format book list into readable context string for LLM prompt."""
        if not books:
            return "Không tìm thấy sách phù hợp trong hệ thống."

        lines = ["| STT | Tên sách | Tác giả | Trạng thái | Chi tiết mượn |\n|:---:|:---|:---|:---|:---|"]
        for i, b in enumerate(books, 1):
            status_map = {
                "AVAILABLE": "✅ Sẵn sàng",
                "BORROWED": "❌ Đã mượn",
                "RESERVED": "🔒 Giữ chỗ",
                "DAMAGED": "⚠️ Sửa chữa",
                "LOST": "🚫 Mất",
            }
            status_str = status_map.get(str(b["status"]), b["status"])
            
            borrow_info = "-"
            if b["status"] == "BORROWED" and b["due_date"]:
                borrow_info = f"Hạn trả: {b['due_date']}"
                if b["days_overdue"] > 0:
                    borrow_info += f" (🚨 Quá hạn {b['days_overdue']} ngày)"
            elif b["status"] == "RESERVED":
                borrow_info = "Đang chờ sinh viên khác lấy sách"

            lines.append(
                f"| {i} | **{b['title']}** | {b['author']} | {status_str} | {borrow_info} |"
            )
        return "\n".join(lines)

    def format_policy_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format policy chunks into readable context string for LLM prompt."""
        if not chunks:
            return "Không tìm thấy thông tin quy định liên quan."

        parts = []
        for chunk in chunks:
            parts.append(chunk["text"])
        return "\n\n---\n\n".join(parts)


# Singleton instance
rag_service = RAGService()
