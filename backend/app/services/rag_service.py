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
    """

    async def search_books_semantic(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        try:
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
            return books
        except Exception as e:
            logger.error(f"Semantic book search failed: {e}")
            await db.rollback()
            return []

    async def search_books_hybrid(
        self,
        db: AsyncSession,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        entities: Optional[Dict[str, Any]] = None,
        intent: str = "book_search"
    ) -> List[Dict[str, Any]]:
        """
        Refined Hybrid search (Pro-Max):
        Combines Semantic (pgvector), Lexical (FTS - Websearch), and Metadata Boosting.
        """
        try:
            # 1. Input Preparation
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            
            # Use entities from AI/LLM instead of hardcoded rules
            lang_boost = None
            target_lang_keyword = None
            
            if entities:
                lang_boost = entities.get("language")
                # Map common codes to display names if needed for exact search
                lang_map = {"en": "Tiếng Anh", "vi": "Tiếng Việt", "ja": "Tiếng Nhật", "zh": "Tiếng Trung"}
                target_lang_keyword = lang_map.get(lang_boost) if lang_boost else None

            # 2. Text Preprocessing for FTS
            # Clean and expand search terms - Extract meaningful tokens using NLP
            tagged_tokens = pos_tag(query_text.lower())
            allowed_tags = {'N', 'NP', 'V', 'A', 'M', 'Np', 'Nc'}
            
            # Words to ignore to avoid noise, but avoiding a large hardcoded list 
            # We rely on POS tag 'N', 'NP' for nouns which are usually our search terms
            meaningful_keys = []
            for token, tag in tagged_tokens:
                if tag in allowed_tags:
                    clean = re.sub(r'[^\w\s]', '', token).strip()
                    if clean and len(clean) > 1:
                        meaningful_keys.append(clean)
            
            topic_entity = ""
            if entities and isinstance(entities, dict):
                topic_entity = entities.get("topic", "") or entities.get("book_title", "")
            
            # Combine meaningful keys for FTS
            if topic_entity:
                # If we have a clear topic from AI, use it as the core FTS query
                fts_query_str = topic_entity
                # If there are other meaningful keys not in the topic, add them lightly
                extra_keys = [k for k in meaningful_keys if k.lower() not in topic_entity.lower()]
                if extra_keys:
                    fts_query_str += " " + " ".join(extra_keys)
            else:
                fts_query_str = " ".join(list(set(meaningful_keys))) if meaningful_keys else query_text
            
            if not fts_query_str.strip():
                fts_query_str = query_text

            # 3. SQL Construction with RRF-inspired scoring
            sql = text("""
                WITH semantic_hits AS (
                    SELECT book_id, 1 - (embedding <=> cast(:query_vec as vector)) as sim,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> cast(:query_vec as vector)) as rank
                    FROM books
                    WHERE embedding IS NOT NULL
                    LIMIT 40
                ),
                keyword_hits AS (
                    SELECT book_id, 
                           ts_rank_cd(fts, websearch_to_tsquery('simple', :fts_query)) as rank_score,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(fts, websearch_to_tsquery('simple', :fts_query)) DESC
                           ) as rank
                    FROM books
                    WHERE fts @@ websearch_to_tsquery('simple', :fts_query)
                       OR (title ILIKE :topic_exact AND :topic_exact != '' AND :intent != 'book_search')
                       OR (subject_category ILIKE :topic_exact AND :topic_exact != '')
                    LIMIT 40
                ),
                exact_hits AS (
                    SELECT book_id, 
                           (CASE 
                                WHEN title ILIKE :exact_match THEN 6.0 -- Exact user phrase match
                                WHEN title ILIKE :topic_exact AND :topic_exact != '' AND :intent != 'book_search' THEN 4.0 -- AI Topic phrase match
                                WHEN title ILIKE :lang_kw AND :lang_kw != '' THEN 2.0 -- Language keyword match
                                WHEN subject_category ILIKE :topic_exact AND :topic_exact != '' THEN 4.5 -- AI Category match (Boosted for book_search)
                                ELSE 0.5
                            END) as match_score
                    FROM books
                    WHERE (title ILIKE :exact_match AND :intent != 'book_search')
                       OR (title ILIKE :topic_exact AND :topic_exact != '' AND :intent != 'book_search')
                       OR (title ILIKE :lang_kw AND :lang_kw != '')
                       OR (subject_category ILIKE :topic_exact AND :topic_exact != '')
                )
                SELECT 
                    b.book_id, b.title, b.author, b.subject_category, 
                    b.smart_category, b.description, b.status, b.publisher,
                    b.publication_year, b.language,
                    t.due_date, t.days_overdue,
                    -- Advanced Scoring Formula
                    COALESCE(e.match_score, 0) + 
                    (1.0 / (60 + COALESCE(s.rank, 500))) * 12 + -- Semantic weight (Heaviest)
                    (1.0 / (60 + COALESCE(k.rank, 500))) * 6 +  -- Keyword weight
                    (CASE WHEN b.language = :lang_boost THEN 0.5 ELSE 0 END) +
                    (CASE WHEN b.status = 'AVAILABLE' THEN 0.3 ELSE 0 END) AS final_score
                FROM books b
                LEFT JOIN semantic_hits s ON b.book_id = s.book_id
                LEFT JOIN keyword_hits k ON b.book_id = k.book_id
                LEFT JOIN exact_hits e ON b.book_id = e.book_id
                LEFT JOIN transactions t ON (
                    b.book_id = t.book_id 
                    AND t.status IN ('ACTIVE', 'OVERDUE')
                )
                WHERE s.book_id IS NOT NULL 
                   OR k.book_id IS NOT NULL 
                   OR e.book_id IS NOT NULL
                   OR (b.title ILIKE :topic_exact AND :topic_exact != '')
                ORDER BY final_score DESC
                LIMIT :top_k
            """)
            
            params = {
                "query_vec": vec_str,
                "fts_query": fts_query_str,
                "exact_match": f"%{query_text}%",
                "topic_exact": f"%{topic_entity}%" if topic_entity else f"%{query_text}%",
                "lang_kw": f"%{target_lang_keyword}%" if target_lang_keyword else "",
                "lang_boost": lang_boost,
                "intent": intent,
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
            
            logger.info(f"Hybrid Pro-Max Search '{query_text}' -> {len(books)} results (FTS: {fts_query_str})")
            return books
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            await db.rollback()
            return await self.search_books_semantic(db, query_embedding, top_k)

    async def search_policy(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
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
            return chunks
        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            await db.rollback()
            return []

    def format_books_for_context(self, books: List[Dict[str, Any]]) -> str:
        if not books:
            return "Không tìm thấy sách phù hợp trong hệ thống."

        lines = ["| STT | Tên sách | Tác giả | Thể loại | Mô tả | Trạng thái | Chi tiết mượn |\n|:---:|:---|:---|:---|:---|:---|:---|"]
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
            
            # Shorten description for context
            desc = b["description"] or "Không có mô tả"
            if len(desc) > 120:
                desc = desc[:117] + "..."
            
            category = b["category"] or "Chưa phân loại"
            
            lines.append(
                f"| {i} | **{b['title']}** | {b['author']} | {category} | {desc} | {status_str} | {borrow_info} |"
            )
        return "\n".join(lines)

    def format_policy_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "Không tìm thấy thông tin quy định liên quan."
        return "\n\n---\n\n".join([c["text"] for c in chunks])


# Singleton instance
rag_service = RAGService()
