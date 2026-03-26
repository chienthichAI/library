"""
RAG Pipeline Summary module holding Document Ingestion and Query processing logic.
Pipeline Diagram: Document Loading -> Preprocessing -> Chunking -> Embeddings -> Vector DB -> Retrieval -> Generation

Supports two modes:
1. General mode: Direct LLM responses for library-related questions (no documents needed)
2. RAG mode: Full retrieval-augmented generation when documents are uploaded
"""
import os
from typing import List, Optional, Any, Dict
import logging
import re
from datetime import datetime, timedelta

import httpx
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.rag.loaders import DocumentLoaderService
from app.rag.preprocessors import Preprocessor
from app.rag.chunkers import ChunkerService
from app.rag.embeddings import EmbeddingsService
from app.rag.vector_store import VectorStoreService
from app.rag.retriever import RetrieverService
from app.rag.generator import GeneratorService

from app.database import async_session_maker
from app.models import PolicyChunk, ChatHistory
from sqlalchemy import select, insert

logger = logging.getLogger(__name__)


# System prompt for general library assistant mode
LIBRARY_SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh của thư viện SmartLib - Đại học FPT.

Nhiệm vụ chính:
- Hướng dẫn sinh viên quy trình mượn/trả sách tại Kiosk AI.
- Giải đáp thắc mắc về nội quy, giờ giấc, phí phạt và các dịch vụ thư viện.
- Tư vấn và gợi ý tài liệu học tập dựa trên chuyên ngành (IT, Kinh tế, Thiết kế, Ngôn ngữ).
- Kiểm tra hệ thống tồn kho của sách tại thư viện.

Thông tin thư viện SmartLib:
- Giờ mở cửa: 
  + Thứ 2 - Thứ 6: 7:30 - 21:00 (Phục vụ mượn trả & phòng tự học)
  + Thứ 7: 8:00 - 17:00
  + Chủ nhật & Ngày lễ: Nghỉ
- Hạn mức mượn: 
  + Tối đa 5 cuốn sách/sinh viên.
  + Thời hạn: 14 ngày (có thể gia hạn thêm 7 ngày qua ứng dụng di động hoặc tại Kiosk).
- Phí phạt & Đền bù:
  + Trả muộn: 2.000đ/ngày/cuốn.
  + Mất sách/Hư hỏng nặng: Đền bù 150% giá trị sách theo giá thị trường hiện tại.
  

Quy trình mượn sách tại Kiosk:
1. Sinh viên đứng trước camera để AI xác thực khuôn mặt (Face ID).
2. Đặt các cuốn sách cần mượn lên bàn quét AI (AI tự động nhận diện bìa & barcode).
3. Kiểm tra danh sách sách hiện ra trên màn hình.
4. Bấm "Xác nhận mượn".
5. Hệ thống gửi thông báo và biên lai điện tử về Email sinh viên (@fpt.edu.vn).

Quy trình trả sách:
1. Xác thực khuôn mặt tại Kiosk.
2. Vui lòng điền đúng thông tin và xác nhận thông tin cá nhân là chính xác.
3. Đặt sách cần trả vào khay nhận diện.
4. AI kiểm tra tình trạng sách và ghi nhận trả thành công.
5. Nếu có phí phạt quá hạn, hệ thống sẽ hiển thị mã QR để sinh viên thanh toán qua ví điện tử hoặc trừ vào tài khoản sinh viên.

Nội quy phòng đọc & Khu vực tự học:
- Giữ yên lặng tuyệt đối tại khu vực Silent Zone.
- Không mang đồ ăn có mùi hoặc nước uống không có nắp đậy vào phòng máy.
- Trang phục kín đáo, lịch sự (theo quy định đồng phục của Đại học FPT).
- Sau khi dùng sách tại chỗ, vui lòng đặt lại đúng vị trí trên kệ hoặc xe đẩy sách.

Thông tin liên hệ:
- Hotline: 0763537027
- Email: letanphap6543z@gmail.com
- Địa chỉ: Tầng 1, Tòa tháp Alpha, Đại học FPT AI Campus Quy Nhơn.

Quy tắc trả lời của bạn:
- Luôn sử dụng tiếng Việt thân thiện, chuyên nghiệp nhưng vẫn trẻ trung (phù hợp với sinh viên FPT).
- Nếu sinh viên hỏi về các vấn đề ngoài thư viện, hãy khéo léo dẫn dắt về lại chủ đề thư viện.
- Trả lời rõ ràng theo dạng danh sách (bullet points) cho các quy trình phức tạp.
- Khi không chắc chắn, hãy khuyên sinh viên gặp trực tiếp cán bộ thư viện tại quầy Information Desk.
"""


class RAGPipeline:
    def __init__(self):
        self.vector_store = None
        self.generator_service = None
        self._embeddings = None

        # Fast FAQ indexes (lexical + semantic)
        self._faq_lexical = self._build_faq_lexical()
        self._faq_vs: FAISS | None = None

        # Semantic index over books from DB (cached)
        self._books_vs: FAISS | None = None
        self._books_vs_built_at: datetime | None = None

        # Retrieval thresholds
        self.policy_k = 3
        self.max_books_to_index = 100 # Reduced from 2000 to save RAM
        self.cache_threshold = 0.15 # Chat history cache (very strict)
        self.faq_threshold = 0.447  # FAQ semantic cache (~0.90 similarity)

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = EmbeddingsService.get_embeddings()
        return self._embeddings

    @staticmethod
    def _build_faq_lexical():
        # ultra-fast keyword map to avoid LLM/retrieval latency for common kiosk FAQs
        return [
            (re.compile(r"(giờ|thời gian)\s*(mở|đóng)\s*cửa", re.I),
             "Giờ mở cửa SmartLib:\n- Thứ 2 - Thứ 6: 7:30 - 21:00\n- Thứ 7: 8:00 - 17:00\n- Chủ nhật & ngày lễ: Nghỉ"),
            (re.compile(r"(mượn|borrowing).*(tối đa|bao nhiêu)|hạn mức mượn", re.I),
             "Hạn mức mượn: tối đa 5 cuốn/sinh viên. Thời hạn mượn 14 ngày (có thể gia hạn thêm 7 ngày)."),
            (re.compile(r"(phạt|fine).*(trễ|muộn)|trả muộn", re.I),
             "Phí phạt trả muộn: 2.000đ/ngày/cuốn. Nếu mất/hỏng nặng: đền bù 150% giá trị sách."),
            (re.compile(r"quy trình.*mượn|mượn sách.*kiosk", re.I),
             "Quy trình mượn sách tại Kiosk:\n- Xác thực khuôn mặt\n- Đặt sách lên bàn quét AI\n- Kiểm tra danh sách sách\n- Bấm 'Xác nhận mượn'\n- Nhận biên lai điện tử qua email"),
            (re.compile(r"quy trình.*trả|trả sách.*kiosk", re.I),
             "Quy trình trả sách:\n- Xác thực khuôn mặt\n- Xác nhận thông tin\n- Đặt sách vào khay nhận diện\n- Hệ thống ghi nhận trả\n- Nếu có phí phạt: thanh toán QR hoặc trừ tài khoản"),
        ]

    async def _ensure_faq_semantic_index(self):
        if self._faq_vs is not None:
            return
        emb = self._get_embeddings()
        faqs = [
            ("Giờ mở cửa thư viện?", "Thứ 2-6: 7:30-21:00; Thứ 7: 8:00-17:00; CN/ngày lễ nghỉ."),
            ("Hạn mức mượn sách là bao nhiêu?", "Tối đa 5 cuốn/sinh viên. Thời hạn 14 ngày; có thể gia hạn 7 ngày."),
            ("Phí phạt trả muộn tính thế nào?", "2.000đ/ngày/cuốn. Mất/hỏng nặng: đền 150% giá trị sách."),
            ("Quy trình mượn sách ở kiosk?", "Xác thực FaceID → đặt sách lên bàn quét → kiểm tra → xác nhận mượn."),
            ("Quy trình trả sách ở kiosk?", "Xác thực FaceID → xác nhận → đặt sách vào khay → trả thành công → xử lý phí phạt nếu có."),
        ]
        docs = [
            Document(page_content=f"Q: {q}\nA: {a}", metadata={"type": "faq", "q": q})
            for q, a in faqs
        ]
        self._faq_vs = FAISS.from_documents(docs, emb)

    async def _ensure_books_semantic_index(self, *, ttl_minutes: int = 20, max_books: int = 2000):
        # rebuild periodically to reflect DB updates without long startup delays
        if self._books_vs is not None and self._books_vs_built_at is not None:
            if datetime.utcnow() - self._books_vs_built_at < timedelta(minutes=ttl_minutes):
                return

        emb = self._get_embeddings()
        from app.database import async_session_maker
        from sqlalchemy import select
        from app.models.book import Book

        async with async_session_maker() as session:
            # Only pull a reasonable number of recent/popular books to avoid OOM
            stmt = select(Book).order_by(Book.created_at.desc()).limit(self.max_books_to_index)
            result = await session.execute(stmt)
            books = result.scalars().all()

        docs: list[Document] = []
        for b in books:
            content = (
                f"Tiêu đề: {b.title}\n"
                f"Tác giả: {b.author or 'Khuyết danh'}\n"
                f"Chủ đề: {b.subject_category or 'N/A'}\n"
                f"Mã sách: {b.book_id}\n"
                f"Barcode: {b.barcode}\n"
                f"Mô tả: {b.description or ''}"
            )
            docs.append(Document(page_content=content, metadata={"type": "book", "book_id": b.book_id, "title": b.title}))

        self._books_vs = FAISS.from_documents(docs, emb) if docs else None
        self._books_vs_built_at = datetime.utcnow()

    def ingest_document(self, file_path: str, doc_type: str = "pdf"):
        logger.info(f"Starting ingestion pipeline for {file_path}")
        
        # 1. Document Loading
        if doc_type == "pdf":
            docs = DocumentLoaderService.load_pdf(file_path)
        elif doc_type == "csv":
            docs = DocumentLoaderService.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {doc_type}")
            
        # 2. Preprocessing
        clean_docs = Preprocessor.clean_documents(docs)
        
        # 3. Chunking
        chunker = ChunkerService()
        chunks = chunker.split_documents(clean_docs)
        
        # 4. Embeddings
        embeddings_model = EmbeddingsService.get_embeddings()
        
        # 5. Vector DB
        vs_service = VectorStoreService(embeddings_model)
        self.vector_store = vs_service.create_from_documents(chunks)
        
        # Reset generator so it picks up new retriever
        self.generator_service = None
        
        logger.info("Ingestion pipeline completed.")
        return len(chunks)

    async def ask_question(self, query: str, session_id: str = "default", student_id: str = None) -> str:
        """
        Answer a question with semantic caching and persistent history.
        """
        logger.info(f"RAG Pipeline: Processing query '{query[:50]}...'")
        emb_model = self._get_embeddings()
        query_vector = await emb_model.aembed_query(query)
        
        # Ensure query_vector is a standard Python list (crucial for pgvector/Supabase compatibility)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        elif not isinstance(query_vector, list):
            query_vector = list(query_vector)

        # 1. Semantic Cache Check (Reply Fast)
        # a) FAQ Cache (High similarity > 90% threshold)
        faq_cache_hit = await self._check_faq_cache(query, query_vector)
        if faq_cache_hit:
            logger.info(f"FAQ semantic cache hit for: {query[:50]}...")
            await self._save_interaction(session_id, student_id, "human", query, query_vector)
            await self._save_interaction(session_id, student_id, "ai", faq_cache_hit, None, {"type": "faq_cache"})
            return faq_cache_hit

        # b) Chat History Cache
        cached_answer = await self._check_semantic_cache(query_vector)
        if cached_answer:
            logger.info(f"Interaction cache hit for: {query[:50]}...")
            await self._save_interaction(session_id, student_id, "human", query, query_vector)
            await self._save_interaction(session_id, student_id, "ai", cached_answer, None, {"type": "history_cache"})
            return cached_answer

        # 2. Main Logic
        if self.vector_store:
            answer = await self._ask_with_rag(query)
        else:
            answer = await self._ask_general(query, query_vector)

        # 3. Save to History
        await self._save_interaction(session_id, student_id, "human", query, query_vector)
        await self._save_interaction(session_id, student_id, "ai", answer)

        return answer

    async def _check_faq_cache(self, query: str, query_vector: List[float]) -> str | None:
        """Checks if the query is nearly identical to a pre-defined FAQ."""
        await self._ensure_faq_semantic_index()
        if self._faq_vs is None:
            return None
            
        hits = self._faq_vs.similarity_search_with_score(query, k=1)
        if hits and hits[0][1] is not None and hits[0][1] < self.faq_threshold:
            # Extract only the Answer part
            content = hits[0][0].page_content
            if "A:" in content:
                return content.split("A:", 1)[-1].strip()
            return content
        return None

    async def _check_semantic_cache(self, query_vector: List[float]) -> str | None:
        """Checks if a very similar user question has been answered before."""
        async with async_session_maker() as session:
            # 1. Find the closest user question
            stmt = select(ChatHistory.session_id, ChatHistory.created_at, ChatHistory.embedding.l2_distance(query_vector).label("dist")) \
                   .where(ChatHistory.role == "human") \
                   .where(ChatHistory.embedding.isnot(None)) \
                   .order_by("dist").limit(1)
            
            result = await session.execute(stmt)
            best_match = result.first()
            
            if best_match and best_match.dist < self.cache_threshold:
                # 2. Find the assistant response that followed it in the same session
                ans_stmt = select(ChatHistory.content) \
                           .where(ChatHistory.session_id == best_match.session_id) \
                           .where(ChatHistory.role == "ai") \
                           .where(ChatHistory.created_at >= best_match.created_at) \
                           .order_by(ChatHistory.created_at.asc()).limit(1)
                ans_res = await session.execute(ans_stmt)
                answer = ans_res.scalar()
                if answer:
                    return f"[Cache Hit] {answer}"

        return None

    async def _save_interaction(self, session_id: str, student_id: Optional[str], role: str, content: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Saves a single message to the chat_history table.
        Robustly handles vector conversion and errors.
        """
        try:
            # Clean embedding (ensures float list and dimension check)
            safe_embedding = None
            if embedding is not None:
                if hasattr(embedding, "tolist"):
                    safe_embedding = embedding.tolist()
                elif isinstance(embedding, (list, tuple)):
                    safe_embedding = list(embedding)
                else:
                    safe_embedding = list(embedding)
                
                # Verify dimension (must be 768 for vietnamese-sbert)
                if len(safe_embedding) != 768:
                    logger.warning(f"Embedding dimension mismatch: expected 768, got {len(safe_embedding)}. Skipping vector save.")
                    safe_embedding = None


            async with async_session_maker() as session:
                new_msg = ChatHistory(
                    session_id=session_id,
                    student_id=student_id,
                    role=role,
                    content=content or "No content",
                    embedding=safe_embedding,
                    extra_metadata=metadata or {}
                )
                session.add(new_msg)
                await session.commit()
                logger.debug(f"Saved {role} message to history.")
        except Exception as e:
            logger.error(f"Critical error saving {role} message to chat_history: {e}", exc_info=True)
            # Re-raise to ensure the API response captures the issue
            raise e

    async def _get_policy_context(self, query_vector: Optional[List[float]]) -> List[Document]:
        """Retrieves relevant policy chunks as Documents for re-ranking."""
        async with async_session_maker() as session:
            stmt = select(PolicyChunk.chunk_text, PolicyChunk.embedding.l2_distance(query_vector).label("dist")) \
                   .order_by("dist") \
                   .limit(self.policy_k)
            result = await session.execute(stmt)
            rows = result.all()
            
            return [
                Document(page_content=row.chunk_text, metadata={"type": "policy", "score": row.dist})
                for row in rows
            ]

    async def _ask_general(self, query: str, query_vector: Optional[List[float]] = None) -> str:
        """Direct LLM response for general library questions."""
        logger.info(f"General mode: answering query: {query}")
        # 0) ultra-fast lexical FAQ shortcut
        for pattern, answer in self._faq_lexical:
            if pattern.search(query or ""):
                return answer

        try:
            # 1) Context Preparation & Combined Retrieval
            await self._ensure_faq_semantic_index()
            
            # Semantic search only if related to books/inventory
            is_book_query = re.search(r"(sách|book|tìm|tra cứu|tác giả|quyển|cuốn|thể loại)", query, re.I)
            if is_book_query:
                await self._ensure_books_semantic_index(max_books=self.max_books_to_index)
            
            if query_vector is None:
                query_vector = await self._get_embeddings().aembed_query(query)

            # Collect candidates from multiple sources
            candidates: List[Document] = []
            
            # FAQ candidates
            if self._faq_vs is not None:
                faq_hits = self._faq_vs.similarity_search_with_score(query, k=3)
                for doc, score in faq_hits:
                    doc.metadata["score"] = score
                    candidates.append(doc)

            # Books candidates
            if self._books_vs is not None:
                book_hits = self._books_vs.similarity_search_with_score(query, k=4)
                for doc, score in book_hits:
                    doc.metadata["score"] = score
                    candidates.append(doc)

            # Policy candidates
            policy_docs = await self._get_policy_context(query_vector)
            candidates.extend(policy_docs)

            # 2) Re-ranking: Simple sorting by vector distance scores
            # Filter: we only want reasonably relevant chunks (distance < 0.8)
            filtered = [c for c in candidates if c.metadata.get("score", 1.0) < 0.8]
            filtered.sort(key=lambda x: x.metadata.get("score", 1.0))
            
            # Limit: Take only the top 3 most relevant chunks to keep RAM and prompt clean
            top_chunks = filtered[:3]
            
            # Special case: results from system search (not embeddings)
            tool_ctx = ""
            try:
                from app.rag.tools import search_books, check_student_info
                if re.search(r"(sách|book|tìm|tra cứu)", query, re.I):
                    tool_ctx = await search_books.ainvoke({"query": query})
                m = re.search(r"\b([A-Za-z]{0,2}\d{6,12})\b", query or "")
                if m and re.search(r"(phạt|mượn|trả|thông tin|nợ)", query, re.I):
                    res = await check_student_info.ainvoke({"student_id": m.group(1)})
                    tool_ctx = (tool_ctx + "\n\n" if tool_ctx else "") + res
            except Exception as tool_err:
                logger.warning(f"Tool ctx build failed: {tool_err}")

            # 3) Final Context Assembly
            context_parts = []
            if tool_ctx:
                context_parts.append("Dữ liệu hệ thống Kiosk:\n" + tool_ctx)
            
            for dc in top_chunks:
                ctype = dc.metadata.get("type", "info")
                context_parts.append(f"[{ctype.upper()}] {dc.page_content}")

            context = "\n\n".join(context_parts).strip()

            # 4) Qwen3:4b via Ollama
            model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            user = (
                f"Câu hỏi: {query}\n\n"
                f"Ngữ cảnh (nếu có):\n{context}\n\n"
                "Hãy trả lời đúng trọng tâm, tiếng Việt. "
                "Nếu có gợi ý sách, hãy liệt kê kèm mã sách."
            )

            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": LIBRARY_SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            msg = data.get("message") or {}
            content = msg.get("content") or data.get("response")
            return content or "Xin lỗi, tôi chưa thể trả lời lúc này."
        except Exception as e:
            logger.error(f"General LLM error: {e}")
            return "Xin lỗi, tôi đang gặp sự cố kết nối. Vui lòng thử lại sau."

    async def _ask_with_rag(self, query: str) -> str:
        """Full RAG pipeline response when documents are available."""
        logger.info(f"RAG mode: answering query: {query}")
        try:
            # 6. Retrieval
            retriever = RetrieverService.create_retriever(self.vector_store, k=5)
            
            # 7. Generation
            if not self.generator_service:
                self.generator_service = GeneratorService(retriever)
                
            return await self.generator_service.generate_response(query)
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            # Fallback to general mode
            logger.info("Falling back to general mode...")
            return await self._ask_general(query)
