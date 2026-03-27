"""
SmartLib - Central Chat Orchestration Service
The main RAG chatbot pipeline:
  1. Load history from DB
  2. Embed query (bge-m3)
  3. Detect intent (qwen3.5:2b)
  4. Route to appropriate tool
  5. Build enriched prompt with retrieved context
  6. Generate response (qwen3.5:2b)
  7. Save Q&A to DB
"""
import uuid
from typing import Optional, List, Dict, Any
from cachetools import TTLCache
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, delete, desc, asc

from app.services.embedding_service import embedding_service
from app.services.intent_service import intent_service
from app.services.rag_service import rag_service
from app.services.tools.debt_tools import check_student_debt
from app.services.tools.policy_tools import query_policy
from app.services.tools.action_tools import (
    renew_book_tool, 
    reserve_book_tool, 
    get_personalized_recommendations
)
from app.models.chat_history import ChatHistory
from app.models.student import Student
from app.models.transaction import Transaction
from app.models.book import Book


# === System prompt templates ===

SYSTEM_PROMPT_BASE = """
# ROLE: Bạn là **SmartLib AI** - Trợ lý thư viện.
**TUYÊN BỐ DANH TÍNH (BẮT BUỘC)**: 
1. Bạn là thư ký AI, còn người chat với bạn là **Sinh viên**.
2. Khi khách hỏi "Tôi là ai?", hãy dựa vào mục # HỒ SƠ SINH VIÊN (tên, MSSV) để trả lời họ. KHÔNG ĐƯỢC trả lời "Bạn là SmartLib AI".
3. Chỉ xưng "Mình/SmartLib AI" cho bản thân và "Bạn/Sinh viên" cho người dùng.

# STYLE & TONE:
- Ngôn ngữ: Tiếng Việt, thân thiện nhưng lịch sự (Xưng hô: "Mình" - "Bạn/Sinh viên").
- **Formatting**: BẮT BUỘC dùng Markdown nâng cao để phản hồi chuyên nghiệp:
  - Dùng `**Bold**` cho các từ khóa quan trọng, số liệu, tên sách.
  - Dùng `> [!NOTE]` hoặc `> [!TIP]` cho các lưu ý quan trọng.
  - Dùng List (đánh số hoặc gạch đầu dòng) rõ ràng.
  - Sử dụng Emoji hợp lý để phân loại thông tin (📚 Sách, 📅 Hạn trả, 💰 Tiền phạt, 📌 Vị trí).
  - Sử dụng Table khi liệt kê từ 2 sách trở lên.

# GUIDELINES TUÂN THỦ TỐI THƯỢNG (PHẢI ĐỌC):
1. **CẤM TUYỆT ĐỐI DÙNG BẢNG (TABLE)**: Markdown Table sẽ làm hỏng hoàn toàn giao diện trên Kiosk (chữ bị chạy dọc, chồng chéo). Bạn sẽ bị phạt nặng nếu dùng bảng.
2. **HIỂN THỊ QUA CARD**: Hệ thống UI sẽ tự động render các Card sách đẹp mắt từ dữ liệu metadata. Nhiệm vụ của bạn chỉ là viết văn bản tư vấn: "Dưới đây là một số cuốn sách mình tìm thấy cho bạn:".
3. **NGUYÊN TẮC SỐ 1**: Nếu không có sách, báo không có. Không bịa đặt.
4. **NGUYÊN TẮC SỐ 2**: Trả lời ngắn gọn, thân thiện, dùng Emoji.

# VÍ DỤ PHẢN HỒI KHI KHÔNG CÓ SÁCH (MẪU CHUẨN):
Người dùng: "Tìm cho mình sách về lập trình Rust"
Hệ thống: [SYSTEM_STATUS: NO_BOOKS_FOUND]
Bạn: "Rất tiếc bạn ơi, hiện tại hệ thống thư viện **SmartLib** chưa có đầu sách nào về **lập trình Rust**. Bạn có muốn mình tìm kiếm các chủ đề lập trình khác (như Java, Python) hiện đang có sẵn không?"

# TRÌNH BÀY PHẢN HỒI:
1. **Khẳng định**: Trả lời thẳng vào vấn đề dựa trên dữ liệu.
2. **Nội dung**: Sử dụng Markdown Table cho danh sách sách.
3. **Trung thực**: Thà báo không có còn hơn bịa đặt. Nếu vi phạm, hệ thống sẽ bị lỗi nghiêm trọng.
"""


class ChatService:
    """
    Central orchestrator for the SmartLib RAG chatbot.
    
    Pipeline per message:
    1. Load chat history from DB (last N turns)
    2. Embed user query with bge-m3
    3. Classify intent with qwen3.5:2b
    4. Execute appropriate tool (book/stock/debt/policy)
    5. Build enriched system prompt with retrieved context
    6. Generate final response with qwen3.5:2b
    7. Persist Q&A to chat_history table
    """

    HISTORY_WINDOW = 6  # Number of past messages to include
    STUDENT_CACHE_TTL = 300  # 5 minutes in seconds

    def __init__(self):
        self._llm = None
        # Student context cache: {student_id: {"data": {...}, "expires_at": float}}
        self._student_cache: Dict[str, Any] = {}
        self._semantic_cache = TTLCache(maxsize=512, ttl=600)  # 10 minutes

    def _get_llm(self):
        """Lazy load LLM service."""
        if self._llm is None:
            from app.services.llm_service import ai_assistant
            self._llm = ai_assistant
        return self._llm

    async def process_message(
        self,
        db: AsyncSession,
        message: str,
        session_id: str,
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline for a single user message.
        
        Args:
            db: Async database session
            message: User's message text
            session_id: Unique session identifier
            student_id: Authenticated student ID (optional)
            
        Returns:
            Dict with reply, intent, sources, and metadata
        """
        logger.info(f"[Chat] session={session_id}, student={student_id}, msg='{message[:60]}...'")

        # === Step 1: Load conversation history (Student-first persistence) ===
        history = await self._load_history(db, session_id, student_id)        # === Step 0: Lexical FAQ Check (Fast Path) ===
        import re
        lexical_faqs = [
            (r"(giờ|thời gian)\s*(mở|đóng)\s*cửa", 
             "🕒 **Giờ mở cửa SmartLib:**\n- Thứ 2 - Thứ 6: **7:30 - 21:00**\n- Thứ 7: **8:00 - 17:00**\n- Chủ nhật & Ngày lễ: **Nghỉ**\n\n*Bạn cần biết thêm thông tin gì khác không?*"),
            (r"(mượn|borrowing).*(tối đa|bao nhiêu)|hạn mức mượn", 
             "📚 **Hạn mức mượn sách:**\n- Tối đa **5 quyển sách** cùng một lúc.\n- Thời hạn mượn: **14 ngày**.\n- Gia hạn: Tối đa **2 lần**, mỗi lần **7 ngày**."),
            (r"(phạt|fine).*(trễ|muộn)|trả muộn", 
             "💰 **Phí phạt trả sách quá hạn:**\n- **10.000 VNĐ/ngày/quyển**.\n- Mất sách: Đền bù **100% giá trị sách** + phí xử lý."),
        ]
        for pattern, faq_reply in lexical_faqs:
            if re.search(pattern, message, re.IGNORECASE):
                logger.info(f"[Chat] Lexical FAQ hit for: {message}")
                return await self._finalize_response(db, session_id, faq_reply, "policy_query", {}, [], student_id, message, metadata={"faq_lexical": True})

        # === Step 2: Load Student Context (Personalization) ===
        student_context = await self._load_student_context(db, student_id)
        student_name = student_context.get("name", "bạn")

        # === Step 2: Embed query ===
        query_embedding = await embedding_service.embed(message)
        if not query_embedding:
            logger.warning("[Chat] Embedding failed, falling back to empty vector")
            query_embedding = [0.0] * 1024 # Placeholder (1024d standardized)

        # === Step 3 (Fast path): Semantic cache from chat_history ===
        # If user repeats (or nearly repeats) a previous question, return the old answer quickly.
        cached = await self._semantic_cache_lookup(db, session_id, student_id, message, query_embedding)
        if cached:
            logger.info("[Chat] Hit semantic cache, returning cached answer.")
            return await self._finalize_response(
                db,
                session_id,
                cached["reply"],
                cached.get("intent", "cached"),
                cached.get("entities", {}),
                cached.get("sources", []),
                student_id,
                message,
                metadata={**(cached.get("metadata") or {}), "cache_hit": True},
            )

        # === Step 4: Detect intent ===
        intent_result = await intent_service.classify(message, history)
        intent = intent_result.get("intent", "general_chat")
        entities = intent_result.get("entities", {})
        logger.info(f"[Chat] Intent: {intent} | Entities: {entities}")

        # === Step 5: Tool Execution ===
        tool_context = ""
        suggestions = []
        sources = []
        metadata = {
            "intent": intent, 
            "entities": entities,
            "student_id": student_id,
            "student_name": student_name
        }

        # --- FALLBACK RETRIEVAL ---
        # Rely on intent_service (LLM) classification to handle all cases
        if intent == "book_search":
            # 1. SEARCH FOR BOOKS with AI-extracted entities
            result = await rag_service.search_books_hybrid(
                db=db,
                query_text=message,
                query_embedding=query_embedding,
                top_k=5,
                entities=entities,
                intent=intent
            )
            tool_context = rag_service.format_books_for_context(result)
            suggestions = result[:3]
            sources = [{"type": "book", "data": b} for b in result]
        
        elif intent == "stock_check":
            # For stock check, we still use hybrid search with AI entities to find the right book
            result = await rag_service.search_books_hybrid(
                db=db,
                query_text=message,
                query_embedding=query_embedding,
                top_k=5,
                entities=entities,
                intent=intent
            )
            tool_context = rag_service.format_books_for_context(result)
            suggestions = result[:3]
            sources = [{"type": "book_stock", "data": b} for b in result]

        elif intent == "debt_check":
            if not student_id:
                tool_context = "⚠️ Vui lòng đăng nhập (nhận diện khuôn mặt) để mình tra cứu thông tin nợ phạt và sách đang mượn chính xác của bạn nhé."
            else:
                result = await check_student_debt(db, message, entities=entities, session_student_id=student_id)
                tool_context = result["context"]

        elif intent == "policy_query":
            # Optimization: FAQ/Policy short-circuit (Skip LLM for fast FAQ)
            result = await query_policy(db, message, top_k=2) # Reduced chunks to 2
            tool_context = result["context"]
            # For policy, we do NOT want to show book cards, so sources remains empty
            sources = [] 
            
            # If high confidence (similarity > 0.50) or specifically marked as found, 
            # we can skip the expensive LLM generation and reply instantly.
            top_similarity = result.get("chunks")[0].get("similarity", 0) if result.get("chunks") else 0
            if result.get("found") and top_similarity > 0.50:
                logger.info(f"[Chat] FAQ Short-circuit (sim={top_similarity}) - Skipping LLM")
                reply = f"📌 **Thông tin quy định bạn cần tìm đây:**\n\n{tool_context}\n\n*Bạn còn thắc mắc gì khác về quy định không?*"
                # For policy queries, sources must be EMPTY to avoid "Không rõ" cards in UI
                return await self._finalize_response(db, session_id, reply, intent, entities, [], student_id, message, metadata={**metadata, "short_circuit": True})

        elif intent == "renew_book":
            if not student_id:
                tool_context = "⚠️ Bạn cần đăng nhập (nhận diện khuôn mặt) để mình có thể gia hạn sách giúp bạn nhé."
            else:
                result = await renew_book_tool(db, student_id, entities=entities)
                tool_context = result["context"]

        elif intent == "reserve_book":
            if not student_id:
                tool_context = "⚠️ Bạn cần đăng nhập để mình hỗ trợ đặt trước sách nhé."
            else:
                result = await reserve_book_tool(db, student_id, entities=entities)
                tool_context = result["context"]

        elif intent == "return_book":
            # For return book, we guide them to the kiosk hardware/procedure
            tool_context = (
                "📚 **Thủ tục trả sách TỰ ĐỘNG tại SmartLib Kiosk**:\n"
                "1. Bạn không cần gặp nhân viên. Hãy chọn nút **'Trả sách'** trên màn hình chính của Kiosk này.\n"
                "2. Đưa mã vạch (barcode) ở bìa sau của sách vào vùng quét laser.\n"
                "3. Hệ thống sẽ nhận diện sách và tự động hoàn tất giao dịch trong **3 giây**.\n"
                "4. Nếu có nợ phạt, hệ thống sẽ hiển thị số tiền và bạn có thể quét mã QR để thanh toán ngay.\n"
                "5. Sau khi quét xong, hãy đặt sách vào hộc **'Trả sách tự động'** bên dưới."
            )

        # === Step 5.1: Personalized Recommendations (Optional Add-on) ===
        recommendation_context = ""
        if student_id and intent in ["general_chat", "book_search"]:
            # Only trigger if history is small (early session) or specifically general chat
            rec_result = await get_personalized_recommendations(db, student_id)
            if rec_result.get("has_recommendations"):
                recommendation_context = rec_result["context"]

        # === Step 6: Build enriched prompt ===
        # Always use student personal context if available (borrowing status, etc.)
        full_student_info = student_context.get("context", "")
        
        system_prompt = self._build_system_prompt(
            intent, 
            tool_context, 
            full_student_info,
            recommendation_context
        )
        llm_messages = [{"role": "system", "content": system_prompt}]
        for h in history[-self.HISTORY_WINDOW:]:
            llm_messages.append(h)
        llm_messages.append({"role": "user", "content": message})

        # === Step 7: Generate response ===
        llm = self._get_llm()
        
        # Inject an even stricter reminder for book search if empty
        if intent == "book_search" and "[SYSTEM_STATUS: NO_BOOKS_FOUND]" in tool_context:
            llm_messages.append({"role": "system", "content": "NHẮC NHỞ: Kết quả tìm kiếm ĐANG TRỐNG. Bạn KHÔNG ĐƯỢC PHÉP gợi ý bất kỳ cuốn sách cụ thể nào. Chỉ báo là không có."})

        response = await llm.chat(
            llm_messages, 
            temperature=0.2, # Lower temperature to prevent hallucination
            num_predict=500
        )
        reply = response.get("message", {}).get("content", "")

        return await self._finalize_response(db, session_id, reply, intent, entities, sources, student_id, message, metadata)

    async def _finalize_response(
        self, db, session_id, reply, intent, entities, sources, student_id, message_text, metadata=None
    ) -> Dict[str, Any]:
        """Final helper to save history and return results."""
        # LLM output is already normalized by llm_service; just handle empty reply.
        if not reply:
            reply = "Xin lỗi, mình đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

        # Step 7: Persist history
        await self._save_message(db, session_id, "human", message_text, student_id)
        await self._save_message(db, session_id, "ai", reply, student_id, metadata)

        # Step 8: Prepare CLEAN suggestions for Frontend (Unwrap and Group)
        cleaned_suggestions = []
        seen_keys = {}
        
        for s in sources:
            # Sources can be [{'type': 'book', 'data': {...}}, ...] or simple dicts
            data = s.get("data") if isinstance(s, dict) and "data" in s else s
            if not isinstance(data, dict): continue
            
            title = data.get("title", "Không rõ").strip()
            author = data.get("author", "Đang cập nhật").strip()
            key = (title.lower(), author.lower())
            
            if key not in seen_keys:
                seen_keys[key] = True
                cleaned_suggestions.append({
                    "book_id": data.get("book_id"),
                    "title": title,
                    "author": author,
                    "status": data.get("status", "AVAILABLE")
                })
        
        return {
            "reply": reply,
            "intent": intent,
            "entities": entities,
            "sources": cleaned_suggestions[:5], # This will be mapped to 'suggestions' in API route
            "session_id": session_id,
            "success": True,
        }

    def _build_system_prompt(self, intent: str, tool_context: str, student_info: str = "", recommendations: str = "") -> str:
        """Construct intent-specific system prompt with dual-flow logic (Public vs Personalized)."""
        prompt = SYSTEM_PROMPT_BASE
        
        # --- Mode 1: PERSONALIZED (Logged in) ---
        if student_info:
            prompt += (
                "\n\n# 👤 TRẠNG THÁI: ĐÃ ĐĂNG NHẬP (LUỒNG CÁ NHÂN HÓA)\n"
                "- Bạn đang phục vụ riêng cho sinh viên này.\n"
                "- Ưu tiên trả lời dựa trên thông tin cá nhân (nợ, sách đang mượn).\n"
                f"**HỒ SƠ SINH VIÊN**:\n{student_info}"
            )
            if recommendations:
                prompt += f"\n\n> [!TIP]\n> **Gợi ý dành riêng cho bạn**:\n{recommendations}"
        
        # --- Mode 2: PUBLIC (Not logged in) ---
        else:
            prompt += (
                "\n\n# 🌐 TRẠNG THÁI: KHÁCH (LUỒNG CÔNG CỘNG)\n"
                "- Tuyệt đối KHÔNG trả lời thông tin cá nhân.\n"
                "- Chỉ hỗ trợ: tìm sách, kiểm tra kho, hỏi quy trình.\n"
                "- Luôn kết thúc bằng việc nhắc nhở đăng nhập (Face Auth) để dùng full tính năng.\n"
            )

        # --- Intent Routing ---
        if intent == "book_search":
            prompt += (
                "\n\n**CHỈ THỊ ĐẶC BIỆT**: Tuyệt đối KHÔNG sử dụng Markdown Table (bảng).\n"
                "Danh sách sách sẽ được hiển thị bằng các THẺ UI riêng biệt.\n"
                "Hãy viết câu trả lời thân thiện, gợi ý người dùng xem danh sách sách ở các thẻ Card bên dưới.\n\n"
                f"--- KẾT QUẢ TÌM KIẾM ---\n{tool_context}"
            )
        elif intent == "stock_check":
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Kiểm tra vị trí và tình trạng sách.\n"
                "Hãy chỉ rõ **Kệ (Shelf/Loc)** và trạng thái mượn.\n\n"
                f"--- THÔNG TIN SÁCH ---\n{tool_context}"
            )
        elif intent == "debt_check":
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Tra cứu nợ nần và tiền phạt.\n"
                "Trình bày theo dạng **Bảng kê khai** nếu sinh viên đang mượn nhiều sách.\n"
                "Sử dụng Blockquote để nhắc họ về quy định đóng phạt nếu có nợ.\n\n"
                f"--- DỮ LIỆU NỢ ---\n{tool_context}"
            )
        elif intent == "policy_query":
            prompt += (
                "\n\n**NHIỆM VỤ QUAN TRỌNG (CẤM SAI LỆCH)**: Bạn đang trả lời về QUY ĐỊNH thư viện.\n"
                "- **TUYỆT ĐỐI KHÔNG** được bịa ra con số (ví dụ: tối đa 5 cuốn thì không được ghi là 10).\n"
                "- Phải trích dẫn chính xác dữ liệu từ văn bản bên dưới.\n"
                "- Nếu thông tin không có trong văn bản, hãy nói 'Xin lỗi, mình không tìm thấy quy định này trong hệ thống'.\n"
                "- Không dùng kiến thức bên ngoài.\n\n"
                f"--- VĂN BẢN QUY ĐỊNH CHUẨN ---\n{tool_context}"
            )
        elif intent == "renew_book":
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Thực hiện gia hạn sách.\n"
                "Thông báo kết quả gia hạn cho sinh viên (Thành công/Thất bại).\n"
                "Nếu thành công, nhắc họ hạn trả mới. Nếu thất bại, giải thích lý do (quá hạn, đã gia hạn rồi).\n\n"
                f"--- KẾT QUẢ GIA HẠN ---\n{tool_context}"
            )
        elif intent == "reserve_book":
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Thực hiện đặt trước sách.\n"
                "Thông báo kết quả đặt trước và vị trí hàng chờ nếu thành công.\n"
                "Nếu sách đang có sẵn, bảo họ không cần đặt trước mà mượn luôn.\n\n"
                f"--- KẾT QUẢ ĐẶT TRƯỚC ---\n{tool_context}"
            )
        elif intent == "return_book":
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Hướng dẫn trả sách TỰ ĐỘNG.\n"
                "Tuyệt đối KHÔNG bảo sinh viên gặp nhân viên trừ khi có sự cố kỹ thuật.\n"
                "Hãy nhấn mạnh vào sự tiện lợi và tốc độ của Kiosk tự phục vụ.\n\n"
                f"--- QUY TRÌNH TRẢ SÁCH ---\n{tool_context}"
            )
        else:
            prompt += (
                "\n\n**Nhiệm vụ hiện tại**: Trò chuyện thân thiện với sinh viên.\n"
                "Bạn có thể giới thiệu các chức năng: tìm sách, kiểm tra sách, xem nợ phạt, hỏi quy định.\n"
                "\n\n**Nhiệm vụ hiện tại**: Trò chuyện thân thiện và hỗ trợ danh tính.\n"
                "- Nếu họ hỏi 'Tôi là ai' mà chưa đăng nhập: Hãy báo họ là **Khách** và mời dùng Face Auth.\n"
                "- Nếu họ hỏi 'Tôi là ai' đã đăng nhập: Trích xuất tên từ # HỒ SƠ SINH VIÊN để trả lời.\n"
                "- Nếu họ hỏi 'Bạn là ai': Khẳng định mình là **SmartLib AI**.\n"
            )

        return prompt

    async def _load_history(
        self, db: AsyncSession, session_id: str, student_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Load recent chat history.
        Prioritizes student_id for global persistence (User Memory).
        Falls back to session_id for anonymous users.
        """
        try:
            # Persistent memory strategy: prioritize student over session
            filter_clause = ChatHistory.student_id == student_id if student_id else ChatHistory.session_id == session_id
            
            stmt = (
                select(ChatHistory.role, ChatHistory.content)
                .where(filter_clause)
                .order_by(desc(ChatHistory.created_at))
                .limit(self.HISTORY_WINDOW)
            )
            result = await db.execute(stmt)
            rows = result.all()

            # Reverse to chronological order
            messages = []
            for role, content in reversed(rows):
                messages.append({
                    "role": "assistant" if role == "ai" else "user",
                    "content": content
                })

            logger.debug(f"Loaded {len(messages)} persistent history messages (student={student_id}, session={session_id})")
            return messages
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            await db.rollback()
            return []

    async def _save_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,  # 'human' or 'ai'
        content: str,
        student_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Persist a single message to chat_history table."""
        try:
            chat_entry = ChatHistory(
                session_id=session_id,
                role=role,
                content=content,
                student_id=student_id,
                extra_metadata=metadata or {}
            )
            db.add(chat_entry)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to save message to history: {e}")
            await db.rollback()

    async def _semantic_cache_lookup(
        self,
        db: AsyncSession,
        session_id: str,
        student_id: Optional[str],
        message: str,
        query_embedding: List[float],
        *,
        similarity_threshold: float = 0.92,
        window_hours: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """
        Semantic cache: find a highly similar previous user message and reuse its paired AI answer.
        Uses pgvector cosine distance on chat_history.embedding (768d).
        """
        try:
            # in-memory shortcut for exact repeats in same session
            key = (student_id or "anon", session_id, message.strip().lower())
            if key in self._semantic_cache:
                return self._semantic_cache[key]

            # Ensure dim = 1024 for DB vector
            if len(query_embedding) != 1024:
                query_embedding = (query_embedding[:1024] if len(query_embedding) > 1024 else query_embedding + [0.0] * (1024 - len(query_embedding)))

            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

            # Restrict search scope: student-first, otherwise session; recent timeframe only.
            filter_sql = "student_id = :student_id" if student_id else "session_id = :session_id"
            params = {"student_id": student_id, "session_id": session_id, "query_vec": vec_str}

            sql = text(f"""
                WITH candidates AS (
                    SELECT id, created_at,
                           1 - (embedding <=> cast(:query_vec as vector)) AS similarity
                    FROM chat_history
                    WHERE role = 'human'
                      AND embedding IS NOT NULL
                      AND {filter_sql}
                      AND created_at >= (now() - interval '{window_hours} hours')
                    ORDER BY embedding <=> cast(:query_vec as vector)
                    LIMIT 1
                )
                SELECT c.id as human_id, c.similarity,
                       a.content as ai_content, a.extra_metadata
                FROM candidates c
                JOIN chat_history a
                  ON a.session_id = :session_id
                 AND a.role = 'ai'
                 AND a.created_at >= c.created_at
                ORDER BY a.created_at ASC
                LIMIT 1
            """)

            result = await db.execute(sql, params)
            row = result.mappings().first()
            if not row:
                return None

            sim = float(row["similarity"] or 0.0)
            if sim < similarity_threshold:
                return None

            payload = {
                "reply": row["ai_content"],
                "metadata": row.get("extra_metadata") or {},
                "intent": "cached",
                "entities": {},
                "sources": [],
            }
            self._semantic_cache[key] = payload
            return payload
        except Exception as e:
            logger.warning(f"[Chat] semantic cache lookup failed: {e}")
            return None

    async def get_history(
        self, db: AsyncSession, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve full chat history for a session."""
        try:
            stmt = (
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .order_by(asc(ChatHistory.created_at))
                .limit(limit)
            )
            result = await db.execute(stmt)
            rows = result.scalars().all()
            
            return [
                {
                    "id": str(r.id),
                    "session_id": r.session_id,
                    "role": r.role,
                    "content": r.content,
                    "student_id": r.student_id,
                    "metadata": r.extra_metadata,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            await db.rollback()
            return []

    async def clear_history(self, db: AsyncSession, session_id: str) -> bool:
        """Clear all messages for a session."""
        try:
            stmt = delete(ChatHistory).where(ChatHistory.session_id == session_id)
            await db.execute(stmt)
            await db.commit()
            logger.info(f"Cleared history for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            await db.rollback()
            return False

    async def _load_student_context(self, db: AsyncSession, student_id: Optional[str]) -> Dict[str, Any]:
        """Fetch student details and borrowing status for personalization.
        
        Results are cached in-memory for STUDENT_CACHE_TTL seconds to avoid
        hitting the DB on every message in a conversation.
        """
        if not student_id:
            return {"name": "bạn", "context": ""}

        import time

        # --- Cache hit check ---
        cached = self._student_cache.get(student_id)
        if cached and cached["expires_at"] > time.monotonic():
            logger.debug(f"[Cache] Student context HIT for {student_id}")
            return cached["data"]

        try:
            # Fetch student info
            from sqlalchemy import select
            result = await db.execute(select(Student).where(Student.student_id == student_id))
            student = result.scalar_one_or_none()
            if not student:
                return {"name": "bạn", "context": ""}
            
            # Fetch active transactions
            t_result = await db.execute(
                select(Transaction, Book.title)
                .join(Book, Transaction.book_id == Book.book_id)
                .where(Transaction.student_id == student_id)
                .where(Transaction.status.in_(["ACTIVE", "OVERDUE"]))
            )
            borrows = t_result.all()
            
            context_parts = [
                f"Tên: {student.full_name}",
                f"Mã SV: {student.student_id}",
                f"Số dư nợ phạt: {student.fine_balance:,.0f} VND"
            ]
            
            if borrows:
                context_parts.append("Sách đang mượn:")
                for t, title in borrows:
                    status_str = "QUÁ HẠN" if t.status == "OVERDUE" else "Đang mượn"
                    context_parts.append(f"- {title} (Hạn: {t.due_date}) [{status_str}]")
            else:
                context_parts.append("Hiện tại không mượn sách nào.")

            result_data = {
                "name": student.full_name,
                "context": "\n".join(context_parts)
            }

            # --- Store in cache ---
            self._student_cache[student_id] = {
                "data": result_data,
                "expires_at": time.monotonic() + self.STUDENT_CACHE_TTL,
            }
            logger.debug(f"[Cache] Student context MISS → cached for {student_id} ({self.STUDENT_CACHE_TTL}s)")

            return result_data
        except Exception as e:
            logger.error(f"Failed to load student context: {e}")
            return {"name": "bạn", "context": ""}


# Singleton instance
chat_service = ChatService()
