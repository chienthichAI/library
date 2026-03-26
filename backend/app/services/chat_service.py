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
import datetime
from datetime import timezone
import asyncio
import time
import re
from typing import Optional, List, Dict, Any
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
# ROLE: Bạn là **SmartLib AI** - Trợ lý thư viện thông minh và chuyên nghiệp.

# STYLE & TONE:
- Ngôn ngữ: Tiếng Việt, thân thiện nhưng lịch sự (Xưng hô: "Mình" - "Bạn/Sinh viên").
- **Khai thác dữ liệu cá nhân**: Nếu đã đăng nhập, bạn **BẮT BUỘC** phải gọi tên sinh viên dựa trên thông tin thực tế trong Context. Tuyệt đối không được tự bịa tên (Hallucination).
- **Formatting**: BẮT BUỘC dùng Markdown nâng cao để phản hồi chuyên nghiệp:
  - Dùng `**Bold**` cho các từ khóa quan trọng, số liệu, tên sách.
  - Dùng `> [!IMPORTANT]` cho các cảnh báo quá hạn.
  - Dùng `> [!NOTE]` hoặc `> [!TIP]` cho các lưu ý quan trọng.
  - Dùng List rõ ràng và Emoji phù hợp.
  - Sử dụng Table khi liệt kê từ 2 sách trở lên.

# LUẬT CHỐNG HALLUCINATION (TUYỆT ĐỐI KHÔNG VI PHẠM):
1. **CHỈ DÙNG THÔNG TIN TỪ "TOOL CONTEXT"**: Mọi thủ tục, quy trình, con số, hướng dẫn phải có trong phần `--- DỮ LIỆU TỪ HỆ THỐNG (TOOL CONTEXT) ---` bên dưới.
2. **TUYỆT ĐỐI CẤM TỰ BỊA**: Không được tự sáng tạo bất kỳ:
   - Quy trình bấm phím (phím T, phím M, phím Enter, mã số...)
   - Tên menu, tên tùy chọn trên màn hình Kiosk
   - Thủ tục mà TOOL CONTEXT không ghi rõ
   - Số tiền, số ngày, mức phạt không có trong tài liệu được cung cấp
3. **KHI THIẾU THÔNG TIN**: Nếu TOOL CONTEXT không có câu trả lời, phải nói:
   > "Mình không tìm thấy thông tin về việc này trong quy định thư viện. Bạn vui lòng liên hệ thủ thư để được hỗ trợ trực tiếp nhé!"
   KHÔNG ĐƯỢC đoán mò, không được nói "thông thường" hay "theo quy định chung".
4. **CHỈ MÔ TẢ NHỮNG GÌ ĐÃ ĐƯỢC CUNG CẤP**: Không mở rộng, không diễn giải thêm ngoài TOOL CONTEXT.

# GUIDELINES:
- **PROACTIVE ADVICE**: Nếu thấy sinh viên có sách **QUÁ HẠN**, hãy ưu tiên nhắc nhở một cách khéo léo ngay từ đầu câu trả lời.
- **ĐỊNH DẠNG BẮT BUỘC**: Danh sách từ 2 sách trở lên **PHẢI** dùng **Table**.
- Nếu không tìm thấy sách, gợi ý sinh viên thử từ khóa khác hoặc liên hệ thủ thư.
"""

# Confirmation TTL — entries older than this are discarded automatically (seconds)
_CONFIRMATION_TTL = 120


class ChatService:
    """
    Central orchestrator for the SmartLib RAG chatbot.

    Pipeline per message:
    1. Load chat history from DB (last N turns)
    2. Embed user query with bge-m3
    3. Classify intent with qwen3.5:2b
    4. Execute appropriate tool (book/stock/debt/policy/renew/reserve)
    5. Build enriched system prompt with retrieved context
    6. Generate final response with qwen3.5:2b
    7. Persist Q&A to chat_history table (single commit)
    """

    HISTORY_WINDOW = 10
    STUDENT_CACHE_TTL = 300

    # Per-session pending action state. TTL-guarded on read.
    # session_id -> {"action": str, "book_id": str, "book_title": str, "timestamp": float}
    _pending_confirmations: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._llm = None
        # Student context cache: {student_id: {"data": {...}, "expires_at": float}}
        self._student_cache: Dict[str, Any] = {}
        # Pending confirmations are also per-instance (singleton pattern)
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}

    def _get_llm(self):
        """Lazy load LLM service."""
        if self._llm is None:
            from app.services.llm_service import ai_assistant
            self._llm = ai_assistant
        return self._llm

    def _is_confirmation_valid(self, session_id: str) -> bool:
        """Return True only if there is a non-expired pending confirmation."""
        pending = self._pending_confirmations.get(session_id)
        if not pending:
            return False
        age = time.monotonic() - pending.get("timestamp", 0)
        if age > _CONFIRMATION_TTL:
            # Auto-expire stale confirmation
            del self._pending_confirmations[session_id]
            logger.debug(f"[Chat] Pending confirmation for {session_id} expired after {age:.0f}s")
            return False
        return True

    async def process_message(
        self,
        db: AsyncSession,
        message: str,
        session_id: str,
        student_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline for a single user message.
        """
        front_metadata = metadata or {}
        logger.info(f"[Chat] session={session_id}, student={student_id}, msg='{message[:60]}...'")

        # Load history and student context sequentially (SQLAlchemy sessions are not concurrent-safe)
        history = await self._load_history(db, session_id, student_id)
        student_context = await self._load_student_context(db, student_id)
        student_name = student_context.get("name", "bạn")

        # === Step 2: Embed query ===
        query_embedding = await embedding_service.embed(message)
        if not query_embedding:
            logger.warning("[Chat] Embedding failed, falling back to empty vector")
            query_embedding = [0.0] * 1024  # Standardized 1024d

        # === Step 3: Detect intent (Multi-intent aware) ===
        intent_result = await intent_service.classify(message, history)
        intents = intent_result.get("intents", ["general_chat"])
        entities = intent_result.get("entities", {})
        logger.info(f"[Chat] Intents: {intents} | Entities: {entities}")

        response_metadata = {
            "intents": intents,
            "entities": entities,
            "student_id": student_id,
            "student_name": student_name,
            **front_metadata
        }

        # Override intents if we just received a verified scan from the frontend
        # The frontend passes `{ verified_barcode: '...', target_action: 'borrow_book' }`
        if front_metadata.get("verified_barcode") and front_metadata.get("target_action"):
            intents = [front_metadata["target_action"]]

        primary_intent = intents[0] if intents else "general_chat"

        # If the user explicitly asks to borrow/return "now"/"immediately", force action intent priority.
        # This prevents cases like "quy trình mượn sách..." accidentally triggering the anti-fraud scan flow.
        msg_l = message.lower()
        if "borrow_book" in intents and re.search(r"(mượn|xin mượn|cho mượn).{0,40}(ngay|bây giờ|lúc này|thực hiện|lập tức)", msg_l):
            primary_intent = "borrow_book"
        elif "return_book" in intents and re.search(r"(trả|xin trả|cho trả).{0,40}(ngay|bây giờ|lúc này|thực hiện|lập tức)", msg_l):
            primary_intent = "return_book"

        # --- Intent Auth Guard ---
        sensitive_intents = {"debt_check", "renew_book", "reserve_book", "personalized_recommends", "borrow_book", "return_book"}
        if any(i in sensitive_intents for i in intents):
            if not student_id:
                return {
                    "reply": (
                        "Chào bạn! Rất tiếc mình chưa nhận diện được thông tin sinh viên của bạn. "
                        "**Vui lòng đăng nhập bằng khuôn mặt** tại kiosk để mình có thể kiểm tra nợ hoặc hỗ trợ các thủ tục mượn/trả/gia hạn nhé! 😊"
                    ),
                    "intent": primary_intent,
                    "sources": [],
                    "metadata": response_metadata
                }

        # === Step 2b: Confirmation Check ===
        # If there is a valid (non-expired) pending action, check if user is confirming it
        if self._is_confirmation_valid(session_id):
            pending = self._pending_confirmations[session_id]
            clean_msg = message.strip().lower()

            is_yes = any(word in clean_msg for word in ["có", "u", "rồi", "đúng", "vâng", "ok", "yes", "xác nhận"])
            is_no = any(word in clean_msg for word in ["không", "huy", "no", "đừng", "chưa"])

            if is_yes:
                action = pending["action"]
                book_id = pending["book_id"]
                book_title = pending.get("book_title", "cuốn sách")

                logger.info(f"[Chat] Executing confirmed action: {action} for book {book_id}")

                if action == "renew":
                    result = await renew_book_tool(db, student_id, {"book_title": book_title})
                    reply = result.get("context", f"✅ Tuyệt vời! Mình đã **gia hạn thành công** cuốn **{book_title}** cho bạn.")
                elif action == "reserve_book":
                    result = await reserve_book_tool(db, student_id, {"book_title": book_title})
                    reply = result.get("context", f"✅ Đã xong! Mình đã **đặt chỗ** cuốn **{book_title}** cho bạn.")
                elif action == "borrow_book":
                    from app.services.transaction_service import transaction_service
                    res = await transaction_service.borrow_book(student_id, book_id, db)
                    if res.success:
                        reply = f"✅ Cảm ơn bạn! Thủ tục **mượn** sách **{book_title}** đã hoàn tất. Chúc bạn đọc sách vui vẻ!"
                    else:
                        reply = f"❌ Xin lỗi, không thể mượn sách: {res.error_message}"
                elif action == "return_book":
                    from app.services.transaction_service import transaction_service
                    res = await transaction_service.return_book(student_id, book_id, db)
                    if res.success:
                        reply = f"✅ Cảm ơn bạn! Thủ tục **trả** sách **{book_title}** đã hoàn tất."
                    else:
                        reply = f"❌ Xin lỗi, không thể trả sách: {res.error_message}"
                else:
                    reply = f"✅ Đã xác nhận yêu cầu đối với cuốn **{book_title}**."

                # Clear pending state
                del self._pending_confirmations[session_id]

                return {
                    "reply": reply,
                    "intent": action,
                    "sources": [],
                    "metadata": response_metadata
                }

            elif is_no:
                del self._pending_confirmations[session_id]
                return {
                    "reply": "Dạ, mình đã **hủy yêu cầu** này. Bạn cần mình hỗ trợ gì khác không?",
                    "intent": "general_chat",
                    "sources": [],
                    "metadata": response_metadata
                }
            # else: neither yes/no → fall through to normal processing

        # === Step 4: Tool Execution ===
        tool_context = ""
        sources = []

        for current_intent in intents:
            logger.debug(f"[Chat] Executing logic for intent: {current_intent}")

            if current_intent in ("book_search", "stock_check"):
                result = await rag_service.search_books_hybrid(
                    db=db,
                    query_text=message,
                    query_embedding=query_embedding,
                    top_k=5,
                    entities=entities,
                    intent=current_intent
                )
                books_ctx = rag_service.format_books_for_context(result)
                tool_context += f"\n[KHO SÁCH & TRẠNG THÁI]:\n{books_ctx}"
                sources.extend([{"type": "book", "data": b} for b in result])

            elif current_intent == "debt_check":
                if student_id:
                    # Correct signature: (db, query, entities, session_student_id)
                    debt_result = await check_student_debt(
                        db, message, entities, session_student_id=student_id
                    )
                    tool_context += f"\n[THÔNG TIN NỢ/SÁCH ĐANG MƯỢN]:\n{debt_result['context']}"
                else:
                    tool_context += "\n[THÔNG TIN NỢ]: Sinh viên chưa đăng nhập, không thể kiểm tra nợ."

            elif current_intent == "renew_book":
                # Only available for authenticated users (already guarded above)
                if student_id:
                    renew_result = await renew_book_tool(db, student_id, entities)
                    tool_context += f"\n[KẾT QUẢ GIA HẠN]:\n{renew_result['context']}"
                    # If the tool asks for clarification (ambiguous books), LLM will relay it
                else:
                    tool_context += "\n[GIA HẠN]: Cần đăng nhập để gia hạn sách."

            elif current_intent == "reserve_book":
                if student_id:
                    reserve_result = await reserve_book_tool(db, student_id, entities)
                    tool_context += f"\n[KẾT QUẢ ĐẶT TRƯỚC]:\n{reserve_result['context']}"
                else:
                    tool_context += "\n[ĐẶT TRƯỚC]: Cần đăng nhập để đặt trước sách."

            if current_intent in ("policy_query", "return_book", "borrow_book"):
                # For procedures, always search the RAG Policy
                policy_result = await query_policy(db, message)
                policy_ctx = policy_result.get('context', '')
                policy_found = policy_result.get('found', False)
                if policy_ctx and policy_found:
                    tool_context += f"\n[QUY ĐỊNH THƯ VIỆN & THỦ TỤC {current_intent.upper()}]:\n{policy_ctx}"
                else:
                    # CRITICAL: Do NOT instruct the LLM to guess — force it to admit ignorance.
                    tool_context += f"\n[QUY ĐỊNH {current_intent.upper()}]: Không tìm thấy thông tin quy định cụ thể trong cơ sở dữ liệu. KHÔNG được tự bịa thủ tục — hãy hướng dẫn sinh viên liên hệ thủ thư."

            if current_intent in ("borrow_book", "return_book"):
                # Anti-Fraud: Require verified barcode scan
                # Only require scan when the action intent is primary. If borrow/return appears as a secondary
                # intent alongside "policy_query", we should answer policy instead of starting scan flow.
                if current_intent != primary_intent:
                    continue
                verified_code = front_metadata.get("verified_barcode")
                book_title = entities.get("book_title") or front_metadata.get("entities", {}).get("book_title")

                action_vn = "mượn" if current_intent == "borrow_book" else "trả"

                if not verified_code:
                    return {
                        "reply": f"Dạ, để hoàn tất thủ tục **{action_vn}** cuốn **{book_title or 'sách này'}**, bạn vui lòng **đưa mã vạch của sách** lên trước camera của Kiosk nhé!",
                        "intent": current_intent,
                        "metadata": {**response_metadata, "requires_action": "SCAN_BOOK", "target_action": current_intent},
                        "sources": [],
                        "is_scanner_trigger": True
                    }
                else:
                    try:
                        scanned_book = (
                            await db.execute(select(Book).where(Book.barcode == verified_code))
                        ).scalar_one_or_none()

                        if not scanned_book:
                            return {
                                "reply": "❌ Xin lỗi, hệ thống không nhận dạng được mã vạch này. Bạn vui lòng **quét lại mã vạch** giúp mình nhé!",
                                "intent": current_intent,
                                "metadata": {**response_metadata, "requires_action": "SCAN_BOOK", "target_action": current_intent},
                                "sources": [],
                                "is_scanner_trigger": True
                            }

                        # Set up a pending action for confirmation
                        self._pending_confirmations[session_id] = {
                            "action": current_intent,
                            "book_id": scanned_book.book_id,
                            "book_title": scanned_book.title,
                            "timestamp": time.monotonic()
                        }

                        is_match = (
                            (book_title and book_title.lower() in scanned_book.title.lower()) or
                            (book_title and scanned_book.title.lower() in book_title.lower())
                        )

                        if book_title and not is_match:
                            return {
                                "reply": f"⚠️ Cuốn sách bạn vừa quét là **{scanned_book.title}**, có vẻ khác với cuốn **{book_title}** mà bạn yêu cầu ban đầu.\n\nBạn có chắc chắn muốn **{action_vn}** cuốn **{scanned_book.title}** này không? (Trả lời 'Có' hoặc 'Không')",
                                "intent": "confirm_action",
                                "sources": [],
                                "metadata": response_metadata
                            }
                        else:
                            return {
                                "reply": f"✅ Đã quét thành công cuốn **{scanned_book.title}**.\n\nBạn hãy xác nhận lại lần cuối: Bạn có chắc chắn muốn **{action_vn}** cuốn sách này không? (Trả lời 'Có' hoặc 'Không')",
                                "intent": "confirm_action",
                                "sources": [],
                                "metadata": response_metadata
                            }
                    except Exception as e:
                        logger.error(f"[Scan] Error fetching book: {e}")
                        return {
                            "reply": "❌ Xảy ra lỗi khi xác thực sách. Vui lòng thử lại.",
                            "intent": "general_chat",
                            "sources": [],
                            "metadata": response_metadata
                        }

        # === Step 5: Build enriched prompt ===
        recommendation_context = ""
        if student_id and "book_search" in intents:
            rec_result = await get_personalized_recommendations(db, student_id)
            if rec_result.get("has_recommendations"):
                recommendation_context = rec_result["context"]

        full_student_info = student_context.get("context", "")
        urgent_msgs = student_context.get("urgent_alerts", [])

        system_prompt = self._build_system_prompt(
            intents,
            tool_context,
            student_name,
            full_student_info,
            recommendation_context,
            urgent_msgs
        )

        # Prepare a structured profile for the frontend
        student_profile = None
        if student_id:
            student_profile = {
                "id": student_id,
                "full_name": student_name,
                "fine_balance": student_context.get("fine_balance", 0),
                "last_login": datetime.datetime.now(timezone.utc).isoformat()
            }

        llm_messages = [{"role": "system", "content": system_prompt}]

        # Force name consistency to prevent hallucination from history.
        # Injected here (before history) so the model sees it early in context.
        if student_id and student_name != "bạn":
            llm_messages.append({
                "role": "system",
                "content": (
                    f"⚠️ NHẮC NHỞ QUAN TRỌNG: Sinh viên hiện tại tên là **{student_name}**. "
                    f"Chỉ được xưng hô bằng tên này. Không được dùng bất kỳ tên khác."
                )
            })

        for h in history[-self.HISTORY_WINDOW:]:
            llm_messages.append(h)
        llm_messages.append({"role": "user", "content": message})

        # === Step 6: Generate response ===
        llm = self._get_llm()
        response = await llm.chat(
            llm_messages,
            temperature=0.1,  # Lower temperature = extreme grounding = less hallucination
            top_p=0.85,
            num_ctx=4096,   # Explicit context window — avoids silent truncation on long conversations
            num_predict=600
        )
        reply = response.get("message", {}).get("content", "")

        # strip_think_tags is applied inside llm_service already, but apply here as safety net
        reply = _strip_think_tags(reply)

        return await self._finalize_response(
            db, session_id, reply, primary_intent, entities, sources,
            student_id, message, response_metadata, student_profile
        )


    async def _finalize_response(
        self, db, session_id, reply, intent, entities, sources, student_id, message_text, metadata=None, student_profile=None
    ) -> Dict[str, Any]:
        """Final helper to save history and return results."""
        if not reply:
            reply = "Xin lỗi, mình đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

        # Batch both messages into a single DB transaction
        await self._save_messages(db, session_id, message_text, reply, student_id, metadata)

        return {
            "reply": reply,
            "intent": intent,
            "entities": entities,
            "sources": sources[:5],
            "session_id": session_id,
            "success": True,
            "student_profile": student_profile
        }

    def _build_system_prompt(self, intents: List[str], tool_context: str, student_name: str = "bạn", student_info: str = "", recommendations: str = "", alerts: List[str] = None) -> str:
        """Construct intent-specific system prompt with dual-flow logic (Public vs Personalized)."""
        prompt = SYSTEM_PROMPT_BASE

        # --- Mode 1: PERSONALIZED (Logged in) ---
        if student_info:
            prompt += (
                f"\n\n# 🚨 QUY TẮC ĐỊNH DANH BẮT BUỘC (PERSONALIZED MODE)\n"
                f"- Bạn ĐANG nói chuyện với sinh viên: **{student_name}**.\n"
                f"- **BẮT BUỘC** câu chào đầu tiên phải là: 'Chào bạn {student_name}...' hoặc 'Chào {student_name}...'.\n"
                f"- Thông tin thực tế từ hệ thống:\n{student_info}\n"
                f"- Hãy nhớ: Tên duy nhất bạn được phép dùng là **{student_name}**."
            )

            if alerts:
                alert_text = "\n".join([f"- {a}" for a in alerts])
                prompt += f"\n\n> [!IMPORTANT]\n> **THÔNG BÁO KHẨN CẤP**:\n{alert_text}\n> Hãy nhắc sinh viên xử lý các sách này trước khi tư vấn tiếp."

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
        prompt += "\n\n**HƯỚNG DẪN XỬ LÝ THEO Ý ĐỊNH**:"

        if "book_search" in intents:
            prompt += (
                "\n- **Tìm sách**: Trình bày kết quả rõ ràng. "
                "Dùng **Table** để so sánh nếu có nhiều sách. Giới thiệu sơ qua chủ đề sách."
            )
        if "stock_check" in intents:
            prompt += (
                "\n- **Kiểm tra kho**: Chỉ rõ **Kệ (Shelf/Loc)** và trạng thái mượn."
            )
        if "debt_check" in intents:
            prompt += (
                "\n- **Kiểm tra nợ**: Trình bày dạng **Bảng kê khai**. "
                "Nhắc nhở về quy định đóng phạt nếu có nợ."
            )
        if "policy_query" in intents:
            prompt += (
                "\n- **Quy định**: Trích dẫn số liệu cụ thể từ ngữ cảnh (ngày, tiền phạt/ngày, v.v.)."
            )
        if "renew_book" in intents or "reserve_book" in intents:
            prompt += (
                "\n- **Gia hạn/Đặt trước**: Thông báo kết quả Thành công/Thất bại cụ thể. "
                "Nếu thành công, nhắc hạn trả mới hoặc vị trí hàng chờ."
            )
        if "return_book" in intents:
            prompt += (
                "\n- **Trả sách**: Chỉ mô tả các bước có trong TOOL CONTEXT. KHÔNG tự bịa thêm bước, tên phím, hay menu."
            )
        if "general_chat" in intents:
            prompt += (
                "\n- **Trò chuyện**: Thân thiện, giới thiệu chức năng thư viện."
            )

        # ANTI-HALLUCINATION GROUNDING: injected directly before context so the model sees it last
        prompt += (
            "\n\n⚠️ **QUY TẮC CUỐI CÙNG**: Phản hồi PHẢI dựa 100% vào TOOL CONTEXT bên dưới. "
            "Nếu TOOL CONTEXT không có thông tin, nói rõ là không tìm thấy và hướng sinh viên liên hệ thủ thư. "
            "TUYỆT ĐỐI không thêm thông tin không có trong TOOL CONTEXT."
        )

        prompt += f"\n\n--- DỮ LIỆU TỪ HỆ THỐNG (TOOL CONTEXT) ---\n{tool_context}\n--- KẾT THÚC TOOL CONTEXT ---"

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
            filter_clause = ChatHistory.student_id == student_id if student_id else ChatHistory.session_id == session_id

            stmt = (
                select(ChatHistory.role, ChatHistory.content)
                .where(filter_clause)
                .order_by(desc(ChatHistory.created_at))
                .limit(self.HISTORY_WINDOW)
            )
            result = await db.execute(stmt)
            rows = result.all()

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

    async def _save_messages(
        self,
        db: AsyncSession,
        session_id: str,
        user_message: str,
        ai_reply: str,
        student_id: Optional[str] = None,
        ai_metadata: Optional[Dict] = None,
    ) -> None:
        """Persist both the user message and AI reply in a single DB commit."""
        try:
            user_entry = ChatHistory(
                session_id=session_id,
                role="human",
                content=user_message,
                student_id=student_id,
                extra_metadata={}
            )
            ai_entry = ChatHistory(
                session_id=session_id,
                role="ai",
                content=ai_reply,
                student_id=student_id,
                extra_metadata=ai_metadata or {}
            )
            db.add(user_entry)
            db.add(ai_entry)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to save messages to history: {e}")
            await db.rollback()

    # Keep _save_message for backward compatibility (e.g. tests/other callers)
    async def _save_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
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

        # --- Cache hit check ---
        cached = self._student_cache.get(student_id)
        if cached and cached["expires_at"] > time.monotonic():
            logger.debug(f"[Cache] Student context HIT for {student_id}")
            return cached["data"]

        try:
            result = await db.execute(select(Student).where(Student.student_id == student_id))
            student = result.scalar_one_or_none()
            if not student:
                return {"name": "bạn", "context": ""}

            t_result = await db.execute(
                select(Transaction, Book.title)
                .join(Book, Transaction.book_id == Book.book_id)
                .where(Transaction.student_id == student_id)
                .where(Transaction.status.in_(["ACTIVE", "OVERDUE"]))
            )
            borrows = t_result.all()

            urgent_alerts = []
            now = datetime.datetime.now(timezone.utc).date()

            context_parts = [
                f"Tên đầy đủ: {student.full_name}",
                f"Mã số sinh viên: {student.student_id}",
                f"Trạng thái tài khoản: Đang hoạt động",
                f"Số dư nợ phạt: {student.fine_balance:,.0f} VND",
                f"Email: {getattr(student, 'email', 'N/A')}",
                f"Số điện thoại: {getattr(student, 'phone', 'N/A')}"
            ]

            if borrows:
                context_parts.append("Sách đang mượn:")
                for t, title in borrows:
                    status_str = "QUÁ HẠN" if t.status == "OVERDUE" else "Đang mượn"
                    context_parts.append(f"- {title} (Hạn: {t.due_date}) [{status_str}]")

                    if t.status == "OVERDUE":
                        urgent_alerts.append(f"Cuốn '{title}' đã QUÁ HẠN.")
                    elif t.due_date and (t.due_date.date() - now).days <= 2:
                        urgent_alerts.append(f"Cuốn '{title}' sắp hết hạn (Hạn: {t.due_date}).")
            else:
                context_parts.append("Hiện tại không mượn sách nào.")

            result_data = {
                "name": student.full_name,
                "fine_balance": student.fine_balance,
                "context": "\n".join(context_parts),
                "urgent_alerts": urgent_alerts
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


def _strip_think_tags(text: str) -> str:
    """Removes <think>...</think> from LLM responses if present."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# Singleton instance
chat_service = ChatService()
