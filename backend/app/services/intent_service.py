"""
SmartLib - Intent Detection Service

Fast intent routing:
- Use PhoBERT intent classifier (if available) for quick intent prediction.
- Always extract `student_id` via regex format (no keyword intent rules).
- Fallback to LLM JSON classifier when PhoBERT is unavailable, unmapped, or low confidence.
"""

import asyncio
import difflib
import json
import os
import re
from typing import Dict, Any, Optional, Tuple

from loguru import logger


INTENT_PROMPT = """# BẮT BUỘC: CHỈ TRẢ VỀ JSON. TUYỆT ĐỐI KHÔNG GIẢI THÍCH.

Bạn là engine phân loại ý định (intent classifier) cho thư viện SmartLib. 
Hãy phân loại câu hỏi của người dùng vào 1 trong các loại sau:

1. book_search: Tìm sách, gợi ý sách, hỏi về chủ đề (VD: "Tìm sách Python", "Sách của AI").
2. stock_check: Hỏi về vị trí, còn sách không, ai đang mượn (VD: "Cuốn Java còn không?", "Sách ở kệ nào?").
3. debt_check: Kiểm tra nợ, tiền phạt, thông tin cá nhân của mình, sách đang mượn (VD: "Mình nợ bao nhiêu", "Thông tin tôi", "Sách mình đang mượn").
4. policy_query: Quy định, giờ mở cửa, cách làm thẻ, hạn mượn (VD: "Mở cửa đến mấy giờ?", "Mượn được bao lâu?").
5. renew_book: Gia hạn sách đang mượn (VD: "Cho mình gia hạn", "Mượn thêm thời gian").
6. reserve_book: Đặt trước sách (VD: "Đặt trước cuốn Python").
7. return_book: Hỏi về thủ tục trả sách hoặc thông báo trả sách (VD: "Muốn trả sách", "Thủ tục trả sách thế nào").
8. general_chat: Chào hỏi, cảm ơn, tán gẫu (VD: "Chào bạn", "Cảm ơn").

# ĐỊNH DẠNG TRẢ VỀ (JSON ONLY):
{
  "intent": "tên_intent",
  "entities": {
    "book_title": "tên sách cụ thể nếu có",
    "student_id": "MSSV nếu có",
    "topic": "chủ đề tìm kiếm hoặc thể loại nếu có (VD: lập trình, kinh tế)",
    "language": "mã ngôn ngữ nếu yêu cầu cụ thể (vi, en, ja, zh, km)"
  },
  "confidence": 1.0
}

# VÍ DỤ:
User: "Tìm sách python" -> {"intent": "book_search", "entities": {"topic": "python"}, "confidence": 1.0}
User: "Sách tiếng Anh về kinh tế" -> {"intent": "book_search", "entities": {"topic": "kinh tế", "language": "en"}, "confidence": 1.0}
User: "Mình muốn trả sách" -> {"intent": "return_book", "entities": {}, "confidence": 1.0}
User: "alo" -> {"intent": "general_chat", "entities": {}, "confidence": 1.0}
"""


from app.config import settings


class IntentService:
    """
    Hybrid intent classifier for the SmartLib chatbot.
    """

    def __init__(self):
        self._llm = None  # Lazy import to avoid circular deps
        self._phobert = None
        self._phobert_available = False
        self._phobert_loaded = False

    @property
    def _phobert_model_name(self) -> str:
        return settings.intent_phobert_model

    @property
    def _phobert_threshold(self) -> float:
        # If settings has it, use it, else default to 0.45
        return settings.intent_phobert_confidence_threshold or 0.45

    def _get_llm(self):
        """Lazy load LLM service to avoid circular imports."""
        if self._llm is None:
            from app.services.llm_service import ai_intent_classifier
            self._llm = ai_intent_classifier
        return self._llm

    def _extract_student_id(self, message: str) -> Optional[str]:
        """
        Extract student_id by format only (e.g. QE190047).
        No keyword-based intent detection is used.
        """
        m = re.search(r"\b([A-Z]{2})\s*(\d{3,})\b", message, flags=re.IGNORECASE)
        if not m:
            return None
        return f"{m.group(1).upper()}{m.group(2)}"

    def _map_phobert_label_to_intent(self, label: str) -> Optional[str]:
        """
        Map PhoBERT label to our intent set via fuzzy matching of label strings.
        """
        intent_set = ("book_search", "stock_check", "debt_check", "policy_query", "renew_book", "reserve_book", "general_chat")

        normalized = re.sub(r"[^a-zA-Z0-9_]+", "", (label or "")).lower()
        if not normalized:
            return None

        if normalized in intent_set:
            return normalized

        # Fuzzy match against our canonical intent keys
        best = difflib.get_close_matches(normalized, intent_set, n=1, cutoff=0.55)
        return best[0] if best else None

    async def _get_phobert(self) -> bool:
        if self._phobert_loaded:
            return self._phobert_available

        self._phobert_loaded = True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Ensure local paths use forward slashes for safer tokenizer loading in transformers
            model_path = self._phobert_model_name.replace("\\", "/")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            model.eval()

            import torch

            # Determine device based on settings and availability
            if settings.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                
            model.to(device)

            id2label = getattr(model.config, "id2label", None) or {}

            self._phobert = {
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
                "id2label": id2label,
            }
            self._phobert_available = True
            logger.info(f"PhoBERT intent classifier loaded: {self._phobert_model_name} (device={device})")
        except Exception as e:
            self._phobert_available = False
            logger.warning(f"PhoBERT intent classifier not available: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

        return self._phobert_available

    def _predict_intent_with_phobert_sync(self, message: str) -> Tuple[Optional[str], float]:
        """
        Synchronous PhoBERT inference. Called via thread executor.
        Returns (mapped_intent, confidence).
        """
        import torch

        ph = self._phobert
        if not ph:
            return None, 0.0

        tokenizer = ph["tokenizer"]
        model = ph["model"]
        device = ph["device"]
        id2label = ph["id2label"]

        inputs = tokenizer(message, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        idx = int(probs.argmax().item())
        confidence = float(probs[idx].item())
        raw_label = id2label.get(idx) if isinstance(id2label, dict) else None
        if raw_label is None:
            raw_label = str(idx)

        mapped_intent = self._map_phobert_label_to_intent(str(raw_label))
        return mapped_intent, confidence

    async def _classify_with_phobert(
        self, message: str
    ) -> Dict[str, Any]:
        if not await self._get_phobert():
            return {"intent": None, "entities": {}, "confidence": 0.0}

        mapped_intent, confidence = await asyncio.to_thread(
            self._predict_intent_with_phobert_sync, message
        )

        return {
            "intent": mapped_intent,
            "entities": {},
            "confidence": confidence,
        }

    async def classify(
        self, message: str, history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Classify user intent.
        - PhoBERT for intent prediction (fast).
        - LLM JSON fallback (accurate).
        """
        # Always extract student_id via regex first
        student_id = self._extract_student_id(message)
        base_entities: Dict[str, Any] = {}
        if student_id:
            base_entities["student_id"] = student_id

        # 1) Try PhoBERT fast path
        try:
            ph_result = await self._classify_with_phobert(message)
            ph_intent = ph_result.get("intent")
            ph_conf = float(ph_result.get("confidence", 0.0) or 0.0)
            if ph_intent and ph_conf >= self._phobert_threshold:
                logger.info(f"PhoBERT Intent: {ph_intent} | Confidence: {ph_conf}")
                return {"intent": ph_intent, "entities": base_entities | ph_result.get("entities", {}), "confidence": ph_conf}
        except Exception as e:
            logger.warning(f"PhoBERT intent classification failed: {e}")

        # 2) LLM Fallback (accurate)
        logger.info("PhoBERT unavailable or low confidence. Falling back to LLM intent classification.")
        llm = self._get_llm()
        
        # Include context if history is provided
        prompt = INTENT_PROMPT
        if history:
            # Last 2 messages for context
            context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-2:]])
            prompt += f"\n\nContext:\n{context}"
        
        try:
            response = await llm.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.01, # Stable high precision
                format="json"
            )
            
            content = response.get("message", {}).get("content", "")
            logger.debug(f"[Intent LLM Raw] {content}")
            
            # Strip think-tags BEFORE JSON parsing (Qwen models emit <think>...</think>)
            content = _strip_think_tags(content)
            logger.debug(f"[Intent LLM Cleaned] {content}")
            json_data = _extract_json(content)
            
            if json_data and "intent" in json_data:
                # Merge entities
                entities = base_entities.copy()
                llm_entities = json_data.get("entities", {})
                if isinstance(llm_entities, dict):
                    entities.update({k: v for k, v in llm_entities.items() if v})
                
                # Auto-correct: If general_chat but has book entities, it's likely book_search
                if json_data["intent"] == "general_chat":
                    if entities.get("book_title") or entities.get("topic"):
                        json_data["intent"] = "book_search"
                        logger.info(f"Auto-corrected intent to book_search due to entities: {entities}")
                
                return {
                    "intent": json_data["intent"],
                    "entities": entities,
                    "confidence": json_data.get("confidence", 0.9)
                }
            else:
                 logger.warning(f"Failed to parse JSON or intent missing from content: {content[:100]}...")
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")

        # Final fallback
        return {"intent": "general_chat", "entities": base_entities, "confidence": 0.0}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from Qwen 3.5 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Optional[Dict]:
    """Exhaustive search for valid JSON starting from any '{'."""
    # Try direct parse
    text_clean = text.strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # Hunt for JSON inside the text
    starts = [m.start() for m in re.finditer(r'\{', text)]
    for start in starts:
        ends = [m.start() for m in re.finditer(r'\}', text[start:])]
        # Check from longest possible candidate downward
        for end_rel in reversed(ends):
            candidate = text[start : start + end_rel + 1]
            try:
                # Basic bracket count check for quick filter
                if candidate.count('{') == candidate.count('}'):
                    return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None

# Singleton instance
intent_service = IntentService()
