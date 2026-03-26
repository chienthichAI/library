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
4. policy_query: Quy trình, thủ tục mượn/trả, quy định, giờ mở cửa, cách làm thẻ (VD: "Thủ tục trả sách thế nào", "Quy trình mượn sách", "Mở cửa đến mấy giờ?").
5. renew_book: Gia hạn sách đang mượn (VD: "Cho mình gia hạn", "Mượn thêm thời gian").
6. reserve_book: Đặt trước sách (VD: "Đặt trước cuốn Python").
7. borrow_book: Yêu cầu thực hiện MƯỢN SÁCH ngay lúc này (VD: "Cho tôi mượn cuốn sách này", "Mình muốn mượn sách").
8. return_book: Yêu cầu thực hiện TRẢ SÁCH ngay lúc này (VD: "Muốn trả cuốn sách này", "Trả sách giúp mình").
9. general_chat: Chào hỏi, cảm ơn, tán gẫu (VD: "Chào bạn", "Cảm ơn").

# ĐỊNH DẠNG TRẢ VỀ (JSON ONLY):
{
  "intents": ["tên_intent_1", "tên_intent_2"], 
  "entities": {
    "book_title": "tên sách cụ thể nếu có",
    "student_id": "MSSV nếu có",
    "topic": "chủ đề tìm kiếm hoặc thể loại nếu có (VD: lập trình, kinh tế)",
    "language": "mã ngôn ngữ nếu yêu cầu cụ thể (vi, en, ja, zh, km)"
  },
  "confidence": 1.0
}

# VÍ DỤ:
User: "Tìm sách python" -> {"intents": ["book_search"], "entities": {"topic": "python"}, "confidence": 1.0}
User: "Sách tiếng Anh về kinh tế" -> {"intents": ["book_search"], "entities": {"topic": "kinh tế", "language": "en"}, "confidence": 1.0}
User: "Mình muốn trả sách" -> {"intents": ["return_book"], "entities": {}, "confidence": 1.0}
User: "Thủ tục trả sách" -> {"intents": ["policy_query"], "entities": {}, "confidence": 1.0}
User: "alo" -> {"intents": ["general_chat"], "entities": {}, "confidence": 1.0}
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
        intent_set = (
            "book_search", "stock_check", "debt_check", "policy_query", 
            "renew_book", "reserve_book", "general_chat", "borrow_book", "return_book"
        )

        normalized = re.sub(r"[^a-zA-Z0-9_]+", "", (label or "")).lower()
        if not normalized:
            return None
            
        # Common synonyms from various NLU models
        synonyms = {
            "greeting": "general_chat", "thanks": "general_chat", "bye": "general_chat",
            "search": "book_search", "find": "book_search", "stock": "stock_check",
            "location": "stock_check", "policy": "policy_query", "help": "policy_query",
            "info": "policy_query", "borrow": "borrow_book", "return": "return_book",
            "fine": "debt_check", "money": "debt_check"
        }
        
        # Explicit mapping for the newly finetuned PhoBERT labels
        finetuned_mapping = {
            "find_book": "book_search",
            "check_stock": "stock_check",
            "check_debt": "debt_check",
            "general_chat": "general_chat",
            "policy_query": "policy_query",
            "renew_book": "renew_book",
            "reserve_book": "reserve_book",   # ← was missing
            "borrow_book": "borrow_book",
            "return_book": "return_book"
        }
        
        if normalized in finetuned_mapping:
            return finetuned_mapping[normalized]

        if normalized in intent_set:
            return normalized
        if normalized in synonyms:
            return synonyms[normalized]

        # Fuzzy match against our canonical intent keys
        best = difflib.get_close_matches(normalized, intent_set, n=1, cutoff=0.55)
        return best[0] if best else None

    async def _get_phobert(self) -> bool:
        if self._phobert_loaded:
            return self._phobert_available

        self._phobert_loaded = True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Ensure local paths are correctly found relative to the project root
            model_path = self._phobert_model_name.replace("\\", "/")
            if not os.path.exists(model_path) and not os.path.isabs(model_path):
                # Fallback to app-prefixed path if needed (standard for our structure)
                alt_path = os.path.join("backend" if "backend" not in os.getcwd() else ".", model_path)
                if os.path.exists(alt_path):
                    model_path = alt_path
            
            # If it looks like a local path (starts with / or has path separators), check if it exists
            if ("/" in model_path or "\\" in model_path) and not os.path.isdir(model_path) and "." not in model_path.split("/")[0]:
                 # Only skip if it's explicitly a local path and missing
                 logger.info(f"Local PhoBERT directory not found, will attempt to download from Hugging Face: {model_path}")
            
            # AutoTokenizer and AutoModel will handle both local paths and HF repo IDs
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
        Classify user intent (Multi-intent aware).
        """
        # Always extract student_id via regex first
        student_id = self._extract_student_id(message)
        base_entities: Dict[str, Any] = {}
        if student_id:
            base_entities["student_id"] = student_id

        # POINT 2: Domain Protection
        # If query contains library keywords, we should be skeptical of PhoBERT (smart-home model)
        lib_keywords = ["sách", "book", "mượn", "trả", "nợ", "phạt", "kệ", "tác giả", "tạp chí", "luận văn", "quy định", "thư viện"]
        is_library_query = any(word in message.lower() for word in lib_keywords)

        # 1) Try PhoBERT fast path (mostly for simple single intents)
        # Initialize ph_result before try so it's always accessible in the LLM fallback log below
        ph_result: Dict[str, Any] = {"intent": None, "entities": {}, "confidence": 0.0}
        try:
            ph_result = await self._classify_with_phobert(message)
            ph_intent = ph_result.get("intent")
            ph_conf = float(ph_result.get("confidence", 0.0) or 0.0)
            
            logger.debug(f"[DEBUG] PhoBERT raw output: intent='{ph_intent}', conf={ph_conf}")
            
            # If phobert predicts a smart home intent (like light/fan inferred via mapping) 
            # while it's a library query, we reject it.
            if ph_intent and ph_conf >= self._phobert_threshold:
                if not is_library_query or ph_intent != "general_chat": 
                   logger.info(f"PhoBERT Intent: {ph_intent} | Confidence: {ph_conf}")
                   return {
                       "intents": [ph_intent], # Return as list for consistency
                       "entities": base_entities | ph_result.get("entities", {}), 
                       "confidence": ph_conf
                   }
            else:
                logger.debug(f"PhoBERT did not pass threshold or no intent. Threshold={self._phobert_threshold}")
        except Exception as e:
            logger.warning(f"PhoBERT intent classification failed: {e}")

        # 2) LLM Fallback (accurate & multi-intent aware - Point 5)
        logger.info(f"PhoBERT rejected/low-confidence (intent={ph_result.get('intent')}, conf={ph_result.get('confidence')}). Using LLM for multi-intent detection.")
        llm = self._get_llm()
        
        prompt = INTENT_PROMPT
        if history:
            context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-2:]])
            prompt += f"\n\nContext:\n{context}"
        
        try:
            response = await llm.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.01,
                format="json"
            )
            
            content = _strip_think_tags(response.get("message", {}).get("content", ""))
            json_data = _extract_json(content)
            
            if json_data:
                intents = json_data.get("intents", [])
                if not intents and "intent" in json_data:
                    intents = [json_data["intent"]]
                
                # Default if empty
                if not intents: intents = ["general_chat"]
                
                # Merge entities
                entities = base_entities.copy()
                llm_entities = json_data.get("entities", {})
                if isinstance(llm_entities, dict):
                    entities.update({k: v for k, v in llm_entities.items() if v})
                
                # Auto-correct book_search
                if "general_chat" in intents and (entities.get("book_title") or entities.get("topic")):
                    intents = [i for i in intents if i != "general_chat"] + ["book_search"]
                
                return {
                    # Keep LLM-provided order while removing duplicates.
                    "intents": list(dict.fromkeys(intents)),
                    "entities": entities,
                    "confidence": json_data.get("confidence", 0.9)
                }
        except Exception as e:
            logger.error(f"LLM multi-intent classification failed: {e}")

        # Final fallback
        return {"intents": ["general_chat"], "entities": base_entities, "confidence": 0.0}


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
