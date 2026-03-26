from sentence_transformers import CrossEncoder
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Reranks candidate results using a Cross-Encoder model.
    Using thanhtantran/Vietnamese_Reranker which is based on BGE-M3.
    This significantly improves the precision of the top results.
    """
    _instance = None
    
    @classmethod
    def get_reranker(cls):
        if cls._instance is None:
            logger.info("Initializing Vietnamese Reranker (thanhtantran/Vietnamese_Reranker)...")
            try:
                # thanhtantran/Vietnamese_Reranker is a fine-tuned BGE-M3 for Vietnamese
                cls._instance = CrossEncoder(
                    "thanhtantran/Vietnamese_Reranker",
                    device="cpu", # Default to CPU for local kiosk safety
                    max_length=512
                )
            except Exception as e:
                logger.error(f"Failed to load Reranker model: {e}")
                return None
        return cls._instance
