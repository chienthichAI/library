from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Converts text chunks into numerical vectors that capture semantic meaning.
    Using AITeamVN/Vietnamese_Embedding which provides SOTA performance at 1024d.
    BGE-M3 base with specialized Vietnamese fine-tuning.
    """
    _instance = None
    
    @classmethod
    def get_embeddings(cls):
        if cls._instance is None:
            logger.info("Initializing HuggingFaceEmbeddings (AITeamVN/Vietnamese_Embedding - 1024d)...")
            # For BGE models, normalization is recommended for cosine similarity/distances.
            cls._instance = HuggingFaceEmbeddings(
                model_name="AITeamVN/Vietnamese_Embedding",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return cls._instance
