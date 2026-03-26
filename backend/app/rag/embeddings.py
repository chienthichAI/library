from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Converts text chunks into numerical vectors that capture semantic meaning.
    Using keepitreal/vietnamese-sbert which provides high performance at 1024d.
    Note: 'embeddings.position_ids | UNEXPECTED' warning is a known BERT/BGE load report 
    artifact that can be safely ignored.
    """
    _instance = None
    
    @classmethod
    def get_embeddings(cls):
        if cls._instance is None:
            logger.info("Initializing HuggingFaceEmbeddings (keepitreal/vietnamese-sbert)...")
            # For BGE models, normalization is recommended for cosine similarity/distances.
            cls._instance = HuggingFaceEmbeddings(
                model_name="keepitreal/vietnamese-sbert",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return cls._instance
