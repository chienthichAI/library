from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Converts text chunks into numerical vectors that capture semantic meaning.
    Using keepitreal/vietnamese-sbert which provides high performance at 768d.
    Note: 'embeddings.position_ids | UNEXPECTED' warning is a known BERT/BGE load report 
    artifact that can be safely ignored.
    """
    _instance = None
    
    @classmethod
    def get_embeddings(cls):
        if cls._instance is None:
            logger.info("Initializing HuggingFaceEmbeddings (keepitreal/vietnamese-sbert) with 1024d padding...")
            base_embeddings = HuggingFaceEmbeddings(
                model_name="keepitreal/vietnamese-sbert",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Wrap to pad 768 -> 1024
            class PaddedEmbeddings:
                def __init__(self, base):
                    self.base = base
                
                def _pad(self, vec):
                    if len(vec) >= 1024: return vec[:1024]
                    return vec + [0.0] * (1024 - len(vec))

                def embed_query(self, text):
                    return self._pad(self.base.embed_query(text))

                async def aembed_query(self, text):
                    return self._pad(await self.base.aembed_query(text))

                def embed_documents(self, texts):
                    return [self._pad(v) for v in self.base.embed_documents(texts)]

                async def aembed_documents(self, texts):
                    return [self._pad(v) for v in await self.base.aembed_documents(texts)]

            cls._instance = PaddedEmbeddings(base_embeddings)
            
        return cls._instance
