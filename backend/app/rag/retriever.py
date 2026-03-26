import logging

logger = logging.getLogger(__name__)

class RetrieverService:
    """
    Finds the most relevant document chunks based on user query similarity.
    """
    
    @staticmethod
    def create_retriever(db, k: int = 5):
        logger.info(f"Creating retriever with top_k={k}")
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
