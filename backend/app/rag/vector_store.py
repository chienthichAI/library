from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Stores embeddings and enables fast similarity search.
    Using FAISS as it is fast, scalable, and open-source.
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None
        
    def create_from_documents(self, documents: List[Document]):
        logger.info(f"Creating FAISS vector store from {len(documents)} documents...")
        self.db = FAISS.from_documents(documents, self.embeddings)
        logger.info("\u2705 Vector store created successfully.")
        return self.db
