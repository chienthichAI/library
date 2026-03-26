from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class ChunkerService:
    """
    Step 3: Text Chunking
    Splits large documents into smaller pieces for precise retrieval.
    Configured to chunk_size=300, chunk_overlap=50 for optimal handling.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,     # Max characters per chunk
            chunk_overlap=200,   # Overlap between chunks to maintain context
            separators=["\n\n", "\n", ".", " "]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} chunks")
        return docs

    def split_policy_text(self, text: str, section_title: str = "General", source: str = "Manual") -> List[dict]:
        """
        Splits a single policy text into formatted chunk dictionaries for DB insertion.
        """
        chunks = self.text_splitter.split_text(text)
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append({
                "chunk_text": chunk,
                "chunk_index": i,
                "section_title": section_title,
                "source": source
            })
        logger.info(f"Policy split into {len(formatted_chunks)} chunks")
        return formatted_chunks
