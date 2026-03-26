import re
from typing import List
from langchain_core.documents import Document

class Preprocessor:
    """
    Step 2: Preprocessing
    Langchain doesn't provide built-in preprocessing. Custom code required.
    """
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'Page \d+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        return text.strip()

    @staticmethod
    def clean_documents(documents: List[Document]) -> List[Document]:
        """Apply preprocessing to a list of loaded documents"""
        for doc in documents:
            doc.page_content = Preprocessor.preprocess_text(doc.page_content)
        return documents
