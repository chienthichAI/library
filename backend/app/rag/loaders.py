from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, CSVLoader
import logging

logger = logging.getLogger(__name__)

class DocumentLoaderService:
    """
    Step 1: Document Loading
    Converts various document formats into structured text that can be processed.
    """
    
    @staticmethod
    def load_pdf(file_path: str):
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from PDF.")
        return documents
        
    @staticmethod
    def load_csv(file_path: str):
        logger.info(f"Loading CSV: {file_path}")
        loader = CSVLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from CSV.")
        return documents

    @staticmethod
    def load_text(file_path: str):
        loader = TextLoader(file_path)
        return loader.load()
