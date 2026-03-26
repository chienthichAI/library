import asyncio
import os
import logging
from app.rag.pipeline import RAGPipeline
from dotenv import load_dotenv

async def main():
    load_dotenv('.env')
    # Set logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    
    pipeline = RAGPipeline()
    
    print("\n--- Testing General Question (should use Policy/FAQ/Books) ---")
    query = "Quy định mượn sách như thế nào?"
    answer = await pipeline.ask_question(query, session_id="test_session_1")
    print(f"Q: {query}")
    print(f"A: {answer}")
    
    print("\n--- Testing Semantic Cache (should hit) ---")
    # Small variation of the same question
    query2 = "Quy định mượn sách ra sao?"
    answer2 = await pipeline.ask_question(query2, session_id="test_session_2")
    print(f"Q: {query2}")
    print(f"A: {answer2}")

if __name__ == "__main__":
    asyncio.run(main())
