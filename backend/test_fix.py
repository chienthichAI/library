
import asyncio
from app.rag.pipeline import RAGPipeline
import logging

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO)

async def test():
    pipeline = RAGPipeline()
    print("\nTesting question that should be saved to history...")
    response = await pipeline.ask_question("Quy trình mượn sách tại thư viện?", session_id="test_session_123")
    print(f"\nResponse: {response[:100]}...")
    print("\n✅ Successfully completed test interaction without error.")

if __name__ == "__main__":
    asyncio.run(test())
