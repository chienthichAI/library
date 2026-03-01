import asyncio
import os
from dotenv import load_dotenv

# Load explicitly
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from app.database import engine, Base
# Import all models to ensure they are registered with Base metadata
from app.models import *

async def create_tables():
    print("Creating missing tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully, check the database.")

if __name__ == "__main__":
    asyncio.run(create_tables())
