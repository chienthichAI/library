import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

database_url = os.environ.get("DATABASE_URL")
# Clean up sqlalchemy specific prefix
if database_url.startswith("postgresql+asyncpg://"):
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

async def run():
    print(f"Connecting to {database_url}...")
    try:
        conn = await asyncpg.connect(database_url)
        print("Connected.")
        
        # Test if table exists
        exists = await conn.fetchval("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'audit_logs')")
        if exists:
            print("Table 'audit_logs' ALREADY EXISTS.")
        else:
            print("Table 'audit_logs' DOES NOT EXIST. Creating it now...")
            await conn.execute("""
            CREATE TABLE audit_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                event_type VARCHAR(50) NOT NULL,
                student_id VARCHAR(20) REFERENCES students(student_id) ON DELETE SET NULL,
                similarity_score DOUBLE PRECISION,
                liveness_score DOUBLE PRECISION,
                processing_time_ms DOUBLE PRECISION,
                details JSONB DEFAULT '{}'::jsonb,
                gpu_fallback BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
            CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
            CREATE INDEX idx_audit_logs_student_id ON audit_logs(student_id);
            """)
            print("Created successfully.")
            
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run())
