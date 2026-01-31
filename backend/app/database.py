"""
SmartLib Kiosk - Database Connection and Session Management
Configured for Supabase PostgreSQL
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.config import settings


# Create async engine for Supabase PostgreSQL
# Using NullPool for serverless/connection pooler compatibility
# statement_cache_size=0 required for pgbouncer in transaction mode
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    poolclass=NullPool,  # Important for Supabase connection pooler
    connect_args={
        "statement_cache_size": 0,  # Disable prepared statement caching for pgbouncer
        "prepared_statement_cache_size": 0,  # Additional safety for pgbouncer
    }
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


async def get_db() -> AsyncSession:
    """
    Dependency for getting database session.
    Usage in FastAPI endpoints:
        async def endpoint(db: AsyncSession = Depends(get_db)):
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database connection.
    
    Note: For Supabase with pgbouncer, we skip table creation since:
    1. Tables already exist in Supabase
    2. pgbouncer in transaction mode has issues with prepared statements
    
    We only test basic connectivity here.
    """
    try:
        # Test basic connectivity with a simple query
        from sqlalchemy import text
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            await session.commit()
    except Exception as e:
        # Log but don't fail - tables already exist in Supabase
        import logging
        logging.warning(f"Database init check: {e}")


async def close_db():
    """Close database connection."""
    await engine.dispose()
