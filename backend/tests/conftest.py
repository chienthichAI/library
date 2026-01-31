"""
SmartLib Kiosk - Test Configuration

Pytest fixtures and configuration for testing.
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import os

# Set dummy env vars for testing BEFORE importing app
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/testdb"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test-key"
os.environ["SECRET_KEY"] = "test-secret-key"

from app.main import app
from app.database import async_session_maker, engine
from app.config import settings


# ============================================
# Pytest Configuration
# ============================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# HTTP Client Fixtures
# ============================================

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for API testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================
# Database Fixtures
# ============================================

@pytest.fixture
async def db_session():
    """Create a database session for testing."""
    async with async_session_maker() as session:
        yield session
        await session.rollback()


# ============================================
# Mock Fixtures for ML Models
# ============================================

@pytest.fixture
def mock_face_detector():
    """Mock face detector for testing without GPU."""
    with patch("app.ml.face_detector.FaceDetector") as mock:
        detector = MagicMock()
        detector.detect.return_value = [
            MagicMock(
                x1=100, y1=100, x2=200, y2=200,
                confidence=0.99,
                landmarks=None
            )
        ]
        mock.return_value = detector
        yield detector


@pytest.fixture
def mock_face_recognizer():
    """Mock face recognizer for testing without GPU."""
    with patch("app.ml.face_recognition.FaceRecognizer") as mock:
        recognizer = MagicMock()
        # Return a fake 512-dim embedding
        fake_embedding = np.random.randn(512).astype(np.float32)
        fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
        
        recognizer.extract_embedding.return_value = MagicMock(
            embedding=fake_embedding,
            confidence=0.95,
            is_valid=True
        )
        mock.return_value = recognizer
        yield recognizer


@pytest.fixture
def mock_anti_spoofing():
    """Mock anti-spoofing detector."""
    with patch("app.ml.anti_spoofing.AntiSpoofing") as mock:
        detector = MagicMock()
        detector.check.return_value = MagicMock(
            is_real=True,
            confidence=0.95
        )
        mock.return_value = detector
        yield detector


@pytest.fixture
def mock_book_detector():
    """Mock book detector for testing without GPU."""
    with patch("app.ml.book_detector.BookDetector") as mock:
        detector = MagicMock()
        detector.detect.return_value = MagicMock(
            books=[MagicMock(
                class_name="book",
                confidence=0.95,
                bbox=(50, 50, 400, 600)
            )],
            barcodes=[MagicMock(
                class_name="barcode",
                confidence=0.90,
                bbox=(100, 500, 300, 550)
            )],
            has_book=True,
            has_barcode=True
        )
        mock.return_value = detector
        yield detector


# ============================================
# Sample Data Fixtures
# ============================================

@pytest.fixture
def sample_student_data():
    """Sample student data for testing."""
    return {
        "student_id": "SE171234",
        "name": "Nguyen Van Test",
        "email": "test@fpt.edu.vn",
        "role": "STUDENT"
    }


@pytest.fixture
def sample_book_data():
    """Sample book data for testing."""
    return {
        "book_id": "BK001",
        "title": "Introduction to AI",
        "author": "Test Author",
        "isbn_13": "9781234567890",
        "barcode": "1234567890123",
        "status": "AVAILABLE"
    }


@pytest.fixture
def sample_image_bytes():
    """Generate a sample image for testing."""
    import io
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()
