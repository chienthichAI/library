"""
SmartLib Kiosk - API Integration Tests

Tests for core API endpoints.
"""
import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint returns API info."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "status" in data
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "services" in data


class TestBooksAPI:
    """Tests for Books API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_books(self, async_client: AsyncClient):
        """Test listing all books."""
        response = await async_client.get("/api/v1/books/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_book_not_found(self, async_client: AsyncClient):
        """Test getting a non-existent book returns 404."""
        response = await async_client.get("/api/v1/books/NONEXISTENT123")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_detect_book_requires_image(self, async_client: AsyncClient):
        """Test book detection endpoint requires image."""
        response = await async_client.post("/api/v1/books/detect")
        
        # Should return 422 (unprocessable entity) without image
        assert response.status_code == 422


class TestAuthAPI:
    """Tests for Authentication API endpoints."""
    
    @pytest.mark.asyncio
    async def test_verify_face_requires_image(self, async_client: AsyncClient):
        """Test face verification requires image upload."""
        response = await async_client.post("/api/v1/auth/verify-face")
        
        # Should return 422 without image
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_check_quality_requires_image(self, async_client: AsyncClient):
        """Test quality check requires image upload."""
        response = await async_client.post("/api/v1/auth/check-quality")
        
        # Should return 422 without image
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_register_face_requires_student_id(
        self, async_client: AsyncClient, sample_image_bytes
    ):
        """Test face registration requires student_id."""
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = await async_client.post(
            "/api/v1/auth/register-face",
            files=files
            # Missing student_id form field
        )
        
        # Should return 422 - missing required field
        assert response.status_code == 422


class TestTransactionsAPI:
    """Tests for Transactions API endpoints."""
    
    @pytest.mark.asyncio
    async def test_borrow_requires_valid_data(self, async_client: AsyncClient):
        """Test borrow endpoint requires student_id and book_id."""
        response = await async_client.post(
            "/api/v1/transactions/borrow",
            json={}  # Empty body
        )
        
        # Should return 422 - missing required fields
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_return_requires_valid_data(self, async_client: AsyncClient):
        """Test return endpoint requires student_id and book_id."""
        response = await async_client.post(
            "/api/v1/transactions/return",
            json={}  # Empty body
        )
        
        # Should return 422 - missing required fields
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_get_history_requires_student_id(self, async_client: AsyncClient):
        """Test transaction history endpoint returns data for valid student."""
        response = await async_client.get("/api/v1/transactions/history/SE171234")
        
        # Should return 200 even if no transactions exist
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "transactions" in data
