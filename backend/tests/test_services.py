"""
SmartLib Kiosk - ML Services Tests

Unit tests for authentication and book identification services.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass


class TestAuthenticationService:
    """Tests for AuthenticationService logic."""
    
    @pytest.mark.asyncio
    async def test_select_best_face_single_face(self):
        """When only one face detected, return it."""
        from app.services.authentication_service import AuthenticationService
        from app.ml.face_detector import DetectedFace
        
        service = AuthenticationService()
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Use real DetectedFace class
        face = DetectedFace(
            bbox=(100, 100, 200, 200),
            confidence=0.95
        )
            
        faces = [face]
        result = service._select_best_face(mock_image, faces)
        
        assert result is not None
        assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_select_best_face_multiple_faces_picks_largest(self):
        """When multiple faces, pick the largest/most centered one."""
        from app.services.authentication_service import AuthenticationService
        from app.ml.face_detector import DetectedFace
        
        service = AuthenticationService()
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Small face in corner
        small_face = DetectedFace(
            bbox=(10, 10, 50, 50),
            confidence=0.99
        )
        
        # Large face in center (320, 240) - bbox around center
        # Center is (320, 240). Width 240 means x range [200, 440]
        # Height 240 means y range [120, 360]
        large_face = DetectedFace(
            bbox=(200, 120, 440, 360),
            confidence=0.90
        )
        
        faces = [small_face, large_face]
        result = service._select_best_face(mock_image, faces)
        
        # Should pick the larger, more centered face
        assert result is not None
        assert result.width == 240
    
    @pytest.mark.asyncio
    async def test_select_best_face_empty_returns_none(self):
        """When no faces, return None."""
        from app.services.authentication_service import AuthenticationService
        
        service = AuthenticationService()
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = service._select_best_face(mock_image, [])
        assert result is None


class TestTransactionService:
    """Tests for TransactionService logic."""
    
    @pytest.mark.asyncio
    async def test_borrow_validates_student_exists(self):
        """Borrow should fail if student doesn't exist."""
        from app.services.transaction_service import TransactionService
        
        service = TransactionService()
        mock_db = AsyncMock()
        
        # Mock _get_student directly to return None
        with patch.object(service, '_get_student', return_value=None):
            result = await service.borrow_book("unknown", "book1", mock_db)
        
        assert not result.success
        assert result.error_message is not None
        assert "Sinh viên không tồn tại" in result.error_message, f"Got unexpected error: {result.error_message}"
    
    @pytest.mark.asyncio
    async def test_fine_calculation_overdue(self):
        """Test fine calculation for overdue books."""
        from app.models.transaction import Transaction
        from datetime import datetime, date, timedelta
        
        transaction = Transaction(
            student_id="SE171234",
            book_id="BOOK_001",
            borrow_date=datetime.utcnow() - timedelta(days=20),
            due_date=date.today() - timedelta(days=5),  # 5 days overdue
        )
        transaction.return_date = datetime.utcnow()
        
        fine = transaction.calculate_fine(fine_per_day=10000)
        
        assert transaction.days_overdue == 5
        assert fine == 50000  # 5 days * 10000 VND
    
    @pytest.mark.asyncio
    async def test_fine_calculation_not_overdue(self):
        """Test no fine for on-time returns."""
        from app.models.transaction import Transaction
        from datetime import datetime, date, timedelta
        
        transaction = Transaction(
            student_id="SE171234",
            book_id="BOOK_001",
            borrow_date=datetime.utcnow() - timedelta(days=5),
            due_date=date.today() + timedelta(days=5),  # 5 days remaining
        )
        transaction.return_date = datetime.utcnow()
        
        fine = transaction.calculate_fine(fine_per_day=10000)
        
        assert transaction.days_overdue == 0
        assert fine == 0


class TestImageQualityChecker:
    """Tests for ImageQualityChecker."""
    
    def test_brightness_check_dark_image(self):
        """Dark images should have low brightness score."""
        from app.ml.quality_checker import ImageQualityChecker
        
        checker = ImageQualityChecker()
        dark_image = np.zeros((480, 640, 3), dtype=np.uint8) + 20  # Very dark
        
        score = checker._check_brightness(dark_image)
        
        assert score < 0.5  # Should be low score
    
    def test_brightness_check_bright_image(self):
        """Well-lit images should have good brightness score."""
        from app.ml.quality_checker import ImageQualityChecker
        
        checker = ImageQualityChecker()
        bright_image = np.zeros((480, 640, 3), dtype=np.uint8) + 120  # Good brightness
        
        score = checker._check_brightness(bright_image)
        
        assert score >= 0.5


class TestBookIdentificationService:
    """Tests for BookIdentificationService logic."""
    
    def test_pick_best_barcode_prefers_isbn(self):
        """Should prefer ISBN barcodes over regular barcodes."""
        from app.services.book_identification_service import BookIdentificationService
        from dataclasses import dataclass
        
        service = BookIdentificationService()
        
        @dataclass
        class MockBarcode:
            data: str
            is_isbn: bool
            confidence: float
        
        barcodes = [
            MockBarcode(data="REGULAR123", is_isbn=False, confidence=0.99),
            MockBarcode(data="9781234567890", is_isbn=True, confidence=0.90),
        ]
        
        result = service._pick_best_barcode(barcodes)
        
        assert result.is_isbn is True
        assert result.data == "9781234567890"
