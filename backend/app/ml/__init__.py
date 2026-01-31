"""
SmartLib Kiosk - AI/ML Modules Package
Deep Neural Network implementations for Face Recognition and Book Detection
"""
from app.ml.face_detector import FaceDetector
from app.ml.face_recognition import FaceRecognizer
from app.ml.anti_spoofing import AntiSpoofing
from app.ml.book_detector import BookDetector
from app.ml.barcode_reader import BarcodeReader
from app.ml.ocr_service import OCRService

__all__ = [
    "FaceDetector",
    "FaceRecognizer", 
    "AntiSpoofing",
    "BookDetector",
    "BarcodeReader",
    "OCRService"
]
