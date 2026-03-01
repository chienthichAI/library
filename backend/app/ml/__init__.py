"""
SmartLib Kiosk - AI/ML Modules Package
Deep Neural Network implementations for Face Recognition and Book Detection
"""
import sys
import os

# --- Windows DLL Conflict Resolution (MANDATORY for GPU on Windows) ---
if sys.platform == 'win32':
    import site
    from loguru import logger
    
    # Add all nvidia-related bin paths to DLL search path to resolve name conflicts with torch
    for base_path in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = os.path.join(base_path, "nvidia")
        if os.path.exists(nvidia_root):
            for root, dirs, files in os.walk(nvidia_root):
                if 'bin' in dirs:
                    bin_dir = os.path.join(root, 'bin')
                    try:
                        os.add_dll_directory(bin_dir)
                        logger.debug(f"Added DLL directory: {bin_dir}")
                    except Exception:
                        pass
        
        # Also add torch/lib as fallback
        torch_lib = os.path.join(base_path, "torch", "lib")
        if os.path.exists(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
                logger.debug(f"Added Torch DLL directory: {torch_lib}")
            except Exception:
                pass
# ---------------------------------------------------------------------

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
