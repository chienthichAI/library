import inspect
import os
from loguru import logger

os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

try:
    from norfair import Tracker
    print("NORFAIR_TRACKER_SIG:", inspect.signature(Tracker.__init__))
except Exception as e:
    print("NORFAIR_ERROR:", e)

try:
    from paddleocr import PaddleOCR
    print("PADDLEOCR_SIG:", inspect.signature(PaddleOCR.__init__))
except Exception as e:
    print("PADDLEOCR_ERROR:", e)
