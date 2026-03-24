import inspect
import os

# Disable check to avoid slow startup for this test
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

try:
    from norfair import Tracker
    print("NORFAIR_TRACKER_ARGS:", inspect.getfullargspec(Tracker.__init__).args)
except Exception as e:
    print("NORFAIR_ERROR:", e)

try:
    from paddleocr import PaddleOCR
    print("PADDLEOCR_ARGS:", inspect.getfullargspec(PaddleOCR.__init__).args)
except Exception as e:
    print("PADDLEOCR_ERROR:", e)
