import time
import sys
import os
sys.path.append('d:/Antigravity/Library/library/backend')

os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from app.ml.face_detector import FaceDetector
from app.ml.face_recognition import FaceRecognizer
import numpy as np
import cv2

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)

print("Initializing buffalo_s...")
fd = FaceDetector(model_name="buffalo_s", det_size=(320, 320), use_gpu=True)
fd.initialize()

print("Warming up...")
fd.detect(img, extract_embedding=False)

print("Benchmarking detection only...")
t0 = time.time()
faces = fd.detect(img, max_faces=5, extract_embedding=False)
t1 = time.time()
print(f"Detection only: {(t1-t0)*1000:.2f}ms")

print("Benchmarking with embeddings...")
t0 = time.time()
faces = fd.detect(img, max_faces=5, extract_embedding=True)
t1 = time.time()
print(f"Detection + Embedding: {(t1-t0)*1000:.2f}ms")

fr = FaceRecognizer(face_analysis_instance=fd._model, use_gpu=True)
fr.initialize()
face_112 = np.zeros((112, 112, 3), dtype=np.uint8)
t0 = time.time()
fr.extract_embedding(face_112)
t1 = time.time()
print(f"Recognition alone: {(t1-t0)*1000:.2f}ms")
