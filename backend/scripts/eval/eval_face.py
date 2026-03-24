import os
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.face_recognition import FaceRecognizer
from insightface.app import FaceAnalysis
from scripts.eval.utils import calculate_metrics, print_report

def create_dummy_dataset(output_dir: str):
    """Creates a small dummy dataset of face pairs for testing the evaluation script."""
    logger.info(f"Creating dummy dataset at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # We will create random noise images that mimic faces (112x112)
    # This is purely to ensure the evaluation pipeline runs without errors.
    labels_file = os.path.join(output_dir, "pairs.txt")
    
    with open(labels_file, "w") as f:
        f.write("image1,image2,is_same\n")
        
        for i in range(10):  # 10 pairs
            is_same = 1 if i < 5 else 0
            img1_path = os.path.join(output_dir, f"img1_{i}.jpg")
            img2_path = os.path.join(output_dir, f"img2_{i}.jpg")
            
            # Generate random images
            img1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            img2 = img1.copy() if is_same else np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            cv2.imwrite(img1_path, img1)
            cv2.imwrite(img2_path, img2)
            
            f.write(f"img1_{i}.jpg,img2_{i}.jpg,{is_same}\n")
            
    logger.info("Dummy dataset created successfully.")

def evaluate_face_recognition(dataset_dir: str, threshold: float = 0.6):
    """Evaluates the Face Recognition model on a dataset of image pairs."""
    logger.info("Initializing InsightFace (buffalo_l) to use REAL model...")
    face_app = FaceAnalysis(name='buffalo_l')
    # Use CPU by default for eval to ensure it works everywhere
    face_app.prepare(ctx_id=-1)
    
    recognizer = FaceRecognizer(face_analysis_instance=face_app)
    recognizer.initialize()
    
    pairs_file = os.path.join(dataset_dir, "pairs.txt")
    if not os.path.exists(pairs_file):
        logger.error(f"Cannot find labels file: {pairs_file}")
        return
        
    y_true = []
    y_pred = []
    
    with open(pairs_file, "r") as f:
        lines = f.readlines()[1:] # skip header
        
    logger.info(f"Starting evaluation on {len(lines)} pairs...")
    
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
            
        img1_name, img2_name, is_same_gt = parts
        is_same_gt = int(is_same_gt)
        
        img1_path = os.path.join(dataset_dir, img1_name)
        img2_path = os.path.join(dataset_dir, img2_name)
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            continue
            
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Convert to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Extract embeddings
        emb1 = recognizer.extract_embedding(img1)
        emb2 = recognizer.extract_embedding(img2)
        
        if not emb1.is_valid or not emb2.is_valid:
            y_pred.append(0) # Prediction failed, assume not same
            y_true.append(is_same_gt)
            continue
            
        is_same_pred, conf = recognizer.is_same_person(emb1.embedding, emb2.embedding, threshold=threshold)
        
        y_true.append(is_same_gt)
        y_pred.append(1 if is_same_pred else 0)
        
    metrics = calculate_metrics(y_true, y_pred)
    print_report("Face Recognition Evaluation Report", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Recognition model")
    parser.add_argument("--dataset", type=str, default="data/eval/face_pairs", help="Path to dataset directory")
    parser.add_argument("--dummy", action="store_true", help="Generate a dummy dataset to test the pipeline")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for matching")
    
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_dataset(args.dataset)
        
    if os.path.exists(args.dataset):
        evaluate_face_recognition(args.dataset, args.threshold)
    else:
        logger.error(f"Dataset directory not found: {args.dataset}. Use --dummy to generate one.")
