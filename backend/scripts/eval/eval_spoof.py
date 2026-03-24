import os
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.anti_spoofing import AntiSpoofing
from scripts.eval.utils import calculate_metrics, print_report

def create_dummy_dataset(output_dir: str):
    """Creates a small dummy dataset of real and spoof faces."""
    logger.info(f"Creating dummy dataset at {output_dir}")
    real_dir = os.path.join(output_dir, "real")
    spoof_dir = os.path.join(output_dir, "spoof")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)
    
    # Generate 5 real and 5 spoof images
    for i in range(5):
        # Real images: relatively clean
        img_real = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        # Spoof images: add some noise/blur to simulate spoof
        img_spoof = cv2.GaussianBlur(img_real, (5, 5), 0)
        
        cv2.imwrite(os.path.join(real_dir, f"real_{i}.jpg"), img_real)
        cv2.imwrite(os.path.join(spoof_dir, f"spoof_{i}.jpg"), img_spoof)
            
    logger.info("Dummy dataset created successfully.")

def evaluate_anti_spoofing(dataset_dir: str, threshold: float = 0.5):
    """Evaluates the Anti-Spoofing model on real and spoof images."""
    spoof_detector = AntiSpoofing(threshold=threshold)
    success = spoof_detector.initialize()
    if not success:
        logger.error("Failed to initialize real AntiSpoofing model.")
        logger.error("Please ensure the ONNX weights are available in models/ directory, or set 'ALLOW_HEURISTIC_SPOOF=true'.")
        sys.exit(1)
    
    real_dir = os.path.join(dataset_dir, "real")
    spoof_dir = os.path.join(dataset_dir, "spoof")
    
    if not os.path.exists(real_dir) or not os.path.exists(spoof_dir):
        logger.error(f"Dataset must contain 'real' and 'spoof' subdirectories.")
        return
        
    y_true = []
    y_pred = []
    
    # Process real images (label = 1)
    real_files = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    for f in real_files:
        img_path = os.path.join(real_dir, f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = spoof_detector.detect(img)
        y_true.append(1)
        y_pred.append(1 if result.is_real else 0)
        
    # Process spoof images (label = 0)
    spoof_files = [f for f in os.listdir(spoof_dir) if f.endswith(('.jpg', '.png'))]
    for f in spoof_files:
        img_path = os.path.join(spoof_dir, f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = spoof_detector.detect(img)
        y_true.append(0)
        y_pred.append(1 if result.is_real else 0)
        
    if not y_true:
        logger.warning("No images found for evaluation.")
        return
        
    metrics = calculate_metrics(y_true, y_pred)
    print_report("Anti-Spoofing Evaluation Report", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Anti-Spoofing model")
    parser.add_argument("--dataset", type=str, default="data/eval/spoof_dataset", help="Path to dataset directory")
    parser.add_argument("--dummy", action="store_true", help="Generate a dummy dataset to test the pipeline")
    parser.add_argument("--threshold", type=float, default=0.5, help="Liveness score threshold")
    
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_dataset(args.dataset)
        
    if os.path.exists(args.dataset):
        evaluate_anti_spoofing(args.dataset, args.threshold)
    else:
        logger.error(f"Dataset directory not found: {args.dataset}. Use --dummy to generate one.")
