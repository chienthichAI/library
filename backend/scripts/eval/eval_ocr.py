import os
import argparse
import sys
import numpy as np
import cv2
import json
from pathlib import Path
from loguru import logger

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.book_detector import BookDetector
from app.ml.ocr_service import OCRService

def create_dummy_ocr_dataset(output_dir: str):
    """Creates a dummy dataset with text to test OCR pipeline."""
    logger.info(f"Creating dummy OCR dataset at {output_dir}")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    ground_truth = []
    
    for i in range(5):
        # Create a white canvas
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Draw a fake book bounding box
        cv2.rectangle(img, (100, 100), (500, 700), (200, 200, 200), -1)
        
        # Draw some text
        title = f"BOOK TITLE {i}"
        author = f"Author Name {i}"
        
        cv2.putText(img, title, (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, author, (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
        
        img_filename = f"book_{i}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, img)
        
        ground_truth.append({
            "image": img_filename,
            "expected_texts": [title, author]
        })
        
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    logger.info("Dummy OCR dataset created successfully.")

def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Very basic Word Error Rate calculation."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Simple intersection approach for basic testing. 
    # Real WER uses Levenshtein distance on words.
    matched = len(set(ref_words).intersection(set(hyp_words)))
    if len(ref_words) == 0:
        return 0.0
    return 1.0 - (matched / len(ref_words))

def evaluate_ocr(dataset_dir: str):
    """Evaluates the OCR and Book Detection pipeline."""
    # Ensure paddle isn't strictly required if the environment lacks it (mock fallback)
    detector = BookDetector()
    detector.initialize()
    
    ocr = OCRService()
    ocr.initialize()
    
    labels_path = os.path.join(dataset_dir, "labels.json")
    images_dir = os.path.join(dataset_dir, "images")
    
    if not os.path.exists(labels_path):
        logger.error(f"Cannot find labels file: {labels_path}")
        return
        
    with open(labels_path, "r") as f:
        ground_truth = json.load(f)
        
    total_samples = len(ground_truth)
    total_wer = 0.0
    books_detected = 0
    
    logger.info(f"Evaluating {total_samples} images...")
    
    for item in ground_truth:
        img_path = os.path.join(images_dir, item["image"])
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect Books
        det_result = detector.detect(img)
        if det_result.has_book:
            books_detected += 1
            
        # OCR
        extracted_texts = []
        ocr_results = ocr.extract_text(img)
        for r in ocr_results:
            extracted_texts.append(r.text)
            
        combined_pred = " ".join(extracted_texts)
        combined_ref = " ".join(item["expected_texts"])
        
        wer = calculate_word_error_rate(combined_ref, combined_pred)
        total_wer += wer
        
    avg_wer = total_wer / total_samples if total_samples > 0 else 0
    detection_rate = books_detected / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*40}")
    print("OCR and Book Detection Evaluation Report")
    print(f"{'='*40}")
    print(f"Total Samples          : {total_samples}")
    print(f"Book Detection Rate    : {detection_rate*100:.2f}%")
    print(f"Average WER (OCR)      : {avg_wer*100:.2f}% (Lower is better)")
    print(f"Estimated Accuracy     : {(1.0 - avg_wer)*100:.2f}%")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Book Detection & OCR")
    parser.add_argument("--dataset", type=str, default="data/eval/ocr_dataset", help="Path to OCR dataset directory")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy OCR dataset")
    
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_ocr_dataset(args.dataset)
        
    if os.path.exists(args.dataset):
        evaluate_ocr(args.dataset)
    else:
        logger.error(f"Dataset not found: {args.dataset}. Use --dummy to generate one.")
