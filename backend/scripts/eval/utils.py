import os
import numpy as np
from typing import List, Dict, Any

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate Accuracy, Precision, Recall, F1-Score, FAR, FRR."""
    tp, tn, fp, fn = 0, 0, 0, 0
    for gt, pred in zip(y_true, y_pred):
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1

    total = len(y_true)
    if total == 0:
        return {}

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Total Samples": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "FAR (False Accept Rate)": far,
        "FRR (False Reject Rate)": frr,
        "Confusion Matrix": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }
    }

def print_report(title: str, metrics: Dict[str, Any]):
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<25}: {v*100:.2f}%")
        else:
            print(f"{k:<25}: {v}")
    print(f"{'='*40}\n")
