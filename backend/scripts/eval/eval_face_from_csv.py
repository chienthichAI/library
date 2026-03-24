"""
Evaluate face recognition metrics from daily CSV logs.

Usage examples:
  python scripts/eval/eval_face_from_csv.py
  python scripts/eval/eval_face_from_csv.py --csv logs/face_eval/face_eval_2026-03-25.csv
  python scripts/eval/eval_face_from_csv.py --only-events AUTH_SUCCESS AUTH_MISMATCH
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_EVENTS = {"AUTH_SUCCESS", "AUTH_MISMATCH", "AUTH_FAIL", "SPOOF_DETECTED"}
UNKNOWN_LABEL = "UNKNOWN"


def normalize_label(value: str) -> str:
    if not value:
        return UNKNOWN_LABEL
    v = value.strip()
    return v if v else UNKNOWN_LABEL


def latest_daily_csv(backend_root: Path) -> Path:
    folder = backend_root / "logs" / "face_eval"
    files = sorted(folder.glob("face_eval_*.csv"))
    if not files:
        raise FileNotFoundError(f"No face eval CSV found in: {folder}")
    return files[-1]


def read_rows(csv_path: Path, events: set[str]) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if normalize_label(r.get("event_type", "")) in events]
    return rows


def build_pairs(rows: List[dict]) -> Tuple[List[str], List[str], int]:
    y_true: List[str] = []
    y_pred: List[str] = []
    skipped = 0
    for r in rows:
        gt = normalize_label(r.get("ground_truth_student_id", ""))
        pred = normalize_label(r.get("predicted_student_id", ""))
        if gt == UNKNOWN_LABEL and not r.get("ground_truth_student_id", "").strip():
            skipped += 1
            continue
        y_true.append(gt)
        y_pred.append(pred)
    return y_true, y_pred, skipped


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    idx = {lb: i for i, lb in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return matrix


def per_class_stats(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    total = len(y_true)
    for lb in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lb and p == lb)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lb and p == lb)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lb and p != lb)
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(1 for t in y_true if t == lb)
        stats[lb] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return stats


def print_report(y_true: List[str], y_pred: List[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total else 0.0
    stats = per_class_stats(y_true, y_pred, labels)

    macro_precision = sum(stats[lb]["precision"] for lb in labels) / len(labels) if labels else 0.0
    macro_recall = sum(stats[lb]["recall"] for lb in labels) / len(labels) if labels else 0.0
    macro_f1 = sum(stats[lb]["f1"] for lb in labels) / len(labels) if labels else 0.0

    support_total = sum(stats[lb]["support"] for lb in labels) or 1
    weighted_f1 = sum(stats[lb]["f1"] * stats[lb]["support"] for lb in labels) / support_total

    print("=" * 72)
    print("FACE RECOGNITION EVALUATION")
    print("=" * 72)
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {macro_precision * 100:.2f}%")
    print(f"Macro Recall: {macro_recall * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print()
    print("Per-class metrics")
    print("-" * 72)
    print(f"{'Label':20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for lb in labels:
        s = stats[lb]
        print(f"{lb:20} {s['precision']*100:9.2f}% {s['recall']*100:9.2f}% {s['f1']*100:9.2f}% {int(s['support']):10d}")

    print()
    print("Confusion Matrix (rows=true, cols=pred)")
    print("-" * 72)
    matrix = confusion_matrix(y_true, y_pred, labels)
    header = " " * 14 + " ".join(f"{lb[:10]:>10}" for lb in labels)
    print(header)
    for i, lb in enumerate(labels):
        row = " ".join(f"{matrix[i][j]:10d}" for j in range(len(labels)))
        print(f"{lb[:12]:>12}  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face recognition metrics from CSV logs")
    parser.add_argument("--csv", type=str, default="", help="Path to face_eval_YYYY-MM-DD.csv")
    parser.add_argument("--only-events", nargs="*", default=list(DEFAULT_EVENTS), help="Event types to include")
    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parents[2]
    csv_path = Path(args.csv) if args.csv else latest_daily_csv(backend_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    events = set(args.only_events)
    rows = read_rows(csv_path, events)
    if not rows:
        print(f"No rows found in {csv_path} with events {sorted(events)}")
        return

    y_true, y_pred, skipped = build_pairs(rows)
    print(f"Input CSV: {csv_path}")
    print(f"Rows read: {len(rows)}")
    print(f"Rows skipped (missing ground truth): {skipped}")
    print()

    if not y_true:
        print("No labeled rows to evaluate.")
        print("Fill `ground_truth_student_id` first, then run again.")
        return

    print_report(y_true, y_pred)


if __name__ == "__main__":
    main()
