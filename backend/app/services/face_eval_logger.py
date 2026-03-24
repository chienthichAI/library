"""
CSV logger for face-auth evaluation.

Creates one file per day so test sessions can run continuously,
then be evaluated in batch at the end of day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import uuid
from typing import Optional


@dataclass
class FaceEvalLogEntry:
    event_type: str
    predicted_student_id: str
    predicted_student_name: str
    confidence: float
    liveness_score: float
    is_real_face: bool
    quality_score: float
    processing_time_ms: float
    error_message: str = ""
    source: str = "api"
    track_id: str = ""
    # Fill this later (manually or by test client) for metrics.
    ground_truth_student_id: str = ""


class FaceEvalCsvLogger:
    """Append-only CSV logger with daily rotation by filename."""

    FIELDNAMES = [
        "event_id",
        "timestamp",
        "event_type",
        "source",
        "track_id",
        "ground_truth_student_id",
        "predicted_student_id",
        "predicted_student_name",
        "confidence",
        "liveness_score",
        "is_real_face",
        "quality_score",
        "processing_time_ms",
        "error_message",
    ]

    def __init__(self) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.log_dir = backend_root / "logs" / "face_eval"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _daily_file_path(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"face_eval_{date_str}.csv"

    def log(self, entry: FaceEvalLogEntry) -> str:
        """Append one event row and return generated event_id."""
        file_path = self._daily_file_path()
        event_id = str(uuid.uuid4())
        now_iso = datetime.now().isoformat(timespec="seconds")

        row = {
            "event_id": event_id,
            "timestamp": now_iso,
            "event_type": entry.event_type,
            "source": entry.source or "api",
            "track_id": entry.track_id or "",
            "ground_truth_student_id": entry.ground_truth_student_id or "",
            "predicted_student_id": entry.predicted_student_id or "UNKNOWN",
            "predicted_student_name": entry.predicted_student_name or "",
            "confidence": f"{entry.confidence:.6f}",
            "liveness_score": f"{entry.liveness_score:.6f}",
            "is_real_face": str(bool(entry.is_real_face)).lower(),
            "quality_score": f"{entry.quality_score:.6f}",
            "processing_time_ms": f"{entry.processing_time_ms:.2f}",
            "error_message": entry.error_message or "",
        }

        write_header = not file_path.exists() or file_path.stat().st_size == 0
        with file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return event_id


_LOGGER: Optional[FaceEvalCsvLogger] = None


def get_face_eval_logger() -> FaceEvalCsvLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = FaceEvalCsvLogger()
    return _LOGGER
