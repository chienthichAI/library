"""
SmartLib Kiosk - Audit Log Model
Layer 7: Production Hardening
Immutable audit log for security tracing and performance monitoring.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    event_type = Column(String, index=True) # E.g., AUTH_SUCCESS, AUTH_FAIL, SPOOF_DETECTED, REGISTER_SUCCESS
    
    # Optional linking
    student_id = Column(String, ForeignKey("students.student_id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Metrics
    similarity_score = Column(Float, nullable=True) # For auth matches
    liveness_score = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    
    # Details (JSON payload for flexibility)
    details = Column(JSON, default={})
    
    # Hardware Status at time of event
    gpu_fallback = Column(Boolean, default=False)
