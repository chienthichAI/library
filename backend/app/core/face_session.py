"""
Face-verified session token utilities.

This lightweight token is used to bind follow-up transactions/admin actions
to a recently verified identity from face authentication.
"""
import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict

from fastapi import Depends, Header, HTTPException

from app.config import settings


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("utf-8"))


def create_face_session_token(
    student_id: str,
    role: str,
    ttl_seconds: int = 300,
) -> str:
    now = int(time.time())
    payload = {
        "student_id": student_id,
        "role": role,
        "iat": now,
        "exp": now + ttl_seconds,
        "typ": "face_session",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    signature = hmac.new(
        settings.secret_key.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def decode_face_session_token(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError("Invalid token format")

    payload_b64, sig_b64 = parts
    expected_sig = hmac.new(
        settings.secret_key.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    provided_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(provided_sig, expected_sig):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("typ") != "face_session":
        raise ValueError("Invalid token type")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Token expired")
    return payload


def _extract_bearer_token(authorization: str) -> str:
    if not authorization:
        raise ValueError("Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ValueError("Invalid Authorization header")
    return token


def require_face_session(
    authorization: str = Header(..., alias="Authorization"),
) -> Dict[str, Any]:
    try:
        token = _extract_bearer_token(authorization)
        return decode_face_session_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or expired face session")


def require_admin_session(
    claims: Dict[str, Any] = Depends(require_face_session),
) -> Dict[str, Any]:
    if str(claims.get("role", "")).upper() != "ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden: admin access required")
    return claims
