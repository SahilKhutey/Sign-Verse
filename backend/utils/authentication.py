"""
Authentication Utilities — Password hashing and JWT tokens.
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES


_PBKDF2_ITERATIONS = 200_000
_PBKDF2_ALGO = "sha256"


def _pbkdf2_hash(password: str, salt: bytes, iterations: int) -> str:
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode(), salt, iterations)
    return dk.hex()


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC (slow hash)."""
    salt = os.urandom(16)
    iterations = _PBKDF2_ITERATIONS
    digest = _pbkdf2_hash(password, salt, iterations)
    return f"pbkdf2${iterations}${salt.hex()}${digest}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against stored hash (supports legacy format)."""
    if hashed.startswith("pbkdf2$"):
        try:
            _, iterations_str, salt_hex, digest = hashed.split("$", 3)
            iterations = int(iterations_str)
            salt = bytes.fromhex(salt_hex)
            check = _pbkdf2_hash(password, salt, iterations)
            return hmac.compare_digest(check, digest)
        except Exception:
            return False

    # Legacy format: "salt:sha256"
    try:
        salt, stored_hash = hashed.split(":")
        check = hashlib.sha256((password + salt).encode()).hexdigest()
        return hmac.compare_digest(check, stored_hash)
    except Exception:
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_token(payload: dict) -> str:
    """Create a signed access token with expiry."""
    now = int(time.time())
    body_obj = dict(payload)
    body_obj["iat"] = now
    body_obj["exp"] = now + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    body_obj["type"] = "access"
    body_obj.setdefault("jti", uuid.uuid4().hex)

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(body_obj).encode())

    signature = hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{body}.{signature}"


def verify_token(token: str) -> dict:
    """Verify and decode access token."""
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header, body, signature = parts

    expected_sig = hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(body))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None

    if payload.get("type") not in (None, "access"):
        return None

    return payload


def create_refresh_token(payload: dict, days: int = 14) -> str:
    """Create a signed refresh token."""
    now = int(time.time())
    body_obj = dict(payload)
    body_obj["iat"] = now
    body_obj["exp"] = now + (days * 24 * 60 * 60)
    body_obj["type"] = "refresh"
    body_obj.setdefault("jti", uuid.uuid4().hex)

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(body_obj).encode())
    signature = hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{body}.{signature}"


def verify_refresh_token(token: str) -> dict:
    """Verify and decode refresh token."""
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header, body, signature = parts
    expected_sig = hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(body))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None

    if payload.get("type") != "refresh":
        return None

    return payload


def hash_token(token: str) -> str:
    """Hash refresh token before persisting."""
    return hashlib.sha256(token.encode()).hexdigest()


def get_current_user(token: str) -> dict:
    """FastAPI dependency to verify token and return current user."""
    payload = verify_token(token)
    if not payload:
        raise Exception("Invalid or expired token")
    return payload
