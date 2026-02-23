"""
jwt_auth.py
───────────
JWT creation, verification, and token models for AttendAI.

Requirements:
    pip install python-jose[cryptography] passlib[bcrypt]

Environment variables (set in .env or your config):
    JWT_SECRET_KEY   – long random string  (CHANGE THIS IN PRODUCTION)
    JWT_ALGORITHM    – default: HS256
    ACCESS_TOKEN_EXPIRE_MINUTES  – default: 60
    REFRESH_TOKEN_EXPIRE_DAYS    – default: 7
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal

from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────
SECRET_KEY              = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_USE_A_LONG_RANDOM_STRING_IN_PROD")
ALGORITHM               = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))


# ── Token payload model ───────────────────────────────────────────
class TokenPayload(BaseModel):
    sub: str                          # "user_id:subdomain"  e.g. "42:acme"
    role: Literal["admin", "employee"]
    subdomain: str
    user_id: int
    exp: Optional[datetime] = None
    token_type: Literal["access", "refresh"] = "access"


# ── Token response model ──────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MIN * 60   # seconds
    role: str
    subdomain: str


# ── Create tokens ─────────────────────────────────────────────────
def _build_token(
    user_id:    int,
    subdomain:  str,
    role:       str,
    token_type: str,
    expires_delta: timedelta,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":        f"{user_id}:{subdomain}",
        "role":       role,
        "subdomain":  subdomain,
        "user_id":    user_id,
        "token_type": token_type,
        "iat":        now,
        "exp":        now + expires_delta,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: int, subdomain: str, role: str) -> str:
    return _build_token(
        user_id, subdomain, role,
        token_type="access",
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN),
    )


def create_refresh_token(user_id: int, subdomain: str, role: str) -> str:
    return _build_token(
        user_id, subdomain, role,
        token_type="refresh",
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def create_token_pair(user_id: int, subdomain: str, role: str) -> dict:
    """Returns both access + refresh tokens as a dict."""
    return {
        "access_token":  create_access_token(user_id, subdomain, role),
        "refresh_token": create_refresh_token(user_id, subdomain, role),
        "token_type":    "bearer",
        "expires_in":    ACCESS_TOKEN_EXPIRE_MIN * 60,
        "role":          role,
        "subdomain":     subdomain,
    }


# ── Verify / decode ───────────────────────────────────────────────
def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT.
    Raises jose.JWTError on failure (caller should convert to HTTPException).
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return TokenPayload(**payload)


def verify_access_token(token: str) -> TokenPayload:
    payload = decode_token(token)
    if payload.token_type != "access":
        raise JWTError("Not an access token")
    return payload


def verify_refresh_token(token: str) -> TokenPayload:
    payload = decode_token(token)
    if payload.token_type != "refresh":
        raise JWTError("Not a refresh token")
    return payload