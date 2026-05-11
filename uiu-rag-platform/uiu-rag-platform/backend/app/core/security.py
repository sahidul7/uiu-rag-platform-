"""
UIU Academic RAG Platform — Security & Authentication
JWT tokens + Google OAuth + UIU email domain restriction.

Flow:
  Student clicks "Sign in with Google"
  → Google OAuth returns profile (email, name, picture)
  → We check email ends with @uiu.ac.bd (if restriction enabled)
  → Issue JWT access + refresh tokens
  → Frontend stores tokens, sends with every request
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.exceptions import (
    DomainRestrictionError,
    ExpiredTokenError,
    InvalidTokenError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password Hashing ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(data: dict[str, Any]) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload["type"] = "access"
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict[str, Any]) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload["type"] = "refresh"
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        err = str(e).lower()
        if "expired" in err:
            raise ExpiredTokenError("Token has expired")
        raise InvalidTokenError(f"Invalid token: {e}")


def create_token_pair(user_id: str, email: str) -> dict[str, str]:
    data = {"sub": user_id, "email": email}
    return {
        "access_token": create_access_token(data),
        "refresh_token": create_refresh_token(data),
        "token_type": "bearer",
    }


# ── UIU Email Restriction ─────────────────────────────────────────────────────
def validate_uiu_email(email: str) -> None:
    """
    Enforce UIU email domain restriction if enabled.
    Raises DomainRestrictionError for non-UIU emails.
    """
    if not settings.RESTRICT_TO_UIU_EMAIL:
        return
    if not email.lower().endswith(f"@{settings.UIU_EMAIL_DOMAIN}"):
        raise DomainRestrictionError(
            f"Access restricted to @{settings.UIU_EMAIL_DOMAIN} email addresses.",
            details={"email": email, "required_domain": settings.UIU_EMAIL_DOMAIN},
        )
