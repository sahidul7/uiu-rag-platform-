"""
UIU Academic RAG Platform — Authentication Endpoints

Endpoints:
  POST /auth/google          — Exchange Google OAuth code for JWT tokens
  POST /auth/refresh         — Refresh access token
  GET  /auth/me              — Get current user profile
  POST /auth/logout          — Invalidate refresh token
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.core.exceptions import DomainRestrictionError, ExpiredTokenError, InvalidTokenError
from app.core.logging import get_logger
from app.core.security import create_token_pair, decode_token, validate_uiu_email
from app.db.session import get_db
from app.models.document import User

router = APIRouter()
logger = get_logger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ── Schemas ───────────────────────────────────────────────────────────────────
class GoogleAuthRequest(BaseModel):
    code: str
    redirect_uri: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_admin: bool


# ── Google OAuth ──────────────────────────────────────────────────────────────
@router.post("/google", response_model=TokenResponse)
async def google_auth(body: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """
    Exchange Google OAuth authorization code for platform JWT tokens.
    Enforces UIU email restriction if enabled.
    """
    # 1. Exchange code for Google access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": body.code,
            "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
            "redirect_uri": body.redirect_uri,
            "grant_type": "authorization_code",
        })

    if token_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate with Google.",
        )

    google_access_token = token_resp.json().get("access_token")

    # 2. Fetch user profile from Google
    async with httpx.AsyncClient() as client:
        profile_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {google_access_token}"},
        )

    if profile_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to fetch Google profile.",
        )

    profile = profile_resp.json()
    email: str = profile.get("email", "").lower()
    name: str = profile.get("name", "")
    google_sub: str = profile.get("sub", "")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No email returned from Google.",
        )

    # 3. UIU email domain check
    try:
        validate_uiu_email(email)
    except DomainRestrictionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message,
        )

    # 4. Upsert user in database
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user:
        user.last_login = datetime.now(timezone.utc)
        user.full_name = name or user.full_name
        user.google_sub = google_sub
    else:
        user = User(
            email=email,
            full_name=name,
            google_sub=google_sub,
            is_active=True,
            is_admin=False,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)

    logger.info("User authenticated via Google", email=email, user_id=user.id)

    # 5. Issue JWT tokens
    tokens = create_token_pair(user.id, user.email)
    return {
        **tokens,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
        },
    }


# ── Refresh Token ─────────────────────────────────────────────────────────────
@router.post("/refresh", response_model=dict)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Issue a new access token using a valid refresh token."""
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise InvalidTokenError("Not a refresh token")
    except (ExpiredTokenError, InvalidTokenError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    tokens = create_token_pair(user.id, user.email)
    return tokens


# ── Current User ──────────────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_admin": current_user.is_admin,
    }


# ── Google OAuth URL ──────────────────────────────────────────────────────────
@router.get("/google/url")
async def get_google_oauth_url():
    """Return the Google OAuth URL for frontend redirect."""
    if not settings.GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth not configured.",
        )

    params = {
        "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{query}"}
