"""JWT issuing/verification + password hashing. Phase 4a."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User

JWT_ALGORITHM = "HS256"
DEFAULT_EXPIRES_MINUTES = 60

_VALID_ROLES = frozenset({"clinician", "admin", "readonly"})
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_MINUTES = 15


class AuthError(Exception):
    """Raised when token verification, signing, or password operations fail."""


# Bcrypt initialization is non-trivial — create once at import time, not per call.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise AuthError("JWT_SECRET env var is required")
    return secret


def create_access_token(
    actor_id: str,
    actor_role: str | None,
    expires_minutes: int = DEFAULT_EXPIRES_MINUTES,
) -> str:
    if not actor_id:
        raise ValueError("actor_id must be non-empty")
    if expires_minutes <= 0:
        raise ValueError(f"expires_minutes must be > 0, got {expires_minutes}")

    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "sub": actor_id,
        "role": actor_role,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        # jti is unique per token — required for the revoked_tokens blacklist.
        "jti": uuid.uuid4().hex,
    }

    try:
        return jwt.encode(claims, _get_secret(), algorithm=JWT_ALGORITHM)
    except JWTError as e:
        raise AuthError("failed to sign token") from e


def verify_token(token: str) -> dict[str, Any]:
    if not token:
        raise ValueError("token must be non-empty")

    try:
        # ExpiredSignatureError must be caught before JWTError — it is a subclass.
        claims = jwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except JWTError as e:
        raise AuthError("invalid token") from e

    if not claims.get("sub"):
        raise AuthError("token missing 'sub' claim")
    if not claims.get("jti"):
        raise AuthError("token missing 'jti' claim")

    return claims


def hash_password(plain: str) -> str:
    if not plain:
        raise ValueError("password must be non-empty")
    try:
        return _pwd_context.hash(plain)
    except Exception as e:
        raise AuthError("failed to hash password") from e


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        raise ValueError("plain and hashed must both be non-empty")
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception as e:
        raise AuthError("failed to verify password") from e


async def authenticate_user(
    username: str,
    password: str,
    session: AsyncSession,
) -> User:
    if not username:
        raise ValueError("username must be non-empty")
    if not password:
        raise ValueError("password must be non-empty")

    result = await session.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    # Same error message for unknown username and bad password — prevents user enumeration.
    if user is None or not user.is_active:
        raise AuthError("invalid credentials")

    now = datetime.now(timezone.utc)
    if user.locked_until is not None and user.locked_until > now:
        raise AuthError("account locked")

    try:
        ok = verify_password(password, user.password_hash)
    except AuthError:
        # Hashing/format issue — treat as auth failure but don't penalise the
        # user's lockout counter for what is a server-side hash format bug.
        raise AuthError("invalid credentials")

    if not ok:
        user.failed_login_attempts = user.failed_login_attempts + 1
        if user.failed_login_attempts >= _MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=_LOCKOUT_MINUTES)
        await session.commit()
        raise AuthError("invalid credentials")

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = now
    await session.commit()
    return user


async def register_user(
    username: str,
    password: str,
    role: str,
    session: AsyncSession,
) -> User:
    if not username:
        raise ValueError("username must be non-empty")
    if not password:
        raise ValueError("password must be non-empty")
    if role not in _VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        # Same message regardless of which unique constraint hit — caller doesn't need
        # to enumerate which field clashed.
        raise AuthError("username already taken") from e
    await session.refresh(user)
    return user
