"""FastAPI routes — auth, ingest (async via Celery), query, health. Phase 5 hardening."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, AsyncIterator

import structlog
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agents.celery_app import celery_app
from agents.ingestion_worker import ingest_task
from agents.orchestrator import OrchestratorError, handle_query
from core.auth import (
    AuthError, authenticate_user, create_access_token, hash_password,
    register_user, verify_token,
)
from core.cache import CacheError, get_cached_query_result, invalidate_patient_cache, set_cached_query_result
from core.middleware import (
    RequestIDMiddleware, SecurityHeadersMiddleware, configure_logging, setup_metrics,
)
from core.s3 import S3Error, ensure_bucket_exists, upload_document
from db.models import RefreshToken, RevokedToken, User

_log = structlog.get_logger()


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _redis_storage_uri() -> str:
    # Falls back to in-process memory if REDIS_URL is unset (dev/test).
    return os.environ.get("REDIS_URL", "memory://")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # configure_logging must run first so every subsequent log line is structured JSON.
    configure_logging()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var is required")
    if db_url.startswith("postgresql://") and "+" not in db_url.split("://", 1)[0]:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_async_engine(db_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app.state.engine = engine
    app.state.session_factory = session_factory

    await ensure_bucket_exists()

    # Bootstrap default users so existing tests pass without manual seeding.
    # Only creates them when the users table is completely empty — uses count()
    # to avoid a per-username query that would require both rows to exist.
    async with session_factory() as session:
        row = await session.execute(select(func.count()).select_from(User))
        count = row.scalar()
        if count == 0:
            for username, role in [("doctor1", "clinician"), ("admin", "admin")]:
                try:
                    await register_user(
                        username=username,
                        password="password123",
                        role=role,
                        session=session,
                    )
                except AuthError:
                    # Rare race on concurrent startups — silently skip.
                    pass

    try:
        yield
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="Medical Records RAG", lifespan=lifespan)

# CORS — dev-permissive so the file:// website can call the API. Tighten
# allow_origins to a specific host list in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cache", "X-Request-ID"],
)

# Outermost middleware first (i.e., added last = outermost).
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

# slowapi rate limiter — uses Redis when REDIS_URL is set, in-memory otherwise.
# Limiting is keyed on the remote IP address rather than the authenticated actor
# because slowapi does not natively support JWT-keyed limits; IP is an acceptable
# approximation for most deployment topologies (reverse proxy must forward the
# real IP via X-Forwarded-For).
limiter = Limiter(key_func=get_remote_address, storage_uri=_redis_storage_uri())
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus /metrics endpoint — must be called after all middleware is registered.
setup_metrics(app)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8)
    role: str = Field(pattern=r"^(clinician|admin|readonly)$")


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    query_text: str = Field(min_length=1, max_length=2000)


class CitationOut(BaseModel):
    chunk_id: str
    document_id: str
    section: str | None
    document_date: str | None
    score: float


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]


class IngestTaskResponse(BaseModel):
    task_id: str


class IngestStatusResponse(BaseModel):
    task_id: str
    state: str          # PENDING | STARTED | SUCCESS | FAILURE
    document_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


# HTTPBearer keeps the green Authorize button in Swagger UI.
# auto_error=False so we can return 401 instead of FastAPI's default 403.
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_actor(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
    session: Annotated[AsyncSession, Depends(get_session)] = None,
) -> dict[str, Any]:
    if credentials is None or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = verify_token(credentials.credentials.strip())
    except AuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jti = claims.get("jti")
    if jti:
        revoked = await session.execute(
            select(RevokedToken).where(RevokedToken.jti == jti)
        )
        if revoked.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Bind actor_id to structlog context so it appears in all log lines for this request.
    structlog.contextvars.bind_contextvars(actor_id=claims["sub"])

    return {
        "actor_id": claims["sub"],
        "actor_role": claims.get("role"),
        "jti": jti,
        "exp": claims.get("exp"),
    }


async def require_admin(
    actor: Annotated[dict[str, Any], Depends(get_current_actor)],
) -> dict[str, Any]:
    if actor.get("actor_role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin role required",
        )
    return actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_query_result(result: Any) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "refused": result.refused,
        "citations": [
            {
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "section": c.section,
                "document_date": c.document_date.isoformat() if c.document_date else None,
                "score": c.score,
            }
            for c in result.citations
        ],
    }


async def _check_db(request: Request) -> str:
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "fail"


async def _check_redis() -> str:
    try:
        import redis.asyncio as redis
        url = os.environ.get("REDIS_URL")
        if not url:
            return "fail"
        client = redis.from_url(url)
        await client.ping()
        await client.aclose()
        return "ok"
    except Exception:
        return "fail"


async def _check_s3() -> str:
    try:
        await ensure_bucket_exists()
        return "ok"
    except Exception:
        return "fail"


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/auth/login", response_model=TokenPair)
async def login(
    request: Request,
    body: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TokenPair:
    try:
        user = await authenticate_user(
            username=body.username,
            password=body.password,
            session=session,
        )
    except AuthError as exc:
        detail = "account locked" if "locked" in str(exc) else "invalid credentials"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

    try:
        access_token = create_access_token(actor_id=user.username, actor_role=user.role)
    except AuthError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to issue token",
        )

    raw_refresh = uuid.uuid4().hex
    refresh_row = RefreshToken(
        user_id=user.id,
        token_hash=_sha256_hex(raw_refresh),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    session.add(refresh_row)
    await session.commit()

    return TokenPair(access_token=access_token, refresh_token=raw_refresh)


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    body: RegisterRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict[str, Any], Depends(require_admin)],
) -> dict[str, Any]:
    try:
        user = await register_user(
            username=body.username,
            password=body.password,
            role=body.role,
            session=session,
        )
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return {"user_id": str(user.id), "username": user.username, "role": user.role}


@app.post("/auth/refresh", response_model=AccessTokenResponse)
async def refresh_token(
    body: RefreshRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AccessTokenResponse:
    token_hash = _sha256_hex(body.refresh_token)
    result = await session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    row = result.scalar_one_or_none()

    # Uniform 401 — don't reveal whether the token doesn't exist, is expired, or is revoked.
    now = datetime.now(timezone.utc)
    if row is None or row.revoked or row.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid refresh token",
        )

    user_result = await session.execute(select(User).where(User.id == row.user_id))
    user = user_result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid refresh token",
        )

    try:
        access_token = create_access_token(actor_id=user.username, actor_role=user.role)
    except AuthError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to issue token",
        )

    return AccessTokenResponse(access_token=access_token)


@app.post("/auth/logout")
async def logout(
    actor: Annotated[dict[str, Any], Depends(get_current_actor)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    jti = actor.get("jti")
    exp = actor.get("exp")

    if jti and exp:
        revoked_row = RevokedToken(
            jti=jti,
            expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
        )
        session.add(revoked_row)
        try:
            await session.flush()
        except IntegrityError:
            # jti already in revoked_tokens — idempotent, continue to refresh revocation.
            await session.rollback()

    # Mass-revoke all refresh tokens for this user on logout.
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == select(User.id).where(User.username == actor["actor_id"]).scalar_subquery())
        .values(revoked=True)
    )
    await session.commit()

    return {"detail": "logged out"}


# ---------------------------------------------------------------------------
# Ingest routes
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestTaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    request: Request,
    actor: Annotated[dict[str, Any], Depends(get_current_actor)],
    patient_id: Annotated[str, Form()],
    source_type: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    document_date: Annotated[str | None, Form()] = None,
) -> IngestTaskResponse:
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is empty",
        )

    parsed_date: date | None = None
    if document_date:
        try:
            parsed_date = date.fromisoformat(document_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document_date must be YYYY-MM-DD",
            )

    document_id = uuid.uuid4()
    s3_filename = file.filename or "uploaded"

    try:
        s3_key = await upload_document(
            patient_id=patient_id,
            document_id=str(document_id),
            filename=s3_filename,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except S3Error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 upload failed",
        )

    task = ingest_task.delay(
        source_type=source_type,
        content=content,
        patient_id=patient_id,
        document_date_iso=parsed_date.isoformat() if parsed_date else None,
        ingested_by=actor["actor_id"],
        source_uri=s3_key,
        original_filename=file.filename,
        document_id=str(document_id),
    )

    # Fire-and-forget cache invalidation. Even though ingestion is async the
    # cache for this patient should be stale the moment a new document is queued,
    # so we clear it eagerly. CacheError is non-fatal — cache is best-effort.
    try:
        await invalidate_patient_cache(patient_id)
    except CacheError:
        pass

    return IngestTaskResponse(task_id=task.id)


@app.get("/ingest/{task_id}/status", response_model=IngestStatusResponse)
async def ingest_status(
    task_id: str,
    _actor: Annotated[dict[str, Any], Depends(get_current_actor)],
) -> IngestStatusResponse:
    result = celery_app.AsyncResult(task_id)
    state = result.state

    if state == "SUCCESS":
        return IngestStatusResponse(task_id=task_id, state=state, document_id=result.result)
    if state == "FAILURE":
        return IngestStatusResponse(task_id=task_id, state=state, error=str(result.result))
    # PENDING, STARTED, REVOKED, RETRY, or any other Celery state.
    return IngestStatusResponse(task_id=task_id, state=state)


# ---------------------------------------------------------------------------
# Query route
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    actor: Annotated[dict[str, Any], Depends(get_current_actor)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    # 1. Check cache first.
    try:
        cached = await get_cached_query_result(body.patient_id, body.query_text)
    except CacheError:
        cached = None

    if cached is not None:
        return Response(
            content=QueryResponse(**cached).model_dump_json(),
            media_type="application/json",
            headers={"X-Cache": "HIT"},
        )

    # 2. Cache miss — run the full orchestrator pipeline.
    try:
        result = await handle_query(
            query_text=body.query_text,
            patient_id=body.patient_id,
            actor_id=actor["actor_id"],
            actor_role=actor["actor_role"],
            session=session,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except OrchestratorError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="query failed",
        )

    serialized = _serialize_query_result(result)

    # 3. Populate cache for next caller. Non-fatal if Redis is unavailable.
    try:
        await set_cached_query_result(body.patient_id, body.query_text, serialized)
    except CacheError:
        pass

    return Response(
        content=QueryResponse(**serialized).model_dump_json(),
        media_type="application/json",
        headers={"X-Cache": "MISS"},
    )


# ---------------------------------------------------------------------------
# Health routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    db_status = await _check_db(request)
    redis_status = await _check_redis()
    s3_status = await _check_s3()

    checks = {"db": db_status, "redis": redis_status, "s3": s3_status}
    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    # Always 200 — health consumers differentiate via the body, not HTTP status.
    return {"status": overall, "checks": checks}


@app.get("/health/ready")
async def health_ready(request: Request) -> Response:
    db_status = await _check_db(request)
    redis_status = await _check_redis()
    s3_status = await _check_s3()

    checks = {"db": db_status, "redis": redis_status, "s3": s3_status}
    all_ok = all(v == "ok" for v in checks.values())
    body = {"status": "ready" if all_ok else "not_ready", "checks": checks}

    return Response(
        content=json.dumps(body),
        media_type="application/json",
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    # Liveness only confirms the process is responsive — no dependency checks.
    return {"status": "alive"}
