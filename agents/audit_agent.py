"""Audit agent — access checks + response logging. Append-only. Phase 3."""

from __future__ import annotations

import os

from sqlalchemy import func, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from agents.synthesis_agent import SynthesisResult
from core.vector_store import ChunkResult
from db.models import AuditLog


class AuditError(Exception):
    """Raised when an audit log write fails. Wraps the underlying DB error."""


def _encryption_key() -> str:
    key = os.environ.get("DB_ENCRYPTION_KEY")
    if not key:
        raise AuditError("DB_ENCRYPTION_KEY env var is required for audit encryption")
    return key


async def check_access(
    actor_id: str,
    actor_role: str | None,
    patient_id: str,
    session: AsyncSession,
) -> bool:
    if not patient_id:
        raise ValueError("patient_id must be non-empty")

    # Phase 1 permissive rule: any non-empty actor_id is allowed.
    # Phase 4 replaces this with real RBAC (role + patient-consent checks).
    allowed = bool(actor_id and actor_id.strip())

    action = "query" if allowed else "access_denied"
    decision = "allowed" if allowed else "denied"

    try:
        await session.execute(
            insert(AuditLog).values(
                actor_id=actor_id,
                actor_role=actor_role,
                action=action,
                patient_id=patient_id,
                query_text=None,
                chunks_retrieved=None,
                response_text=None,
                decision=decision,
            )
        )
        # Commit here — audit writes are independent of the calling transaction so the
        # row persists even if the caller crashes or rolls back its own transaction.
        await session.commit()
    except SQLAlchemyError as e:
        raise AuditError("failed to write access-check audit row") from e

    return allowed


async def log_response(
    actor_id: str,
    actor_role: str | None,
    patient_id: str,
    query_text: str,
    chunks: list[ChunkResult],
    result: SynthesisResult,
    session: AsyncSession,
) -> None:
    if not actor_id:
        raise ValueError("actor_id must be non-empty")
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if not query_text:
        raise ValueError("query_text must be non-empty")

    # str() because uuid.UUID is not JSON-serialisable and JSONB requires plain types.
    chunks_retrieved = [{"chunk_id": str(c.chunk_id), "score": c.score} for c in chunks]

    decision = "refused_no_context" if result.refused else "answered"

    try:
        # Encrypt the user-supplied query and LLM response at write time via
        # pgcrypto. Plaintext columns stay NULL for new rows; old rows keep
        # their original plaintext until a backfill job moves them across.
        key = _encryption_key()
        await session.execute(
            insert(AuditLog).values(
                actor_id=actor_id,
                actor_role=actor_role,
                action="query",
                patient_id=patient_id,
                query_text=None,
                chunks_retrieved=chunks_retrieved,
                response_text=None,
                query_text_enc=func.pgp_sym_encrypt(query_text, key),
                response_text_enc=func.pgp_sym_encrypt(result.answer, key),
                decision=decision,
            )
        )
        await session.commit()
    except SQLAlchemyError as e:
        raise AuditError("failed to write response audit row") from e
