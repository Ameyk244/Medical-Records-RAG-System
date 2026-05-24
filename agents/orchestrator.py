"""Orchestrator — sequences access check, retrieval, synthesis, audit. Phase 3."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from agents.audit_agent import AuditError, check_access, log_response
from agents.query_agent import QueryError, query_chunks
from agents.synthesis_agent import SynthesisError, SynthesisResult, synthesise


class OrchestratorError(Exception):
    """Raised when the orchestration flow fails. Wraps the underlying cause."""


async def handle_query(
    query_text: str,
    patient_id: str,
    actor_id: str,
    actor_role: str | None,
    session: AsyncSession,
) -> SynthesisResult:
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be non-empty")
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if not actor_id:
        raise ValueError("actor_id must be non-empty")

    try:
        allowed = await check_access(actor_id, actor_role, patient_id, session)
    except AuditError as e:
        raise OrchestratorError("audit access-check write failed") from e

    if allowed is False:
        return SynthesisResult(answer="Access denied.", citations=[], refused=True)

    chunks = []
    result: SynthesisResult | None = None
    pipeline_exc: Exception | None = None

    try:
        chunks = await query_chunks(query_text, patient_id, session)
        result = await synthesise(query_text, chunks)
    except (QueryError, SynthesisError) as e:
        pipeline_exc = e
        # Build a refused stub so log_response always receives a valid result object,
        # regardless of where in the pipeline the failure occurred.
        result = SynthesisResult(
            answer=f"REFUSED: synthesis pipeline failed ({type(e).__name__}).",
            citations=[],
            refused=True,
        )

    # Always log — chunks may be [] and result may be the failure stub.
    try:
        await log_response(actor_id, actor_role, patient_id, query_text, chunks, result, session)
    except AuditError as e:
        # Audit failure is surfaced as the primary cause: pipeline failures are recoverable,
        # but an audit gap is not. Chain the pipeline exception as context if both failed.
        if pipeline_exc is not None:
            raise OrchestratorError("pipeline failed AND audit write failed") from e
        raise OrchestratorError("audit response-write failed") from e

    if pipeline_exc is not None:
        raise OrchestratorError("query pipeline failed (audit row written)") from pipeline_exc

    return result
