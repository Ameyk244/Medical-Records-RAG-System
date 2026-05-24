"""Query agent — embed query, retrieve top-k chunks. Phase 2."""

from __future__ import annotations

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import EmbeddingError, embed_texts
from core.vector_store import ChunkResult, VectorStoreError, search_chunks


class QueryError(Exception):
    """Raised when the query pipeline fails. Wraps the underlying cause."""


async def query_chunks(
    query_text: str,
    patient_id: str,
    session: AsyncSession,
    *,
    top_k: int = 5,
    document_date_from: date | None = None,
    document_date_to: date | None = None,
) -> list[ChunkResult]:
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be non-empty")
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    try:
        # input_type="query" (not "document") — Voyage produces asymmetric embeddings
        # for retrieval; query and document vectors live in different subspaces.
        # See core/embeddings.py for details.
        embeddings = await embed_texts([query_text.strip()], input_type="query")
    except EmbeddingError as e:
        raise QueryError("failed to embed query") from e

    query_vec = embeddings[0]

    try:
        results = await search_chunks(
            query_embedding=query_vec,
            patient_id=patient_id,
            top_k=top_k,
            session=session,
            document_date_from=document_date_from,
            document_date_to=document_date_to,
        )
    except VectorStoreError as e:
        raise QueryError("failed to search chunks") from e

    return results
