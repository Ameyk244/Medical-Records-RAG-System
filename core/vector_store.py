"""pgvector store — insert chunks and cosine-similarity search."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Chunk


class VectorStoreError(Exception):
    """Raised when a pgvector operation fails."""


@dataclass(frozen=True)
class ChunkRow:
    document_id: uuid.UUID
    patient_id: str
    chunk_index: int
    text: str
    token_count: int
    section: str | None
    document_date: date | None
    embedding: list[float]


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    section: str | None
    document_date: date | None
    score: float


async def upsert_chunks(chunks: list[ChunkRow], session: AsyncSession) -> None:
    if not chunks:
        return

    rows = [
        {
            "document_id": c.document_id,
            "patient_id": c.patient_id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "token_count": c.token_count,
            "section": c.section,
            "document_date": c.document_date,
            "embedding": c.embedding,
        }
        for c in chunks
    ]

    # Insert-only — no ON CONFLICT DO UPDATE. Re-ingesting a document issues a new
    # document_id; the old chunks are removed via CASCADE on the FK, not overwritten.
    try:
        await session.execute(insert(Chunk).values(rows))
    except SQLAlchemyError as e:
        raise VectorStoreError("failed to insert chunks") from e


async def search_chunks(
    query_embedding: list[float],
    patient_id: str,
    top_k: int,
    session: AsyncSession,
    *,
    document_date_from: date | None = None,
    document_date_to: date | None = None,
) -> list[ChunkResult]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if not patient_id:
        raise ValueError("patient_id must be non-empty")

    # cosine_distance maps to pgvector's <=> operator, which uses the HNSW index
    # declared in db/models.py (vector_cosine_ops).
    distance = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.section,
            Chunk.document_date,
            distance.label("distance"),
        )
        .where(Chunk.patient_id == patient_id)
        .order_by(distance)
        .limit(top_k)
    )

    if document_date_from is not None:
        stmt = stmt.where(Chunk.document_date >= document_date_from)
    if document_date_to is not None:
        stmt = stmt.where(Chunk.document_date <= document_date_to)

    try:
        result = await session.execute(stmt)
    except SQLAlchemyError as e:
        raise VectorStoreError("failed to search chunks") from e

    rows = result.all()

    return [
        ChunkResult(
            chunk_id=row.id,
            document_id=row.document_id,
            text=row.text,
            section=row.section,
            document_date=row.document_date,
            score=1 - row.distance,  # convert distance → similarity (higher is better)
        )
        for row in rows
    ]
