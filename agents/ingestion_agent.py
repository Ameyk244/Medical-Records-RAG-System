"""Ingestion agent — parse, chunk, embed, upsert. Phase 1."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import date
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.chunker import Chunk as TextChunk, chunk_document
from core.document_parser import parse_document
from core.embeddings import EMBEDDING_MODEL, embed_texts
from core.vector_store import ChunkRow, upsert_chunks
from db.models import Document


REPO_ROOT = Path(__file__).resolve().parents[1]


class IngestionError(Exception):
    """Raised when the ingestion pipeline fails. Wraps the underlying cause."""


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key or not value:
            continue
        if key not in os.environ or not os.environ[key].strip():
            os.environ[key] = value


def _load_env_files() -> None:
    _load_env_file(REPO_ROOT / ".env")
    _load_env_file(REPO_ROOT / ".env.example")


async def ingest_document(
    source_type: str,
    content: bytes,
    patient_id: str,
    document_date: date | None,
    ingested_by: str | None,
    source_uri: str,
    session: AsyncSession,
    *,
    original_filename: str | None = None,
    document_id: uuid.UUID | None = None,
) -> uuid.UUID:
    # Validate inputs before touching the database so we don't leave stray rows.
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if source_type not in {"pdf", "fhir_json", "hl7", "csv"}:
        raise ValueError(f"unsupported source_type: {source_type!r}")
    if not source_uri:
        raise ValueError("source_uri must be non-empty")

    # Phase A — insert the document row immediately so operational dashboards
    # can observe the in-progress state. We commit here so the row is visible
    # even if Phase B fails and we need to mark it "failed".
    # If document_id is passed in (e.g. by the HTTP route to match a pre-built
    # S3 key), use it; otherwise let Document's default uuid4 generator fire.
    document_kwargs: dict = {
        "patient_id": patient_id,
        "source_type": source_type,
        "source_uri": source_uri,
        "original_filename": original_filename,
        "document_date": document_date,
        "ingested_by": ingested_by,
        "chunk_count": 0,
        "embedding_model": EMBEDDING_MODEL,
        "status": "ingesting",
    }
    if document_id is not None:
        document_kwargs["id"] = document_id
    document = Document(**document_kwargs)
    session.add(document)
    await session.commit()
    await session.refresh(document)
    doc_id = document.id

    # Phase B — run the full pipeline and commit chunks + status="ready" atomically.
    # Phase C (inside the except) marks the doc "failed" in a fresh transaction
    # if anything in Phase B goes wrong, so the failure is visible in the DB.
    try:
        sections = parse_document(source_type, content, filename=original_filename)
        text_chunks: list[TextChunk] = chunk_document(sections)

        if not text_chunks:
            raise IngestionError("document produced zero chunks after chunking")

        embeddings = await embed_texts([tc.text for tc in text_chunks], input_type="document")

        if len(embeddings) != len(text_chunks):
            raise IngestionError(
                f"embedding count {len(embeddings)} != chunk count {len(text_chunks)}"
            )

        chunk_rows: list[ChunkRow] = [
            ChunkRow(
                document_id=doc_id,
                patient_id=patient_id,
                chunk_index=tc.chunk_index,
                text=tc.text,
                token_count=tc.token_count,
                section=tc.section,
                document_date=document_date,
                embedding=emb,
            )
            for tc, emb in zip(text_chunks, embeddings)
        ]

        await upsert_chunks(chunk_rows, session)
        await session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(chunk_count=len(text_chunks), status="ready")
        )
        await session.commit()
        return doc_id

    except ValueError:
        # Bad input — propagate untouched. Do not mark doc failed; do not wrap.
        raise
    except Exception as e:
        await session.rollback()
        # Phase C — mark the document row as "failed" in a fresh transaction so
        # operators can identify and reprocess it.
        await session.execute(
            update(Document).where(Document.id == doc_id).values(status="failed")
        )
        await session.commit()
        raise IngestionError(f"ingestion failed for document {doc_id}") from e


def _cli() -> None:
    _load_env_files()

    parser = argparse.ArgumentParser(
        description="Ingest a single document into the medical records RAG system."
    )
    parser.add_argument("--file", required=True, help="Path to the document on disk")
    parser.add_argument("--patient_id", required=True, help="Patient identifier (e.g. P123)")
    parser.add_argument(
        "--source_type",
        required=True,
        choices=["pdf", "fhir_json", "hl7", "csv"],
        help="Document type — determines which parser is used",
    )
    parser.add_argument(
        "--document_date",
        default=None,
        help="Clinical event date (YYYY-MM-DD). Defaults to None.",
    )
    parser.add_argument(
        "--ingested_by",
        default=None,
        help="User id of the ingesting user (Phase 4+ populates this).",
    )
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.is_file():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_bytes()
    doc_date = date.fromisoformat(args.document_date) if args.document_date else None
    source_uri = f"file://{file_path}"

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("error: DATABASE_URL env var is required", file=sys.stderr)
        sys.exit(1)

    # Accept the standard postgresql:// form from .env and rewrite to the
    # psycopg async dialect that SQLAlchemy 2.x requires for asyncpg/psycopg.
    if db_url.startswith("postgresql://") and "+" not in db_url.split("://", 1)[0]:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    doc_id = asyncio.run(
        _cli_run(
            db_url=db_url,
            source_type=args.source_type,
            content=content,
            patient_id=args.patient_id,
            document_date=doc_date,
            ingested_by=args.ingested_by,
            source_uri=source_uri,
            original_filename=file_path.name,
        )
    )
    print(f"ingested document {doc_id} (patient={args.patient_id}, type={args.source_type})")


async def _cli_run(
    *,
    db_url: str,
    source_type: str,
    content: bytes,
    patient_id: str,
    document_date: date | None,
    ingested_by: str | None,
    source_uri: str,
    original_filename: str | None,
) -> uuid.UUID:
    engine = create_async_engine(db_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            return await ingest_document(
                source_type=source_type,
                content=content,
                patient_id=patient_id,
                document_date=document_date,
                ingested_by=ingested_by,
                source_uri=source_uri,
                session=session,
                original_filename=original_filename,
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    _cli()
