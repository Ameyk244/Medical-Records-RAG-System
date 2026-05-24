"""Celery worker — async ingestion task wrapper."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import date

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.celery_app import celery_app
from agents.ingestion_agent import ingest_document


def _db_url() -> str:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def _run_ingest(
    source_type: str,
    content: bytes,
    patient_id: str,
    document_date_iso: str | None,
    ingested_by: str | None,
    source_uri: str,
    original_filename: str | None,
    document_id: str | None,
) -> str:
    engine = create_async_engine(_db_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            parsed_date = date.fromisoformat(document_date_iso) if document_date_iso else None
            doc_uuid = uuid.UUID(document_id) if document_id else None
            doc_id = await ingest_document(
                source_type=source_type,
                content=content,
                patient_id=patient_id,
                document_date=parsed_date,
                ingested_by=ingested_by,
                source_uri=source_uri,
                session=session,
                original_filename=original_filename,
                document_id=doc_uuid,
            )
            return str(doc_id)
    finally:
        await engine.dispose()


@celery_app.task(name="agents.ingestion_worker.ingest_task", bind=True)
def ingest_task(
    self,
    source_type: str,
    content: bytes,
    patient_id: str,
    document_date_iso: str | None,
    ingested_by: str | None,
    source_uri: str,
    original_filename: str | None,
    document_id: str | None,
) -> str:
    # Celery worker is sync; drive the async pipeline with asyncio.run().
    return asyncio.run(
        _run_ingest(
            source_type=source_type,
            content=content,
            patient_id=patient_id,
            document_date_iso=document_date_iso,
            ingested_by=ingested_by,
            source_uri=source_uri,
            original_filename=original_filename,
            document_id=document_id,
        )
    )
