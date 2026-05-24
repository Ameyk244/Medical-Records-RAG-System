"""Phase 4b HTTP end-to-end — ingest writes to S3 + source_uri is the S3 key."""
import asyncio
import sys
from pathlib import Path

import aioboto3
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
import os
import uuid

from api.routes import app, lifespan
from core.s3 import _client_kwargs, _bucket
from db.models import Document

BASE_URL = "http://test"
SAMPLE_CSV = Path("tests/sample_labs.csv")


def _db_url() -> str:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def _fetch_document_row(doc_id: str) -> dict | None:
    engine = create_async_engine(_db_url(), future=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            result = await session.execute(
                select(Document.id, Document.source_uri, Document.status, Document.chunk_count)
                .where(Document.id == uuid.UUID(doc_id))
            )
            row = result.first()
            if row is None:
                return None
            return {
                "id": str(row.id),
                "source_uri": row.source_uri,
                "status": row.status,
                "chunk_count": row.chunk_count,
            }
    finally:
        await engine.dispose()


async def _s3_object_exists(key: str) -> bool:
    session = aioboto3.Session()
    async with session.client("s3", **_client_kwargs()) as s3:
        try:
            await s3.head_object(Bucket=_bucket(), Key=key)
            return True
        except ClientError:
            return False


async def main() -> None:
    failures: list[str] = []

    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as client:
            # Get a JWT
            r = await client.post(
                "/auth/login",
                json={"username": "doctor1", "password": "password123"},
            )
            assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
            token = r.json()["access_token"]
            print("✓ logged in")

            # Ingest
            csv_bytes = SAMPLE_CSV.read_bytes()
            r = await client.post(
                "/ingest",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("sample_labs.csv", csv_bytes, "text/csv")},
                data={
                    "patient_id": "P123",
                    "source_type": "csv",
                    "document_date": "2025-11-05",
                },
            )
            if r.status_code != 200:
                print(f"✗ FAIL — ingest returned {r.status_code}: {r.text[:200]}")
                failures.append(f"ingest returned {r.status_code}")
                sys.exit(1)

            doc_id = r.json()["document_id"]
            print(f"✓ ingested document_id={doc_id}")

    # --- Test 1: source_uri is a documents/... key, not file:// ---
    row = await _fetch_document_row(doc_id)
    if row is None:
        print(f"✗ FAIL — document row not found in DB")
        failures.append("document row not found")
    else:
        print(f"  source_uri: {row['source_uri']}")
        print(f"  status: {row['status']}")
        print(f"  chunk_count: {row['chunk_count']}")
        if row["source_uri"].startswith("documents/") and "file://" not in row["source_uri"]:
            print("✓ PASS — source_uri is an S3 key, not file://")
        else:
            print(f"✗ FAIL — source_uri is not an S3 key: {row['source_uri']!r}")
            failures.append(f"source_uri wrong: {row['source_uri']}")

    # --- Test 2: the object actually exists in MinIO ---
    if row and row["source_uri"]:
        exists = await _s3_object_exists(row["source_uri"])
        if exists:
            print(f"✓ PASS — object exists in MinIO at key={row['source_uri']}")
        else:
            print(f"✗ FAIL — object not in MinIO at key={row['source_uri']}")
            failures.append(f"S3 object missing: {row['source_uri']}")

    # --- Test 3: source_uri embeds the same document_id we got back ---
    if row:
        if doc_id in row["source_uri"]:
            print("✓ PASS — document_id appears inside the S3 key (linkage preserved)")
        else:
            print(f"✗ FAIL — document_id {doc_id} not in source_uri {row['source_uri']!r}")
            failures.append("doc_id not embedded in source_uri")

    print("\n========== SUMMARY ==========")
    if failures:
        print(f"✗ {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✓ All Phase 4b checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
