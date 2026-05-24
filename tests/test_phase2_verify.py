"""Phase 2 extra verifications — refusal path, filename fix, multi-doc citations."""
import asyncio
import os
from datetime import date
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.ingestion_agent import ingest_document
from agents.query_agent import query_chunks
from agents.synthesis_agent import synthesise


def _db_url() -> str:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def scenario_1_refusal(session_factory):
    print("\n========== SCENARIO 1: refusal path ==========")
    q = "What were the findings of the patient's MRI scan of the brain?"
    print(f"Query: {q!r}")
    print("Expected: refused=True — there's no MRI data anywhere in the records.\n")
    async with session_factory() as session:
        chunks = await query_chunks(q, "P123", session, top_k=5)
        print(f"Retrieved {len(chunks)} chunks (similarity search still returns top-k)")
        result = await synthesise(q, chunks)
        print(f"refused={result.refused}")
        print(f"Answer: {result.answer}")
        print(f"Citations: {len(result.citations)}")


async def scenario_2_filename_fix(session_factory):
    print("\n========== SCENARIO 2: filename passthrough bug fix ==========")
    print("Re-ingesting tests/sample_labs.csv — expecting NEW chunk's section='Sample Labs'\n")
    content = Path("tests/sample_labs.csv").read_bytes()
    async with session_factory() as session:
        doc_id = await ingest_document(
            source_type="csv",
            content=content,
            patient_id="P123",
            document_date=date(2025, 11, 2),  # +1 day so it's distinguishable from the first ingest
            ingested_by=None,
            source_uri="file://tests/sample_labs.csv",
            session=session,
            original_filename="sample_labs.csv",
        )
        print(f"New document id: {doc_id}")
        chunks = await query_chunks("hemoglobin level", "P123", session, top_k=3)
        print("Retrieved chunks (showing section per chunk):")
        for c in chunks:
            print(f"  doc={c.document_id} date={c.document_date} section={c.section!r}")


async def scenario_3_multi_doc(session_factory):
    print("\n========== SCENARIO 3: multi-doc retrieval + multi-citation ==========")
    print("Ingesting tests/sample_meds.csv, then asking a question spanning labs + meds.\n")
    content = Path("tests/sample_meds.csv").read_bytes()
    async with session_factory() as session:
        await ingest_document(
            source_type="csv",
            content=content,
            patient_id="P123",
            document_date=date(2025, 9, 15),
            ingested_by=None,
            source_uri="file://tests/sample_meds.csv",
            session=session,
            original_filename="sample_meds.csv",
        )
        q = "What medications is the patient on and which lab values were flagged as elevated?"
        chunks = await query_chunks(q, "P123", session, top_k=5)
        print(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] section={c.section!r} date={c.document_date} score={c.score:.4f}")
        result = await synthesise(q, chunks)
        print(f"\nrefused={result.refused}")
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nCitations ({len(result.citations)}):")
        for c in result.citations:
            print(f"  - section={c.section!r} date={c.document_date} score={c.score:.4f}")


async def _wait(seconds: int, why: str) -> None:
    # Voyage free tier without a payment method = 3 RPM. We space the calls
    # to stay under the limit. Remove these sleeps if you add a payment method.
    print(f"\n[waiting {seconds}s — {why}]")
    await asyncio.sleep(seconds)


async def main():
    engine = create_async_engine(_db_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await scenario_1_refusal(session_factory)
        await _wait(70, "Voyage rate limit (3 RPM free tier) — clearing rolling window")
        await scenario_2_filename_fix(session_factory)
        await _wait(70, "Voyage rate limit (3 RPM free tier) — clearing rolling window")
        await scenario_3_multi_doc(session_factory)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
