"""Phase 2 end-to-end smoke test — query + synthesise. Not a pytest."""
import asyncio
import os

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.query_agent import query_chunks
from agents.synthesis_agent import synthesise


async def main() -> None:
    db_url = os.environ["DATABASE_URL"]
    if db_url.startswith("postgresql://") and "+" not in db_url.split("://", 1)[0]:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_async_engine(db_url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    query = "Were any lab values flagged as elevated or abnormal?"
    patient = "P123"

    try:
        async with session_factory() as session:
            chunks = await query_chunks(query, patient, session, top_k=5)
            print(f"\n=== Retrieved {len(chunks)} chunks ===")
            for i, c in enumerate(chunks, 1):
                print(f"[{i}] score={c.score:.4f} section={c.section} date={c.document_date}")
                print(f"    {c.text[:200]}")

            result = await synthesise(query, chunks)
            print(f"\n=== Synthesis (refused={result.refused}) ===")
            print(f"Answer:\n{result.answer}\n")
            print(f"Citations ({len(result.citations)}):")
            for c in result.citations:
                print(f"  - chunk_id={c.chunk_id} section={c.section} score={c.score:.4f}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
