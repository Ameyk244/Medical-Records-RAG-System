"""Ad-hoc query — pass any question as a CLI arg.

Usage:  PYTHONPATH=. ./.venv/bin/python tests/ask.py "your question here"
"""
import asyncio
import os
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.query_agent import query_chunks
from agents.synthesis_agent import synthesise


async def main(query: str) -> None:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_async_engine(url, future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as session:
            chunks = await query_chunks(query, "P123", session, top_k=5)
            print(f"\n--- Retrieved {len(chunks)} chunks ---")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] section={c.section!r} date={c.document_date} score={c.score:.4f}")

            result = await synthesise(query, chunks)
            print(f"\n--- Synthesis (refused={result.refused}) ---")
            print(f"\n{result.answer}\n")
            print(f"--- Citations ({len(result.citations)}) ---")
            for c in result.citations:
                print(f"  - section={c.section!r} date={c.document_date} score={c.score:.4f}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python tests/ask.py "your question here"', file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
