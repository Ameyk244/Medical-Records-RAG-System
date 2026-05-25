"""Tiny local query runner for the real retrieval + synthesis pipeline.

Usage:
    python3 test_query.py "Who is the patient?"
    python3 test_query.py --patient-id P456 "What allergies are documented?"
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


class CliConfigError(Exception):
    """Raised when local CLI dependencies/config are missing."""


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
    # Load real local config first, then fill missing defaults from the example.
    _load_env_file(REPO_ROOT / ".env")
    _load_env_file(REPO_ROOT / ".env.example")


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise CliConfigError("DATABASE_URL env var is required")
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _is_psycopg_wrapper_error(exc: ImportError) -> bool:
    message = str(exc)
    return (
        "no pq wrapper available" in message
        or "psycopg_binary" in message
        or "libpq library not found" in message
    )


def _require_runtime_dependencies(*, include_synthesis: bool) -> None:
    missing: list[str] = []
    required_modules = [
        ("sqlalchemy", "sqlalchemy[asyncio]"),
        ("greenlet", "greenlet"),
        ("psycopg", "psycopg[binary]"),
        ("pgvector", "pgvector"),
        ("voyageai", "voyageai"),
    ]
    if include_synthesis:
        required_modules.append(("google.genai", "google-genai"))

    for module_name, package_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        packages = " ".join(f"'{package}'" for package in missing)
        raise CliConfigError(
            "Missing dependencies for the real query pipeline. "
            f"Run: pip install {packages}\n"
            "Or install everything with: pip install -r requirements.txt"
        )


def _require_api_keys(*, include_synthesis: bool) -> None:
    required_keys = ["VOYAGE_API_KEY"]
    if include_synthesis:
        required_keys.append("GOOGLE_API_KEY")

    missing = [
        key
        for key in required_keys
        if not os.environ.get(key, "").strip()
    ]
    if missing:
        raise CliConfigError(
            "Missing API key(s): "
            + ", ".join(missing)
            + ". Add them to .env or export them before running this real query pipeline."
        )


def _is_voyage_auth_error(exc: Exception) -> bool:
    exc_type = type(exc)
    return exc_type.__name__ == "AuthenticationError" and exc_type.__module__.startswith(
        "voyageai"
    )


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain = [exc]
    current = exc
    while current.__cause__ is not None:
        current = current.__cause__
        chain.append(current)
    return chain


def _is_gemini_quota_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        message = str(item)
        if "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message:
            return True
    return False


def _preview(text: str, limit: int = 300) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


async def _run(
    query_text: str,
    patient_id: str,
    top_k: int,
    *,
    synthesize_answer: bool,
) -> None:
    _require_runtime_dependencies(include_synthesis=synthesize_answer)
    _require_api_keys(include_synthesis=synthesize_answer)

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # type: ignore[import-untyped]

    from agents.query_agent import _infer_chunk_section_category, query_chunks

    try:
        engine = create_async_engine(_db_url(), future=True)
    except ImportError as exc:
        if _is_psycopg_wrapper_error(exc):
            raise CliConfigError(
                "Postgres driver is installed without a usable libpq wrapper. "
                "Run: pip install 'psycopg[binary]'"
            ) from exc
        raise
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as session:
            try:
                chunks = await query_chunks(query_text, patient_id, session, top_k=top_k)
            except Exception as exc:
                if _is_voyage_auth_error(exc):
                    raise CliConfigError(
                        "Voyage rejected VOYAGE_API_KEY. Check the key in .env."
                    ) from exc
                raise

            print(f"\nRETRIEVED CHUNKS ({len(chunks)})")
            for index, chunk in enumerate(chunks, start=1):
                category = _infer_chunk_section_category(chunk) or "unknown"
                print("\n" + "-" * 80)
                print(f"chunk index: {index}")
                print(f"section: {chunk.section or 'unknown'}")
                print(f"semantic category: {category}")
                print(f"retrieval score: {chunk.score:.4f}")
                print(f"preview: {_preview(chunk.text)}")

            print("\nFINAL ANSWER")
            if not synthesize_answer:
                print("SKIPPED: retrieval-only mode.")
                return

            try:
                from agents.synthesis_agent import synthesise

                result = await synthesise(query_text, chunks)
            except Exception as exc:
                if _is_gemini_quota_error(exc):
                    print(
                        "UNAVAILABLE: Gemini quota is exhausted. "
                        "Retrieved chunks above are still valid."
                    )
                    return
                raise
            print(result.answer)
    finally:
        await engine.dispose()


def main() -> None:
    _load_env_files()

    parser = argparse.ArgumentParser(
        description="Run a local query through the existing retrieval and synthesis pipeline."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language question to ask about the patient records",
    )
    parser.add_argument(
        "--patient-id",
        default=os.environ.get("PATIENT_ID", "P123"),
        help="Patient id to query; defaults to PATIENT_ID or P123",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved chunks and skip Gemini synthesis",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        print("error: --top-k must be > 0", file=sys.stderr)
        sys.exit(2)

    query_text = " ".join(args.query).strip()
    if not query_text:
        print("error: query must be non-empty", file=sys.stderr)
        sys.exit(2)

    try:
        asyncio.run(
            _run(
                query_text,
                patient_id=args.patient_id,
                top_k=args.top_k,
                synthesize_answer=not args.retrieval_only,
            )
        )
    except CliConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
