"""Interactive PDF ingestion + query session.

Pick a PDF, auto-detect its patient id, ingest it if needed, then ask queries
for that patient without retyping commands.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import re
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


class SessionConfigError(Exception):
    """Raised when local config or dependencies are missing."""


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


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SessionConfigError("DATABASE_URL env var is required")
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _require_modules(*, include_synthesis: bool) -> None:
    required_modules = [
        ("sqlalchemy", "sqlalchemy[asyncio]"),
        ("greenlet", "greenlet"),
        ("psycopg", "psycopg[binary]"),
        ("pgvector", "pgvector"),
        ("pdfplumber", "pdfplumber"),
        ("voyageai", "voyageai"),
    ]
    if include_synthesis:
        required_modules.append(("google.genai", "google-genai"))

    missing = [
        package_name
        for module_name, package_name in required_modules
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        packages = " ".join(f"'{package}'" for package in missing)
        raise SessionConfigError(
            "Missing dependencies. "
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
        raise SessionConfigError(
            "Missing API key(s): "
            + ", ".join(missing)
            + ". Add them to .env or export them before running."
        )


def _discover_pdfs() -> list[Path]:
    search_roots = [REPO_ROOT / "tests", REPO_ROOT]
    pdfs: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        for path in sorted(root.glob("*.pdf")):
            resolved = path.resolve()
            if resolved not in seen:
                pdfs.append(resolved)
                seen.add(resolved)
    return pdfs


def _select_pdf(pdf_arg: str | None) -> Path:
    if pdf_arg:
        pdf_path = Path(pdf_arg).expanduser().resolve()
        if not pdf_path.is_file():
            raise SessionConfigError(f"PDF file not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise SessionConfigError(f"Expected a PDF file, got: {pdf_path}")
        return pdf_path

    pdfs = _discover_pdfs()
    if not pdfs:
        raise SessionConfigError("No PDF files found in repo root or tests/")

    print("\nAvailable PDFs")
    for index, path in enumerate(pdfs, start=1):
        print(f"{index}. {path.relative_to(REPO_ROOT)}")

    while True:
        choice = input("\nSelect PDF number or enter a PDF path: ").strip()
        if not choice:
            continue
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(pdfs):
                return pdfs[selected_index - 1]
            print("Invalid number.")
            continue
        pdf_path = Path(choice).expanduser().resolve()
        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
            return pdf_path
        print("Invalid PDF path.")


def _parse_pdf_metadata(pdf_path: Path) -> tuple[str | None, date | None]:
    from core.document_parser import parse_document

    sections = parse_document("pdf", pdf_path.read_bytes(), filename=pdf_path.name)
    text = "\n".join(
        f"{section.heading or ''}\n{section.text}"
        for section in sections
    )

    patient_id_match = re.search(
        r"(?im)\bPatient\s+ID\s*[:\-]\s*([A-Za-z0-9_-]+)\b",
        text,
    )
    patient_id = patient_id_match.group(1).strip() if patient_id_match else None

    document_date = None
    for label in ("Discharge Date", "Admission Date", "Document Date"):
        match = re.search(
            rf"(?im)\b{re.escape(label)}\s*[:\-]\s*(\d{{4}}-\d{{2}}-\d{{2}})\b",
            text,
        )
        if match:
            document_date = date.fromisoformat(match.group(1))
            break

    return patient_id, document_date


def _prompt_for_patient_id(detected_patient_id: str | None) -> str:
    if detected_patient_id:
        entered = input(f"Patient id [{detected_patient_id}]: ").strip()
        return entered or detected_patient_id

    while True:
        entered = input("Patient id could not be detected. Enter patient id: ").strip()
        if entered:
            return entered


def _prompt_for_document_date(detected_date: date | None) -> date | None:
    default = detected_date.isoformat() if detected_date else ""
    entered = input(f"Document date YYYY-MM-DD [{default or 'none'}]: ").strip()
    if not entered:
        return detected_date
    return date.fromisoformat(entered)


def _preview(text: str, limit: int = 300) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


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


async def _ready_document_exists(session, patient_id: str, filename: str) -> bool:
    from sqlalchemy import select

    from db.models import Document

    result = await session.execute(
        select(Document.id)
        .where(Document.patient_id == patient_id)
        .where(Document.original_filename == filename)
        .where(Document.status == "ready")
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _ingest_if_needed(
    session,
    pdf_path: Path,
    patient_id: str,
    document_date: date | None,
    *,
    force_ingest: bool,
) -> None:
    from agents.ingestion_agent import ingest_document

    exists = await _ready_document_exists(session, patient_id, pdf_path.name)
    if exists and not force_ingest:
        print(f"Ready document already exists for patient {patient_id}: {pdf_path.name}")
        print("Skipping ingest. Use --force-ingest to ingest again.")
        return

    print(f"Ingesting {pdf_path.name} for patient {patient_id}...")
    doc_id = await ingest_document(
        source_type="pdf",
        content=pdf_path.read_bytes(),
        patient_id=patient_id,
        document_date=document_date,
        ingested_by=None,
        source_uri=f"file://{pdf_path}",
        session=session,
        original_filename=pdf_path.name,
    )
    print(f"Ingested document {doc_id}")


def _print_chunks(chunks, infer_category) -> None:
    print(f"\nRETRIEVED CHUNKS ({len(chunks)})")
    for index, chunk in enumerate(chunks, start=1):
        category = infer_category(chunk) or "unknown"
        print("\n" + "-" * 80)
        print(f"chunk index: {index}")
        print(f"section: {chunk.section or 'unknown'}")
        print(f"semantic category: {category}")
        print(f"retrieval score: {chunk.score:.4f}")
        print(f"preview: {_preview(chunk.text)}")


async def _query_loop(session, patient_id: str, *, retrieval_only: bool, top_k: int) -> None:
    from agents.query_agent import _infer_chunk_section_category, query_chunks

    print("\nEnter questions. Commands: :quit, :exit, :retrieval-only, :synthesis")
    current_retrieval_only = retrieval_only

    while True:
        prompt_suffix = "retrieval-only" if current_retrieval_only else "synthesis"
        query = input(f"\n{patient_id} [{prompt_suffix}]> ").strip()
        if not query:
            continue
        if query in {":quit", ":exit"}:
            return
        if query == ":retrieval-only":
            current_retrieval_only = True
            print("Synthesis disabled.")
            continue
        if query == ":synthesis":
            current_retrieval_only = False
            print("Synthesis enabled.")
            continue

        try:
            chunks = await query_chunks(query, patient_id, session, top_k=top_k)
        except Exception as exc:
            if _is_voyage_auth_error(exc):
                print("error: Voyage rejected VOYAGE_API_KEY. Check .env.")
                continue
            raise

        _print_chunks(chunks, _infer_chunk_section_category)

        print("\nFINAL ANSWER")
        if current_retrieval_only:
            print("SKIPPED: retrieval-only mode.")
            continue

        try:
            from agents.synthesis_agent import synthesise

            result = await synthesise(query, chunks)
        except Exception as exc:
            if _is_gemini_quota_error(exc):
                print(
                    "UNAVAILABLE: Gemini quota is exhausted. "
                    "Retrieved chunks above are still valid."
                )
                continue
            raise
        print(result.answer)


async def _run(args: argparse.Namespace) -> None:
    _require_modules(include_synthesis=not args.retrieval_only)
    _require_api_keys(include_synthesis=not args.retrieval_only)

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    pdf_path = _select_pdf(args.pdf)
    detected_patient_id, detected_date = _parse_pdf_metadata(pdf_path)

    print(f"\nSelected PDF: {pdf_path.relative_to(REPO_ROOT)}")
    if detected_patient_id:
        print(f"Detected patient id: {detected_patient_id}")
    if detected_date:
        print(f"Detected document date: {detected_date.isoformat()}")

    patient_id = args.patient_id or _prompt_for_patient_id(detected_patient_id)
    document_date = args.document_date or _prompt_for_document_date(detected_date)

    engine = create_async_engine(_db_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            await _ingest_if_needed(
                session,
                pdf_path,
                patient_id,
                document_date,
                force_ingest=args.force_ingest,
            )
            await _query_loop(
                session,
                patient_id,
                retrieval_only=args.retrieval_only,
                top_k=args.top_k,
            )
    finally:
        await engine.dispose()


def main() -> None:
    _load_env_files()

    parser = argparse.ArgumentParser(
        description="Select a PDF, ingest it, and query that patient interactively."
    )
    parser.add_argument("pdf", nargs="?", help="PDF path. If omitted, show a PDF picker.")
    parser.add_argument("--patient-id", help="Override detected patient id")
    parser.add_argument(
        "--document-date",
        type=date.fromisoformat,
        help="Override detected document date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Gemini synthesis by default",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Ingest even if a ready document with same patient id and filename exists",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        print("error: --top-k must be > 0", file=sys.stderr)
        sys.exit(2)

    try:
        asyncio.run(_run(args))
    except SessionConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
