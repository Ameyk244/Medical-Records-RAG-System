"""Temporary PDF extraction debugger.

This script stops before embeddings/vector writes and prints what ingestion
actually gives the vector DB candidate chunks.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - diagnostic convenience only
    load_dotenv = None

ClinicalEntities = dict[str, dict[str, str] | list[str]]

ENTITY_LIST_KEYS = (
    "diagnoses",
    "medications",
    "allergies",
    "procedures",
    "labs",
    "risk_factors",
)

CHECK_PATTERNS: dict[str, tuple[str, ...]] = {
    "Patient Name": (
        r"\bpatient\s+name\b",
        r"\bname\s*[:\-]",
    ),
    "DOB": (
        r"\bdob\b",
        r"\bdate\s+of\s+birth\b",
        r"\bbirth\s+date\b",
    ),
    "Age": (
        r"\bage\s*[:\-]?\s*\d{1,3}\b",
        r"\b\d{1,3}\s*(?:year|yr)s?\s*old\b",
    ),
    "Sex": (
        r"\bsex\s*[:\-]?\s*(?:male|female|m|f)\b",
        r"\bgender\s*[:\-]?\s*(?:male|female|m|f)\b",
    ),
    "History": (
        r"\bhistory\b",
        r"\bpast\s+medical\b",
        r"\bpmh\b",
        r"\bsocial\s+history\b",
        r"\bfamily\s+history\b",
    ),
    "Allergies": (
        r"\ballerg(?:y|ies|ic)\b",
        r"\badverse\s+reaction\b",
    ),
}


def _print_banner(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _extract_raw_pdf_text(content: bytes) -> str:
    """Mirror the PDF parser's word/line extraction before section splitting."""
    import pdfplumber

    lines: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue

            lines_map: dict[float, list[dict[str, Any]]] = {}
            for word in words:
                top = word["top"]
                placed = False
                for key in lines_map:
                    if abs(key - top) < 3:
                        lines_map[key].append(word)
                        placed = True
                        break
                if not placed:
                    lines_map[top] = [word]

            if lines:
                lines.append("")
            lines.append(f"[page {page_number}]")
            for top_key in sorted(lines_map):
                line_words = lines_map[top_key]
                line_words.sort(key=lambda w: w["x0"])
                lines.append(" ".join(w["text"] for w in line_words))

    return "\n".join(lines).strip()


def _dedupe_append(values: list[str], value: str) -> None:
    if value and value.lower() not in {existing.lower() for existing in values}:
        values.append(value)


def _merge_entities(entity_sets: list[ClinicalEntities]) -> ClinicalEntities:
    merged: ClinicalEntities = {
        "demographics": {},
        "diagnoses": [],
        "medications": [],
        "allergies": [],
        "procedures": [],
        "labs": [],
        "risk_factors": [],
    }

    demographics = merged["demographics"]
    assert isinstance(demographics, dict)

    for entities in entity_sets:
        incoming_demographics = entities.get("demographics")
        if isinstance(incoming_demographics, dict):
            for key in ("patient_name", "DOB", "age", "sex"):
                value = incoming_demographics.get(key)
                if value and not demographics.get(key):
                    demographics[key] = value

        for key in ENTITY_LIST_KEYS:
            merged_values = merged[key]
            incoming_values = entities.get(key)
            if isinstance(merged_values, list) and isinstance(incoming_values, list):
                for value in incoming_values:
                    _dedupe_append(merged_values, value)

    return merged


def _print_entities(entities: ClinicalEntities) -> None:
    demographics = entities.get("demographics")
    demographics = demographics if isinstance(demographics, dict) else {}

    for key in ("patient_name", "DOB", "age", "sex"):
        print(f"{key}: {demographics.get(key) or 'MISSING'}")

    for key in ENTITY_LIST_KEYS:
        values = entities.get(key)
        if isinstance(values, list) and values:
            print(f"{key}: {', '.join(values)}")
        else:
            print(f"{key}: MISSING")


def _chunk_indexes_matching(chunks: list[Any], patterns: tuple[str, ...]) -> list[int]:
    matches: list[int] = []
    for chunk in chunks:
        if any(re.search(pattern, chunk.text, flags=re.IGNORECASE) for pattern in patterns):
            matches.append(chunk.chunk_index)
    return matches


def _patterns_with_entity_values(
    base_patterns: tuple[str, ...],
    entities: ClinicalEntities,
    label: str,
) -> tuple[str, ...]:
    demographics = entities.get("demographics")
    demographics = demographics if isinstance(demographics, dict) else {}

    extra_patterns: list[str] = []
    if label == "Patient Name" and demographics.get("patient_name"):
        extra_patterns.append(re.escape(demographics["patient_name"]))
    if label == "DOB" and demographics.get("DOB"):
        extra_patterns.append(re.escape(demographics["DOB"]))

    return (*base_patterns, *extra_patterns)


def _print_checks(chunks: list[Any], entities: ClinicalEntities) -> None:
    all_chunk_text = "\n\n".join(chunk.text for chunk in chunks)

    for label, base_patterns in CHECK_PATTERNS.items():
        patterns = _patterns_with_entity_values(base_patterns, entities, label)
        chunk_indexes = _chunk_indexes_matching(chunks, patterns)
        status = "FOUND" if chunk_indexes else "MISSING"
        print(f"{label}: {status}")
        print(f"  chunks: {chunk_indexes if chunk_indexes else 'none'}")

    demographics = entities.get("demographics")
    demographics = demographics if isinstance(demographics, dict) else {}
    present_demographics = {
        key: value
        for key in ("patient_name", "DOB", "age", "sex")
        if (value := demographics.get(key)) and str(value).lower() in all_chunk_text.lower()
    }
    print("\nDemographic entity values present in final chunks:")
    print(json.dumps(present_demographics, indent=2) if present_demographics else "none")


def debug_pdf(pdf_path: Path, raw_chars: int, chunk_limit: int) -> None:
    from core.chunker import chunk_document
    from core.clinical_entities import extract_clinical_entities
    from core.document_parser import parse_document

    content = pdf_path.read_bytes()

    raw_text = _extract_raw_pdf_text(content)
    sections = parse_document("pdf", content, filename=pdf_path.name)
    chunks = chunk_document(sections)

    raw_entities = extract_clinical_entities(raw_text)
    chunk_entities = _merge_entities(
        [chunk.clinical_entities for chunk in chunks if chunk.clinical_entities]
    )

    _print_banner("RAW EXTRACTED TEXT")
    print(f"file: {pdf_path}")
    print(f"raw_chars_total: {len(raw_text)}")
    print(f"\n{raw_text[:raw_chars]}")
    if len(raw_text) > raw_chars:
        print(f"\n... truncated after {raw_chars} chars ...")

    _print_banner("DETECTED SECTIONS")
    print(f"section_count: {len(sections)}")
    for index, section in enumerate(sections, start=1):
        heading = section.heading or "(no heading)"
        print(f"{index}. {heading} [{len(section.text)} chars]")

    _print_banner("EXTRACTED CLINICAL ENTITIES")
    print("From raw extracted text:")
    _print_entities(raw_entities)
    print("\nAggregated from final chunks before embeddings:")
    _print_entities(chunk_entities)

    _print_banner("FINAL CHUNKS")
    print(f"chunk_count: {len(chunks)}")
    for chunk in chunks[:chunk_limit]:
        print(
            "\n"
            + "-" * 50
            + f"\nchunk_index={chunk.chunk_index} "
            + f"section={chunk.section!r} "
            + f"category={chunk.semantic_category!r} "
            + f"tokens={chunk.token_count}"
        )
        print("-" * 50)
        print(chunk.text)

    _print_banner("CHECKS")
    _print_checks(chunks, chunk_entities)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print raw PDF extraction, parser sections, clinical entities, "
            "and final pre-embedding chunks."
        )
    )
    parser.add_argument("pdf", help="Path to one sample PDF")
    parser.add_argument(
        "--raw-chars",
        type=int,
        default=3000,
        help="Number of raw extracted text characters to print",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Number of final pre-embedding chunks to print",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"error: file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"error: expected a PDF file, got: {pdf_path}")
    if args.raw_chars <= 0:
        raise SystemExit("error: --raw-chars must be > 0")
    if args.chunks <= 0:
        raise SystemExit("error: --chunks must be > 0")

    debug_pdf(pdf_path, raw_chars=args.raw_chars, chunk_limit=args.chunks)


if __name__ == "__main__":
    main()
