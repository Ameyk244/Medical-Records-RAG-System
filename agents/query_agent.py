"""Query agent — embed query, retrieve top-k chunks. Phase 2."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import EmbeddingError, embed_texts
from core.hybrid_retrieval import INTENT_WEIGHT, hybrid_score, lexical_score
from core.query_intent import QueryIntent, classify_query_intent
from core.vector_store import ChunkResult, VectorStoreError, search_chunks


class QueryError(Exception):
    """Raised when the query pipeline fails. Wraps the underlying cause."""


_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")

_SECTION_HEADING_FALLBACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("demographics", ("patient", "demographic", "identification", "birth", "gender", "sex")),
    ("history", ("history", "past medical", "pmh", "social history", "family history", "smoking")),
    ("diagnosis", ("diagnosis", "diagnoses", "condition", "problem list", "assessment", "impression")),
    ("medications", ("medication", "medications", "medicine", "rx", "prescription")),
    ("allergies", ("allergy", "allergies", "adverse reaction")),
    ("labs", ("lab", "labs", "laboratory", "observation", "diagnostic report", "result")),
    ("procedures", ("procedure", "procedures", "surgery", "operation", "angiography", "pci")),
    ("hospital_course", ("hospital course", "encounter", "visit", "progress", "clinical course")),
    ("discharge", ("discharge", "disposition")),
    ("followup", ("follow up", "follow-up", "followup", "care plan", "appointment")),
)


def _infer_chunk_section_category(chunk: ChunkResult) -> str | None:
    """Read Phase 1 section metadata without requiring a schema migration."""
    match = _SECTION_CATEGORY_RE.search(chunk.text)
    if match:
        return match.group(1).strip().lower()

    if not chunk.section:
        return None

    normalized_section = re.sub(r"[_\-]+", " ", chunk.section.lower())
    normalized_section = re.sub(r"\s+", " ", normalized_section).strip()
    if normalized_section in {"patient", "patient information", "patient details"}:
        return "demographics"

    for category, patterns in _SECTION_HEADING_FALLBACKS:
        if any(pattern in normalized_section for pattern in patterns):
            return category

    return None


def _intent_signal(chunk: ChunkResult, intent: QueryIntent) -> float:
    if not intent.target_sections or intent.boost_weight <= 0:
        return 0.0

    section_category = _infer_chunk_section_category(chunk)
    if section_category not in set(intent.target_sections):
        return 0.0

    # Convert Phase 2's per-intent boost into the 0..1 signal expected by
    # hybrid_score, preserving roughly the same final section contribution.
    return intent.boost_weight / INTENT_WEIGHT if INTENT_WEIGHT > 0 else 0.0


def _apply_hybrid_reranking(
    query_text: str,
    results: list[ChunkResult],
    intent: QueryIntent,
) -> list[ChunkResult]:
    if not results:
        return results

    reranked_results: list[ChunkResult] = []
    for result in results:
        lexical = lexical_score(query_text, result.text)
        intent_boost = _intent_signal(result, intent)

        # pgvector remains the candidate source of truth. Hybrid reranking adds
        # exact lexical evidence for names/dates/abbreviations and a clinical
        # section prior for intent-matched demographics/history/allergy chunks.
        reranked_results.append(
            replace(
                result,
                score=hybrid_score(
                    semantic_score=result.score,
                    lexical_score_value=lexical,
                    intent_boost=intent_boost,
                ),
            )
        )

    return sorted(reranked_results, key=lambda r: r.score, reverse=True)


async def query_chunks(
    query_text: str,
    patient_id: str,
    session: AsyncSession,
    *,
    top_k: int = 5,
    document_date_from: date | None = None,
    document_date_to: date | None = None,
) -> list[ChunkResult]:
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be non-empty")
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    query_text = query_text.strip()
    intent = classify_query_intent(query_text)
    candidate_k = max(top_k, top_k * 3)

    try:
        # input_type="query" (not "document") — Voyage produces asymmetric embeddings
        # for retrieval; query and document vectors live in different subspaces.
        # See core/embeddings.py for details.
        embeddings = await embed_texts([query_text], input_type="query")
    except EmbeddingError as e:
        raise QueryError("failed to embed query") from e

    query_vec = embeddings[0]

    try:
        results = await search_chunks(
            query_embedding=query_vec,
            patient_id=patient_id,
            top_k=candidate_k,
            session=session,
            document_date_from=document_date_from,
            document_date_to=document_date_to,
        )
    except VectorStoreError as e:
        raise QueryError("failed to search chunks") from e

    return _apply_hybrid_reranking(query_text, results, intent)[:top_k]
