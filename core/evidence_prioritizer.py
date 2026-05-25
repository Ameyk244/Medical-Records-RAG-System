"""Clinical evidence prioritization for retrieved chunks.

This module does not replace retrieval and does not hide evidence. It produces
deterministic annotations that help synthesis understand which retrieved
records are likely more current, authoritative, and clinically salient.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from core.vector_store import ChunkResult

EvidenceAssessment = dict[str, Any]

_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")

_SECTION_RELIABILITY: dict[str, float] = {
    "discharge": 0.30,
    "procedures": 0.26,
    "diagnosis": 0.24,
    "labs": 0.22,
    "hospital_course": 0.20,
    "medications": 0.18,
    "allergies": 0.18,
    "followup": 0.16,
    "history": 0.12,
    "demographics": 0.10,
    "general": 0.04,
}

_SALIENT_TERMS: tuple[tuple[str, float], ...] = (
    ("nstemi", 0.08),
    ("stemi", 0.08),
    ("myocardial infarction", 0.08),
    ("sepsis", 0.08),
    ("stroke", 0.08),
    ("pci", 0.07),
    ("stent", 0.07),
    ("cabg", 0.07),
    ("angiography", 0.06),
    ("troponin", 0.06),
    ("downtrending", 0.05),
    ("elevated", 0.04),
    ("discharged", 0.06),
    ("stable", 0.04),
    ("allergy", 0.05),
    ("penicillin", 0.05),
    ("aspirin", 0.04),
    ("clopidogrel", 0.04),
    ("atorvastatin", 0.04),
    ("smoker", 0.04),
    ("tobacco", 0.04),
)


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _section_category(chunk: ChunkResult) -> str:
    match = _SECTION_CATEGORY_RE.search(chunk.text)
    if match:
        return match.group(1).strip().lower()

    section = _normalize(chunk.section or "")
    if "discharge" in section:
        return "discharge"
    if "procedure" in section or "surgery" in section:
        return "procedures"
    if "diagnos" in section or "assessment" in section or "impression" in section:
        return "diagnosis"
    if "lab" in section or "observation" in section:
        return "labs"
    if "course" in section or "progress" in section or "encounter" in section:
        return "hospital_course"
    if "medication" in section or "rx" in section:
        return "medications"
    if "allerg" in section:
        return "allergies"
    if "history" in section:
        return "history"
    if "patient" in section or "demographic" in section:
        return "demographics"
    if "follow" in section or "plan" in section:
        return "followup"
    return "general"


def _salience_score(text: str) -> tuple[float, list[str]]:
    normalized = _normalize(text)
    matched: list[str] = []
    score = 0.0
    for term, weight in _SALIENT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            matched.append(term)
            score += weight
    return min(score, 0.28), matched[:8]


def _recency_scores(chunks: list[ChunkResult]) -> dict[int, float]:
    dated = [
        (idx, chunk.document_date)
        for idx, chunk in enumerate(chunks)
        if chunk.document_date is not None
    ]
    if not dated:
        return {}

    dates = [item[1] for item in dated if item[1] is not None]
    assert dates
    oldest = min(dates)
    newest = max(dates)
    span_days = max((newest - oldest).days, 1)

    scores: dict[int, float] = {}
    for idx, chunk_date in dated:
        if chunk_date is None:
            continue
        scores[idx] = ((chunk_date - oldest).days / span_days) * 0.14
    return scores


def _confidence_label(priority_score: float, matched_salience: list[str]) -> str:
    if priority_score >= 0.72 and matched_salience:
        return "strong"
    if priority_score >= 0.50:
        return "moderate"
    return "limited"


def prioritize_evidence(chunks: list[ChunkResult]) -> list[EvidenceAssessment]:
    """Annotate retrieved chunks by reliability, salience, recency, and score."""
    recency_scores = _recency_scores(chunks)
    assessments: list[EvidenceAssessment] = []

    for idx, chunk in enumerate(chunks, start=1):
        category = _section_category(chunk)
        section_score = _SECTION_RELIABILITY.get(category, _SECTION_RELIABILITY["general"])
        salience, matched_salience = _salience_score(chunk.text)
        recency = recency_scores.get(idx - 1, 0.0)
        semantic_component = min(max(chunk.score, 0.0), 1.0) * 0.28
        priority_score = round(section_score + salience + recency + semantic_component, 4)

        rationale: list[str] = [f"section={category}"]
        if matched_salience:
            rationale.append(f"salient_terms={', '.join(matched_salience)}")
        if chunk.document_date:
            rationale.append(f"dated={chunk.document_date.isoformat()}")

        assessments.append(
            {
                "context_number": idx,
                "source_chunk": str(chunk.chunk_id),
                "section": chunk.section,
                "section_category": category,
                "document_date": chunk.document_date.isoformat() if chunk.document_date else None,
                "retrieval_score": round(chunk.score, 4),
                "priority_score": priority_score,
                "confidence_guidance": _confidence_label(priority_score, matched_salience),
                "rationale": "; ".join(rationale),
            }
        )

    return sorted(
        assessments,
        key=lambda item: (
            item["priority_score"],
            item["document_date"] or "",
            -item["context_number"],
        ),
        reverse=True,
    )
